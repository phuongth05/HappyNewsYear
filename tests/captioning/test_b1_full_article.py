import json
from dataclasses import fields
from pathlib import Path

import pytest

from kric.captioning.blip_full_article import BlipFullArticleCaptioner
from kric.captioning.config import (
    B1ExperimentConfig,
    FullArticleContextConfig,
    GenerationConfig,
    ModelConfig,
)
from kric.captioning.runner import (
    _assign_article_samples,
    _assert_b0_b1_comparable,
    _run_paired_comparisons,
    run_b1,
)
from kric.captioning.types import FullArticleInput, GeneratedCaption
from kric.evaluation.io import load_predictions


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def prepare_for_model(self, ids, **kwargs):
        return {"input_ids": [101, *ids, 102], "attention_mask": [1] * (len(ids) + 2)}


class FakeProcessor:
    tokenizer = FakeTokenizer()


class FakeB1Captioner:
    load_seconds = 0.0

    def __init__(self):
        self.received = ()

    def load(self):
        return None

    def generate(self, inputs):
        self.received = tuple(inputs)
        return [
            GeneratedCaption(
                item.sample_id,
                f"context caption for {item.sample_id}",
                {
                    "original_article_tokens": len(item.article_text.split()),
                    "used_article_tokens": min(3, len(item.article_text.split())),
                    "truncation_ratio": max(0, len(item.article_text.split()) - 3)
                    / len(item.article_text.split()),
                    "max_context_tokens": 3,
                    "generation_tokens": 30,
                },
            )
            for item in inputs
        ]

    def model_info(self):
        return {"name": "fake", "tokenizer": {}, "context_policy": {}}


def _write_fixture(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    images = data / "images"
    images.mkdir(parents=True)
    annotations = {
        "story": {
            "article": "one two three four five",
            "images": {"0": "Reference caption."},
        }
    }
    (data / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
    (data / "splits.json").write_text(json.dumps({"story_0.jpg": "val"}), encoding="utf-8")
    (images / "story_0.jpg").write_bytes(b"fake")
    (tmp_path / "goodnews.yaml").write_text(
        "\n".join(
            (
                "dataset: goodnews",
                "root_dir: data",
                "annotations_path: annotations.json",
                "splits_path: splits.json",
                "images_root: images",
            )
        ),
        encoding="utf-8",
    )
    shared = """
seed: 9
dataset:
  name: goodnews
  config: goodnews.yaml
  split: dev
  max_samples: 1
model:
  type: blip
  name: fake/model
  revision: fixed-sha
generation:
  do_sample: false
context:
  max_context_tokens: 3
  truncation: head
output_dir: {output}
evaluation:
  enabled: false
"""
    (tmp_path / "b0.yaml").write_text(
        "experiment: B0_image_only\n" + shared.format(output="b0-run"),
        encoding="utf-8",
    )
    (tmp_path / "b1.yaml").write_text(
        "experiment: B1_full_article\n"
        + shared.format(output="b1-run")
        + """
comparison:
  baseline_config: b0.yaml
  baseline_per_sample: b0-run/metrics_per_sample.csv
  metrics: [CIDEr]
""",
        encoding="utf-8",
    )
    return tmp_path / "b1.yaml"


def test_full_article_contract_excludes_reference_and_metadata() -> None:
    assert [field.name for field in fields(FullArticleInput)] == [
        "sample_id",
        "image_path",
        "article_text",
    ]


def test_article_token_accounting_uses_head_limit() -> None:
    captioner = BlipFullArticleCaptioner(
        ModelConfig("blip", "fake", "sha"),
        GenerationConfig(),
        FullArticleContextConfig(max_context_tokens=3),
        1,
    )
    captioner.processor = FakeProcessor()
    encoded, stats = captioner._encode_article("one two three four five")
    assert encoded["input_ids"] == [101, 0, 1, 2, 102]
    assert stats == {
        "original_article_tokens": 5,
        "used_article_tokens": 3,
        "truncation_ratio": 0.4,
        "max_context_tokens": 3,
        "generation_tokens": 30,
    }


def test_b1_runner_logs_context_without_exposing_reference(tmp_path: Path) -> None:
    config = B1ExperimentConfig.from_file(_write_fixture(tmp_path))
    fake = FakeB1Captioner()
    result = run_b1(config, captioner=fake)
    assert isinstance(fake.received[0], FullArticleInput)
    assert not hasattr(fake.received[0], "reference_caption")
    record = load_predictions(result["predictions"])[0]
    assert record.metadata["original_article_tokens"] == 5
    assert record.metadata["used_article_tokens"] == 3
    assert record.metadata["truncation_ratio"] == 0.4
    assert result["context_statistics"]["truncated_sample_rate"] == 1.0


def test_comparability_check_rejects_changed_generation(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    text = path.read_text(encoding="utf-8").replace("do_sample: false", "do_sample: false\n  num_beams: 2", 1)
    path.write_text(text, encoding="utf-8")
    config = B1ExperimentConfig.from_file(path)
    with pytest.raises(ValueError, match="generation"):
        _assert_b0_b1_comparable(config)


def test_b1_runs_paired_bootstrap_on_aligned_outputs(tmp_path: Path) -> None:
    config = B1ExperimentConfig.from_file(_write_fixture(tmp_path))
    config.output_dir.mkdir()
    header = "sample_id,CIDEr\n"
    config.comparison.baseline_per_sample.parent.mkdir(exist_ok=True)
    config.comparison.baseline_per_sample.write_text(
        header + "story_0,0.2\n", encoding="utf-8"
    )
    (config.output_dir / "metrics_per_sample.csv").write_text(
        header + "story_0,0.4\n", encoding="utf-8"
    )
    result = _run_paired_comparisons(config)
    assert result["status"] == "complete"
    comparison = json.loads(
        (config.output_dir / "comparisons" / "CIDEr.json").read_text(encoding="utf-8")
    )
    assert comparison["mean_delta"] == pytest.approx(0.2)


def test_random_article_assignment_is_seeded_and_has_no_fixed_points() -> None:
    class Sample:
        def __init__(self, sample_id):
            self.sample_id = sample_id

    samples = [Sample(str(index)) for index in range(5)]
    first = _assign_article_samples(samples, "B1_random_article", 2026)
    second = _assign_article_samples(samples, "B1_random_article", 2026)
    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert all(sample.sample_id != donor.sample_id for sample, donor in zip(samples, first))

    new_model = _assign_article_samples(
        samples, "B1_instructblip_random_article", 2026
    )
    assert all(
        sample.sample_id != donor.sample_id
        for sample, donor in zip(samples, new_model)
    )
