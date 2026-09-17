import json
from dataclasses import replace
from pathlib import Path

import kric.captioning.runner as runner_module
from kric.captioning.config import EvaluationConfig
from kric.captioning.config import B0ExperimentConfig
from kric.captioning.runner import run_b0
from kric.captioning.types import GeneratedCaption, ImageOnlyInput
from kric.evaluation.io import load_predictions


class SpyCaptioner:
    load_seconds = 0.0

    def __init__(self) -> None:
        self.received = ()

    def load(self):
        return None

    def generate(self, inputs):
        self.received = tuple(inputs)
        return [GeneratedCaption(item.sample_id, f"caption for {item.sample_id}") for item in inputs]

    def model_info(self):
        return {
            "name": "fake",
            "requested_revision": "test-sha",
            "resolved_revision": "test-sha",
            "tokenizer": {"bos_token_id": 1, "eos_token_id": 2},
        }


def _write_fixture(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    images = data / "images"
    images.mkdir(parents=True)
    annotations = {
        "story-z": {"article": "Secret article Z.", "images": {"0": "Reference Z."}},
        "story-a": {"article": "Secret article A.", "images": {"0": "Reference A."}},
    }
    splits = {"story-z_0.jpg": "val", "story-a_0.jpg": "val"}
    (data / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
    (data / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    for sample_id in ("story-z_0", "story-a_0"):
        (images / f"{sample_id}.jpg").write_bytes(b"fake image")
    dataset_config = tmp_path / "goodnews.yaml"
    dataset_config.write_text(
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
    experiment = tmp_path / "b0.yaml"
    experiment.write_text(
        "\n".join(
            (
                "experiment: B0_image_only",
                "seed: 11",
                "dataset:",
                "  name: goodnews",
                "  config: goodnews.yaml",
                "  split: dev",
                "  max_samples: 1",
                "model:",
                "  type: blip",
                "  name: fake/model",
                "  revision: test-sha",
                "generation:",
                "  do_sample: false",
                "output_dir: run",
                "evaluation:",
                "  enabled: false",
            )
        ),
        encoding="utf-8",
    )
    return experiment


def test_runner_passes_only_image_contract_and_writes_evaluator_format(tmp_path: Path) -> None:
    config = B0ExperimentConfig.from_file(_write_fixture(tmp_path))
    spy = SpyCaptioner()
    result = run_b0(config, captioner=spy)

    assert len(spy.received) == 1
    assert isinstance(spy.received[0], ImageOnlyInput)
    assert spy.received[0].sample_id == "story-a_0"
    assert not hasattr(spy.received[0], "article_text")
    assert not hasattr(spy.received[0], "reference_caption")

    records = load_predictions(result["predictions"])
    assert records[0].prediction == "caption for story-a_0"
    assert records[0].reference == "Reference A."
    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_contract"] == ["image"]
    assert manifest["article_fields_passed_to_model"] is False
    assert manifest["reference_passed_to_model"] is False
    assert (config.output_dir / "generation_config.json").is_file()
    assert (config.output_dir / "token_settings.json").is_file()
    assert (config.output_dir / "runtime.json").is_file()
    assert (config.output_dir / "resolved_config.json").is_file()
    assert (config.output_dir / "gpu_memory.json").is_file()


def test_runner_evaluates_the_saved_prediction_artifact(tmp_path: Path, monkeypatch) -> None:
    config = B0ExperimentConfig.from_file(_write_fixture(tmp_path))
    config = replace(config, evaluation=EvaluationConfig(enabled=True, metrics=("cider",)))
    observed = {}

    def fake_evaluation(received_config, predictions):
        observed["predictions"] = predictions
        assert predictions.is_file()
        return {"seconds": 0.01}

    monkeypatch.setattr(runner_module, "_run_evaluation", fake_evaluation)
    result = run_b0(config, captioner=SpyCaptioner())
    assert observed["predictions"] == Path(result["predictions"])
