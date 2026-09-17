from dataclasses import fields, replace
from pathlib import Path

import pytest

from kric.captioning.config import (
    B1ExperimentConfig,
    FullArticleContextConfig,
    GenerationConfig,
    ModelConfig,
    PromptConfig,
)
from kric.captioning.instructblip import (
    InstructBlipArticleCaptioner,
    InstructBlipImageOnlyCaptioner,
)
from kric.captioning.runner import _assert_b0_b1_comparable
from kric.captioning.runner import _assign_article_samples
from kric.captioning.types import FullArticleInput, ImageOnlyInput


class WordTokenizer:
    model_max_length = 512

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


class FakeProcessor:
    tokenizer = WordTokenizer()
    qformer_tokenizer = WordTokenizer()


def _captioner(cls, *, max_context_tokens=3):
    item = cls(
        ModelConfig(
            "instructblip", "Salesforce/instructblip-flan-t5-xl", "fixed-sha"
        ),
        GenerationConfig(max_new_tokens=30),
        PromptConfig(
            instruction="Describe the image in one factual news-style sentence.",
            context_prefix="Article context:\n",
            context_suffix="\n\n",
        ),
        FullArticleContextConfig(max_context_tokens=max_context_tokens),
        2026,
    )
    item.processor = FakeProcessor()
    item.language_max_positions = 512
    item.qformer_max_positions = 512
    item.num_query_tokens = 32
    return item


def test_b0_contract_cannot_carry_article_or_reference() -> None:
    assert [field.name for field in fields(ImageOnlyInput)] == ["sample_id", "image_path"]
    with pytest.raises(TypeError):
        ImageOnlyInput("id", "image.jpg", article_text="forbidden")


def test_b1_contract_carries_article_but_never_reference() -> None:
    assert [field.name for field in fields(FullArticleInput)] == [
        "sample_id",
        "image_path",
        "article_text",
    ]
    with pytest.raises(TypeError):
        FullArticleInput("id", "image.jpg", "article", reference_caption="forbidden")


def test_prompts_differ_only_by_article_context() -> None:
    b0 = _captioner(InstructBlipImageOnlyCaptioner)
    b1 = _captioner(InstructBlipArticleCaptioner)
    image_prompt = b0._build_image_prompt()
    article_prompt, _ = b1._build_article_prompt("correct article words")
    assert image_prompt == "Describe the image in one factual news-style sentence."
    assert article_prompt == (
        "Article context:\ncorrect article words\n\n" + image_prompt
    )


def test_context_token_accounting_is_exact() -> None:
    captioner = _captioner(InstructBlipArticleCaptioner, max_context_tokens=3)
    prompt, stats = captioner._build_article_prompt("one two three four five")
    assert "one two three" in prompt
    assert "four" not in prompt
    assert stats == {
        "original_article_tokens": 5,
        "used_article_tokens": 3,
        "truncation_ratio": 0.4,
        "max_context_tokens": 3,
        "generation_tokens": 30,
    }


def test_correct_and_random_article_assignment_for_new_experiments() -> None:
    class Sample:
        def __init__(self, sample_id, article_text):
            self.sample_id = sample_id
            self.article_text = article_text

    samples = [Sample(str(index), f"article {index}") for index in range(4)]
    correct = _assign_article_samples(
        samples, "B1_instructblip_article_context", 2026
    )
    random = _assign_article_samples(samples, "B1_instructblip_random_article", 2026)
    assert [item.article_text for item in correct] == [item.article_text for item in samples]
    assert all(item.sample_id != donor.sample_id for item, donor in zip(samples, random))


def test_real_configs_are_controlled_and_decoding_equality_is_enforced(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "configs" / "experiments" / "b1_instructblip_article_context.yaml"
    config = B1ExperimentConfig.from_file(source)
    baseline = _assert_b0_b1_comparable(config)
    assert baseline.model == config.model
    assert baseline.prompt == config.prompt
    assert baseline.context == config.context
    assert baseline.generation == config.generation

    changed = tmp_path / "b0.yaml"
    text = config.comparison.baseline_config.read_text(encoding="utf-8")
    changed.write_text(text.replace("num_beams: 3", "num_beams: 2"), encoding="utf-8")
    unfair = replace(
        config,
        comparison=replace(config.comparison, baseline_config=changed),
    )
    with pytest.raises(ValueError, match="generation"):
        _assert_b0_b1_comparable(unfair)


def test_prompt_and_context_mismatches_are_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = B1ExperimentConfig.from_file(
        root / "configs" / "experiments" / "b1_instructblip_article_context.yaml"
    )
    original = config.comparison.baseline_config.read_text(encoding="utf-8")
    for old, new, expected in (
        ("factual news-style", "brief news-style", "prompt"),
        ("max_context_tokens: 384", "max_context_tokens: 300", "context"),
    ):
        changed = tmp_path / f"b0-{expected}.yaml"
        changed.write_text(original.replace(old, new), encoding="utf-8")
        unfair = replace(
            config,
            comparison=replace(config.comparison, baseline_config=changed),
        )
        with pytest.raises(ValueError, match=expected):
            _assert_b0_b1_comparable(unfair)
