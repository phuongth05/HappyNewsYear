"""Configuration schemas for controlled captioning baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _resolve(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str
    config_path: Path
    split: str = "dev"
    max_samples: int | None = None
    subset_strategy: str = "first_by_sample_id"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    type: str
    name: str
    revision: str
    device: str = "auto"
    dtype: str = "float32"
    batch_size: int = 1


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    do_sample: bool = False
    num_beams: int = 3
    max_new_tokens: int = 30
    min_new_tokens: int = 1
    no_repeat_ngram_size: int = 2
    length_penalty: float = 1.0
    early_stopping: bool = True

    def to_generate_kwargs(self) -> dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
        }


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    enabled: bool = True
    metrics: tuple[str, ...] = ("cider",)
    allow_metric_errors: bool = False
    entity_extractor: str = "metadata"
    spacy_model: str = "en_core_web_sm"


@dataclass(frozen=True, slots=True)
class PromptConfig:
    """A shared prompt template; B1 only inserts the context block."""

    instruction: str = ""
    context_prefix: str = "Article context:\n"
    context_suffix: str = "\n\n"


@dataclass(frozen=True, slots=True)
class FullArticleContextConfig:
    max_context_tokens: int = 256
    truncation: str = "head"

    @property
    def max_article_tokens(self) -> int:
        """Backward-compatible alias for the exploratory BLIP implementation."""

        return self.max_context_tokens


def _context_config(raw: Mapping[str, Any]) -> FullArticleContextConfig:
    values = dict(raw)
    legacy = values.pop("max_article_tokens", None)
    if legacy is not None:
        if "max_context_tokens" in values:
            raise ValueError("use only one of max_context_tokens or max_article_tokens")
        values["max_context_tokens"] = legacy
    return FullArticleContextConfig(**values)


def _prompt_config(raw: Mapping[str, Any]) -> PromptConfig:
    return PromptConfig(**dict(raw))


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    baseline_config: Path
    baseline_per_sample: Path
    metrics: tuple[str, ...] = ("CIDEr", "EntityPrecision", "EntityRecall", "EntityF1")
    confidence: float = 0.95
    resamples: int = 10_000
    seed: int = 2026


@dataclass(frozen=True, slots=True)
class B0ExperimentConfig:
    experiment: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    generation: GenerationConfig
    output_dir: Path
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    context: FullArticleContextConfig = field(default_factory=FullArticleContextConfig)
    source_path: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "B0ExperimentConfig":
        source = Path(path).resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError("experiment config must contain a mapping")
        if raw.get("experiment") not in {"B0_image_only", "B0_instructblip_image_only"}:
            raise ValueError(
                "B0 runner requires experiment: B0_image_only or "
                "B0_instructblip_image_only"
            )

        dataset = raw.get("dataset")
        model = raw.get("model")
        generation = raw.get("generation", {})
        evaluation = raw.get("evaluation", {})
        prompt = raw.get("prompt", {})
        context = raw.get("context", {})
        for name, value in (("dataset", dataset), ("model", model)):
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
        if not all(isinstance(value, Mapping) for value in (generation, evaluation, prompt, context)):
            raise TypeError("generation, evaluation, prompt, and context must be mappings")

        dataset_config = DatasetConfig(
            name=str(dataset.get("name", "")).lower(),
            config_path=_resolve(dataset["config"], source.parent),
            split=str(dataset.get("split", "dev")).lower(),
            max_samples=(
                None if dataset.get("max_samples") is None else int(dataset["max_samples"])
            ),
            subset_strategy=str(dataset.get("subset_strategy", "first_by_sample_id")),
        )
        model_config = ModelConfig(
            type=str(model.get("type", "")).lower(),
            name=str(model.get("name", "")),
            revision=str(model.get("revision", "")),
            device=str(model.get("device", "auto")),
            dtype=str(model.get("dtype", "float32")),
            batch_size=int(model.get("batch_size", 1)),
        )
        generation_config = GenerationConfig(**dict(generation))
        metrics = evaluation.get("metrics", ["cider"])
        evaluation_config = EvaluationConfig(
            enabled=bool(evaluation.get("enabled", True)),
            metrics=tuple(str(item) for item in metrics),
            allow_metric_errors=bool(evaluation.get("allow_metric_errors", False)),
            entity_extractor=str(evaluation.get("entity_extractor", "metadata")),
            spacy_model=str(evaluation.get("spacy_model", "en_core_web_sm")),
        )
        config = cls(
            experiment=str(raw["experiment"]),
            seed=int(raw.get("seed", 2026)),
            dataset=dataset_config,
            model=model_config,
            generation=generation_config,
            output_dir=_resolve(raw["output_dir"], source.parent),
            evaluation=evaluation_config,
            prompt=_prompt_config(prompt),
            context=_context_config(context),
            source_path=source,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.dataset.name != "goodnews":
            raise ValueError("B0 currently supports only dataset.name: goodnews")
        if self.dataset.split not in {"train", "dev", "test"}:
            raise ValueError("dataset.split must be train, dev, or test")
        if self.dataset.max_samples is not None and self.dataset.max_samples <= 0:
            raise ValueError("dataset.max_samples must be positive")
        if self.dataset.subset_strategy != "first_by_sample_id":
            raise ValueError("only deterministic first_by_sample_id selection is supported")
        expected_type = (
            "instructblip" if self.experiment == "B0_instructblip_image_only" else "blip"
        )
        if self.model.type != expected_type:
            raise ValueError(f"{self.experiment} requires model.type: {expected_type}")
        if not self.model.name or not self.model.revision:
            raise ValueError("model.name and immutable model.revision are required")
        if self.model.batch_size <= 0:
            raise ValueError("model.batch_size must be positive")
        if self.generation.do_sample:
            raise ValueError("B0 requires deterministic decoding: do_sample must be false")
        if self.generation.num_beams <= 0 or self.generation.max_new_tokens <= 0:
            raise ValueError("beam count and max_new_tokens must be positive")
        if self.evaluation.enabled and not self.evaluation.metrics:
            raise ValueError("evaluation.metrics must not be empty when evaluation is enabled")
        if self.context.max_context_tokens <= 0 or self.context.truncation != "head":
            raise ValueError("context requires positive max_context_tokens and head truncation")
        if self.model.type == "instructblip" and not self.prompt.instruction.strip():
            raise ValueError("InstructBLIP experiments require prompt.instruction")


@dataclass(frozen=True, slots=True)
class B1ExperimentConfig:
    experiment: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    generation: GenerationConfig
    context: FullArticleContextConfig
    output_dir: Path
    evaluation: EvaluationConfig
    comparison: ComparisonConfig
    prompt: PromptConfig = field(default_factory=PromptConfig)
    source_path: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "B1ExperimentConfig":
        source = Path(path).resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError("experiment config must contain a mapping")
        if raw.get("experiment") not in {
            "B1_full_article",
            "B1_random_article",
            "B1_instructblip_article_context",
            "B1_instructblip_random_article",
        }:
            raise ValueError(
                "B1 runner requires experiment: B1_full_article or B1_random_article"
            )

        dataset = raw.get("dataset")
        model = raw.get("model")
        generation = raw.get("generation", {})
        context = raw.get("context", {})
        prompt = raw.get("prompt", {})
        evaluation = raw.get("evaluation", {})
        comparison = raw.get("comparison")
        for name, value in (("dataset", dataset), ("model", model), ("comparison", comparison)):
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
        for name, value in (("generation", generation), ("context", context), ("evaluation", evaluation), ("prompt", prompt)):
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")

        metrics = evaluation.get("metrics", ["cider"])
        comparison_metrics = comparison.get(
            "metrics", ["CIDEr", "EntityPrecision", "EntityRecall", "EntityF1"]
        )
        config = cls(
            experiment=str(raw["experiment"]),
            seed=int(raw.get("seed", 2026)),
            dataset=DatasetConfig(
                name=str(dataset.get("name", "")).lower(),
                config_path=_resolve(dataset["config"], source.parent),
                split=str(dataset.get("split", "dev")).lower(),
                max_samples=(
                    None if dataset.get("max_samples") is None else int(dataset["max_samples"])
                ),
                subset_strategy=str(dataset.get("subset_strategy", "first_by_sample_id")),
            ),
            model=ModelConfig(
                type=str(model.get("type", "")).lower(),
                name=str(model.get("name", "")),
                revision=str(model.get("revision", "")),
                device=str(model.get("device", "auto")),
                dtype=str(model.get("dtype", "float32")),
                batch_size=int(model.get("batch_size", 1)),
            ),
            generation=GenerationConfig(**dict(generation)),
            context=_context_config(context),
            output_dir=_resolve(raw["output_dir"], source.parent),
            evaluation=EvaluationConfig(
                enabled=bool(evaluation.get("enabled", True)),
                metrics=tuple(str(item) for item in metrics),
                allow_metric_errors=bool(evaluation.get("allow_metric_errors", False)),
                entity_extractor=str(evaluation.get("entity_extractor", "metadata")),
                spacy_model=str(evaluation.get("spacy_model", "en_core_web_sm")),
            ),
            comparison=ComparisonConfig(
                baseline_config=_resolve(comparison["baseline_config"], source.parent),
                baseline_per_sample=_resolve(comparison["baseline_per_sample"], source.parent),
                metrics=tuple(str(item) for item in comparison_metrics),
                confidence=float(comparison.get("confidence", 0.95)),
                resamples=int(comparison.get("resamples", 10_000)),
                seed=int(comparison.get("seed", 2026)),
            ),
            prompt=_prompt_config(prompt),
            source_path=source,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.dataset.name != "goodnews":
            raise ValueError("B1 currently supports only dataset.name: goodnews")
        if self.dataset.split not in {"train", "dev", "test"}:
            raise ValueError("dataset.split must be train, dev, or test")
        if self.dataset.max_samples is not None and self.dataset.max_samples <= 0:
            raise ValueError("dataset.max_samples must be positive")
        if self.dataset.subset_strategy != "first_by_sample_id":
            raise ValueError("only deterministic first_by_sample_id selection is supported")
        expected_type = (
            "instructblip"
            if self.experiment.startswith("B1_instructblip_")
            else "blip"
        )
        if self.model.type != expected_type or not self.model.name or not self.model.revision:
            raise ValueError(f"{self.experiment} requires a pinned {expected_type} model")
        if self.model.batch_size != 1:
            raise ValueError(
                "B1 requires batch_size: 1 because BLIP conditional prompts have variable lengths"
            )
        if self.generation.do_sample:
            raise ValueError("B1 requires deterministic decoding: do_sample must be false")
        if self.generation.num_beams <= 0 or self.generation.max_new_tokens <= 0:
            raise ValueError("beam count and max_new_tokens must be positive")
        if self.context.max_context_tokens <= 0:
            raise ValueError("context.max_context_tokens must be positive")
        if self.context.truncation != "head":
            raise ValueError("B1 currently supports only deterministic head truncation")
        if not self.evaluation.metrics or not self.comparison.metrics:
            raise ValueError("evaluation and comparison metrics must not be empty")
        if not 0 < self.comparison.confidence < 1 or self.comparison.resamples <= 0:
            raise ValueError("invalid paired-bootstrap configuration")
        if self.model.type == "instructblip" and not self.prompt.instruction.strip():
            raise ValueError("InstructBLIP experiments require prompt.instruction")


def load_experiment_config(
    path: str | Path,
) -> B0ExperimentConfig | B1ExperimentConfig:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("experiment config must contain a mapping")
    experiment = raw.get("experiment")
    if experiment in {"B0_image_only", "B0_instructblip_image_only"}:
        return B0ExperimentConfig.from_file(source)
    if experiment in {
        "B1_full_article",
        "B1_random_article",
        "B1_instructblip_article_context",
        "B1_instructblip_random_article",
    }:
        return B1ExperimentConfig.from_file(source)
    raise ValueError(f"unsupported experiment: {experiment!r}")
