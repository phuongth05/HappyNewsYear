"""Configuration schema for the B0 image-only experiment."""

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


@dataclass(frozen=True, slots=True)
class B0ExperimentConfig:
    experiment: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    generation: GenerationConfig
    output_dir: Path
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    source_path: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "B0ExperimentConfig":
        source = Path(path).resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError("experiment config must contain a mapping")
        if raw.get("experiment") != "B0_image_only":
            raise ValueError("B0 runner requires experiment: B0_image_only")

        dataset = raw.get("dataset")
        model = raw.get("model")
        generation = raw.get("generation", {})
        evaluation = raw.get("evaluation", {})
        for name, value in (("dataset", dataset), ("model", model)):
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
        if not isinstance(generation, Mapping) or not isinstance(evaluation, Mapping):
            raise TypeError("generation and evaluation must be mappings")

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
        )
        config = cls(
            experiment=str(raw["experiment"]),
            seed=int(raw.get("seed", 2026)),
            dataset=dataset_config,
            model=model_config,
            generation=generation_config,
            output_dir=_resolve(raw["output_dir"], source.parent),
            evaluation=evaluation_config,
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
        if self.model.type != "blip":
            raise ValueError("B0 currently supports model.type: blip")
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
