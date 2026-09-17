"""Shared deterministic sample selection for experiments and preflight checks."""

from __future__ import annotations

from typing import Any

from .base import DatasetAdapter, DatasetSplit


def select_samples(
    dataset: DatasetAdapter,
    *,
    split: DatasetSplit | str,
    max_samples: int | None,
    strategy: str = "first_by_sample_id",
) -> list[Any]:
    if strategy != "first_by_sample_id":
        raise ValueError(f"unsupported sample-selection strategy: {strategy!r}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    samples = sorted(
        dataset.iter_samples(split), key=lambda sample: sample.sample_id
    )
    return samples if max_samples is None else samples[:max_samples]
