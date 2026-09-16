"""Paired bootstrap confidence intervals for aligned per-sample metrics."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping
from typing import Any


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_ci(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Estimate a CI for candidate-minus-baseline using paired sample IDs."""

    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))[:10]
        missing_baseline = sorted(set(candidate) - set(baseline))[:10]
        raise ValueError(
            f"paired IDs differ: missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )
    if not baseline:
        raise ValueError("paired bootstrap requires at least one sample")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    sample_ids = sorted(baseline)
    deltas = [float(candidate[sample_id]) - float(baseline[sample_id]) for sample_id in sample_ids]
    rng = random.Random(seed)
    bootstrap_means = [
        statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(resamples)
    ]
    alpha = 1 - confidence
    return {
        "n": len(deltas),
        "mean_delta": statistics.fmean(deltas),
        "confidence": confidence,
        "ci_low": _quantile(bootstrap_means, alpha / 2),
        "ci_high": _quantile(bootstrap_means, 1 - alpha / 2),
        "probability_candidate_better": sum(value > 0 for value in bootstrap_means) / resamples,
        "resamples": resamples,
        "seed": seed,
        "direction": "candidate_minus_baseline",
    }

