"""Leakage-safe cosine threshold calibration for M4 claim support."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


OBJECTIVES = ("max_f1", "precision_at_least_0.90_then_max_recall")


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    sample_id: str
    claim_id: str
    caption_variant: str
    score: float
    label: int
    source_label: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.claim_id:
            raise ValueError("calibration identifiers must not be empty")
        if self.label not in {0, 1}:
            raise ValueError("binary calibration label must be 0 or 1")
        if not math.isfinite(self.score):
            raise ValueError("calibration score must be finite")


def join_calibration_rows(
    annotation_rows: Sequence[Mapping[str, str]],
    review_rows: Sequence[Mapping[str, str]],
) -> tuple[list[CalibrationExample], dict[str, Any]]:
    """Join frozen labels to frozen scores without modifying either source."""

    def keyed(rows: Sequence[Mapping[str, str]], name: str) -> dict[tuple[str, str], Mapping[str, str]]:
        result: dict[tuple[str, str], Mapping[str, str]] = {}
        for row in rows:
            key = (str(row.get("sample_id", "")), str(row.get("claim_id", "")))
            if not all(key) or key in result:
                raise ValueError(f"{name} contains an empty or duplicate review key: {key}")
            result[key] = row
        return result

    annotations = keyed(annotation_rows, "annotations")
    review = keyed(review_rows, "review")
    if len(annotations) != 120 or len(review) != 120 or set(annotations) != set(review):
        raise ValueError("calibration requires identical frozen 120-claim key sets")
    examples: list[CalibrationExample] = []
    excluded_uncertain = 0
    counts = {"supported": 0, "unsupported": 0, "uncertain": 0}
    for key, score_row in review.items():
        annotation = annotations[key]
        label = str(annotation.get("support_label", "")).strip().casefold()
        if label not in counts:
            raise ValueError(f"invalid or missing frozen support label for {key}: {label!r}")
        counts[label] += 1
        if label == "uncertain":
            excluded_uncertain += 1
            continue
        score_text = str(score_row.get("cosine_top1_score", "")).strip()
        if not score_text:
            raise ValueError(f"missing cosine top-1 score for {key}")
        if str(annotation.get("caption_variant", "")) != str(
            score_row.get("caption_variant", "")
        ):
            raise ValueError(f"caption variant mismatch for {key}")
        examples.append(
            CalibrationExample(
                sample_id=key[0],
                claim_id=key[1],
                caption_variant=str(score_row.get("caption_variant", "")),
                score=float(score_text),
                label=1 if label == "supported" else 0,
                source_label=label,
            )
        )
    return examples, {
        "total_frozen_claims": 120,
        "included_binary_claims": len(examples),
        "excluded_uncertain_claims": excluded_uncertain,
        "label_counts": counts,
    }


def stratified_fold_assignments(
    examples: Sequence[CalibrationExample], *, n_splits: int = 5, seed: int = 2026
) -> list[int]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    assignments = [-1] * len(examples)
    rng = random.Random(seed)
    for label in (0, 1):
        indices = [index for index, example in enumerate(examples) if example.label == label]
        if len(indices) < n_splits:
            raise ValueError(f"label {label} has fewer than {n_splits} examples")
        indices.sort(key=lambda index: (examples[index].sample_id, examples[index].claim_id))
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            assignments[index] = offset % n_splits
    if any(value < 0 for value in assignments):
        raise AssertionError("not every calibration example received a fold")
    return assignments


def confusion(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, int]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must be aligned")
    return {
        "tn": sum(y == 0 and p == 0 for y, p in zip(labels, predictions, strict=True)),
        "fp": sum(y == 0 and p == 1 for y, p in zip(labels, predictions, strict=True)),
        "fn": sum(y == 1 and p == 0 for y, p in zip(labels, predictions, strict=True)),
        "tp": sum(y == 1 and p == 1 for y, p in zip(labels, predictions, strict=True)),
    }


def threshold_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    matrix = confusion(labels, predictions)
    precision = matrix["tp"] / (matrix["tp"] + matrix["fp"]) if matrix["tp"] + matrix["fp"] else 0.0
    recall = matrix["tp"] / (matrix["tp"] + matrix["fn"]) if matrix["tp"] + matrix["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "confusion_matrix": matrix}


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            favorable += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return favorable / (len(positives) * len(negatives))


def auprc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels, strict=True), key=lambda item: -item[0])
    true_positives = 0
    false_positives = 0
    area = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        group_tp = group_fp = 0
        while index < len(ordered) and ordered[index][0] == score:
            if ordered[index][1] == 1:
                group_tp += 1
            else:
                group_fp += 1
            index += 1
        previous_recall = true_positives / positives
        true_positives += group_tp
        false_positives += group_fp
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
    return area


def _candidate_thresholds(scores: Sequence[float]) -> list[float]:
    if not scores:
        raise ValueError("threshold selection requires training examples")
    unique = sorted(set(float(score) for score in scores), reverse=True)
    return [math.nextafter(unique[0], math.inf), *unique]


def select_threshold(
    examples: Sequence[CalibrationExample], objective: str
) -> dict[str, Any]:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown threshold objective: {objective}")
    labels = [example.label for example in examples]
    scores = [example.score for example in examples]
    candidates = []
    for threshold in _candidate_thresholds(scores):
        predictions = [int(score >= threshold) for score in scores]
        metrics = threshold_metrics(labels, predictions)
        candidates.append(
            {
                "threshold": threshold,
                **metrics,
                "predicted_positive_count": sum(predictions),
            }
        )
    if objective == "max_f1":
        selected = max(
            candidates,
            key=lambda row: (
                row["f1"], row["precision"], row["recall"], row["threshold"]
            ),
        )
        constraint_met = True
    else:
        eligible = [
            row
            for row in candidates
            if row["predicted_positive_count"] > 0 and row["precision"] >= 0.90
        ]
        if eligible:
            selected = max(
                eligible,
                key=lambda row: (
                    row["recall"], row["precision"], row["f1"], row["threshold"]
                ),
            )
            constraint_met = True
        else:
            selected = candidates[0]
            constraint_met = False
    return {**selected, "objective": objective, "constraint_met": constraint_met}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_metric_intervals(
    labels: Sequence[int],
    predictions: Sequence[int],
    scores: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if not labels or len(labels) != len(predictions) or len(labels) != len(scores):
        raise ValueError("bootstrap inputs must be non-empty and aligned")
    rng = random.Random(seed)
    sampled: dict[str, list[float]] = {
        "precision": [], "recall": [], "f1": [], "auroc": [], "auprc": []
    }
    for _ in range(resamples):
        indices = [rng.randrange(len(labels)) for _ in labels]
        ys = [labels[index] for index in indices]
        ps = [predictions[index] for index in indices]
        ss = [scores[index] for index in indices]
        metrics = threshold_metrics(ys, ps)
        for name in ("precision", "recall", "f1"):
            sampled[name].append(float(metrics[name]))
        roc, pr = auroc(ys, ss), auprc(ys, ss)
        if roc is not None:
            sampled["auroc"].append(roc)
        if pr is not None:
            sampled["auprc"].append(pr)
    alpha = 1 - confidence
    result: dict[str, Any] = {}
    for name, values in sampled.items():
        result[name] = {
            "confidence": confidence,
            "ci_low": _quantile(values, alpha / 2) if values else None,
            "ci_high": _quantile(values, 1 - alpha / 2) if values else None,
            "valid_resamples": len(values),
            "requested_resamples": resamples,
            "seed": seed,
        }
    return result


def cross_validate_thresholds(
    examples: Sequence[CalibrationExample],
    *,
    n_splits: int = 5,
    seed: int = 2026,
    bootstrap_resamples: int = 10_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assignments = stratified_fold_assignments(examples, n_splits=n_splits, seed=seed)
    reports: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        folds = []
        objective_predictions: list[int] = [0] * len(examples)
        thresholds = []
        for fold in range(n_splits):
            train_indices = [index for index, value in enumerate(assignments) if value != fold]
            heldout_indices = [index for index, value in enumerate(assignments) if value == fold]
            selection = select_threshold([examples[index] for index in train_indices], objective)
            threshold = float(selection["threshold"])
            thresholds.append(threshold)
            heldout_labels = [examples[index].label for index in heldout_indices]
            heldout_scores = [examples[index].score for index in heldout_indices]
            heldout_predictions = [int(score >= threshold) for score in heldout_scores]
            for index, prediction in zip(heldout_indices, heldout_predictions, strict=True):
                objective_predictions[index] = prediction
                example = examples[index]
                prediction_rows.append(
                    {
                        "objective": objective,
                        "fold": fold,
                        "sample_id": example.sample_id,
                        "claim_id": example.claim_id,
                        "caption_variant": example.caption_variant,
                        "cosine_top1_score": example.score,
                        "gold_label": example.label,
                        "gold_support_label": example.source_label,
                        "threshold": threshold,
                        "prediction": prediction,
                    }
                )
            folds.append(
                {
                    "fold": fold,
                    "threshold": threshold,
                    "constraint_met_on_training": selection["constraint_met"],
                    "train_count": len(train_indices),
                    "heldout_count": len(heldout_indices),
                    "train_claim_ids": [examples[index].claim_id for index in train_indices],
                    "heldout_claim_ids": [examples[index].claim_id for index in heldout_indices],
                    "heldout": {
                        **threshold_metrics(heldout_labels, heldout_predictions),
                        "auroc": auroc(heldout_labels, heldout_scores),
                        "auprc": auprc(heldout_labels, heldout_scores),
                    },
                }
            )
        labels = [example.label for example in examples]
        scores = [example.score for example in examples]
        aggregate = threshold_metrics(labels, objective_predictions)
        aggregate.update({"auroc": auroc(labels, scores), "auprc": auprc(labels, scores)})
        reports[objective] = {
            "folds": folds,
            "mean_threshold": statistics.fmean(thresholds),
            "median_threshold": statistics.median(thresholds),
            "out_of_fold": aggregate,
            "bootstrap_95_ci": bootstrap_metric_intervals(
                labels,
                objective_predictions,
                scores,
                resamples=bootstrap_resamples,
                seed=seed,
            ),
        }
    return reports, prediction_rows


__all__ = [
    "OBJECTIVES",
    "CalibrationExample",
    "auprc",
    "auroc",
    "bootstrap_metric_intervals",
    "cross_validate_thresholds",
    "join_calibration_rows",
    "select_threshold",
    "stratified_fold_assignments",
    "threshold_metrics",
]
