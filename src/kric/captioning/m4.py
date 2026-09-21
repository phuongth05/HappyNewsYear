"""Controlled M4 support-filtered generation and three-method comparison."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from kric.captioning.m3 import MODEL_NAME, MODEL_REVISION, build_atomic_model_inputs
from kric.evaluation.bootstrap import paired_bootstrap_ci
from kric.evaluation.io import load_predictions, write_json
from kric.matching.ranking import write_jsonl


METRICS = ("CIDEr", "EntityPrecision", "EntityRecall", "EntityF1")


def generate_filtered_variant(
    *,
    samples: Sequence[Any],
    contexts: Sequence[Mapping[str, Any]],
    expected_ids: Sequence[str],
    captioner: Any,
    output_dir: Path,
) -> list[dict[str, Any]]:
    if len(expected_ids) != 49 or len(set(expected_ids)) != 49:
        raise ValueError("M4 filtered generation requires the frozen ordered 49 IDs")
    inputs = build_atomic_model_inputs(samples, contexts, expected_ids)
    generated = captioner.generate(inputs)
    if [item.sample_id for item in generated] != list(expected_ids):
        raise RuntimeError("captioner returned missing, duplicate, or reordered M4 IDs")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    context_by_id = {str(row["sample_id"]): row for row in contexts}
    rows: list[dict[str, Any]] = []
    for item in generated:
        context = context_by_id[item.sample_id]
        used = int(item.context_stats["used_article_tokens"])
        if used != int(context["context_tokens"]):
            raise RuntimeError(f"M4 generation token accounting drift for {item.sample_id}")
        if used > int(context["b2_context_token_budget"]):
            raise RuntimeError(f"M4 context exceeds frozen token budget for {item.sample_id}")
        sample = sample_by_id[item.sample_id]
        rows.append(
            {
                "sample_id": item.sample_id,
                "prediction": item.text,
                "reference": sample.reference_caption,
                "metadata": {
                    "dataset": "goodnews",
                    "official_split": sample.metadata.get("official_split"),
                    "image_path": sample.image_path,
                    "experiment": "m4_support_filtered_token_matched",
                    "model_name": MODEL_NAME,
                    "model_revision": MODEL_REVISION,
                    "context_tokens": used,
                    "context_kind": context["context_kind"],
                    "included_evidence_ids": context["included_evidence_ids"],
                    "dropped_evidence_ids": context["dropped_evidence_ids"],
                    "threshold": context["threshold"],
                    "fallback_used": context["fallback_used"],
                },
            }
        )
    write_jsonl(output_dir / "predictions.jsonl", rows)
    return rows


def _metric_rows(path: Path, expected_ids: Sequence[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in ("sample_id", *METRICS)):
            raise ValueError(f"missing required metric columns: {path}")
        for row in reader:
            sample_id = str(row["sample_id"])
            if sample_id in result:
                raise ValueError(f"duplicate metric sample ID: {sample_id}")
            result[sample_id] = {metric: float(row[metric]) for metric in METRICS}
    if any(sample_id not in result for sample_id in expected_ids):
        raise ValueError(f"metrics do not contain every primary sample: {path}")
    return {sample_id: result[sample_id] for sample_id in expected_ids}


def _predictions(path: Path, expected_ids: Sequence[str]) -> dict[str, str]:
    values = {record.sample_id: record.prediction for record in load_predictions(path)}
    if any(sample_id not in values for sample_id in expected_ids):
        raise ValueError(f"predictions do not contain every primary sample: {path}")
    return {sample_id: values[sample_id] for sample_id in expected_ids}


def compare_three_methods(
    *,
    method_dirs: Mapping[str, Path],
    primary_ids: Sequence[str],
    output_dir: Path,
    resamples: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    expected_methods = {"b2_semantic_k3", "m3_atomic_token_matched", "m4_support_filtered"}
    if set(method_dirs) != expected_methods:
        raise ValueError("comparison requires B2, M3 token-matched, and M4 filtered methods")
    metrics = {
        name: _metric_rows(path / "metrics_per_sample.csv", primary_ids)
        for name, path in method_dirs.items()
    }
    predictions = {
        name: _predictions(path / "predictions.jsonl", primary_ids)
        for name, path in method_dirs.items()
    }
    method_means = {
        name: {
            metric: statistics.fmean(values[sample_id][metric] for sample_id in primary_ids)
            for metric in METRICS
        }
        for name, values in metrics.items()
    }
    pairs = {
        "m3_vs_b2": ("b2_semantic_k3", "m3_atomic_token_matched"),
        "m4_vs_b2": ("b2_semantic_k3", "m4_support_filtered"),
        "m4_vs_m3": ("m3_atomic_token_matched", "m4_support_filtered"),
    }
    comparisons: dict[str, Any] = {}
    for label, (baseline, candidate) in pairs.items():
        by_metric = {}
        for metric in METRICS:
            left = {sample_id: metrics[baseline][sample_id][metric] for sample_id in primary_ids}
            right = {sample_id: metrics[candidate][sample_id][metric] for sample_id in primary_ids}
            result = paired_bootstrap_ci(
                left, right, confidence=0.95, resamples=resamples, seed=seed
            )
            deltas = [right[sample_id] - left[sample_id] for sample_id in primary_ids]
            result.update(
                {
                    "wins": sum(delta > 0 for delta in deltas),
                    "ties": sum(delta == 0 for delta in deltas),
                    "losses": sum(delta < 0 for delta in deltas),
                }
            )
            by_metric[metric] = result
        comparisons[label] = {
            "baseline": baseline,
            "candidate": candidate,
            "prediction_change_rate": sum(
                predictions[baseline][sample_id] != predictions[candidate][sample_id]
                for sample_id in primary_ids
            )
            / len(primary_ids),
            "metrics": by_metric,
        }

    fields = ["sample_id"]
    for method in method_dirs:
        fields.extend([f"{method}_{metric}" for metric in METRICS])
    for pair in pairs:
        fields.extend([f"{pair}_{metric}_delta" for metric in METRICS])
    per_sample_path = output_dir / "comparison_per_sample.csv"
    temporary = per_sample_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id in primary_ids:
            row: dict[str, Any] = {"sample_id": sample_id}
            for method in method_dirs:
                for metric in METRICS:
                    row[f"{method}_{metric}"] = metrics[method][sample_id][metric]
            for pair, (baseline, candidate) in pairs.items():
                for metric in METRICS:
                    row[f"{pair}_{metric}_delta"] = (
                        metrics[candidate][sample_id][metric]
                        - metrics[baseline][sample_id][metric]
                    )
            writer.writerow(row)
    temporary.replace(per_sample_path)
    report = {
        "sample_count": len(primary_ids),
        "primary_ids": list(primary_ids),
        "method_means": method_means,
        "comparisons": comparisons,
        "bootstrap": {"resamples": resamples, "seed": seed, "confidence": 0.95},
        "interpretation": (
            "CIDEr and entity overlap are not direct factuality measures. Any factuality "
            "claim requires dedicated human or claim-support evaluation."
        ),
        "human_calibration_claims_used_as_generation_gold": False,
    }
    write_json(output_dir / "comparison.json", report)
    return report


__all__ = ["METRICS", "compare_three_methods", "generate_filtered_variant"]
