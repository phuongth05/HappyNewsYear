"""Analyze saved B2 variants against the fixed B1 article-context baseline."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.evaluation.bootstrap import paired_bootstrap_ci  # noqa: E402
from kric.evaluation.io import load_predictions, write_json  # noqa: E402


METRICS = ("CIDEr", "EntityPrecision", "EntityRecall", "EntityF1")


def _summary(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "metrics.json").read_text(encoding="utf-8"))


def _per_sample(directory: Path) -> dict[str, dict[str, float]]:
    result = {}
    with (directory / "metrics_per_sample.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["sample_id"]] = {
                metric: float(row[metric]) for metric in METRICS
            }
    return result


def analyze_b2(
    b1_dir: Path,
    variants: dict[str, Path],
    *,
    resamples: int = 10_000,
    seed: int = 2026,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    b1_records = load_predictions(b1_dir / "predictions.jsonl")
    b1_by_id = {record.sample_id: record for record in b1_records}
    ordered_ids = [record.sample_id for record in b1_records]
    b1_metrics = _per_sample(b1_dir)
    b1_summary = _summary(b1_dir)["metrics"]
    result: dict[str, Any] = {
        "samples": len(ordered_ids),
        "baseline": {metric: b1_summary.get(metric) for metric in METRICS},
        "variants": {},
        "paired_bootstrap_vs_b1": {},
        "bm25_vs_semantic_selection_difference_rate": {},
        "selection_decision": {
            "status": "requires_metrics_and_human_qualitative_review",
            "rule": "consider EntityF1, CIDEr, context efficiency, and coded qualitative behavior; never test data",
        },
    }
    qualitative = [
        {
            "sample_id": sample_id,
            "reference_caption": b1_by_id[sample_id].reference,
            "b1_caption": b1_by_id[sample_id].prediction,
            "b1_context_tokens": int(
                b1_by_id[sample_id].metadata["used_article_tokens"]
            ),
            "variants": {},
            "human_category": None,
            "human_notes": "",
        }
        for sample_id in ordered_ids
    ]
    qualitative_by_id = {row["sample_id"]: row for row in qualitative}
    selected_ids_by_variant: dict[str, dict[str, tuple[int, ...]]] = {}

    for name, directory in variants.items():
        records = load_predictions(directory / "predictions.jsonl")
        by_id = {record.sample_id: record for record in records}
        if [record.sample_id for record in records] != ordered_ids:
            raise ValueError(f"{name} ordered IDs differ from B1")
        summary = _summary(directory)["metrics"]
        per_sample = _per_sample(directory)
        context_tokens = [int(record.metadata["context_token_count"]) for record in records]
        b1_tokens = [
            int(b1_by_id[record.sample_id].metadata["used_article_tokens"])
            for record in records
        ]
        selected_counts = [
            int(record.metadata["selected_sentence_count"]) for record in records
        ]
        scores = [
            float(score)
            for record in records
            for score in record.metadata["selected_ranking_scores"]
        ]
        result["variants"][name] = {
            "metrics": {metric: summary.get(metric) for metric in METRICS},
            "mean_selected_sentence_count": statistics.fmean(selected_counts),
            "mean_context_tokens": statistics.fmean(context_tokens),
            "median_context_tokens": statistics.median(context_tokens),
            "percentage_using_fewer_tokens_than_b1": 100.0
            * sum(left < right for left, right in zip(context_tokens, b1_tokens, strict=True))
            / len(records),
            "retrieval_scores": {
                "count": len(scores),
                "mean": statistics.fmean(scores) if scores else None,
                "median": statistics.median(scores) if scores else None,
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
            },
        }
        result["paired_bootstrap_vs_b1"][name] = {
            metric: paired_bootstrap_ci(
                {sample_id: b1_metrics[sample_id][metric] for sample_id in ordered_ids},
                {sample_id: per_sample[sample_id][metric] for sample_id in ordered_ids},
                resamples=resamples,
                seed=seed,
            )
            for metric in METRICS
        }
        selected_ids_by_variant[name] = {
            record.sample_id: tuple(record.metadata["selected_sentence_ids"])
            for record in records
        }
        for record in records:
            qualitative_by_id[record.sample_id]["variants"][name] = {
                "caption": record.prediction,
                "retrieval_method": record.metadata["retrieval_method"],
                "k": record.metadata["retrieval_k"],
                "selected_sentence_ids": record.metadata["selected_sentence_ids"],
                "selected_sentence_texts": record.metadata["selected_sentence_texts"],
                "selected_ranking_scores": record.metadata["selected_ranking_scores"],
                "context_token_count": record.metadata["context_token_count"],
                "fraction_of_article_represented": record.metadata[
                    "fraction_of_article_represented"
                ],
            }

    for k in (1, 3, 5):
        bm25_name = f"bm25_k{k}"
        semantic_name = f"semantic_k{k}"
        if bm25_name in selected_ids_by_variant and semantic_name in selected_ids_by_variant:
            changed = sum(
                selected_ids_by_variant[bm25_name][sample_id]
                != selected_ids_by_variant[semantic_name][sample_id]
                for sample_id in ordered_ids
            )
            result["bm25_vs_semantic_selection_difference_rate"][f"k{k}"] = (
                changed / len(ordered_ids)
            )
    return result, qualitative


def write_review_markdown(path: Path, qualitative: list[dict[str, Any]]) -> None:
    lines = [
        "# B2 qualitative review worksheet",
        "",
        "Assign categories manually; no model or heuristic has labeled success/failure.",
        "",
        "Categories: A removes irrelevant information; B loses important fact; C BM25/semantic differ; D copying reduced; E visually relevant but not reference-relevant.",
        "",
        "| sample_id | category | notes |",
        "|---|---|---|",
    ]
    lines.extend(f"| {row['sample_id']} |  |  |" for row in qualitative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1-dir", required=True, type=Path)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME=RUN_DIR; repeat for all six variants",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qualitative-output", required=True, type=Path)
    parser.add_argument("--review-markdown", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    variants = {}
    for value in args.variant:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or name in variants:
            raise ValueError(f"invalid or duplicate --variant: {value!r}")
        variants[name] = Path(raw_path).expanduser().resolve()
    expected = {f"{method}_k{k}" for method in ("bm25", "semantic") for k in (1, 3, 5)}
    if set(variants) != expected:
        raise ValueError(f"expected variants {sorted(expected)}, got {sorted(variants)}")
    result, qualitative = analyze_b2(
        args.b1_dir.expanduser().resolve(),
        variants,
        resamples=args.resamples,
        seed=args.seed,
    )
    write_json(args.output, result)
    write_json(args.qualitative_output, qualitative)
    write_review_markdown(args.review_markdown, qualitative)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
