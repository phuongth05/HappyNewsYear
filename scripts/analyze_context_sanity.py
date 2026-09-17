"""Analyze aligned B0, B1-correct, and B1-random saved experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.evaluation.bootstrap import paired_bootstrap_ci  # noqa: E402
from kric.evaluation.io import load_predictions, write_json  # noqa: E402
from kric.data.goodnews import GoodNewsConfig, GoodNewsDataset  # noqa: E402


REPORTED_METRICS = ("CIDEr", "EntityPrecision", "EntityRecall", "EntityF1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-dir", required=True, type=Path)
    parser.add_argument("--b1-dir", required=True, type=Path)
    parser.add_argument("--b1-random-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--qualitative-output", type=Path)
    parser.add_argument("--qualitative-markdown-output", type=Path)
    parser.add_argument("--article-excerpt-chars", type=int, default=500)
    parser.add_argument("--representative-count", type=int, default=10)
    return parser.parse_args()


def _read_summary(directory: Path) -> dict:
    return json.loads((directory / "metrics.json").read_text(encoding="utf-8"))


def _read_per_sample(directory: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with (directory / "metrics_per_sample.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["sample_id"]] = {
                metric: float(row[metric])
                for metric in REPORTED_METRICS
                if row.get(metric) not in (None, "")
            }
    return result


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _context_statistics(records) -> dict:
    lengths = [int(record.metadata["original_article_tokens"]) for record in records]
    used_lengths = [int(record.metadata["used_article_tokens"]) for record in records]
    ratios = [float(record.metadata["truncation_ratio"]) for record in records]
    limits = {
        int(record.metadata["max_context_tokens"])
        for record in records
        if "max_context_tokens" in record.metadata
    }
    if len(limits) > 1:
        raise ValueError("condition contains inconsistent max_context_tokens")
    context_limit = next(iter(limits), max(used_lengths))
    truncated = sum(value > 0 for value in ratios)
    result = {
        "samples": len(records),
        "max_context_tokens": context_limit,
        "mean_article_tokens": statistics.fmean(lengths),
        "median_article_tokens": statistics.median(lengths),
        "mean_used_article_tokens": statistics.fmean(used_lengths),
        "median_used_article_tokens": statistics.median(used_lengths),
        "truncated_samples": truncated,
        "percentage_truncated": 100.0 * truncated / len(ratios),
        "truncation_ratio_distribution": {
            "min": min(ratios),
            "p25": _quantile(ratios, 0.25),
            "median": statistics.median(ratios),
            "p75": _quantile(ratios, 0.75),
            "p90": _quantile(ratios, 0.90),
            "p95": _quantile(ratios, 0.95),
            "max": max(ratios),
            "mean": statistics.fmean(ratios),
        },
    }
    if context_limit == 256:
        result["percentage_truncated_at_256"] = result["percentage_truncated"]
    return result


def analyze(b0_dir: Path, b1_dir: Path, random_dir: Path, *, resamples: int, seed: int) -> dict:
    condition_dirs = {"b0": b0_dir, "b1_correct": b1_dir, "b1_random": random_dir}
    records = {
        name: load_predictions(directory / "predictions.jsonl")
        for name, directory in condition_dirs.items()
    }
    by_id = {
        name: {record.sample_id: record for record in values}
        for name, values in records.items()
    }
    ids = set(by_id["b0"])
    if any(set(values) != ids for values in by_id.values()):
        raise ValueError("B0, B1, and B1-random sample IDs are not identical")
    for sample_id in ids:
        if by_id["b1_correct"][sample_id].metadata["selected_article_sample_id"] != sample_id:
            raise ValueError(f"B1-correct has mismatched context for {sample_id}")
        if by_id["b1_random"][sample_id].metadata["selected_article_sample_id"] == sample_id:
            raise ValueError(f"B1-random has matching context for {sample_id}")

    summaries = {name: _read_summary(directory) for name, directory in condition_dirs.items()}
    metrics = {
        name: {metric: summary["metrics"].get(metric) for metric in REPORTED_METRICS}
        for name, summary in summaries.items()
    }
    per_sample = {name: _read_per_sample(directory) for name, directory in condition_dirs.items()}
    paired = {}
    comparisons = (
        ("b0", "b1_correct", "b1_correct_vs_b0"),
        ("b0", "b1_random", "b1_random_vs_b0"),
        ("b1_random", "b1_correct", "b1_correct_vs_b1_random"),
    )
    for baseline_name, candidate_name, comparison_name in comparisons:
        paired[comparison_name] = {}
        for metric in REPORTED_METRICS:
            if all(
                all(metric in per_sample[name][sample_id] for sample_id in ids)
                for name in (baseline_name, candidate_name)
            ):
                paired[comparison_name][metric] = paired_bootstrap_ci(
                    {sample_id: per_sample[baseline_name][sample_id][metric] for sample_id in ids},
                    {sample_id: per_sample[candidate_name][sample_id][metric] for sample_id in ids},
                    resamples=resamples,
                    seed=seed,
                )

    changed = sum(
        by_id["b1_correct"][sample_id].prediction != by_id["b0"][sample_id].prediction
        for sample_id in ids
    )
    qualitative = []
    for sample_id in sorted(ids):
        qualitative.append(
            {
                "sample_id": sample_id,
                "reference": by_id["b0"][sample_id].reference,
                "b0": by_id["b0"][sample_id].prediction,
                "b1_correct": by_id["b1_correct"][sample_id].prediction,
                "b1_random": by_id["b1_random"][sample_id].prediction,
                "correct_article_sample_id": by_id["b1_correct"][sample_id].metadata[
                    "selected_article_sample_id"
                ],
                "random_article_sample_id": by_id["b1_random"][sample_id].metadata[
                    "selected_article_sample_id"
                ],
            }
        )
        if len(qualitative) == 10:
            break

    return {
        "samples": len(ids),
        "metrics": metrics,
        "prediction_change_rate": changed / len(ids),
        "correct_vs_random_entity_gain": (
            metrics["b1_correct"]["EntityF1"] - metrics["b1_random"]["EntityF1"]
            if metrics["b1_correct"]["EntityF1"] is not None
            and metrics["b1_random"]["EntityF1"] is not None
            else None
        ),
        "paired_bootstrap": paired,
        "context_statistics": {
            "b1_correct": _context_statistics(records["b1_correct"]),
            "b1_random": _context_statistics(records["b1_random"]),
        },
        "unsupported_claim_rate": {
            "status": "deferred",
            "reason": "requires atomic claims and evidence-support decisions",
        },
        "qualitative_candidates": qualitative,
    }


def build_qualitative_examples(
    b0_dir: Path,
    b1_dir: Path,
    random_dir: Path,
    dataset_root: Path,
    *,
    article_excerpt_chars: int = 500,
) -> list[dict]:
    """Build aligned, human-inspection-only rows without automatic judgments."""

    if article_excerpt_chars <= 0:
        raise ValueError("article_excerpt_chars must be positive")
    records = {
        name: load_predictions(directory / "predictions.jsonl")
        for name, directory in {
            "b0": b0_dir,
            "b1": b1_dir,
            "b1_random": random_dir,
        }.items()
    }
    by_id = {
        name: {record.sample_id: record for record in condition}
        for name, condition in records.items()
    }
    ordered_ids = [record.sample_id for record in records["b0"]]
    if any(set(condition) != set(ordered_ids) for condition in by_id.values()):
        raise ValueError("qualitative inputs do not contain identical sample IDs")

    dataset = GoodNewsDataset(
        GoodNewsConfig(
            annotations_path=dataset_root / "article+caption.json",
            splits_path=dataset_root / "img_splits.json",
            images_root=dataset_root / "images",
        )
    )
    samples = {sample.sample_id: sample for sample in dataset.iter_samples("dev")}
    missing = sorted(set(ordered_ids) - set(samples))
    if missing:
        raise ValueError(f"prediction IDs are absent from GoodNews dev: {missing[:10]}")

    result = []
    for sample_id in ordered_ids:
        b0 = by_id["b0"][sample_id]
        b1 = by_id["b1"][sample_id]
        random_record = by_id["b1_random"][sample_id]
        if len({b0.reference, b1.reference, random_record.reference}) != 1:
            raise ValueError(f"reference mismatch across conditions for {sample_id}")
        correct_id = str(b1.metadata["selected_article_sample_id"])
        random_id = str(random_record.metadata["selected_article_sample_id"])
        if correct_id != sample_id or random_id == sample_id:
            raise ValueError(f"invalid article assignment for {sample_id}")
        result.append(
            {
                "sample_id": sample_id,
                "reference_caption": b0.reference,
                "b0_caption": b0.prediction,
                "b1_caption": b1.prediction,
                "b1_random_caption": random_record.prediction,
                "article_excerpt": textwrap.shorten(
                    samples[sample_id].article_text,
                    width=article_excerpt_chars,
                    placeholder="…",
                ),
                "correct_article_sample_id": correct_id,
                "random_article_donor_sample_id": random_id,
            }
        )
    return result


def write_qualitative_markdown(
    path: Path, examples: list[dict], *, representative_count: int = 10
) -> None:
    """Write an evenly spaced deterministic subset; no success labels are inferred."""

    if representative_count <= 0 or representative_count > len(examples):
        raise ValueError("representative_count must be within the example count")
    if representative_count == 1:
        indices = [0]
    else:
        indices = [
            round(index * (len(examples) - 1) / (representative_count - 1))
            for index in range(representative_count)
        ]

    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# GoodNews validation: qualitative examples",
        "",
        "Deterministic evenly spaced samples; no automatic success/failure classification.",
        "",
        "| sample_id | reference | B0 | B1 | B1-random | article excerpt | random donor |",
        "|---|---|---|---|---|---|---|",
    ]
    for index in indices:
        row = examples[index]
        lines.append(
            "| "
            + " | ".join(
                cell(row[key])
                for key in (
                    "sample_id",
                    "reference_caption",
                    "b0_caption",
                    "b1_caption",
                    "b1_random_caption",
                    "article_excerpt",
                    "random_article_donor_sample_id",
                )
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    result = analyze(
        args.b0_dir,
        args.b1_dir,
        args.b1_random_dir,
        resamples=args.resamples,
        seed=args.seed,
    )
    write_json(args.output, result)
    qualitative_arguments = (
        args.dataset_root,
        args.qualitative_output,
        args.qualitative_markdown_output,
    )
    if any(value is not None for value in qualitative_arguments):
        if not all(value is not None for value in qualitative_arguments):
            raise ValueError(
                "--dataset-root, --qualitative-output, and "
                "--qualitative-markdown-output must be supplied together"
            )
        examples = build_qualitative_examples(
            args.b0_dir,
            args.b1_dir,
            args.b1_random_dir,
            args.dataset_root.resolve(),
            article_excerpt_chars=args.article_excerpt_chars,
        )
        write_json(args.qualitative_output, examples)
        write_qualitative_markdown(
            args.qualitative_markdown_output,
            examples,
            representative_count=args.representative_count,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
