"""Paired bootstrap comparison of one metric in two per-sample CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.evaluation.bootstrap import paired_bootstrap_ci  # noqa: E402
from kric.evaluation.io import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_metric(path: Path, metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames or metric not in reader.fieldnames:
            raise ValueError(f"{path} must contain sample_id and {metric!r} columns")
        for line_number, row in enumerate(reader, start=2):
            sample_id = row["sample_id"]
            if sample_id in values:
                raise ValueError(f"duplicate sample_id {sample_id!r} in {path}")
            if row[metric] in (None, ""):
                raise ValueError(f"missing {metric} for {sample_id!r} at line {line_number}")
            values[sample_id] = float(row[metric])
    return values


def main() -> int:
    args = parse_args()
    result = paired_bootstrap_ci(
        load_metric(args.baseline, args.metric),
        load_metric(args.candidate, args.metric),
        confidence=args.confidence,
        resamples=args.resamples,
        seed=args.seed,
    )
    result.update(
        {
            "metric": args.metric,
            "baseline": str(args.baseline.resolve()),
            "candidate": str(args.candidate.resolve()),
        }
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Comparison: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

