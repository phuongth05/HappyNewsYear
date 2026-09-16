"""Audit a configured dataset and emit a machine-readable JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.data.audit import audit_dataset  # noqa: E402
from kric.data.goodnews import GoodNewsDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("goodnews",))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", default="all", choices=("all", "train", "dev", "val", "test"))
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit 1 when any data or split issue is detected",
    )
    return parser.parse_args()


def print_summary(report: dict) -> None:
    print(f"Dataset: {report['dataset']}")
    print(f"Requested split: {report['requested_split']}")
    print(f"Samples: {report['number_of_samples']}")
    print("Split counts:", json.dumps(report["split_counts"], sort_keys=True))
    print("Issues:")
    for name, details in report["issues"].items():
        print(f"  {name}: {details['count']}")
    print("Statistics:")
    for name, values in report["statistics"].items():
        print(f"  {name}: {json.dumps(values, sort_keys=True)}")


def main() -> int:
    args = parse_args()
    dataset = GoodNewsDataset.from_config(args.config)
    report = audit_dataset(
        dataset,
        split=None if args.split == "all" else args.split,
        long_article_words=dataset.config.long_article_words,
        max_examples=dataset.config.max_issue_examples,
    )
    print_summary(report)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Report: {output}")
    issue_count = sum(item["count"] for item in report["issues"].values())
    return 1 if args.fail_on_issues and issue_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

