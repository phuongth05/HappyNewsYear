#!/usr/bin/env python
"""Summarise complete or partially complete atomic-evidence human reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kric.evidence.human_review import analyse_review_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True, help="Primary manual_review_100.csv")
    parser.add_argument("--second-review", help="Optional second annotator CSV")
    parser.add_argument("--output", required=True, help="JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyse_review_files(args.review, args.second_review)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    labelled = report["summary"]["labels"]
    print(f"Rows: {report['summary']['rows']}")
    for field in ("atomicity", "faithfulness", "type_correct"):
        print(f"{field}: labelled={labelled[field]['labelled']} missing={labelled[field]['missing']}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
