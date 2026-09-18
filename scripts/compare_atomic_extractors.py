#!/usr/bin/env python
"""Build a neutral, human-reviewable spaCy/LLM comparison artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kric.evidence.extractor_comparison import (
    audit_units,
    build_comparison_rows,
    comparison_coverage,
    token_matched_context_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy-review", required=True, help="Existing manual_review_100.csv")
    parser.add_argument("--llm-evidence", required=True, help="llm_atomic_evidence_100.jsonl")
    parser.add_argument("--llm-failures", help="Optional failures.jsonl")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--human-review-output")
    parser.add_argument("--audit-output", required=True)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    with Path(args.spacy_review).open("r", encoding="utf-8-sig", newline="") as handle:
        spacy_rows = [dict(row) for row in csv.DictReader(handle)]
    llm_rows = _jsonl(Path(args.llm_evidence))
    llm_failures = _jsonl(Path(args.llm_failures)) if args.llm_failures else []
    if len(spacy_rows) != 100:
        raise ValueError("spaCy review must contain the authoritative 100 source sentences")
    comparison, spacy_audit_rows, llm_audit_rows = build_comparison_rows(
        spacy_rows, llm_rows, llm_failures
    )

    fieldnames = list(comparison[0])
    csv_rows = []
    for row in comparison:
        csv_row = dict(row)
        csv_row["spacy_units"] = json.dumps(row["spacy_units"], ensure_ascii=False)
        csv_row["llm_units"] = json.dumps(row["llm_units"], ensure_ascii=False)
        csv_rows.append(csv_row)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    if args.human_review_output:
        human_review_output = Path(args.human_review_output)
        human_review_output.parent.mkdir(parents=True, exist_ok=True)
        with human_review_output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in comparison),
            encoding="utf-8",
        )
    audit = {
        **comparison_coverage(comparison),
        "human_preference_is_blank_by_design": True,
        "spacy": audit_units(spacy_audit_rows, "units"),
        "llm": audit_units(llm_audit_rows, "units"),
        "token_matched_without_padding": {
            "spacy": token_matched_context_stats(spacy_audit_rows, "units"),
            "llm": token_matched_context_stats(llm_audit_rows, "units"),
        },
    }
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output_csv}")
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()
