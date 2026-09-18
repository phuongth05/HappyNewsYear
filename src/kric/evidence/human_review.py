"""Utilities for analysing optional human review of atomic evidence.

Missing labels are deliberately excluded from denominators.  This module never
turns an empty review cell into a negative judgement.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


LABELS = {
    "atomicity": ("good", "too_coarse", "over_split"),
    "faithfulness": ("supported", "unsupported", "uncertain"),
    "type_correct": ("yes", "no"),
}
KEY_FIELDS = ("sample_id", "source_sentence_id")


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def load_review_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _unit_types(row: Mapping[str, str]) -> set[str]:
    raw = _clean(row.get("atomic_units_json"))
    if not raw:
        return set()
    try:
        units = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return set()
    if not isinstance(units, list):
        return set()
    result: set[str] = set()
    for unit in units:
        if isinstance(unit, dict) and _clean(unit.get("type")):
            result.add(_clean(unit["type"]))
    return result


def _cohen_kappa(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    pairs = list(pairs)
    if not pairs:
        return {"n": 0, "percent_agreement": None, "cohen_kappa": None}
    agree = sum(left == right for left, right in pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    labels = set(left_counts) | set(right_counts)
    n = len(pairs)
    observed = agree / n
    expected = sum((left_counts[label] / n) * (right_counts[label] / n) for label in labels)
    kappa = None if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return {"n": n, "percent_agreement": observed * 100.0, "cohen_kappa": kappa}


def summarize_reviews(rows: list[Mapping[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": len(rows), "labels": {}, "invalid_labels": {}}
    for field, allowed in LABELS.items():
        valid = Counter()
        invalid = Counter()
        missing = 0
        for row in rows:
            value = _clean(row.get(field)).lower()
            if not value:
                missing += 1
            elif value in allowed:
                valid[value] += 1
            else:
                invalid[value] += 1
        denominator = sum(valid.values())
        summary["labels"][field] = {
            "counts": {label: valid[label] for label in allowed},
            "labelled": denominator,
            "missing": missing,
            "distribution": {
                label: (valid[label] / denominator if denominator else None) for label in allowed
            },
        }
        summary["invalid_labels"][field] = dict(sorted(invalid.items()))

    per_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        judgement = _clean(row.get("type_correct")).lower()
        if judgement not in LABELS["type_correct"]:
            continue
        for evidence_type in _unit_types(row):
            per_type[evidence_type][judgement] += 1
    summary["per_type_accuracy"] = {
        evidence_type: {
            "yes": counts["yes"],
            "no": counts["no"],
            "labelled_rows_containing_type": counts["yes"] + counts["no"],
            "accuracy": counts["yes"] / (counts["yes"] + counts["no"]),
        }
        for evidence_type, counts in sorted(per_type.items())
    }

    notes = Counter(_clean(row.get("notes")) for row in rows if _clean(row.get("notes")))
    categories = Counter(
        _clean(row.get("failure_category"))
        for row in rows
        if _clean(row.get("failure_category"))
    )
    summary["common_failure_notes"] = [
        {"note": note, "count": count} for note, count in notes.most_common()
    ]
    summary["failure_categories"] = [
        {"category": category, "count": count} for category, count in categories.most_common()
    ]
    return summary


def agreement_report(
    first: list[Mapping[str, str]], second: list[Mapping[str, str]]
) -> dict[str, Any]:
    def key(row: Mapping[str, str]) -> tuple[str, str]:
        return tuple(_clean(row.get(field)) for field in KEY_FIELDS)  # type: ignore[return-value]

    def index_unique(rows: list[Mapping[str, str]], name: str) -> dict[tuple[str, str], Mapping[str, str]]:
        indexed: dict[tuple[str, str], Mapping[str, str]] = {}
        for row in rows:
            row_key = key(row)
            if not all(row_key):
                raise ValueError(f"{name} review contains an empty annotation key")
            if row_key in indexed:
                raise ValueError(f"{name} review contains duplicate annotation key {row_key}")
            indexed[row_key] = row
        return indexed

    first_by_key = index_unique(first, "first")
    second_by_key = index_unique(second, "second")
    overlap = sorted(set(first_by_key) & set(second_by_key))
    result: dict[str, Any] = {
        "first_rows": len(first),
        "second_rows": len(second),
        "overlapping_rows": len(overlap),
        "fields": {},
    }
    for field, allowed in LABELS.items():
        pairs: list[tuple[str, str]] = []
        for row_key in overlap:
            left = _clean(first_by_key[row_key].get(field)).lower()
            right = _clean(second_by_key[row_key].get(field)).lower()
            if left in allowed and right in allowed:
                pairs.append((left, right))
        field_report = _cohen_kappa(pairs)
        field_report["valid_overlap_count"] = field_report["n"]
        result["fields"][field] = field_report
    return result


def analyse_review_files(
    first_path: str | Path, second_path: str | Path | None = None
) -> dict[str, Any]:
    first = load_review_csv(first_path)
    report: dict[str, Any] = {
        "review_file": str(Path(first_path)),
        "summary": summarize_reviews(first),
    }
    if second_path is not None:
        second = load_review_csv(second_path)
        report["second_review_file"] = str(Path(second_path))
        report["second_summary"] = summarize_reviews(second)
        report["agreement"] = agreement_report(first, second)
    return report
