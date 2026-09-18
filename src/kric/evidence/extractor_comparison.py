"""Side-by-side audit helpers for atomic evidence extractors."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
import json
from typing import Any, Iterable, Mapping


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def audit_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _summary(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def audit_units(records: Iterable[Mapping[str, Any]], units_field: str) -> dict[str, Any]:
    rows = list(records)
    units_per_sentence: list[int] = []
    unit_token_lengths: list[int] = []
    types: Counter[str] = Counter()
    normalized_texts_by_sample: dict[str, Counter[str]] = defaultdict(Counter)
    unresolved_spans = 0
    total_units = 0
    for row in rows:
        units = row.get(units_field, [])
        if not isinstance(units, list):
            units = []
        units_per_sentence.append(len(units))
        total_units += len(units)
        for unit in units:
            if not isinstance(unit, dict):
                continue
            text = str(unit.get("text", "")).strip()
            unit_token_lengths.append(len(audit_tokens(text)))
            types[str(unit.get("type", "unknown"))] += 1
            sample_id = str(row.get("sample_id", ""))
            normalized_texts_by_sample[sample_id][" ".join(text.casefold().split())] += 1
            span = unit.get("source_span")
            source = str(row.get("source_sentence", ""))
            exact_span = isinstance(span, dict) and span.get("alignment") == "exact"
            if isinstance(span, dict) and not exact_span:
                start = span.get("start")
                end = span.get("end")
                span_text = span.get("text")
                exact_span = (
                    isinstance(start, int)
                    and isinstance(end, int)
                    and isinstance(span_text, str)
                    and 0 <= start < end <= len(source)
                    and source[start:end] == span_text
                )
            if not exact_span:
                unresolved_spans += 1
    duplicate_units = sum(
        count
        for sample_counts in normalized_texts_by_sample.values()
        for count in sample_counts.values()
        if count > 1
    )
    return {
        "source_sentences": len(rows),
        "atomic_units": total_units,
        "units_per_sentence": _summary(units_per_sentence),
        "evidence_token_length": _summary(unit_token_lengths),
        "type_distribution": dict(sorted(types.items())),
        "exact_duplicate_units": duplicate_units,
        "exact_duplicate_rate": duplicate_units / total_units if total_units else None,
        "exact_duplicate_rate_definition": (
            "fraction of units participating in an exact normalized-text duplicate "
            "within the same sample"
        ),
        "unresolved_source_spans": unresolved_spans,
        "audit_tokenizer": "unicode_regex_v1",
    }


def build_comparison_rows(
    spacy_rows: Iterable[Mapping[str, str]],
    llm_rows: Iterable[Mapping[str, Any]],
    llm_failures: Iterable[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    spacy_rows = list(spacy_rows)
    llm_rows = list(llm_rows)
    llm_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in llm_rows:
        key = (str(row.get("sample_id", "")), str(row.get("source_sentence_id", "")))
        if not all(key):
            raise ValueError("LLM output contains an empty source key")
        if key in llm_by_key:
            raise ValueError(f"LLM output contains duplicate source key {key}")
        llm_by_key[key] = row
    failures_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in llm_failures:
        key = (str(row.get("sample_id", "")), str(row.get("source_sentence_id", "")))
        if not all(key):
            raise ValueError("LLM failure contains an empty source key")
        if key in failures_by_key:
            raise ValueError(f"LLM failures contain duplicate source key {key}")
        failures_by_key[key] = row

    comparison: list[dict[str, str]] = []
    spacy_audit_rows: list[dict[str, Any]] = []
    llm_audit_rows: list[dict[str, Any]] = []
    seen_spacy: set[tuple[str, str]] = set()
    for source in spacy_rows:
        key = (str(source.get("sample_id", "")), str(source.get("source_sentence_id", "")))
        if not all(key):
            raise ValueError("spaCy review contains an empty source key")
        if key in seen_spacy:
            raise ValueError(f"spaCy review contains duplicate source key {key}")
        seen_spacy.add(key)
        source_sentence = str(source.get("source_sentence", ""))
        spacy_units = json.loads(source.get("atomic_units_json", "[]") or "[]")
        if not isinstance(spacy_units, list):
            raise ValueError(f"spaCy atomic units are not a list for {key}")
        llm = llm_by_key.get(key)
        failure = failures_by_key.get(key)
        if llm is not None and failure is not None:
            raise ValueError(f"source key appears in both LLM evidence and failures: {key}")
        if llm is not None:
            if llm.get("source_sentence") != source_sentence:
                raise ValueError(f"source sentence changed for {key}")
            llm_units = llm.get("propositions", [])
            if not isinstance(llm_units, list):
                raise ValueError(f"LLM propositions are not a list for {key}")
            llm_status = "success"
            failure_type = ""
            failure_message = ""
        else:
            llm_units = []
            llm_status = "failed"
            failure_type = str(
                (failure or {}).get(
                    "error_type",
                    (failure or {}).get("failure_type", "missing_evidence"),
                )
            )
            failure_message = str(
                (failure or {}).get(
                    "error",
                    (failure or {}).get(
                        "message",
                        "No LLM evidence row or explicit failure metadata was provided.",
                    ),
                )
            )
            if failure is not None:
                failure_source = failure.get("source_sentence")
                if failure_source is not None and str(failure_source) != source_sentence:
                    raise ValueError(f"failure source sentence changed for {key}")
        comparison.append(
            {
                "sample_id": key[0],
                "source_sentence_id": key[1],
                "source_rank": str(source.get("source_rank", "")),
                "source_sentence": source_sentence,
                "spacy_units": spacy_units,
                "spacy_units_json": json.dumps(spacy_units, ensure_ascii=False),
                "spacy_unit_count": len(spacy_units),
                "llm_status": llm_status,
                "llm_units": llm_units,
                "llm_units_json": json.dumps(llm_units, ensure_ascii=False),
                "llm_unit_count": len(llm_units),
                "llm_failure_type": failure_type,
                "llm_failure_message": failure_message,
                "spacy_atomicity": str(source.get("atomicity", "")),
                "spacy_faithfulness": str(source.get("faithfulness", "")),
                "spacy_type_correct": str(source.get("type_correct", "")),
                "spacy_notes": str(source.get("notes", "")),
                "llm_atomicity": "",
                "llm_faithfulness": "",
                "llm_type_correct": "",
                "preferred_extractor": "",
                "notes": "",
            }
        )
        base = {
            "sample_id": key[0],
            "source_sentence_id": key[1],
            "source_rank": source.get("source_rank", ""),
            "source_sentence": source_sentence,
        }
        spacy_audit_rows.append({**base, "units": spacy_units})
        llm_audit_rows.append({**base, "units": llm_units})
    extra = set(llm_by_key) - seen_spacy
    if extra:
        raise ValueError(f"LLM output contains unexpected frozen source sentences: {sorted(extra)}")
    extra_failures = set(failures_by_key) - seen_spacy
    if extra_failures:
        raise ValueError(
            f"LLM failures contain unexpected frozen source sentences: {sorted(extra_failures)}"
        )
    return comparison, spacy_audit_rows, llm_audit_rows


def comparison_coverage(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    success = sum(row.get("llm_status") == "success" for row in rows)
    failed = sum(row.get("llm_status") == "failed" for row in rows)
    return {
        "total_source_sentences": len(rows),
        "llm_success_rows": success,
        "llm_failed_rows": failed,
        "llm_completion_rate": success / len(rows) if rows else 0.0,
        "spaCy_rows": len(rows),
        "comparison_rows": len(rows),
    }


def token_matched_context_stats(
    records: Iterable[Mapping[str, Any]], units_field: str
) -> dict[str, Any]:
    """Greedily pack units up to source-sentence tokens, never adding padding."""
    by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_sample[str(row.get("sample_id", ""))].append(row)
    used_lengths: list[int] = []
    budgets: list[int] = []
    dropped_units = 0
    for sample_rows in by_sample.values():
        ordered = sorted(sample_rows, key=lambda row: int(row.get("source_rank") or 0))
        budget = sum(len(audit_tokens(str(row.get("source_sentence", "")))) for row in ordered)
        budgets.append(budget)
        used = 0
        for row in ordered:
            units = row.get(units_field, [])
            if not isinstance(units, list):
                continue
            for unit in units:
                length = len(audit_tokens(str(unit.get("text", ""))))
                if used + length <= budget:
                    used += length
                else:
                    dropped_units += 1
        used_lengths.append(used)
    return {
        "samples": len(by_sample),
        "source_context_token_budget": _summary(budgets),
        "token_matched_atomic_context": _summary(used_lengths),
        "units_dropped": dropped_units,
        "padding_tokens_added": 0,
        "audit_tokenizer": "unicode_regex_v1",
    }
