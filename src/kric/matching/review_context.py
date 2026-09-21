"""Provenance-safe context enrichment for the frozen M4 review set."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kric.evaluation.io import file_sha256


HUMAN_FIELDS = ("support_label", "best_evidence_ids", "matcher_preference", "notes")
ENRICHMENT_FIELDS = (
    "cosine_top1_source_sentence_id",
    "cosine_top1_source_sentence_text",
    "nli_top1_source_sentence_id",
    "nli_top1_source_sentence_text",
    "union_top3_evidence_context_json",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def read_review_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("review CSV has no header")
        return list(reader.fieldnames), [dict(row) for row in reader]


@dataclass(frozen=True)
class FrozenEvidence:
    sample_id: str
    evidence_id: str
    text: str
    evidence_type: str
    source_sentence_id: str
    source_sentence_text: str


class ProvenanceResolutionError(ValueError):
    """Raised when frozen evidence provenance cannot be resolved exactly."""


def _source_key(sample_id: Any, sentence_id: Any) -> tuple[str, str]:
    return str(sample_id), str(sentence_id)


def build_frozen_indexes(
    atomic_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    context_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, FrozenEvidence], dict[tuple[str, str], str]]:
    selected_sources: dict[tuple[str, str], str] = {}
    selected_samples: set[str] = set()
    for row in selected_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in selected_samples:
            raise ProvenanceResolutionError(f"duplicate or empty selected-evidence sample_id: {sample_id!r}")
        selected_samples.add(sample_id)
        sentence_ids = row.get("selected_sentence_ids")
        sentence_texts = row.get("selected_sentence_texts")
        if not isinstance(sentence_ids, list) or not isinstance(sentence_texts, list):
            raise ProvenanceResolutionError(f"invalid selected sentence arrays for sample {sample_id}")
        if len(sentence_ids) != len(sentence_texts):
            raise ProvenanceResolutionError(f"selected sentence array length mismatch for sample {sample_id}")
        for sentence_id, sentence_text in zip(sentence_ids, sentence_texts, strict=True):
            key = _source_key(sample_id, sentence_id)
            text = str(sentence_text)
            if not text:
                raise ProvenanceResolutionError(f"empty selected source sentence for {key}")
            if key in selected_sources and selected_sources[key] != text:
                raise ProvenanceResolutionError(f"source sentence differs across selected evidence for {key}")
            selected_sources[key] = text

    evidence_by_id: dict[str, FrozenEvidence] = {}
    atomic_samples: set[str] = set()
    for row in atomic_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in atomic_samples:
            raise ProvenanceResolutionError(f"duplicate or empty atomic-evidence sample_id: {sample_id!r}")
        atomic_samples.add(sample_id)
        units = row.get("atomic_units")
        if not isinstance(units, list):
            raise ProvenanceResolutionError(f"atomic_units must be a list for sample {sample_id}")
        for unit in units:
            if not isinstance(unit, Mapping):
                raise ProvenanceResolutionError(f"invalid atomic unit for sample {sample_id}")
            evidence_id = str(unit.get("evidence_id", ""))
            if not evidence_id or evidence_id in evidence_by_id:
                raise ProvenanceResolutionError(f"duplicate or empty evidence_id: {evidence_id!r}")
            provenance = unit.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ProvenanceResolutionError(f"missing provenance for evidence {evidence_id}")
            provenance_sample = str(provenance.get("sample_id", ""))
            if provenance_sample != sample_id:
                raise ProvenanceResolutionError(
                    f"evidence {evidence_id} belongs to sample {provenance_sample}, not {sample_id}"
                )
            sentence_id = str(provenance.get("source_sentence_id", ""))
            key = _source_key(sample_id, sentence_id)
            if key not in selected_sources:
                raise ProvenanceResolutionError(f"source_sentence_id cannot be resolved for evidence {evidence_id}: {key}")
            source_text = str(provenance.get("source_sentence", ""))
            if source_text != selected_sources[key]:
                raise ProvenanceResolutionError(f"source sentence differs across frozen artifacts for {key}")
            evidence_text = str(unit.get("text", ""))
            evidence_type = str(unit.get("type", ""))
            if not evidence_text or not evidence_type:
                raise ProvenanceResolutionError(f"incomplete atomic evidence {evidence_id}")
            evidence_by_id[evidence_id] = FrozenEvidence(
                sample_id=sample_id,
                evidence_id=evidence_id,
                text=evidence_text,
                evidence_type=evidence_type,
                source_sentence_id=sentence_id,
                source_sentence_text=source_text,
            )

    context_samples: set[str] = set()
    for row in context_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in context_samples:
            raise ProvenanceResolutionError(f"duplicate or empty atomic-context sample_id: {sample_id!r}")
        context_samples.add(sample_id)
        for evidence_id_value in row.get("included_evidence_ids", []):
            evidence_id = str(evidence_id_value)
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                raise ProvenanceResolutionError(f"atomic context contains unresolved evidence_id {evidence_id}")
            if evidence.sample_id != sample_id:
                raise ProvenanceResolutionError(
                    f"atomic context introduces cross-sample evidence {evidence_id}: "
                    f"{evidence.sample_id} != {sample_id}"
                )
    if atomic_samples != selected_samples or context_samples != atomic_samples:
        raise ProvenanceResolutionError("frozen source artifacts do not contain identical sample sets")
    return evidence_by_id, selected_sources


def _parse_union(value: str, *, claim_id: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProvenanceResolutionError(f"malformed union evidence JSON for claim {claim_id}") from exc
    if not isinstance(parsed, list):
        raise ProvenanceResolutionError(f"union evidence must be a list for claim {claim_id}")
    if any(not isinstance(item, dict) for item in parsed):
        raise ProvenanceResolutionError(f"union evidence entries must be objects for claim {claim_id}")
    return parsed


def _record_matcher(
    record: dict[str, Any], matcher: str, rank: Any, score: Any
) -> None:
    selected_by = record.setdefault("selected_by", [])
    if matcher not in selected_by:
        selected_by.append(matcher)
    rank_key = f"{matcher}_rank"
    score_key = f"{matcher}_score"
    normalized_rank = int(rank)
    normalized_score = float(score)
    if record.get(rank_key) not in (None, normalized_rank):
        raise ProvenanceResolutionError(
            f"conflicting {rank_key} for evidence {record['evidence_id']}"
        )
    if record.get(score_key) is not None and float(record[score_key]) != normalized_score:
        raise ProvenanceResolutionError(
            f"conflicting {score_key} for evidence {record['evidence_id']}"
        )
    record[rank_key] = normalized_rank
    record[score_key] = normalized_score


def enrich_review_rows(
    rows: Sequence[Mapping[str, str]],
    evidence_by_id: Mapping[str, FrozenEvidence],
) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []
    for source_row in rows:
        row = dict(source_row)
        sample_id = str(row.get("sample_id", ""))
        claim_id = str(row.get("claim_id", ""))
        union = _parse_union(row.get("union_top3_evidence_json", ""), claim_id=claim_id)
        records: OrderedDict[str, dict[str, Any]] = OrderedDict()

        def resolve(evidence_id: str, expected_text: str | None = None) -> FrozenEvidence:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                raise ProvenanceResolutionError(f"evidence_id cannot be resolved: {evidence_id}")
            if evidence.sample_id != sample_id:
                raise ProvenanceResolutionError(
                    f"evidence {evidence_id} belongs to another sample: "
                    f"{evidence.sample_id} != {sample_id}"
                )
            if expected_text is not None and expected_text != evidence.text:
                raise ProvenanceResolutionError(f"evidence text mismatch for {evidence_id}")
            return evidence

        for item in union:
            evidence_id = str(item.get("evidence_id", ""))
            evidence = resolve(evidence_id, str(item.get("evidence_text", "")))
            if str(item.get("evidence_type", "")) != evidence.evidence_type:
                raise ProvenanceResolutionError(f"evidence type mismatch for {evidence_id}")
            if str(item.get("source_sentence_id", "")) != evidence.source_sentence_id:
                raise ProvenanceResolutionError(f"source_sentence_id mismatch for {evidence_id}")
            record = records.setdefault(
                evidence_id,
                {
                    "evidence_id": evidence.evidence_id,
                    "evidence_text": evidence.text,
                    "evidence_type": evidence.evidence_type,
                    "source_sentence_id": evidence.source_sentence_id,
                    "source_sentence_text": evidence.source_sentence_text,
                    "selected_by": [],
                    "cosine_rank": None,
                    "nli_rank": None,
                    "cosine_score": None,
                    "nli_score": None,
                },
            )
            matcher = str(item.get("selected_by", ""))
            if matcher not in {"cosine", "nli"}:
                raise ProvenanceResolutionError(f"unknown matcher for evidence {evidence_id}: {matcher}")
            _record_matcher(record, matcher, item.get("rank"), item.get("score"))

        top1_values: dict[str, FrozenEvidence] = {}
        for matcher in ("cosine", "nli"):
            evidence_id = str(row.get(f"{matcher}_top1_id", ""))
            evidence = resolve(evidence_id, str(row.get(f"{matcher}_top1_text", "")))
            top1_values[matcher] = evidence
            record = records.setdefault(
                evidence_id,
                {
                    "evidence_id": evidence.evidence_id,
                    "evidence_text": evidence.text,
                    "evidence_type": evidence.evidence_type,
                    "source_sentence_id": evidence.source_sentence_id,
                    "source_sentence_text": evidence.source_sentence_text,
                    "selected_by": [],
                    "cosine_rank": None,
                    "nli_rank": None,
                    "cosine_score": None,
                    "nli_score": None,
                },
            )
            _record_matcher(record, matcher, 1, row[f"{matcher}_top1_score"])

        row["cosine_top1_source_sentence_id"] = top1_values["cosine"].source_sentence_id
        row["cosine_top1_source_sentence_text"] = top1_values["cosine"].source_sentence_text
        row["nli_top1_source_sentence_id"] = top1_values["nli"].source_sentence_id
        row["nli_top1_source_sentence_text"] = top1_values["nli"].source_sentence_text
        row["union_top3_evidence_context_json"] = json.dumps(
            list(records.values()), ensure_ascii=False, separators=(",", ":")
        )
        enriched.append(row)
    return enriched


def write_review_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def source_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in paths.items()}


__all__ = [
    "ENRICHMENT_FIELDS",
    "HUMAN_FIELDS",
    "FrozenEvidence",
    "ProvenanceResolutionError",
    "build_frozen_indexes",
    "enrich_review_rows",
    "read_jsonl",
    "read_review_csv",
    "source_hashes",
    "write_review_csv",
]
