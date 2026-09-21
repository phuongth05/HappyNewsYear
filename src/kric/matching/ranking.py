"""Deterministic same-sample evidence ranking for caption claims."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def evidence_by_sample(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in result:
            raise ValueError(f"empty or duplicate atomic-evidence sample: {sample_id!r}")
        units = row.get("atomic_units")
        if not isinstance(units, list):
            raise ValueError(f"atomic_units must be a list for {sample_id}")
        normalized = []
        seen: set[str] = set()
        for unit in units:
            evidence_id = str(unit.get("evidence_id", ""))
            provenance = unit.get("provenance", {})
            if not evidence_id or evidence_id in seen:
                raise ValueError(f"empty or duplicate evidence ID for {sample_id}")
            if str(provenance.get("sample_id")) != sample_id:
                raise ValueError(f"cross-sample evidence provenance for {sample_id}")
            seen.add(evidence_id)
            normalized.append(
                {
                    "evidence_id": evidence_id,
                    "evidence_text": str(unit["text"]),
                    "evidence_type": str(unit["type"]),
                    "source_sentence_id": str(provenance.get("source_sentence_id", "")),
                }
            )
        result[sample_id] = normalized
    return result


def rank_claims_against_same_sample_evidence(
    claims: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    scorer: Any,
) -> list[dict[str, Any]]:
    evidence = evidence_by_sample(evidence_rows)
    seen_claims: set[str] = set()
    output: list[dict[str, Any]] = []
    for claim in claims:
        sample_id = str(claim.get("sample_id", ""))
        claim_id = str(claim.get("claim_id", ""))
        if not sample_id or sample_id not in evidence:
            raise ValueError(f"claim has no same-sample evidence: {sample_id!r}")
        if not claim_id or claim_id in seen_claims:
            raise ValueError(f"empty or duplicate claim ID: {claim_id!r}")
        seen_claims.add(claim_id)
        candidates = evidence[sample_id]
        pairs = [(item["evidence_text"], str(claim["claim_text"])) for item in candidates]
        scores = scorer.score_pairs(pairs)
        if len(scores) != len(candidates):
            raise RuntimeError("matcher returned the wrong number of pair scores")
        ranked = []
        for item, score in zip(candidates, scores, strict=True):
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(f"non-finite matcher score for {claim_id}")
            ranked.append({**item, "score": value})
        ranked.sort(key=lambda item: (-item["score"], item["evidence_id"]))
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        output.append(
            {
                "sample_id": sample_id,
                "caption_variant": str(claim["caption_variant"]),
                "claim_id": claim_id,
                "claim_text": str(claim["claim_text"]),
                "claim_type": str(claim["claim_type"]),
                "ranked_evidence": ranked,
                "best_score": ranked[0]["score"] if ranked else None,
                "top1_evidence_id": ranked[0]["evidence_id"] if ranked else None,
                "top3_evidence_ids": [item["evidence_id"] for item in ranked[:3]],
            }
        )
    return output


def ranking_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "claims": len(rows),
        "caption_variants": dict(
            sorted(Counter(str(row["caption_variant"]) for row in rows).items())
        ),
        "claim_types": dict(sorted(Counter(str(row["claim_type"]) for row in rows).items())),
        "claims_without_candidate_evidence": sum(
            not row.get("ranked_evidence") for row in rows
        ),
        "same_sample_only": True,
    }


__all__ = [
    "evidence_by_sample",
    "rank_claims_against_same_sample_evidence",
    "ranking_audit",
    "read_jsonl",
    "write_jsonl",
]