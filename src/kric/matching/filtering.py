"""Support-filtered atomic context construction for M4.2."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

from kric.evidence.llm_frozen_pipeline import _fit_closest


def _ordered_atomic_units(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    sample_id = str(row.get("sample_id", ""))
    sentence_ids = [str(value) for value in row.get("selected_sentence_ids", [])]
    if len(sentence_ids) != len(set(sentence_ids)):
        raise ValueError(f"duplicate selected source sentence IDs for {sample_id}")
    try:
        ordered_sentence_ids = sorted(sentence_ids, key=int)
    except ValueError as error:
        raise ValueError(f"non-numeric source sentence ID for {sample_id}") from error
    sentence_order = {
        sentence_id: index for index, sentence_id in enumerate(ordered_sentence_ids)
    }
    normalized: list[tuple[int, int, dict[str, Any]]] = []
    seen: set[str] = set()
    for unit_index, raw in enumerate(row.get("atomic_units", [])):
        unit = dict(raw)
        evidence_id = str(unit.get("evidence_id", ""))
        provenance = unit.get("provenance")
        if not evidence_id or evidence_id in seen or not isinstance(provenance, Mapping):
            raise ValueError(f"invalid or duplicate atomic evidence for {sample_id}")
        if str(provenance.get("sample_id", "")) != sample_id:
            raise ValueError(f"cross-sample atomic evidence {evidence_id} for {sample_id}")
        source_id = str(provenance.get("source_sentence_id", ""))
        if source_id not in sentence_order:
            raise ValueError(f"unresolved atomic source sentence for {evidence_id}")
        if not str(unit.get("text", "")).strip():
            raise ValueError(f"empty atomic evidence text for {evidence_id}")
        seen.add(evidence_id)
        normalized.append((sentence_order[source_id], unit_index, unit))
    normalized.sort(key=lambda value: (value[0], value[1]))
    return [unit for _, _, unit in normalized]


def build_support_filtered_contexts(
    *,
    primary_ids: Sequence[str],
    claim_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    atomic_rows: Sequence[Mapping[str, Any]],
    token_matched_rows: Sequence[Mapping[str, Any]],
    threshold: float,
    token_counter: Callable[[str], int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(primary_ids) != 49 or len(set(primary_ids)) != 49:
        raise ValueError("M4.2 requires the frozen ordered 49 primary IDs")
    if not math.isfinite(threshold):
        raise ValueError("support threshold must be finite")
    primary_set = set(primary_ids)

    claims_by_id: dict[str, Mapping[str, Any]] = {}
    claims_by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for claim in claim_rows:
        if str(claim.get("caption_variant")) != "atomic_token_matched":
            raise ValueError("filtered context claims must be atomic_token_matched only")
        sample_id = str(claim.get("sample_id", ""))
        claim_id = str(claim.get("claim_id", ""))
        if sample_id not in primary_set or not claim_id or claim_id in claims_by_id:
            raise ValueError(f"invalid, duplicate, or non-primary claim: {claim_id}")
        claims_by_id[claim_id] = claim
        claims_by_sample[sample_id].append(claim)
    if any(not claims_by_sample[sample_id] for sample_id in primary_ids):
        raise ValueError("every primary sample must have at least one token-matched claim")

    rankings_by_claim: dict[str, Mapping[str, Any]] = {}
    for ranking in ranking_rows:
        if str(ranking.get("caption_variant")) != "atomic_token_matched":
            continue
        claim_id = str(ranking.get("claim_id", ""))
        if not claim_id or claim_id in rankings_by_claim:
            raise ValueError(f"duplicate or empty token-matched ranking claim: {claim_id}")
        rankings_by_claim[claim_id] = ranking
    if set(rankings_by_claim) != set(claims_by_id):
        raise ValueError("token-matched claim and cosine ranking key sets differ")

    atomic_by_sample = {str(row.get("sample_id", "")): row for row in atomic_rows}
    budgets_by_sample = {
        str(row.get("sample_id", "")): row for row in token_matched_rows
    }
    if len(atomic_by_sample) != len(atomic_rows) or len(budgets_by_sample) != len(token_matched_rows):
        raise ValueError("duplicate atomic or token-matched context sample IDs")
    if any(sample_id not in atomic_by_sample or sample_id not in budgets_by_sample for sample_id in primary_ids):
        raise ValueError("atomic evidence or token-matched budget is missing a primary sample")

    output: list[dict[str, Any]] = []
    sample_audits: list[dict[str, Any]] = []
    fallback_count = 0
    for sample_id in primary_ids:
        ordered_units = _ordered_atomic_units(atomic_by_sample[sample_id])
        unit_by_id = {str(unit["evidence_id"]): unit for unit in ordered_units}
        qualifying: set[str] = set()
        top1_candidates: list[tuple[float, str]] = []
        claim_to_qualifying: dict[str, set[str]] = {}
        for claim in claims_by_sample[sample_id]:
            claim_id = str(claim["claim_id"])
            ranking = rankings_by_claim[claim_id]
            if str(ranking.get("sample_id", "")) != sample_id:
                raise ValueError(f"cross-sample ranking for claim {claim_id}")
            ranked = ranking.get("ranked_evidence")
            if not isinstance(ranked, list) or not ranked:
                raise ValueError(f"claim has no cosine evidence ranking: {claim_id}")
            passing: set[str] = set()
            for item in ranked:
                evidence_id = str(item.get("evidence_id", ""))
                unit = unit_by_id.get(evidence_id)
                if unit is None:
                    raise ValueError(f"ranking contains unknown or cross-sample evidence {evidence_id}")
                provenance = unit["provenance"]
                if (
                    str(item.get("evidence_text", "")) != str(unit.get("text", ""))
                    or str(item.get("evidence_type", "")) != str(unit.get("type", ""))
                    or str(item.get("source_sentence_id", ""))
                    != str(provenance.get("source_sentence_id", ""))
                ):
                    raise ValueError(f"ranking/evidence provenance mismatch for {evidence_id}")
                score = float(item.get("score"))
                if not math.isfinite(score):
                    raise ValueError(f"non-finite cosine score for {claim_id}:{evidence_id}")
                if score >= threshold:
                    passing.add(evidence_id)
                    qualifying.add(evidence_id)
            top = ranked[0]
            top1_candidates.append((float(top["score"]), str(top["evidence_id"])))
            claim_to_qualifying[claim_id] = passing

        threshold_qualifying = set(qualifying)
        selection_pool = set(threshold_qualifying)
        fallback_used = False
        fallback_evidence_id = None
        if not selection_pool:
            fallback_used = True
            fallback_count += 1
            fallback_evidence_id = max(top1_candidates, key=lambda value: (value[0], value[1]))[1]
            selection_pool.add(fallback_evidence_id)

        ordered_qualifying = [
            unit for unit in ordered_units if str(unit["evidence_id"]) in selection_pool
        ]
        if len(ordered_qualifying) != len(selection_pool):
            raise AssertionError("qualifying evidence was lost during deterministic ordering")
        budget_row = budgets_by_sample[sample_id]
        budget = int(budget_row.get("b2_context_token_budget", 0))
        if budget <= 0 or int(budget_row.get("context_tokens", 0)) > budget:
            raise ValueError(f"invalid frozen token-matched budget for {sample_id}")
        fitted, context, context_tokens, dropped_for_budget = _fit_closest(
            ordered_qualifying, token_counter, budget
        )
        included_ids = [str(unit["evidence_id"]) for unit in fitted]
        if selection_pool and not included_ids:
            raise ValueError(f"no selected evidence fits the frozen token budget for {sample_id}")
        if fallback_used and fallback_evidence_id not in included_ids:
            raise ValueError(f"deterministic fallback evidence does not fit budget for {sample_id}")
        included_set = set(included_ids)
        all_ids = [str(unit["evidence_id"]) for unit in ordered_units]
        threshold_dropped = [
            evidence_id for evidence_id in all_ids if evidence_id not in threshold_qualifying
        ]
        claims_covered_before_budget = sum(bool(value) for value in claim_to_qualifying.values())
        claims_covered_after_budget = sum(
            bool(value & included_set) for value in claim_to_qualifying.values()
        )

        row = {
            "sample_id": sample_id,
            "context_kind": "m4_support_filtered_token_matched",
            "context": context,
            "context_tokens": context_tokens,
            "b2_context_token_budget": budget,
            "threshold": threshold,
            "included_evidence_ids": included_ids,
            "threshold_eligible_evidence_ids": [
                str(unit["evidence_id"])
                for unit in ordered_units
                if str(unit["evidence_id"]) in threshold_qualifying
            ],
            "threshold_dropped_evidence_ids": threshold_dropped,
            "budget_dropped_evidence_ids": dropped_for_budget,
            "dropped_evidence_ids": [
                evidence_id for evidence_id in all_ids if evidence_id not in included_set
            ],
            "fallback_used": fallback_used,
            "fallback_evidence_id": fallback_evidence_id,
            "claim_count": len(claim_to_qualifying),
            "claims_covered_before_budget": claims_covered_before_budget,
            "claims_covered_after_budget": claims_covered_after_budget,
            "claim_coverage": claims_covered_after_budget / len(claim_to_qualifying),
        }
        output.append(row)
        sample_audits.append(
            {
                "sample_id": sample_id,
                "atomic_evidence_count": len(all_ids),
                "threshold_eligible_count": len(threshold_qualifying),
                "evidence_retained_count": len(included_ids),
                "dropped_evidence_count": len(all_ids) - len(included_ids),
                "threshold_dropped_count": len(threshold_dropped),
                "budget_dropped_count": len(dropped_for_budget),
                "context_tokens": context_tokens,
                "token_budget": budget,
                "fallback_used": fallback_used,
                "claim_count": len(claim_to_qualifying),
                "claims_covered_after_budget": claims_covered_after_budget,
                "claim_coverage": row["claim_coverage"],
            }
        )

    token_counts = [int(row["context_tokens"]) for row in output]
    retained_counts = [len(row["included_evidence_ids"]) for row in output]
    audit = {
        "samples": len(output),
        "threshold": threshold,
        "fallback_count": fallback_count,
        "cross_sample_evidence_count": 0,
        "samples_over_budget": 0,
        "evidence_retained": {
            "total": sum(retained_counts),
            "mean": statistics.fmean(retained_counts),
            "median": statistics.median(retained_counts),
            "min": min(retained_counts),
            "max": max(retained_counts),
        },
        "context_tokens": {
            "mean": statistics.fmean(token_counts),
            "median": statistics.median(token_counts),
            "min": min(token_counts),
            "max": max(token_counts),
        },
        "mean_claim_coverage": statistics.fmean(
            float(row["claim_coverage"]) for row in output
        ),
        "per_sample": sample_audits,
    }
    return output, audit


__all__ = ["build_support_filtered_contexts"]
