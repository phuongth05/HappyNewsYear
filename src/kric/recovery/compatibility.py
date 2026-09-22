from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

TOKEN = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    return " ".join(TOKEN.findall(text.casefold()))


def jaccard(left: str, right: str) -> float:
    a, b = set(TOKEN.findall(left.casefold())), set(TOKEN.findall(right.casefold()))
    return len(a & b) / len(a | b) if a or b else 1.0


def map_reviewed_claims(
    originals: Sequence[Mapping[str, Any]], regenerated: Sequence[Mapping[str, Any]],
    cosine_scorer: Callable[[Sequence[tuple[str, str]]], Sequence[float]],
) -> list[dict[str, Any]]:
    original_ids = [str(row.get("claim_id", "")) for row in originals]
    if any(not claim_id for claim_id in original_ids) or len(original_ids) != len(set(original_ids)):
        raise ValueError("empty or duplicate original claim ID")
    by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    regen_ids: set[str] = set()
    for row in regenerated:
        claim_id = str(row.get("claim_id", ""))
        if not claim_id or claim_id in regen_ids: raise ValueError("duplicate regenerated claim ID")
        regen_ids.add(claim_id); by_sample[str(row.get("sample_id", ""))].append(row)
    candidates: list[tuple[tuple[Any, ...], int, str, Mapping[str, Any], float, float, bool]] = []
    for index, old in enumerate(originals):
        same = by_sample.get(str(old.get("sample_id", "")), [])
        pairs = [(str(old.get("claim_text", "")), str(new.get("claim_text", ""))) for new in same]
        scores = list(cosine_scorer(pairs)) if pairs else []
        for new, cosine in zip(same, scores, strict=True):
            exact = normalize(str(old.get("claim_text", ""))) == normalize(str(new.get("claim_text", "")))
            jac = jaccard(str(old.get("claim_text", "")), str(new.get("claim_text", "")))
            variant = str(old.get("caption_variant", "")) == str(new.get("caption_variant", ""))
            priority = (int(variant), int(exact), float(cosine), jac, str(new["claim_id"]))
            candidates.append((priority, index, str(new["claim_id"]), new, float(cosine), jac, exact))
    candidates.sort(key=lambda item: item[0], reverse=True)
    assigned_old: set[int] = set(); assigned_new: set[str] = set(); chosen: dict[int, tuple] = {}
    for item in candidates:
        _, index, claim_id, *_ = item
        if index not in assigned_old and claim_id not in assigned_new:
            assigned_old.add(index); assigned_new.add(claim_id); chosen[index] = item
    output = []
    for index, old in enumerate(originals):
        item = chosen.get(index)
        if item is None:
            output.append({**dict(old), "regenerated_claim_id": "", "regenerated_claim_text": "", "regenerated_caption_variant": "", "variant_match": False, "normalized_exact_match": False, "token_jaccard": None, "sentence_transformer_cosine": None, "compatibility_band": "unmatched", "high_confidence_mapping": False})
            continue
        _, _, _, new, cosine, jac, exact = item
        band = "exact_normalized" if exact else "cosine_ge_0.95" if cosine >= .95 else "cosine_0.90_0.95" if cosine >= .90 else "cosine_0.80_0.90" if cosine >= .80 else "cosine_lt_0.80"
        output.append({**dict(old), "regenerated_claim_id": new["claim_id"], "regenerated_claim_text": new.get("claim_text", ""), "regenerated_caption_variant": new.get("caption_variant", ""), "variant_match": str(old.get("caption_variant", "")) == str(new.get("caption_variant", "")), "normalized_exact_match": exact, "token_jaccard": jac, "sentence_transformer_cosine": cosine, "compatibility_band": band, "high_confidence_mapping": exact or cosine >= .95})
    return output


def compatibility_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows); exact = sum(bool(row["normalized_exact_match"]) for row in rows); high = sum(bool(row["high_confidence_mapping"]) for row in rows)
    def grouped(field: str) -> dict[str, Any]:
        result = {}
        for value in sorted({str(row.get(field, "")) for row in rows}):
            subset = [row for row in rows if str(row.get(field, "")) == value]
            result[value] = {"count": len(subset), "high_confidence_rate": sum(bool(row["high_confidence_mapping"]) for row in subset) / len(subset)}
        return result
    cosines = [float(row["sentence_transformer_cosine"]) for row in rows if row["sentence_transformer_cosine"] is not None]
    return {"reviewed_claims": total, "exact_match_rate": exact / total if total else 0, "high_semantic_match_rate": high / total if total else 0, "unmatched_reviewed_claims": total - sum(bool(row["regenerated_claim_id"]) for row in rows), "bands": dict(Counter(str(row["compatibility_band"]) for row in rows)), "per_variant": grouped("caption_variant"), "per_claim_type": grouped("claim_type"), "cosine": {"mean": statistics.fmean(cosines) if cosines else None, "median": statistics.median(cosines) if cosines else None}, "annotation_transfer_performed": False}
