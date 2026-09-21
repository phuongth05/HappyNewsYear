"""Deterministic balanced sampling for M4 claim-support human review."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict, deque
from typing import Any, Mapping, Sequence

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN.findall(text)}


def _near_duplicate(text: str, selected: Sequence[str], threshold: float = 0.9) -> bool:
    tokens = _tokens(text)
    normalized = " ".join(text.casefold().split())
    for prior in selected:
        if normalized == " ".join(prior.casefold().split()):
            return True
        other = _tokens(prior)
        union = tokens | other
        if union and len(tokens & other) / len(union) >= threshold:
            return True
    return False


def sample_review_claims(
    claims: Sequence[Mapping[str, Any]],
    cosine_rows: Sequence[Mapping[str, Any]],
    nli_rows: Sequence[Mapping[str, Any]],
    *,
    target: int = 120,
    seed: int = 2026,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target <= 0:
        raise ValueError("review target must be positive")
    claims_by_id = {str(row["claim_id"]): row for row in claims}
    cosine = {str(row["claim_id"]): row for row in cosine_rows}
    nli = {str(row["claim_id"]): row for row in nli_rows}
    if len(claims_by_id) != len(claims) or set(claims_by_id) != set(cosine) or set(cosine) != set(nli):
        raise ValueError("claim and matcher key sets must be identical and unique")
    variants = ("b2", "atomic_full", "atomic_token_matched")
    rng = random.Random(seed)
    groups: dict[tuple[str, str], deque[str]] = {}
    types_by_variant: dict[str, list[str]] = defaultdict(list)
    raw_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for claim_id, claim in claims_by_id.items():
        variant = str(claim["caption_variant"])
        if variant not in variants:
            raise ValueError(f"unknown caption variant: {variant}")
        raw_groups[(variant, str(claim["claim_type"]))].append(claim_id)
    for key, values in raw_groups.items():
        values.sort()
        rng.shuffle(values)
        groups[key] = deque(values)
        types_by_variant[key[0]].append(key[1])
    for variant in variants:
        types_by_variant[variant] = sorted(set(types_by_variant[variant]))
    quota = {variant: target // len(variants) for variant in variants}
    for variant in variants[: target % len(variants)]:
        quota[variant] += 1
    chosen: list[str] = []
    chosen_texts: list[str] = []
    counts: Counter[str] = Counter()
    progress = True
    while len(chosen) < target and progress:
        progress = False
        for variant in variants:
            if counts[variant] >= quota[variant]:
                continue
            for claim_type in types_by_variant[variant]:
                queue = groups[(variant, claim_type)]
                while queue:
                    claim_id = queue.popleft()
                    text = str(claims_by_id[claim_id]["claim_text"])
                    if _near_duplicate(text, chosen_texts):
                        continue
                    chosen.append(claim_id)
                    chosen_texts.append(text)
                    counts[variant] += 1
                    progress = True
                    break
                if counts[variant] >= quota[variant] or len(chosen) >= target:
                    break
    rows = []
    for claim_id in chosen:
        claim = claims_by_id[claim_id]
        cosine_row = cosine[claim_id]
        nli_row = nli[claim_id]
        cosine_top = cosine_row.get("ranked_evidence", [])[:1]
        nli_top = nli_row.get("ranked_evidence", [])[:1]
        union = []
        seen: set[str] = set()
        for matcher_name, ranking in (
            ("cosine", cosine_row.get("ranked_evidence", [])[:3]),
            ("nli", nli_row.get("ranked_evidence", [])[:3]),
        ):
            for item in ranking:
                evidence_id = str(item["evidence_id"])
                if evidence_id in seen:
                    continue
                seen.add(evidence_id)
                union.append({**dict(item), "selected_by": matcher_name})
        rows.append(
            {
                "sample_id": claim["sample_id"],
                "caption_variant": claim["caption_variant"],
                "caption": claim["caption"],
                "claim_id": claim_id,
                "claim_text": claim["claim_text"],
                "claim_type": claim["claim_type"],
                "cosine_top1_id": cosine_top[0]["evidence_id"] if cosine_top else "",
                "cosine_top1_text": cosine_top[0]["evidence_text"] if cosine_top else "",
                "cosine_top1_score": cosine_top[0]["score"] if cosine_top else "",
                "nli_top1_id": nli_top[0]["evidence_id"] if nli_top else "",
                "nli_top1_text": nli_top[0]["evidence_text"] if nli_top else "",
                "nli_top1_score": nli_top[0]["score"] if nli_top else "",
                "union_top3_evidence_json": json.dumps(union, ensure_ascii=False),
                "support_label": "",
                "best_evidence_ids": "",
                "matcher_preference": "",
                "notes": "",
            }
        )
    manifest = {
        "seed": seed,
        "requested_claims": target,
        "selected_claims": len(rows),
        "caption_variants": dict(sorted(Counter(row["caption_variant"] for row in rows).items())),
        "claim_types": dict(sorted(Counter(row["claim_type"] for row in rows).items())),
        "near_duplicate_jaccard_threshold": 0.9,
        "human_fields_prefilled": False,
    }
    return rows, manifest


__all__ = ["sample_review_claims"]