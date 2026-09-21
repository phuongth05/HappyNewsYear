import pytest

from kric.matching.filtering import build_support_filtered_contexts


def _inputs():
    primary_ids = [f"s{index:02d}" for index in range(49)]
    claims = []
    rankings = []
    atomic = []
    matched = []
    for sample_index, sample_id in enumerate(primary_ids):
        units = [
            {
                "evidence_id": f"{sample_id}-e0a",
                "text": f"Alpha {sample_id}.",
                "type": "entity",
                "provenance": {"sample_id": sample_id, "source_sentence_id": "0"},
            },
            {
                "evidence_id": f"{sample_id}-e0b",
                "text": f"Beta detail {sample_id}.",
                "type": "attribute",
                "provenance": {"sample_id": sample_id, "source_sentence_id": "0"},
            },
            {
                "evidence_id": f"{sample_id}-e1",
                "text": f"Gamma event {sample_id} happened.",
                "type": "event",
                "provenance": {"sample_id": sample_id, "source_sentence_id": "1"},
            },
        ]
        atomic.append(
            {
                "sample_id": sample_id,
                "selected_sentence_ids": [0, 1],
                "atomic_units": units,
            }
        )
        matched.append(
            {
                "sample_id": sample_id,
                "context_tokens": 8,
                "b2_context_token_budget": 8,
            }
        )
        claim_count = 2 if sample_index == 0 else 1
        for claim_index in range(claim_count):
            claim_id = f"{sample_id}-c{claim_index}"
            claims.append(
                {
                    "sample_id": sample_id,
                    "caption_variant": "atomic_token_matched",
                    "claim_id": claim_id,
                    "claim_text": "A claim.",
                    "claim_type": "event",
                }
            )
            scores = (0.8, 0.7, 0.2) if claim_index == 0 else (0.75, 0.65, 0.1)
            order = (1, 0, 2) if claim_index == 0 else (1, 2, 0)
            ranked = []
            for rank, (unit_index, score) in enumerate(zip(order, scores, strict=True), start=1):
                unit = units[unit_index]
                ranked.append(
                    {
                        "evidence_id": unit["evidence_id"],
                        "evidence_text": unit["text"],
                        "evidence_type": unit["type"],
                        "source_sentence_id": unit["provenance"]["source_sentence_id"],
                        "score": score,
                        "rank": rank,
                    }
                )
            rankings.append(
                {
                    "sample_id": sample_id,
                    "caption_variant": "atomic_token_matched",
                    "claim_id": claim_id,
                    "ranked_evidence": ranked,
                }
            )
    return primary_ids, claims, rankings, atomic, matched


def _build(*, threshold=0.6, mutate=None):
    values = list(_inputs())
    if mutate:
        mutate(values)
    return build_support_filtered_contexts(
        primary_ids=values[0],
        claim_rows=values[1],
        ranking_rows=values[2],
        atomic_rows=values[3],
        token_matched_rows=values[4],
        threshold=threshold,
        token_counter=lambda text: len(text.split()),
    )


def test_same_sample_duplicate_removal_order_and_budget_enforcement():
    contexts, audit = _build()
    first = contexts[0]
    assert len(contexts) == 49
    assert first["included_evidence_ids"] == ["s00-e0a", "s00-e0b"]
    assert len(first["included_evidence_ids"]) == len(set(first["included_evidence_ids"]))
    assert first["context_tokens"] <= first["b2_context_token_budget"]
    assert audit["samples_over_budget"] == 0
    assert audit["cross_sample_evidence_count"] == 0


def test_source_sentence_then_within_sentence_order_is_deterministic():
    def mutate(values):
        values[3][0]["selected_sentence_ids"] = [1, 0]
        values[4][0]["b2_context_token_budget"] = 20

    contexts, _ = _build(threshold=0.1, mutate=mutate)
    assert contexts[0]["included_evidence_ids"] == ["s00-e0a", "s00-e0b", "s00-e1"]

def test_fallback_uses_highest_top1_and_is_recorded():
    contexts, audit = _build(threshold=0.99)
    first = contexts[0]
    assert first["fallback_used"] is True
    assert first["fallback_evidence_id"] == "s00-e0b"
    assert first["included_evidence_ids"] == ["s00-e0b"]
    assert audit["fallback_count"] == 49


def test_tight_budget_drops_evidence_deterministically():
    def mutate(values):
        for row in values[4]:
            row["context_tokens"] = 4
            row["b2_context_token_budget"] = 4

    first, _ = _build(mutate=mutate)
    second, _ = _build(mutate=mutate)
    assert first == second
    assert all(row["context_tokens"] <= 4 for row in first)
    assert any(row["budget_dropped_evidence_ids"] for row in first)


def test_cross_sample_ranking_is_rejected():
    def mutate(values):
        values[2][0]["sample_id"] = "s01"

    with pytest.raises(ValueError, match="cross-sample ranking"):
        _build(mutate=mutate)


def test_ranking_evidence_provenance_mismatch_is_rejected():
    def mutate(values):
        values[2][0]["ranked_evidence"][0]["evidence_text"] = "Altered evidence."

    with pytest.raises(ValueError, match="provenance mismatch"):
        _build(mutate=mutate)
