import copy

import pytest

from kric.matching.nli import NliMatcher
from kric.matching.ranking import rank_claims_against_same_sample_evidence
from kric.matching.review import sample_review_claims


class RecordingScorer:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def score_pairs(self, pairs):
        self.pairs = list(pairs)
        return list(self.scores)


def _evidence():
    return [
        {
            "sample_id": "s1",
            "atomic_units": [
                {
                    "evidence_id": "e-b",
                    "text": "Alice spoke in Paris.",
                    "type": "event",
                    "provenance": {"sample_id": "s1", "source_sentence_id": "2"},
                },
                {
                    "evidence_id": "e-a",
                    "text": "Alice attended the summit.",
                    "type": "event",
                    "provenance": {"sample_id": "s1", "source_sentence_id": "1"},
                },
            ],
        },
        {
            "sample_id": "s2",
            "atomic_units": [
                {
                    "evidence_id": "other",
                    "text": "Bob was elsewhere.",
                    "type": "location",
                    "provenance": {"sample_id": "s2", "source_sentence_id": "0"},
                }
            ],
        },
    ]


def test_same_sample_only_nli_ordering_and_topk_are_deterministic():
    claim = {
        "sample_id": "s1",
        "caption_variant": "b2",
        "claim_id": "c1",
        "claim_text": "Alice was in Paris.",
        "claim_type": "location",
    }
    scorer = RecordingScorer([0.8, 0.8])
    first = rank_claims_against_same_sample_evidence([claim], _evidence(), scorer)
    assert scorer.pairs == [
        ("Alice spoke in Paris.", "Alice was in Paris."),
        ("Alice attended the summit.", "Alice was in Paris."),
    ]
    ranked = first[0]["ranked_evidence"]
    assert [row["evidence_id"] for row in ranked] == ["e-a", "e-b"]
    assert first[0]["top1_evidence_id"] == "e-a"
    assert first[0]["top3_evidence_ids"] == ["e-a", "e-b"]
    assert all(row["evidence_id"] != "other" for row in ranked)
    second = rank_claims_against_same_sample_evidence(
        [claim], _evidence(), RecordingScorer([0.8, 0.8])
    )
    assert first == second


def test_cross_sample_evidence_provenance_is_rejected():
    evidence = _evidence()
    evidence[0]["atomic_units"][0]["provenance"]["sample_id"] = "s2"
    with pytest.raises(ValueError, match="cross-sample"):
        rank_claims_against_same_sample_evidence(
            [
                {
                    "sample_id": "s1",
                    "caption_variant": "b2",
                    "claim_id": "c",
                    "claim_text": "claim",
                    "claim_type": "event",
                }
            ],
            evidence,
            RecordingScorer([1.0, 0.0]),
        )


def _review_inputs():
    claims = []
    cosine = []
    nli = []
    variants = ("b2", "atomic_full", "atomic_token_matched")
    claim_types = ("entity", "event", "location")
    for variant_index, variant in enumerate(variants):
        for index in range(4):
            claim_id = f"{variant}-{index}"
            claim = {
                "sample_id": f"s-{variant_index}-{index}",
                "caption_variant": variant,
                "caption": f"Caption {variant_index} {index}",
                "claim_id": claim_id,
                "claim_text": f"Unique factual {variant} item word{index}",
                "claim_type": claim_types[index % len(claim_types)],
            }
            claims.append(claim)
            ranking = [
                {
                    "evidence_id": f"e-{claim_id}-{rank}",
                    "evidence_text": f"Evidence {claim_id} {rank}",
                    "evidence_type": "event",
                    "source_sentence_id": str(rank),
                    "score": 1.0 - rank / 10,
                    "rank": rank + 1,
                }
                for rank in range(3)
            ]
            base = {key: claim[key] for key in ("sample_id", "caption_variant", "claim_id", "claim_text", "claim_type")}
            cosine.append({**base, "ranked_evidence": copy.deepcopy(ranking)})
            nli.append({**base, "ranked_evidence": list(reversed(copy.deepcopy(ranking)))})
    return claims, cosine, nli


def test_review_sampling_is_deterministic_balanced_and_labels_blank():
    claims, cosine, nli = _review_inputs()
    rows_a, manifest_a = sample_review_claims(claims, cosine, nli, target=9, seed=2026)
    rows_b, manifest_b = sample_review_claims(claims, cosine, nli, target=9, seed=2026)
    assert rows_a == rows_b
    assert manifest_a == manifest_b
    assert manifest_a["caption_variants"] == {
        "atomic_full": 3,
        "atomic_token_matched": 3,
        "b2": 3,
    }
    assert all(
        row[field] == ""
        for row in rows_a
        for field in ("support_label", "best_evidence_ids", "matcher_preference", "notes")
    )
    assert all(len(__import__("json").loads(row["union_top3_evidence_json"])) >= 3 for row in rows_a)

def test_nli_matcher_passes_premise_before_hypothesis():
    import torch

    class Tokenizer:
        def __init__(self):
            self.received = None

        def __call__(self, premises, hypotheses, **kwargs):
            self.received = (premises, hypotheses, kwargs)
            return {"input_ids": torch.tensor([[1, 2]])}

    class Output:
        logits = torch.tensor([[0.0, 2.0, 0.0]])

    class Model:
        def __call__(self, **_kwargs):
            return Output()

    matcher = NliMatcher(batch_size=1)
    matcher.tokenizer = Tokenizer()
    matcher.model = Model()
    matcher.device = "cpu"
    scores = matcher.score_pairs([("Evidence premise.", "Claim hypothesis.")])
    assert matcher.tokenizer.received[0] == ["Evidence premise."]
    assert matcher.tokenizer.received[1] == ["Claim hypothesis."]
    assert scores[0] > 0.7