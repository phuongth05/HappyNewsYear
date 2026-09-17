from pathlib import Path

import pytest

from kric.evaluation.clipscore import EmbeddingCache
from kric.retrieval.sentences import (
    ClipSentenceRanker,
    SentenceCandidate,
    bm25_rank,
)


def test_bm25_ranks_headline_relevant_sentence_and_preserves_ids() -> None:
    candidates = [
        SentenceCandidate(0, "The football team won the city championship."),
        SentenceCandidate(1, "Markets closed lower after a volatile session."),
        SentenceCandidate(2, "The coach praised the football players."),
    ]
    ranked = bm25_rank(candidates, "Football team wins championship")
    assert ranked[0].sentence_id == 0
    assert {item.sentence_id for item in ranked} == {0, 1, 2}
    assert [item.rank for item in ranked] == [1, 2, 3]


def test_bm25_ties_are_broken_by_original_sentence_id() -> None:
    candidates = [SentenceCandidate(1, "beta"), SentenceCandidate(0, "alpha")]
    ranked = bm25_rank(candidates, "unmatched")
    assert [item.sentence_id for item in ranked] == [0, 1]


def test_clip_ranking_uses_image_sentence_cosine_without_text_query(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"cache-key-only")
    candidates = [
        SentenceCandidate(0, "visually aligned"),
        SentenceCandidate(1, "not aligned"),
    ]
    ranker = ClipSentenceRanker(
        tmp_path / "cache.sqlite3",
        model_name="fake/clip",
        model_revision="fixed",
        device="cpu",
    )
    ranker.model = object()  # cached vectors make model/processor calls unnecessary
    ranker.device = "cpu"
    with EmbeddingCache(ranker.cache_path) as cache:
        cache.put(ranker._image_key(image), [1.0, 0.0])
        cache.put(ranker._text_key(candidates[0].text), [0.9, 0.1])
        cache.put(ranker._text_key(candidates[1].text), [0.1, 0.9])
    ranked = ranker.rank(image, candidates)
    assert [item.sentence_id for item in ranked] == [0, 1]
    assert ranked[0].score == pytest.approx(0.9)
