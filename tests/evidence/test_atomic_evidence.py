import json
from pathlib import Path

import pytest

from kric.evidence.pipeline import _extract_with_retries, run_atomic_extraction
from kric.evidence.schema import (
    AtomicEvidence,
    FrozenSentence,
    SourceSpan,
    stable_evidence_id,
)


def _unit(sentence: FrozenSentence) -> AtomicEvidence:
    span = SourceSpan(0, len(sentence.text), sentence.text)
    text = sentence.text.rstrip(".") + " occurred."
    return AtomicEvidence(
        evidence_id=stable_evidence_id(
            sentence.sample_id, sentence.sentence_id, "event", text, span
        ),
        text=text,
        type="event",
        source_sentence_id=sentence.sentence_id,
        source_span=span,
        source_rank=sentence.source_rank,
        metadata={"ranking_score": sentence.ranking_score, "extractor_rule": "fake"},
    )


class RecordingExtractor:
    def __init__(self):
        self.received = []

    def info(self):
        return {
            "method": "test",
            "model_name": "test",
            "model_version": "1",
            "rule_set_version": "1",
            "prompt_version": "none",
        }

    def cache_key(self, sentence):
        return f"{sentence.sample_id}:{sentence.sentence_id}:{sentence.text}"

    def extract(self, sentence):
        self.received.append(sentence)
        return [_unit(sentence)]


def _frozen_fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    ids = [f"sample-{index:02d}" for index in range(50)]
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    selected = tmp_path / "selected_evidence.jsonl"
    with selected.open("w", encoding="utf-8") as handle:
        for sample_index, sample_id in enumerate(ids):
            row = {
                "sample_id": sample_id,
                "retrieval_method": "semantic",
                "retrieval_k": 3,
                "selected_sentence_count": 3,
                "selected_sentence_ids": [1, 4, 8],
                "selected_sentence_texts": [
                    f"Selected alpha {sample_index}.",
                    f"Selected beta {sample_index}.",
                    f"Selected gamma {sample_index}.",
                ],
                "selected_ranking_scores": [0.9, 0.8, 0.7],
                "selected_ranks": [1, 2, 3],
                "context_token_count": 12,
                "unselected_sentence_text": "This must never reach the extractor.",
            }
            handle.write(json.dumps(row) + "\n")
    return selected, ids_path, ids


def _word_count(text: str) -> int:
    return len(text.split())


def test_atomic_schema_rejects_malformed_and_ids_are_stable() -> None:
    sentence = FrozenSentence("sample", 2, "A person spoke.", 1, 0.9)
    first = _unit(sentence)
    second = _unit(sentence)
    assert first.evidence_id == second.evidence_id
    with pytest.raises(ValueError, match="missing fields"):
        AtomicEvidence.from_dict({"text": "broken"})


def test_malformed_structured_output_is_retried() -> None:
    sentence = FrozenSentence("sample", 2, "A person spoke.", 1, 0.9)

    class RetryExtractor:
        calls = 0

        def extract(self, received):
            self.calls += 1
            if self.calls == 1:
                return [{"text": "missing schema"}]
            return [_unit(received)]

    extractor = RetryExtractor()
    units, failures = _extract_with_retries(extractor, sentence)
    assert failures == 1
    assert extractor.calls == 2
    assert units[0].source_sentence_id == 2


def test_pipeline_only_exposes_frozen_top3_and_reuses_cache(tmp_path: Path) -> None:
    selected, ids_path, ids = _frozen_fixture(tmp_path)
    cache = tmp_path / "cache.sqlite3"
    extractor = RecordingExtractor()
    report = run_atomic_extraction(
        selected,
        ids_path,
        tmp_path / "first",
        extractor=extractor,
        token_counter=_word_count,
        cache_path=cache,
    )
    assert len(extractor.received) == 150
    assert all("unselected" not in sentence.text.lower() for sentence in extractor.received)
    assert {sentence.sample_id for sentence in extractor.received} == set(ids)
    assert report["input"]["reference_caption_exposed"] is False
    assert report["input"]["generated_caption_exposed"] is False
    assert report["input"]["full_article_exposed"] is False
    assert report["manual_review"]["included"] == 100
    contexts = [
        json.loads(line)
        for line in (tmp_path / "first" / "atomic_contexts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["sample_id"] for row in contexts] == ids
    assert all(row["atomic_context_tokens"] <= 384 for row in contexts)
    assert all(row["token_matched_context_tokens"] <= 12 for row in contexts)
    assert all("atomic_context_dropped_unit_count" in row for row in contexts)
    assert all("atomic_evidence_tokens_unformatted" in row for row in contexts)

    cached_extractor = RecordingExtractor()
    cached = run_atomic_extraction(
        selected,
        ids_path,
        tmp_path / "second",
        extractor=cached_extractor,
        token_counter=_word_count,
        cache_path=cache,
    )
    assert cached_extractor.received == []
    assert cached["cache"] == {
        "path": str(cache.resolve()),
        "hits": 150,
        "misses": 0,
    }


def test_source_order_and_provenance_are_preserved(tmp_path: Path) -> None:
    selected, ids_path, _ = _frozen_fixture(tmp_path)
    run_atomic_extraction(
        selected,
        ids_path,
        tmp_path / "output",
        extractor=RecordingExtractor(),
        token_counter=_word_count,
        cache_path=tmp_path / "cache.sqlite3",
    )
    first = json.loads(
        (tmp_path / "output" / "atomic_evidence.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert first["selected_sentence_ids"] == [1, 4, 8]
    assert [unit["source_sentence_id"] for unit in first["atomic_units"]] == [1, 4, 8]
    assert all(unit["source_span"] is not None for unit in first["atomic_units"])
