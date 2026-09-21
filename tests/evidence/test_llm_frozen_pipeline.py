import hashlib
import json
from pathlib import Path

import pytest

from kric.evidence.llm_frozen_pipeline import (
    RETRIEVAL_MODEL,
    RETRIEVAL_REVISION,
    _fit_closest,
    load_verified_frozen_b2,
    prepare_cache,
    run_frozen_llm_extraction,
)


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _frozen_artifacts(tmp_path: Path):
    ids = [f"sample-{index:02d}" for index in range(50)]
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    selected_rows = []
    ranking_rows = []
    for sample_index, sample_id in enumerate(ids):
        ranked = []
        for rank, sentence_id in enumerate((5, 1, 3), start=1):
            ranked.append(
                {
                    "rank": rank,
                    "score": 1.0 - rank / 10,
                    "sentence_id": sentence_id,
                    "text": f"Frozen {sample_index} sentence {sentence_id}.",
                }
            )
        by_id = {item["sentence_id"]: item for item in ranked}
        article_order = [1, 3, 5]
        selected_rows.append(
            {
                "sample_id": sample_id,
                "retrieval_method": "semantic",
                "retrieval_k": 3,
                "selected_sentence_count": 3,
                "selected_sentence_ids": article_order,
                "selected_sentence_texts": [by_id[value]["text"] for value in article_order],
                "selected_ranking_scores": [by_id[value]["score"] for value in article_order],
                "selected_ranks": [by_id[value]["rank"] for value in article_order],
                "context_token_count": 12,
            }
        )
        ranking_rows.append(
            {
                "sample_id": sample_id,
                "ranked_sentences": ranked,
                "retriever": {
                    "method": "semantic",
                    "name": RETRIEVAL_MODEL,
                    "revision": RETRIEVAL_REVISION,
                },
            }
        )
    selected_path = tmp_path / "selected.jsonl"
    rankings_path = tmp_path / "rankings.jsonl"
    _write_jsonl(selected_path, selected_rows)
    _write_jsonl(rankings_path, ranking_rows)
    rankings_hash = hashlib.sha256(rankings_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "split": "dev",
                    "max_samples": 50,
                    "subset_strategy": "first_by_sample_id",
                },
                "retrieval": {
                    "method": "semantic",
                    "k": 3,
                    "model_name": RETRIEVAL_MODEL,
                    "model_revision": RETRIEVAL_REVISION,
                },
                "rankings_sha256": rankings_hash,
            }
        ),
        encoding="utf-8",
    )
    resolved_path = tmp_path / "resolved.json"
    resolved_path.write_text(
        json.dumps(
            {
                "retrieval": {
                    "method": "semantic",
                    "k": 3,
                    "model_name": RETRIEVAL_MODEL,
                    "model_revision": RETRIEVAL_REVISION,
                }
            }
        ),
        encoding="utf-8",
    )
    return selected_path, rankings_path, manifest_path, resolved_path, ids_path, selected_rows


class FakeExtractor:
    def __init__(self, failed_key=None):
        self.received = []
        self.failed_key = failed_key

    def extract(
        self,
        sentence,
        *,
        sample_id,
        source_sentence_id,
        source_rank,
        ranking_score,
    ):
        self.received.append(sentence)
        if (sample_id, source_sentence_id) == self.failed_key:
            raise RuntimeError("synthetic extraction failure")
        text = sentence.removesuffix(".") + " happened."
        return {
            "sample_id": sample_id,
            "source_sentence_id": source_sentence_id,
            "source_rank": source_rank,
            "ranking_score": ranking_score,
            "source_sentence": sentence,
            "response_model": "fake-model",
            "response_snapshot": None,
            "propositions": [
                {
                    "evidence_id": f"ae-{sample_id}-{source_sentence_id}",
                    "text": text,
                    "type": "event",
                    "source_span": {
                        "text": sentence,
                        "start": 0,
                        "end": len(sentence),
                        "alignment": "exact",
                    },
                    "provenance": {
                        "sample_id": sample_id,
                        "source_sentence_id": source_sentence_id,
                        "source_rank": source_rank,
                        "ranking_score": ranking_score,
                        "source_sentence": sentence,
                    },
                }
            ],
        }

    def stats(self):
        return {
            "request_count": len(self.received),
            "cache_hit_count": 0,
            "retry_count": 0,
            "failed_request_count": int(self.failed_key is not None),
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }


def _word_tokens(text: str) -> int:
    return len(text.split())


def test_frozen_loader_cross_checks_rankings_and_exact_150(tmp_path):
    paths = _frozen_artifacts(tmp_path)
    rows = load_verified_frozen_b2(*paths[:5])
    assert len(rows) == 50
    assert sum(len(row["selected_sentence_ids"]) for row in rows) == 150
    assert rows[0]["selected_sentence_ids"] == [1, 3, 5]


def test_frozen_loader_rejects_leakage_fields(tmp_path):
    paths = _frozen_artifacts(tmp_path)
    selected_path = paths[0]
    rows = [json.loads(line) for line in selected_path.read_text().splitlines()]
    rows[0]["reference_caption"] = "must not be exposed"
    _write_jsonl(selected_path, rows)
    with pytest.raises(ValueError, match="forbidden input fields"):
        load_verified_frozen_b2(*paths[:5])


def test_pipeline_writes_50_ordered_samples_and_bounded_contexts(tmp_path):
    *_, selected_rows = _frozen_artifacts(tmp_path)
    extractor = FakeExtractor()
    output = tmp_path / "output"
    audit = run_frozen_llm_extraction(
        selected_rows=selected_rows,
        extractor=extractor,
        token_counter=_word_tokens,
        output_dir=output,
    )
    assert len(extractor.received) == 150
    assert audit["source_sentences_completed"] == 150
    assert audit["samples_with_all_3_sentences_completed"] == 50
    evidence = [json.loads(line) for line in (output / "atomic_evidence.jsonl").read_text().splitlines()]
    full = [json.loads(line) for line in (output / "atomic_contexts_full.jsonl").read_text().splitlines()]
    matched = [
        json.loads(line)
        for line in (output / "atomic_contexts_token_matched.jsonl").read_text().splitlines()
    ]
    assert len(evidence) == len(full) == len(matched) == 50
    assert evidence[0]["selected_sentence_ids"] == [1, 3, 5]
    assert [unit["provenance"]["source_sentence_id"] for unit in evidence[0]["atomic_units"]] == [
        "1",
        "3",
        "5",
    ]
    assert all(row["context_tokens"] <= 384 for row in full)
    assert all(row["context_tokens"] <= row["b2_context_token_budget"] for row in matched)
    assert (output / "failures.jsonl").read_text(encoding="utf-8") == ""


def test_failure_is_explicit_and_never_replaced_with_spacy(tmp_path):
    *_, selected_rows = _frozen_artifacts(tmp_path)
    failed_key = (selected_rows[0]["sample_id"], "3")
    output = tmp_path / "output"
    audit = run_frozen_llm_extraction(
        selected_rows=selected_rows,
        extractor=FakeExtractor(failed_key),
        token_counter=_word_tokens,
        output_dir=output,
    )
    assert audit["source_sentences_failed"] == 1
    assert audit["samples_with_all_3_sentences_completed"] == 49
    assert audit["spacy_fallback_used"] is False
    failure = json.loads((output / "failures.jsonl").read_text().strip())
    assert (failure["sample_id"], failure["source_sentence_id"]) == failed_key
    evidence = json.loads((output / "atomic_evidence.jsonl").read_text().splitlines()[0])
    failed = next(row for row in evidence["sentences"] if row["status"] == "failed")
    assert failed["propositions"] == []


def test_cache_seed_never_overwrites_existing_destination(tmp_path):
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "cache" / "cache.sqlite3"
    source.write_bytes(b"validated-cache")
    first = prepare_cache(destination, source)
    assert first["seeded"] is True
    assert destination.read_bytes() == b"validated-cache"
    destination.write_bytes(b"expanded-run-cache")
    second = prepare_cache(destination, source)
    assert second["existing_cache_preserved"] is True
    assert second["seeded"] is False
    assert destination.read_bytes() == b"expanded-run-cache"


def test_token_matching_uses_closest_order_preserving_subset_without_padding():
    units = [
        {"evidence_id": "long", "text": "one two three four five six"},
        {"evidence_id": "short-a", "text": "seven"},
        {"evidence_id": "short-b", "text": "eight"},
    ]
    selected, _context, tokens, dropped = _fit_closest(units, _word_tokens, 5)
    assert [unit["evidence_id"] for unit in selected] == ["short-a", "short-b"]
    assert tokens == 5
    assert dropped == ["long"]
