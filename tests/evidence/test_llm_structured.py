import csv
import json
import sqlite3
from inspect import signature

import pytest

from kric.evidence.llm_structured import (
    LlmEndpointConfig,
    RETRY_FORMAT_REMINDER,
    StructuredAtomicExtractor,
    StructuredExtractionError,
    align_source_span,
    stable_evidence_id,
    validate_payload,
)
from scripts.extract_llm_atomic_evidence import _read_source


def _response(content, model="test-model", **extra):
    return {
        "id": "response-1",
        "model": model,
        "choices": [{"message": {"content": json.dumps(content)}}],
        **extra,
    }


def test_exact_source_span_alignment_has_offsets():
    span = align_source_span("Barack Obama spoke in Paris.", "Barack Obama")
    assert span == {"text": "Barack Obama", "start": 0, "end": 12, "alignment": "exact"}


def test_unresolved_span_never_invents_offsets():
    span = align_source_span("Obama spoke.", "Barack Obama")
    assert span["alignment"] == "unresolved"
    assert span["start"] is None and span["end"] is None


def test_extractor_sends_only_one_sentence_and_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    calls = []

    def post(url, headers, body, timeout):
        calls.append(body)
        return _response(
            {
                "propositions": [
                    {
                        "text": "Barack Obama spoke in Paris.",
                        "type": "event",
                        "source_span_text": "Barack Obama spoke in Paris",
                    }
                ]
            }
        )

    extractor = StructuredAtomicExtractor(
        LlmEndpointConfig(
            api_url="https://example.invalid/v1/chat/completions",
            model="test-model",
            api_key_env="TEST_LLM_KEY",
        ),
        tmp_path / "cache.sqlite3",
        post_json=post,
    )
    sentence = "Barack Obama spoke in Paris."
    first = extractor.extract(sentence, sample_id="s1", source_sentence_id="sent1")
    second = extractor.extract(sentence, sample_id="s1", source_sentence_id="sent1")
    assert "temperature" not in calls[0]
    assert calls[0]["messages"][1] == {"role": "user", "content": sentence}
    assert "reference" not in json.dumps(calls[0]).lower()
    assert first["propositions"][0]["source_span"]["alignment"] == "exact"
    assert second["cache_hit"] is True
    assert len(calls) == 1
    assert extractor.stats()["request_count"] == 1
    assert extractor.stats()["cache_hit_count"] == 1


def test_cache_reuses_only_content_and_rebuilds_current_provenance(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    calls = []

    def post(_url, _headers, body, _timeout):
        calls.append(body)
        return _response(
            {
                "propositions": [
                    {
                        "text": "The council approved the plan.",
                        "type": "event",
                        "source_span_text": "The council approved the plan",
                    }
                ]
            }
        )

    cache_path = tmp_path / "cache.sqlite3"
    extractor = StructuredAtomicExtractor(
        LlmEndpointConfig("https://example.invalid", "test-model", "TEST_LLM_KEY"),
        cache_path,
        post_json=post,
    )
    sentence = "The council approved the plan."
    first = extractor.extract(
        sentence,
        sample_id="sample-a",
        source_sentence_id="sentence-a",
        source_rank=1,
        ranking_score=0.91,
    )
    second = extractor.extract(
        sentence,
        sample_id="sample-b",
        source_sentence_id="sentence-b",
        source_rank=3,
        ranking_score=0.42,
    )

    assert len(calls) == 1
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert first["propositions"][0]["evidence_id"] != second["propositions"][0]["evidence_id"]
    assert second["sample_id"] == "sample-b"
    assert second["source_sentence_id"] == "sentence-b"
    assert second["ranking_score"] == 0.42
    assert second["propositions"][0]["provenance"] == {
        "sample_id": "sample-b",
        "source_sentence_id": "sentence-b",
        "source_rank": 3,
        "ranking_score": 0.42,
        "source_sentence": sentence,
    }
    with sqlite3.connect(cache_path) as connection:
        content = connection.execute(
            "SELECT content_json FROM extraction_content_cache"
        ).fetchone()[0]
    assert "sample-a" not in content
    assert "sentence-a" not in content


def test_extractor_api_and_request_exclude_forbidden_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    calls = []

    def post(_url, _headers, body, _timeout):
        calls.append(body)
        return _response({"propositions": []})

    extractor = StructuredAtomicExtractor(
        LlmEndpointConfig("https://example.invalid", "test-model", "TEST_LLM_KEY"),
        tmp_path / "cache.sqlite3",
        post_json=post,
    )
    forbidden_parameters = {
        "reference_caption",
        "generated_caption",
        "image",
        "article_text",
        "unselected_sentences",
        "gold_entities",
        "human_labels",
    }
    assert forbidden_parameters.isdisjoint(signature(extractor.extract).parameters)
    sentence = "Only this frozen selected sentence is permitted."
    extractor.extract(sentence, sample_id="s", source_sentence_id="x")
    assert calls[0]["messages"][1] == {"role": "user", "content": sentence}
    serialized = json.dumps(calls[0])
    for secret_marker in (
        "REFERENCE_MARKER",
        "GENERATED_MARKER",
        "ARTICLE_MARKER",
        "UNSELECTED_MARKER",
        "GOLD_ENTITY_MARKER",
        "HUMAN_LABEL_MARKER",
    ):
        assert secret_marker not in serialized


def test_review_source_loader_discards_labels_and_non_sentence_inputs(tmp_path):
    path = tmp_path / "review.csv"
    fieldnames = [
        "sample_id",
        "source_sentence_id",
        "source_rank",
        "source_sentence",
        "atomic_units_json",
        "atomicity",
        "faithfulness",
        "type_correct",
        "notes",
        "reference_caption",
        "full_article",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(100):
            writer.writerow(
                {
                    "sample_id": f"s{index}",
                    "source_sentence_id": f"x{index}",
                    "source_rank": "1",
                    "source_sentence": f"Frozen sentence {index}.",
                    "atomic_units_json": json.dumps(
                        [{"metadata": {"ranking_score": 0.75}}]
                    ),
                    "atomicity": "good",
                    "faithfulness": "supported",
                    "type_correct": "yes",
                    "notes": "HUMAN_LABEL_MARKER",
                    "reference_caption": "REFERENCE_MARKER",
                    "full_article": "ARTICLE_MARKER",
                }
            )
    rows = _read_source(path)
    assert len(rows) == 100
    assert set(rows[0]) == {
        "sample_id",
        "source_sentence_id",
        "source_rank",
        "source_sentence",
        "ranking_score",
    }
    assert rows[0]["ranking_score"] == "0.75"
    serialized = json.dumps(rows)
    assert "HUMAN_LABEL_MARKER" not in serialized
    assert "REFERENCE_MARKER" not in serialized
    assert "ARTICLE_MARKER" not in serialized


def test_malformed_response_is_retried(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    responses = [
        _response({"wrong": []}),
        _response({"propositions": []}),
    ]

    calls = []

    def post(_url, _headers, body, _timeout):
        calls.append(body)
        return responses.pop(0)

    extractor = StructuredAtomicExtractor(
        LlmEndpointConfig("https://example.invalid", "test", "TEST_LLM_KEY", max_attempts=2),
        tmp_path / "cache.sqlite3",
        post_json=post,
    )
    result = extractor.extract("A valid sentence.", sample_id="s", source_sentence_id="x")
    assert result["propositions"] == []
    assert responses == []
    assert extractor.stats()["request_count"] == 2
    assert extractor.stats()["retry_count"] == 1
    assert extractor.stats()["failed_request_count"] == 1
    assert extractor.stats()["malformed_response_count"] == 1
    assert calls[0]["messages"][1] == calls[1]["messages"][1]
    assert RETRY_FORMAT_REMINDER not in calls[0]["messages"][0]["content"]
    assert RETRY_FORMAT_REMINDER not in calls[1]["messages"][0]["content"]
    assert calls[0]["response_format"] == calls[1]["response_format"]


def test_malformed_raw_attempt_is_preserved_and_only_valid_content_is_cached(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    sentence = "The council approved the plan."
    malformed_content = (
        '{"propositions":[{"text":"The council approved the plan.",'
        '"type":"event","source_span_text":"The council approved'
    )
    responses = [
        {
            "id": "bad-response",
            "model": "gpt-5.6-terra",
            "choices": [{"message": {"content": malformed_content}}],
        },
        _response(
            {
                "propositions": [
                    {
                        "text": "The council approved the plan.",
                        "type": "event",
                        "source_span_text": "The council approved the plan",
                    }
                ]
            },
            model="gpt-5.6-terra",
        ),
    ]
    calls = []

    def post(_url, _headers, body, _timeout):
        calls.append(body)
        return responses.pop(0)

    cache_path = tmp_path / "cache.sqlite3"
    extractor = StructuredAtomicExtractor(
        LlmEndpointConfig(
            "https://api.openai.com/v1",
            "gpt-5.6-terra",
            "TEST_LLM_KEY",
            max_attempts=2,
        ),
        cache_path,
        post_json=post,
    )
    result = extractor.extract(
        sentence,
        sample_id="sample-current",
        source_sentence_id="sentence-current",
        source_rank=2,
        ranking_score=0.77,
    )
    cached = extractor.extract(
        sentence,
        sample_id="sample-cache",
        source_sentence_id="sentence-cache",
        source_rank=3,
        ranking_score=0.55,
    )

    assert len(calls) == 2
    assert calls[0]["messages"][1]["content"] == sentence
    assert calls[1]["messages"][1]["content"] == sentence
    assert calls[0]["response_format"] == calls[1]["response_format"]
    assert RETRY_FORMAT_REMINDER not in calls[0]["messages"][0]["content"]
    assert calls[1]["messages"][0]["content"].endswith(RETRY_FORMAT_REMINDER)
    retry_json = json.dumps(calls[1])
    for forbidden in (
        "reference_caption",
        "generated_caption",
        "full_article",
        "gold_entities",
        "human_labels",
    ):
        assert forbidden not in retry_json
    assert result["propositions"][0]["provenance"]["sample_id"] == "sample-current"
    assert cached["cache_hit"] is True
    assert cached["propositions"][0]["provenance"]["sample_id"] == "sample-cache"
    assert cached["propositions"][0]["provenance"]["source_sentence_id"] == "sentence-cache"
    with sqlite3.connect(cache_path) as connection:
        attempts = connection.execute(
            "SELECT attempt, response_json, error FROM response_attempts ORDER BY id"
        ).fetchall()
        finals = connection.execute(
            "SELECT content_json FROM extraction_content_cache"
        ).fetchall()
    assert len(attempts) == 2
    stored_failed_response = json.loads(attempts[0][1])
    assert stored_failed_response["choices"][0]["message"]["content"] == malformed_content
    assert "Unterminated string" in attempts[0][2]
    assert attempts[1][2] is None
    assert len(finals) == 1
    assert "bad-response" not in finals[0][0]


def test_missing_credentials_fail_before_network(tmp_path):
    extractor = StructuredAtomicExtractor(
        LlmEndpointConfig("https://example.invalid", "test", "MISSING_TEST_KEY"),
        tmp_path / "cache.sqlite3",
        post_json=lambda *_: pytest.fail("network should not be called"),
    )
    with pytest.raises(StructuredExtractionError, match="MISSING_TEST_KEY"):
        extractor.extract("Sentence.", sample_id="s", source_sentence_id="x")


def test_missing_credentials_abort_even_when_content_is_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    extractor = StructuredAtomicExtractor(
        LlmEndpointConfig("https://example.invalid/v1", "test", "TEST_LLM_KEY"),
        tmp_path / "cache.sqlite3",
        post_json=lambda *_: _response({"propositions": []}),
    )
    extractor.extract("Sentence.", sample_id="s1", source_sentence_id="x1")
    monkeypatch.delenv("TEST_LLM_KEY")
    with pytest.raises(StructuredExtractionError, match="TEST_LLM_KEY"):
        extractor.extract("Sentence.", sample_id="s2", source_sentence_id="x2")


@pytest.mark.parametrize(
    "payload",
    [
        {"propositions": [{"text": "Fact.", "type": "unsupported", "source_span_text": "Fact"}]},
        {"propositions": [{"text": "   ", "type": "event", "source_span_text": "Fact"}]},
        {"propositions": [{"text": "Fact.", "type": "event", "source_span_text": "   "}]},
    ],
)
def test_structured_validation_rejects_invalid_units(payload):
    with pytest.raises(StructuredExtractionError):
        validate_payload(payload)


def test_duplicate_handling_and_stable_ids_are_deterministic():
    proposition = {"text": "A fact.", "type": "event", "source_span_text": "A fact"}
    with pytest.raises(StructuredExtractionError, match="duplicates"):
        validate_payload({"propositions": [proposition, dict(proposition)]})
    first = stable_evidence_id("s", "x", 0, "A fact.", "event")
    second = stable_evidence_id("s", "x", 0, "A fact.", "event")
    assert first == second


def test_openai_base_url_response_metadata_and_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    calls = []

    def post(url, _headers, _body, _timeout):
        calls.append(url)
        return _response(
            {"propositions": []},
            model="actual-model-id",
            model_snapshot="snapshot-123",
            usage={"prompt_tokens": 41, "completion_tokens": 7},
        )

    config = LlmEndpointConfig(
        "https://api.openai.com/v1/",
        "requested-model-id",
        "TEST_LLM_KEY",
    )
    extractor = StructuredAtomicExtractor(
        config,
        tmp_path / "cache.sqlite3",
        post_json=post,
    )
    result = extractor.extract("Sentence.", sample_id="s", source_sentence_id="x")
    assert calls == ["https://api.openai.com/v1/chat/completions"]
    assert result["requested_model"] == "requested-model-id"
    assert result["response_model"] == "actual-model-id"
    assert result["response_snapshot"] == "snapshot-123"
    assert extractor.stats() == {
        "request_count": 1,
        "successful_request_count": 1,
        "cache_hit_count": 0,
        "retry_count": 0,
        "failed_request_count": 0,
        "malformed_response_count": 0,
        "total_input_tokens": 41,
        "total_output_tokens": 7,
    }


def test_gpt_5_6_terra_omits_temperature(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    calls = []

    def post(_url, _headers, body, _timeout):
        calls.append(body)
        return _response({"propositions": []}, model="gpt-5.6-terra")

    extractor = StructuredAtomicExtractor(
        LlmEndpointConfig(
            "https://api.openai.com/v1",
            "gpt-5.6-terra",
            "TEST_LLM_KEY",
            temperature=None,
        ),
        tmp_path / "cache.sqlite3",
        post_json=post,
    )
    extractor.extract("Sentence.", sample_id="s", source_sentence_id="x")
    assert "temperature" not in calls[0]


def test_compatible_model_can_include_configured_temperature(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    calls = []

    def post(_url, _headers, body, _timeout):
        calls.append(body)
        return _response({"propositions": []})

    extractor = StructuredAtomicExtractor(
        LlmEndpointConfig(
            "https://example.invalid/v1",
            "compatible-model",
            "TEST_LLM_KEY",
            temperature=0.25,
        ),
        tmp_path / "cache.sqlite3",
        post_json=post,
    )
    extractor.extract("Sentence.", sample_id="s", source_sentence_id="x")
    assert calls[0]["temperature"] == 0.25


def test_provider_default_temperature_preserves_cache_and_provenance(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    calls = []

    def post(_url, _headers, body, _timeout):
        calls.append(body)
        return _response(
            {
                "propositions": [
                    {
                        "text": "A statement was made.",
                        "type": "event",
                        "source_span_text": "A statement was made",
                    }
                ]
            }
        )

    extractor = StructuredAtomicExtractor(
        LlmEndpointConfig(
            "https://api.openai.com/v1",
            "gpt-5.6-terra",
            "TEST_LLM_KEY",
        ),
        tmp_path / "cache.sqlite3",
        post_json=post,
    )
    sentence = "A statement was made."
    extractor.extract(sentence, sample_id="a", source_sentence_id="x", source_rank=1)
    cached = extractor.extract(
        sentence,
        sample_id="b",
        source_sentence_id="y",
        source_rank=2,
        ranking_score=0.5,
    )
    assert len(calls) == 1
    assert cached["cache_hit"] is True
    assert cached["propositions"][0]["provenance"]["sample_id"] == "b"
    assert cached["propositions"][0]["provenance"]["source_sentence_id"] == "y"
    assert cached["propositions"][0]["provenance"]["ranking_score"] == 0.5


def test_temperature_configuration_is_part_of_content_cache_key(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    calls = []

    def post(_url, _headers, body, _timeout):
        calls.append(body)
        return _response({"propositions": []})

    cache_path = tmp_path / "cache.sqlite3"
    provider_default = StructuredAtomicExtractor(
        LlmEndpointConfig(
            "https://example.invalid/v1",
            "compatible-model",
            "TEST_LLM_KEY",
            temperature=None,
        ),
        cache_path,
        post_json=post,
    )
    explicit = StructuredAtomicExtractor(
        LlmEndpointConfig(
            "https://example.invalid/v1",
            "compatible-model",
            "TEST_LLM_KEY",
            temperature=0.25,
        ),
        cache_path,
        post_json=post,
    )
    provider_default.extract("Sentence.", sample_id="a", source_sentence_id="x")
    explicit.extract("Sentence.", sample_id="b", source_sentence_id="y")
    assert len(calls) == 2
    assert "temperature" not in calls[0]
    assert calls[1]["temperature"] == 0.25
