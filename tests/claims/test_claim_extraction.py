import json
import sqlite3
from types import SimpleNamespace

import pytest

from kric.claims.llm_claim_extractor import StructuredClaimExtractor
from kric.claims.pipeline import run_claim_extraction, validate_m4_sources
from kric.claims.schema import normalize_claims
from kric.evidence.llm_structured import LlmEndpointConfig, StructuredExtractionError


def _response(content):
    return {
        "id": "response-1",
        "model": "mock-model",
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _config(max_attempts=3):
    return LlmEndpointConfig(
        api_url="https://example.test/v1",
        model="mock-model",
        api_key_env="M4_TEST_KEY",
        model_revision="revision",
        max_attempts=max_attempts,
    )


def test_claim_schema_validation_and_duplicate_removal():
    claims = normalize_claims(
        {
            "claims": [
                {"text": "Alice is speaking.", "type": "event"},
                {"text": "  alice   is speaking. ", "type": "event"},
            ]
        }
    )
    assert [claim.text for claim in claims] == ["Alice is speaking."]
    with pytest.raises(ValueError, match="unsupported claim type"):
        normalize_claims({"claims": [{"text": "fact", "type": "unknown"}]})
    with pytest.raises(ValueError, match="empty text"):
        normalize_claims({"claims": [{"text": " ", "type": "event"}]})


def test_malformed_json_retry_cache_and_no_leakage(tmp_path, monkeypatch):
    monkeypatch.setenv("M4_TEST_KEY", "secret-not-printed")
    bodies = []
    responses = iter(
        [
            _response('{"claims":[{"text":"unterminated'),
            _response(json.dumps({"claims": [{"text": "Alice speaks.", "type": "event"}]})),
        ]
    )

    def post(_url, _headers, body, _timeout):
        bodies.append(body)
        return next(responses)

    extractor = StructuredClaimExtractor(_config(), tmp_path / "cache.sqlite3", post)
    caption = "Alice speaks at a podium."
    result = extractor.extract(caption, sample_id="s1", caption_variant="b2")
    assert result["claims"][0]["text"] == "Alice speaks."
    assert len(bodies) == 2
    assert bodies[0]["messages"][1] == {"role": "user", "content": caption}
    assert bodies[1]["messages"][1]["content"] == caption
    assert "Do not truncate" not in bodies[0]["messages"][0]["content"]
    assert "Do not truncate" in bodies[1]["messages"][0]["content"]
    serialized = json.dumps(bodies)
    for forbidden in ("REFERENCE_MARKER", "ARTICLE_MARKER", "EVIDENCE_MARKER", "IMAGE_MARKER"):
        assert forbidden not in serialized
    cached = extractor.extract(
        caption, sample_id="different-sample", caption_variant="atomic_full"
    )
    assert cached["cache_hit"] is True
    assert cached["sample_id"] == "different-sample"
    assert cached["caption_variant"] == "atomic_full"
    assert cached["claims"][0]["claim_id"] != result["claims"][0]["claim_id"]
    assert extractor.stats()["request_count"] == 2
    assert extractor.stats()["cache_hit_count"] == 1


def test_failed_extraction_retains_all_raw_attempts_and_no_final_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("M4_TEST_KEY", "secret")

    def post(_url, _headers, _body, _timeout):
        return _response('{"claims": [')

    cache = tmp_path / "cache.sqlite3"
    extractor = StructuredClaimExtractor(_config(max_attempts=2), cache, post)
    with pytest.raises(StructuredExtractionError, match="failed after 2 attempts"):
        extractor.extract("Caption.", sample_id="s", caption_variant="b2")
    with sqlite3.connect(cache) as connection:
        attempts = connection.execute("SELECT COUNT(*) FROM response_attempts").fetchone()[0]
        finals = connection.execute("SELECT COUNT(*) FROM extraction_content_cache").fetchone()[0]
    assert attempts == 2
    assert finals == 0


class FakePipelineExtractor:
    method = "fake-claim-extractor"

    def extract(self, caption, *, sample_id, caption_variant):
        if sample_id == "s00" and caption_variant == "b2":
            raise RuntimeError("preserved synthetic failure")
        return {
            "sample_id": sample_id,
            "caption_variant": caption_variant,
            "caption": caption,
            "claims": [
                {"claim_id": f"{caption_variant}-{sample_id}", "text": caption, "type": "event"}
            ],
        }

    def stats(self):
        return {"request_count": 0}


def test_pipeline_preserves_failed_caption_without_substitution(tmp_path, monkeypatch):
    ids = [f"s{index:02d}" for index in range(49)]
    paths = {}
    for variant in ("b2", "atomic_full", "atomic_token_matched"):
        path = tmp_path / f"{variant}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "prediction": f"{variant} caption {sample_id}",
                        "reference": "REFERENCE_MUST_NOT_BE_PASSED",
                        "metadata": {"article": "ARTICLE_MUST_NOT_BE_PASSED"},
                    }
                )
                + "\n"
                for sample_id in ids
            ),
            encoding="utf-8",
        )
        paths[variant] = path
    monkeypatch.setattr(
        "kric.claims.pipeline.validate_m4_sources",
        lambda **_kwargs: (ids, paths, {"validated": True}),
    )
    output = tmp_path / "output"
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    manifest = run_claim_extraction(
        extractor=FakePipelineExtractor(),
        b2_dir=tmp_path,
        m3_dir=tmp_path,
        atomic_dir=tmp_path,
        primary_ids_path=ids_path,
        output_dir=output,
    )
    failures = [json.loads(line) for line in (output / "failures.jsonl").read_text().splitlines()]
    assert len(failures) == 1
    assert failures[0]["sample_id"] == "s00"
    assert failures[0]["caption_variant"] == "b2"
    b2_claims = [json.loads(line) for line in (output / "b2_claims.jsonl").read_text().splitlines()]
    assert all(row["sample_id"] != "s00" for row in b2_claims)
    assert manifest["failed_captions"] == 1
    assert manifest["reference_passed_to_extractor"] is False

def test_m4_provenance_rejects_changed_atomic_evidence(tmp_path, monkeypatch):
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    ids_49 = json.loads(
        (root / "configs/dataset/m3_complete_49_ids.json").read_text(encoding="utf-8")
    )
    ids_50 = json.loads(
        (root / "configs/dataset/goodnews_validation_50_ids.json").read_text(encoding="utf-8")
    )
    ids_path = tmp_path / "m3_complete_49_ids.json"
    ids_path.write_text(json.dumps(ids_49), encoding="utf-8")
    (tmp_path / "goodnews_validation_50_ids.json").write_text(
        json.dumps(ids_50), encoding="utf-8"
    )
    b2 = tmp_path / "b2"
    atomic = tmp_path / "atomic"
    m3 = tmp_path / "m3"
    b2.mkdir()
    atomic.mkdir()
    m3.mkdir()
    (b2 / "resolved_config.json").write_bytes(
        (
            root
            / "artifacts/frozen_b2/goodnews_b2_validation_50/semantic_k3/resolved_config.json"
        ).read_bytes()
    )
    (atomic / "atomic_evidence.jsonl").write_text("changed\n", encoding="utf-8")
    (atomic / "manifest.json").write_text(
        json.dumps(
            {
                "outputs": {
                    "atomic_evidence.jsonl": {"sha256": "expected-frozen-hash"}
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kric.claims.pipeline.validate_m3_b2_provenance",
        lambda **_kwargs: "validated-test-b2",
    )
    with pytest.raises(ValueError, match="atomic evidence hash mismatch"):
        validate_m4_sources(
            b2_dir=b2,
            m3_dir=m3,
            atomic_dir=atomic,
            primary_ids_path=ids_path,
        )