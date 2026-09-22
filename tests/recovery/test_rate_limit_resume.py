from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from kric.recovery.common import sha256
from kric.recovery.rate_limit import RateLimitedPostJson


def test_rate_limiter_spaces_attempts_without_changing_payload():
    now = [0.0]
    sleeps = []
    bodies = []

    def clock():
        return now[0]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    def post(_url, _headers, body, _timeout):
        bodies.append(body)
        return {"ok": True}

    limited = RateLimitedPostJson(post, 21, clock=clock, sleep=sleep)
    body = {"model": "gpt-5.6-terra", "messages": [{"content": "same"}]}
    limited("url", {}, body, 1)
    limited("url", {}, body, 1)
    limited("url", {}, body, 1)
    assert sleeps == [21.0, 21.0]
    assert bodies == [body, body, body]


def _runner_module():
    path = Path("scripts/run_m4_claim_recovery_v2.py")
    spec = importlib.util.spec_from_file_location("m4_recovery_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resume_requires_exact_sources_ids_and_llm_configuration(tmp_path: Path):
    runner = _runner_module()
    output = tmp_path / "m4_claims_regenerated_v2"
    output.mkdir()
    (output / "claim_response_cache.sqlite3").write_bytes(b"cache")
    source = tmp_path / "predictions.jsonl"
    source.write_text('{"sample_id":"s1","prediction":"caption"}\n', encoding="utf-8")
    config = {
        "api_url": "https://api.openai.com/v1",
        "model": "gpt-5.6-terra",
        "model_revision": None,
        "temperature": None,
    }
    manifest = {
        "experiment": runner.EXPERIMENT,
        "claim_universe_status": "regenerated_non_deterministic",
        "ordered_primary_ids": ["s1"],
        "variants": ["b2"],
        "source_prediction_sha256": {"b2": sha256(source)},
        "llm_configuration": {
            "api_url": config["api_url"],
            "requested_model": config["model"],
            "model_revision": None,
            "requested_temperature": None,
            "effective_temperature": "provider_default",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert runner._validate_resume(
        output, ids=["s1"], sources={"b2": source}, config=config
    )
    source.write_text('{"sample_id":"s1","prediction":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source hashes changed"):
        runner._validate_resume(
            output, ids=["s1"], sources={"b2": source}, config=config
        )
