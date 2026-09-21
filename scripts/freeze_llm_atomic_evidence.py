#!/usr/bin/env python
"""Freeze llm_structured_atomic_v1 evidence for all 150 B2 semantic-k3 sentences."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.captioning.instructblip import _InstructBlipCaptioner  # noqa: E402
from kric.evidence.llm_frozen_pipeline import (  # noqa: E402
    RETRIEVAL_MODEL,
    RETRIEVAL_REVISION,
    load_verified_frozen_b2,
    prepare_cache,
    run_frozen_llm_extraction,
)
from kric.evidence.llm_structured import (  # noqa: E402
    PROMPT_VERSION,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    LlmEndpointConfig,
    StructuredAtomicExtractor,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"available": True, "commit": commit, "dirty": bool(status.strip())}
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {"available": False, "commit": None, "dirty": None, "detail": str(error)}


def _token_counter(name: str, revision: str):
    from transformers import InstructBlipProcessor

    processor = InstructBlipProcessor.from_pretrained(name, revision=revision)
    tokenizer = processor.tokenizer
    return lambda text: len(_InstructBlipCaptioner._token_ids(tokenizer, text))


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _validate_config(raw: dict) -> None:
    expected = {
        "experiment": "M3_llm_atomic_frozen_50",
        "extractor.method": PROMPT_VERSION,
        "extractor.api_url": "https://api.openai.com/v1",
        "extractor.model": "gpt-5.6-terra",
        "extractor.temperature": None,
        "extractor.structured_output_mode": "json_schema",
        "source.retrieval_method": "semantic",
        "source.retrieval_k": 3,
        "source.retrieval_model": RETRIEVAL_MODEL,
        "source.retrieval_revision": RETRIEVAL_REVISION,
        "context.max_tokens": 384,
    }
    actual = {
        "experiment": raw.get("experiment"),
        "extractor.method": raw.get("extractor", {}).get("method"),
        "extractor.api_url": raw.get("extractor", {}).get("api_url"),
        "extractor.model": raw.get("extractor", {}).get("model"),
        "extractor.temperature": raw.get("extractor", {}).get("temperature"),
        "extractor.structured_output_mode": raw.get("extractor", {}).get(
            "structured_output_mode"
        ),
        "source.retrieval_method": raw.get("source", {}).get("retrieval_method"),
        "source.retrieval_k": int(raw.get("source", {}).get("retrieval_k", 0)),
        "source.retrieval_model": raw.get("source", {}).get("retrieval_model"),
        "source.retrieval_revision": raw.get("source", {}).get("retrieval_revision"),
        "context.max_tokens": int(raw.get("context", {}).get("max_tokens", 0)),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"frozen M3 configuration changed: {mismatches}")
    if raw.get("extractor", {}).get("api_key_env") != "OPENAI_API_KEY":
        raise ValueError("frozen M3 extractor requires OPENAI_API_KEY")
    if int(raw.get("extractor", {}).get("max_attempts", 0)) != 3:
        raise ValueError("frozen M3 malformed-response policy requires max_attempts=3")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiments" / "m3_llm_atomic_frozen_50.yaml",
    )
    parser.add_argument(
        "--cache-source",
        type=Path,
        help="Validated prior LLM cache used only to seed an absent output cache",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(raw)
    base = config_path.parent
    source = raw["source"]
    output_dir = _resolve(base, raw["output_dir"])
    cache_path = output_dir / "cache" / "llm_response_cache.sqlite3"
    configured_cache_source = raw.get("cache_source")
    cache_source = (
        args.cache_source.expanduser().resolve()
        if args.cache_source
        else _resolve(base, configured_cache_source)
        if configured_cache_source
        else None
    )
    paths = {
        "selected_evidence": _resolve(base, source["selected_evidence"]),
        "rankings": _resolve(base, source["rankings"]),
        "manifest": _resolve(base, source["manifest"]),
        "resolved_config": _resolve(base, source["resolved_config"]),
        "ids": _resolve(base, source["ids"]),
    }
    selected_rows = load_verified_frozen_b2(
        paths["selected_evidence"],
        paths["rankings"],
        paths["manifest"],
        paths["resolved_config"],
        paths["ids"],
    )
    api_key_env = raw["extractor"]["api_key_env"]
    preflight_errors = []
    if not cache_path.is_file() and cache_source is None:
        preflight_errors.append(
            "no validated prior cache was provided; pass --cache-source or restore the "
            "existing output cache so reviewed results are not re-sampled"
        )
    elif cache_source is not None and not cache_source.is_file() and not cache_path.is_file():
        preflight_errors.append(f"validated cache source not found: {cache_source}")
    if not os.environ.get(api_key_env, "").strip():
        preflight_errors.append(
            f"missing API credential in environment variable {api_key_env}"
        )
    if preflight_errors:
        raise SystemExit(
            f"validated {len(selected_rows)} samples and "
            f"{sum(len(row['selected_sentence_ids']) for row in selected_rows)} source sentences; "
            + "; ".join(preflight_errors)
            + "; aborting before tokenizer/model loading and before any request"
        )
    cache_report = prepare_cache(cache_path, cache_source)
    endpoint = LlmEndpointConfig(
        api_url=raw["extractor"]["api_url"],
        model=raw["extractor"]["model"],
        api_key_env=api_key_env,
        model_revision=raw["extractor"].get("model_revision"),
        structured_output_mode=raw["extractor"]["structured_output_mode"],
        temperature=None,
        timeout_seconds=float(raw["extractor"]["timeout_seconds"]),
        max_attempts=int(raw["extractor"]["max_attempts"]),
    )
    extractor = StructuredAtomicExtractor(endpoint, cache_path)
    audit = run_frozen_llm_extraction(
        selected_rows=selected_rows,
        extractor=extractor,
        token_counter=_token_counter(
            raw["tokenizer"]["name"], raw["tokenizer"]["revision"]
        ),
        output_dir=output_dir,
        context_budget=int(raw["context"]["max_tokens"]),
    )
    cache_report["sha256_after_run"] = _sha256(cache_path)
    audit["cache"] = cache_report
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    evidence_rows = _read_jsonl(output_dir / "atomic_evidence.jsonl")
    response_models = sorted(
        {
            str(sentence["response_model"])
            for sample in evidence_rows
            for sentence in sample["sentences"]
            if sentence.get("response_model")
        }
    )
    response_snapshots = sorted(
        {
            str(sentence["response_snapshot"])
            for sample in evidence_rows
            for sentence in sample["sentences"]
            if sentence.get("response_snapshot")
        }
    )
    output_names = (
        "atomic_evidence.jsonl",
        "atomic_contexts_full.jsonl",
        "atomic_contexts_token_matched.jsonl",
        "audit.json",
        "failures.jsonl",
    )
    manifest = {
        "schema_version": 1,
        "experiment": "M3_llm_atomic_frozen_50",
        "extractor": PROMPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "evidence_schema_version": SCHEMA_VERSION,
        "requested_model": endpoint.model,
        "requested_model_revision": endpoint.model_revision,
        "actual_response_model_ids": response_models,
        "provider_response_snapshots": response_snapshots or ["unavailable"],
        "requested_temperature": None,
        "effective_temperature": "provider_default",
        "structured_output": "strict json_schema",
        "api_url": endpoint.api_url,
        "api_key_env": endpoint.api_key_env,
        "api_key_stored_or_logged": False,
        "run_started_at": audit["run_started_at"],
        "run_completed_at": audit["run_completed_at"],
        "git": _git_state(),
        "source": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "retrieval": {
            "method": "semantic",
            "k": 3,
            "model": RETRIEVAL_MODEL,
            "revision": RETRIEVAL_REVISION,
        },
        "ordered_sample_ids": [str(row["sample_id"]) for row in selected_rows],
        "ordered_source_keys": [
            [str(row["sample_id"]), str(sentence_id)]
            for row in selected_rows
            for sentence_id in row["selected_sentence_ids"]
        ],
        "cache": cache_report,
        "api_accounting": audit["api"],
        "caption_generation_run": False,
        "spacy_fallback_used": False,
        "forbidden_inputs": [
            "image",
            "full_article",
            "unselected_sentences",
            "reference_caption",
            "generated_caption",
            "gold_entities",
            "human_labels",
            "spacy_output",
        ],
        "outputs": {
            name: {
                "path": str((output_dir / name).resolve()),
                "sha256": _sha256(output_dir / name),
            }
            for name in output_names
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"Full contexts: {output_dir / 'atomic_contexts_full.jsonl'}")
    print(f"Token-matched contexts: {output_dir / 'atomic_contexts_token_matched.jsonl'}")
    return 0 if audit["source_sentences_failed"] == 0 else 2


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
