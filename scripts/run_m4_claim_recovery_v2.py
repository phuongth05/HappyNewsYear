"""Regenerate an explicitly new, non-deterministic M4 claim universe."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kric.claims.llm_claim_extractor import StructuredClaimExtractor  # noqa: E402
from kric.evidence.llm_structured import (  # noqa: E402
    LlmEndpointConfig,
    StructuredAtomicExtractor as AtomicTransport,
)
from kric.recovery.common import (  # noqa: E402
    read_jsonl,
    require_recovery_output,
    sha256,
    write_json,
    write_jsonl,
)
from kric.recovery.rate_limit import RateLimitedPostJson  # noqa: E402

EXPERIMENT = "M4_caption_claim_decomposition_49_regenerated_v2"
ALLOWED_VARIANTS = {"b2", "atomic_full", "atomic_token_matched"}


def _sources(specifications: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for specification in specifications:
        variant, separator, raw_path = specification.partition("=")
        if not separator or variant not in ALLOWED_VARIANTS or variant in sources:
            raise ValueError("sources must be unique variant=predictions.jsonl entries")
        sources[variant] = Path(raw_path).expanduser().resolve()
    return sources


def _validate_resume(
    output_dir: Path,
    *,
    ids: list[str],
    sources: dict[str, Path],
    config: dict,
) -> str:
    manifest_path = output_dir / "manifest.json"
    cache_path = output_dir / "claim_response_cache.sqlite3"
    if not manifest_path.is_file() or not cache_path.is_file():
        raise FileNotFoundError("resume requires the prior manifest and response cache")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("resume manifest has the wrong experiment")
    if manifest.get("claim_universe_status") != "regenerated_non_deterministic":
        raise ValueError("resume manifest has invalid claim-universe provenance")
    if manifest.get("ordered_primary_ids") != ids:
        raise ValueError("resume primary IDs changed")
    if manifest.get("variants") != list(sources):
        raise ValueError("resume variants or variant order changed")
    expected_hashes = {variant: sha256(path) for variant, path in sources.items()}
    if manifest.get("source_prediction_sha256") != expected_hashes:
        raise ValueError("resume prediction source hashes changed")
    llm = manifest.get("llm_configuration", {})
    expected_llm = {
        "api_url": config["api_url"],
        "requested_model": config["model"],
        "model_revision": config.get("model_revision"),
        "requested_temperature": config.get("temperature"),
        "effective_temperature": (
            "provider_default" if config.get("temperature") is None else config.get("temperature")
        ),
    }
    if llm != expected_llm:
        raise ValueError("resume LLM configuration changed")
    return sha256(manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True, help="variant=predictions.jsonl")
    parser.add_argument(
        "--primary-ids",
        type=Path,
        default=PROJECT_ROOT / "configs/dataset/m3_complete_49_ids.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/m4_claims_49_regenerated_v2",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--min-request-interval-seconds",
        type=float,
        default=21.0,
        help="Minimum spacing between actual API request starts; 21s supports a 3 RPM limit.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ids = json.loads(args.primary_ids.read_text(encoding="utf-8"))
    sources = _sources(args.source)
    output_dir = args.output_dir.expanduser().resolve()
    previous_manifest_sha256 = None
    if args.resume:
        if not output_dir.is_dir():
            raise FileNotFoundError(output_dir)
        previous_manifest_sha256 = _validate_resume(
            output_dir, ids=ids, sources=sources, config=config
        )
    else:
        output_dir = require_recovery_output(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    endpoint = LlmEndpointConfig(
        api_url=config["api_url"],
        model=config["model"],
        api_key_env=config["api_key_env"],
        model_revision=config.get("model_revision"),
        structured_output_mode=config["structured_output_mode"],
        temperature=config.get("temperature"),
        timeout_seconds=float(config["timeout_seconds"]),
        max_attempts=int(config["max_attempts"]),
    )
    transport = RateLimitedPostJson(
        AtomicTransport._urllib_post_json,
        args.min_request_interval_seconds,
    )
    extractor = StructuredClaimExtractor(
        endpoint,
        output_dir / "claim_response_cache.sqlite3",
        post_json=transport,
    )

    all_rows = []
    failures = []
    hashes = {}
    for variant, path in sources.items():
        rows = read_jsonl(path)
        by_sample = {str(row["sample_id"]): row for row in rows}
        if len(by_sample) != len(rows) or any(sample_id not in by_sample for sample_id in ids):
            raise ValueError(f"{variant} has duplicate or missing primary IDs")
        claims = []
        for sample_id in ids:
            prediction = str(by_sample[sample_id]["prediction"])
            try:
                result = extractor.extract(
                    prediction,
                    sample_id=sample_id,
                    caption_variant=variant,
                )
                claims.extend(
                    {
                        "sample_id": sample_id,
                        "caption_variant": variant,
                        "caption": prediction,
                        "claim_id": claim["claim_id"],
                        "claim_text": claim["text"],
                        "claim_type": claim["type"],
                    }
                    for claim in result["claims"]
                )
            except Exception as error:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "caption_variant": variant,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        name = f"{variant}_claims.jsonl"
        write_jsonl(output_dir / name, claims)
        all_rows.extend(claims)
        hashes[variant] = sha256(path)

    write_jsonl(output_dir / "all_claims.jsonl", all_rows)
    write_jsonl(output_dir / "failures.jsonl", failures)
    audit = {
        "requested_captions": len(ids) * len(sources),
        "claims": len(all_rows),
        "failures": len(failures),
        "successful_captions": len(ids) * len(sources) - len(failures),
        "variants": list(sources),
        "resume": args.resume,
        "request_pacing_seconds": args.min_request_interval_seconds,
        "api": extractor.stats(),
    }
    write_json(output_dir / "audit.json", audit)
    names = [
        "all_claims.jsonl",
        "failures.jsonl",
        "audit.json",
        *[f"{variant}_claims.jsonl" for variant in sources],
    ]
    manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT,
        "claim_universe_status": "regenerated_non_deterministic",
        "ordered_primary_ids": ids,
        "variants": list(sources),
        "scope_limitation": (
            None
            if set(sources) == ALLOWED_VARIANTS
            else "only explicitly supplied prediction variants were regenerated; no substitution occurred"
        ),
        "source_prediction_sha256": hashes,
        "llm_configuration": {
            "api_url": config["api_url"],
            "requested_model": config["model"],
            "model_revision": config.get("model_revision"),
            "requested_temperature": config.get("temperature"),
            "effective_temperature": (
                "provider_default"
                if config.get("temperature") is None
                else config.get("temperature")
            ),
        },
        "recovery_execution": {
            "resumed_from_partial_run": args.resume,
            "previous_manifest_sha256": previous_manifest_sha256,
            "minimum_request_interval_seconds": args.min_request_interval_seconds,
            "request_payload_changed_by_rate_limiter": False,
        },
        "api_key_stored_or_logged": False,
        "reference_passed": False,
        "article_passed": False,
        "image_passed": False,
        "evidence_passed": False,
        "outputs": {name: sha256(output_dir / name) for name in names},
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())