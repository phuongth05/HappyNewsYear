"""Extract atomic claims from the frozen 49-sample B2/M3 captions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.claims.llm_claim_extractor import StructuredClaimExtractor  # noqa: E402
from kric.claims.pipeline import run_claim_extraction  # noqa: E402
from kric.evidence.llm_structured import LlmEndpointConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--b2-dir", required=True, type=Path)
    parser.add_argument("--m3-dir", required=True, type=Path)
    parser.add_argument("--atomic-dir", required=True, type=Path)
    parser.add_argument(
        "--primary-ids",
        type=Path,
        default=PROJECT_ROOT / "configs/dataset/m3_complete_49_ids.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if raw.get("experiment") != "M4_caption_claim_decomposition_49":
        raise ValueError("unexpected M4 claim-extraction config")
    api_key_env = str(raw.get("api_key_env", "OPENAI_API_KEY"))
    if not os.environ.get(api_key_env, "").strip():
        raise SystemExit(
            f"missing API credential in environment variable {api_key_env}; aborting before requests"
        )
    endpoint = LlmEndpointConfig(
        api_url=str(raw["api_url"]),
        model=str(raw["model"]),
        api_key_env=api_key_env,
        model_revision=raw.get("model_revision"),
        structured_output_mode=str(raw.get("structured_output_mode", "json_schema")),
        temperature=raw.get("temperature"),
        timeout_seconds=float(raw.get("timeout_seconds", 120)),
        max_attempts=int(raw.get("max_attempts", 3)),
    )
    output = args.output_dir.expanduser().resolve()
    extractor = StructuredClaimExtractor(
        endpoint, output / "cache" / "llm_claim_response_cache.sqlite3"
    )
    manifest = run_claim_extraction(
        extractor=extractor,
        b2_dir=args.b2_dir.expanduser().resolve(),
        m3_dir=args.m3_dir.expanduser().resolve(),
        atomic_dir=args.atomic_dir.expanduser().resolve(),
        primary_ids_path=args.primary_ids.expanduser().resolve(),
        output_dir=output,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if manifest["failed_captions"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())