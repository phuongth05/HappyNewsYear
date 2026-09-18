#!/usr/bin/env python
"""Run llm_structured_atomic_v1 on the frozen 100-sentence review source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from kric.evidence.extractor_comparison import audit_units, token_matched_context_stats
from kric.evidence.llm_structured import (
    PROMPT_VERSION,
    LlmEndpointConfig,
    StructuredAtomicExtractor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-review", required=True, help="Frozen manual_review_100.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--api-url",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible base URL or exact chat-completions URL",
    )
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Omit to use the provider/model default; required for gpt-5.6-terra",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--structured-output-mode", choices=("json_schema", "json_object"), default="json_schema"
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--max-samples",
        type=int,
        choices=(3, 100),
        default=3,
        help="Defaults to the required 3-sentence smoke test",
    )
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="Required with --max-samples 100; never enabled by the smoke workflow",
    )
    return parser.parse_args()


def _read_source(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = [dict(row) for row in csv.DictReader(handle)]
    required = {"sample_id", "source_sentence_id", "source_sentence"}
    missing = required - (set(raw_rows[0]) if raw_rows else set())
    if missing:
        raise ValueError(f"source review is missing columns: {sorted(missing)}")
    if len(raw_rows) != 100:
        raise ValueError(f"expected the frozen 100 review sentences, found {len(raw_rows)}")
    keys = [(row["sample_id"], row["source_sentence_id"]) for row in raw_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate sample/source sentence key in review source")
    if any(not row["source_sentence"].strip() for row in raw_rows):
        raise ValueError("review source contains an empty sentence")
    safe_rows = []
    for row in raw_rows:
        ranking_score = _ranking_score(row)
        safe_rows.append(
            {
                "sample_id": row["sample_id"],
                "source_sentence_id": row["source_sentence_id"],
                "source_rank": str(row.get("source_rank", "")),
                "source_sentence": row["source_sentence"],
                "ranking_score": "" if ranking_score is None else str(ranking_score),
            }
        )
    return safe_rows


def _ranking_score(row: dict[str, str]) -> float | None:
    direct = str(row.get("ranking_score", "")).strip()
    if direct:
        return float(direct)
    raw_units = str(row.get("atomic_units_json", "")).strip()
    if not raw_units:
        return None
    units = json.loads(raw_units)
    scores = {
        float(unit["metadata"]["ranking_score"])
        for unit in units
        if isinstance(unit, dict)
        and isinstance(unit.get("metadata"), dict)
        and unit["metadata"].get("ranking_score") is not None
    }
    if len(scores) > 1:
        raise ValueError(
            f"inconsistent ranking scores for {row['sample_id']}:{row['source_sentence_id']}"
        )
    return next(iter(scores), None)


def main() -> None:
    args = parse_args()
    if args.max_samples == 100 and not args.confirm_full_run:
        raise SystemExit(
            "refusing the 100-sentence run without explicit --confirm-full-run"
        )
    if not os.environ.get(args.api_key_env, "").strip():
        raise SystemExit(
            f"missing API credential in environment variable {args.api_key_env}; "
            "aborting before any request"
        )
    started_at = datetime.now(timezone.utc).isoformat()
    source_path = Path(args.source_review)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = _read_source(source_path)
    rows = all_rows[: args.max_samples]
    endpoint = LlmEndpointConfig(
        api_url=args.api_url,
        model=args.model,
        api_key_env=args.api_key_env,
        model_revision=args.model_revision,
        structured_output_mode=args.structured_output_mode,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
    )
    cache_dir = output_dir / "cache"
    extractor = StructuredAtomicExtractor(endpoint, cache_dir / "llm_response_cache.sqlite3")
    results = []
    failures = []
    for index, row in enumerate(rows, start=1):
        rank_text = str(row.get("source_rank", "")).strip()
        ranking_score = _ranking_score(row)
        try:
            result = extractor.extract(
                row["source_sentence"],
                sample_id=row["sample_id"],
                source_sentence_id=row["source_sentence_id"],
                source_rank=int(rank_text) if rank_text else None,
                ranking_score=ranking_score,
            )
            if (
                result["sample_id"] != row["sample_id"]
                or result["source_sentence_id"] != row["source_sentence_id"]
                or result["source_sentence"] != row["source_sentence"]
                or result["ranking_score"] != ranking_score
            ):
                raise RuntimeError("provenance reconstruction validation failed")
            cached = extractor.extract(
                row["source_sentence"],
                sample_id=row["sample_id"],
                source_sentence_id=row["source_sentence_id"],
                source_rank=int(rank_text) if rank_text else None,
                ranking_score=ranking_score,
            )
            if not cached["cache_hit"] or cached["propositions"] != result["propositions"]:
                raise RuntimeError("cache behavior validation failed")
            results.append(result)
            print(
                f"[{index:03d}/{len(rows):03d}] "
                f"{row['sample_id']} {row['source_sentence_id']}"
            )
        except Exception as error:
            failures.append(
                {
                    "sample_id": row["sample_id"],
                    "source_sentence_id": row["source_sentence_id"],
                    "source_rank": int(rank_text) if rank_text else None,
                    "ranking_score": ranking_score,
                    "source_sentence": row["source_sentence"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    evidence_path = output_dir / "llm_atomic_evidence.jsonl"
    evidence_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    audit = audit_units(results, "propositions")
    audit["token_matched_to_review_source"] = token_matched_context_stats(
        results, "propositions"
    )
    audit.update(
        {
            "requested_source_sentences": len(rows),
            "completed_source_sentences": len(results),
            "failed_source_sentences": len(failures),
            "completion_rate": len(results) / len(rows) if rows else 0.0,
            "api": extractor.stats(),
        }
    )
    audit_path = output_dir / "llm_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    contexts = []
    by_sample = defaultdict(list)
    for result in results:
        by_sample[result["sample_id"]].append(result)
    for sample_id, sample_rows in by_sample.items():
        ordered = sorted(
            sample_rows,
            key=lambda item: (
                int(item["source_sentence_id"]),
                int(item.get("source_rank") or 0),
            ),
        )
        units = [
            unit
            for item in ordered
            for unit in item.get("propositions", [])
        ]
        contexts.append(
            {
                "sample_id": sample_id,
                "source_sentence_keys": [
                    [item["sample_id"], item["source_sentence_id"]] for item in ordered
                ],
                "atomic_context": "Evidence:\n"
                + "\n".join(f"- {unit['text']}" for unit in units),
                "atomic_evidence_ids": [unit["evidence_id"] for unit in units],
            }
        )
    (output_dir / "llm_contexts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in contexts),
        encoding="utf-8",
    )
    failures_path = output_dir / "failures.jsonl"
    if failures:
        failures_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
            encoding="utf-8",
        )
    elif failures_path.exists():
        failures_path.unlink()
    response_models = sorted(
        {str(row["response_model"]) for row in results if row.get("response_model")}
    )
    response_snapshots = sorted(
        {str(row["response_snapshot"]) for row in results if row.get("response_snapshot")}
    )
    completed_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "extractor": PROMPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "schema_version": results[0]["schema_version"] if results else "atomic_propositions_schema_v1",
        "run_started_at": started_at,
        "run_completed_at": completed_at,
        "source_review": str(source_path),
        "source_review_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "available_source_rows": len(all_rows),
        "requested_source_rows": len(rows),
        "ordered_source_keys": [
            [row["sample_id"], row["source_sentence_id"]] for row in rows
        ],
        "api_url": args.api_url,
        "api_key_env": args.api_key_env,
        "api_key_stored_or_logged": False,
        "requested_model": args.model,
        "requested_model_revision": args.model_revision,
        "actual_response_model_ids": response_models,
        "provider_response_snapshots": response_snapshots or ["unavailable"],
        "structured_output_mode": args.structured_output_mode,
        "requested_temperature": args.temperature,
        "effective_temperature": (
            "provider_default" if args.temperature is None else args.temperature
        ),
        "max_attempts": args.max_attempts,
        "cache": str(cache_dir / "llm_response_cache.sqlite3"),
        "api_accounting": extractor.stats(),
        "failures": len(failures),
        "caption_generation_run": False,
        "learned_matching_run": False,
        "forbidden_inputs": [
            "image",
            "reference_caption",
            "generated_caption",
            "full_article",
            "unselected_sentences",
            "gold_entities",
            "human_labels",
        ],
    }
    (output_dir / "llm_extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("### Extracted examples")
    for result in results:
        print(
            json.dumps(
                {
                    "sample_id": result["sample_id"],
                    "source_sentence_id": result["source_sentence_id"],
                    "source_sentence": result["source_sentence"],
                    "propositions": result["propositions"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print(f"Wrote {evidence_path}")
    print(f"Wrote {audit_path}")
    if failures:
        raise SystemExit(f"smoke extraction had {len(failures)} failure(s)")


if __name__ == "__main__":
    main()
