"""Frozen-input M4 claim-extraction pipeline."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from kric.captioning.b2_regenerated import validate_m3_b2_provenance
from kric.captioning.m3 import validate_generator_control
from kric.evaluation.io import file_sha256, load_predictions, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ("b2", "atomic_full", "atomic_token_matched")
OUTPUT_NAMES = {
    "b2": "b2_claims.jsonl",
    "atomic_full": "atomic_full_claims.jsonl",
    "atomic_token_matched": "atomic_token_matched_claims.jsonl",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 49 or len(set(value)) != 49:
        raise ValueError("M4 requires the exact 49 unique primary IDs")
    return [str(item) for item in value]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_m4_sources(
    *,
    b2_dir: Path,
    m3_dir: Path,
    atomic_dir: Path,
    primary_ids_path: Path,
) -> tuple[list[str], dict[str, Path], dict[str, Any]]:
    ids = _ids(primary_ids_path)
    canonical_ids = _ids(PROJECT_ROOT / "configs/dataset/m3_complete_49_ids.json")
    if ids != canonical_ids:
        raise ValueError("primary IDs differ from the frozen M3 complete-49 set")
    atomic_manifest = _json(atomic_dir / "manifest.json")
    b2_resolved = _json(b2_dir / "resolved_config.json")
    validate_generator_control(b2_resolved)
    b2_status = validate_m3_b2_provenance(
        b2_dir=b2_dir,
        atomic_manifest=atomic_manifest,
        resolved=b2_resolved,
        expected_ids=[
            str(value)
            for value in json.loads(
                (primary_ids_path.parent / "goodnews_validation_50_ids.json").read_text(
                    encoding="utf-8"
                )
            )
        ],
    )
    atomic_output = atomic_manifest.get("outputs", {}).get("atomic_evidence.jsonl", {})
    if atomic_output.get("sha256") != file_sha256(atomic_dir / "atomic_evidence.jsonl"):
        raise ValueError("frozen atomic evidence hash mismatch")
    m3_manifest = _json(m3_dir / "manifest.json")
    paths = {
        "b2": b2_dir / "predictions.jsonl",
        "atomic_full": m3_dir / "atomic_full" / "predictions.jsonl",
        "atomic_token_matched": m3_dir
        / "atomic_token_matched"
        / "predictions.jsonl",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    variants = m3_manifest.get("workflow", {}).get("variants", {})
    for variant in ("m3_atomic_full", "m3_atomic_token_matched"):
        directory_name = variant.removeprefix("m3_")
        expected_hash = variants.get(variant, {}).get("predictions_sha256")
        if expected_hash != file_sha256(paths[directory_name]):
            raise ValueError(f"M3 prediction hash mismatch: {variant}")
    for variant, path in paths.items():
        records = load_predictions(path)
        filtered = [record.sample_id for record in records if record.sample_id in set(ids)]
        if filtered != ids:
            raise ValueError(f"{variant} does not contain the exact ordered primary 49")
    return ids, paths, {
        "b2_provenance_status": b2_status,
        "atomic_manifest_sha256": file_sha256(atomic_dir / "manifest.json"),
        "atomic_evidence_sha256": file_sha256(atomic_dir / "atomic_evidence.jsonl"),
        "m3_manifest_sha256": file_sha256(m3_dir / "manifest.json"),
    }


def run_claim_extraction(
    *,
    extractor: Any,
    b2_dir: Path,
    m3_dir: Path,
    atomic_dir: Path,
    primary_ids_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    ids, paths, provenance = validate_m4_sources(
        b2_dir=b2_dir,
        m3_dir=m3_dir,
        atomic_dir=atomic_dir,
        primary_ids_path=primary_ids_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    counts: Counter[str] = Counter()
    claim_lengths: list[int] = []
    for variant in VARIANTS:
        path = paths[variant]
        source_hashes[variant] = file_sha256(path)
        records = [record for record in load_predictions(path) if record.sample_id in set(ids)]
        variant_rows: list[dict[str, Any]] = []
        for record in records:
            try:
                result = extractor.extract(
                    record.prediction,
                    sample_id=record.sample_id,
                    caption_variant=variant,
                )
                if (
                    result["sample_id"] != record.sample_id
                    or result["caption_variant"] != variant
                    or result["caption"] != record.prediction
                ):
                    raise RuntimeError("claim provenance reconstruction failed")
                for claim in result["claims"]:
                    row = {
                        "sample_id": record.sample_id,
                        "caption_variant": variant,
                        "caption": record.prediction,
                        "claim_id": claim["claim_id"],
                        "claim_text": claim["text"],
                        "claim_type": claim["type"],
                    }
                    variant_rows.append(row)
                    all_rows.append(row)
                    counts[claim["type"]] += 1
                    claim_lengths.append(len(claim["text"].split()))
            except Exception as error:
                failures.append(
                    {
                        "sample_id": record.sample_id,
                        "caption_variant": variant,
                        "caption": record.prediction,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        _write_jsonl(output_dir / OUTPUT_NAMES[variant], variant_rows)
    _write_jsonl(output_dir / "all_claims.jsonl", all_rows)
    _write_jsonl(output_dir / "failures.jsonl", failures)
    audit = {
        "primary_samples": len(ids),
        "caption_variants": len(VARIANTS),
        "requested_captions": len(ids) * len(VARIANTS),
        "failed_captions": len(failures),
        "successful_captions": len(ids) * len(VARIANTS) - len(failures),
        "claims": len(all_rows),
        "claims_by_type": dict(sorted(counts.items())),
        "claim_words": {
            "mean": statistics.fmean(claim_lengths) if claim_lengths else None,
            "median": statistics.median(claim_lengths) if claim_lengths else None,
        },
        "api": extractor.stats(),
    }
    write_json(output_dir / "audit.json", audit)
    manifest = {
        "schema_version": 1,
        "experiment": "M4_caption_claim_decomposition_49",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ordered_primary_ids": ids,
        "primary_ids_sha256": file_sha256(primary_ids_path),
        "source_prediction_sha256": source_hashes,
        "provenance": provenance,
        "extractor": getattr(extractor, "method", type(extractor).__name__),
        "llm_configuration": {
            "api_url": getattr(getattr(extractor, "config", None), "api_url", None),
            "requested_model": getattr(getattr(extractor, "config", None), "model", None),
            "model_revision": getattr(getattr(extractor, "config", None), "model_revision", None),
            "api_key_env": getattr(getattr(extractor, "config", None), "api_key_env", None),
            "api_key_stored_or_logged": False,
            "structured_output_mode": getattr(
                getattr(extractor, "config", None), "structured_output_mode", None
            ),
            "requested_temperature": getattr(
                getattr(extractor, "config", None), "temperature", None
            ),
            "effective_temperature": (
                "provider_default"
                if getattr(getattr(extractor, "config", None), "temperature", None) is None
                else getattr(getattr(extractor, "config", None), "temperature", None)
            ),
        },
        "cache": {
            "path": str(getattr(getattr(extractor, "cache", None), "path", "")),
            "raw_response_attempts_retained": True,
            "malformed_outputs_cached_as_final": False,
        },
        "api_accounting": extractor.stats(),
        "failed_captions": len(failures),
        "reference_passed_to_extractor": False,
        "article_passed_to_extractor": False,
        "image_passed_to_extractor": False,
        "evidence_passed_to_extractor": False,
        "outputs": {
            name: file_sha256(output_dir / name)
            for name in (*OUTPUT_NAMES.values(), "all_claims.jsonl", "audit.json", "failures.jsonl")
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


__all__ = ["OUTPUT_NAMES", "VARIANTS", "run_claim_extraction", "validate_m4_sources"]