"""Build frozen M4.2 support-filtered token-matched evidence contexts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.captioning.instructblip import _InstructBlipCaptioner  # noqa: E402
from kric.captioning.m3 import MODEL_NAME, MODEL_REVISION  # noqa: E402
from kric.evaluation.io import file_sha256, write_json  # noqa: E402
from kric.matching.calibration import OBJECTIVES  # noqa: E402
from kric.matching.cosine import COSINE_MODEL, COSINE_REVISION  # noqa: E402
from kric.matching.filtering import build_support_filtered_contexts  # noqa: E402
from kric.matching.ranking import read_jsonl, write_jsonl  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _output_hash(manifest: Mapping[str, Any], name: str) -> str | None:
    value = manifest.get("outputs", {}).get(name)
    return value.get("sha256") if isinstance(value, Mapping) else value


def _validate_hash(manifest: Mapping[str, Any], name: str, path: Path) -> None:
    expected = _output_hash(manifest, name)
    if expected != file_sha256(path):
        raise ValueError(f"frozen artifact hash mismatch: {name}")


def _token_counter():
    from transformers import InstructBlipProcessor

    processor = InstructBlipProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    tokenizer = processor.tokenizer
    return lambda text: len(_InstructBlipCaptioner._token_ids(tokenizer, text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-dir", required=True, type=Path)
    parser.add_argument("--claims-dir", required=True, type=Path)
    parser.add_argument("--matcher-dir", required=True, type=Path)
    parser.add_argument("--atomic-dir", required=True, type=Path)
    parser.add_argument("--m3-dir", required=True, type=Path)
    parser.add_argument(
        "--primary-ids",
        type=Path,
        default=PROJECT_ROOT / "configs/dataset/m3_complete_49_ids.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    directories = {
        "threshold": args.threshold_dir.expanduser().resolve(),
        "claims": args.claims_dir.expanduser().resolve(),
        "matcher": args.matcher_dir.expanduser().resolve(),
        "atomic": args.atomic_dir.expanduser().resolve(),
        "m3": args.m3_dir.expanduser().resolve(),
    }
    primary_path = args.primary_ids.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    primary_ids = json.loads(primary_path.read_text(encoding="utf-8"))
    canonical = json.loads(
        (PROJECT_ROOT / "configs/dataset/m3_complete_49_ids.json").read_text(encoding="utf-8")
    )
    if primary_ids != canonical or len(primary_ids) != 49:
        raise ValueError("primary IDs differ from the frozen ordered M3 complete-49 set")

    threshold_manifest_path = directories["threshold"] / "threshold_manifest.json"
    threshold_report_path = directories["threshold"] / "calibration_report.json"
    threshold_predictions_path = directories["threshold"] / "fold_predictions.csv"
    threshold_manifest = _json(threshold_manifest_path)
    if threshold_manifest.get("provenance_status") != "frozen":
        raise ValueError("support threshold has not been explicitly frozen")
    calibration = threshold_manifest.get("calibration", {})
    selected_objective = calibration.get("selected_objective")
    threshold = calibration.get("selected_threshold")
    if selected_objective not in OBJECTIVES or threshold is None:
        raise ValueError("threshold manifest has no valid selected objective/threshold")
    if threshold_manifest.get("outputs", {}).get("calibration_report.json") != file_sha256(
        threshold_report_path
    ):
        raise ValueError("calibration report hash mismatch")
    if threshold_manifest.get("outputs", {}).get("fold_predictions.csv") != file_sha256(
        threshold_predictions_path
    ):
        raise ValueError("calibration fold-prediction hash mismatch")

    claims_manifest = _json(directories["claims"] / "manifest.json")
    matcher_manifest = _json(directories["matcher"] / "manifest.json")
    atomic_manifest = _json(directories["atomic"] / "manifest.json")
    m3_manifest = _json(directories["m3"] / "manifest.json")
    if claims_manifest.get("ordered_primary_ids") != primary_ids:
        raise ValueError("claims do not use the frozen ordered primary IDs")
    if matcher_manifest.get("ordered_primary_ids") != primary_ids:
        raise ValueError("matcher rankings do not use the frozen ordered primary IDs")
    cosine_model = matcher_manifest.get("models", {}).get("cosine", {})
    if (
        cosine_model.get("model") != COSINE_MODEL
        or cosine_model.get("revision") != COSINE_REVISION
        or cosine_model.get("score") != "cosine_similarity"
    ):
        raise ValueError("cosine matcher model/revision/score changed")

    claims_path = directories["claims"] / "atomic_token_matched_claims.jsonl"
    rankings_path = directories["matcher"] / "cosine_rankings.jsonl"
    atomic_path = directories["atomic"] / "atomic_evidence.jsonl"
    matched_path = directories["atomic"] / "atomic_contexts_token_matched.jsonl"
    draft_path = directories["m3"] / "atomic_token_matched" / "predictions.jsonl"
    _validate_hash(claims_manifest, claims_path.name, claims_path)
    _validate_hash(matcher_manifest, rankings_path.name, rankings_path)
    _validate_hash(atomic_manifest, atomic_path.name, atomic_path)
    _validate_hash(atomic_manifest, matched_path.name, matched_path)
    if matcher_manifest.get("claims_sha256") != file_sha256(
        directories["claims"] / "all_claims.jsonl"
    ):
        raise ValueError("matcher/claim provenance mismatch")
    if matcher_manifest.get("atomic_evidence_sha256") != file_sha256(atomic_path):
        raise ValueError("matcher/atomic provenance mismatch")
    expected_draft_hash = m3_manifest.get("workflow", {}).get("variants", {}).get(
        "m3_atomic_token_matched", {}
    ).get("predictions_sha256")
    actual_draft_hash = file_sha256(draft_path)
    if (
        expected_draft_hash != actual_draft_hash
        or claims_manifest.get("source_prediction_sha256", {}).get("atomic_token_matched")
        != actual_draft_hash
    ):
        raise ValueError("claim decomposition does not derive from the frozen token-matched draft")

    contexts, audit = build_support_filtered_contexts(
        primary_ids=primary_ids,
        claim_rows=read_jsonl(claims_path),
        ranking_rows=read_jsonl(rankings_path),
        atomic_rows=read_jsonl(atomic_path),
        token_matched_rows=read_jsonl(matched_path),
        threshold=float(threshold),
        token_counter=_token_counter(),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts_path = output_dir / "filtered_contexts.jsonl"
    audit_path = output_dir / "audit.json"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(contexts_path, contexts)
    audit.update(
        {
            "threshold_objective": selected_objective,
            "ordered_primary_ids": primary_ids,
            "provenance_hashes": {
                "threshold_manifest": file_sha256(threshold_manifest_path),
                "claims": file_sha256(claims_path),
                "cosine_rankings": file_sha256(rankings_path),
                "atomic_evidence": file_sha256(atomic_path),
                "atomic_token_matched_contexts": file_sha256(matched_path),
                "atomic_token_matched_draft_predictions": file_sha256(draft_path),
                "m3_manifest": file_sha256(directories["m3"] / "manifest.json"),
                "primary_ids": file_sha256(primary_path),
            },
        }
    )
    write_json(audit_path, audit)
    manifest = {
        "schema_version": 1,
        "experiment": "M4_support_filtered_token_matched_contexts_49",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ordered_primary_ids": primary_ids,
        "threshold": {
            "objective": selected_objective,
            "value": float(threshold),
            "manifest_sha256": file_sha256(threshold_manifest_path),
            "calibration_report_sha256": file_sha256(threshold_report_path),
        },
        "cosine_matcher": {
            "model": COSINE_MODEL,
            "revision": COSINE_REVISION,
            "rankings_sha256": file_sha256(rankings_path),
            "scores_recomputed": False,
        },
        "frozen_inputs": audit["provenance_hashes"],
        "tokenizer": {"model": MODEL_NAME, "revision": MODEL_REVISION},
        "token_budget_policy": "per-sample frozen Atomic Token-Matched b2_context_token_budget",
        "cross_sample_evidence_allowed": False,
        "fallback": "highest cosine top-1 across draft claims; score then evidence_id tie-break",
        "outputs": {
            "filtered_contexts.jsonl": file_sha256(contexts_path),
            "audit.json": file_sha256(audit_path),
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
