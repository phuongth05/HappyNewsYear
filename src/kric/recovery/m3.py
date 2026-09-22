from __future__ import annotations

import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from kric.captioning.m3 import MODEL_NAME, MODEL_REVISION, RETRIEVAL_MODEL, RETRIEVAL_REVISION
from kric.evidence.llm_frozen_pipeline import _fit_closest

from .common import FROZEN_SELECTED_SHA256, read_jsonl, require_recovery_output, sha256, unique_ids, write_json, write_jsonl

PACKING_VERSION = "llm_frozen_pipeline._fit_closest/order_preserving_max_tokens_v1"


def audit_inputs(
    *, atomic_evidence: Path, atomic_contexts_full: Path, selected_evidence: Path,
    human_review_csv: Sequence[Mapping[str, Any]], frozen_annotations: Sequence[Mapping[str, Any]],
    primary_ids: Sequence[str],
) -> dict[str, Any]:
    paths = (atomic_evidence, atomic_contexts_full, selected_evidence)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing recovery inputs: {missing}")
    atomic, full, selected = map(read_jsonl, paths)
    ids = {
        "atomic_evidence": unique_ids(atomic, "atomic evidence"),
        "atomic_contexts_full": unique_ids(full, "atomic full contexts"),
        "selected_evidence": unique_ids(selected, "selected evidence"),
    }
    if ids["atomic_evidence"] != ids["atomic_contexts_full"] or ids["atomic_evidence"] != ids["selected_evidence"]:
        raise ValueError("M3 backup ID/order mismatch")
    expected = set(map(str, primary_ids))
    expected.add("4fd292608eb7c8105d86b39a_0")
    unknown = sorted(set(ids["atomic_evidence"]) - expected)
    if unknown:
        raise ValueError(f"atomic evidence contains IDs outside expected dataset: {unknown}")
    selected_hash = sha256(selected_evidence)
    if selected_hash != FROZEN_SELECTED_SHA256:
        raise ValueError("selected evidence does not match frozen B2 hash")
    from .common import unique_claim_keys
    review_keys = unique_claim_keys(human_review_csv, "human review")
    annotation_keys = unique_claim_keys(frozen_annotations, "frozen annotations")
    if annotation_keys != review_keys:
        raise ValueError("frozen annotation claim IDs do not match human-review rows")
    return {
        "status": "valid", "row_counts": {"atomic_evidence": len(atomic), "atomic_contexts_full": len(full), "selected_evidence": len(selected), "human_review": len(human_review_csv), "frozen_annotations": len(frozen_annotations)},
        "sha256": {path.name: sha256(path) for path in paths}, "selected_evidence_matches_frozen_hash": True,
        "sample_id_coverage": {"backup_rows": len(ids["atomic_evidence"]), "primary_49_covered": len(set(map(str, primary_ids)) & set(ids["atomic_evidence"])), "unknown_ids": []},
        "human_review_claim_keys_unique": True, "annotation_keys_match_review": True,
    }


def reconstruct_token_matched(
    *, atomic_evidence: Path, atomic_contexts_full: Path, selected_evidence: Path,
    primary_ids: Sequence[str], output_dir: Path, token_counter: Callable[[str], int],
) -> dict[str, Any]:
    output_dir = require_recovery_output(output_dir)
    atomic = read_jsonl(atomic_evidence); full = read_jsonl(atomic_contexts_full); selected = read_jsonl(selected_evidence)
    atomic_ids = unique_ids(atomic, "atomic evidence")
    if atomic_ids != unique_ids(full, "full contexts") or atomic_ids != unique_ids(selected, "selected evidence"):
        raise ValueError("backup ID/order mismatch")
    if sha256(selected_evidence) != FROZEN_SELECTED_SHA256:
        raise ValueError("selected evidence hash mismatch")
    if not set(map(str, primary_ids)).issubset(set(atomic_ids)):
        raise ValueError("primary 49 IDs are not covered by recovery inputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(atomic_evidence, output_dir / "atomic_evidence.jsonl")
    shutil.copyfile(atomic_contexts_full, output_dir / "atomic_contexts_full.jsonl")
    selected_by_id = {str(row["sample_id"]): row for row in selected}
    matched: list[dict[str, Any]] = []
    for row in atomic:
        sample_id = str(row["sample_id"]); budget = int(selected_by_id[sample_id]["context_token_count"])
        units, context, tokens, dropped = _fit_closest(row["atomic_units"], token_counter, budget)
        matched.append({"sample_id": sample_id, "context_kind": "atomic_token_matched", "context": context, "context_tokens": tokens, "b2_context_token_budget": budget, "token_difference_from_b2": tokens - budget, "absolute_token_difference_from_b2": abs(tokens - budget), "included_evidence_ids": [unit["evidence_id"] for unit in units], "dropped_evidence_ids": dropped, "all_source_sentences_completed": int(row.get("failed_source_sentences", 0)) == 0})
    write_jsonl(output_dir / "atomic_contexts_token_matched.jsonl", matched)
    differences = [row["absolute_token_difference_from_b2"] for row in matched]
    audit = {"samples": len(matched), "deterministically_reconstructed_samples": len(matched), "mean_absolute_token_difference_from_b2": statistics.fmean(differences), "reconstruction_failures": 0}
    write_json(output_dir / "audit.json", audit)
    outputs = {name: sha256(output_dir / name) for name in ("atomic_evidence.jsonl", "atomic_contexts_full.jsonl", "atomic_contexts_token_matched.jsonl", "audit.json")}
    manifest = {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(), "recovery_status": "deterministically_reconstructed_from_partial_original_artifacts", "original_missing_provenance": ["atomic_contexts_token_matched.jsonl", "manifest.json", "audit.json", "failures.jsonl"], "source_artifact_hashes": {"atomic_evidence.jsonl": sha256(atomic_evidence), "atomic_contexts_full.jsonl": sha256(atomic_contexts_full), "selected_evidence.jsonl": sha256(selected_evidence)}, "ordered_sample_ids": atomic_ids, "ordered_primary_ids": list(map(str, primary_ids)), "tokenizer": {"model": MODEL_NAME, "revision": MODEL_REVISION}, "retrieval": {"model": RETRIEVAL_MODEL, "revision": RETRIEVAL_REVISION, "method": "semantic", "k": 3}, "packing_algorithm": PACKING_VERSION, "every_reconstructed_sample_deterministic": True, "caption_generation_run": False, "failures_file_created": False, "outputs": outputs}
    write_json(output_dir / "manifest.json", manifest)
    return manifest
