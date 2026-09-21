"""Strict provenance for a B2 run regenerated from frozen scientific inputs."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from kric.evaluation.io import file_sha256, load_predictions, write_json

PROVENANCE_STATUS = "regenerated_from_exact_frozen_inputs"
REGENERATED_MANIFEST_NAME = "regenerated_baseline_manifest.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 50 or len(set(value)) != 50:
        raise ValueError("B2 provenance requires exactly 50 unique ordered IDs")
    return [str(item) for item in value]


def scientific_projection(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Return only scientific settings; machine-specific paths are excluded."""

    dataset = resolved.get("dataset", {})
    model = resolved.get("model", {})
    retrieval = resolved.get("retrieval", {})
    evaluation = resolved.get("evaluation", {})
    return {
        "experiment": resolved.get("experiment"),
        "seed": resolved.get("seed"),
        "dataset": {
            "name": dataset.get("name"),
            "split": dataset.get("split"),
            "max_samples": dataset.get("max_samples"),
            "subset_strategy": dataset.get("subset_strategy"),
        },
        "model": {
            "type": model.get("type"),
            "name": model.get("name"),
            "revision": model.get("revision"),
            "dtype": model.get("dtype"),
            "batch_size": model.get("batch_size"),
        },
        "prompt": resolved.get("prompt"),
        "generation": resolved.get("generation"),
        "context": resolved.get("context"),
        "retrieval": {
            "method": retrieval.get("method"),
            "k": retrieval.get("k"),
            "model_name": retrieval.get("model_name"),
            "model_revision": retrieval.get("model_revision"),
        },
        "evaluation": {
            "enabled": evaluation.get("enabled"),
            "metrics": evaluation.get("metrics"),
            "allow_metric_errors": evaluation.get("allow_metric_errors"),
            "entity_extractor": evaluation.get("entity_extractor"),
            "spacy_model": evaluation.get("spacy_model"),
        },
    }


def scientific_projection_sha256(resolved: Mapping[str, Any]) -> str:
    payload = json.dumps(
        scientific_projection(resolved),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_ordered_ids(
    *, selected_path: Path, predictions_path: Path, metrics_path: Path, expected: Sequence[str]
) -> None:
    selected = _jsonl(selected_path)
    selected_ids = [str(row.get("sample_id")) for row in selected]
    prediction_ids = [record.sample_id for record in load_predictions(predictions_path)]
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metric_ids = [str(row.get("sample_id")) for row in csv.DictReader(handle)]
    for label, actual in (
        ("selected evidence", selected_ids),
        ("predictions", prediction_ids),
        ("per-sample metrics", metric_ids),
    ):
        if actual != list(expected):
            raise ValueError(f"{label} IDs/order differ from the frozen 50")


def _validate_run_manifest(manifest: Mapping[str, Any], resolved: Mapping[str, Any]) -> None:
    if manifest.get("experiment") != "B2_instructblip_sentence_context":
        raise ValueError("unexpected B2 experiment")
    model = manifest.get("model", {})
    resolved_model = resolved.get("model", {})
    if (
        model.get("name") != resolved_model.get("name")
        or model.get("requested_revision") != resolved_model.get("revision")
        or model.get("resolved_revision") != resolved_model.get("revision")
        or model.get("prompt") != resolved.get("prompt")
    ):
        raise ValueError("B2 run manifest model/revision/prompt mismatch")
    retrieval = manifest.get("retrieval", {})
    resolved_retrieval = resolved.get("retrieval", {})
    for key in ("method", "k", "model_name", "model_revision"):
        if retrieval.get(key) != resolved_retrieval.get(key):
            raise ValueError(f"B2 run manifest retrieval mismatch: {key}")
    if manifest.get("seed") != resolved.get("seed"):
        raise ValueError("B2 run manifest seed mismatch")


def create_regenerated_baseline_manifest(
    *,
    historical_dir: Path,
    regenerated_dir: Path,
    ids_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate exact-input equivalence and write a non-historical provenance manifest."""

    required_historical = ("manifest.json", "resolved_config.json", "selected_evidence.jsonl")
    required_regenerated = (
        "manifest.json",
        "resolved_config.json",
        "selected_evidence.jsonl",
        "predictions.jsonl",
        "metrics.json",
        "metrics_per_sample.csv",
    )
    for directory, names in (
        (historical_dir, required_historical),
        (regenerated_dir, required_regenerated),
    ):
        missing = [name for name in names if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing B2 artifacts in {directory}: {missing}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite provenance manifest: {output_path}")

    expected_ids = _ids(ids_path)
    historical_manifest = _json(historical_dir / "manifest.json")
    regenerated_manifest = _json(regenerated_dir / "manifest.json")
    historical_resolved = _json(historical_dir / "resolved_config.json")
    regenerated_resolved = _json(regenerated_dir / "resolved_config.json")
    _validate_run_manifest(historical_manifest, historical_resolved)
    _validate_run_manifest(regenerated_manifest, regenerated_resolved)

    historical_projection = scientific_projection(historical_resolved)
    regenerated_projection = scientific_projection(regenerated_resolved)
    if regenerated_projection != historical_projection:
        changed = {
            key: {
                "historical": historical_projection.get(key),
                "regenerated": regenerated_projection.get(key),
            }
            for key in historical_projection
            if historical_projection.get(key) != regenerated_projection.get(key)
        }
        raise ValueError(f"regenerated B2 scientific settings changed: {changed}")

    historical_selected = historical_dir / "selected_evidence.jsonl"
    regenerated_selected = regenerated_dir / "selected_evidence.jsonl"
    selected_hash = file_sha256(historical_selected)
    if file_sha256(regenerated_selected) != selected_hash:
        raise ValueError("regenerated selected evidence differs from frozen evidence")
    historical_rankings_hash = historical_manifest.get("rankings_sha256")
    regenerated_rankings_hash = regenerated_manifest.get("rankings_sha256")
    if not historical_rankings_hash or regenerated_rankings_hash != historical_rankings_hash:
        raise ValueError("regenerated rankings hash differs from frozen rankings")
    _validate_ordered_ids(
        selected_path=regenerated_selected,
        predictions_path=regenerated_dir / "predictions.jsonl",
        metrics_path=regenerated_dir / "metrics_per_sample.csv",
        expected=expected_ids,
    )
    if [str(row.get("sample_id")) for row in _jsonl(historical_selected)] != expected_ids:
        raise ValueError("historical selected evidence IDs/order differ from frozen 50")

    prediction_hash = file_sha256(regenerated_dir / "predictions.jsonl")
    prediction_meta = regenerated_manifest.get("predictions", {})
    if prediction_meta.get("sha256") != prediction_hash or prediction_meta.get("samples") != 50:
        raise ValueError("regenerated run manifest does not authenticate its predictions")
    runtime = (
        _json(regenerated_dir / "runtime.json")
        if (regenerated_dir / "runtime.json").is_file()
        else {}
    )
    model = regenerated_manifest.get("model", {})
    result = {
        "schema_version": 1,
        "provenance_status": PROVENANCE_STATUS,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "historical_manifest": {
            "path": str((historical_dir / "manifest.json").resolve()),
            "sha256": file_sha256(historical_dir / "manifest.json"),
            "created_utc": historical_manifest.get("created_utc"),
            "git_commit": historical_manifest.get("git", {}).get("commit"),
            "historical_predictions_sha256": historical_manifest.get("predictions", {}).get(
                "sha256"
            ),
        },
        "frozen_inputs": {
            "selected_evidence_sha256": selected_hash,
            "rankings_sha256": historical_rankings_hash,
            "ordered_sample_ids": expected_ids,
            "ordered_sample_ids_sha256": hashlib.sha256(
                json.dumps(expected_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "scientific_config_sha256": scientific_projection_sha256(
                historical_resolved
            ),
        },
        "regenerated_run": {
            "manifest_path": str((regenerated_dir / "manifest.json").resolve()),
            "manifest_sha256": file_sha256(regenerated_dir / "manifest.json"),
            "resolved_config_sha256": file_sha256(
                regenerated_dir / "resolved_config.json"
            ),
        },
        "regenerated_outputs": {
            "predictions_sha256": prediction_hash,
            "metrics_sha256": file_sha256(regenerated_dir / "metrics.json"),
            "metrics_per_sample_sha256": file_sha256(
                regenerated_dir / "metrics_per_sample.csv"
            ),
            "regenerated_utc": regenerated_manifest.get("created_utc"),
            "environment_versions": {
                "torch": model.get("torch_version"),
                "transformers": model.get("transformers_version"),
                "python": runtime.get("python"),
                "platform": runtime.get("platform"),
                "packages": runtime.get("packages", {}),
                "model_class": model.get("class"),
                "processor_class": model.get("processor_class"),
            },
        },
        "allowed_differences": [
            "dataset.config",
            "output_dir",
            "comparison.baseline_config",
            "comparison.baseline_per_sample",
            "retrieval.rankings_path",
        ],
    }
    if not result["historical_manifest"]["historical_predictions_sha256"]:
        raise ValueError("historical manifest does not record its prediction hash")
    write_json(output_path, result)
    return result


def validate_m3_b2_provenance(
    *,
    b2_dir: Path,
    atomic_manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    expected_ids: Sequence[str],
) -> str:
    """Accept only the exact historical run or an authenticated regenerated run."""

    run_manifest_path = b2_dir / "manifest.json"
    run_manifest = _json(run_manifest_path)
    predictions_path = b2_dir / "predictions.jsonl"
    actual_prediction_hash = file_sha256(predictions_path)
    expected_historical_manifest_hash = (
        atomic_manifest.get("source", {}).get("manifest", {}).get("sha256")
    )
    if not expected_historical_manifest_hash:
        raise ValueError("atomic manifest lacks historical B2 manifest provenance")

    is_historical_manifest = (
        file_sha256(run_manifest_path) == expected_historical_manifest_hash
    )
    prediction_meta = run_manifest.get("predictions", {})
    _validate_ordered_ids(
        selected_path=b2_dir / "selected_evidence.jsonl",
        predictions_path=predictions_path,
        metrics_path=b2_dir / "metrics_per_sample.csv",
        expected=expected_ids,
    )
    if (
        is_historical_manifest
        and prediction_meta.get("sha256") == actual_prediction_hash
        and prediction_meta.get("samples") == 50
    ):
        return "exact_historical_frozen_predictions"

    provenance_path = b2_dir / REGENERATED_MANIFEST_NAME
    if not provenance_path.is_file():
        raise ValueError(
            "B2 predictions are not historical and no validated regenerated baseline manifest exists"
        )
    provenance = _json(provenance_path)
    if provenance.get("provenance_status") != PROVENANCE_STATUS:
        raise ValueError("invalid regenerated B2 provenance status")
    if (
        provenance.get("historical_manifest", {}).get("sha256")
        != expected_historical_manifest_hash
    ):
        raise ValueError("regenerated baseline references the wrong historical manifest")
    frozen = provenance.get("frozen_inputs", {})
    selected_path = b2_dir / "selected_evidence.jsonl"
    expected_selected_hash = (
        atomic_manifest.get("source", {}).get("selected_evidence", {}).get("sha256")
    )
    if (
        frozen.get("selected_evidence_sha256") != file_sha256(selected_path)
        or frozen.get("selected_evidence_sha256") != expected_selected_hash
    ):
        raise ValueError("regenerated baseline selected-evidence hash mismatch")
    expected_rankings_hash = (
        atomic_manifest.get("source", {}).get("rankings", {}).get("sha256")
    )
    if (
        frozen.get("rankings_sha256") != run_manifest.get("rankings_sha256")
        or frozen.get("rankings_sha256") != expected_rankings_hash
    ):
        raise ValueError("regenerated baseline rankings hash mismatch")
    if frozen.get("scientific_config_sha256") != scientific_projection_sha256(resolved):
        raise ValueError("regenerated baseline scientific configuration mismatch")
    if frozen.get("ordered_sample_ids") != list(expected_ids):
        raise ValueError("regenerated baseline ordered IDs mismatch")
    regenerated_run = provenance.get("regenerated_run", {})
    if regenerated_run.get("manifest_sha256") != file_sha256(run_manifest_path):
        raise ValueError("regenerated run manifest hash mismatch")
    if regenerated_run.get("resolved_config_sha256") != file_sha256(
        b2_dir / "resolved_config.json"
    ):
        raise ValueError("regenerated resolved-config hash mismatch")
    outputs = provenance.get("regenerated_outputs", {})
    expected_outputs = {
        "predictions_sha256": predictions_path,
        "metrics_sha256": b2_dir / "metrics.json",
        "metrics_per_sample_sha256": b2_dir / "metrics_per_sample.csv",
    }
    for key, path in expected_outputs.items():
        if outputs.get(key) != file_sha256(path):
            raise ValueError(f"regenerated B2 output hash mismatch: {key}")
    if prediction_meta.get("sha256") != actual_prediction_hash or prediction_meta.get("samples") != 50:
        raise ValueError("regenerated run manifest does not authenticate predictions")
    _validate_ordered_ids(
        selected_path=selected_path,
        predictions_path=predictions_path,
        metrics_path=b2_dir / "metrics_per_sample.csv",
        expected=expected_ids,
    )
    return PROVENANCE_STATUS


__all__ = [
    "PROVENANCE_STATUS",
    "REGENERATED_MANIFEST_NAME",
    "create_regenerated_baseline_manifest",
    "scientific_projection",
    "scientific_projection_sha256",
    "validate_m3_b2_provenance",
]