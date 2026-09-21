import copy
import csv
import json
from pathlib import Path

import pytest

from kric.captioning.b2_regenerated import (
    PROVENANCE_STATUS,
    create_regenerated_baseline_manifest,
    validate_m3_b2_provenance,
)
from kric.evaluation.io import file_sha256


ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _resolved(path_prefix: str):
    return {
        "experiment": "B2_instructblip_sentence_context",
        "seed": 2026,
        "dataset": {
            "name": "goodnews",
            "config": f"{path_prefix}/goodnews.yaml",
            "split": "dev",
            "max_samples": 50,
            "subset_strategy": "first_by_sample_id",
        },
        "model": {
            "type": "instructblip",
            "name": "Salesforce/instructblip-flan-t5-xl",
            "revision": "bf9bd58cb3bb06e88c03b80f9cc346b90094bc6d",
            "device": "auto",
            "dtype": "float16",
            "batch_size": 1,
        },
        "prompt": {
            "instruction": "Describe the image in one factual news-style sentence.",
            "context_prefix": "Article context:\n",
            "context_suffix": "\n\n",
        },
        "context": {"max_context_tokens": 384, "truncation": "head"},
        "generation": {
            "do_sample": False,
            "num_beams": 3,
            "max_new_tokens": 30,
            "min_new_tokens": 1,
            "no_repeat_ngram_size": 2,
            "length_penalty": 1.0,
            "early_stopping": True,
        },
        "retrieval": {
            "method": "semantic",
            "k": 3,
            "rankings_path": f"{path_prefix}/semantic.jsonl",
            "model_name": "openai/clip-vit-base-patch32",
            "model_revision": "b97b0100e55e367c057773c2a614676470b0d575",
        },
        "evaluation": {
            "enabled": True,
            "metrics": ["cider", "entity"],
            "allow_metric_errors": False,
            "entity_extractor": "spacy",
            "spacy_model": "en_core_web_sm",
        },
        "comparison": {
            "baseline_config": f"{path_prefix}/b1.yaml",
            "baseline_per_sample": f"{path_prefix}/b1_metrics.csv",
        },
        "output_dir": f"{path_prefix}/semantic_k3",
    }


def _run_manifest(resolved, prediction_hash, *, created):
    return {
        "schema_version": 1,
        "created_utc": created,
        "experiment": "B2_instructblip_sentence_context",
        "seed": 2026,
        "retrieval": dict(resolved["retrieval"]),
        "rankings_sha256": "rankings-frozen-sha256",
        "model": {
            "name": resolved["model"]["name"],
            "requested_revision": resolved["model"]["revision"],
            "resolved_revision": resolved["model"]["revision"],
            "prompt": resolved["prompt"],
            "torch_version": "test-torch",
            "transformers_version": "test-transformers",
            "class": "InstructBlipForConditionalGeneration",
            "processor_class": "InstructBlipProcessor",
        },
        "git": {"commit": "test-commit"},
        "predictions": {
            "sha256": prediction_hash,
            "samples": 50,
        },
    }


def _artifacts(tmp_path: Path):
    historical = tmp_path / "historical"
    regenerated = tmp_path / "regenerated"
    historical.mkdir()
    regenerated.mkdir()
    ids = [f"sample-{index:02d}" for index in range(50)]
    ids_path = tmp_path / "ids.json"
    _write_json(ids_path, ids)
    selected = [
        {
            "sample_id": sample_id,
            "retrieval_method": "semantic",
            "retrieval_k": 3,
            "selected_sentence_ids": [0, 1, 2],
            "selected_sentence_texts": ["a", "b", "c"],
            "selected_ranks": [1, 2, 3],
            "selected_ranking_scores": [0.9, 0.8, 0.7],
            "context_token_count": 3,
        }
        for sample_id in ids
    ]
    _write_jsonl(historical / "selected_evidence.jsonl", selected)
    _write_jsonl(regenerated / "selected_evidence.jsonl", selected)
    predictions = [
        {
            "sample_id": sample_id,
            "prediction": f"prediction {index}",
            "reference": f"reference {index}",
            "metadata": {},
        }
        for index, sample_id in enumerate(ids)
    ]
    _write_jsonl(regenerated / "predictions.jsonl", predictions)
    (regenerated / "metrics.json").write_text("{}\n", encoding="utf-8")
    with (regenerated / "metrics_per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "CIDEr"])
        writer.writeheader()
        for sample_id in ids:
            writer.writerow({"sample_id": sample_id, "CIDEr": 0})
    historical_resolved = _resolved("/historical/location")
    regenerated_resolved = _resolved("/different/regenerated/location")
    _write_json(historical / "resolved_config.json", historical_resolved)
    _write_json(regenerated / "resolved_config.json", regenerated_resolved)
    historical_manifest = _run_manifest(
        historical_resolved,
        "historical-predictions-no-longer-available",
        created="2026-09-17T00:00:00+00:00",
    )
    regenerated_manifest = _run_manifest(
        regenerated_resolved,
        file_sha256(regenerated / "predictions.jsonl"),
        created="2026-09-21T00:00:00+00:00",
    )
    _write_json(historical / "manifest.json", historical_manifest)
    _write_json(regenerated / "manifest.json", regenerated_manifest)
    return historical, regenerated, ids_path, ids


def _create(tmp_path: Path):
    historical, regenerated, ids_path, ids = _artifacts(tmp_path)
    output = regenerated / "regenerated_baseline_manifest.json"
    manifest = create_regenerated_baseline_manifest(
        historical_dir=historical,
        regenerated_dir=regenerated,
        ids_path=ids_path,
        output_path=output,
    )
    return historical, regenerated, ids_path, ids, output, manifest


def test_path_only_differences_are_allowed(tmp_path):
    historical, regenerated, _, _, output, manifest = _create(tmp_path)
    assert output.is_file()
    assert manifest["provenance_status"] == PROVENANCE_STATUS
    assert manifest["historical_manifest"]["sha256"] == file_sha256(
        historical / "manifest.json"
    )
    assert manifest["regenerated_outputs"]["predictions_sha256"] == file_sha256(
        regenerated / "predictions.jsonl"
    )


def test_altered_selected_evidence_is_rejected(tmp_path):
    historical, regenerated, ids_path, _ = _artifacts(tmp_path)
    rows = [json.loads(line) for line in (regenerated / "selected_evidence.jsonl").read_text().splitlines()]
    rows[0]["selected_sentence_texts"][0] = "changed"
    _write_jsonl(regenerated / "selected_evidence.jsonl", rows)
    with pytest.raises(ValueError, match="selected evidence"):
        create_regenerated_baseline_manifest(
            historical_dir=historical,
            regenerated_dir=regenerated,
            ids_path=ids_path,
            output_path=regenerated / "provenance.json",
        )


def test_reordered_sample_ids_are_rejected(tmp_path):
    historical, regenerated, ids_path, _ = _artifacts(tmp_path)
    rows = [json.loads(line) for line in (regenerated / "predictions.jsonl").read_text().splitlines()]
    rows[0], rows[1] = rows[1], rows[0]
    _write_jsonl(regenerated / "predictions.jsonl", rows)
    manifest = json.loads((regenerated / "manifest.json").read_text())
    manifest["predictions"]["sha256"] = file_sha256(regenerated / "predictions.jsonl")
    _write_json(regenerated / "manifest.json", manifest)
    with pytest.raises(ValueError, match="predictions IDs/order"):
        create_regenerated_baseline_manifest(
            historical_dir=historical,
            regenerated_dir=regenerated,
            ids_path=ids_path,
            output_path=regenerated / "provenance.json",
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value["prompt"].update(instruction="Changed prompt"), "prompt"),
        (lambda value: value["generation"].update(num_beams=4), "scientific settings"),
        (lambda value: value["model"].update(revision="changed-revision"), "revision"),
    ],
)
def test_changed_scientific_setting_is_rejected(tmp_path, change, message):
    historical, regenerated, ids_path, _ = _artifacts(tmp_path)
    resolved = json.loads((regenerated / "resolved_config.json").read_text())
    change(resolved)
    _write_json(regenerated / "resolved_config.json", resolved)
    if "prompt" in message:
        manifest = json.loads((regenerated / "manifest.json").read_text())
        manifest["model"]["prompt"] = resolved["prompt"]
        _write_json(regenerated / "manifest.json", manifest)
    with pytest.raises(ValueError, match=message):
        create_regenerated_baseline_manifest(
            historical_dir=historical,
            regenerated_dir=regenerated,
            ids_path=ids_path,
            output_path=regenerated / "provenance.json",
        )


def test_exact_historical_predictions_remain_accepted(tmp_path):
    historical, regenerated, _, ids = _artifacts(tmp_path)
    for name in ("predictions.jsonl", "metrics.json", "metrics_per_sample.csv"):
        (historical / name).write_bytes((regenerated / name).read_bytes())
    manifest = json.loads((historical / "manifest.json").read_text())
    manifest["predictions"]["sha256"] = file_sha256(
        historical / "predictions.jsonl"
    )
    _write_json(historical / "manifest.json", manifest)
    atomic_manifest = {
        "source": {
            "manifest": {"sha256": file_sha256(historical / "manifest.json")},
            "selected_evidence": {
                "sha256": file_sha256(historical / "selected_evidence.jsonl")
            },
            "rankings": {"sha256": "rankings-frozen-sha256"},
        }
    }
    resolved = json.loads((historical / "resolved_config.json").read_text())
    assert validate_m3_b2_provenance(
        b2_dir=historical,
        atomic_manifest=atomic_manifest,
        resolved=resolved,
        expected_ids=ids,
    ) == "exact_historical_frozen_predictions"


def test_regenerated_predictions_require_validated_manifest(tmp_path):
    historical, regenerated, _, ids = _artifacts(tmp_path)
    atomic_manifest = {
        "source": {
            "manifest": {"sha256": file_sha256(historical / "manifest.json")},
            "selected_evidence": {
                "sha256": file_sha256(historical / "selected_evidence.jsonl")
            },
            "rankings": {"sha256": "rankings-frozen-sha256"},
        }
    }
    resolved = json.loads((regenerated / "resolved_config.json").read_text())
    with pytest.raises(ValueError, match="no validated regenerated"):
        validate_m3_b2_provenance(
            b2_dir=regenerated,
            atomic_manifest=atomic_manifest,
            resolved=resolved,
            expected_ids=ids,
        )
    output = regenerated / "regenerated_baseline_manifest.json"
    create_regenerated_baseline_manifest(
        historical_dir=historical,
        regenerated_dir=regenerated,
        ids_path=tmp_path / "ids.json",
        output_path=output,
    )
    assert (
        validate_m3_b2_provenance(
            b2_dir=regenerated,
            atomic_manifest=atomic_manifest,
            resolved=resolved,
            expected_ids=ids,
        )
        == PROVENANCE_STATUS
    )
    provenance = json.loads(output.read_text())
    provenance["regenerated_outputs"]["predictions_sha256"] = "arbitrary"
    _write_json(output, provenance)
    with pytest.raises(ValueError, match="output hash mismatch"):
        validate_m3_b2_provenance(
            b2_dir=regenerated,
            atomic_manifest=atomic_manifest,
            resolved=resolved,
            expected_ids=ids,
        )