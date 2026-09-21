"""Run controlled M4 support-filtered generation, evaluation, and comparison."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.captioning.instructblip import InstructBlipArticleCaptioner  # noqa: E402
from kric.captioning.m3 import (  # noqa: E402
    EXPECTED_GENERATION,
    EXPECTED_PROMPT,
    FAILURE_SAMPLE_ID,
    MODEL_NAME,
    MODEL_REVISION,
    controlled_captioner_settings,
    load_frozen_m3_inputs,
    validate_generator_control,
)
from kric.captioning.m4 import compare_three_methods, generate_filtered_variant  # noqa: E402
from kric.captioning.runner import (  # noqa: E402
    _finish_gpu_memory_tracking,
    _git_state,
    _start_gpu_memory_tracking,
    _versions,
)
from kric.data.goodnews import GoodNewsConfig, GoodNewsDataset  # noqa: E402
from kric.data.selection import select_samples  # noqa: E402
from kric.evaluation.io import file_sha256, write_json  # noqa: E402
from kric.matching.ranking import read_jsonl  # noqa: E402


def _run(command: list[str], label: str) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    result = {
        "label": label,
        "command": command,
        "return_code": completed.returncode,
        "seconds": time.perf_counter() - started,
    }
    if completed.returncode:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")
    return result


def _resolve(config: Mapping[str, Any], key: str) -> Path:
    source = Path(config["_source_path"])
    value = Path(str(config[key])).expanduser()
    return value.resolve() if value.is_absolute() else (source.parent / value).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("M4 generation config must be a mapping")
    config = copy.deepcopy(dict(raw))
    if config.get("experiment") != "M4_support_filtered_token_matched_generation":
        raise ValueError("unexpected M4 filtered-generation experiment")
    validate_generator_control(config)
    if config.get("context", {}).get("budget_policy") != "frozen_atomic_token_matched_per_sample":
        raise ValueError("M4 must preserve the Atomic Token-Matched budget policy")
    if config.get("analysis") != {
        "primary_sample_count": 49,
        "excluded_sample_id": FAILURE_SAMPLE_ID,
        "resamples": 10_000,
        "confidence": 0.95,
        "seed": 2026,
    }:
        raise ValueError("M4 comparison settings differ from the frozen protocol")
    config["_source_path"] = source
    return config


def _load_samples(dataset_root: Path, frozen_ids: tuple[str, ...], primary_ids: tuple[str, ...]):
    root = dataset_root.expanduser().resolve()
    dataset = GoodNewsDataset(
        GoodNewsConfig(
            annotations_path=root / "article+caption.json",
            splits_path=root / "img_splits.json",
            images_root=root / "images",
        )
    )
    dataset.assert_split_integrity()
    samples = select_samples(dataset, split="dev", max_samples=50, strategy="first_by_sample_id")
    if [sample.sample_id for sample in samples] != list(frozen_ids):
        raise ValueError("GoodNews does not resolve the exact frozen ordered 50 IDs")
    primary_set = set(primary_ids)
    selected = [sample for sample in samples if sample.sample_id in primary_set]
    if [sample.sample_id for sample in selected] != list(primary_ids):
        raise ValueError("GoodNews does not resolve the exact ordered primary 49 IDs")
    missing = [sample.sample_id for sample in selected if not Path(sample.image_path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing M4 images: {missing}")
    return selected


def _evaluation_command(predictions: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate.py"),
        "--predictions",
        str(predictions),
        "--output",
        str(output_dir / "metrics.json"),
        "--per-sample-output",
        str(output_dir / "metrics_per_sample.csv"),
        "--metrics",
        "cider,entity",
        "--entity-extractor",
        "spacy",
        "--spacy-model",
        "en_core_web_sm",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--b2-dir", required=True, type=Path)
    parser.add_argument("--atomic-dir", required=True, type=Path)
    parser.add_argument("--m3-dir", required=True, type=Path)
    parser.add_argument("--filtered-context-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = _load_config(args.config)
    b2_dir = args.b2_dir.expanduser().resolve()
    atomic_dir = args.atomic_dir.expanduser().resolve()
    m3_dir = args.m3_dir.expanduser().resolve()
    filtered_dir = args.filtered_context_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {predictions_path}")
    started = time.perf_counter()

    frozen = load_frozen_m3_inputs(
        b2_dir=b2_dir,
        atomic_dir=atomic_dir,
        ids_50_path=_resolve(config, "ids_50"),
        ids_49_path=_resolve(config, "primary_ids_49"),
    )
    filtered_manifest_path = filtered_dir / "manifest.json"
    filtered_context_path = filtered_dir / "filtered_contexts.jsonl"
    filtered_manifest = json.loads(filtered_manifest_path.read_text(encoding="utf-8"))
    if filtered_manifest.get("experiment") != "M4_support_filtered_token_matched_contexts_49":
        raise ValueError("unexpected filtered-context manifest")
    if filtered_manifest.get("ordered_primary_ids") != list(frozen.ids_49):
        raise ValueError("filtered contexts do not use the ordered primary 49")
    if filtered_manifest.get("outputs", {}).get("filtered_contexts.jsonl") != file_sha256(
        filtered_context_path
    ):
        raise ValueError("filtered context hash mismatch")
    if filtered_manifest.get("frozen_inputs", {}).get("atomic_evidence") != file_sha256(
        atomic_dir / "atomic_evidence.jsonl"
    ):
        raise ValueError("filtered context atomic provenance mismatch")
    contexts = read_jsonl(filtered_context_path)
    if [str(row.get("sample_id")) for row in contexts] != list(frozen.ids_49):
        raise ValueError("filtered context IDs/order differ from primary 49")
    samples = _load_samples(args.dataset_root, frozen.ids_50, frozen.ids_49)

    output_dir.mkdir(parents=True, exist_ok=True)
    workflow = {"commands": []}
    workflow["commands"].append(
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/check_evaluation_dependencies.py"),
                "--metrics",
                "cider,entity",
                "--entity-extractor",
                "spacy",
                "--spacy-model",
                "en_core_web_sm",
                "--output",
                str(output_dir / "evaluation_preflight.json"),
            ],
            "Evaluation dependency preflight",
        )
    )
    workflow["commands"].append(
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/gpu_preflight.py"),
                "--output",
                str(output_dir / "gpu_preflight.json"),
            ],
            "GPU preflight",
        )
    )

    model, generation, prompt, context_config, seed = controlled_captioner_settings(
        frozen.b2_resolved
    )
    captioner = InstructBlipArticleCaptioner(model, generation, prompt, context_config, seed)
    gpu_start = _start_gpu_memory_tracking()
    load_started = time.perf_counter()
    captioner.load()
    model_load_seconds = time.perf_counter() - load_started
    model_info = captioner.model_info()
    if model_info.get("resolved_revision") != MODEL_REVISION:
        raise RuntimeError("loaded model revision differs from frozen M3")
    generation_started = time.perf_counter()
    prediction_rows = generate_filtered_variant(
        samples=samples,
        contexts=contexts,
        expected_ids=frozen.ids_49,
        captioner=captioner,
        output_dir=output_dir,
    )
    generation_seconds = time.perf_counter() - generation_started
    workflow["commands"].append(
        _run(_evaluation_command(predictions_path, output_dir), "Evaluate M4 filtered generation")
    )
    workflow["gpu_memory"] = _finish_gpu_memory_tracking(gpu_start)

    m3_manifest = json.loads((m3_dir / "manifest.json").read_text(encoding="utf-8"))
    m3_variant = m3_manifest.get("workflow", {}).get("variants", {}).get(
        "m3_atomic_token_matched", {}
    )
    m3_token_dir = m3_dir / "atomic_token_matched"
    for name, key in (
        ("predictions.jsonl", "predictions_sha256"),
        ("metrics.json", "metrics_sha256"),
        ("metrics_per_sample.csv", "metrics_per_sample_sha256"),
    ):
        if m3_variant.get(key) != file_sha256(m3_token_dir / name):
            raise ValueError(f"M3 atomic-token-matched provenance mismatch: {name}")
    comparison = compare_three_methods(
        method_dirs={
            "b2_semantic_k3": b2_dir,
            "m3_atomic_token_matched": m3_token_dir,
            "m4_support_filtered": output_dir,
        },
        primary_ids=frozen.ids_49,
        output_dir=output_dir,
        resamples=10_000,
        seed=2026,
    )

    resolved = copy.deepcopy(frozen.b2_resolved)
    resolved.update(
        {
            "experiment": "M4_support_filtered_token_matched_generation",
            "output_dir": str(output_dir),
            "context_selection": {
                "kind": "support_filtered_token_matched",
                "source": str(filtered_context_path),
                "source_sha256": file_sha256(filtered_context_path),
                "threshold": filtered_manifest["threshold"],
                "per_sample_budget_source": "frozen Atomic Token-Matched",
            },
        }
    )
    validate_generator_control(resolved)
    write_json(output_dir / "resolved_config.json", resolved)
    manifest = {
        "schema_version": 1,
        "experiment": "M4_support_filtered_token_matched_generation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_sample_count": 49,
        "ordered_primary_ids": list(frozen.ids_49),
        "generator_control": {
            "model": MODEL_NAME,
            "revision": MODEL_REVISION,
            "prompt": EXPECTED_PROMPT,
            "generation": EXPECTED_GENERATION,
            "seed": 2026,
            "max_context_tokens": 384,
            "all_controls_equal_to_m3_atomic_token_matched": True,
            "only_context_selection_changed": True,
        },
        "frozen_inputs": {
            "b2_predictions_sha256": file_sha256(b2_dir / "predictions.jsonl"),
            "m3_manifest_sha256": file_sha256(m3_dir / "manifest.json"),
            "m3_atomic_token_matched_predictions_sha256": file_sha256(
                m3_token_dir / "predictions.jsonl"
            ),
            "filtered_context_manifest_sha256": file_sha256(filtered_manifest_path),
            "filtered_contexts_sha256": file_sha256(filtered_context_path),
            "atomic_evidence_sha256": file_sha256(atomic_dir / "atomic_evidence.jsonl"),
            "primary_ids_sha256": file_sha256(_resolve(config, "primary_ids_49")),
        },
        "outputs": {
            "predictions.jsonl": file_sha256(predictions_path),
            "metrics.json": file_sha256(output_dir / "metrics.json"),
            "metrics_per_sample.csv": file_sha256(output_dir / "metrics_per_sample.csv"),
            "comparison.json": file_sha256(output_dir / "comparison.json"),
            "comparison_per_sample.csv": file_sha256(
                output_dir / "comparison_per_sample.csv"
            ),
        },
        "evaluation": {
            "metrics": ["CIDEr", "EntityPrecision", "EntityRecall", "EntityF1"],
            "paired_bootstrap_resamples": 10_000,
            "paired_bootstrap_seed": 2026,
            "human_calibration_claims_used_as_generation_gold": False,
            "factuality_claim_from_cider_alone": False,
        },
        "workflow": {
            **workflow,
            "model": model_info,
            "model_load_seconds": model_load_seconds,
            "generation_seconds": generation_seconds,
            "generated_samples": len(prediction_rows),
            "comparison_completed": bool(comparison),
        },
        "git": _git_state(PROJECT_ROOT),
        "packages": _versions(),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
