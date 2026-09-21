"""Run frozen M3 Atomic Full and Token-Matched caption generation and analysis."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
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
    build_atomic_model_inputs,
    controlled_captioner_settings,
    generate_atomic_variant,
    load_frozen_m3_inputs,
    run_primary_analysis,
    validate_generator_control,
)
from kric.captioning.runner import (  # noqa: E402
    _finish_gpu_memory_tracking,
    _git_state,
    _start_gpu_memory_tracking,
    _versions,
)
from kric.data.goodnews import GoodNewsConfig, GoodNewsDataset  # noqa: E402
from kric.data.selection import select_samples  # noqa: E402
from kric.evaluation.io import file_sha256, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--b2-dir", required=True, type=Path)
    parser.add_argument("--atomic-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--stage", choices=("all", "generate", "analyze"), default="all"
    )
    return parser.parse_args()


def load_m3_config(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("M3 config must contain a mapping")
    config = dict(raw)
    if config.get("experiment") != "M3_atomic_caption_generation":
        raise ValueError("unexpected M3 experiment name")
    validate_generator_control(config)
    evaluation = config.get("evaluation", {})
    if (
        evaluation.get("metrics") != ["cider", "entity"]
        or evaluation.get("entity_extractor") != "spacy"
        or evaluation.get("spacy_model") != "en_core_web_sm"
        or evaluation.get("allow_metric_errors") is not False
    ):
        raise ValueError("M3 requires CIDEr and spaCy entity metrics; SPICE is excluded")
    analysis = config.get("analysis", {})
    if analysis != {
        "primary_sample_count": 49,
        "excluded_sample_id": FAILURE_SAMPLE_ID,
        "resamples": 10_000,
        "confidence": 0.95,
        "seed": 2026,
    }:
        raise ValueError("M3 paired-analysis settings differ from the frozen protocol")
    config["_source_path"] = source
    return config


def _resolve_config_path(config: Mapping[str, Any], key: str) -> Path:
    source = Path(config["_source_path"])
    path = Path(str(config[key])).expanduser()
    return path.resolve() if path.is_absolute() else (source.parent / path).resolve()


def _run(command: list[str], label: str) -> dict[str, Any]:
    print(f"\n=== {label} ===", flush=True)
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


def _dataset(dataset_root: Path) -> GoodNewsDataset:
    root = dataset_root.expanduser().resolve()
    dataset = GoodNewsDataset(
        GoodNewsConfig(
            annotations_path=root / "article+caption.json",
            splits_path=root / "img_splits.json",
            images_root=root / "images",
        )
    )
    dataset.assert_split_integrity()
    return dataset


def _load_exact_samples(dataset_root: Path, expected_ids: tuple[str, ...]) -> list[Any]:
    samples = select_samples(
        _dataset(dataset_root),
        split="dev",
        max_samples=50,
        strategy="first_by_sample_id",
    )
    actual = [sample.sample_id for sample in samples]
    if actual != list(expected_ids):
        raise ValueError("GoodNews dataset does not resolve the frozen ordered 50 IDs")
    missing = [sample.sample_id for sample in samples if not Path(sample.image_path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing M3 images: {missing}")
    return samples


def _evaluation_command(predictions: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate.py"),
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


def _context_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [int(row["metadata"]["context_tokens"]) for row in rows]
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _variant_resolved(
    frozen_resolved: Mapping[str, Any],
    *,
    experiment: str,
    output_dir: Path,
    atomic_dir: Path,
    context_file: str,
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(frozen_resolved))
    resolved["experiment"] = experiment
    resolved["output_dir"] = str(output_dir)
    resolved["evidence_representation"] = {
        "kind": experiment,
        "source": str((atomic_dir / context_file).resolve()),
        "source_sha256": file_sha256(atomic_dir / context_file),
        "evidence_types_exposed_to_generator": False,
        "recomputed_or_reextracted": False,
    }
    validate_generator_control(resolved)
    return resolved


def _generate(
    *,
    config: Mapping[str, Any],
    dataset_root: Path,
    b2_dir: Path,
    atomic_dir: Path,
    output_root: Path,
) -> tuple[Any, dict[str, Any]]:
    frozen = load_frozen_m3_inputs(
        b2_dir=b2_dir,
        atomic_dir=atomic_dir,
        ids_50_path=_resolve_config_path(config, "ids_50"),
        ids_49_path=_resolve_config_path(config, "primary_ids_49"),
    )
    samples = _load_exact_samples(dataset_root, frozen.ids_50)
    for sample, frozen_prediction in zip(samples, frozen.b2_predictions, strict=True):
        if sample.reference_caption != frozen_prediction["reference"]:
            raise ValueError(
                f"dataset/B2 reference mismatch for frozen sample {sample.sample_id}"
            )
    preflight = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "check_evaluation_dependencies.py"),
        "--metrics",
        "cider,entity",
        "--entity-extractor",
        "spacy",
        "--spacy-model",
        "en_core_web_sm",
        "--output",
        str(output_root / "evaluation_preflight.json"),
    ]
    workflow = {"commands": [_run(preflight, "Evaluation dependency preflight")]}
    workflow["commands"].append(
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "gpu_preflight.py"),
                "--output",
                str(output_root / "gpu_preflight.json"),
            ],
            "GPU preflight",
        )
    )
    model, generation, prompt, context, seed = controlled_captioner_settings(
        frozen.b2_resolved
    )
    captioner = InstructBlipArticleCaptioner(
        model, generation, prompt, context, seed
    )
    gpu_start = _start_gpu_memory_tracking()
    load_started = time.perf_counter()
    captioner.load()
    model_load_seconds = time.perf_counter() - load_started
    model_info = captioner.model_info()
    if model_info.get("resolved_revision") != MODEL_REVISION:
        raise RuntimeError("loaded model revision differs from frozen B2")

    variants = (
        (
            "atomic_full",
            "m3_atomic_full",
            list(frozen.full_contexts),
            "atomic_contexts_full.jsonl",
        ),
        (
            "atomic_token_matched",
            "m3_atomic_token_matched",
            list(frozen.matched_contexts),
            "atomic_contexts_token_matched.jsonl",
        ),
    )
    summaries: dict[str, Any] = {}
    for directory_name, experiment, contexts, source_name in variants:
        output_dir = output_root / directory_name
        predictions_path = output_dir / "predictions.jsonl"
        if predictions_path.exists():
            raise FileExistsError(f"refusing to overwrite existing run: {predictions_path}")
        # Validate ID/order before any variant generation begins.
        build_atomic_model_inputs(samples, contexts, frozen.ids_50)
        started = time.perf_counter()
        rows = generate_atomic_variant(
            variant=experiment,
            samples=samples,
            contexts=contexts,
            frozen=frozen,
            captioner=captioner,
            output_dir=output_dir,
        )
        generation_seconds = time.perf_counter() - started
        resolved = _variant_resolved(
            frozen.b2_resolved,
            experiment=experiment,
            output_dir=output_dir,
            atomic_dir=atomic_dir,
            context_file=source_name,
        )
        write_json(output_dir / "resolved_config.json", resolved)
        workflow["commands"].append(
            _run(_evaluation_command(predictions_path, output_dir), f"Evaluate {experiment}")
        )
        summaries[experiment] = {
            "samples": len(rows),
            "generation_seconds": generation_seconds,
            "context_tokens": _context_statistics(rows),
            "predictions_sha256": file_sha256(predictions_path),
            "metrics_sha256": file_sha256(output_dir / "metrics.json"),
            "metrics_per_sample_sha256": file_sha256(
                output_dir / "metrics_per_sample.csv"
            ),
        }
    workflow["gpu_memory"] = _finish_gpu_memory_tracking(gpu_start)
    workflow["model_load_seconds"] = model_load_seconds
    workflow["model"] = model_info
    workflow["variants"] = summaries
    return frozen, workflow


def _analysis_only(
    config: Mapping[str, Any], b2_dir: Path, atomic_dir: Path, output_root: Path
):
    return load_frozen_m3_inputs(
        b2_dir=b2_dir,
        atomic_dir=atomic_dir,
        ids_50_path=_resolve_config_path(config, "ids_50"),
        ids_49_path=_resolve_config_path(config, "primary_ids_49"),
    )


def main() -> int:
    args = parse_args()
    config = load_m3_config(args.config)
    b2_dir = args.b2_dir.expanduser().resolve()
    atomic_dir = args.atomic_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.stage in {"all", "generate"}:
        frozen, workflow = _generate(
            config=config,
            dataset_root=args.dataset_root,
            b2_dir=b2_dir,
            atomic_dir=atomic_dir,
            output_root=output_root,
        )
    else:
        frozen = _analysis_only(config, b2_dir, atomic_dir, output_root)
        workflow = {"commands": [], "stage": "analyze"}
    comparison = None
    if args.stage in {"all", "analyze"}:
        comparison = run_primary_analysis(
            frozen=frozen, b2_dir=b2_dir, output_root=output_root
        )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "M3_atomic_caption_generation",
        "stage": args.stage,
        "generator_control": {
            "model": MODEL_NAME,
            "revision": MODEL_REVISION,
            "prompt": EXPECTED_PROMPT,
            "generation": EXPECTED_GENERATION,
            "seed": 2026,
            "max_context_tokens": 384,
        },
        "frozen_inputs": {
            "b2_directory": str(b2_dir),
            "b2_predictions_sha256": file_sha256(b2_dir / "predictions.jsonl"),
            "atomic_directory": str(atomic_dir),
            "atomic_manifest_sha256": file_sha256(atomic_dir / "manifest.json"),
            "ids_50_sha256": file_sha256(_resolve_config_path(config, "ids_50")),
            "primary_ids_49_sha256": file_sha256(
                _resolve_config_path(config, "primary_ids_49")
            ),
        },
        "b2_generation_reused": True,
        "b2_generation_rerun": False,
        "atomic_extraction_invoked": False,
        "spacy_fallback_used": False,
        "primary_sample_count": 49,
        "excluded_failure_sample": FAILURE_SAMPLE_ID,
        "evaluation": {
            "required_metrics": ["CIDEr", "EntityPrecision", "EntityRecall", "EntityF1"],
            "entity_extractor": "spacy/en_core_web_sm",
            "spice": "not_invoked",
        },
        "workflow": workflow,
        "comparison_completed": comparison is not None,
        "git": _git_state(PROJECT_ROOT),
        "packages": _versions(),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())