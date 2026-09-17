"""Run the six controlled B2 sentence-selection variants against saved B1 outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.captioning.config import B1ExperimentConfig, load_experiment_config  # noqa: E402
from kric.evaluation.io import file_sha256, load_predictions, write_json  # noqa: E402
from scripts.run_goodnews_validation import (  # noqa: E402
    DEFAULT_IDS,
    assert_prediction_ids,
    create_output_bundle,
    load_validated_ids,
    validate_exact_subset,
)


VARIANTS = {
    f"{method}_k{k}": PROJECT_ROOT
    / "configs"
    / "experiments"
    / f"b2_instructblip_{method}_k{k}.yaml"
    for method in ("bm25", "semantic")
    for k in (1, 3, 5)
}
BASELINE_SOURCE = (
    PROJECT_ROOT / "configs" / "experiments" / "b1_instructblip_article_context.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--b1-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    return parser.parse_args()


def _run(command: list[str], label: str) -> dict[str, Any]:
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(command), flush=True)
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


def _validate_b1(b1_dir: Path, expected_ids: list[str]) -> dict[str, Any]:
    required = (
        "predictions.jsonl",
        "metrics.json",
        "metrics_per_sample.csv",
        "resolved_config.json",
    )
    missing = [name for name in required if not (b1_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"B1 directory is missing required artifacts: {missing}")
    records = load_predictions(b1_dir / "predictions.jsonl")
    if [record.sample_id for record in records] != expected_ids:
        raise RuntimeError("saved B1 outputs do not use the approved ordered 50 IDs")
    resolved = json.loads((b1_dir / "resolved_config.json").read_text(encoding="utf-8"))
    canonical = B1ExperimentConfig.from_file(BASELINE_SOURCE)
    expected = {
        "seed": canonical.seed,
        "model": {
            "type": canonical.model.type,
            "name": canonical.model.name,
            "revision": canonical.model.revision,
            "device": canonical.model.device,
            "dtype": canonical.model.dtype,
            "batch_size": canonical.model.batch_size,
        },
        "generation": canonical.generation.to_generate_kwargs(),
        "prompt": {
            "instruction": canonical.prompt.instruction,
            "context_prefix": canonical.prompt.context_prefix,
            "context_suffix": canonical.prompt.context_suffix,
        },
        "context": {
            "max_context_tokens": canonical.context.max_context_tokens,
            "truncation": canonical.context.truncation,
        },
    }
    for key, value in expected.items():
        if resolved.get(key) != value:
            raise RuntimeError(f"saved B1 resolved config mismatch: {key}")
    dataset = resolved.get("dataset", {})
    if (
        dataset.get("split") != "dev"
        or dataset.get("max_samples") != 50
        or dataset.get("subset_strategy") != "first_by_sample_id"
    ):
        raise RuntimeError("saved B1 dataset selection is not the approved validation subset")
    return {
        "directory": str(b1_dir),
        "samples": len(records),
        "artifacts": {
            name: {"path": str(b1_dir / name), "sha256": file_sha256(b1_dir / name)}
            for name in required
        },
    }


def _runtime_configs(
    dataset_root: Path, b1_dir: Path, output_root: Path
) -> dict[str, Path]:
    config_dir = output_root / "workflow_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = config_dir / "goodnews.runtime.yaml"
    dataset_config.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "goodnews",
                    "root": str(dataset_root),
                    "annotations_path": "article+caption.json",
                    "splits_path": "img_splits.json",
                    "images_root": "images",
                    "image_extension": ".jpg",
                    "entities_source": "unknown",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    baseline_raw = yaml.safe_load(BASELINE_SOURCE.read_text(encoding="utf-8"))
    baseline_raw["dataset"]["config"] = str(dataset_config)
    baseline_raw["output_dir"] = str(b1_dir)
    baseline_config = config_dir / "b1.runtime.yaml"
    baseline_config.write_text(yaml.safe_dump(baseline_raw, sort_keys=False), encoding="utf-8")

    runtime_paths = {}
    for name, source in VARIANTS.items():
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        method = raw["retrieval"]["method"]
        raw["dataset"]["config"] = str(dataset_config)
        raw["retrieval"]["rankings_path"] = str(
            output_root / "retrieval" / f"{method}.jsonl"
        )
        raw["output_dir"] = str(output_root / name)
        raw["comparison"]["baseline_config"] = str(baseline_config)
        raw["comparison"]["baseline_per_sample"] = str(
            b1_dir / "metrics_per_sample.csv"
        )
        runtime_path = config_dir / f"{name}.runtime.yaml"
        runtime_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        load_experiment_config(runtime_path)
        runtime_paths[name] = runtime_path
    return runtime_paths


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    b1_dir = args.b1_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    bundle = args.bundle.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root must be absent or empty: {output_root}")
    if bundle.exists():
        raise FileExistsError(f"bundle already exists: {bundle}")
    output_root.mkdir(parents=True, exist_ok=True)
    expected_ids = load_validated_ids(DEFAULT_IDS)
    subset = validate_exact_subset(dataset_root, expected_ids)
    write_json(output_root / "subset_preflight.json", subset)
    b1_snapshot = _validate_b1(b1_dir, expected_ids)
    write_json(output_root / "b1_baseline_snapshot.json", b1_snapshot)
    runtime_paths = _runtime_configs(dataset_root, b1_dir, output_root)
    workflow: dict[str, Any] = {
        "status": "running",
        "dataset_root": str(dataset_root),
        "b1_dir": str(b1_dir),
        "output_root": str(output_root),
        "commands": [],
    }
    preflight = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "gpu_preflight.py"),
        "--output",
        str(output_root / "gpu_preflight.json"),
    ]
    workflow["commands"].append(_run(preflight, "GPU preflight"))
    retrieval_dir = output_root / "retrieval"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    for method in ("bm25", "semantic"):
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "rank_goodnews_sentences.py"),
            "--dataset-root",
            str(dataset_root),
            "--ids",
            str(DEFAULT_IDS),
            "--method",
            method,
            "--output",
            str(retrieval_dir / f"{method}.jsonl"),
        ]
        if method == "semantic":
            command.extend(
                ["--cache", str(retrieval_dir / "clip_sentence_embeddings.sqlite3")]
            )
        workflow["commands"].append(_run(command, f"{method} sentence ranking"))

    for name in (
        "bm25_k1",
        "bm25_k3",
        "bm25_k5",
        "semantic_k1",
        "semantic_k3",
        "semantic_k5",
    ):
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_captioning.py"),
            "--config",
            str(runtime_paths[name]),
        ]
        workflow["commands"].append(_run(command, name))
        assert_prediction_ids(output_root / name / "predictions.jsonl", expected_ids)

    analysis_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "analyze_b2_validation.py"),
        "--b1-dir",
        str(b1_dir),
    ]
    for name in VARIANTS:
        analysis_command.extend(["--variant", f"{name}={output_root / name}"])
    analysis_command.extend(
        [
            "--output",
            str(output_root / "b2_analysis.json"),
            "--qualitative-output",
            str(output_root / "qualitative_examples.json"),
            "--review-markdown",
            str(output_root / "qualitative_review.md"),
            "--resamples",
            "10000",
            "--seed",
            "2026",
        ]
    )
    workflow["commands"].append(_run(analysis_command, "B2 analysis"))
    workflow.update(
        {
            "status": "complete",
            "all_variants_share_validated_ids": True,
            "variants": list(VARIANTS),
        }
    )
    write_json(output_root / "workflow_runtime.json", workflow)
    bundle_result = create_output_bundle(output_root, bundle)
    print(json.dumps({"workflow": workflow, "bundle": bundle_result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
