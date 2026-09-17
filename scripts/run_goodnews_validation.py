"""Run the controlled 50-sample GoodNews validation workflow on a CUDA host."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.captioning.config import (  # noqa: E402
    B0ExperimentConfig,
    B1ExperimentConfig,
    load_experiment_config,
)
from kric.data.goodnews import GoodNewsConfig, GoodNewsDataset  # noqa: E402
from kric.data.selection import select_samples  # noqa: E402
from kric.evaluation.io import file_sha256, load_predictions, write_json  # noqa: E402


MODEL_NAME = "Salesforce/instructblip-flan-t5-xl"
MODEL_REVISION = "bf9bd58cb3bb06e88c03b80f9cc346b90094bc6d"
EXPECTED_SEED = 2026
EXPECTED_SAMPLES = 50
CONDITION_SOURCES = {
    "b0": PROJECT_ROOT / "configs" / "experiments" / "b0_instructblip_image_only.yaml",
    "b1": PROJECT_ROOT / "configs" / "experiments" / "b1_instructblip_article_context.yaml",
    "b1_random": PROJECT_ROOT
    / "configs"
    / "experiments"
    / "b1_instructblip_random_article.yaml",
}
DEFAULT_IDS = (
    PROJECT_ROOT / "configs" / "dataset" / "goodnews_validation_50_ids.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="GoodNews root containing article+caption.json, img_splits.json, and images/.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
    )
    return parser.parse_args()


def load_validated_ids(path: Path) -> list[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise TypeError("validated ID manifest must be a JSON list of strings")
    if len(values) != EXPECTED_SAMPLES or len(set(values)) != EXPECTED_SAMPLES:
        raise ValueError("validated ID manifest must contain exactly 50 unique IDs")
    return values


def validate_exact_subset(dataset_root: Path, expected_ids: list[str]) -> dict[str, Any]:
    """Repeat the exact subset preflight before any model is loaded."""

    config = GoodNewsConfig(
        annotations_path=dataset_root / "article+caption.json",
        splits_path=dataset_root / "img_splits.json",
        images_root=dataset_root / "images",
    )
    dataset = GoodNewsDataset(config)
    dataset.assert_split_integrity()
    samples = select_samples(
        dataset,
        split="dev",
        max_samples=EXPECTED_SAMPLES,
        strategy="first_by_sample_id",
    )
    actual_ids = [sample.sample_id for sample in samples]
    if actual_ids != expected_ids:
        raise RuntimeError(
            "selected GoodNews IDs differ from the validated 50-sample manifest"
        )

    checks = {
        "captions_non_empty": True,
        "articles_non_empty": True,
        "images_present": True,
        "images_pillow_valid": True,
    }
    for sample in samples:
        if not sample.reference_caption.strip():
            checks["captions_non_empty"] = False
            raise RuntimeError(f"empty reference caption: {sample.sample_id}")
        if not sample.article_text.strip():
            checks["articles_non_empty"] = False
            raise RuntimeError(f"empty article: {sample.sample_id}")
        image_path = Path(sample.image_path)
        if not image_path.is_file():
            checks["images_present"] = False
            raise RuntimeError(f"missing image: {sample.sample_id}: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as error:
            checks["images_pillow_valid"] = False
            raise RuntimeError(f"corrupt image: {sample.sample_id}: {error}") from error
    return {
        "valid": all(checks.values()),
        "dataset_root": str(dataset_root),
        "split": "dev",
        "selection": "first_by_sample_id",
        "samples": len(samples),
        "sample_ids": actual_ids,
        "checks": checks,
    }


def create_runtime_configs(dataset_root: Path, output_root: Path) -> dict[str, Path]:
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

    runtime_paths = {
        name: config_dir / f"{name}.runtime.yaml" for name in CONDITION_SOURCES
    }
    for name, source in CONDITION_SOURCES.items():
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        raw["dataset"]["config"] = str(dataset_config)
        raw["output_dir"] = str(output_root / name)
        if name != "b0":
            raw["comparison"]["baseline_config"] = str(runtime_paths["b0"])
            raw["comparison"]["baseline_per_sample"] = str(
                output_root / "b0" / "metrics_per_sample.csv"
            )
        runtime_paths[name].write_text(
            yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
        )
    return runtime_paths


def validate_scientific_controls(runtime_paths: dict[str, Path]) -> dict[str, Any]:
    configs = {name: load_experiment_config(path) for name, path in runtime_paths.items()}
    b0 = configs["b0"]
    if not isinstance(b0, B0ExperimentConfig):
        raise TypeError("B0 runtime config has the wrong experiment type")
    for name in ("b1", "b1_random"):
        if not isinstance(configs[name], B1ExperimentConfig):
            raise TypeError(f"{name} runtime config has the wrong experiment type")

    controlled_fields = (
        "seed",
        "dataset",
        "model",
        "generation",
        "evaluation",
        "prompt",
        "context",
    )
    for name in ("b1", "b1_random"):
        mismatches = [
            field
            for field in controlled_fields
            if getattr(configs[name], field) != getattr(b0, field)
        ]
        if mismatches:
            raise RuntimeError(f"unfair {name}/B0 config mismatch: {mismatches}")

    if (
        b0.seed != EXPECTED_SEED
        or b0.dataset.split != "dev"
        or b0.dataset.max_samples != EXPECTED_SAMPLES
        or b0.dataset.subset_strategy != "first_by_sample_id"
        or b0.model.name != MODEL_NAME
        or b0.model.revision != MODEL_REVISION
        or b0.model.device != "auto"
        or b0.model.dtype != "float16"
        or b0.model.batch_size != 1
        or b0.generation.do_sample
    ):
        raise RuntimeError("runtime configs violate the pinned validation protocol")
    return {
        "valid": True,
        "controlled_fields": list(controlled_fields),
        "only_conceptual_difference": "article_context_access",
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "dtype": "float16",
        "batch_size": 1,
        "seed": EXPECTED_SEED,
    }


def assert_prediction_ids(predictions: Path, expected_ids: list[str]) -> None:
    actual_ids = [record.sample_id for record in load_predictions(predictions)]
    if actual_ids != expected_ids:
        raise RuntimeError(f"prediction sample IDs differ from validated manifest: {predictions}")


def run_command(command: list[str], label: str) -> dict[str, Any]:
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
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}; settings were not changed or retried"
        )
    return result


def create_output_bundle(output_root: Path, bundle: Path) -> dict[str, Any]:
    if bundle.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {bundle}")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle.with_suffix(bundle.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    included: list[str] = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                relative = Path(output_root.name) / path.relative_to(output_root)
                archive.write(path, relative.as_posix())
                included.append(relative.as_posix())
    temporary.replace(bundle)
    return {
        "path": str(bundle),
        "sha256": file_sha256(bundle),
        "files": len(included),
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    bundle = args.bundle.expanduser().resolve()
    expected_ids = load_validated_ids(DEFAULT_IDS)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"GoodNews dataset root does not exist: {dataset_root}")
    if bundle.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {bundle}")
    if output_root == dataset_root or output_root in dataset_root.parents:
        raise ValueError("output root must not contain or equal the input dataset root")
    for condition in CONDITION_SOURCES:
        predictions = output_root / condition / "predictions.jsonl"
        if predictions.exists():
            raise FileExistsError(f"refusing to overwrite existing run: {predictions}")
    output_root.mkdir(parents=True, exist_ok=True)

    workflow: dict[str, Any] = {
        "status": "preflight",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "validated_ids_manifest": str(DEFAULT_IDS),
        "validated_ids_sha256": file_sha256(DEFAULT_IDS),
        "commands": [],
    }
    subset_report = validate_exact_subset(dataset_root, expected_ids)
    write_json(output_root / "subset_preflight.json", subset_report)
    runtime_paths = create_runtime_configs(dataset_root, output_root)
    controls = validate_scientific_controls(runtime_paths)
    write_json(output_root / "scientific_controls.json", controls)

    evaluation_preflight_command = [
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
    workflow["commands"].append(
        run_command(evaluation_preflight_command, "Evaluation dependency preflight")
    )

    preflight_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "gpu_preflight.py"),
        "--output",
        str(output_root / "gpu_preflight.json"),
    ]
    workflow["commands"].append(run_command(preflight_command, "GPU preflight"))

    for condition in ("b0", "b1", "b1_random"):
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_captioning.py"),
            "--config",
            str(runtime_paths[condition]),
        ]
        workflow["commands"].append(run_command(command, condition))
        assert_prediction_ids(output_root / condition / "predictions.jsonl", expected_ids)
        # Model-process cleanup runs in run_captioning.py; this collects only
        # orchestration objects and intentionally does not initialize CUDA here.
        gc.collect()

    analysis_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "analyze_context_sanity.py"),
        "--b0-dir",
        str(output_root / "b0"),
        "--b1-dir",
        str(output_root / "b1"),
        "--b1-random-dir",
        str(output_root / "b1_random"),
        "--output",
        str(output_root / "context_sanity.json"),
        "--dataset-root",
        str(dataset_root),
        "--qualitative-output",
        str(output_root / "qualitative_examples.json"),
        "--qualitative-markdown-output",
        str(output_root / "qualitative_examples.md"),
        "--representative-count",
        "10",
        "--resamples",
        "10000",
        "--seed",
        str(EXPECTED_SEED),
    ]
    workflow["commands"].append(run_command(analysis_command, "context sanity analysis"))

    for condition in ("b0", "b1", "b1_random"):
        assert_prediction_ids(output_root / condition / "predictions.jsonl", expected_ids)
    workflow.update(
        {
            "status": "complete",
            "all_conditions_share_validated_ids": True,
            "samples_per_condition": EXPECTED_SAMPLES,
        }
    )
    write_json(output_root / "workflow_runtime.json", workflow)
    bundle_result = create_output_bundle(output_root, bundle)
    print(json.dumps({"workflow": workflow, "bundle": bundle_result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
