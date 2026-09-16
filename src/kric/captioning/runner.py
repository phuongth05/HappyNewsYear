"""Reproducible B0 experiment orchestration and artifact writing."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from kric.data.goodnews import GoodNewsDataset
from kric.evaluation.io import file_sha256, write_json

from .blip import BlipImageOnlyCaptioner
from .config import B0ExperimentConfig
from .types import GeneratedCaption, ImageOnlyInput


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Captioner(Protocol):
    load_seconds: float

    def load(self) -> None: ...

    def generate(self, inputs: Sequence[ImageOnlyInput]) -> list[GeneratedCaption]: ...

    def model_info(self) -> dict[str, Any]: ...


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"available": True, "commit": commit, "dirty": bool(status.strip())}
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "reason": "git_unavailable_or_not_a_repository",
            "detail": str(error),
        }


def _versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("kric", "torch", "transformers", "Pillow", "PyYAML"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


def _resolved_config(config: B0ExperimentConfig) -> dict[str, Any]:
    return {
        "experiment": config.experiment,
        "seed": config.seed,
        "dataset": {
            "name": config.dataset.name,
            "config": str(config.dataset.config_path),
            "split": config.dataset.split,
            "max_samples": config.dataset.max_samples,
            "subset_strategy": config.dataset.subset_strategy,
        },
        "model": {
            "type": config.model.type,
            "name": config.model.name,
            "revision": config.model.revision,
            "device": config.model.device,
            "dtype": config.model.dtype,
            "batch_size": config.model.batch_size,
        },
        "generation": config.generation.to_generate_kwargs(),
        "output_dir": str(config.output_dir),
        "evaluation": {
            "enabled": config.evaluation.enabled,
            "metrics": list(config.evaluation.metrics),
            "allow_metric_errors": config.evaluation.allow_metric_errors,
        },
    }


def _run_evaluation(config: B0ExperimentConfig, predictions: Path) -> dict[str, Any]:
    metrics_path = config.output_dir / "metrics.json"
    per_sample_path = config.output_dir / "metrics_per_sample.csv"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate.py"),
        "--predictions",
        str(predictions),
        "--output",
        str(metrics_path),
        "--per-sample-output",
        str(per_sample_path),
        "--metrics",
        ",".join(config.evaluation.metrics),
    ]
    if config.evaluation.allow_metric_errors:
        command.append("--allow-metric-errors")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    result = {
        "command": command,
        "return_code": completed.returncode,
        "seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "metrics_path": str(metrics_path),
        "per_sample_path": str(per_sample_path),
    }
    write_json(config.output_dir / "evaluation_runtime.json", result)
    if completed.returncode:
        raise RuntimeError(
            "evaluation failed; see evaluation_runtime.json: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return result


def run_b0(
    config: B0ExperimentConfig,
    *,
    captioner: Captioner | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate B0 captions, persist artifacts, then evaluate the saved file."""

    predictions_path = config.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing run: {predictions_path}")

    random.seed(config.seed)
    dataset = GoodNewsDataset.from_config(config.dataset.config_path)
    dataset.assert_split_integrity()
    samples = sorted(
        dataset.iter_samples(config.dataset.split), key=lambda sample: sample.sample_id
    )
    if config.dataset.max_samples is not None:
        samples = samples[: config.dataset.max_samples]
    if not samples:
        raise ValueError(f"no samples found in split {config.dataset.split!r}")
    missing_images = [sample.image_path for sample in samples if not Path(sample.image_path).is_file()]
    if missing_images:
        raise FileNotFoundError(f"selected subset has missing images: {missing_images[:10]}")

    # This is the sole model boundary. References, articles, entities, and
    # dataset metadata remain in `samples` and cannot enter `model_inputs`.
    model_inputs = tuple(
        ImageOnlyInput(sample_id=sample.sample_id, image_path=sample.image_path)
        for sample in samples
    )
    generator = captioner or BlipImageOnlyCaptioner(
        config.model, config.generation, config.seed
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = _resolved_config(config)
    write_json(config.output_dir / "config.resolved.json", resolved)
    write_json(config.output_dir / "generation_config.json", resolved["generation"])

    started = time.perf_counter()
    load_started = time.perf_counter()
    generator.load()
    model_load_seconds = time.perf_counter() - load_started
    generation_started = time.perf_counter()
    generated = generator.generate(model_inputs)
    generation_seconds = time.perf_counter() - generation_started
    by_id = {item.sample_id: item.text for item in generated}
    if len(by_id) != len(samples) or set(by_id) != {sample.sample_id for sample in samples}:
        raise RuntimeError("captioner returned missing or duplicate sample IDs")

    rows = [
        {
            "sample_id": sample.sample_id,
            "prediction": by_id[sample.sample_id],
            "reference": sample.reference_caption,
            "metadata": {
                "dataset": "goodnews",
                "official_split": sample.metadata.get("official_split"),
                "image_path": sample.image_path,
                "experiment": config.experiment,
                "model_name": config.model.name,
                "model_revision": config.model.revision,
            },
        }
        for sample in samples
    ]
    _write_jsonl(predictions_path, rows)

    model_info = generator.model_info()
    write_json(config.output_dir / "model.json", model_info)
    write_json(
        config.output_dir / "token_settings.json",
        {**resolved["generation"], "tokenizer": model_info.get("tokenizer", {})},
    )
    qualitative = [
        {
            "sample_id": row["sample_id"],
            "image_path": row["metadata"]["image_path"],
            "prediction": row["prediction"],
            "reference": row["reference"],
        }
        for row in rows[:10]
    ]
    write_json(config.output_dir / "qualitative_examples.json", qualitative)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config.experiment,
        "input_contract": ["image"],
        "article_fields_passed_to_model": False,
        "reference_passed_to_model": False,
        "processor_text_inputs_allowed": False,
        "seed": config.seed,
        "dataset": resolved["dataset"],
        "model": model_info,
        "git": _git_state(PROJECT_ROOT),
        "source_config": {
            "path": str(config.source_path) if config.source_path else None,
            "sha256": file_sha256(config.source_path) if config.source_path else None,
        },
        "predictions": {
            "path": str(predictions_path),
            "sha256": file_sha256(predictions_path),
            "samples": len(rows),
        },
    }
    write_json(config.output_dir / "manifest.json", manifest)

    runtime = {
        "status": "generated",
        "samples": len(rows),
        "model_load_seconds": model_load_seconds,
        "generation_seconds": generation_seconds,
        "seconds_per_sample": generation_seconds / len(rows),
        "samples_per_second": len(rows) / generation_seconds if generation_seconds else None,
        "total_before_evaluation_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _versions(),
    }
    write_json(config.output_dir / "runtime.json", runtime)

    evaluation = None
    if config.evaluation.enabled:
        evaluation = _run_evaluation(config, predictions_path)
        runtime["status"] = "complete"
        runtime["evaluation_seconds"] = evaluation["seconds"]
        runtime["total_seconds"] = time.perf_counter() - started
        write_json(config.output_dir / "runtime.json", runtime)
    return {
        "predictions": str(predictions_path),
        "metrics": str(config.output_dir / "metrics.json") if evaluation else None,
        "samples": len(rows),
        "qualitative_examples": qualitative,
    }
