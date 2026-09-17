"""Reproducible controlled-baseline orchestration and artifact writing."""

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
from kric.data.selection import select_samples
from kric.evaluation.io import file_sha256, write_json

from .blip import BlipImageOnlyCaptioner
from .blip_full_article import BlipFullArticleCaptioner
from .config import B0ExperimentConfig, B1ExperimentConfig, B2ExperimentConfig
from .instructblip import InstructBlipArticleCaptioner, InstructBlipImageOnlyCaptioner
from .types import FullArticleInput, GeneratedCaption, ImageOnlyInput


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Captioner(Protocol):
    load_seconds: float

    def load(self) -> None: ...

    def generate(
        self, inputs: Sequence[ImageOnlyInput] | Sequence[FullArticleInput]
    ) -> list[GeneratedCaption]: ...

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


def _start_gpu_memory_tracking() -> dict[str, Any]:
    """Reset CUDA peaks immediately before model loading."""

    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "reason": "torch_not_installed"}
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "total_vram_bytes": total_bytes,
        "free_vram_before_load_bytes": free_bytes,
    }


def _finish_gpu_memory_tracking(start: dict[str, Any]) -> dict[str, Any]:
    """Capture model-load and generation peaks without changing inference."""

    if not start.get("cuda_available"):
        return start
    import torch

    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        **start,
        "total_vram_bytes": total_bytes,
        "free_vram_after_generation_bytes": free_bytes,
        "current_allocated_bytes": torch.cuda.memory_allocated(),
        "current_reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _resolved_config(
    config: B0ExperimentConfig | B1ExperimentConfig | B2ExperimentConfig,
) -> dict[str, Any]:
    resolved = {
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
        "prompt": {
            "instruction": config.prompt.instruction,
            "context_prefix": config.prompt.context_prefix,
            "context_suffix": config.prompt.context_suffix,
        },
        "context": {
            "max_context_tokens": config.context.max_context_tokens,
            "truncation": config.context.truncation,
        },
        "output_dir": str(config.output_dir),
        "evaluation": {
            "enabled": config.evaluation.enabled,
            "metrics": list(config.evaluation.metrics),
            "allow_metric_errors": config.evaluation.allow_metric_errors,
            "entity_extractor": config.evaluation.entity_extractor,
            "spacy_model": config.evaluation.spacy_model,
        },
    }
    if isinstance(config, (B1ExperimentConfig, B2ExperimentConfig)):
        resolved["comparison"] = {
            "baseline_config": str(config.comparison.baseline_config),
            "baseline_per_sample": str(config.comparison.baseline_per_sample),
            "metrics": list(config.comparison.metrics),
            "confidence": config.comparison.confidence,
            "resamples": config.comparison.resamples,
            "seed": config.comparison.seed,
        }
    if isinstance(config, B2ExperimentConfig):
        resolved["retrieval"] = {
            "method": config.retrieval.method,
            "k": config.retrieval.k,
            "rankings_path": str(config.retrieval.rankings_path),
            "model_name": config.retrieval.model_name,
            "model_revision": config.retrieval.model_revision,
        }
    return resolved


def _run_evaluation(
    config: B0ExperimentConfig | B1ExperimentConfig | B2ExperimentConfig,
    predictions: Path,
) -> dict[str, Any]:
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
        "--entity-extractor",
        config.evaluation.entity_extractor,
        "--spacy-model",
        config.evaluation.spacy_model,
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


def _assert_b0_b1_comparable(config: B1ExperimentConfig) -> B0ExperimentConfig:
    baseline = B0ExperimentConfig.from_file(config.comparison.baseline_config)
    mismatches = []
    for field_name in (
        "seed",
        "dataset",
        "model",
        "generation",
        "evaluation",
        "prompt",
        "context",
    ):
        if getattr(baseline, field_name) != getattr(config, field_name):
            mismatches.append(field_name)
    if mismatches:
        raise ValueError(
            "B0/B1 causal comparison is not controlled; mismatched fields: "
            + ", ".join(mismatches)
        )
    return baseline


def _assert_b1_b2_comparable(config: B2ExperimentConfig) -> B1ExperimentConfig:
    baseline = B1ExperimentConfig.from_file(config.comparison.baseline_config)
    mismatches = []
    for field_name in (
        "seed",
        "dataset",
        "model",
        "generation",
        "evaluation",
        "prompt",
        "context",
    ):
        if getattr(baseline, field_name) != getattr(config, field_name):
            mismatches.append(field_name)
    if mismatches:
        raise ValueError(
            "B1/B2 causal comparison is not controlled; mismatched fields: "
            + ", ".join(mismatches)
        )
    return baseline


def _run_paired_comparisons(
    config: B1ExperimentConfig | B2ExperimentConfig,
) -> dict[str, Any]:
    baseline = config.comparison.baseline_per_sample
    candidate = config.output_dir / "metrics_per_sample.csv"
    output_dir = config.output_dir / "comparisons"
    if not baseline.is_file():
        result = {
            "status": "not_run",
            "reason": "baseline per-sample metrics are missing",
            "baseline": str(baseline),
            "candidate": str(candidate),
        }
        write_json(config.output_dir / "comparison_runtime.json", result)
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"status": "complete", "metrics": {}}
    for metric in config.comparison.metrics:
        output = output_dir / f"{metric}.json"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "compare_metrics.py"),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--metric",
            metric,
            "--output",
            str(output),
            "--confidence",
            str(config.comparison.confidence),
            "--resamples",
            str(config.comparison.resamples),
            "--seed",
            str(config.comparison.seed),
        ]
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
        results["metrics"][metric] = {
            "return_code": completed.returncode,
            "output": str(output),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode:
            results["status"] = "error"
    write_json(config.output_dir / "comparison_runtime.json", results)
    if results["status"] == "error":
        raise RuntimeError("one or more paired bootstrap comparisons failed")
    return results


def _assign_article_samples(samples: Sequence[Any], experiment: str, seed: int) -> list[Any]:
    if experiment in {"B1_full_article", "B1_instructblip_article_context"}:
        return list(samples)
    if experiment not in {"B1_random_article", "B1_instructblip_random_article"}:
        raise ValueError(f"unsupported article assignment for {experiment!r}")
    if len(samples) < 2:
        raise ValueError("B1-random requires at least two selected samples")
    offset = random.Random(seed).randrange(1, len(samples))
    donors = list(samples[offset:]) + list(samples[:offset])
    if any(sample.sample_id == donor.sample_id for sample, donor in zip(samples, donors)):
        raise RuntimeError("random article assignment must be a derangement")
    return donors


def _make_b0_captioner(config: B0ExperimentConfig) -> Captioner:
    if config.model.type == "instructblip":
        return InstructBlipImageOnlyCaptioner(
            config.model, config.generation, config.prompt, config.context, config.seed
        )
    return BlipImageOnlyCaptioner(config.model, config.generation, config.seed)


def _make_b1_captioner(config: B1ExperimentConfig) -> Captioner:
    if config.model.type == "instructblip":
        return InstructBlipArticleCaptioner(
            config.model, config.generation, config.prompt, config.context, config.seed
        )
    return BlipFullArticleCaptioner(
        config.model, config.generation, config.context, config.seed
    )


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
    samples = select_samples(
        dataset,
        split=config.dataset.split,
        max_samples=config.dataset.max_samples,
        strategy=config.dataset.subset_strategy,
    )
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
    generator = captioner or _make_b0_captioner(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = _resolved_config(config)
    write_json(config.output_dir / "config.resolved.json", resolved)
    write_json(config.output_dir / "resolved_config.json", resolved)
    write_json(config.output_dir / "generation_config.json", resolved["generation"])

    started = time.perf_counter()
    gpu_memory_start = _start_gpu_memory_tracking()
    load_started = time.perf_counter()
    generator.load()
    model_load_seconds = time.perf_counter() - load_started
    generation_started = time.perf_counter()
    generated = generator.generate(model_inputs)
    generation_seconds = time.perf_counter() - generation_started
    gpu_memory = _finish_gpu_memory_tracking(gpu_memory_start)
    write_json(config.output_dir / "gpu_memory.json", gpu_memory)
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
        {
            **resolved["generation"],
            "context": resolved["context"],
            "tokenizer": model_info.get("tokenizer", {}),
        },
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
        "input_contract": (
            ["image", "fixed_instruction"]
            if config.model.type == "instructblip"
            else ["image"]
        ),
        "article_fields_passed_to_model": False,
        "reference_passed_to_model": False,
        "processor_text_inputs_allowed": config.model.type == "instructblip",
        "prompt": resolved["prompt"],
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
        "gpu_memory": gpu_memory,
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


def run_b1(
    config: B1ExperimentConfig,
    *,
    captioner: Captioner | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate full-article captions with the same controlled B0 settings."""

    _assert_b0_b1_comparable(config)
    predictions_path = config.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing run: {predictions_path}")

    random.seed(config.seed)
    dataset = GoodNewsDataset.from_config(config.dataset.config_path)
    dataset.assert_split_integrity()
    samples = select_samples(
        dataset,
        split=config.dataset.split,
        max_samples=config.dataset.max_samples,
        strategy=config.dataset.subset_strategy,
    )
    if not samples:
        raise ValueError(f"no samples found in split {config.dataset.split!r}")
    missing_images = [sample.image_path for sample in samples if not Path(sample.image_path).is_file()]
    if missing_images:
        raise FileNotFoundError(f"selected subset has missing images: {missing_images[:10]}")

    # Reference captions and annotations are excluded. Context assignment is
    # determined only by selected sample IDs and the fixed seed.
    article_samples = _assign_article_samples(samples, config.experiment, config.seed)
    selected_article_ids = {
        sample.sample_id: donor.sample_id
        for sample, donor in zip(samples, article_samples, strict=True)
    }
    model_inputs = tuple(
        FullArticleInput(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            article_text=donor.article_text,
        )
        for sample, donor in zip(samples, article_samples, strict=True)
    )
    generator = captioner or _make_b1_captioner(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = _resolved_config(config)
    write_json(config.output_dir / "config.resolved.json", resolved)
    write_json(config.output_dir / "resolved_config.json", resolved)
    write_json(config.output_dir / "generation_config.json", resolved["generation"])

    started = time.perf_counter()
    gpu_memory_start = _start_gpu_memory_tracking()
    load_started = time.perf_counter()
    generator.load()
    model_load_seconds = time.perf_counter() - load_started
    generation_started = time.perf_counter()
    generated = generator.generate(model_inputs)
    generation_seconds = time.perf_counter() - generation_started
    gpu_memory = _finish_gpu_memory_tracking(gpu_memory_start)
    write_json(config.output_dir / "gpu_memory.json", gpu_memory)
    by_id = {item.sample_id: item for item in generated}
    if len(by_id) != len(samples) or set(by_id) != {sample.sample_id for sample in samples}:
        raise RuntimeError("captioner returned missing or duplicate sample IDs")
    required_stats = {
        "original_article_tokens",
        "used_article_tokens",
        "truncation_ratio",
        "max_context_tokens",
        "generation_tokens",
    }
    for item in generated:
        if set(item.context_stats) != required_stats:
            raise RuntimeError(
                f"captioner returned incomplete context stats for {item.sample_id}"
            )

    rows = []
    for sample in samples:
        item = by_id[sample.sample_id]
        rows.append(
            {
                "sample_id": sample.sample_id,
                "prediction": item.text,
                "reference": sample.reference_caption,
                "metadata": {
                    "dataset": "goodnews",
                    "official_split": sample.metadata.get("official_split"),
                    "image_path": sample.image_path,
                    "experiment": config.experiment,
                    "model_name": config.model.name,
                    "model_revision": config.model.revision,
                    "selected_article_sample_id": selected_article_ids[sample.sample_id],
                    **item.context_stats,
                },
            }
        )
    _write_jsonl(predictions_path, rows)

    original_total = sum(
        int(row["metadata"]["original_article_tokens"]) for row in rows
    )
    used_total = sum(int(row["metadata"]["used_article_tokens"]) for row in rows)
    truncated = sum(
        row["metadata"]["used_article_tokens"]
        < row["metadata"]["original_article_tokens"]
        for row in rows
    )
    context_summary = {
        "samples": len(rows),
        "max_context_tokens": config.context.max_context_tokens,
        "generation_tokens": config.generation.max_new_tokens,
        "truncation": config.context.truncation,
        "truncation_ratio_definition": "(original-used)/original",
        "original_article_tokens_total": original_total,
        "used_article_tokens_total": used_total,
        "mean_original_article_tokens": original_total / len(rows),
        "mean_used_article_tokens": used_total / len(rows),
        "truncated_samples": truncated,
        "truncated_sample_rate": truncated / len(rows),
        "corpus_truncation_ratio": (
            (original_total - used_total) / original_total if original_total else 0.0
        ),
    }
    write_json(config.output_dir / "context_statistics.json", context_summary)

    model_info = generator.model_info()
    write_json(config.output_dir / "model.json", model_info)
    write_json(
        config.output_dir / "token_settings.json",
        {
            **resolved["generation"],
            "context": resolved["context"],
            "tokenizer": model_info.get("tokenizer", {}),
        },
    )
    qualitative = [
        {
            "sample_id": row["sample_id"],
            "image_path": row["metadata"]["image_path"],
            "prediction": row["prediction"],
            "reference": row["reference"],
            "context_stats": {
                key: row["metadata"][key] for key in required_stats
            },
            "selected_article_sample_id": row["metadata"]["selected_article_sample_id"],
        }
        for row in rows[:10]
    ]
    write_json(config.output_dir / "qualitative_examples.json", qualitative)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config.experiment,
        "input_contract": ["image", "article_text"],
        "article_fields_passed_to_model": True,
        "reference_passed_to_model": False,
        "context_selection_uses_reference": False,
        "prompt": resolved["prompt"],
        "context_assignment": (
            "matching_sample"
            if config.experiment in {"B1_full_article", "B1_instructblip_article_context"}
            else "seeded_cyclic_derangement"
        ),
        "seed": config.seed,
        "dataset": resolved["dataset"],
        "context": resolved["context"],
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
        "gpu_memory": gpu_memory,
    }
    write_json(config.output_dir / "runtime.json", runtime)

    evaluation = comparison = None
    if config.evaluation.enabled:
        evaluation = _run_evaluation(config, predictions_path)
        comparison = _run_paired_comparisons(config)
        runtime["status"] = "complete"
        runtime["evaluation_seconds"] = evaluation["seconds"]
        runtime["total_seconds"] = time.perf_counter() - started
        write_json(config.output_dir / "runtime.json", runtime)
    return {
        "predictions": str(predictions_path),
        "metrics": str(config.output_dir / "metrics.json") if evaluation else None,
        "comparison": comparison,
        "context_statistics": context_summary,
        "samples": len(rows),
        "qualitative_examples": qualitative,
    }


def _load_b2_rankings(
    path: Path, samples: Sequence[Any], config: B2ExperimentConfig
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in rows:
                raise ValueError(f"invalid/duplicate ranking ID at line {line_number}")
            rows[sample_id] = row
    expected_ids = {sample.sample_id for sample in samples}
    if set(rows) != expected_ids:
        raise ValueError("ranking artifact IDs differ from the selected B2 samples")
    for sample in samples:
        row = rows[sample.sample_id]
        if row.get("candidate_sentence_count") != len(sample.article_sentences):
            raise ValueError(f"candidate count mismatch for {sample.sample_id}")
        retriever = row.get("retriever", {})
        if retriever.get("method") != config.retrieval.method:
            raise ValueError(f"retrieval method mismatch for {sample.sample_id}")
        if config.retrieval.method == "semantic" and (
            retriever.get("name") != config.retrieval.model_name
            or retriever.get("revision") != config.retrieval.model_revision
        ):
            raise ValueError(f"semantic model mismatch for {sample.sample_id}")
        ranked = row.get("ranked_sentences")
        if not isinstance(ranked, list) or len(ranked) != len(sample.article_sentences):
            raise ValueError(f"incomplete sentence ranking for {sample.sample_id}")
        sentence_ids = [int(item["sentence_id"]) for item in ranked]
        ranks = [int(item["rank"]) for item in ranked]
        if set(sentence_ids) != set(range(len(sample.article_sentences))):
            raise ValueError(f"sentence IDs mismatch for {sample.sample_id}")
        if ranks != list(range(1, len(ranked) + 1)):
            raise ValueError(f"ranking order is invalid for {sample.sample_id}")
        for item in ranked:
            sentence_id = int(item["sentence_id"])
            if item["text"] != sample.article_sentences[sentence_id]:
                raise ValueError(f"sentence text mismatch for {sample.sample_id}:{sentence_id}")
    return rows


def _select_b2_context(
    generator: InstructBlipArticleCaptioner,
    sample: Any,
    ranking: dict[str, Any],
    k: int,
) -> tuple[str, dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    skipped_for_budget: list[int] = []

    def context_text(items: Sequence[dict[str, Any]]) -> str:
        return "\n".join(
            item["text"] for item in sorted(items, key=lambda value: int(value["sentence_id"]))
        )

    for candidate in ranking["ranked_sentences"]:
        if len(selected) == k:
            break
        trial = [*selected, candidate]
        _, trial_stats = generator._build_article_prompt(context_text(trial))
        if trial_stats["used_article_tokens"] == trial_stats["original_article_tokens"]:
            selected.append(candidate)
        else:
            skipped_for_budget.append(int(candidate["sentence_id"]))
    fallback_truncated = False
    if not selected:
        selected = [ranking["ranked_sentences"][0]]
        fallback_truncated = True
    selected_in_article_order = sorted(
        selected, key=lambda item: int(item["sentence_id"])
    )
    context = context_text(selected_in_article_order)
    _, context_stats = generator._build_article_prompt(context)
    tokenizer = generator.processor.tokenizer
    article_token_count = len(generator._token_ids(tokenizer, sample.article_text))
    used_tokens = int(context_stats["used_article_tokens"])
    selection = {
        "candidate_sentence_count": len(sample.article_sentences),
        "selected_sentence_count": len(selected_in_article_order),
        "selected_sentence_ids": [
            int(item["sentence_id"]) for item in selected_in_article_order
        ],
        "selected_ranking_scores": [
            float(item["score"]) for item in selected_in_article_order
        ],
        "selected_ranks": [int(item["rank"]) for item in selected_in_article_order],
        "selected_sentence_texts": [item["text"] for item in selected_in_article_order],
        "skipped_sentence_ids_for_budget": skipped_for_budget,
        "fallback_sentence_truncated": fallback_truncated,
        "article_token_count": article_token_count,
        "selected_context_tokens_before_model_limit": int(
            context_stats["original_article_tokens"]
        ),
        "context_token_count": used_tokens,
        "fraction_of_article_represented": (
            used_tokens / article_token_count if article_token_count else 0.0
        ),
        "fraction_of_sentences_represented": (
            len(selected_in_article_order) / len(sample.article_sentences)
        ),
        "context_truncation_ratio": float(context_stats["truncation_ratio"]),
        "max_context_tokens": int(context_stats["max_context_tokens"]),
        "generation_tokens": int(context_stats["generation_tokens"]),
    }
    return context, selection


def run_b2(
    config: B2ExperimentConfig,
    *,
    captioner: Captioner | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate captions from reference-free top-k sentence context."""

    _assert_b1_b2_comparable(config)
    predictions_path = config.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing run: {predictions_path}")
    random.seed(config.seed)
    dataset = GoodNewsDataset.from_config(config.dataset.config_path)
    dataset.assert_split_integrity()
    samples = select_samples(
        dataset,
        split=config.dataset.split,
        max_samples=config.dataset.max_samples,
        strategy=config.dataset.subset_strategy,
    )
    if len(samples) != 50:
        raise ValueError(f"B2 requires exactly 50 selected samples, got {len(samples)}")
    rankings = _load_b2_rankings(config.retrieval.rankings_path, samples, config)
    generator = captioner or _make_b1_captioner(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = _resolved_config(config)
    write_json(config.output_dir / "config.resolved.json", resolved)
    write_json(config.output_dir / "resolved_config.json", resolved)
    write_json(config.output_dir / "generation_config.json", resolved["generation"])

    started = time.perf_counter()
    gpu_memory_start = _start_gpu_memory_tracking()
    load_started = time.perf_counter()
    generator.load()
    model_load_seconds = time.perf_counter() - load_started
    if not isinstance(generator, InstructBlipArticleCaptioner):
        raise TypeError("B2 sentence-budget selection requires InstructBlipArticleCaptioner")
    selection_started = time.perf_counter()
    contexts: dict[str, str] = {}
    selections: dict[str, dict[str, Any]] = {}
    for sample in samples:
        context, selection = _select_b2_context(
            generator, sample, rankings[sample.sample_id], config.retrieval.k
        )
        contexts[sample.sample_id] = context
        selections[sample.sample_id] = selection
    selection_seconds = time.perf_counter() - selection_started
    model_inputs = tuple(
        FullArticleInput(sample.sample_id, sample.image_path, contexts[sample.sample_id])
        for sample in samples
    )
    generation_started = time.perf_counter()
    generated = generator.generate(model_inputs)
    generation_seconds = time.perf_counter() - generation_started
    gpu_memory = _finish_gpu_memory_tracking(gpu_memory_start)
    write_json(config.output_dir / "gpu_memory.json", gpu_memory)
    generated_by_id = {item.sample_id: item for item in generated}
    if set(generated_by_id) != {sample.sample_id for sample in samples}:
        raise RuntimeError("B2 captioner returned missing or unexpected sample IDs")

    rows = []
    evidence_rows = []
    for sample in samples:
        item = generated_by_id[sample.sample_id]
        selection = selections[sample.sample_id]
        if int(item.context_stats["used_article_tokens"]) != selection["context_token_count"]:
            raise RuntimeError(f"B2 token accounting drift for {sample.sample_id}")
        metadata = {
            "dataset": "goodnews",
            "official_split": sample.metadata.get("official_split"),
            "image_path": sample.image_path,
            "experiment": config.experiment,
            "model_name": config.model.name,
            "model_revision": config.model.revision,
            "retrieval_method": config.retrieval.method,
            "retrieval_k": config.retrieval.k,
            **selection,
        }
        rows.append(
            {
                "sample_id": sample.sample_id,
                "prediction": item.text,
                "reference": sample.reference_caption,
                "metadata": metadata,
            }
        )
        evidence_rows.append(
            {
                "sample_id": sample.sample_id,
                "retrieval_method": config.retrieval.method,
                "retrieval_k": config.retrieval.k,
                **selection,
            }
        )
    _write_jsonl(predictions_path, rows)
    _write_jsonl(config.output_dir / "selected_evidence.jsonl", evidence_rows)

    context_tokens = [selection["context_token_count"] for selection in selections.values()]
    selected_counts = [selection["selected_sentence_count"] for selection in selections.values()]
    selected_scores = [
        score
        for selection in selections.values()
        for score in selection["selected_ranking_scores"]
    ]
    context_summary = {
        "samples": len(samples),
        "retrieval_method": config.retrieval.method,
        "k": config.retrieval.k,
        "max_context_tokens": config.context.max_context_tokens,
        "mean_selected_sentence_count": sum(selected_counts) / len(selected_counts),
        "mean_context_tokens": sum(context_tokens) / len(context_tokens),
        "mean_selected_retrieval_score": (
            sum(selected_scores) / len(selected_scores) if selected_scores else None
        ),
        "samples_with_budget_skips": sum(
            bool(selection["skipped_sentence_ids_for_budget"])
            for selection in selections.values()
        ),
        "samples_with_truncated_fallback": sum(
            bool(selection["fallback_sentence_truncated"])
            for selection in selections.values()
        ),
    }
    write_json(config.output_dir / "context_statistics.json", context_summary)
    model_info = generator.model_info()
    write_json(config.output_dir / "model.json", model_info)
    write_json(
        config.output_dir / "token_settings.json",
        {**resolved["generation"], "context": resolved["context"], "tokenizer": model_info.get("tokenizer", {})},
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config.experiment,
        "input_contract": ["image", "selected_article_sentences", "fixed_instruction"],
        "reference_passed_to_retriever": False,
        "generated_caption_passed_to_retriever": False,
        "reference_passed_to_model": False,
        "selected_context_preserves_article_order": True,
        "seed": config.seed,
        "dataset": resolved["dataset"],
        "retrieval": resolved["retrieval"],
        "rankings_sha256": file_sha256(config.retrieval.rankings_path),
        "model": model_info,
        "git": _git_state(PROJECT_ROOT),
        "predictions": {"path": str(predictions_path), "sha256": file_sha256(predictions_path), "samples": len(rows)},
    }
    write_json(config.output_dir / "manifest.json", manifest)
    runtime = {
        "status": "generated",
        "samples": len(rows),
        "model_load_seconds": model_load_seconds,
        "sentence_selection_seconds": selection_seconds,
        "generation_seconds": generation_seconds,
        "samples_per_second": len(rows) / generation_seconds if generation_seconds else None,
        "total_before_evaluation_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _versions(),
        "gpu_memory": gpu_memory,
    }
    write_json(config.output_dir / "runtime.json", runtime)
    evaluation = comparison = None
    if config.evaluation.enabled:
        evaluation = _run_evaluation(config, predictions_path)
        comparison = _run_paired_comparisons(config)
        runtime.update(
            {
                "status": "complete",
                "evaluation_seconds": evaluation["seconds"],
                "total_seconds": time.perf_counter() - started,
            }
        )
        write_json(config.output_dir / "runtime.json", runtime)
    return {
        "predictions": str(predictions_path),
        "metrics": str(config.output_dir / "metrics.json") if evaluation else None,
        "comparison": comparison,
        "context_statistics": context_summary,
        "samples": len(rows),
    }
