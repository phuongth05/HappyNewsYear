"""Frozen M3 atomic-context generation and paired analysis."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from kric.captioning.b2_regenerated import validate_m3_b2_provenance
from kric.evaluation.bootstrap import paired_bootstrap_ci
from kric.evaluation.io import file_sha256, load_predictions, write_json

from .config import (
    FullArticleContextConfig,
    GenerationConfig,
    ModelConfig,
    PromptConfig,
)
from .types import FullArticleInput


MODEL_NAME = "Salesforce/instructblip-flan-t5-xl"
MODEL_REVISION = "bf9bd58cb3bb06e88c03b80f9cc346b90094bc6d"
RETRIEVAL_MODEL = "openai/clip-vit-base-patch32"
RETRIEVAL_REVISION = "b97b0100e55e367c057773c2a614676470b0d575"
FAILURE_SAMPLE_ID = "4fd292608eb7c8105d86b39a_0"
EXPECTED_GENERATION = {
    "do_sample": False,
    "num_beams": 3,
    "max_new_tokens": 30,
    "min_new_tokens": 1,
    "no_repeat_ngram_size": 2,
    "length_penalty": 1.0,
    "early_stopping": True,
}
EXPECTED_PROMPT = {
    "instruction": "Describe the image in one factual news-style sentence.",
    "context_prefix": "Article context:\n",
    "context_suffix": "\n\n",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_ids(path: Path, expected: int) -> list[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) != expected or len(set(values)) != expected:
        raise ValueError(f"{path} must contain {expected} unique ordered IDs")
    return [str(value) for value in values]


def validate_primary_ids(ids_50: Sequence[str], ids_49: Sequence[str]) -> None:
    if len(ids_50) != 50 or len(set(ids_50)) != 50:
        raise ValueError("M3 requires the frozen 50 unique IDs")
    expected = [sample_id for sample_id in ids_50 if sample_id != FAILURE_SAMPLE_ID]
    if list(ids_49) != expected or len(ids_49) != 49:
        raise ValueError("primary M3 IDs must be the ordered frozen 50 minus the failure sample")


def validate_generator_control(resolved: Mapping[str, Any]) -> None:
    expected = {
        "seed": 2026,
        "model.type": "instructblip",
        "model.name": MODEL_NAME,
        "model.revision": MODEL_REVISION,
        "model.dtype": "float16",
        "model.batch_size": 1,
        "prompt": EXPECTED_PROMPT,
        "generation": EXPECTED_GENERATION,
        "context.max_context_tokens": 384,
        "context.truncation": "head",
        "retrieval.method": "semantic",
        "retrieval.k": 3,
        "retrieval.model_name": RETRIEVAL_MODEL,
        "retrieval.model_revision": RETRIEVAL_REVISION,
        "dataset.name": "goodnews",
        "dataset.split": "dev",
        "dataset.max_samples": 50,
        "dataset.subset_strategy": "first_by_sample_id",
        "evaluation.metrics": ["cider", "entity"],
        "evaluation.entity_extractor": "spacy",
        "evaluation.spacy_model": "en_core_web_sm",
        "evaluation.allow_metric_errors": False,
    }
    actual = {
        "seed": resolved.get("seed"),
        "model.type": resolved.get("model", {}).get("type"),
        "model.name": resolved.get("model", {}).get("name"),
        "model.revision": resolved.get("model", {}).get("revision"),
        "model.dtype": resolved.get("model", {}).get("dtype"),
        "model.batch_size": resolved.get("model", {}).get("batch_size"),
        "prompt": resolved.get("prompt"),
        "generation": resolved.get("generation"),
        "context.max_context_tokens": resolved.get("context", {}).get("max_context_tokens"),
        "context.truncation": resolved.get("context", {}).get("truncation"),
        "retrieval.method": resolved.get("retrieval", {}).get("method"),
        "retrieval.k": resolved.get("retrieval", {}).get("k"),
        "retrieval.model_name": resolved.get("retrieval", {}).get("model_name"),
        "retrieval.model_revision": resolved.get("retrieval", {}).get("model_revision"),
        "dataset.name": resolved.get("dataset", {}).get("name"),
        "dataset.split": resolved.get("dataset", {}).get("split"),
        "dataset.max_samples": resolved.get("dataset", {}).get("max_samples"),
        "dataset.subset_strategy": resolved.get("dataset", {}).get(
            "subset_strategy"
        ),
        "evaluation.metrics": resolved.get("evaluation", {}).get(
            "metrics"
        ),
        "evaluation.entity_extractor": resolved.get("evaluation", {}).get(
            "entity_extractor"
        ),
        "evaluation.spacy_model": resolved.get("evaluation", {}).get(
            "spacy_model"
        ),
        "evaluation.allow_metric_errors": resolved.get("evaluation", {}).get(
            "allow_metric_errors"
        ),
    }
    mismatches = {key: (expected[key], actual[key]) for key in expected if actual[key] != expected[key]}
    if mismatches:
        raise ValueError(f"M3 generator differs from frozen B2: {mismatches}")


def validate_atomic_audit(audit: Mapping[str, Any]) -> None:
    """Reject any extraction state other than the frozen 149/150 no-fallback run."""

    if (
        int(audit.get("source_sentences_requested", 0)) != 150
        or int(audit.get("source_sentences_completed", 0)) != 149
        or int(audit.get("source_sentences_failed", 0)) != 1
        or audit.get("samples_affected_by_extraction_failures") != [FAILURE_SAMPLE_ID]
        or audit.get("spacy_fallback_used") is not False
    ):
        raise ValueError("atomic audit does not match the frozen 149/150 extraction")


def validate_context_budgets(
    full_rows: Sequence[Mapping[str, Any]],
    matched_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Validate recorded M3 budgets without recomputing or extracting evidence."""

    budgets = {str(row["sample_id"]): int(row["context_token_count"]) for row in selected_rows}
    if len(budgets) != len(selected_rows):
        raise ValueError("duplicate B2 source IDs")
    if len(full_rows) != len(matched_rows) or len(full_rows) != len(selected_rows):
        raise ValueError("M3 context variants are not aligned")
    for full_row, matched_row in zip(full_rows, matched_rows, strict=True):
        sample_id = str(full_row["sample_id"])
        if sample_id != str(matched_row["sample_id"]) or sample_id not in budgets:
            raise ValueError("atomic context variants are not paired")
        if int(full_row["context_tokens"]) > 384:
            raise ValueError(f"Atomic Full exceeds 384 tokens for {sample_id}")
        b2_budget = budgets[sample_id]
        if (
            int(matched_row["b2_context_token_budget"]) != b2_budget
            or int(matched_row["context_tokens"]) > b2_budget
        ):
            raise ValueError(f"Atomic Token-Matched exceeds B2 budget for {sample_id}")


@dataclass(frozen=True)
class FrozenM3Inputs:
    ids_50: tuple[str, ...]
    ids_49: tuple[str, ...]
    b2_predictions: tuple[dict[str, Any], ...]
    selected_evidence: tuple[dict[str, Any], ...]
    atomic_evidence: tuple[dict[str, Any], ...]
    full_contexts: tuple[dict[str, Any], ...]
    matched_contexts: tuple[dict[str, Any], ...]
    b2_resolved: dict[str, Any]
    b2_provenance_status: str


def load_frozen_m3_inputs(
    *, b2_dir: Path, atomic_dir: Path, ids_50_path: Path, ids_49_path: Path
) -> FrozenM3Inputs:
    ids_50 = _load_ids(ids_50_path, 50)
    ids_49 = _load_ids(ids_49_path, 49)
    validate_primary_ids(ids_50, ids_49)
    required_b2 = (
        "manifest.json",
        "resolved_config.json",
        "selected_evidence.jsonl",
        "predictions.jsonl",
        "metrics.json",
        "metrics_per_sample.csv",
    )
    required_atomic = (
        "manifest.json",
        "audit.json",
        "atomic_evidence.jsonl",
        "atomic_contexts_full.jsonl",
        "atomic_contexts_token_matched.jsonl",
        "failures.jsonl",
    )
    for directory, names in ((b2_dir, required_b2), (atomic_dir, required_atomic)):
        missing = [name for name in names if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing frozen artifacts in {directory}: {missing}")
    b2_manifest = json.loads((b2_dir / "manifest.json").read_text(encoding="utf-8"))
    b2_resolved = json.loads((b2_dir / "resolved_config.json").read_text(encoding="utf-8"))
    atomic_manifest = json.loads((atomic_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_generator_control(b2_resolved)
    if b2_manifest.get("experiment") != "B2_instructblip_sentence_context":
        raise ValueError("unexpected frozen B2 experiment")
    prediction_path = b2_dir / "predictions.jsonl"

    model = b2_manifest.get("model", {})
    if (
        model.get("name") != MODEL_NAME
        or model.get("requested_revision") != MODEL_REVISION
        or model.get("resolved_revision") != MODEL_REVISION
        or model.get("prompt") != EXPECTED_PROMPT
    ):
        raise ValueError("frozen B2 model/prompt provenance mismatch")
    selected = _jsonl(b2_dir / "selected_evidence.jsonl")
    b2_predictions = [
        {
            "sample_id": record.sample_id,
            "prediction": record.prediction,
            "reference": record.reference,
            "metadata": record.metadata,
        }
        for record in load_predictions(prediction_path)
    ]
    for name, rows in (("B2 predictions", b2_predictions), ("selected evidence", selected)):
        if [str(row.get("sample_id")) for row in rows] != ids_50:
            raise ValueError(f"{name} IDs/order differ from frozen 50")
    b2_provenance_status = validate_m3_b2_provenance(
        b2_dir=b2_dir,
        atomic_manifest=atomic_manifest,
        resolved=b2_resolved,
        expected_ids=ids_50,
    )
    for prediction, evidence in zip(b2_predictions, selected, strict=True):
        metadata = prediction["metadata"]
        for field in (
            "retrieval_method",
            "retrieval_k",
            "selected_sentence_ids",
            "selected_sentence_texts",
            "selected_ranks",
            "selected_ranking_scores",
            "context_token_count",
        ):
            if metadata.get(field) != evidence.get(field):
                raise ValueError(f"B2 prediction/evidence mismatch for {prediction['sample_id']}:{field}")

    audit = json.loads((atomic_dir / "audit.json").read_text(encoding="utf-8"))
    if atomic_manifest.get("caption_generation_run") is not False:
        raise ValueError("atomic manifest does not represent extraction-only frozen evidence")
    source = atomic_manifest.get("source", {}).get("selected_evidence", {})
    if source.get("sha256") != file_sha256(b2_dir / "selected_evidence.jsonl"):
        raise ValueError("atomic evidence was not derived from the frozen B2 selection")
    if atomic_manifest.get("ordered_sample_ids") != ids_50:
        raise ValueError("atomic manifest sample IDs/order differ from frozen 50")
    outputs = atomic_manifest.get("outputs", {})
    for name in required_atomic[1:-1]:
        expected_hash = outputs.get(name, {}).get("sha256")
        if expected_hash != file_sha256(atomic_dir / name):
            raise ValueError(f"frozen atomic output hash mismatch: {name}")
    failure_hash = outputs.get("failures.jsonl", {}).get("sha256")
    if failure_hash != file_sha256(atomic_dir / "failures.jsonl"):
        raise ValueError("frozen atomic output hash mismatch: failures.jsonl")
    validate_atomic_audit(audit)
    atomic_evidence = _jsonl(atomic_dir / "atomic_evidence.jsonl")
    full = _jsonl(atomic_dir / "atomic_contexts_full.jsonl")
    matched = _jsonl(atomic_dir / "atomic_contexts_token_matched.jsonl")
    failures = _jsonl(atomic_dir / "failures.jsonl")
    if len(failures) != 1 or str(failures[0].get("sample_id")) != FAILURE_SAMPLE_ID:
        raise ValueError("frozen failure artifact does not contain the one expected failure")
    for name, rows in (("atomic evidence", atomic_evidence), ("full contexts", full), ("matched contexts", matched)):
        if [str(row.get("sample_id")) for row in rows] != ids_50:
            raise ValueError(f"{name} IDs/order differ from frozen 50")
    b2_by_id = {str(row["sample_id"]): row for row in selected}
    for evidence_row, full_row, matched_row in zip(
        atomic_evidence, full, matched, strict=True
    ):
        sample_id = str(full_row["sample_id"])
        if sample_id != str(matched_row["sample_id"]):
            raise ValueError("atomic context variants are not paired")
        selected_row = b2_by_id[sample_id]
        for field in (
            "selected_sentence_ids",
            "selected_sentence_texts",
            "selected_ranks",
            "selected_ranking_scores",
        ):
            if evidence_row.get(field) != selected_row.get(field):
                raise ValueError(f"atomic/B2 source provenance mismatch for {sample_id}:{field}")
        expected_complete = sample_id != FAILURE_SAMPLE_ID
        if bool(full_row.get("all_source_sentences_completed")) != expected_complete or bool(
            matched_row.get("all_source_sentences_completed")
        ) != expected_complete:
            raise ValueError(f"atomic completion status mismatch for {sample_id}")

    validate_context_budgets(full, matched, selected)
    return FrozenM3Inputs(
        tuple(ids_50), tuple(ids_49), tuple(b2_predictions), tuple(selected),
        tuple(atomic_evidence), tuple(full), tuple(matched), b2_resolved,
        b2_provenance_status,
    )


def controlled_captioner_settings(resolved: Mapping[str, Any]):
    """Build both M3 variants from the exact frozen B2 generator settings."""

    validate_generator_control(resolved)
    model = ModelConfig(**dict(resolved["model"]))
    generation = GenerationConfig(**dict(resolved["generation"]))
    prompt = PromptConfig(**dict(resolved["prompt"]))
    context = FullArticleContextConfig(
        max_context_tokens=int(resolved["context"]["max_context_tokens"]),
        truncation=str(resolved["context"]["truncation"]),
    )
    return model, generation, prompt, context, int(resolved["seed"])


def build_atomic_model_inputs(
    samples: Sequence[Any], contexts: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]
) -> tuple[FullArticleInput, ...]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    if list(sample_by_id) != list(expected_ids):
        raise ValueError("dataset sample IDs/order differ from frozen M3 IDs")
    if [str(row.get("sample_id")) for row in contexts] != list(expected_ids):
        raise ValueError("atomic context IDs/order differ from frozen M3 IDs")
    return tuple(
        FullArticleInput(sample_id, sample_by_id[sample_id].image_path, str(row["context"]))
        for sample_id, row in zip(expected_ids, contexts, strict=True)
    )


def generate_atomic_variant(
    *,
    variant: str,
    samples: Sequence[Any],
    contexts: Sequence[Mapping[str, Any]],
    frozen: FrozenM3Inputs,
    captioner: Any,
    output_dir: Path,
) -> list[dict[str, Any]]:
    if variant not in {"m3_atomic_full", "m3_atomic_token_matched"}:
        raise ValueError(f"unsupported M3 variant: {variant}")
    inputs = build_atomic_model_inputs(samples, contexts, frozen.ids_50)
    generated = captioner.generate(inputs)
    by_id = {item.sample_id: item for item in generated}
    if list(by_id) != list(frozen.ids_50):
        raise RuntimeError("captioner returned missing, duplicate, or reordered M3 IDs")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    rows = []
    for context_row in contexts:
        sample_id = str(context_row["sample_id"])
        item = by_id[sample_id]
        used = int(item.context_stats["used_article_tokens"])
        recorded = int(context_row["context_tokens"])
        if used != recorded or used > 384:
            raise RuntimeError(f"generation token accounting drift for {sample_id}")
        if variant == "m3_atomic_token_matched" and used > int(
            context_row["b2_context_token_budget"]
        ):
            raise RuntimeError(f"token-matched context exceeds B2 budget for {sample_id}")
        sample = sample_by_id[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "prediction": item.text,
                "reference": sample.reference_caption,
                "metadata": {
                    "dataset": "goodnews",
                    "official_split": sample.metadata.get("official_split"),
                    "image_path": sample.image_path,
                    "experiment": variant,
                    "model_name": MODEL_NAME,
                    "model_revision": MODEL_REVISION,
                    "context_tokens": used,
                    "context_kind": context_row["context_kind"],
                    "included_evidence_ids": context_row["included_evidence_ids"],
                    "dropped_evidence_ids": context_row["dropped_evidence_ids"],
                    "all_source_sentences_completed": context_row[
                        "all_source_sentences_completed"
                    ],
                },
            }
        )
    _write_jsonl(output_dir / "predictions.jsonl", rows)
    return rows


def _read_metrics(path: Path, ids: Sequence[str]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise ValueError(f"duplicate metric ID: {sample_id}")
            required = ("CIDEr", "EntityPrecision", "EntityRecall", "EntityF1")
            rows[sample_id] = {key: float(row[key]) for key in required}
            if row.get("EntityPredicted") not in (None, ""):
                rows[sample_id]["EntityPredicted"] = float(row["EntityPredicted"])
    if set(rows) != set(ids):
        raise ValueError("per-sample metric IDs differ from the frozen 50")
    return rows


def _comparison(
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
    ids: Sequence[str],
    metric: str,
) -> dict[str, Any]:
    left = {sample_id: baseline[sample_id][metric] for sample_id in ids}
    right = {sample_id: candidate[sample_id][metric] for sample_id in ids}
    result = paired_bootstrap_ci(left, right, resamples=10_000, seed=2026)
    deltas = {sample_id: right[sample_id] - left[sample_id] for sample_id in ids}
    result["wins"] = sum(value > 0 for value in deltas.values())
    result["ties"] = sum(value == 0 for value in deltas.values())
    result["losses"] = sum(value < 0 for value in deltas.values())
    return result


def run_primary_analysis(
    *, frozen: FrozenM3Inputs, b2_dir: Path, output_root: Path
) -> dict[str, Any]:
    methods = {
        "b2_semantic_k3": b2_dir,
        "atomic_full": output_root / "atomic_full",
        "atomic_token_matched": output_root / "atomic_token_matched",
    }
    metrics = {
        name: _read_metrics(path / "metrics_per_sample.csv", frozen.ids_50)
        for name, path in methods.items()
    }
    predictions = {
        name: {record.sample_id: record.prediction for record in load_predictions(path / "predictions.jsonl")}
        for name, path in methods.items()
    }
    for name, values in predictions.items():
        if set(values) != set(frozen.ids_50):
            raise ValueError(f"{name} prediction IDs differ from frozen 50")
    pairs = {
        "atomic_token_matched_vs_b2": ("b2_semantic_k3", "atomic_token_matched"),
        "atomic_full_vs_b2": ("b2_semantic_k3", "atomic_full"),
        "atomic_full_vs_atomic_token_matched": ("atomic_token_matched", "atomic_full"),
    }
    bootstrap = {}
    comparison_summary = {}
    for label, (baseline_name, candidate_name) in pairs.items():
        bootstrap[label] = {
            metric: _comparison(
                metrics[baseline_name], metrics[candidate_name], frozen.ids_49, metric
            )
            for metric in ("CIDEr", "EntityF1")
        }
        comparison_summary[label] = {
            "baseline": baseline_name,
            "candidate": candidate_name,
            "prediction_change_rate": sum(
                predictions[baseline_name][sample_id]
                != predictions[candidate_name][sample_id]
                for sample_id in frozen.ids_49
            )
            / len(frozen.ids_49),
            "metrics": bootstrap[label],
        }
    method_means = {
        name: {
            metric: statistics.fmean(
                values[sample_id][metric] for sample_id in frozen.ids_49
            )
            for metric in ("CIDEr", "EntityPrecision", "EntityRecall", "EntityF1")
        }
        for name, values in metrics.items()
    }
    primary_dir = output_root / "primary_49"
    primary_dir.mkdir(parents=True, exist_ok=True)
    write_json(primary_dir / "paired_bootstrap.json", bootstrap)
    write_json(
        primary_dir / "m3_comparison.json",
        {
            "primary_ids": list(frozen.ids_49),
            "excluded_failure_sample": FAILURE_SAMPLE_ID,
            "sample_count": len(frozen.ids_49),
            "method_means": method_means,
            "comparisons": comparison_summary,
            "significance_policy": "interpret paired confidence intervals; raw differences alone are insufficient",
        },
    )
    columns = ["sample_id"]
    for method in methods:
        columns.extend(
            [f"{method}_prediction", *[f"{method}_{metric}" for metric in ("CIDEr", "EntityPrecision", "EntityRecall", "EntityF1")]]
        )
    per_sample_path = primary_dir / "per_sample_comparison.csv"
    with per_sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sample_id in frozen.ids_49:
            row: dict[str, Any] = {"sample_id": sample_id}
            for method in methods:
                row[f"{method}_prediction"] = predictions[method][sample_id]
                for metric in ("CIDEr", "EntityPrecision", "EntityRecall", "EntityF1"):
                    row[f"{method}_{metric}"] = metrics[method][sample_id][metric]
            writer.writerow(row)
    qualitative = build_qualitative_candidates(frozen, predictions, metrics)
    write_json(primary_dir / "qualitative_examples.json", qualitative)
    lines = [
        "# M3 qualitative review",
        "",
        "These examples are selected by metric/entity/context heuristics and require human factuality review.",
        "",
    ]
    for item in qualitative:
        lines.extend(
            [
                f"## {item['category']}: {item['sample_id']}",
                "",
                f"- Selection basis: {item['selection_basis']}",
                f"- B2: {item['b2_caption']}",
                f"- Atomic Full: {item['atomic_full_caption']}",
                f"- Atomic Token-Matched: {item['atomic_token_matched_caption']}",
                f"- Reference: {item['reference_caption']}",
                "",
            ]
        )
    (primary_dir / "qualitative_review.md").write_text("\n".join(lines), encoding="utf-8")
    return comparison_summary


def build_qualitative_candidates(
    frozen: FrozenM3Inputs,
    predictions: Mapping[str, Mapping[str, str]],
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> list[dict[str, Any]]:
    b2_evidence = {str(row["sample_id"]): row for row in frozen.selected_evidence}
    atomic = {str(row["sample_id"]): row for row in frozen.atomic_evidence}
    full = {str(row["sample_id"]): row for row in frozen.full_contexts}
    matched = {str(row["sample_id"]): row for row in frozen.matched_contexts}
    reference = {str(row["sample_id"]): row["reference"] for row in frozen.b2_predictions}
    scored = []
    for sample_id in frozen.ids_49:
        scored.append(
            (
                sample_id,
                metrics["atomic_token_matched"][sample_id]["EntityF1"]
                - metrics["b2_semantic_k3"][sample_id]["EntityF1"],
                metrics["atomic_full"][sample_id]["EntityF1"]
                - metrics["b2_semantic_k3"][sample_id]["EntityF1"],
                len(matched[sample_id]["dropped_evidence_ids"]),
            )
        )
    choices = [
        ("atomic_token_matched_metric_gain", max(scored, key=lambda item: item[1])[0], "largest EntityF1 delta versus B2; not a factuality label"),
        ("atomic_full_metric_gain", max(scored, key=lambda item: item[2])[0], "largest EntityF1 delta versus B2; not a factuality label"),
        ("b2_metric_advantage", min(scored, key=lambda item: min(item[1], item[2]))[0], "largest Atomic EntityF1 deficit versus B2; not a factuality label"),
        (
            "atomic_named_entity_change",
            next(
                (
                    item[0]
                    for item in scored
                    if metrics["atomic_full"][item[0]].get("EntityPredicted")
                    != metrics["b2_semantic_k3"][item[0]].get("EntityPredicted")
                ),
                scored[0][0],
            ),
            "spaCy predicted-entity count changed; correctness requires human confirmation",
        ),
        ("token_matching_dropped_evidence", max(scored, key=lambda item: item[3])[0], "largest number of evidence units dropped by token matching"),
        ("full_contains_evidence_dropped_by_matched", max(scored, key=lambda item: item[3])[0], "Atomic Full includes evidence IDs omitted from Token-Matched"),
    ]
    result = []
    for category, sample_id, basis in choices:
        result.append(
            {
                "category": category,
                "selection_basis": basis,
                "requires_human_factuality_review": True,
                "sample_id": sample_id,
                "reference_caption": reference[sample_id],
                "b2_caption": predictions["b2_semantic_k3"][sample_id],
                "atomic_full_caption": predictions["atomic_full"][sample_id],
                "atomic_token_matched_caption": predictions["atomic_token_matched"][sample_id],
                "b2_selected_sentences": b2_evidence[sample_id]["selected_sentence_texts"],
                "atomic_evidence": atomic[sample_id]["atomic_units"],
                "full_dropped_evidence_ids": full[sample_id]["dropped_evidence_ids"],
                "token_matched_dropped_evidence_ids": matched[sample_id]["dropped_evidence_ids"],
            }
        )
    return result
