"""Freeze LLM atomic evidence from the exact B2 semantic-k3 selection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .llm_structured import (
    EVIDENCE_TYPES,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    stable_evidence_id,
)


RETRIEVAL_MODEL = "openai/clip-vit-base-patch32"
RETRIEVAL_REVISION = "b97b0100e55e367c057773c2a614676470b0d575"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_float(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def load_verified_frozen_b2(
    selected_evidence_path: Path,
    rankings_path: Path,
    manifest_path: Path,
    resolved_config_path: Path,
    ids_path: Path,
) -> list[dict[str, Any]]:
    """Return the authoritative 50 rows after cross-artifact verification."""

    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError("M3 requires the approved 50 unique sample IDs")
    ids = [str(value) for value in ids]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    expected_retrieval = {
        "method": "semantic",
        "k": 3,
        "model_name": RETRIEVAL_MODEL,
        "model_revision": RETRIEVAL_REVISION,
    }
    for name, source in (("manifest", manifest), ("resolved config", resolved)):
        retrieval = source.get("retrieval", {})
        actual = {
            "method": retrieval.get("method"),
            "k": int(retrieval.get("k", 0)),
            "model_name": retrieval.get("model_name", retrieval.get("name")),
            "model_revision": retrieval.get("model_revision", retrieval.get("revision")),
        }
        if actual != expected_retrieval:
            raise ValueError(f"{name} does not describe the frozen semantic-k3 retriever")
    dataset = manifest.get("dataset", {})
    if (
        dataset.get("split") != "dev"
        or int(dataset.get("max_samples", 0)) != 50
        or dataset.get("subset_strategy") != "first_by_sample_id"
    ):
        raise ValueError("frozen B2 manifest has an unexpected dataset selection")
    if manifest.get("rankings_sha256") != _sha256(rankings_path):
        raise ValueError("ranking artifact hash does not match the frozen B2 manifest")

    selected_rows = _read_jsonl(selected_evidence_path)
    ranking_rows = _read_jsonl(rankings_path)
    selected_ids = [str(row.get("sample_id", "")) for row in selected_rows]
    ranking_ids = [str(row.get("sample_id", "")) for row in ranking_rows]
    if selected_ids != ids or ranking_ids != ids:
        raise ValueError("frozen B2 artifacts do not match the approved ordered sample IDs")

    for selected, ranking in zip(selected_rows, ranking_rows, strict=True):
        sample_id = str(selected["sample_id"])
        forbidden = {
            key
            for key in selected
            if any(
                marker in key.casefold()
                for marker in ("reference", "caption", "human_label", "spacy")
            )
            or key.casefold() in {"image", "image_path", "article_text", "full_article"}
        }
        if forbidden:
            raise ValueError(f"forbidden input fields for {sample_id}: {sorted(forbidden)}")
        if selected.get("retrieval_method") != "semantic" or int(
            selected.get("retrieval_k", 0)
        ) != 3:
            raise ValueError(f"{sample_id} is not frozen semantic-k3 evidence")
        fields = (
            selected.get("selected_sentence_ids"),
            selected.get("selected_sentence_texts"),
            selected.get("selected_ranking_scores"),
            selected.get("selected_ranks"),
        )
        if not all(isinstance(value, list) and len(value) == 3 for value in fields):
            raise ValueError(f"{sample_id} must contain exactly three selected sentences")
        sentence_ids, texts, scores, ranks = fields
        integer_ids = [int(value) for value in sentence_ids]
        if integer_ids != sorted(integer_ids) or len(set(integer_ids)) != 3:
            raise ValueError(f"{sample_id} selected sentences are not in article order")
        if sorted(int(value) for value in ranks) != [1, 2, 3]:
            raise ValueError(f"{sample_id} does not contain retrieval ranks 1, 2, and 3")
        retriever = ranking.get("retriever", {})
        if {
            "method": retriever.get("method"),
            "name": retriever.get("name"),
            "revision": retriever.get("revision"),
        } != {
            "method": "semantic",
            "name": RETRIEVAL_MODEL,
            "revision": RETRIEVAL_REVISION,
        }:
            raise ValueError(f"{sample_id} ranking retriever provenance changed")
        ranked = ranking.get("ranked_sentences")
        if not isinstance(ranked, list):
            raise ValueError(f"malformed ranking row for {sample_id}")
        by_id = {int(item["sentence_id"]): item for item in ranked}
        if len(by_id) != len(ranked):
            raise ValueError(f"duplicate ranked sentence IDs for {sample_id}")
        for sentence_id, text, score, rank in zip(
            integer_ids, texts, scores, ranks, strict=True
        ):
            candidate = by_id.get(sentence_id)
            if (
                candidate is None
                or str(candidate.get("text")) != str(text)
                or int(candidate.get("rank", 0)) != int(rank)
                or not _same_float(candidate.get("score"), score)
            ):
                raise ValueError(
                    f"selected evidence differs from frozen ranking for {sample_id}:{sentence_id}"
                )
        if int(selected.get("selected_sentence_count", 0)) != 3:
            raise ValueError(f"selected sentence count changed for {sample_id}")
        context_tokens = int(selected.get("context_token_count", 0))
        if context_tokens <= 0 or context_tokens > 384:
            raise ValueError(f"invalid frozen B2 token budget for {sample_id}")
    return selected_rows


def load_reviewed_comparison(
    path: Path,
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    expected_reviewed_rows: int = 100,
    expected_success_rows: int = 99,
    expected_failure_rows: int = 1,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    """Recover exact reviewed LLM units after strict frozen-source validation."""

    frozen: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in selected_rows:
        for sentence_id, source_rank, score, text in zip(
            sample["selected_sentence_ids"],
            sample["selected_ranks"],
            sample["selected_ranking_scores"],
            sample["selected_sentence_texts"],
            strict=True,
        ):
            key = (str(sample["sample_id"]), str(sentence_id))
            if key in frozen:
                raise ValueError(f"duplicate frozen source key: {key}")
            frozen[key] = {
                "sample_id": key[0],
                "source_sentence_id": key[1],
                "source_rank": int(source_rank),
                "ranking_score": float(score),
                "source_sentence": str(text),
            }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "source_sentence_id",
            "source_rank",
            "source_sentence",
            "llm_status",
            "llm_units_json",
            "llm_unit_count",
            "llm_failure_type",
            "llm_failure_message",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"reviewed comparison is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if len(rows) != expected_reviewed_rows:
        raise ValueError(
            f"expected {expected_reviewed_rows} reviewed rows, found {len(rows)}"
        )

    recovered: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    success_count = failure_count = 0
    for row in rows:
        key = (str(row["sample_id"]), str(row["source_sentence_id"]))
        if not all(key) or key in seen:
            raise ValueError(f"empty or duplicate reviewed source key: {key}")
        seen.add(key)
        source = frozen.get(key)
        if source is None:
            raise ValueError(f"reviewed source is not in frozen semantic-k3: {key}")
        if (
            int(row["source_rank"]) != source["source_rank"]
            or row["source_sentence"] != source["source_sentence"]
        ):
            raise ValueError(f"reviewed source provenance mismatch for {key}")
        status = str(row["llm_status"]).strip().casefold()
        if status == "failed":
            failure_count += 1
            if int(row["llm_unit_count"] or 0) != 0:
                raise ValueError(f"failed reviewed row contains LLM units for {key}")
            try:
                failed_units = json.loads(row["llm_units_json"] or "[]")
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed llm_units_json for failed row {key}") from error
            if failed_units != []:
                raise ValueError(f"failed reviewed row contains LLM evidence for {key}")
            continue
        if status != "success":
            raise ValueError(f"invalid llm_status for reviewed row {key}: {status!r}")
        success_count += 1
        try:
            units = json.loads(row["llm_units_json"])
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed llm_units_json for {key}") from error
        if not isinstance(units, list) or len(units) != int(row["llm_unit_count"]):
            raise ValueError(f"reviewed LLM unit count mismatch for {key}")
        evidence_ids: set[str] = set()
        for index, unit in enumerate(units):
            if not isinstance(unit, dict):
                raise ValueError(f"reviewed evidence unit is not an object for {key}")
            required_unit = {
                "evidence_id",
                "text",
                "type",
                "source_span",
                "provenance",
            }
            if required_unit - set(unit):
                raise ValueError(f"reviewed evidence unit has an invalid schema for {key}")
            text = unit["text"]
            evidence_type = unit["type"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"reviewed evidence unit has empty text for {key}")
            if evidence_type not in EVIDENCE_TYPES:
                raise ValueError(f"reviewed evidence unit has invalid type for {key}")
            evidence_id = str(unit["evidence_id"])
            if (
                evidence_id
                != stable_evidence_id(key[0], key[1], index, text, evidence_type)
                or evidence_id in evidence_ids
            ):
                raise ValueError(f"reviewed evidence ID validation failed for {key}")
            evidence_ids.add(evidence_id)
            provenance = unit["provenance"]
            if not isinstance(provenance, dict) or (
                str(provenance.get("sample_id")) != key[0]
                or str(provenance.get("source_sentence_id")) != key[1]
                or int(provenance.get("source_rank", 0)) != source["source_rank"]
                or not _same_float(
                    provenance.get("ranking_score"), source["ranking_score"]
                )
                or provenance.get("source_sentence") != source["source_sentence"]
            ):
                raise ValueError(f"reviewed evidence provenance mismatch for {key}")
            span = unit["source_span"]
            if not isinstance(span, dict) or not isinstance(span.get("text"), str):
                raise ValueError(f"reviewed evidence source span is malformed for {key}")
            alignment = span.get("alignment")
            if alignment == "exact":
                start, end = span.get("start"), span.get("end")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or start < 0
                    or end <= start
                    or end > len(source["source_sentence"])
                    or source["source_sentence"][start:end] != span["text"]
                ):
                    raise ValueError(f"reviewed exact source span mismatch for {key}")
            elif alignment == "unresolved":
                if span.get("start") is not None or span.get("end") is not None:
                    raise ValueError(f"reviewed unresolved span has offsets for {key}")
            else:
                raise ValueError(f"reviewed evidence span alignment is invalid for {key}")
        recovered[key] = units
    if success_count != expected_success_rows or failure_count != expected_failure_rows:
        raise ValueError(
            "reviewed comparison status counts changed: "
            f"success={success_count}, failed={failure_count}"
        )
    return recovered, {
        "source_file": str(path.resolve()),
        "source_file_sha256": _sha256(path),
        "reviewed_rows": len(rows),
        "recovered_success_rows": success_count,
        "reviewed_failure_rows": failure_count,
        "previous_failure_rows": failure_count,
        "validation_failures": 0,
    }


def prepare_cache(cache_path: Path, cache_source: Path | None) -> dict[str, Any]:
    """Seed a run cache once, never replacing an existing destination cache."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "path": str(cache_path.resolve()),
        "seed_source": str(cache_source.resolve()) if cache_source else None,
        "seeded": False,
        "existing_cache_preserved": cache_path.exists(),
    }
    if cache_path.exists():
        report["sha256_before_run"] = _sha256(cache_path)
        return report
    if cache_source is not None:
        if not cache_source.is_file():
            raise FileNotFoundError(f"validated cache source not found: {cache_source}")
        shutil.copy2(cache_source, cache_path)
        report["seeded"] = True
        report["existing_cache_preserved"] = False
        report["seed_source_sha256"] = _sha256(cache_source)
        report["sha256_before_run"] = _sha256(cache_path)
    return report


def format_atomic_context(units: Sequence[Mapping[str, Any]]) -> str:
    return "Evidence:\n" + "\n".join(f"- {unit['text']}" for unit in units)


def _fit_closest(
    units: Sequence[Mapping[str, Any]],
    token_counter: Callable[[str], int],
    budget: int,
) -> tuple[list[Mapping[str, Any]], str, int, list[str]]:
    """Select an order-preserving subset with maximum exact rendered token use."""

    if budget <= 0:
        raise ValueError("context token budget must be positive")
    states: dict[int, tuple[int, ...]] = {}

    def tokens(indices: tuple[int, ...]) -> int:
        return token_counter(format_atomic_context([units[index] for index in indices]))

    empty_tokens = tokens(())
    if empty_tokens > budget:
        raise ValueError("context budget is smaller than the empty Evidence header")
    states[empty_tokens] = ()
    for index in range(len(units)):
        additions: dict[int, tuple[int, ...]] = {}
        for indices in list(states.values()):
            candidate = (*indices, index)
            count = tokens(candidate)
            if count <= budget:
                current = states.get(count, additions.get(count))
                if current is None or candidate < current:
                    additions[count] = candidate
        for count, candidate in additions.items():
            current = states.get(count)
            if current is None or candidate < current:
                states[count] = candidate
    best_count = max(states)
    best_indices = states[best_count]
    selected = [units[index] for index in best_indices]
    selected_ids = {str(unit["evidence_id"]) for unit in selected}
    dropped = [
        str(unit["evidence_id"])
        for unit in units
        if str(unit["evidence_id"]) not in selected_ids
    ]
    return selected, format_atomic_context(selected), best_count, dropped


def _distribution(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _validate_result(result: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    expected = {
        "sample_id": str(source["sample_id"]),
        "source_sentence_id": str(source["source_sentence_id"]),
        "source_rank": int(source["source_rank"]),
        "ranking_score": float(source["ranking_score"]),
        "source_sentence": str(source["source_sentence"]),
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise RuntimeError(f"extractor changed frozen provenance field {field}")
    propositions = result.get("propositions")
    if not isinstance(propositions, list):
        raise RuntimeError("extractor propositions are not a list")
    for unit in propositions:
        if not isinstance(unit, Mapping) or unit.get("type") not in EVIDENCE_TYPES:
            raise RuntimeError("extractor returned malformed evidence")
        provenance = unit.get("provenance")
        if not isinstance(provenance, Mapping):
            raise RuntimeError("evidence unit is missing provenance")
        for field, value in expected.items():
            if provenance.get(field) != value:
                raise RuntimeError(f"evidence unit changed frozen provenance field {field}")


def run_frozen_llm_extraction(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    extractor: Any,
    token_counter: Callable[[str], int],
    output_dir: Path,
    context_budget: int = 384,
    recovered_evidence: Mapping[
        tuple[str, str], Sequence[Mapping[str, Any]]
    ] | None = None,
) -> dict[str, Any]:
    """Extract, aggregate, audit, and freeze contexts without caption generation."""

    if context_budget != 384:
        raise ValueError("the frozen M3 full-context budget must remain 384 tokens")
    if len(selected_rows) != 50:
        raise ValueError("M3 requires exactly 50 frozen samples")
    recovered_evidence = recovered_evidence or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    failures: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    full_context_rows: list[dict[str, Any]] = []
    matched_context_rows: list[dict[str, Any]] = []
    units_per_sentence: list[int] = []
    units_per_sample: list[int] = []
    types: Counter[str] = Counter()
    unresolved_spans = 0
    duplicate_units = 0
    total_units = 0
    all_three_completed = 0
    affected_samples: list[str] = []
    recovered_outputs_reused = 0
    api_extraction_targets = 0

    for row in selected_rows:
        sample_id = str(row["sample_id"])
        sentence_results: list[dict[str, Any]] = []
        sample_failures: list[dict[str, Any]] = []
        for sentence_id, source_rank, ranking_score, source_text in zip(
            row["selected_sentence_ids"],
            row["selected_ranks"],
            row["selected_ranking_scores"],
            row["selected_sentence_texts"],
            strict=True,
        ):
            source = {
                "sample_id": sample_id,
                "source_sentence_id": str(sentence_id),
                "source_rank": int(source_rank),
                "ranking_score": float(ranking_score),
                "source_sentence": str(source_text),
            }
            key = (source["sample_id"], source["source_sentence_id"])
            if key in recovered_evidence:
                units = list(recovered_evidence[key])
                result = {
                    **source,
                    "extractor": PROMPT_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "propositions": units,
                    "cache_hit": False,
                }
                _validate_result(result, source)
                units_per_sentence.append(len(units))
                sentence_results.append(
                    {
                        **source,
                        "status": "success",
                        "origin": "reviewed_comparison_recovery",
                        **result,
                    }
                )
                recovered_outputs_reused += 1
                continue
            api_extraction_targets += 1
            try:
                result = extractor.extract(
                    source["source_sentence"],
                    sample_id=source["sample_id"],
                    source_sentence_id=source["source_sentence_id"],
                    source_rank=source["source_rank"],
                    ranking_score=source["ranking_score"],
                )
                _validate_result(result, source)
                units_per_sentence.append(len(result["propositions"]))
                sentence_results.append(
                    {
                        **source,
                        "status": "success",
                        "origin": "api_extraction",
                        **dict(result),
                    }
                )
            except Exception as error:  # Preserve explicit per-source failures.
                failure = {
                    **source,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                sample_failures.append(failure)
                sentence_results.append(
                    {
                        **source,
                        "status": "failed",
                        "origin": "api_extraction",
                        "propositions": [],
                        **failure,
                    }
                )
        if not sample_failures:
            all_three_completed += 1
        else:
            affected_samples.append(sample_id)
        units = [
            unit
            for sentence in sentence_results
            for unit in sentence.get("propositions", [])
        ]
        units_per_sample.append(len(units))
        total_units += len(units)
        types.update(str(unit["type"]) for unit in units)
        unresolved_spans += sum(
            not isinstance(unit.get("source_span"), Mapping)
            or unit["source_span"].get("alignment") != "exact"
            for unit in units
        )
        normalized = [" ".join(str(unit["text"]).casefold().split()) for unit in units]
        counts = Counter(normalized)
        duplicate_units += sum(count for count in counts.values() if count > 1)

        full_units, full_context, full_tokens, full_dropped = _fit_closest(
            units, token_counter, context_budget
        )
        b2_budget = int(row["context_token_count"])
        matched_units, matched_context, matched_tokens, matched_dropped = _fit_closest(
            units, token_counter, b2_budget
        )
        evidence_rows.append(
            {
                "sample_id": sample_id,
                "retrieval_method": "semantic",
                "retrieval_k": 3,
                "retrieval_model": RETRIEVAL_MODEL,
                "retrieval_revision": RETRIEVAL_REVISION,
                "selected_sentence_ids": [int(value) for value in row["selected_sentence_ids"]],
                "selected_ranks": [int(value) for value in row["selected_ranks"]],
                "selected_ranking_scores": [
                    float(value) for value in row["selected_ranking_scores"]
                ],
                "selected_sentence_texts": [str(value) for value in row["selected_sentence_texts"]],
                "completed_source_sentences": 3 - len(sample_failures),
                "failed_source_sentences": len(sample_failures),
                "sentences": sentence_results,
                "atomic_units": units,
            }
        )
        full_context_rows.append(
            {
                "sample_id": sample_id,
                "context_kind": "atomic_full_384",
                "context": full_context,
                "context_tokens": full_tokens,
                "max_context_tokens": context_budget,
                "included_evidence_ids": [unit["evidence_id"] for unit in full_units],
                "dropped_evidence_ids": full_dropped,
                "all_source_sentences_completed": not sample_failures,
            }
        )
        matched_context_rows.append(
            {
                "sample_id": sample_id,
                "context_kind": "atomic_token_matched",
                "context": matched_context,
                "context_tokens": matched_tokens,
                "b2_context_token_budget": b2_budget,
                "token_difference_from_b2": matched_tokens - b2_budget,
                "absolute_token_difference_from_b2": abs(matched_tokens - b2_budget),
                "included_evidence_ids": [unit["evidence_id"] for unit in matched_units],
                "dropped_evidence_ids": matched_dropped,
                "all_source_sentences_completed": not sample_failures,
            }
        )

    _write_jsonl(output_dir / "atomic_evidence.jsonl", evidence_rows)
    _write_jsonl(output_dir / "atomic_contexts_full.jsonl", full_context_rows)
    _write_jsonl(output_dir / "atomic_contexts_token_matched.jsonl", matched_context_rows)
    _write_jsonl(output_dir / "failures.jsonl", failures)
    b2_tokens = [int(row["context_token_count"]) for row in selected_rows]
    full_tokens = [int(row["context_tokens"]) for row in full_context_rows]
    matched_tokens = [int(row["context_tokens"]) for row in matched_context_rows]
    audit = {
        "status": "complete" if not failures else "complete_with_failures",
        "source_sentences_requested": 150,
        "source_sentences_completed": 150 - len(failures),
        "final_completed_source_sentences": 150 - len(failures),
        "source_sentences_failed": len(failures),
        "failures": len(failures),
        "reviewed_outputs_reused": recovered_outputs_reused,
        "api_extraction_targets": api_extraction_targets,
        "new_api_extractions": extractor.stats().get("successful_request_count", 0),
        "api_retries": extractor.stats().get("retry_count", 0),
        "samples": 50,
        "samples_with_all_3_sentences_completed": all_three_completed,
        "samples_affected_by_extraction_failures": affected_samples,
        "total_atomic_units": total_units,
        "units_per_source_sentence": _distribution(units_per_sentence),
        "units_per_sample": _distribution(units_per_sample),
        "type_distribution": dict(sorted(types.items())),
        "unresolved_spans": unresolved_spans,
        "exact_duplicate_units": duplicate_units,
        "exact_duplicate_rate": duplicate_units / total_units if total_units else 0.0,
        "context_tokens": {
            "b2": _distribution(b2_tokens),
            "atomic_full": _distribution(full_tokens),
            "atomic_token_matched": _distribution(matched_tokens),
        },
        "units_dropped_for_384_limit": sum(
            len(row["dropped_evidence_ids"]) for row in full_context_rows
        ),
        "units_dropped_for_token_matching": sum(
            len(row["dropped_evidence_ids"]) for row in matched_context_rows
        ),
        "mean_absolute_token_difference_from_b2": statistics.fmean(
            abs(atomic - b2)
            for atomic, b2 in zip(matched_tokens, b2_tokens, strict=True)
        ),
        "api": extractor.stats(),
        "run_started_at": started_at,
        "run_completed_at": datetime.now(timezone.utc).isoformat(),
        "caption_generation_run": False,
        "spacy_fallback_used": False,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


__all__ = [
    "RETRIEVAL_MODEL",
    "RETRIEVAL_REVISION",
    "format_atomic_context",
    "load_reviewed_comparison",
    "load_verified_frozen_b2",
    "prepare_cache",
    "run_frozen_llm_extraction",
]
