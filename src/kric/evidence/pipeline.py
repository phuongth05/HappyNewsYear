"""Frozen-input atomic extraction, context preparation, and quality auditing."""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .cache import AtomicEvidenceCache
from .schema import AtomicEvidence, FrozenSentence
from .spacy_extractor import SpacyAtomicExtractor


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def load_frozen_semantic_k3(
    selected_evidence_path: Path, ids_path: Path
) -> list[tuple[str, list[FrozenSentence], dict[str, Any]]]:
    """Load only the already-selected B2 evidence; no article API is accepted."""

    expected_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if len(expected_ids) != 50 or len(set(expected_ids)) != 50:
        raise ValueError("atomic extraction requires the approved 50 unique sample IDs")
    rows = _read_jsonl(selected_evidence_path)
    if [str(row.get("sample_id", "")) for row in rows] != expected_ids:
        raise ValueError("frozen B2 evidence does not match the approved ordered sample IDs")
    result = []
    for row in rows:
        if row.get("retrieval_method") != "semantic" or int(row.get("retrieval_k", 0)) != 3:
            raise ValueError("atomic extraction requires frozen semantic k=3 evidence")
        forbidden = {key for key in row if "reference" in key.lower() or "caption" in key.lower()}
        if forbidden:
            raise ValueError(f"frozen evidence contains forbidden fields: {sorted(forbidden)}")
        sentence_ids = row.get("selected_sentence_ids")
        texts = row.get("selected_sentence_texts")
        scores = row.get("selected_ranking_scores")
        ranks = row.get("selected_ranks")
        if not all(isinstance(value, list) for value in (sentence_ids, texts, scores, ranks)):
            raise ValueError(f"malformed selected evidence for {row.get('sample_id')}")
        lengths = {len(sentence_ids), len(texts), len(scores), len(ranks)}
        if len(lengths) != 1 or not 1 <= len(sentence_ids) <= 3:
            raise ValueError(f"invalid frozen top-k arrays for {row.get('sample_id')}")
        if [int(value) for value in sentence_ids] != sorted(int(value) for value in sentence_ids):
            raise ValueError("selected sentences must be stored in original article order")
        if len(set(int(value) for value in sentence_ids)) != len(sentence_ids):
            raise ValueError("selected sentence IDs must be unique")
        sentences = [
            FrozenSentence(
                sample_id=str(row["sample_id"]),
                sentence_id=int(sentence_id),
                text=str(text),
                source_rank=int(rank),
                ranking_score=float(score),
            )
            for sentence_id, text, score, rank in zip(
                sentence_ids, texts, scores, ranks, strict=True
            )
        ]
        result.append((str(row["sample_id"]), sentences, row))
    return result


def format_atomic_context(units: Sequence[AtomicEvidence]) -> str:
    return "Evidence:\n" + "\n".join(f"- {unit.text}" for unit in units)


def _extract_with_retries(
    extractor: Any, sentence: FrozenSentence, *, max_attempts: int = 2
) -> tuple[list[AtomicEvidence], int]:
    """Validate structured units and retry a malformed extractor response."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    failures = 0
    last_error: Exception | None = None
    for _attempt in range(max_attempts):
        try:
            raw_units = extractor.extract(sentence)
            if not isinstance(raw_units, list):
                raise ValueError("extractor output must be a list")
            units = []
            for raw in raw_units:
                if isinstance(raw, AtomicEvidence):
                    unit = AtomicEvidence.from_dict(raw.to_dict())
                elif isinstance(raw, Mapping):
                    unit = AtomicEvidence.from_dict(raw)
                else:
                    raise ValueError("each extractor output must be an evidence object")
                if unit.source_sentence_id != sentence.sentence_id:
                    raise ValueError("extractor changed frozen source_sentence_id")
                units.append(unit)
            return units, failures
        except (KeyError, TypeError, ValueError) as error:
            failures += 1
            last_error = error
    raise ValueError(
        f"malformed extractor output after {max_attempts} attempts for "
        f"{sentence.sample_id}:{sentence.sentence_id}"
    ) from last_error


def _fit_units(
    units: Sequence[AtomicEvidence], token_counter: Callable[[str], int], budget: int
) -> tuple[list[AtomicEvidence], str, int]:
    selected: list[AtomicEvidence] = []
    for unit in units:
        trial = [*selected, unit]
        rendered = format_atomic_context(trial)
        if token_counter(rendered) <= budget:
            selected.append(unit)
    rendered = format_atomic_context(selected)
    tokens = token_counter(rendered)
    if tokens > budget:
        raise RuntimeError(f"atomic context budget exceeded: {tokens} > {budget}")
    return selected, rendered, tokens


_WORD = re.compile(r"\b[\w'’-]+\b", flags=re.UNICODE)
_TEMPLATE_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "person",
    "organization",
    "group",
    "location",
    "place",
    "time",
    "event",
    "work",
    "product",
}


def _normalized(text: str) -> str:
    return " ".join(token.casefold() for token in _WORD.findall(text))


def _flags(unit: AtomicEvidence, source: str, token_counter: Callable[[str], int]) -> list[str]:
    flags = []
    token_length = token_counter(unit.text)
    if token_length > 40:
        flags.append("very_long_proposition")
    if ";" in unit.text or len(re.findall(r"\b(?:and|but|while|whereas)\b", unit.text, re.I)) > 1:
        flags.append("possible_multiple_independent_clauses")
    words = _WORD.findall(unit.text)
    if len(words) < 3:
        flags.append("isolated_keyword")
    pronouns = {"he", "she", "it", "they", "him", "her", "them", "this", "that"}
    content = {word.casefold() for word in words if word.casefold() not in _TEMPLATE_WORDS}
    if content and content <= pronouns:
        flags.append("vague_pronoun_only")
    source_words = {word.casefold() for word in _WORD.findall(source)}
    support_terms = content - pronouns
    overlap = len(support_terms & source_words) / len(support_terms) if support_terms else 1.0
    if overlap < 0.5:
        flags.append("possible_source_support_failure")
    if unit.source_span is None:
        flags.append("missing_source_span")
    elif (
        unit.source_span.end > len(source)
        or source[unit.source_span.start : unit.source_span.end] != unit.source_span.text
    ):
        flags.append("invalid_source_span")
    return flags


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


def run_atomic_extraction(
    selected_evidence_path: Path,
    ids_path: Path,
    output_dir: Path,
    *,
    extractor: SpacyAtomicExtractor,
    token_counter: Callable[[str], int],
    cache_path: Path,
    context_budget: int = 384,
    manual_review_count: int = 100,
) -> dict[str, Any]:
    if context_budget != 384:
        raise ValueError("first atomic granularity study requires the frozen 384-token budget")
    frozen = load_frozen_semantic_k3(selected_evidence_path, ids_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_rows = []
    context_rows = []
    review_rows = []
    cache_hits = cache_misses = 0
    units_per_sentence = []
    units_per_sample = []
    evidence_token_lengths = []
    type_counts: Counter[str] = Counter()
    suspicious_counts: Counter[str] = Counter()
    exact_duplicate_units = near_duplicate_units = 0
    duplicate_units = 0
    malformed_units = empty_units = missing_provenance = 0
    malformed_extraction_attempts = 0

    with AtomicEvidenceCache(cache_path) as cache:
        for sample_id, sentences, source_row in frozen:
            sample_units: list[AtomicEvidence] = []
            sentence_records = []
            for sentence in sentences:
                cache_key = extractor.cache_key(sentence)
                units = cache.get(cache_key)
                if units is None:
                    cache_misses += 1
                    units, failures = _extract_with_retries(extractor, sentence)
                    malformed_extraction_attempts += failures
                    cache.put(cache_key, units)
                else:
                    cache_hits += 1
                validated = []
                for unit in units:
                    try:
                        checked = AtomicEvidence.from_dict(unit.to_dict())
                    except (KeyError, TypeError, ValueError):
                        malformed_units += 1
                        continue
                    if not checked.text.strip():
                        empty_units += 1
                        continue
                    if checked.source_sentence_id != sentence.sentence_id:
                        raise RuntimeError("extractor changed frozen source_sentence_id")
                    validated.append(checked)
                units_per_sentence.append(len(validated))
                sample_units.extend(validated)
                rendered_units = []
                for unit in validated:
                    flags = _flags(unit, sentence.text, token_counter)
                    suspicious_counts.update(flags)
                    type_counts[unit.type] += 1
                    length = token_counter(unit.text)
                    evidence_token_lengths.append(length)
                    if unit.source_span is None:
                        missing_provenance += 1
                    rendered_units.append({**unit.to_dict(), "token_length": length, "flags": flags})
                sentence_records.append(
                    {
                        "sentence_id": sentence.sentence_id,
                        "source_rank": sentence.source_rank,
                        "ranking_score": sentence.ranking_score,
                        "source_sentence": sentence.text,
                        "atomic_units": rendered_units,
                    }
                )
                if len(review_rows) < manual_review_count:
                    review_rows.append(
                        {
                            "sample_id": sample_id,
                            "source_sentence_id": sentence.sentence_id,
                            "source_rank": sentence.source_rank,
                            "source_sentence": sentence.text,
                            "atomic_units": rendered_units,
                            "atomicity": "",
                            "faithfulness": "",
                            "type_correct": "",
                            "notes": "",
                        }
                    )

            units_per_sample.append(len(sample_units))
            normalized = [_normalized(unit.text) for unit in sample_units]
            counts = Counter(normalized)
            exact_indices = {index for index, value in enumerate(normalized) if counts[value] > 1}
            exact_duplicate_units += len(exact_indices)
            near_indices = set()
            for left in range(len(normalized)):
                for right in range(left + 1, len(normalized)):
                    if normalized[left] == normalized[right]:
                        continue
                    if difflib.SequenceMatcher(None, normalized[left], normalized[right]).ratio() >= 0.9:
                        near_indices.update((left, right))
            near_duplicate_units += len(near_indices)
            duplicate_units += len(exact_indices | near_indices)

            ordered_units = sorted(
                sample_units,
                key=lambda unit: (
                    unit.source_sentence_id,
                    unit.source_span.start if unit.source_span else 10**9,
                    unit.evidence_id,
                ),
            )
            budget_units, context, atomic_context_tokens = _fit_units(
                ordered_units, token_counter, context_budget
            )
            raw_evidence_tokens = token_counter(
                "\n".join(unit.text for unit in ordered_units)
            )
            b2_tokens = int(source_row["context_token_count"])
            matched_budget = min(context_budget, b2_tokens)
            matched_units, matched_context, matched_tokens = _fit_units(
                ordered_units, token_counter, matched_budget
            )
            evidence_rows.append(
                {
                    "sample_id": sample_id,
                    "retrieval_method": "semantic",
                    "retrieval_k": 3,
                    "selected_sentence_ids": [sentence.sentence_id for sentence in sentences],
                    "selected_sentence_texts": [sentence.text for sentence in sentences],
                    "selected_ranks": [sentence.source_rank for sentence in sentences],
                    "selected_ranking_scores": [sentence.ranking_score for sentence in sentences],
                    "sentences": sentence_records,
                    "atomic_units": [unit.to_dict() for unit in ordered_units],
                }
            )
            context_rows.append(
                {
                    "sample_id": sample_id,
                    "b2_selected_sentence_tokens": b2_tokens,
                    "atomic_evidence_unit_count": len(ordered_units),
                    "atomic_evidence_tokens_unformatted": raw_evidence_tokens,
                    "atomic_context": context,
                    "atomic_context_tokens": atomic_context_tokens,
                    "atomic_context_evidence_ids": [unit.evidence_id for unit in budget_units],
                    "atomic_context_included_unit_count": len(budget_units),
                    "atomic_context_dropped_unit_count": len(ordered_units) - len(budget_units),
                    "token_matched_budget": matched_budget,
                    "token_matched_context": matched_context,
                    "token_matched_context_tokens": matched_tokens,
                    "token_matched_evidence_ids": [unit.evidence_id for unit in matched_units],
                    "token_matched_included_unit_count": len(matched_units),
                    "token_matched_dropped_unit_count": len(ordered_units) - len(matched_units),
                    "context_budget": context_budget,
                }
            )

    _write_jsonl(output_dir / "atomic_evidence.jsonl", evidence_rows)
    _write_jsonl(output_dir / "atomic_contexts.jsonl", context_rows)
    _write_jsonl(output_dir / "manual_review_100.jsonl", review_rows)
    with (output_dir / "manual_review_100.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "source_sentence_id",
                "source_rank",
                "source_sentence",
                "atomic_units_json",
                "atomicity",
                "faithfulness",
                "type_correct",
                "notes",
            ),
        )
        writer.writeheader()
        for row in review_rows:
            writer.writerow(
                {
                    **{key: row[key] for key in writer.fieldnames if key != "atomic_units_json"},
                    "atomic_units_json": json.dumps(row["atomic_units"], ensure_ascii=False),
                }
            )

    total_units = sum(units_per_sample)
    atomic_tokens = [row["atomic_context_tokens"] for row in context_rows]
    raw_atomic_tokens = [row["atomic_evidence_tokens_unformatted"] for row in context_rows]
    b2_tokens = [row["b2_selected_sentence_tokens"] for row in context_rows]
    matched_tokens = [row["token_matched_context_tokens"] for row in context_rows]
    report = {
        "status": "complete",
        "samples": len(frozen),
        "source_sentences": len(units_per_sentence),
        "extractor": extractor.info(),
        "input": {
            "selected_evidence_path": str(selected_evidence_path.resolve()),
            "sha256": hashlib.sha256(selected_evidence_path.read_bytes()).hexdigest(),
            "retrieval_method": "semantic",
            "retrieval_k": 3,
            "reference_caption_exposed": False,
            "generated_caption_exposed": False,
            "full_article_exposed": False,
        },
        "context_budget": context_budget,
        "units_per_sentence": _distribution(units_per_sentence),
        "units_per_sample": _distribution(units_per_sample),
        "evidence_token_length": _distribution(evidence_token_lengths),
        "b2_selected_sentence_tokens": _distribution(b2_tokens),
        "atomic_context_tokens": _distribution(atomic_tokens),
        "atomic_evidence_tokens_unformatted": _distribution(raw_atomic_tokens),
        "token_matched_atomic_tokens": _distribution(matched_tokens),
        "mean_atomic_to_b2_token_ratio": statistics.fmean(
            atomic / b2 if b2 else 0.0
            for atomic, b2 in zip(atomic_tokens, b2_tokens, strict=True)
        ),
        "samples_atomic_over_384": sum(value > context_budget for value in atomic_tokens),
        "samples_with_units_dropped_at_384": sum(
            row["atomic_context_dropped_unit_count"] > 0 for row in context_rows
        ),
        "units_dropped_at_384_total": sum(
            row["atomic_context_dropped_unit_count"] for row in context_rows
        ),
        "token_matched_units_dropped_total": sum(
            row["token_matched_dropped_unit_count"] for row in context_rows
        ),
        "type_distribution": dict(sorted(type_counts.items())),
        "exact_duplicate_rate": exact_duplicate_units / total_units if total_units else 0.0,
        "near_duplicate_rate": near_duplicate_units / total_units if total_units else 0.0,
        "duplicate_evidence_rate": duplicate_units / total_units if total_units else 0.0,
        "duplicate_rate_definition": "fraction of units participating in an exact/SequenceMatcher>=0.90 within-sample pair",
        "malformed_units": malformed_units,
        "malformed_extraction_attempts": malformed_extraction_attempts,
        "empty_units": empty_units,
        "units_without_source_span": missing_provenance,
        "units_without_provenance": 0,
        "suspicious_flags": dict(sorted(suspicious_counts.items())),
        "cache": {"path": str(cache_path.resolve()), "hits": cache_hits, "misses": cache_misses},
        "manual_review": {
            "requested": manual_review_count,
            "included": len(review_rows),
            "jsonl": str((output_dir / "manual_review_100.jsonl").resolve()),
            "csv": str((output_dir / "manual_review_100.csv").resolve()),
            "labels_populated_automatically": False,
        },
    }
    (output_dir / "atomic_evidence_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
