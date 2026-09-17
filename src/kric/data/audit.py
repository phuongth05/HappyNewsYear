"""Dataset quality checks and descriptive statistics."""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .base import DatasetAdapter, DatasetSplit


_WORD_RE = re.compile(r"\b\w+(?:['’-]\w+)?\b", flags=re.UNICODE)


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _distribution(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None, "p95": None}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 3),
        "median": round(float(statistics.median(ordered)), 3),
        "p95": ordered[p95_index],
    }


def _entity_count(entities: Any) -> int | None:
    if entities is None:
        return None
    if isinstance(entities, (list, tuple, set)):
        return len(entities)
    if isinstance(entities, dict):
        if isinstance(entities.get("count"), int):
            return int(entities["count"])
        if isinstance(entities.get("mentions"), list):
            return len(entities["mentions"])
        return sum(
            count
            for value in entities.values()
            if (count := _entity_count(value)) is not None
        )
    if isinstance(entities, str):
        return int(bool(entities.strip()))
    return 1


def audit_dataset(
    dataset: DatasetAdapter,
    *,
    split: DatasetSplit | str | None = None,
    long_article_words: int = 2500,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Audit one dataset without dropping invalid records from the report."""

    requested = DatasetSplit.parse(split) if split is not None else None
    samples = list(dataset.iter_samples(requested))
    issue_ids: dict[str, list[str]] = defaultdict(list)
    split_counts: Counter[str] = Counter()
    article_lengths: list[int] = []
    sentence_counts: list[int] = []
    caption_lengths: list[int] = []
    entity_counts: list[int] = []
    entity_caption_words = 0

    ids: Counter[str] = Counter(sample.sample_id for sample in samples)
    for sample_id, count in ids.items():
        if count > 1:
            issue_ids["duplicate_sample_ids"].append(sample_id)

    for sample in samples:
        official_split = sample.metadata.get("official_split")
        split_counts[str(official_split or "unassigned")] += 1
        article_words = count_words(sample.article_text)
        caption_words = count_words(sample.reference_caption)
        article_lengths.append(article_words)
        sentence_counts.append(len(sample.article_sentences))
        caption_lengths.append(caption_words)

        if not Path(sample.image_path).is_file():
            issue_ids["missing_images"].append(sample.sample_id)
        if not sample.article_text.strip():
            issue_ids["empty_articles"].append(sample.sample_id)
        if not sample.reference_caption.strip():
            issue_ids["empty_captions"].append(sample.sample_id)
        if article_words > long_article_words:
            issue_ids["unusually_long_articles"].append(sample.sample_id)

        entity_count = _entity_count(sample.entities)
        if entity_count is not None:
            entity_counts.append(entity_count)
            entity_caption_words += caption_words

    issue_names = (
        "missing_images",
        "empty_articles",
        "empty_captions",
        "duplicate_sample_ids",
        "unusually_long_articles",
    )
    issues = {
        name: {
            "count": len(issue_ids[name]),
            "sample_ids": issue_ids[name][:max_examples],
        }
        for name in issue_names
    }

    split_errors = list(dataset.split_integrity_errors())
    issues["split_integrity"] = {
        "count": len(split_errors),
        "messages": split_errors[:max_examples],
    }

    entity_statistics: dict[str, Any]
    if entity_counts:
        entity_statistics = {
            "available": True,
            "samples_with_entities": len(entity_counts),
            "sample_coverage": round(len(entity_counts) / len(samples), 6) if samples else 0.0,
            "mentions": _distribution(entity_counts),
            "density_per_caption_word": round(
                sum(entity_counts) / entity_caption_words, 6
            ) if entity_caption_words else None,
        }
    else:
        entity_statistics = {
            "available": False,
            "samples_with_entities": 0,
            "sample_coverage": 0.0,
            "mentions": _distribution([]),
            "density_per_caption_word": None,
        }

    return {
        "dataset": dataset.name,
        "requested_split": requested.value if requested else "all",
        "number_of_samples": len(samples),
        "split_counts": dict(sorted(split_counts.items())),
        "thresholds": {"unusually_long_article_words": long_article_words},
        "issues": issues,
        "statistics": {
            "article_length_words": _distribution(article_lengths),
            "sentence_count": _distribution(sentence_counts),
            "caption_length_words": _distribution(caption_lengths),
            "entity_density": entity_statistics,
        },
    }

