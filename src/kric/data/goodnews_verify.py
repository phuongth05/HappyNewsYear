"""Release-level validation for GoodNews before model inference."""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .goodnews import GoodNewsConfig, _coerce_article, _first, _normalize_sample_id


class _JSONObject(list):
    """Marker used to preserve duplicate object keys during JSON decoding."""


def _load_json(path: Path) -> tuple[Any, list[str]]:
    import json

    raw = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_JSONObject
    )
    duplicates: list[str] = []

    def convert(value: Any, location: str) -> Any:
        if isinstance(value, _JSONObject):
            result: dict[str, Any] = {}
            seen: set[str] = set()
            for key, child in value:
                key = str(key)
                if key in seen:
                    duplicates.append(f"{location}.{key}")
                seen.add(key)
                result[key] = convert(child, f"{location}.{key}")
            return result
        if isinstance(value, list):
            return [convert(child, f"{location}[{index}]") for index, child in enumerate(value)]
        return value

    return convert(raw, "$"), duplicates


def _quantile(values: list[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _distribution(values: list[int]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3) if values else None,
        "median": round(float(statistics.median(values)), 3) if values else None,
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


@dataclass(frozen=True, slots=True)
class _ReleaseSample:
    sample_id: str
    article_id: str
    article_text: str
    caption: str
    image_path: Path


def _issue(count: int, examples: Sequence[Any], max_examples: int) -> dict[str, Any]:
    return {"count": count, "examples": list(examples[:max_examples])}


def _parse_splits(
    raw: Any, errors: list[str]
) -> tuple[dict[str, list[str]], list[str]]:
    assignments: dict[str, list[str]] = defaultdict(list)
    non_official_labels: list[str] = []

    def add(sample_id: object, split: object) -> None:
        normalized_id = _normalize_sample_id(sample_id)
        label = str(split).strip().lower()
        if not normalized_id:
            errors.append("split file contains an empty image ID")
            return
        if label not in {"train", "val", "test"}:
            non_official_labels.append(label)
            return
        assignments[normalized_id].append(label)

    if isinstance(raw, Mapping):
        first_value = next(iter(raw.values()), None)
        if isinstance(first_value, Mapping) and any(
            str(value).startswith(("http://", "https://"))
            for value in first_value.values()
        ):
            errors.append(
                "img_splits.json appears to contain the article/image URL mapping "
                "(img_urls.json), not train/val/test split assignments"
            )
            return {}, non_official_labels
        keys = {str(key).lower() for key in raw}
        if keys and keys.issubset({"train", "val", "test"}):
            for split, sample_ids in raw.items():
                if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, (str, bytes)):
                    errors.append(f"split group {split!r} is not a list")
                    continue
                for sample_id in sample_ids:
                    add(sample_id, split)
        else:
            for sample_id, value in raw.items():
                split = value.get("split") if isinstance(value, Mapping) else value
                add(sample_id, split)
    elif isinstance(raw, list):
        for index, row in enumerate(raw):
            if not isinstance(row, Mapping):
                errors.append(f"split row {index} is not an object")
                continue
            sample_id = _first(row, ("sample_id", "imgid", "image_id", "filename"))
            if sample_id is None or "split" not in row:
                errors.append(f"split row {index} has no image ID or split")
                continue
            add(sample_id, row["split"])
    else:
        errors.append("split JSON must be an object or list")
    return dict(assignments), non_official_labels


def _release_samples(
    raw: Any, config: GoodNewsConfig, schema_errors: list[str]
) -> tuple[list[_ReleaseSample], dict[str, str], Counter[str]]:
    samples: list[_ReleaseSample] = []
    articles: dict[str, str] = {}
    images_per_article: Counter[str] = Counter()
    if not isinstance(raw, Mapping):
        schema_errors.append("article+caption.json must be an article-keyed object")
        return samples, articles, images_per_article

    for raw_article_id, record in raw.items():
        article_id = str(raw_article_id)
        if not isinstance(record, Mapping):
            schema_errors.append(f"article {article_id!r} is not an object")
            continue
        article_text, _ = _coerce_article(_first(record, config.article_fields))
        articles[article_id] = article_text
        images = _first(record, config.images_fields)
        if isinstance(images, Mapping):
            entries = list(images.items())
        elif isinstance(images, list):
            entries = list(enumerate(images))
        else:
            schema_errors.append(f"article {article_id!r} has no object/list images field")
            continue

        for default_index, image_value in entries:
            entry = image_value if isinstance(image_value, Mapping) else {}
            image_index = str(
                _first(entry, ("image_index", "index", "id"))
                if entry
                and _first(entry, ("image_index", "index", "id")) is not None
                else default_index
            )
            caption_value = _first(entry, config.caption_fields) if entry else image_value
            caption = "" if caption_value is None else " ".join(str(caption_value).split())
            explicit_path = _first(entry, config.image_path_fields) if entry else None
            explicit_id = _first(entry, ("sample_id", "imgid", "image_id")) if entry else None
            sample_id = (
                _normalize_sample_id(explicit_id)
                if explicit_id is not None
                else _normalize_sample_id(explicit_path)
                if explicit_path is not None
                else f"{article_id}_{image_index}"
            )
            if not sample_id:
                schema_errors.append(f"article {article_id!r} image {image_index!r} has empty ID")
                continue
            if explicit_path is None:
                image_path = config.images_root / f"{sample_id}{config.image_extension}"
            else:
                candidate = Path(str(explicit_path))
                image_path = candidate if candidate.is_absolute() else config.images_root / candidate
            samples.append(
                _ReleaseSample(sample_id, article_id, article_text, caption, image_path)
            )
            images_per_article[article_id] += 1
    return samples, articles, images_per_article


def verify_goodnews_release(
    config: GoodNewsConfig,
    *,
    count_tokens: Callable[[str], int] | None = None,
    tokenizer_name: str | None = None,
    context_limit: int = 384,
    verify_images: int | None = None,
    verification_seed: int = 2026,
) -> dict[str, Any]:
    """Validate official artifacts without changing or filtering the dataset."""

    if verify_images is not None and verify_images <= 0:
        raise ValueError("verify_images must be positive when sampling is enabled")
    max_examples = config.max_issue_examples
    sampled_verification = verify_images is not None
    artifacts = {
        "article+caption.json": config.annotations_path,
        "img_splits.json": config.splits_path,
        "images": config.images_root,
    }
    missing_artifacts = [name for name, path in artifacts.items() if not path.exists()]
    report: dict[str, Any] = {
        "dataset": "goodnews",
        "mode": (
            "quick"
            if sampled_verification
            else "check_only"
            if count_tokens is None
            else "validation_and_statistics"
        ),
        "verification_scope": "sampled" if sampled_verification else "full",
        "full_image_verification_completed": False,
        "paths": {name: str(path) for name, path in artifacts.items()},
        "checks": {},
        "issues": {},
        "warnings": {},
        "statistics": {"status": "not_computed_check_only"},
    }
    report["checks"]["required_artifacts_exist"] = not missing_artifacts
    report["issues"]["missing_required_artifacts"] = _issue(
        len(missing_artifacts), missing_artifacts, max_examples
    )
    if missing_artifacts:
        report["valid"] = False
        return report

    parse_errors: list[str] = []
    duplicate_json_keys: list[str] = []
    try:
        annotations, duplicates = _load_json(config.annotations_path)
        duplicate_json_keys.extend(
            f"article+caption.json:{location}" for location in duplicates
        )
    except Exception as error:
        annotations = None
        parse_errors.append(f"article+caption.json: {type(error).__name__}: {error}")
    try:
        splits, duplicates = _load_json(config.splits_path)
        duplicate_json_keys.extend(f"img_splits.json:{location}" for location in duplicates)
    except Exception as error:
        splits = None
        parse_errors.append(f"img_splits.json: {type(error).__name__}: {error}")
    report["checks"]["json_parses"] = not parse_errors
    report["issues"]["json_parse_errors"] = _issue(
        len(parse_errors), parse_errors, max_examples
    )
    report["issues"]["duplicate_json_keys"] = _issue(
        len(duplicate_json_keys), duplicate_json_keys, max_examples
    )
    if parse_errors:
        report["valid"] = False
        return report

    schema_errors: list[str] = []
    samples, articles, images_per_article = _release_samples(
        annotations, config, schema_errors
    )
    assignments, non_official_labels = _parse_splits(splits, schema_errors)
    sample_counts = Counter(sample.sample_id for sample in samples)
    duplicate_sample_ids_all = sorted(
        sample_id for sample_id, count in sample_counts.items() if count > 1
    )
    annotation_ids = set(sample_counts)
    split_ids = set(assignments)
    official_annotation_ids = split_ids & annotation_ids
    annotations_outside_splits = annotation_ids - split_ids
    duplicate_sample_ids_official = sorted(
        sample_id
        for sample_id in duplicate_sample_ids_all
        if sample_id in official_annotation_ids
    )
    duplicate_sample_ids_outside = sorted(
        sample_id
        for sample_id in duplicate_sample_ids_all
        if sample_id in annotations_outside_splits
    )
    split_conflicts = {
        sample_id: labels
        for sample_id, labels in assignments.items()
        if len(labels) != 1
    }
    split_ids_missing_annotations = sorted(split_ids - annotation_ids)
    annotation_ids_missing_splits = sorted(annotations_outside_splits)
    empty_articles_all = sorted(
        sample.sample_id for sample in samples if not sample.article_text.strip()
    )
    empty_captions_all = sorted(
        sample.sample_id for sample in samples if not sample.caption.strip()
    )
    empty_articles_official = [
        sample_id for sample_id in empty_articles_all if sample_id in official_annotation_ids
    ]
    empty_articles_outside = [
        sample_id for sample_id in empty_articles_all if sample_id in annotations_outside_splits
    ]
    empty_captions_official = [
        sample_id for sample_id in empty_captions_all if sample_id in official_annotation_ids
    ]
    empty_captions_outside = [
        sample_id for sample_id in empty_captions_all if sample_id in annotations_outside_splits
    ]

    def counts_by_split(sample_ids: Sequence[str]) -> dict[str, int]:
        ids = set(sample_ids)
        counts = Counter(
            labels[0]
            for sample_id, labels in assignments.items()
            if sample_id in ids and len(labels) == 1
        )
        return {
            "train": counts.get("train", 0),
            "val": counts.get("val", 0),
            "test": counts.get("test", 0),
            "total": len(sample_ids),
        }

    empty_captions_official_by_split = counts_by_split(empty_captions_official)
    empty_articles_official_by_split = counts_by_split(empty_articles_official)

    sample_by_id = {sample.sample_id: sample for sample in samples}
    resolved_split_samples = [
        sample_by_id[sample_id]
        for sample_id in sorted(split_ids & annotation_ids)
    ]
    # Both quick and full verification are scoped to the authoritative
    # experimental population. Extra annotations are reported but do not
    # trigger image I/O or invalidate an otherwise usable official split.
    path_check_samples = resolved_split_samples
    missing_images: list[str] = []
    corrupted_images: list[str] = []
    corruption_errors: list[str] = []
    from PIL import Image

    for sample in path_check_samples:
        if not sample.image_path.is_file():
            missing_images.append(sample.sample_id)
    missing_image_ids = set(missing_images)
    existing_path_check_samples = [
        sample
        for sample in path_check_samples
        if sample.sample_id not in missing_image_ids
    ]
    if sampled_verification:
        sample_size = min(verify_images, len(existing_path_check_samples))
        verification_samples = random.Random(verification_seed).sample(
            existing_path_check_samples, sample_size
        )
    else:
        verification_samples = existing_path_check_samples

    verified_attempts = 0
    for sample in verification_samples:
        verified_attempts += 1
        try:
            with Image.open(sample.image_path) as image:
                image.verify()
        except Exception as error:
            corrupted_images.append(sample.sample_id)
            if len(corruption_errors) < max_examples:
                corruption_errors.append(
                    f"{sample.sample_id}: {type(error).__name__}: {error}"
                )

    issue_values: dict[str, Sequence[Any] | Mapping[str, Any]] = {
        "schema_errors": schema_errors,
        "non_official_split_labels": non_official_labels,
        "duplicate_sample_ids_official_splits": duplicate_sample_ids_official,
        "samples_in_multiple_splits": split_conflicts,
        "split_ids_missing_annotations": split_ids_missing_annotations,
        "empty_articles_official_splits": empty_articles_official,
        "empty_captions_official_splits": empty_captions_official,
        "missing_images": missing_images,
        "corrupted_images": corrupted_images,
    }
    for name, values in issue_values.items():
        examples = list(values.items()) if isinstance(values, Mapping) else list(values)
        report["issues"][name] = _issue(len(values), examples, max_examples)
    report["issues"]["corrupted_images"]["errors"] = corruption_errors
    report["issues"]["empty_articles_official_splits"][
        "by_split"
    ] = empty_articles_official_by_split
    report["issues"]["empty_captions_official_splits"][
        "by_split"
    ] = empty_captions_official_by_split

    warning_values: dict[str, Sequence[Any]] = {
        "annotation_ids_missing_splits": annotation_ids_missing_splits,
        "duplicate_sample_ids_outside_official_splits": duplicate_sample_ids_outside,
        "empty_articles_outside_official_splits": empty_articles_outside,
        "empty_captions_outside_official_splits": empty_captions_outside,
    }
    for name, values in warning_values.items():
        report["warnings"][name] = _issue(len(values), values, max_examples)

    total_image_files = sum(
        1 for path in config.images_root.rglob("*") if path.is_file()
    )
    report["image_verification"] = {
        "verification_scope": "sampled" if sampled_verification else "full",
        "seed": verification_seed if sampled_verification else None,
        "requested_sample_size": verify_images,
        "total_image_files_found": total_image_files,
        "total_split_referenced_images": len(split_ids),
        "resolved_split_referenced_images": len(resolved_split_samples),
        "sampled_images_selected": len(verification_samples),
        "sampled_images_verified": verified_attempts,
        "successfully_verified": verified_attempts - len(corrupted_images),
        "corrupted_images_found_in_sample": len(corrupted_images),
        "missing_referenced_images": len(missing_images),
        "sampled_split_distribution": {
            split: sum(
                assignments[sample.sample_id][0] == split
                for sample in verification_samples
                if len(assignments.get(sample.sample_id, [])) == 1
            )
            for split in ("train", "val", "test")
        },
    }
    report["full_image_verification_completed"] = not sampled_verification

    split_counts = Counter(
        labels[0]
        for sample_id, labels in assignments.items()
        if len(labels) == 1 and sample_id in annotation_ids
    )
    report["checks"].update(
        {
            "official_split_labels_only": not non_official_labels,
            "all_split_ids_resolve_to_annotations": not split_ids_missing_annotations,
            "official_samples_have_one_split": not split_conflicts,
            "official_sample_ids_unique": not duplicate_sample_ids_official
            and not duplicate_json_keys,
            "official_captions_non_empty": not empty_captions_official,
            "official_articles_non_empty": not empty_articles_official,
            "all_official_images_exist": not missing_images,
            (
                "sampled_images_decode"
                if sampled_verification
                else "all_images_decode"
            ): not corrupted_images,
        }
    )
    report["release_counts"] = {
        "total_annotation_samples": len(samples),
        "official_split_samples": len(split_ids),
        "annotations_outside_official_splits": len(annotations_outside_splits),
        "unique_articles": len(articles),
        "samples_by_official_split": {
            split: split_counts.get(split, 0) for split in ("train", "val", "test")
        },
    }
    report["content_validation"] = {
        "empty_captions_all_annotations": len(empty_captions_all),
        "empty_captions_official_splits": empty_captions_official_by_split,
        "empty_captions_outside_official_splits": len(empty_captions_outside),
        "empty_caption_partition_consistent": len(empty_captions_all)
        == len(empty_captions_official) + len(empty_captions_outside),
        "empty_articles_all_annotations": len(empty_articles_all),
        "empty_articles_official_splits": empty_articles_official_by_split,
        "empty_articles_outside_official_splits": len(empty_articles_outside),
        "empty_article_partition_consistent": len(empty_articles_all)
        == len(empty_articles_official) + len(empty_articles_outside),
    }

    if count_tokens is not None:
        article_tokens = {
            article_id: count_tokens(text) for article_id, text in articles.items()
        }
        caption_lengths = [count_tokens(sample.caption) for sample in samples]
        per_sample_article_lengths = [
            article_tokens[sample.article_id] for sample in samples
        ]
        unique_article_lengths = list(article_tokens.values())
        report["statistics"] = {
            "status": "computed_from_provided_release",
            "tokenizer": tokenizer_name,
            "context_limit_tokens": context_limit,
            "article_token_length_unique_articles": _distribution(unique_article_lengths),
            "caption_token_length_samples": _distribution(caption_lengths),
            "images_per_article": _distribution(list(images_per_article.values())),
            "percentage_unique_articles_truncated_at_context_limit": round(
                100.0
                * sum(length > context_limit for length in unique_article_lengths)
                / len(unique_article_lengths),
                6,
            )
            if unique_article_lengths
            else None,
            "percentage_sample_contexts_truncated_at_context_limit": round(
                100.0
                * sum(length > context_limit for length in per_sample_article_lengths)
                / len(per_sample_article_lengths),
                6,
            )
            if per_sample_article_lengths
            else None,
        }

    fatal_issue_count = sum(details["count"] for details in report["issues"].values())
    warning_count = sum(details["count"] for details in report["warnings"].values())
    report["fatal_issue_count"] = fatal_issue_count
    report["warning_count"] = warning_count
    report["valid"] = fatal_issue_count == 0 and all(report["checks"].values())
    return report
