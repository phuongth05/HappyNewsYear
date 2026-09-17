"""GoodNews adapter for the official article/caption and image-split files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .base import DatasetAdapter, DatasetSplit
from .schema import DatasetSample
from .sentences import segment_sentences


_OFFICIAL_SPLITS = {"train", "val", "test"}
_DEFAULT_ARTICLE_FIELDS = ("article", "article_text", "text", "body")
_DEFAULT_IMAGES_FIELDS = ("images",)
_DEFAULT_CAPTION_FIELDS = ("caption", "raw", "text")
_DEFAULT_IMAGE_PATH_FIELDS = ("image_path", "filename", "filepath")
_DEFAULT_ENTITY_FIELDS = ("entities", "named_entities")
_DEFAULT_METADATA_FIELDS = ("headline", "url", "web_url", "date", "pub_date", "section")


def _as_tuple(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("field aliases must be a list of strings")
    result = tuple(str(item) for item in value)
    if not result:
        raise ValueError("field alias lists must not be empty")
    return result


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True, slots=True)
class GoodNewsConfig:
    """Paths and field aliases for released variants of GoodNews JSON files."""

    annotations_path: Path
    splits_path: Path
    images_root: Path
    image_extension: str = ".jpg"
    long_article_words: int = 2500
    max_issue_examples: int = 20
    article_fields: tuple[str, ...] = _DEFAULT_ARTICLE_FIELDS
    images_fields: tuple[str, ...] = _DEFAULT_IMAGES_FIELDS
    caption_fields: tuple[str, ...] = _DEFAULT_CAPTION_FIELDS
    image_path_fields: tuple[str, ...] = _DEFAULT_IMAGE_PATH_FIELDS
    entity_fields: tuple[str, ...] = _DEFAULT_ENTITY_FIELDS
    metadata_fields: tuple[str, ...] = _DEFAULT_METADATA_FIELDS
    entities_source: str = "unknown"

    def __post_init__(self) -> None:
        if not self.image_extension.startswith("."):
            raise ValueError("image_extension must start with '.'")
        if self.long_article_words <= 0:
            raise ValueError("long_article_words must be positive")
        if self.max_issue_examples <= 0:
            raise ValueError("max_issue_examples must be positive")

    @classmethod
    def from_file(cls, path: str | Path) -> "GoodNewsConfig":
        config_path = Path(path).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError("dataset config must contain a YAML/JSON mapping")
        if isinstance(raw.get("goodnews"), Mapping):
            raw = raw["goodnews"]
        elif isinstance(raw.get("dataset"), Mapping):
            raw = raw["dataset"]
        name = str(raw.get("dataset", raw.get("name", "goodnews"))).lower()
        if name != "goodnews":
            raise ValueError(f"expected a GoodNews config, got {name!r}")

        config_dir = config_path.parent
        root = _resolve_path(str(raw.get("root", raw.get("root_dir", "."))), config_dir)

        return cls(
            annotations_path=_resolve_path(
                str(raw.get("annotations_path", "article+caption.json")), root
            ),
            splits_path=_resolve_path(
                str(raw.get("splits_path", "img_splits.json")), root
            ),
            images_root=_resolve_path(str(raw.get("images_root", "images")), root),
            image_extension=str(raw.get("image_extension", ".jpg")),
            long_article_words=int(raw.get("long_article_words", 2500)),
            max_issue_examples=int(raw.get("max_issue_examples", 20)),
            article_fields=_as_tuple(raw.get("article_fields"), _DEFAULT_ARTICLE_FIELDS),
            images_fields=_as_tuple(raw.get("images_fields"), _DEFAULT_IMAGES_FIELDS),
            caption_fields=_as_tuple(raw.get("caption_fields"), _DEFAULT_CAPTION_FIELDS),
            image_path_fields=_as_tuple(raw.get("image_path_fields"), _DEFAULT_IMAGE_PATH_FIELDS),
            entity_fields=_as_tuple(raw.get("entity_fields"), _DEFAULT_ENTITY_FIELDS),
            metadata_fields=_as_tuple(raw.get("metadata_fields"), _DEFAULT_METADATA_FIELDS),
            entities_source=str(raw.get("entities_source", "unknown")),
        )


@dataclass(frozen=True, slots=True)
class _LoadedSample:
    sample: DatasetSample
    split: DatasetSplit | None


def _first(mapping: Mapping[str, Any], fields: Sequence[str]) -> Any | None:
    for field_name in fields:
        if field_name in mapping:
            return mapping[field_name]
    return None


def _normalize_sample_id(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    return Path(text).stem if text else ""


def _normalize_official_split(value: object) -> str:
    split = str(value).strip().lower()
    aliases = {"dev": "val", "valid": "val", "validation": "val"}
    split = aliases.get(split, split)
    if split not in _OFFICIAL_SPLITS:
        raise ValueError(f"unknown GoodNews split label: {value!r}")
    return split


def _project_split(official_split: str) -> DatasetSplit:
    return DatasetSplit.DEV if official_split == "val" else DatasetSplit.parse(official_split)


def _load_split_assignments(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assignments: dict[str, list[str]] = defaultdict(list)

    def add(sample_id: object, split: object) -> None:
        normalized_id = _normalize_sample_id(sample_id)
        if not normalized_id:
            raise ValueError("official split file contains an empty sample ID")
        assignments[normalized_id].append(_normalize_official_split(split))

    if isinstance(raw, Mapping):
        if set(str(key).lower() for key in raw).issubset(_OFFICIAL_SPLITS | {"dev"}):
            for split, sample_ids in raw.items():
                if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, (str, bytes)):
                    raise TypeError(f"split group {split!r} must be a list of sample IDs")
                for sample_id in sample_ids:
                    add(sample_id, split)
        else:
            for sample_id, value in raw.items():
                split = value.get("split") if isinstance(value, Mapping) else value
                add(sample_id, split)
    elif isinstance(raw, list):
        for row in raw:
            if not isinstance(row, Mapping):
                raise TypeError("split list entries must be objects")
            sample_id = _first(row, ("sample_id", "imgid", "image_id", "filename"))
            if sample_id is None or "split" not in row:
                raise ValueError("split entries require an ID and a split")
            add(sample_id, row["split"])
    else:
        raise TypeError("official split file must contain an object or list")
    return dict(assignments)


def _coerce_article(value: Any) -> tuple[str, tuple[str, ...]]:
    if value is None:
        return "", ()
    if isinstance(value, str):
        sentences = tuple(segment_sentences(value))
        return " ".join(value.split()), sentences
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        source_parts = [str(part).strip() for part in value if str(part).strip()]
        sentences = tuple(
            sentence
            for part in source_parts
            for sentence in segment_sentences(part)
        )
        return " ".join(source_parts), sentences
    if isinstance(value, Mapping):
        nested = _first(value, ("text", "article", "body", "full", "paragraphs", "sentences"))
        return _coerce_article(nested)
    return _coerce_article(str(value))


class GoodNewsDataset(DatasetAdapter):
    """Adapter for the official GoodNews article/caption and split artifacts."""

    name = "goodnews"

    def __init__(self, config: GoodNewsConfig):
        self.config = config
        self._assignments = _load_split_assignments(config.splits_path)
        self._loaded = tuple(self._load_annotations())

    @classmethod
    def from_config(cls, path: str | Path) -> "GoodNewsDataset":
        return cls(GoodNewsConfig.from_file(path))

    def _article_records(self, raw: Any) -> Iterator[tuple[str, Mapping[str, Any]]]:
        if isinstance(raw, Mapping):
            for article_id, record in raw.items():
                if not isinstance(record, Mapping):
                    raise TypeError(f"article {article_id!r} must be an object")
                yield str(article_id), record
            return
        if isinstance(raw, list):
            for index, record in enumerate(raw):
                if not isinstance(record, Mapping):
                    raise TypeError(f"article row {index} must be an object")
                article_id = _first(record, ("article_id", "id", "_id"))
                if article_id is None:
                    raise ValueError(f"article row {index} has no article ID")
                yield str(article_id), record
            return
        raise TypeError("GoodNews annotations must contain an article-keyed object or list")

    def _image_entries(self, article_id: str, record: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
        images = _first(record, self.config.images_fields)
        if images is None:
            raise ValueError(f"article {article_id!r} has no images field")
        if isinstance(images, Mapping):
            for image_index, value in images.items():
                yield str(image_index), value
            return
        if isinstance(images, list):
            for index, value in enumerate(images):
                image_index = str(
                    _first(value, ("image_index", "index", "id"))
                    if isinstance(value, Mapping)
                    else index
                )
                yield image_index, value
            return
        raise TypeError(f"article {article_id!r} images must be an object or list")

    def _load_annotations(self) -> Iterator[_LoadedSample]:
        raw = json.loads(self.config.annotations_path.read_text(encoding="utf-8"))
        for article_id, record in self._article_records(raw):
            article_text, article_sentences = _coerce_article(_first(record, self.config.article_fields))
            article_metadata = {
                key: record[key]
                for key in self.config.metadata_fields
                if key in record
            }
            for image_index, image_value in self._image_entries(article_id, record):
                entry = image_value if isinstance(image_value, Mapping) else {}
                caption_value = (
                    _first(entry, self.config.caption_fields)
                    if entry
                    else image_value
                )
                caption = "" if caption_value is None else " ".join(str(caption_value).split())

                explicit_path = _first(entry, self.config.image_path_fields) if entry else None
                explicit_id = _first(entry, ("sample_id", "imgid", "image_id")) if entry else None
                if explicit_id is not None:
                    sample_id = _normalize_sample_id(explicit_id)
                elif explicit_path is not None:
                    sample_id = _normalize_sample_id(explicit_path)
                else:
                    sample_id = f"{article_id}_{image_index}"

                if explicit_path is not None:
                    candidate = Path(str(explicit_path))
                    image_path = candidate if candidate.is_absolute() else self.config.images_root / candidate
                else:
                    image_path = self.config.images_root / f"{sample_id}{self.config.image_extension}"

                split_values = self._assignments.get(sample_id, [])
                official_split = split_values[0] if split_values else None
                project_split = _project_split(official_split) if official_split else None
                entities = _first(entry, self.config.entity_fields) if entry else None
                if entities is None:
                    entities = _first(record, self.config.entity_fields)

                metadata = {
                    "dataset": self.name,
                    "article_id": article_id,
                    "image_index": image_index,
                    "official_split": official_split,
                    "entities_source": self.config.entities_source,
                    **article_metadata,
                }
                for key in self.config.metadata_fields:
                    if entry and key in entry:
                        metadata[key] = entry[key]

                yield _LoadedSample(
                    sample=DatasetSample(
                        sample_id=sample_id,
                        image_path=str(image_path.resolve()),
                        article_text=article_text,
                        article_sentences=article_sentences,
                        reference_caption=caption,
                        entities=entities,
                        metadata=metadata,
                    ),
                    split=project_split,
                )

    def iter_samples(self, split: DatasetSplit | str | None = None) -> Iterator[DatasetSample]:
        requested = DatasetSplit.parse(split) if split is not None else None
        for loaded in self._loaded:
            if requested is None or loaded.split == requested:
                yield loaded.sample

    def split_integrity_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        conflicts = {
            sample_id: splits
            for sample_id, splits in self._assignments.items()
            if len(splits) != 1
        }
        if conflicts:
            examples = list(conflicts.items())[: self.config.max_issue_examples]
            errors.append(f"samples assigned multiple times in official splits: {examples}")

        sample_ids = [loaded.sample.sample_id for loaded in self._loaded]
        counts = Counter(sample_ids)
        duplicates = [sample_id for sample_id, count in counts.items() if count > 1]
        if duplicates:
            errors.append(
                f"duplicate annotation sample IDs: {duplicates[:self.config.max_issue_examples]}"
            )

        unknown = sorted(set(self._assignments) - set(sample_ids))
        if unknown:
            errors.append(
                f"official split IDs missing annotations: {unknown[:self.config.max_issue_examples]}"
            )
        return tuple(errors)
