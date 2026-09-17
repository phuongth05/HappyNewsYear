"""Canonical sample types.

``DatasetSample`` is the storage/evaluation representation.  Model and
evidence code must consume ``InferenceSample`` instead; that type deliberately
has no reference-caption or annotation field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _require_string(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class InferenceSample:
    """The only sample view intended for evidence selection and generation."""

    sample_id: str
    image_path: str
    article_text: str
    article_sentences: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("sample_id", "image_path", "article_text"):
            _require_string(name, getattr(self, name))
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if not isinstance(self.article_sentences, tuple) or not all(
            isinstance(sentence, str) for sentence in self.article_sentences
        ):
            raise TypeError("article_sentences must be tuple[str, ...]")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be dict")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["article_sentences"] = list(self.article_sentences)
        return value


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """Canonical stored sample, including evaluation-only fields."""

    sample_id: str
    image_path: str
    article_text: str
    article_sentences: tuple[str, ...]
    reference_caption: str
    entities: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("sample_id", "image_path", "article_text", "reference_caption"):
            _require_string(name, getattr(self, name))
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if not isinstance(self.article_sentences, tuple) or not all(
            isinstance(sentence, str) for sentence in self.article_sentences
        ):
            raise TypeError("article_sentences must be tuple[str, ...]")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be dict")

    def to_inference_sample(self) -> InferenceSample:
        """Return a reference-free view for evidence selection and models.

        ``entities`` is also omitted because legacy entity fields may have been
        extracted from the target caption. Article-derived entities can be
        introduced later under an explicitly provenance-tagged field.
        """

        return InferenceSample(
            sample_id=self.sample_id,
            image_path=self.image_path,
            article_text=self.article_text,
            article_sentences=self.article_sentences,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["article_sentences"] = list(self.article_sentences)
        return value

