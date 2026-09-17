"""Narrow model-facing types for caption generation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ImageOnlyInput:
    """The complete input contract for B0.

    Article, reference-caption, entity, and metadata fields are deliberately
    impossible to pass through this object.
    """

    sample_id: str
    image_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if not isinstance(self.image_path, str) or not self.image_path.strip():
            raise ValueError("image_path must be a non-empty string")


@dataclass(frozen=True, slots=True)
class FullArticleInput:
    """The complete model-facing B1 input, without evaluation-only fields."""

    sample_id: str
    image_path: str
    article_text: str

    def __post_init__(self) -> None:
        for name in ("sample_id", "image_path", "article_text"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        if not self.sample_id.strip() or not self.image_path.strip():
            raise ValueError("sample_id and image_path must not be empty")


@dataclass(frozen=True, slots=True)
class GeneratedCaption:
    sample_id: str
    text: str
    context_stats: dict[str, float | int] = field(default_factory=dict)
