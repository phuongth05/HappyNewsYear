"""Narrow model-facing types for caption generation."""

from __future__ import annotations

from dataclasses import dataclass


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
class GeneratedCaption:
    sample_id: str
    text: str

