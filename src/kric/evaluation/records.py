"""Types shared by evaluation metrics and artifact readers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    sample_id: str
    prediction: str
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("sample_id", "prediction", "reference"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be str, got {type(value).__name__}")
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be dict")


@dataclass(slots=True)
class MetricResult:
    """A metric's aggregate fields and optional aligned per-sample fields."""

    summary: dict[str, Any]
    per_sample: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class MetricUnavailableError(RuntimeError):
    """Raised when a requested metric lacks a required runtime or input."""

