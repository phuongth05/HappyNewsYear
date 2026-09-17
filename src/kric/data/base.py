"""Dataset adapter contract shared by all current and future datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum

from .schema import DatasetSample, InferenceSample


class DatasetSplit(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"

    @classmethod
    def parse(cls, value: str | "DatasetSplit") -> "DatasetSplit":
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower()
        if normalized in {"val", "valid", "validation"}:
            normalized = "dev"
        return cls(normalized)


class SplitIntegrityError(ValueError):
    """Raised when official split assignments overlap or are incomplete."""


class DatasetAdapter(ABC):
    """Stable interface model code can use across dataset implementations."""

    name: str

    @abstractmethod
    def iter_samples(self, split: DatasetSplit | str | None = None) -> Iterator[DatasetSample]:
        """Iterate canonical samples; ``None`` means every official split."""

    @abstractmethod
    def split_integrity_errors(self) -> tuple[str, ...]:
        """Return split assignment errors without hiding them from an audit."""

    def assert_split_integrity(self) -> None:
        errors = self.split_integrity_errors()
        if errors:
            raise SplitIntegrityError("; ".join(errors))

    def iter_inference_samples(self, split: DatasetSplit | str) -> Iterator[InferenceSample]:
        """Yield only inference-safe fields after validating split integrity."""

        self.assert_split_integrity()
        for sample in self.iter_samples(DatasetSplit.parse(split)):
            yield sample.to_inference_sample()

