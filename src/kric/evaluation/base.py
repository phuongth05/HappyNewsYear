"""Metric plugin interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .records import EvaluationRecord, MetricResult


class EvaluationMetric(ABC):
    """One evaluation stage that may emit several related metric fields."""

    name: str

    @abstractmethod
    def evaluate(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        """Evaluate records without invoking a caption-generation model."""

