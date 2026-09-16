"""Saved-prediction evaluation and paired statistical comparison."""

from .bootstrap import paired_bootstrap_ci
from .entities import EntityMetric, MetadataEntityExtractor, SpacyEntityExtractor
from .io import load_predictions
from .records import EvaluationRecord
from .runner import EvaluationRunner

__all__ = [
    "EntityMetric",
    "EvaluationRecord",
    "EvaluationRunner",
    "MetadataEntityExtractor",
    "SpacyEntityExtractor",
    "load_predictions",
    "paired_bootstrap_ci",
]

