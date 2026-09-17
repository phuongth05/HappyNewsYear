"""Unified dataset interfaces and dataset-specific adapters."""

from .base import DatasetAdapter, DatasetSplit, SplitIntegrityError
from .goodnews import GoodNewsConfig, GoodNewsDataset
from .schema import DatasetSample, InferenceSample

__all__ = [
    "DatasetAdapter",
    "DatasetSample",
    "DatasetSplit",
    "GoodNewsConfig",
    "GoodNewsDataset",
    "InferenceSample",
    "SplitIntegrityError",
]

