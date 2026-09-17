"""Caption-generation interfaces and experiment runners."""

from .config import B0ExperimentConfig, B1ExperimentConfig, B2ExperimentConfig, PromptConfig
from .types import FullArticleInput, GeneratedCaption, ImageOnlyInput

__all__ = [
    "B0ExperimentConfig",
    "B1ExperimentConfig",
    "B2ExperimentConfig",
    "FullArticleInput",
    "GeneratedCaption",
    "ImageOnlyInput",
    "PromptConfig",
]
