"""Atomic evidence extraction and auditing."""

from .schema import AtomicEvidence, FrozenSentence, SourceSpan
from .spacy_extractor import SpacyAtomicExtractor

__all__ = ["AtomicEvidence", "FrozenSentence", "SourceSpan", "SpacyAtomicExtractor"]

