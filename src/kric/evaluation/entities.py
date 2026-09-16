"""Named-entity precision, recall, and F1 with pluggable extraction."""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .base import EvaluationMetric
from .records import EvaluationRecord, MetricResult, MetricUnavailableError


@dataclass(frozen=True, slots=True)
class EntityMention:
    text: str
    label: str | None = None


class EntityExtractor(ABC):
    @abstractmethod
    def extract(self, text: str, metadata: dict[str, Any], role: str) -> list[EntityMention]:
        """Extract mentions for ``prediction`` or ``reference``."""


class MetadataEntityExtractor(EntityExtractor):
    """Read precomputed mentions from prediction/reference metadata fields."""

    def __init__(
        self,
        prediction_key: str = "prediction_entities",
        reference_key: str = "reference_entities",
    ):
        self.keys = {"prediction": prediction_key, "reference": reference_key}

    def extract(self, text: str, metadata: dict[str, Any], role: str) -> list[EntityMention]:
        key = self.keys[role]
        if key not in metadata:
            raise MetricUnavailableError(
                f"entity metric requested but metadata.{key} is missing; use --entity-extractor spacy or provide annotations"
            )
        raw = metadata[key]
        if not isinstance(raw, list):
            raise TypeError(f"metadata.{key} must be a list")
        mentions: list[EntityMention] = []
        for item in raw:
            if isinstance(item, str):
                mentions.append(EntityMention(item))
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                label = item.get("label")
                mentions.append(EntityMention(item["text"], str(label) if label is not None else None))
            else:
                raise TypeError(f"invalid entity annotation in metadata.{key}: {item!r}")
        return mentions


class SpacyEntityExtractor(EntityExtractor):
    """Extract entities directly from captions using a fixed spaCy model."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            import spacy
        except ImportError as error:
            raise MetricUnavailableError("spaCy extraction requires `pip install -e .[ner]`") from error
        try:
            self.nlp = spacy.load(model_name)
        except OSError as error:
            raise MetricUnavailableError(
                f"spaCy model {model_name!r} is unavailable; install that exact model before evaluation"
            ) from error
        self.model_name = model_name

    def extract(self, text: str, metadata: dict[str, Any], role: str) -> list[EntityMention]:
        return [EntityMention(entity.text, entity.label_) for entity in self.nlp(text).ents]


def normalize_entity(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("’", "'")
    normalized = re.sub(r"(?:'s)\b", "", normalized)
    normalized = re.sub(r"[^\w\s'-]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _prf(true_positive: int, predicted: int, reference: int) -> tuple[float, float, float]:
    precision = true_positive / predicted if predicted else float(reference == 0)
    recall = true_positive / reference if reference else float(predicted == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


class EntityMetric(EvaluationMetric):
    name = "entity"

    def __init__(self, extractor: EntityExtractor, *, match_labels: bool = False):
        self.extractor = extractor
        self.match_labels = match_labels

    def _key(self, mention: EntityMention) -> tuple[str, str | None]:
        text = normalize_entity(mention.text)
        label = mention.label.casefold() if self.match_labels and mention.label else None
        return text, label

    def evaluate(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        total_tp = total_predicted = total_reference = 0
        per_sample: dict[str, dict[str, float | int | None]] = {}
        for record in records:
            predicted_mentions = self.extractor.extract(record.prediction, record.metadata, "prediction")
            reference_mentions = self.extractor.extract(record.reference, record.metadata, "reference")
            predicted = Counter(self._key(mention) for mention in predicted_mentions if normalize_entity(mention.text))
            reference = Counter(self._key(mention) for mention in reference_mentions if normalize_entity(mention.text))
            true_positive = sum((predicted & reference).values())
            predicted_count = sum(predicted.values())
            reference_count = sum(reference.values())
            precision, recall, f1 = _prf(true_positive, predicted_count, reference_count)
            total_tp += true_positive
            total_predicted += predicted_count
            total_reference += reference_count
            per_sample[record.sample_id] = {
                "EntityPrecision": precision,
                "EntityRecall": recall,
                "EntityF1": f1,
                "EntityTP": true_positive,
                "EntityPredicted": predicted_count,
                "EntityReference": reference_count,
            }

        precision, recall, f1 = _prf(total_tp, total_predicted, total_reference)
        return MetricResult(
            summary={
                "EntityPrecision": precision,
                "EntityRecall": recall,
                "EntityF1": f1,
                "EntityTP": total_tp,
                "EntityPredicted": total_predicted,
                "EntityReference": total_reference,
            },
            per_sample=per_sample,
            details={
                "matching": "NFKC + casefold + whitespace/punctuation normalization; exact mention matching",
                "match_labels": self.match_labels,
                "extractor": type(self.extractor).__name__,
            },
        )

