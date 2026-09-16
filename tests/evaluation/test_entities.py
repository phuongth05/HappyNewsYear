import pytest

from kric.evaluation.entities import EntityMetric, MetadataEntityExtractor, normalize_entity
from kric.evaluation.records import EvaluationRecord


def test_entity_micro_metrics_and_per_sample_scores() -> None:
    records = [
        EvaluationRecord(
            sample_id="a",
            prediction="Alice visited Paris.",
            reference="Alice visited London.",
            metadata={
                "prediction_entities": ["Alice", "Paris"],
                "reference_entities": ["Alice", "London"],
            },
        ),
        EvaluationRecord(
            sample_id="b",
            prediction="Bob spoke.",
            reference="Bob spoke.",
            metadata={
                "prediction_entities": [{"text": "Bob", "label": "PERSON"}],
                "reference_entities": [{"text": "Bob", "label": "PERSON"}],
            },
        ),
    ]
    result = EntityMetric(MetadataEntityExtractor()).evaluate(records)

    assert result.summary["EntityPrecision"] == pytest.approx(2 / 3)
    assert result.summary["EntityRecall"] == pytest.approx(2 / 3)
    assert result.summary["EntityF1"] == pytest.approx(2 / 3)
    assert result.per_sample["a"]["EntityF1"] == pytest.approx(0.5)
    assert result.per_sample["b"]["EntityF1"] == pytest.approx(1.0)


def test_entity_normalization_is_case_and_possessive_insensitive() -> None:
    assert normalize_entity("  NEW York’s  ") == "new york"


def test_both_empty_entity_sets_score_one() -> None:
    record = EvaluationRecord(
        sample_id="empty",
        prediction="A landscape.",
        reference="A landscape.",
        metadata={"prediction_entities": [], "reference_entities": []},
    )
    result = EntityMetric(MetadataEntityExtractor()).evaluate([record])
    assert result.summary["EntityPrecision"] == 1.0
    assert result.summary["EntityRecall"] == 1.0
    assert result.summary["EntityF1"] == 1.0

