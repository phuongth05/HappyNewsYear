import json
from pathlib import Path

import pytest

from kric.evaluation.io import load_predictions


def test_load_jsonl_predictions(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    rows = [
        {"sample_id": "a", "prediction": "Alice spoke.", "reference": "Alice spoke.", "metadata": {}},
        {"sample_id": "b", "prediction": "Bob left.", "reference": "Bob stayed.", "metadata": {}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    records = load_predictions(path)
    assert [record.sample_id for record in records] == ["a", "b"]
    assert records[0].prediction == "Alice spoke."


def test_duplicate_prediction_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps(
            [
                {"sample_id": "same", "prediction": "one", "reference": "one"},
                {"sample_id": "same", "prediction": "two", "reference": "two"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate prediction sample_id"):
        load_predictions(path)

