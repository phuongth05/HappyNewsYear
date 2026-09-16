import csv
import json
from pathlib import Path

from kric.evaluation.deferred import deferred_metrics
from kric.evaluation.entities import EntityMetric, MetadataEntityExtractor
from kric.evaluation.io import load_predictions
from kric.evaluation.runner import EvaluationRunner


def test_runner_writes_summary_and_aligned_per_sample_csv(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    row = {
        "sample_id": "sample-1",
        "prediction": "Alice spoke in Paris.",
        "reference": "Alice spoke in Paris.",
        "metadata": {
            "prediction_entities": ["Alice", "Paris"],
            "reference_entities": ["Alice", "Paris"],
        },
    }
    predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary_path = tmp_path / "metrics.json"
    csv_path = tmp_path / "metrics_per_sample.csv"
    records = load_predictions(predictions)
    runner = EvaluationRunner(
        [
            EntityMetric(MetadataEntityExtractor()),
            deferred_metrics()["unsupported_claim_rate"],
        ]
    )

    summary, had_errors = runner.run(
        records,
        predictions_path=predictions,
        summary_path=summary_path,
        per_sample_path=csv_path,
    )

    assert had_errors is False
    assert summary["metrics"]["EntityF1"] == 1.0
    assert summary["metric_runs"]["unsupported_claim_rate"]["status"] == "deferred"
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["predictions"]["samples"] == 1
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sample_id"] == "sample-1"
    assert float(rows[0]["EntityF1"]) == 1.0

