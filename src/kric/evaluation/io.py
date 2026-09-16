"""Prediction artifact loading and result serialization."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .records import EvaluationRecord


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(path: str | Path) -> list[EvaluationRecord]:
    """Load a JSON array or JSONL file with strict ID/schema validation."""

    prediction_path = Path(path)
    if prediction_path.suffix.lower() == ".jsonl":
        rows: Any = []
        with prediction_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
    else:
        rows = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("prediction artifact must contain a JSON array or JSONL objects")

    records: list[EvaluationRecord] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"prediction row {index} must be an object")
        missing = [key for key in ("sample_id", "prediction", "reference") if key not in row]
        if missing:
            raise ValueError(f"prediction row {index} is missing fields: {missing}")
        record = EvaluationRecord(
            sample_id=row["sample_id"],
            prediction=row["prediction"],
            reference=row["reference"],
            metadata=row.get("metadata", {}),
        )
        if record.sample_id in seen:
            raise ValueError(f"duplicate prediction sample_id: {record.sample_id}")
        seen.add(record.sample_id)
        records.append(record)
    if not records:
        raise ValueError("prediction artifact contains no samples")
    return records


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)


def write_per_sample_csv(
    path: str | Path,
    records: list[EvaluationRecord],
    per_sample: dict[str, dict[str, float | int | None]],
) -> list[str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_columns = sorted({key for values in per_sample.values() for key in values})
    columns = ["sample_id", "prediction", "reference", *metric_columns, "metadata_json"]
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            metric_values = per_sample.get(record.sample_id, {})
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "prediction": record.prediction,
                    "reference": record.reference,
                    **metric_values,
                    "metadata_json": json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                }
            )
    temporary.replace(output)
    return columns

