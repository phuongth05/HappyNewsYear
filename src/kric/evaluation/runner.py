"""Unified evaluation runner for all experiment families."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import EvaluationMetric
from .io import file_sha256, write_json, write_per_sample_csv
from .records import EvaluationRecord


class EvaluationRunner:
    def __init__(self, metrics: Sequence[EvaluationMetric]):
        names = [metric.name for metric in metrics]
        if len(names) != len(set(names)):
            raise ValueError(f"metric plugin names must be unique: {names}")
        self.metrics = tuple(metrics)

    def run(
        self,
        records: list[EvaluationRecord],
        *,
        predictions_path: str | Path,
        summary_path: str | Path,
        per_sample_path: str | Path,
    ) -> tuple[dict[str, Any], bool]:
        aggregate: dict[str, Any] = {}
        per_sample: dict[str, dict[str, float | int | None]] = {
            record.sample_id: {} for record in records
        }
        metric_runs: dict[str, Any] = {}
        had_errors = False

        for metric in self.metrics:
            try:
                result = metric.evaluate(records)
                status = result.details.pop("status", "ok")
                for key, value in result.summary.items():
                    if key in aggregate:
                        raise ValueError(f"duplicate aggregate metric field: {key}")
                    aggregate[key] = value
                for sample_id, values in result.per_sample.items():
                    if sample_id not in per_sample:
                        raise ValueError(f"metric {metric.name} returned unknown sample ID {sample_id}")
                    overlap = set(per_sample[sample_id]) & set(values)
                    if overlap:
                        raise ValueError(
                            f"metric {metric.name} returned duplicate per-sample fields: {sorted(overlap)}"
                        )
                    per_sample[sample_id].update(values)
                metric_runs[metric.name] = {
                    "status": status,
                    "summary_fields": sorted(result.summary),
                    **result.details,
                }
            except Exception as error:  # preserve partial results and a reproducible failure record
                had_errors = True
                metric_runs[metric.name] = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }

        csv_columns = write_per_sample_csv(per_sample_path, records, per_sample)
        prediction_source = Path(predictions_path).resolve()
        summary = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "predictions": {
                "path": str(prediction_source),
                "sha256": file_sha256(prediction_source),
                "samples": len(records),
            },
            "metrics": aggregate,
            "metric_runs": metric_runs,
            "per_sample": {
                "path": str(Path(per_sample_path).resolve()),
                "columns": csv_columns,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": _package_versions(
                    ("torch", "transformers", "pycocoevalcap", "spacy", "Pillow")
                ),
            },
        }
        write_json(summary_path, summary)
        return summary, had_errors


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    packages: dict[str, str] = {}
    for name in names:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    return packages
