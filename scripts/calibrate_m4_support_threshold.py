"""Calibrate frozen M4 cosine support thresholds with stratified held-out CV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.evaluation.io import file_sha256, write_json  # noqa: E402
from kric.matching.calibration import (  # noqa: E402
    OBJECTIVES,
    cross_validate_thresholds,
    join_calibration_rows,
    select_threshold,
)


FOLD_FIELDS = [
    "objective",
    "fold",
    "sample_id",
    "claim_id",
    "caption_variant",
    "cosine_top1_score",
    "gold_label",
    "gold_support_label",
    "threshold",
    "prediction",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _write_fold_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FOLD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--review-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--freeze-objective", choices=OBJECTIVES)
    args = parser.parse_args()
    annotation_path = args.annotations.expanduser().resolve()
    review_path = args.review_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (annotation_path, review_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    examples, dataset_audit = join_calibration_rows(
        _read_csv(annotation_path), _read_csv(review_path)
    )
    reports, fold_rows = cross_validate_thresholds(
        examples, n_splits=5, seed=2026, bootstrap_resamples=10_000
    )
    full_data_fit = {
        objective: select_threshold(examples, objective) for objective in OBJECTIVES
    }
    selected = full_data_fit.get(args.freeze_objective) if args.freeze_objective else None
    if selected is not None and not selected["constraint_met"]:
        raise ValueError(
            f"cannot freeze infeasible objective {args.freeze_objective}: "
            "no positive-prediction threshold satisfies its constraint"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "calibration_report.json"
    predictions_path = output_dir / "fold_predictions.csv"
    manifest_path = output_dir / "threshold_manifest.json"
    report = {
        "schema_version": 1,
        "experiment": "M4_cosine_support_threshold_calibration",
        "protocol": {
            "score": "cosine_top1_similarity",
            "binary_labels": {"supported": 1, "unsupported": 0},
            "excluded_label": "uncertain",
            "folds": 5,
            "stratification": "binary_support_label",
            "seed": 2026,
            "threshold_comparison": "score >= threshold",
            "selection_scope": "training_folds_only",
            "heldout_examples_used_for_threshold_selection": False,
            "bootstrap_resamples": 10_000,
            "bootstrap_confidence": 0.95,
        },
        "data": dataset_audit,
        "objectives": reports,
        "full_binary_data_fit_after_cv_reporting": full_data_fit,
        "metric_notes": {
            "auprc": "non-interpolated average precision over tied score groups",
            "cv_metrics": "computed from one held-out prediction per included claim",
            "deployment_fit": "not used as a CV performance estimate",
        },
    }
    write_json(report_path, report)
    _write_fold_predictions(predictions_path, fold_rows)
    manifest = {
        "schema_version": 1,
        "experiment": "M4_cosine_support_threshold_calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_status": "frozen" if selected is not None else "reported_not_frozen",
        "source_artifacts": {
            "annotations": {
                "path": str(annotation_path),
                "sha256": file_sha256(annotation_path),
            },
            "review_csv": {
                "path": str(review_path),
                "sha256": file_sha256(review_path),
            },
        },
        "calibration": {
            "total_claims": dataset_audit["total_frozen_claims"],
            "included_binary_claims": dataset_audit["included_binary_claims"],
            "excluded_uncertain_claims": dataset_audit["excluded_uncertain_claims"],
            "folds": 5,
            "seed": 2026,
            "objectives_reported": list(OBJECTIVES),
            "selected_objective": args.freeze_objective,
            "selected_threshold": selected["threshold"] if selected else None,
            "selected_threshold_fit_scope": "all non-uncertain calibration claims" if selected else None,
            "cv_performance_source": "held-out fold predictions",
        },
        "outputs": {
            "calibration_report.json": file_sha256(report_path),
            "fold_predictions.csv": file_sha256(predictions_path),
        },
        "matcher": "cosine",
        "nli_used_for_threshold": False,
        "caption_generation_gold_created": False,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
