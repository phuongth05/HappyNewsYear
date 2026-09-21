"""Enrich the frozen M4 review CSV with exact parent-sentence provenance."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.evaluation.io import file_sha256, write_json  # noqa: E402
from kric.matching.review_context import (  # noqa: E402
    ENRICHMENT_FIELDS,
    HUMAN_FIELDS,
    build_frozen_indexes,
    enrich_review_rows,
    read_jsonl,
    read_review_csv,
    source_hashes,
    write_review_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True, type=Path)
    parser.add_argument("--atomic-evidence", required=True, type=Path)
    parser.add_argument("--atomic-contexts", required=True, type=Path)
    parser.add_argument("--selected-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()
    paths = {
        "input_review_csv": args.review_csv.expanduser().resolve(),
        "atomic_evidence": args.atomic_evidence.expanduser().resolve(),
        "atomic_contexts_full": args.atomic_contexts.expanduser().resolve(),
        "selected_evidence": args.selected_evidence.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    original_fields, rows = read_review_csv(paths["input_review_csv"])
    if len(rows) != 120:
        raise ValueError(f"frozen M4 review must contain exactly 120 rows, found {len(rows)}")
    before_order = [(row.get("sample_id"), row.get("claim_id")) for row in rows]
    before_human = [{field: row.get(field, "") for field in HUMAN_FIELDS} for row in rows]
    evidence_by_id, _ = build_frozen_indexes(
        read_jsonl(paths["atomic_evidence"]),
        read_jsonl(paths["selected_evidence"]),
        read_jsonl(paths["atomic_contexts_full"]),
    )
    enriched = enrich_review_rows(rows, evidence_by_id)
    after_order = [(row.get("sample_id"), row.get("claim_id")) for row in enriched]
    after_human = [{field: row.get(field, "") for field in HUMAN_FIELDS} for row in enriched]
    if before_order != after_order:
        raise AssertionError("review row order changed during enrichment")
    if before_human != after_human:
        raise AssertionError("human annotation fields changed during enrichment")

    output = args.output.expanduser().resolve()
    manifest_output = args.manifest_output.expanduser().resolve()
    fields = original_fields + [field for field in ENRICHMENT_FIELDS if field not in original_fields]
    write_review_csv(output, fields, enriched)
    manifest = {
        "schema_version": 1,
        "experiment": "M4_claim_support_context_enrichment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact_hashes": source_hashes(paths),
        "input_csv_sha256": file_sha256(paths["input_review_csv"]),
        "output_csv_sha256": file_sha256(output),
        "row_count": len(enriched),
        "row_order_preserved": True,
        "human_annotation_fields_preserved": True,
        "unresolved_evidence_count": 0,
        "unresolved_source_sentence_count": 0,
        "cross_sample_mismatch_count": 0,
        "source_sentence_mismatch_count": 0,
    }
    write_json(manifest_output, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
