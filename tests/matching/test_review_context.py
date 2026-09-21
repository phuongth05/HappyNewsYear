import csv
import io
import json

import pytest

from kric.matching.annotation_ui import (
    apply_annotations,
    build_annotation_html,
    export_csv_text,
    group_candidate_evidence,
    toggle_best_evidence_id,
)
from kric.matching.review_context import (
    ENRICHMENT_FIELDS,
    HUMAN_FIELDS,
    ProvenanceResolutionError,
    build_frozen_indexes,
    enrich_review_rows,
)


def _sources():
    atomic = [
        {
            "sample_id": "s1",
            "atomic_units": [
                {
                    "evidence_id": "e1",
                    "text": "Alice attended the summit.",
                    "type": "event",
                    "provenance": {
                        "sample_id": "s1",
                        "source_sentence_id": "12",
                        "source_sentence": "Alice attended the summit in Paris on Monday.",
                    },
                },
                {
                    "evidence_id": "e2",
                    "text": "The summit was in Paris.",
                    "type": "location",
                    "provenance": {
                        "sample_id": "s1",
                        "source_sentence_id": "12",
                        "source_sentence": "Alice attended the summit in Paris on Monday.",
                    },
                },
            ],
        }
    ]
    selected = [
        {
            "sample_id": "s1",
            "selected_sentence_ids": [12],
            "selected_sentence_texts": ["Alice attended the summit in Paris on Monday."],
        }
    ]
    contexts = [{"sample_id": "s1", "included_evidence_ids": ["e1", "e2"]}]
    return atomic, selected, contexts


def _review_row(index=0):
    return {
        "sample_id": "s1",
        "caption_variant": "b2",
        "caption": "Alice is pictured at a summit.",
        "claim_id": f"c{index:03d}",
        "claim_text": "Alice attended a summit.",
        "claim_type": "event",
        "cosine_top1_id": "e1",
        "cosine_top1_text": "Alice attended the summit.",
        "cosine_top1_score": "0.75",
        "nli_top1_id": "e2",
        "nli_top1_text": "The summit was in Paris.",
        "nli_top1_score": "0.91",
        "union_top3_evidence_json": json.dumps(
            [
                {
                    "evidence_id": "e1",
                    "evidence_text": "Alice attended the summit.",
                    "evidence_type": "event",
                    "source_sentence_id": "12",
                    "selected_by": "cosine",
                    "rank": 1,
                    "score": 0.75,
                },
                {
                    "evidence_id": "e2",
                    "evidence_text": "The summit was in Paris.",
                    "evidence_type": "location",
                    "source_sentence_id": "12",
                    "selected_by": "nli",
                    "rank": 1,
                    "score": 0.91,
                },
            ]
        ),
        "support_label": "",
        "best_evidence_ids": "",
        "matcher_preference": "",
        "notes": "",
    }


def test_120_rows_order_annotations_and_parent_resolution_are_preserved():
    atomic, selected, contexts = _sources()
    evidence, _ = build_frozen_indexes(atomic, selected, contexts)
    rows = [_review_row(index) for index in range(120)]
    result = enrich_review_rows(rows, evidence)
    assert len(result) == 120
    assert [row["claim_id"] for row in result] == [row["claim_id"] for row in rows]
    assert all(row[field] == "" for row in result for field in HUMAN_FIELDS)
    assert result[0]["cosine_top1_source_sentence_id"] == "12"
    assert result[0]["cosine_top1_source_sentence_text"] == (
        "Alice attended the summit in Paris on Monday."
    )
    assert all(field in result[0] for field in ENRICHMENT_FIELDS)


def test_cross_sample_evidence_is_rejected():
    atomic, selected, contexts = _sources()
    evidence, _ = build_frozen_indexes(atomic, selected, contexts)
    row = _review_row()
    row["sample_id"] = "s2"
    with pytest.raises(ProvenanceResolutionError, match="another sample"):
        enrich_review_rows([row], evidence)


def test_source_sentence_provenance_mismatch_is_rejected():
    atomic, selected, contexts = _sources()
    atomic[0]["atomic_units"][0]["provenance"]["source_sentence"] = "Altered sentence."
    with pytest.raises(ProvenanceResolutionError, match="differs across frozen artifacts"):
        build_frozen_indexes(atomic, selected, contexts)


def test_union_evidence_is_grouped_once_for_duplicate_source_sentence():
    atomic, selected, contexts = _sources()
    evidence, _ = build_frozen_indexes(atomic, selected, contexts)
    result = enrich_review_rows([_review_row()], evidence)[0]
    items = json.loads(result["union_top3_evidence_context_json"])
    groups = group_candidate_evidence(items)
    assert len(groups) == 1
    assert groups[0]["source_sentence_id"] == "12"
    assert [item["evidence_id"] for item in groups[0]["evidence"]] == ["e1", "e2"]


def test_exported_csv_preserves_columns_and_annotations():
    row = _review_row()
    fields = list(row)
    updated = apply_annotations(
        [row],
        {
            row["claim_id"]: {
                "support_label": "supported",
                "best_evidence_ids": "e1",
                "matcher_preference": "cosine",
                "notes": "Direct support.",
            }
        },
    )
    text = export_csv_text(fields, updated)
    parsed = list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))
    assert list(parsed[0]) == fields
    assert parsed[0]["support_label"] == "supported"
    assert parsed[0]["best_evidence_ids"] == "e1"
    assert parsed[0]["notes"] == "Direct support."


def test_best_evidence_toggle_is_ordered_and_reversible():
    assert toggle_best_evidence_id("", "e2", ["e1", "e2"]) == "e2"
    assert toggle_best_evidence_id("e2", "e1", ["e1", "e2"]) == "e1;e2"
    assert toggle_best_evidence_id("e1;e2", "e2", ["e1", "e2"]) == "e1"


def test_html_build_is_deterministic_and_contains_required_controls():
    atomic, selected, contexts = _sources()
    evidence, _ = build_frozen_indexes(atomic, selected, contexts)
    row = enrich_review_rows([_review_row()], evidence)[0]
    fields = list(row)
    first = build_annotation_html(fields, [row], source_sha256="a" * 64)
    second = build_annotation_html(fields, [row], source_sha256="a" * 64)
    assert first == second
    for expected in (
        "Strict Evidence View",
        "Context-Assisted View",
        "Next unannotated",
        "Export CSV",
        "Scores are not calibrated probabilities",
        "localStorage",
    ):
        assert expected in first
