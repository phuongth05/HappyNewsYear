import csv
import json
import sys

import pytest

from kric.evidence.extractor_comparison import (
    audit_units,
    build_comparison_rows,
    comparison_coverage,
    token_matched_context_stats,
)
from scripts.compare_atomic_extractors import main as compare_main


def test_token_matching_never_pads_or_exceeds_source_budget():
    rows = [
        {
            "sample_id": "s1",
            "source_sentence_id": "x1",
            "source_rank": 0,
            "source_sentence": "A short source.",
            "units": [
                {"text": "A short fact.", "type": "event"},
                {"text": "This proposition is much too long for the budget.", "type": "event"},
            ],
        }
    ]
    report = token_matched_context_stats(rows, "units")
    assert report["padding_tokens_added"] == 0
    assert report["token_matched_atomic_context"]["max"] <= report["source_context_token_budget"]["max"]
    assert report["units_dropped"] == 1


def test_audit_reports_unresolved_spans_and_types():
    rows = [
        {
            "propositions": [
                {
                    "text": "Obama spoke.",
                    "type": "event",
                    "source_span": {"alignment": "unresolved"},
                }
            ]
        }
    ]
    report = audit_units(rows, "propositions")
    assert report["atomic_units"] == 1
    assert report["type_distribution"] == {"event": 1}
    assert report["unresolved_source_spans"] == 1


def test_exact_duplicates_are_counted_only_within_sample():
    cross_sample = [
        {"sample_id": "a", "units": [{"text": "Same fact.", "type": "event"}]},
        {"sample_id": "b", "units": [{"text": "Same fact.", "type": "event"}]},
    ]
    assert audit_units(cross_sample, "units")["exact_duplicate_units"] == 0
    within_sample = [
        {
            "sample_id": "a",
            "units": [
                {"text": "Same fact.", "type": "event"},
                {"text": " same   fact. ", "type": "event"},
            ],
        }
    ]
    report = audit_units(within_sample, "units")
    assert report["exact_duplicate_units"] == 2
    assert report["exact_duplicate_rate"] == 1.0


def test_spacy_offset_span_is_recognized_as_exact():
    source = "Obama spoke."
    rows = [
        {
            "source_sentence": source,
            "units": [
                {
                    "text": "Obama spoke.",
                    "type": "event",
                    "source_span": {"start": 0, "end": 12, "text": source},
                }
            ],
        }
    ]
    assert audit_units(rows, "units")["unresolved_source_spans"] == 0


def test_comparison_preserves_spacy_labels_and_leaves_new_review_blank():
    spacy = [
        {
            "sample_id": "s1",
            "source_sentence_id": "x1",
            "source_rank": "2",
            "source_sentence": "Obama spoke.",
            "atomic_units_json": json.dumps([{"text": "Obama spoke.", "type": "event"}]),
            "atomicity": "too_coarse",
            "faithfulness": "supported",
            "type_correct": "yes",
            "notes": "Original note, unchanged.",
        }
    ]
    llm = [
        {
            "sample_id": "s1",
            "source_sentence_id": "x1",
            "source_sentence": "Obama spoke.",
            "propositions": [{"text": "Obama spoke.", "type": "event"}],
        }
    ]
    rows, _, _ = build_comparison_rows(spacy, llm)
    row = rows[0]
    assert row["spacy_atomicity"] == "too_coarse"
    assert row["spacy_faithfulness"] == "supported"
    assert row["spacy_type_correct"] == "yes"
    assert row["spacy_notes"] == "Original note, unchanged."
    assert row["llm_atomicity"] == ""
    assert row["llm_faithfulness"] == ""
    assert row["llm_type_correct"] == ""
    assert row["preferred_extractor"] == ""
    assert row["notes"] == ""


def test_comparison_preserves_100_rows_with_one_explicit_llm_failure():
    spacy = []
    llm = []
    for index in range(100):
        source = f"Frozen source sentence {index}."
        spacy.append(
            {
                "sample_id": f"sample-{index // 3:02d}",
                "source_sentence_id": str(index),
                "source_rank": str((index % 3) + 1),
                "source_sentence": source,
                "atomic_units_json": json.dumps(
                    [{"text": f"spaCy fact {index}.", "type": "event"}]
                ),
                "atomicity": "",
                "faithfulness": "",
                "type_correct": "",
                "notes": "",
            }
        )
        if index != 57:
            llm.append(
                {
                    "sample_id": f"sample-{index // 3:02d}",
                    "source_sentence_id": str(index),
                    "source_sentence": source,
                    "propositions": [
                        {"text": f"LLM fact {index}.", "type": "event"}
                    ],
                }
            )
    failures = [
        {
            "sample_id": "sample-19",
            "source_sentence_id": "57",
            "source_sentence": "Frozen source sentence 57.",
            "error_type": "StructuredExtractionError",
            "error": "failed after maximum retries",
        }
    ]

    rows, _, llm_audit_rows = build_comparison_rows(spacy, llm, failures)
    assert len(rows) == 100
    assert [(row["sample_id"], row["source_sentence_id"]) for row in rows] == [
        (row["sample_id"], row["source_sentence_id"]) for row in spacy
    ]
    failed = rows[57]
    assert failed["llm_status"] == "failed"
    assert failed["llm_units"] == []
    assert failed["llm_unit_count"] == 0
    assert failed["llm_failure_type"] == "StructuredExtractionError"
    assert failed["llm_failure_message"] == "failed after maximum retries"
    assert failed["llm_atomicity"] == ""
    assert failed["llm_faithfulness"] == ""
    assert failed["llm_type_correct"] == ""
    assert failed["preferred_extractor"] == ""
    assert failed["notes"] == ""
    assert llm_audit_rows[57]["units"] == []
    assert comparison_coverage(rows) == {
        "total_source_sentences": 100,
        "llm_success_rows": 99,
        "llm_failed_rows": 1,
        "llm_completion_rate": 0.99,
        "spaCy_rows": 100,
        "comparison_rows": 100,
    }


def test_comparison_rejects_unknown_and_duplicate_llm_keys():
    spacy = [
        {
            "sample_id": "known",
            "source_sentence_id": "1",
            "source_sentence": "Known sentence.",
            "atomic_units_json": "[]",
        }
    ]
    unknown = [
        {
            "sample_id": "unknown",
            "source_sentence_id": "2",
            "source_sentence": "Unknown sentence.",
            "propositions": [],
        }
    ]
    with pytest.raises(ValueError, match="unexpected frozen source sentences"):
        build_comparison_rows(spacy, unknown)

    duplicate = [
        {
            "sample_id": "known",
            "source_sentence_id": "1",
            "source_sentence": "Known sentence.",
            "propositions": [],
        },
        {
            "sample_id": "known",
            "source_sentence_id": "1",
            "source_sentence": "Known sentence.",
            "propositions": [],
        },
    ]
    with pytest.raises(ValueError, match="duplicate source key"):
        build_comparison_rows(spacy, duplicate)


def test_missing_llm_row_without_failure_metadata_is_explicitly_marked():
    spacy = [
        {
            "sample_id": "known",
            "source_sentence_id": "1",
            "source_sentence": "Known sentence.",
            "atomic_units_json": "[]",
        }
    ]
    rows, _, _ = build_comparison_rows(spacy, [])
    assert rows[0]["llm_status"] == "failed"
    assert rows[0]["llm_units"] == []
    assert rows[0]["llm_failure_type"] == "missing_evidence"


def test_comparator_cli_writes_all_100_human_review_rows(
    monkeypatch, tmp_path
):
    spacy_path = tmp_path / "spacy.csv"
    llm_path = tmp_path / "llm.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    output_csv = tmp_path / "comparison.csv"
    human_csv = tmp_path / "human_review.csv"
    audit_path = tmp_path / "audit.json"
    fieldnames = [
        "sample_id",
        "source_sentence_id",
        "source_rank",
        "source_sentence",
        "atomic_units_json",
        "atomicity",
        "faithfulness",
        "type_correct",
        "notes",
    ]
    llm_rows = []
    with spacy_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(100):
            source = f"Frozen source {index}."
            writer.writerow(
                {
                    "sample_id": f"s{index}",
                    "source_sentence_id": str(index),
                    "source_rank": "1",
                    "source_sentence": source,
                    "atomic_units_json": "[]",
                    "atomicity": "",
                    "faithfulness": "",
                    "type_correct": "",
                    "notes": "",
                }
            )
            if index != 42:
                llm_rows.append(
                    {
                        "sample_id": f"s{index}",
                        "source_sentence_id": str(index),
                        "source_sentence": source,
                        "propositions": [],
                    }
                )
    llm_path.write_text(
        "".join(json.dumps(row) + "\n" for row in llm_rows),
        encoding="utf-8",
    )
    failures_path.write_text(
        json.dumps(
            {
                "sample_id": "s42",
                "source_sentence_id": "42",
                "error_type": "StructuredExtractionError",
                "error": "failed after retries",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_atomic_extractors.py",
            "--spacy-review",
            str(spacy_path),
            "--llm-evidence",
            str(llm_path),
            "--llm-failures",
            str(failures_path),
            "--output-csv",
            str(output_csv),
            "--human-review-output",
            str(human_csv),
            "--audit-output",
            str(audit_path),
        ],
    )
    compare_main()

    with human_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        human_rows = list(csv.DictReader(handle))
    assert len(human_rows) == 100
    assert human_rows[42]["llm_status"] == "failed"
    assert human_rows[42]["llm_units"] == "[]"
    assert human_rows[42]["preferred_extractor"] == ""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["total_source_sentences"] == 100
    assert audit["llm_success_rows"] == 99
    assert audit["llm_failed_rows"] == 1
    assert audit["llm_completion_rate"] == 0.99
    assert audit["spaCy_rows"] == 100
    assert audit["comparison_rows"] == 100
