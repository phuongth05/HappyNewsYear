import json

from kric.evidence.extractor_comparison import (
    audit_units,
    build_comparison_rows,
    token_matched_context_stats,
)


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
