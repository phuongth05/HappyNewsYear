import json

import pytest

from kric.evidence.human_review import agreement_report, load_review_csv, summarize_reviews


def _row(atomicity="", faithfulness="", type_correct="", unit_type="entity"):
    return {
        "sample_id": "s1",
        "source_sentence_id": "x1",
        "atomic_units_json": json.dumps([{"type": unit_type}]),
        "atomicity": atomicity,
        "faithfulness": faithfulness,
        "type_correct": type_correct,
        "notes": "",
    }


def test_missing_human_labels_are_not_inferred():
    report = summarize_reviews([_row(), _row("good", "supported", "yes")])
    assert report["labels"]["atomicity"]["labelled"] == 1
    assert report["labels"]["atomicity"]["missing"] == 1
    assert report["labels"]["atomicity"]["counts"]["good"] == 1
    assert report["per_type_accuracy"]["entity"]["accuracy"] == 1.0


def test_agreement_uses_only_pairs_with_two_valid_labels():
    first = [_row("good", "supported", "yes")]
    second = [_row("good", "", "no")]
    report = agreement_report(first, second)
    assert report["fields"]["atomicity"]["n"] == 1
    assert report["fields"]["atomicity"]["valid_overlap_count"] == 1
    assert report["fields"]["atomicity"]["percent_agreement"] == 100.0
    assert report["fields"]["faithfulness"]["n"] == 0
    assert report["fields"]["type_correct"]["percent_agreement"] == 0.0


def test_partial_csv_preserves_existing_values_and_blanks(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text(
        "sample_id,source_sentence_id,atomicity,faithfulness,type_correct,notes\n"
        "s1,x1,good,,yes,Keep this exact note\n"
        "s2,x2,,uncertain,,\n",
        encoding="utf-8",
    )
    rows = load_review_csv(path)
    assert rows[0]["atomicity"] == "good"
    assert rows[0]["faithfulness"] == ""
    assert rows[0]["notes"] == "Keep this exact note"
    assert rows[1]["atomicity"] == ""
    assert rows[1]["faithfulness"] == "uncertain"
    assert rows[1]["type_correct"] == ""
    report = summarize_reviews(rows)
    assert report["labels"]["atomicity"]["labelled"] == 1
    assert report["labels"]["atomicity"]["missing"] == 1


def test_agreement_rejects_duplicate_annotation_keys():
    with pytest.raises(ValueError, match="duplicate annotation key"):
        agreement_report([_row(), _row()], [_row()])
