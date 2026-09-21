import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kric.captioning.m3 import (
    FAILURE_SAMPLE_ID,
    FrozenM3Inputs,
    _comparison,
    build_atomic_model_inputs,
    run_primary_analysis,
    validate_atomic_audit,
    validate_context_budgets,
    validate_generator_control,
    validate_primary_ids,
)
from scripts.run_m3_generation import load_m3_config


ROOT = Path(__file__).resolve().parents[2]


def _frozen_ids():
    ids_50 = json.loads(
        (ROOT / "configs/dataset/goodnews_validation_50_ids.json").read_text(
            encoding="utf-8"
        )
    )
    ids_49 = json.loads(
        (ROOT / "configs/dataset/m3_complete_49_ids.json").read_text(
            encoding="utf-8"
        )
    )
    return ids_50, ids_49


def test_m3_config_is_exactly_controlled_and_primary_ids_are_frozen():
    config = load_m3_config(ROOT / "configs/experiments/m3_atomic_generation.yaml")
    validate_generator_control(config)
    ids_50, ids_49 = _frozen_ids()
    validate_primary_ids(ids_50, ids_49)
    assert len(ids_49) == 49
    assert ids_49 == [value for value in ids_50 if value != FAILURE_SAMPLE_ID]
    assert FAILURE_SAMPLE_ID not in ids_49
    assert config["evaluation"]["metrics"] == ["cider", "entity"]


def test_generator_control_rejects_any_decoding_difference():
    config = load_m3_config(ROOT / "configs/experiments/m3_atomic_generation.yaml")
    config["generation"] = dict(config["generation"])
    config["generation"]["num_beams"] = 2
    with pytest.raises(ValueError, match="differs from frozen B2"):
        validate_generator_control(config)


def test_atomic_audit_forbids_spacy_fallback():
    audit = {
        "source_sentences_requested": 150,
        "source_sentences_completed": 149,
        "source_sentences_failed": 1,
        "samples_affected_by_extraction_failures": [FAILURE_SAMPLE_ID],
        "spacy_fallback_used": False,
    }
    validate_atomic_audit(audit)
    audit["spacy_fallback_used"] = True
    with pytest.raises(ValueError, match="149/150"):
        validate_atomic_audit(audit)


def test_context_budget_guards_full_and_token_matched():
    selected = [{"sample_id": "s", "context_token_count": 120}]
    full = [{"sample_id": "s", "context_tokens": 180}]
    matched = [
        {
            "sample_id": "s",
            "context_tokens": 119,
            "b2_context_token_budget": 120,
        }
    ]
    validate_context_budgets(full, matched, selected)
    full[0]["context_tokens"] = 385
    with pytest.raises(ValueError, match="exceeds 384"):
        validate_context_budgets(full, matched, selected)
    full[0]["context_tokens"] = 180
    matched[0]["context_tokens"] = 121
    with pytest.raises(ValueError, match="exceeds B2 budget"):
        validate_context_budgets(full, matched, selected)


def test_model_inputs_receive_only_frozen_atomic_context():
    sample = SimpleNamespace(
        sample_id="s",
        image_path="image.jpg",
        article_text="FULL_ARTICLE_MUST_NOT_ENTER",
        reference_caption="REFERENCE_MUST_NOT_ENTER",
    )
    contexts = [{"sample_id": "s", "context": "Evidence:\n- Frozen proposition."}]
    model_inputs = build_atomic_model_inputs([sample], contexts, ["s"])
    assert len(model_inputs) == 1
    assert model_inputs[0].article_text == "Evidence:\n- Frozen proposition."
    assert "FULL_ARTICLE" not in model_inputs[0].article_text
    assert "REFERENCE" not in model_inputs[0].article_text
    assert not hasattr(model_inputs[0], "reference_caption")


def test_bootstrap_pairs_by_sample_id_not_mapping_order():
    ids = [f"s-{index}" for index in range(49)]
    baseline = {sample_id: {"EntityF1": float(index)} for index, sample_id in enumerate(ids)}
    candidate = {
        sample_id: {"EntityF1": baseline[sample_id]["EntityF1"] + 1.0}
        for sample_id in reversed(ids)
    }
    result = _comparison(baseline, candidate, ids, "EntityF1")
    assert result["n"] == 49
    assert result["mean_delta"] == 1.0
    assert (result["wins"], result["ties"], result["losses"]) == (49, 0, 0)


def _write_method(directory: Path, ids: list[str], offset: float) -> None:
    directory.mkdir(parents=True)
    predictions = []
    for index, sample_id in enumerate(ids):
        predictions.append(
            {
                "sample_id": sample_id,
                "prediction": f"caption {offset} {index}",
                "reference": f"reference {index}",
                "metadata": {},
            }
        )
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    with (directory / "metrics_per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "CIDEr",
                "EntityPrecision",
                "EntityRecall",
                "EntityF1",
                "EntityPredicted",
            ],
        )
        writer.writeheader()
        for index, sample_id in enumerate(ids):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "CIDEr": index + offset,
                    "EntityPrecision": 0.5,
                    "EntityRecall": 0.5,
                    "EntityF1": 0.5 + offset / 100,
                    "EntityPredicted": index % 3 + offset,
                }
            )


def test_primary_analysis_uses_same_exact_49_for_all_methods(tmp_path):
    ids_50, ids_49 = _frozen_ids()
    b2_dir = tmp_path / "b2"
    output = tmp_path / "m3"
    _write_method(b2_dir, ids_50, 0.0)
    _write_method(output / "atomic_full", ids_50, 2.0)
    _write_method(output / "atomic_token_matched", ids_50, 1.0)
    selected = tuple(
        {"sample_id": sample_id, "selected_sentence_texts": ["sentence"]}
        for sample_id in ids_50
    )
    atomic = tuple(
        {"sample_id": sample_id, "atomic_units": [{"text": "fact"}]}
        for sample_id in ids_50
    )
    contexts = tuple(
        {
            "sample_id": sample_id,
            "dropped_evidence_ids": [],
        }
        for sample_id in ids_50
    )
    frozen = FrozenM3Inputs(
        tuple(ids_50),
        tuple(ids_49),
        tuple(
            {
                "sample_id": sample_id,
                "prediction": "b2",
                "reference": f"reference {index}",
                "metadata": {},
            }
            for index, sample_id in enumerate(ids_50)
        ),
        selected,
        atomic,
        contexts,
        contexts,
        {},
        "synthetic_test_provenance",
    )
    run_primary_analysis(frozen=frozen, b2_dir=b2_dir, output_root=output)
    summary = json.loads(
        (output / "primary_49/m3_comparison.json").read_text(encoding="utf-8")
    )
    assert summary["primary_ids"] == ids_49
    assert summary["sample_count"] == 49
    assert summary["excluded_failure_sample"] == FAILURE_SAMPLE_ID
    assert all(
        metric["n"] == 49
        for comparison in summary["comparisons"].values()
        for metric in comparison["metrics"].values()
    )
    with (output / "primary_49/per_sample_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in rows] == ids_49
    assert FAILURE_SAMPLE_ID not in {row["sample_id"] for row in rows}


def test_generation_code_has_no_atomic_extractor_dependency():
    sources = [
        (ROOT / "src/kric/captioning/m3.py").read_text(encoding="utf-8"),
        (ROOT / "scripts/run_m3_generation.py").read_text(encoding="utf-8"),
    ]
    assert all("kric.evidence" not in source for source in sources)
    assert all("llm_structured" not in source for source in sources)