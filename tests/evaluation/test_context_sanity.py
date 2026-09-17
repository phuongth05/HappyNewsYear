import csv
import json
from pathlib import Path

import pytest

from scripts.analyze_context_sanity import (
    analyze,
    build_qualitative_examples,
    write_qualitative_markdown,
)


METRICS = ("CIDEr", "EntityPrecision", "EntityRecall", "EntityF1")


def _write_condition(
    directory: Path,
    name: str,
    predictions: tuple[str, str],
    entity_f1: float,
) -> None:
    directory.mkdir()
    rows = []
    for index, prediction in enumerate(predictions):
        sample_id = f"s{index}"
        metadata = {}
        if name != "b0":
            metadata = {
                "selected_article_sample_id": (
                    sample_id if name == "b1" else f"s{1 - index}"
                ),
                "original_article_tokens": 200 + index * 200,
                "used_article_tokens": min(256, 200 + index * 200),
                "truncation_ratio": 0.0 if index == 0 else 0.36,
                "max_context_tokens": 256,
            }
        rows.append(
            {
                "sample_id": sample_id,
                "prediction": prediction,
                "reference": f"reference {index}",
                "metadata": metadata,
            }
        )
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "metrics": {
            "CIDEr": 0.3,
            "EntityPrecision": entity_f1,
            "EntityRecall": entity_f1,
            "EntityF1": entity_f1,
        }
    }
    (directory / "metrics.json").write_text(json.dumps(summary), encoding="utf-8")
    with (directory / "metrics_per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", *METRICS))
        writer.writeheader()
        for index in range(2):
            writer.writerow(
                {
                    "sample_id": f"s{index}",
                    "CIDEr": 0.2 + index * 0.1,
                    "EntityPrecision": entity_f1,
                    "EntityRecall": entity_f1,
                    "EntityF1": entity_f1,
                }
            )


def test_three_condition_context_sanity_analysis(tmp_path: Path) -> None:
    b0, b1, random = tmp_path / "b0", tmp_path / "b1", tmp_path / "random"
    _write_condition(b0, "b0", ("base zero", "base one"), 0.3)
    _write_condition(b1, "b1", ("correct zero", "correct one"), 0.6)
    _write_condition(random, "random", ("random zero", "random one"), 0.4)
    result = analyze(b0, b1, random, resamples=100, seed=7)
    assert result["prediction_change_rate"] == 1.0
    assert result["correct_vs_random_entity_gain"] == pytest.approx(0.2)
    assert result["context_statistics"]["b1_correct"]["median_article_tokens"] == 300
    assert result["context_statistics"]["b1_correct"]["percentage_truncated_at_256"] == 50.0
    assert "b1_correct_vs_b1_random" in result["paired_bootstrap"]
    assert len(result["qualitative_candidates"]) == 2


def test_qualitative_outputs_contain_all_aligned_samples(tmp_path: Path) -> None:
    b0, b1, random = tmp_path / "b0", tmp_path / "b1", tmp_path / "random"
    _write_condition(b0, "b0", ("base zero", "base one"), 0.3)
    _write_condition(b1, "b1", ("correct zero", "correct one"), 0.6)
    _write_condition(random, "random", ("random zero", "random one"), 0.4)
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    annotations = {
        "s0": {"article": "The first article has useful context.", "images": {"0": "unused"}},
        "s1": {"article": "The second article has other context.", "images": {"0": "unused"}},
    }
    # Explicit image IDs make the tiny fixture match prediction IDs.
    annotations["s0"]["images"] = {"0": {"sample_id": "s0", "caption": "reference 0"}}
    annotations["s1"]["images"] = {"0": {"sample_id": "s1", "caption": "reference 1"}}
    (dataset / "article+caption.json").write_text(json.dumps(annotations), encoding="utf-8")
    (dataset / "img_splits.json").write_text(
        json.dumps({"s0.jpg": "val", "s1.jpg": "val"}), encoding="utf-8"
    )

    examples = build_qualitative_examples(b0, b1, random, dataset)
    assert len(examples) == 2
    assert examples[0]["article_excerpt"].startswith("The first article")
    assert examples[0]["random_article_donor_sample_id"] == "s1"
    markdown = tmp_path / "qualitative.md"
    write_qualitative_markdown(markdown, examples, representative_count=2)
    assert "no automatic success/failure classification" in markdown.read_text(encoding="utf-8")
