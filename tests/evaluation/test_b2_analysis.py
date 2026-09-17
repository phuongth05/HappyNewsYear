import csv
import json
from pathlib import Path

from scripts.analyze_b2_validation import METRICS, analyze_b2


def _condition(directory: Path, *, b1: bool, method="bm25", k=1) -> None:
    directory.mkdir()
    rows = []
    for index in range(2):
        metadata = {"used_article_tokens": 100} if b1 else {
            "context_token_count": 20 + index,
            "selected_sentence_count": k,
            "selected_sentence_ids": list(range(k)),
            "selected_sentence_texts": [f"sentence {value}" for value in range(k)],
            "selected_ranking_scores": [0.5] * k,
            "fraction_of_article_represented": 0.2,
            "retrieval_method": method,
            "retrieval_k": k,
        }
        rows.append(
            {
                "sample_id": f"s{index}",
                "prediction": f"prediction {index}",
                "reference": f"reference {index}",
                "metadata": metadata,
            }
        )
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        json.dumps({"metrics": {metric: 0.5 for metric in METRICS}}), encoding="utf-8"
    )
    with (directory / "metrics_per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", *METRICS))
        writer.writeheader()
        for index in range(2):
            writer.writerow({"sample_id": f"s{index}", **{metric: 0.5 for metric in METRICS}})


def test_b2_analysis_reports_efficiency_and_all_paired_comparisons(tmp_path: Path) -> None:
    b1 = tmp_path / "b1"
    _condition(b1, b1=True)
    variants = {}
    for method in ("bm25", "semantic"):
        for k in (1, 3, 5):
            name = f"{method}_k{k}"
            variants[name] = tmp_path / name
            _condition(variants[name], b1=False, method=method, k=k)
    result, qualitative = analyze_b2(b1, variants, resamples=100, seed=7)
    assert set(result["variants"]) == set(variants)
    assert result["variants"]["bm25_k1"]["percentage_using_fewer_tokens_than_b1"] == 100.0
    assert set(result["paired_bootstrap_vs_b1"]["semantic_k5"]) == set(METRICS)
    assert len(qualitative) == 2
    assert qualitative[0]["human_category"] is None
