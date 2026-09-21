from kric.matching.calibration import (
    CalibrationExample,
    cross_validate_thresholds,
    join_calibration_rows,
    select_threshold,
    stratified_fold_assignments,
)


def _example(index, label, score):
    return CalibrationExample(
        sample_id=f"s{index // 3}",
        claim_id=f"c{index}",
        caption_variant="atomic_token_matched",
        score=score,
        label=label,
        source_label="supported" if label else "unsupported",
    )


def test_uncertain_is_excluded_from_frozen_120_join():
    annotations = []
    review = []
    for index in range(120):
        label = "uncertain" if index == 119 else "supported" if index % 2 else "unsupported"
        base = {
            "sample_id": f"s{index // 4}",
            "claim_id": f"c{index}",
            "caption_variant": "b2",
        }
        annotations.append({**base, "support_label": label})
        review.append({**base, "cosine_top1_score": str(index / 120)})
    examples, audit = join_calibration_rows(annotations, review)
    assert len(examples) == 119
    assert audit["excluded_uncertain_claims"] == 1
    assert all(example.source_label != "uncertain" for example in examples)


def test_stratified_folds_are_deterministic_and_contain_both_classes():
    examples = [_example(index, index % 2, index / 20) for index in range(20)]
    first = stratified_fold_assignments(examples, n_splits=5, seed=2026)
    second = stratified_fold_assignments(examples, n_splits=5, seed=2026)
    assert first == second
    for fold in range(5):
        assert {examples[index].label for index, value in enumerate(first) if value == fold} == {0, 1}


def test_threshold_objectives_are_correct():
    examples = [
        _example(0, 1, 0.9),
        _example(1, 0, 0.8),
        _example(2, 1, 0.7),
        _example(3, 0, 0.1),
    ]
    max_f1 = select_threshold(examples, "max_f1")
    precise = select_threshold(examples, "precision_at_least_0.90_then_max_recall")
    assert max_f1["threshold"] == 0.7
    assert max_f1["f1"] == 0.8
    assert precise["threshold"] == 0.9
    assert precise["precision"] >= 0.90
    assert precise["recall"] == 0.5


def test_each_fold_threshold_is_selected_from_training_only():
    examples = [_example(index, index % 2, (index * 7 % 31) / 31) for index in range(30)]
    reports, predictions = cross_validate_thresholds(
        examples, n_splits=5, seed=2026, bootstrap_resamples=50
    )
    assert len(predictions) == len(examples) * 2
    by_claim = {example.claim_id: example for example in examples}
    for objective, report in reports.items():
        for fold in report["folds"]:
            train_ids = set(fold["train_claim_ids"])
            heldout_ids = set(fold["heldout_claim_ids"])
            assert train_ids.isdisjoint(heldout_ids)
            assert train_ids | heldout_ids == set(by_claim)
            expected = select_threshold([by_claim[claim_id] for claim_id in train_ids], objective)
            assert fold["threshold"] == expected["threshold"]
