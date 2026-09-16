import pytest

from kric.evaluation.bootstrap import paired_bootstrap_ci


def test_paired_bootstrap_constant_delta() -> None:
    result = paired_bootstrap_ci(
        {"a": 1.0, "b": 2.0, "c": 3.0},
        {"a": 2.0, "b": 3.0, "c": 4.0},
        resamples=200,
        seed=7,
    )
    assert result["mean_delta"] == 1.0
    assert result["ci_low"] == 1.0
    assert result["ci_high"] == 1.0
    assert result["probability_candidate_better"] == 1.0


def test_paired_bootstrap_requires_identical_ids() -> None:
    with pytest.raises(ValueError, match="paired IDs differ"):
        paired_bootstrap_ci({"a": 1.0}, {"b": 1.0})

