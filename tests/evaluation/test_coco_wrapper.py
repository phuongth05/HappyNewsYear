import types

import pytest

import kric.evaluation.coco as coco_module
from kric.evaluation.coco import CocoCaptionMetrics
from kric.evaluation.records import EvaluationRecord


class _Cider:
    def compute_score(self, references, predictions):
        return 0.5, [0.25, 0.75]


def _module(name, **values):
    module = types.ModuleType(name)
    for key, value in values.items():
        setattr(module, key, value)
    return module


def test_coco_wrapper_preserves_per_sample_scores(monkeypatch) -> None:
    modules = {
        "pycocoevalcap": _module("pycocoevalcap"),
        "pycocoevalcap.cider": _module("pycocoevalcap.cider"),
        "pycocoevalcap.cider.cider": _module("pycocoevalcap.cider.cider", Cider=_Cider),
        "pycocoevalcap.spice": _module("pycocoevalcap.spice"),
        "pycocoevalcap.spice.spice": _module("pycocoevalcap.spice.spice"),
        "pycocoevalcap.tokenizer": _module("pycocoevalcap.tokenizer"),
        "pycocoevalcap.tokenizer.ptbtokenizer": _module("pycocoevalcap.tokenizer.ptbtokenizer"),
    }
    for name, module in modules.items():
        monkeypatch.setitem(__import__("sys").modules, name, module)
    monkeypatch.setattr(
        coco_module,
        "_tokenize_with_ptb",
        lambda values, module, timeout: {
            key: [item["caption"] for item in rows] for key, rows in values.items()
        },
    )
    monkeypatch.setattr(
        coco_module,
        "_compute_spice",
        lambda *args, **kwargs: (
            0.6,
            [
                {"All": {"p": 0.5, "r": 1.0, "f": 2 / 3}},
                {"All": {"p": 1.0, "r": 0.5, "f": 2 / 3}},
            ],
        ),
    )
    records = [
        EvaluationRecord("a", "prediction a", "reference a"),
        EvaluationRecord("b", "prediction b", "reference b"),
    ]

    result = CocoCaptionMetrics(("CIDEr", "SPICE")).evaluate(records)

    assert result.summary == {"CIDEr": 0.5, "SPICE": 0.6}
    assert result.per_sample["a"]["CIDEr"] == 0.25
    assert result.per_sample["b"]["CIDEr"] == 0.75
    assert result.per_sample["a"]["SPICE"] == pytest.approx(2 / 3)
    assert result.per_sample["a"]["SPICE_precision"] == 0.5
