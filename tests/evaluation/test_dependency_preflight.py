import sys
from types import SimpleNamespace

import pytest

from kric.evaluation.preflight import (
    EvaluationDependencyError,
    check_evaluation_dependencies,
)


def test_preflight_loads_exact_spacy_model_and_probes_cider(monkeypatch) -> None:
    loaded = []
    fake_spacy = SimpleNamespace(
        __version__="test",
        load=lambda name: loaded.append(name) or SimpleNamespace(pipe_names=["ner"]),
    )
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    report = check_evaluation_dependencies(
        ("cider", "entity"),
        entity_extractor="spacy",
        spacy_model="en_core_web_sm",
        cider_probe=lambda: {"available": True, "ptb_tokenizer_probe": "passed"},
        spice_probe=lambda: {"status": "unavailable", "reason": "optional"},
    )
    assert loaded == ["en_core_web_sm"]
    assert report["python_executable"] == sys.executable
    assert report["checks"]["spacy"]["pipeline"] == ["ner"]
    assert report["checks"]["cider"]["available"] is True
    assert report["checks"]["spice"]["status"] == "unavailable"
    assert report["checks"]["spice"]["required"] is False


def test_preflight_fails_when_spacy_model_cannot_load(monkeypatch) -> None:
    def fail(_name):
        raise OSError("missing")

    monkeypatch.setitem(
        sys.modules, "spacy", SimpleNamespace(__version__="test", load=fail)
    )
    with pytest.raises(EvaluationDependencyError, match="en_core_web_sm"):
        check_evaluation_dependencies(
            ("entity",),
            entity_extractor="spacy",
            spacy_model="en_core_web_sm",
            spice_probe=lambda: {"status": "unavailable"},
        )


def test_preflight_propagates_cider_probe_failure() -> None:
    def fail():
        raise EvaluationDependencyError("missing CIDEr")

    with pytest.raises(EvaluationDependencyError, match="missing CIDEr"):
        check_evaluation_dependencies(
            ("cider",),
            entity_extractor="metadata",
            spacy_model="unused",
            cider_probe=fail,
            spice_probe=lambda: {"status": "unavailable"},
        )


def test_optional_spice_unavailable_is_nonfatal() -> None:
    report = check_evaluation_dependencies(
        ("cider",),
        entity_extractor="metadata",
        spacy_model="unused",
        cider_probe=lambda: {"available": True},
        spice_probe=lambda: {"status": "error", "reason": "not configured"},
    )
    assert report["status"] == "passed"
    assert report["checks"]["spice"] == {
        "status": "error",
        "reason": "not configured",
        "required": False,
    }


def test_explicitly_required_spice_must_be_available() -> None:
    with pytest.raises(EvaluationDependencyError, match="explicitly configured"):
        check_evaluation_dependencies(
            ("spice",),
            entity_extractor="metadata",
            spacy_model="unused",
            spice_probe=lambda: {"status": "unavailable"},
        )
