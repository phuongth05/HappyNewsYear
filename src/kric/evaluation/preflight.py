"""Fail-fast checks for dependencies required by configured evaluation metrics."""

from __future__ import annotations

import sys
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Sequence


class EvaluationDependencyError(RuntimeError):
    """A configured evaluator cannot run in the current Python environment."""


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _probe_cider() -> dict[str, Any]:
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.tokenizer import ptbtokenizer

        from .coco import _tokenize_with_ptb
    except (ImportError, ModuleNotFoundError) as error:
        raise EvaluationDependencyError(
            "CIDEr requires pycocoevalcap; install the exact workflow interpreter "
            "with `python -m pip install -e .[coco-eval]`"
        ) from error

    try:
        tokenized = _tokenize_with_ptb(
            {0: [{"caption": "Evaluation dependency preflight."}]},
            ptbtokenizer,
            timeout_seconds=30,
        )
        Cider()
    except Exception as error:
        raise EvaluationDependencyError(
            "CIDEr/PTB tokenizer preflight failed; verify pycocoevalcap, its "
            "bundled Stanford tokenizer, and Java in this interpreter environment"
        ) from error
    if 0 not in tokenized:
        raise EvaluationDependencyError("CIDEr PTB tokenizer returned an invalid probe result")
    return {
        "available": True,
        "package": "pycocoevalcap",
        "version": _package_version("pycocoevalcap"),
        "ptb_tokenizer_probe": "passed",
    }


def _probe_spice_availability() -> dict[str, Any]:
    """Inspect optional SPICE resources without downloading or scoring."""

    try:
        from pycocoevalcap.spice import spice as spice_module
    except (ImportError, ModuleNotFoundError) as error:
        return {
            "status": "unavailable",
            "required": False,
            "reason": "pycocoevalcap SPICE module is unavailable",
            "detail": str(error),
        }
    try:
        module_dir = Path(spice_module.__file__).resolve().parent
        resources = {
            "spice_jar": module_dir / spice_module.SPICE_JAR,
            "stanford_corenlp_jar": module_dir / "lib" / "stanford-corenlp-3.6.0.jar",
            "stanford_corenlp_models_jar": (
                module_dir / "lib" / "stanford-corenlp-3.6.0-models.jar"
            ),
        }
        missing = [name for name, path in resources.items() if not path.is_file()]
        java = shutil.which("java")
        if missing or java is None:
            return {
                "status": "unavailable",
                "required": False,
                "reason": "optional SPICE runtime resources are incomplete",
                "missing": missing + (["java"] if java is None else []),
                "resources": {name: str(path) for name, path in resources.items()},
            }
        completed = subprocess.run(
            [java, "-version"], capture_output=True, text=True, timeout=15
        )
        if completed.returncode:
            return {
                "status": "error",
                "required": False,
                "reason": "java -version returned a non-zero exit code",
                "return_code": completed.returncode,
            }
        return {
            "status": "available",
            "required": False,
            "java": java,
            "resources": {name: str(path) for name, path in resources.items()},
        }
    except Exception as error:
        return {
            "status": "error",
            "required": False,
            "reason": "unexpected SPICE availability probe failure",
            "detail": str(error),
        }


def check_evaluation_dependencies(
    metrics: Sequence[str],
    *,
    entity_extractor: str,
    spacy_model: str,
    cider_probe: Callable[[], dict[str, Any]] | None = None,
    spice_probe: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate configured dependencies without loading caption-model weights."""

    normalized = {metric.strip().lower() for metric in metrics}
    report: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "metrics": sorted(normalized),
        "checks": {},
    }
    if "entity" in normalized and entity_extractor == "spacy":
        try:
            import spacy
        except ImportError as error:
            raise EvaluationDependencyError(
                "entity evaluation requires spaCy in the exact workflow interpreter; "
                "install with `python -m pip install -e .[ner]`"
            ) from error
        try:
            nlp = spacy.load(spacy_model)
        except (ImportError, OSError) as error:
            raise EvaluationDependencyError(
                f"spaCy model {spacy_model!r} is unavailable in {sys.executable}; "
                f"install it with `python -m spacy download {spacy_model}`"
            ) from error
        report["checks"]["spacy"] = {
            "available": True,
            "version": getattr(spacy, "__version__", _package_version("spacy")),
            "model": spacy_model,
            "pipeline": list(nlp.pipe_names),
        }
        del nlp
    if "cider" in normalized:
        report["checks"]["cider"] = (cider_probe or _probe_cider)()
    spice = (spice_probe or _probe_spice_availability)()
    spice["required"] = "spice" in normalized
    report["checks"]["spice"] = spice
    if spice["required"] and spice.get("status") != "available":
        raise EvaluationDependencyError(
            "SPICE was explicitly configured as a required metric but its "
            f"availability status is {spice.get('status', 'unknown')!r}"
        )
    report["status"] = "passed"
    return report
