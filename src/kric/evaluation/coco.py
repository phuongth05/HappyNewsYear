"""CIDEr and SPICE wrappers using the official COCO caption metric package."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .base import EvaluationMetric
from .records import EvaluationRecord, MetricResult, MetricUnavailableError


class CocoCaptionMetrics(EvaluationMetric):
    """Run CIDEr and/or SPICE with one shared PTB tokenization pass."""

    def __init__(
        self,
        metrics: Sequence[str],
        *,
        timeout_seconds: int = 600,
        spice_java: str = "java",
        spice_memory: str = "8G",
    ):
        normalized = tuple(metric.upper() for metric in metrics)
        unknown = set(normalized) - {"CIDER", "SPICE"}
        if unknown:
            raise ValueError(f"unsupported COCO caption metrics: {sorted(unknown)}")
        self.metrics = normalized
        self.name = "+".join(metric.lower() for metric in normalized)
        self.timeout_seconds = timeout_seconds
        self.spice_java = spice_java
        self.spice_memory = spice_memory

    @staticmethod
    def _spice_fields(value: Any) -> dict[str, float | None]:
        if isinstance(value, dict):
            all_scores = value.get("All", value.get("all", {}))
            if isinstance(all_scores, dict):
                return {
                    "SPICE": _optional_float(all_scores.get("f")),
                    "SPICE_precision": _optional_float(all_scores.get("p")),
                    "SPICE_recall": _optional_float(all_scores.get("r")),
                }
        return {"SPICE": _optional_float(value)}

    def evaluate(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        try:
            from pycocoevalcap.tokenizer import ptbtokenizer
        except ImportError as error:
            raise MetricUnavailableError(
                "CIDEr/SPICE require `pip install -e .[coco-eval]`; SPICE also requires a compatible Java runtime"
            ) from error

        internal_ids = list(range(len(records)))
        references = {
            index: [{"caption": record.reference}]
            for index, record in zip(internal_ids, records)
        }
        predictions = {
            index: [{"caption": record.prediction}]
            for index, record in zip(internal_ids, records)
        }
        references = _tokenize_with_ptb(references, ptbtokenizer, self.timeout_seconds)
        predictions = _tokenize_with_ptb(predictions, ptbtokenizer, self.timeout_seconds)

        summary: dict[str, Any] = {}
        per_sample = {record.sample_id: {} for record in records}
        try:
            package_version = version("pycocoevalcap")
        except PackageNotFoundError:
            package_version = "unknown"
        details: dict[str, Any] = {
            "implementation": "pycocoevalcap",
            "package_version": package_version,
            "tokenizer": "PTBTokenizer",
            "reference_count_per_sample": 1,
        }
        for metric in self.metrics:
            if metric == "CIDER":
                from pycocoevalcap.cider.cider import Cider

                aggregate, sample_scores = Cider().compute_score(references, predictions)
            else:
                from pycocoevalcap.spice import spice as spice_module

                aggregate, sample_scores = _compute_spice(
                    references,
                    predictions,
                    spice_module,
                    java=self.spice_java,
                    memory=self.spice_memory,
                    timeout_seconds=self.timeout_seconds,
                )
            output_name = "CIDEr" if metric == "CIDER" else "SPICE"
            summary[output_name] = float(aggregate)
            if len(sample_scores) != len(records):
                raise RuntimeError(
                    f"{output_name} returned {len(sample_scores)} per-sample scores for {len(records)} records"
                )
            for record, value in zip(records, sample_scores):
                if metric == "SPICE":
                    per_sample[record.sample_id].update(self._spice_fields(value))
                else:
                    per_sample[record.sample_id][output_name] = float(value)
        return MetricResult(summary=summary, per_sample=per_sample, details=details)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _tokenize_with_ptb(captions: dict, module: Any, timeout_seconds: int) -> dict:
    """Invoke the packaged Stanford tokenizer without pycocoevalcap's Windows hang.

    pycocoevalcap passes ``input=`` to a subprocess whose stdin is not a pipe.
    Calling the same bundled tokenizer JAR with its temporary filename is
    equivalent and works on both Windows and Linux without editing the package.
    """

    image_ids = [key for key, values in captions.items() for _ in values]
    sentences = "\n".join(
        item["caption"].replace("\n", " ")
        for values in captions.values()
        for item in values
    )
    module_dir = Path(module.__file__).resolve().parent
    jar = module_dir / module.STANFORD_CORENLP_3_4_1_JAR
    if not jar.is_file():
        raise MetricUnavailableError(f"Stanford PTB tokenizer JAR is missing: {jar}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as handle:
            handle.write(sentences)
            temporary_path = Path(handle.name)
        command = [
            "java",
            "-cp",
            str(jar),
            "edu.stanford.nlp.process.PTBTokenizer",
            "-preserveLines",
            "-lowerCase",
            str(temporary_path),
        ]
        process = subprocess.run(
            command,
            cwd=module_dir,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise MetricUnavailableError("CIDEr/SPICE tokenization requires Java on PATH") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    lines = process.stdout.splitlines()
    if len(lines) < len(image_ids):
        raise RuntimeError(
            f"PTBTokenizer returned {len(lines)} lines for {len(image_ids)} captions: {process.stderr}"
        )
    result: dict[Any, list[str]] = {}
    for image_id, line in zip(image_ids, lines):
        tokens = [token for token in line.rstrip().split(" ") if token not in module.PUNCTUATIONS]
        result.setdefault(image_id, []).append(" ".join(tokens))
    return result


def _compute_spice(
    references: dict,
    predictions: dict,
    module: Any,
    *,
    java: str,
    memory: str,
    timeout_seconds: int,
) -> tuple[float, list[dict[str, dict[str, float]]]]:
    """Run the packaged SPICE JAR with JVM options in the valid order.

    The legacy package emits ``java -jar -Xmx8G``. JVM options must precede
    ``-jar``. The optional LMDB parse cache is omitted because its bundled
    native library is not portable across current operating systems/JVMs.
    """

    image_ids = sorted(references)
    payload = [
        {
            "image_id": image_id,
            "test": predictions[image_id][0],
            "refs": references[image_id],
        }
        for image_id in image_ids
    ]
    module_dir = Path(module.__file__).resolve().parent
    jar = module_dir / module.SPICE_JAR
    if not jar.is_file():
        raise MetricUnavailableError(f"SPICE JAR is missing: {jar}")
    core_nlp_jar = module_dir / "lib" / "stanford-corenlp-3.6.0.jar"
    core_nlp_models = module_dir / "lib" / "stanford-corenlp-3.6.0-models.jar"
    if not core_nlp_jar.is_file() or not core_nlp_models.is_file():
        try:
            from pycocoevalcap.spice import get_stanford_models

            get_stanford_models.get_stanford_models()
        except Exception as error:
            raise MetricUnavailableError(
                "SPICE requires Stanford CoreNLP 3.6.0 artifacts; automatic preparation failed"
            ) from error
    with tempfile.TemporaryDirectory(prefix="kric-spice-") as directory:
        input_path = Path(directory) / "input.json"
        output_path = Path(directory) / "output.json"
        input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        command = [
            java,
            f"-Xmx{memory}",
            "-jar",
            str(jar),
            str(input_path),
            "-out",
            str(output_path),
            "-subset",
            "-silent",
        ]
        try:
            subprocess.run(
                command,
                cwd=module_dir,
                text=True,
                capture_output=True,
                check=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as error:
            raise MetricUnavailableError(f"SPICE Java executable not found: {java}") from error
        except subprocess.CalledProcessError as error:
            raise MetricUnavailableError(
                f"SPICE failed under {java!r}: {(error.stderr or error.stdout).strip()}"
            ) from error
        raw_results = json.loads(output_path.read_text(encoding="utf-8"))

    by_id = {item["image_id"]: item["scores"] for item in raw_results}
    scores: list[dict[str, dict[str, float]]] = []
    f_scores: list[float] = []
    for image_id in image_ids:
        score_set: dict[str, dict[str, float]] = {}
        for category, values in by_id[image_id].items():
            score_set[category] = {
                key: float(value) if value is not None else math.nan
                for key, value in values.items()
            }
        scores.append(score_set)
        f_scores.append(score_set["All"]["f"])
    return sum(f_scores) / len(f_scores), scores
