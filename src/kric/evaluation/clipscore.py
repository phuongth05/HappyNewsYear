"""Reference-free CLIPScore with persistent image/text embedding caching."""

from __future__ import annotations

import hashlib
import sqlite3
from array import array
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import EvaluationMetric
from .records import EvaluationRecord, MetricResult, MetricUnavailableError


class EmbeddingCache:
    """Small SQLite cache keyed by model, input kind, and input fingerprint."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(cache_key TEXT PRIMARY KEY, dimension INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        self.connection.commit()

    def get(self, key: str) -> list[float] | None:
        row = self.connection.execute(
            "SELECT dimension, vector FROM embeddings WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        values = array("f")
        values.frombytes(row[1])
        if len(values) != row[0]:
            raise RuntimeError(f"corrupt embedding cache entry: {key}")
        return list(values)

    def put(self, key: str, values: Sequence[float]) -> None:
        packed = array("f", values)
        self.connection.execute(
            "INSERT OR REPLACE INTO embeddings(cache_key, dimension, vector) VALUES (?, ?, ?)",
            (key, len(packed), packed.tobytes()),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _digest(*parts: object) -> str:
    value = "\0".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _image_key(model_name: str, image_path: Path) -> str:
    stat = image_path.stat()
    return _digest("image", model_name, image_path.resolve(), stat.st_size, stat.st_mtime_ns)


def _text_key(model_name: str, prefix: str, text: str) -> str:
    return _digest("text", model_name, prefix, text)


class ClipScoreMetric(EvaluationMetric):
    """CLIP-S = 2.5 * max(cosine(image, caption), 0)."""

    name = "clipscore"

    def __init__(
        self,
        cache_path: str | Path,
        *,
        model_name: str = "openai/clip-vit-base-patch32",
        model_revision: str | None = None,
        device: str = "auto",
        batch_size: int = 16,
        text_prefix: str = "A photo depicts ",
        image_path_key: str = "image_path",
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self.model_revision = model_revision
        self.cache_model_id = f"{model_name}@{model_revision or 'default'}"
        self.device_name = device
        self.batch_size = batch_size
        self.text_prefix = text_prefix
        self.image_path_key = image_path_key

    @staticmethod
    def _tensor(value: Any) -> Any:
        if hasattr(value, "pooler_output") and value.pooler_output is not None:
            return value.pooler_output
        if hasattr(value, "last_hidden_state"):
            return value.last_hidden_state[:, 0]
        if isinstance(value, tuple):
            return value[0]
        return value

    def evaluate(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        try:
            import torch
            import torch.nn.functional as functional
            from PIL import Image
            from transformers import AutoProcessor, CLIPModel
        except ImportError as error:
            raise MetricUnavailableError(
                "CLIPScore requires `pip install -e .[clipscore]`"
            ) from error

        image_paths: dict[str, Path] = {}
        for record in records:
            raw_path = record.metadata.get(self.image_path_key)
            if not isinstance(raw_path, str) or not raw_path:
                raise MetricUnavailableError(
                    f"CLIPScore requires metadata.{self.image_path_key} for sample {record.sample_id}"
                )
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise MetricUnavailableError(f"CLIPScore image is missing: {path}")
            image_paths[record.sample_id] = path

        requested_device = self.device_name
        device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
        if device == "auto":
            device = "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise MetricUnavailableError(f"requested CLIPScore device {device!r}, but CUDA is unavailable")

        with EmbeddingCache(self.cache_path) as cache:
            vectors: dict[str, tuple[list[float] | None, list[float] | None]] = {}
            missing_records: list[EvaluationRecord] = []
            for record in records:
                image_key = _image_key(self.cache_model_id, image_paths[record.sample_id])
                text_key = _text_key(self.cache_model_id, self.text_prefix, record.prediction)
                image_vector = cache.get(image_key)
                text_vector = cache.get(text_key)
                vectors[record.sample_id] = image_vector, text_vector
                if image_vector is None or text_vector is None:
                    missing_records.append(record)

            if missing_records:
                processor = AutoProcessor.from_pretrained(self.model_name, revision=self.model_revision)
                model = CLIPModel.from_pretrained(self.model_name, revision=self.model_revision).eval().to(device)
                for start in range(0, len(missing_records), self.batch_size):
                    batch = missing_records[start : start + self.batch_size]
                    images = [Image.open(image_paths[item.sample_id]).convert("RGB") for item in batch]
                    texts = [self.text_prefix + item.prediction for item in batch]
                    image_inputs = processor(images=images, return_tensors="pt")
                    text_inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
                    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
                    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
                    with torch.inference_mode():
                        image_features = self._tensor(model.get_image_features(**image_inputs))
                        text_features = self._tensor(model.get_text_features(**text_inputs))
                        image_features = functional.normalize(image_features, dim=-1).cpu()
                        text_features = functional.normalize(text_features, dim=-1).cpu()
                    for record, image_vector_tensor, text_vector_tensor in zip(
                        batch, image_features, text_features
                    ):
                        image_vector = image_vector_tensor.float().tolist()
                        text_vector = text_vector_tensor.float().tolist()
                        cache.put(_image_key(self.cache_model_id, image_paths[record.sample_id]), image_vector)
                        cache.put(
                            _text_key(self.cache_model_id, self.text_prefix, record.prediction),
                            text_vector,
                        )
                        vectors[record.sample_id] = image_vector, text_vector
                    for image in images:
                        image.close()

        per_sample: dict[str, dict[str, float]] = {}
        scores: list[float] = []
        for record in records:
            image_vector, text_vector = vectors[record.sample_id]
            if image_vector is None or text_vector is None:
                raise RuntimeError(f"CLIPScore failed to create embeddings for {record.sample_id}")
            cosine = sum(left * right for left, right in zip(image_vector, text_vector))
            score = 2.5 * max(cosine, 0.0)
            scores.append(score)
            per_sample[record.sample_id] = {"CLIPScore": score}
        return MetricResult(
            summary={"CLIPScore": sum(scores) / len(scores)},
            per_sample=per_sample,
            details={
                "model": self.model_name,
                "model_revision": self.model_revision,
                "formula": "2.5 * max(cosine(image_embedding, text_embedding), 0)",
                "text_prefix": self.text_prefix,
                "cache_path": str(self.cache_path.resolve()),
                "device": device,
            },
        )
