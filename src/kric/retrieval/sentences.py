"""Deterministic, reference-free sentence-ranking baselines for B2."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from kric.evaluation.clipscore import EmbeddingCache


_TOKEN_PATTERN = re.compile(r"[\w]+(?:['’][\w]+)?", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class SentenceCandidate:
    sentence_id: int
    text: str

    def __post_init__(self) -> None:
        if self.sentence_id < 0:
            raise ValueError("sentence_id must be non-negative")
        if not self.text.strip():
            raise ValueError("sentence text must not be empty")


@dataclass(frozen=True, slots=True)
class RankedSentence:
    sentence_id: int
    text: str
    score: float
    rank: int


def lexical_tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text)]


def bm25_rank(
    sentences: Sequence[SentenceCandidate],
    query: str,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[RankedSentence]:
    """Rank article sentences with standard BM25 and deterministic tie-breaking."""

    if not sentences:
        return []
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
    documents = [lexical_tokens(sentence.text) for sentence in sentences]
    query_counts = Counter(lexical_tokens(query))
    average_length = sum(len(document) for document in documents) / len(documents)
    document_frequency = Counter(
        token for document in documents for token in set(document)
    )
    corpus_size = len(documents)
    scored: list[tuple[SentenceCandidate, float]] = []
    for sentence, document in zip(sentences, documents, strict=True):
        frequencies = Counter(document)
        length_norm = 1 - b + b * len(document) / (average_length or 1.0)
        score = 0.0
        for token, query_frequency in query_counts.items():
            frequency = frequencies[token]
            if not frequency:
                continue
            df = document_frequency[token]
            inverse_document_frequency = math.log(
                1.0 + (corpus_size - df + 0.5) / (df + 0.5)
            )
            score += query_frequency * inverse_document_frequency * (
                frequency * (k1 + 1) / (frequency + k1 * length_norm)
            )
        scored.append((sentence, score))
    scored.sort(key=lambda item: (-item[1], item[0].sentence_id))
    return [
        RankedSentence(sentence.sentence_id, sentence.text, float(score), rank)
        for rank, (sentence, score) in enumerate(scored, start=1)
    ]


def _digest(*parts: object) -> str:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).hexdigest()


class ClipSentenceRanker:
    """Rank sentences by cosine similarity to the image in pinned CLIP space."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        model_name: str,
        model_revision: str,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        if not model_name or not model_revision:
            raise ValueError("semantic retrieval requires a pinned model and revision")
        if batch_size <= 0:
            raise ValueError("semantic batch_size must be positive")
        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self.model_revision = model_revision
        self.cache_model_id = f"{model_name}@{model_revision}"
        self.requested_device = device
        self.batch_size = batch_size
        self.device = "unloaded"
        self.processor: Any | None = None
        self.model: Any | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, CLIPModel

        self.device = (
            "cuda"
            if self.requested_device == "auto" and torch.cuda.is_available()
            else "cpu"
            if self.requested_device == "auto"
            else self.requested_device
        )
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("semantic retrieval requested CUDA but CUDA is unavailable")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, revision=self.model_revision
        )
        self.model = (
            CLIPModel.from_pretrained(self.model_name, revision=self.model_revision)
            .eval()
            .to(self.device)
        )

    def _image_key(self, image_path: Path) -> str:
        stat = image_path.stat()
        return _digest(
            "b2-image",
            self.cache_model_id,
            image_path.resolve(),
            stat.st_size,
            stat.st_mtime_ns,
        )

    def _text_key(self, text: str) -> str:
        return _digest("b2-sentence", self.cache_model_id, text)

    @staticmethod
    def _tensor(value: Any) -> Any:
        if hasattr(value, "pooler_output") and value.pooler_output is not None:
            return value.pooler_output
        if hasattr(value, "last_hidden_state"):
            return value.last_hidden_state[:, 0]
        if isinstance(value, tuple):
            return value[0]
        return value

    def rank(
        self, image_path: str | Path, sentences: Sequence[SentenceCandidate]
    ) -> list[RankedSentence]:
        if not sentences:
            return []
        self.load()
        import torch
        import torch.nn.functional as functional
        from PIL import Image

        path = Path(image_path).expanduser().resolve()
        with EmbeddingCache(self.cache_path) as cache:
            image_key = self._image_key(path)
            image_vector = cache.get(image_key)
            if image_vector is None:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                try:
                    inputs = self.processor(images=[image], return_tensors="pt")
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    with torch.inference_mode():
                        features = self._tensor(self.model.get_image_features(**inputs))
                        features = functional.normalize(features, dim=-1)[0].float().cpu()
                    image_vector = features.tolist()
                    cache.put(image_key, image_vector)
                finally:
                    image.close()

            text_vectors: dict[int, list[float]] = {}
            missing: list[SentenceCandidate] = []
            for sentence in sentences:
                vector = cache.get(self._text_key(sentence.text))
                if vector is None:
                    missing.append(sentence)
                else:
                    text_vectors[sentence.sentence_id] = vector
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                inputs = self.processor(
                    text=[sentence.text for sentence in batch],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    features = self._tensor(self.model.get_text_features(**inputs))
                    features = functional.normalize(features, dim=-1).float().cpu()
                for sentence, feature in zip(batch, features, strict=True):
                    vector = feature.tolist()
                    cache.put(self._text_key(sentence.text), vector)
                    text_vectors[sentence.sentence_id] = vector

        scored = [
            (
                sentence,
                sum(
                    left * right
                    for left, right in zip(
                        image_vector, text_vectors[sentence.sentence_id], strict=True
                    )
                ),
            )
            for sentence in sentences
        ]
        scored.sort(key=lambda item: (-item[1], item[0].sentence_id))
        return [
            RankedSentence(sentence.sentence_id, sentence.text, float(score), rank)
            for rank, (sentence, score) in enumerate(scored, start=1)
        ]

    def model_info(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "revision": self.model_revision,
            "device": self.device,
            "batch_size": self.batch_size,
            "cache_path": str(self.cache_path.resolve()),
            "score": "cosine(image_embedding, sentence_embedding)",
        }
