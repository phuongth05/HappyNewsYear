"""Pinned sentence-transformer cosine matcher."""

from __future__ import annotations

from typing import Sequence

COSINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COSINE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class CosineMatcher:
    method = "semantic_cosine_v1"

    def __init__(
        self,
        model_name: str = COSINE_MODEL,
        revision: str = COSINE_REVISION,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.model = None

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            self.model_name, revision=self.revision, device=self.device
        )

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if self.model is None:
            self.load()
        if not pairs:
            return []
        evidence = [pair[0] for pair in pairs]
        claims = [pair[1] for pair in pairs]
        left = self.model.encode(
            evidence,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        right = self.model.encode(
            claims,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [float((a * b).sum()) for a, b in zip(left, right, strict=True)]

    def model_info(self) -> dict[str, object]:
        return {
            "method": self.method,
            "model": self.model_name,
            "revision": self.revision,
            "score": "cosine_similarity",
        }