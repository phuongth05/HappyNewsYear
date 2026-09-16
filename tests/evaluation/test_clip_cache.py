from pathlib import Path

import pytest

from kric.evaluation.clipscore import EmbeddingCache


def test_embedding_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.sqlite3"
    with EmbeddingCache(path) as cache:
        assert cache.get("missing") is None
        cache.put("vector", [0.1, 0.2, 0.3])
        assert cache.get("vector") == pytest.approx([0.1, 0.2, 0.3])

    with EmbeddingCache(path) as reopened:
        assert reopened.get("vector") == pytest.approx([0.1, 0.2, 0.3])

