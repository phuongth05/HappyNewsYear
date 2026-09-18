"""Deterministic SQLite cache for atomic extraction results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Sequence

from .schema import AtomicEvidence


class AtomicEvidenceCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> "AtomicEvidenceCache":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS extraction_cache "
            "(cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.connection.commit()
        return self

    def __exit__(self, *_args) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def get(self, cache_key: str) -> list[AtomicEvidence] | None:
        if self.connection is None:
            raise RuntimeError("cache must be used as a context manager")
        row = self.connection.execute(
            "SELECT payload FROM extraction_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        raw = json.loads(row[0])
        if not isinstance(raw, list):
            raise ValueError("malformed cached extraction payload")
        return [AtomicEvidence.from_dict(item) for item in raw]

    def put(self, cache_key: str, units: Sequence[AtomicEvidence]) -> None:
        if self.connection is None:
            raise RuntimeError("cache must be used as a context manager")
        payload = json.dumps(
            [unit.to_dict() for unit in units], ensure_ascii=False, sort_keys=True
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO extraction_cache(cache_key, payload) VALUES (?, ?)",
            (cache_key, payload),
        )
        self.connection.commit()

