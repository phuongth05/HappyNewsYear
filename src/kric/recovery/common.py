from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

RECOVERY_SUFFIX = "_regenerated_v2"
FROZEN_SELECTED_SHA256 = "d8bdfcef4721f1eb2f6346e4a0e9dbf1ecfa825d4db724431415c34ce720a4f2"
HISTORICAL_B2_PREDICTIONS_SHA256 = "858544a50f6ffc66978d9ea50d1b610cf18453d7a047e6bcd47c6eae91062934"
ORIGINAL_M3_TOKEN_MATCHED_PREDICTIONS_SHA256 = "c3b500c34bf370f1d54a4807ea54bde4cc7fd41ef247b6150acd608f88b35fdb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(dict(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def require_recovery_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if RECOVERY_SUFFIX not in resolved.name:
        raise ValueError(f"recovery output directory must contain {RECOVERY_SUFFIX}: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty recovery output: {resolved}")
    return resolved


def unique_ids(rows: Sequence[Mapping[str, Any]], label: str) -> list[str]:
    ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains empty or duplicate sample IDs")
    return ids


def unique_claim_keys(rows: Sequence[Mapping[str, Any]], label: str) -> set[tuple[str, str]]:
    keys = {(str(row.get("sample_id", "")), str(row.get("claim_id", ""))) for row in rows}
    if any(not all(key) for key in keys) or len(keys) != len(rows):
        raise ValueError(f"{label} contains empty or duplicate claim keys")
    return keys
