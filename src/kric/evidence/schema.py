"""Strict schemas for frozen sentence inputs and atomic evidence outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


EVIDENCE_TYPES = frozenset(
    {
        "entity",
        "object",
        "attribute",
        "relation",
        "event",
        "location",
        "time",
        "external_fact",
    }
)


@dataclass(frozen=True, slots=True)
class SourceSpan:
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("source span must satisfy 0 <= start < end")
        if not self.text:
            raise ValueError("source span text must not be empty")


@dataclass(frozen=True, slots=True)
class FrozenSentence:
    sample_id: str
    sentence_id: int
    text: str
    source_rank: int
    ranking_score: float

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.sentence_id < 0:
            raise ValueError("sentence_id must be non-negative")
        if not self.text.strip():
            raise ValueError("sentence text must not be empty")
        if self.source_rank <= 0:
            raise ValueError("source_rank must be positive")


def stable_evidence_id(
    sample_id: str,
    sentence_id: int,
    evidence_type: str,
    text: str,
    source_span: SourceSpan | None,
) -> str:
    payload = {
        "sample_id": sample_id,
        "sentence_id": sentence_id,
        "type": evidence_type,
        "text": " ".join(text.split()),
        "span": asdict(source_span) if source_span else None,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return f"ae_{digest}"


@dataclass(frozen=True, slots=True)
class AtomicEvidence:
    evidence_id: str
    text: str
    type: str
    source_sentence_id: int
    source_span: SourceSpan | None
    source_rank: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence_id must not be empty")
        if not self.text.strip():
            raise ValueError("evidence text must not be empty")
        if self.type not in EVIDENCE_TYPES:
            raise ValueError(f"unsupported evidence type: {self.type!r}")
        if self.source_sentence_id < 0:
            raise ValueError("source_sentence_id must be non-negative")
        if self.source_rank <= 0:
            raise ValueError("source_rank must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "type": self.type,
            "source_sentence_id": self.source_sentence_id,
            "source_span": asdict(self.source_span) if self.source_span else None,
            "source_rank": self.source_rank,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AtomicEvidence":
        required = {
            "evidence_id",
            "text",
            "type",
            "source_sentence_id",
            "source_rank",
            "metadata",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(f"malformed evidence unit; missing fields: {sorted(missing)}")
        raw_span = value.get("source_span")
        span = None
        if raw_span is not None:
            if not isinstance(raw_span, Mapping):
                raise ValueError("source_span must be an object or null")
            span = SourceSpan(
                start=int(raw_span["start"]),
                end=int(raw_span["end"]),
                text=str(raw_span["text"]),
            )
        metadata = value["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            evidence_id=str(value["evidence_id"]),
            text=str(value["text"]),
            type=str(value["type"]),
            source_sentence_id=int(value["source_sentence_id"]),
            source_span=span,
            source_rank=int(value["source_rank"]),
            metadata=dict(metadata),
        )

