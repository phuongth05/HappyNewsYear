"""Strict schema and stable identities for atomic caption claims."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

CLAIM_TYPES = (
    "entity",
    "object",
    "attribute",
    "relation",
    "event",
    "location",
    "time",
    "external_fact",
)
CLAIM_SCHEMA_VERSION = "atomic_caption_claims_schema_v1"
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    type: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("claim text must be non-empty")
        if self.type not in CLAIM_TYPES:
            raise ValueError(f"unsupported claim type: {self.type!r}")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _normalized(text: str) -> str:
    return _SPACE.sub(" ", text).strip().casefold()


def normalize_claims(payload: Any) -> list[Claim]:
    """Validate strict structured output and remove exact text duplicates stably."""

    if not isinstance(payload, Mapping) or set(payload) != {"claims"}:
        raise ValueError("claim response must contain only 'claims'")
    values = payload["claims"]
    if not isinstance(values, list):
        raise ValueError("claims must be a list")
    result: list[Claim] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or set(value) != {"text", "type"}:
            raise ValueError(f"claim {index} has an invalid shape")
        text = value["text"]
        claim_type = value["type"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"claim {index} has empty text")
        claim = Claim(_SPACE.sub(" ", text).strip(), str(claim_type))
        key = _normalized(claim.text)
        if key in seen:
            continue
        seen.add(key)
        result.append(claim)
    return result


def stable_claim_id(
    sample_id: str, caption_variant: str, index: int, text: str, claim_type: str
) -> str:
    material = "\x1f".join(
        [sample_id, caption_variant, str(index), text, claim_type, CLAIM_SCHEMA_VERSION]
    )
    return "cl_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def claims_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims"],
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "type"],
                    "properties": {
                        "text": {"type": "string", "minLength": 1},
                        "type": {"type": "string", "enum": list(CLAIM_TYPES)},
                    },
                },
            }
        },
    }


__all__ = [
    "CLAIM_SCHEMA_VERSION",
    "CLAIM_TYPES",
    "Claim",
    "claims_json_schema",
    "normalize_claims",
    "stable_claim_id",
]