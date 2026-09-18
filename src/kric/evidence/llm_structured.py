"""Structured, sentence-only atomic evidence extraction over compatible APIs."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


PROMPT_VERSION = "llm_structured_atomic_v1"
SCHEMA_VERSION = "atomic_propositions_schema_v1"
EVIDENCE_TYPES = (
    "entity",
    "object",
    "attribute",
    "relation",
    "event",
    "location",
    "time",
    "external_fact",
)

SYSTEM_PROMPT = """You extract atomic evidence from exactly one news sentence.

Rules:
1. Each proposition must express exactly one fact supported by the sentence.
2. Preserve named entities exactly. Do not add world knowledge or facts absent from the sentence.
3. Resolve a pronoun only when its antecedent is unambiguous within this same sentence.
4. Do not reinterpret cardinals, ages, distances, or quantities as time.
5. Do not infer an organization type unless the sentence states it.
6. Do not output keywords, fragments, duplicate facts, or multi-clause propositions.
7. Preserve the source meaning. Do not force a separate entity proposition merely because an entity is mentioned.
8. source_span_text must be a verbatim substring of the sentence that directly supports the proposition.
9. If no supported atomic proposition can be extracted, return an empty propositions list.

Return only the requested structured object."""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["propositions"],
    "properties": {
        "propositions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "type", "source_span_text"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "type": {"type": "string", "enum": list(EVIDENCE_TYPES)},
                    "source_span_text": {"type": "string", "minLength": 1},
                },
            },
        }
    },
}


class StructuredExtractionError(RuntimeError):
    pass


def stable_evidence_id(
    sample_id: str,
    source_sentence_id: str,
    proposition_index: int,
    text: str,
    evidence_type: str,
) -> str:
    value = "\x1f".join(
        [sample_id, source_sentence_id, str(proposition_index), text, evidence_type, PROMPT_VERSION]
    )
    return "ae_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def align_source_span(sentence: str, span_text: str) -> dict[str, Any]:
    start = sentence.find(span_text)
    if start < 0:
        return {
            "text": span_text,
            "start": None,
            "end": None,
            "alignment": "unresolved",
        }
    return {
        "text": span_text,
        "start": start,
        "end": start + len(span_text),
        "alignment": "exact",
    }


def validate_payload(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, dict) or set(payload) != {"propositions"}:
        raise StructuredExtractionError("response must contain only 'propositions'")
    propositions = payload["propositions"]
    if not isinstance(propositions, list):
        raise StructuredExtractionError("propositions must be a list")
    validated: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, proposition in enumerate(propositions):
        if not isinstance(proposition, dict) or set(proposition) != {
            "text",
            "type",
            "source_span_text",
        }:
            raise StructuredExtractionError(f"proposition {index} has an invalid shape")
        text = proposition["text"]
        evidence_type = proposition["type"]
        span = proposition["source_span_text"]
        if not isinstance(text, str) or not text.strip():
            raise StructuredExtractionError(f"proposition {index} has empty text")
        if evidence_type not in EVIDENCE_TYPES:
            raise StructuredExtractionError(f"proposition {index} has invalid type {evidence_type!r}")
        if not isinstance(span, str) or not span.strip():
            raise StructuredExtractionError(f"proposition {index} has empty source_span_text")
        item = (text.strip(), evidence_type, span)
        if item in seen:
            raise StructuredExtractionError(f"proposition {index} duplicates an earlier proposition")
        seen.add(item)
        validated.append({"text": item[0], "type": item[1], "source_span_text": item[2]})
    return validated


class StructuredResponseCache:
    """SQLite cache retaining every attempt and each validated final response."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS response_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    error TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS extraction_content_cache (
                    cache_key TEXT PRIMARY KEY,
                    content_json TEXT NOT NULL,
                    response_model TEXT,
                    response_id TEXT,
                    created_at REAL NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get_final(self, cache_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content_json FROM extraction_content_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def put_attempt(
        self,
        cache_key: str,
        attempt: int,
        request_body: Mapping[str, Any],
        response: Any = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO response_attempts"
                "(cache_key, attempt, request_json, response_json, error, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    cache_key,
                    attempt,
                    json.dumps(request_body, ensure_ascii=False, sort_keys=True),
                    None if response is None else json.dumps(response, ensure_ascii=False),
                    error,
                    time.time(),
                ),
            )

    def put_final(
        self,
        cache_key: str,
        content: Mapping[str, Any],
        response_model: str | None,
        response_id: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO extraction_content_cache VALUES (?, ?, ?, ?, ?)",
                (
                    cache_key,
                    json.dumps(content, ensure_ascii=False, sort_keys=True),
                    response_model,
                    response_id,
                    time.time(),
                ),
            )


@dataclass(frozen=True)
class LlmEndpointConfig:
    api_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    model_revision: str | None = None
    structured_output_mode: str = "json_schema"
    timeout_seconds: float = 120.0
    max_attempts: int = 3

    @property
    def chat_completions_url(self) -> str:
        base = self.api_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"


class StructuredAtomicExtractor:
    """Extract atomic evidence while sending only one source sentence per call."""

    method = PROMPT_VERSION

    def __init__(
        self,
        config: LlmEndpointConfig,
        cache_path: str | Path,
        post_json: Callable[[str, Mapping[str, str], Mapping[str, Any], float], Any] | None = None,
    ):
        if config.structured_output_mode not in {"json_schema", "json_object"}:
            raise ValueError("structured_output_mode must be json_schema or json_object")
        if config.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if config.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.config = config
        self.cache = StructuredResponseCache(cache_path)
        self._post_json = post_json or self._urllib_post_json
        self._stats = {
            "request_count": 0,
            "successful_request_count": 0,
            "cache_hit_count": 0,
            "retry_count": 0,
            "failed_request_count": 0,
            "malformed_response_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @staticmethod
    def _response_snapshot(response: Mapping[str, Any]) -> str | None:
        for key in ("model_snapshot", "model_version", "snapshot", "version"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _record_usage(self, response: Mapping[str, Any]) -> None:
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            return
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        if isinstance(input_tokens, int) and input_tokens >= 0:
            self._stats["total_input_tokens"] += input_tokens
        if isinstance(output_tokens, int) and output_tokens >= 0:
            self._stats["total_output_tokens"] += output_tokens

    def _cache_key(self, sentence: str) -> str:
        material = json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
                "model": self.config.model,
                "model_revision": self.config.model_revision,
                "api_url": self.config.api_url,
                "sentence": sentence,
                "structured_output_mode": self.config.structured_output_mode,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _materialize(
        self,
        content: Mapping[str, Any],
        *,
        sentence: str,
        sample_id: str,
        source_sentence_id: str,
        source_rank: int | None,
        ranking_score: float | None,
        cache_hit: bool,
    ) -> dict[str, Any]:
        propositions = validate_payload({"propositions": content.get("propositions")})
        units = []
        for index, proposition in enumerate(propositions):
            span = align_source_span(sentence, proposition["source_span_text"])
            units.append(
                {
                    "evidence_id": stable_evidence_id(
                        sample_id,
                        source_sentence_id,
                        index,
                        proposition["text"],
                        proposition["type"],
                    ),
                    "text": proposition["text"],
                    "type": proposition["type"],
                    "source_span": span,
                    "provenance": {
                        "sample_id": sample_id,
                        "source_sentence_id": source_sentence_id,
                        "source_rank": source_rank,
                        "ranking_score": ranking_score,
                        "source_sentence": sentence,
                    },
                }
            )
        return {
            "sample_id": sample_id,
            "source_sentence_id": source_sentence_id,
            "source_rank": source_rank,
            "ranking_score": ranking_score,
            "source_sentence": sentence,
            "extractor": PROMPT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "requested_model": self.config.model,
            "model_revision": self.config.model_revision,
            "response_model": content.get("response_model"),
            "response_snapshot": content.get("response_snapshot"),
            "response_id": content.get("response_id"),
            "propositions": units,
            "cache_hit": cache_hit,
        }

    def _request_body(self, sentence: str) -> dict[str, Any]:
        if self.config.structured_output_mode == "json_schema":
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "atomic_evidence",
                    "strict": True,
                    "schema": OUTPUT_SCHEMA,
                },
            }
        else:
            response_format = {"type": "json_object"}
        return {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ],
            "response_format": response_format,
        }

    @staticmethod
    def _content(response: Mapping[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise StructuredExtractionError("missing chat-completions message content") from exc
        if not isinstance(content, str):
            raise StructuredExtractionError("message content is not a string")
        return content

    @staticmethod
    def _urllib_post_json(
        url: str, headers: Mapping[str, str], body: Mapping[str, Any], timeout: float
    ) -> Any:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise StructuredExtractionError(f"HTTP {exc.code}: {detail}") from exc

    def extract(
        self,
        sentence: str,
        *,
        sample_id: str,
        source_sentence_id: str,
        source_rank: int | None = None,
        ranking_score: float | None = None,
    ) -> dict[str, Any]:
        if not sentence.strip():
            raise ValueError("source sentence must be non-empty")
        api_key = os.environ.get(self.config.api_key_env, "").strip()
        if not api_key:
            raise StructuredExtractionError(
                f"missing API credential in environment variable {self.config.api_key_env}"
            )
        cache_key = self._cache_key(sentence)
        cached = self.cache.get_final(cache_key)
        if cached is not None:
            self._stats["cache_hit_count"] += 1
            return self._materialize(
                cached,
                sentence=sentence,
                sample_id=sample_id,
                source_sentence_id=source_sentence_id,
                source_rank=source_rank,
                ranking_score=ranking_score,
                cache_hit=True,
            )

        body = self._request_body(sentence)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            response: Any = None
            self._stats["request_count"] += 1
            if attempt > 1:
                self._stats["retry_count"] += 1
            try:
                response = self._post_json(
                    self.config.chat_completions_url,
                    headers,
                    body,
                    self.config.timeout_seconds,
                )
                if isinstance(response, Mapping):
                    self._record_usage(response)
                payload = json.loads(self._content(response))
                propositions = validate_payload(payload)
                content = {
                    "propositions": propositions,
                    "response_model": response.get("model") if isinstance(response, dict) else None,
                    "response_snapshot": (
                        self._response_snapshot(response)
                        if isinstance(response, Mapping)
                        else None
                    ),
                    "response_id": response.get("id") if isinstance(response, dict) else None,
                }
                self.cache.put_attempt(cache_key, attempt, body, response=response)
                self.cache.put_final(
                    cache_key, content, content["response_model"], content["response_id"]
                )
                self._stats["successful_request_count"] += 1
                return self._materialize(
                    content,
                    sentence=sentence,
                    sample_id=sample_id,
                    source_sentence_id=source_sentence_id,
                    source_rank=source_rank,
                    ranking_score=ranking_score,
                    cache_hit=False,
                )
            except (StructuredExtractionError, json.JSONDecodeError, TypeError) as exc:
                last_error = exc
                self._stats["failed_request_count"] += 1
                if response is not None:
                    self._stats["malformed_response_count"] += 1
                self.cache.put_attempt(cache_key, attempt, body, response=response, error=str(exc))
        raise StructuredExtractionError(
            f"failed after {self.config.max_attempts} attempts: {last_error}"
        )
