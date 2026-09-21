"""Structured caption-only LLM decomposition into atomic claims."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from kric.evidence.llm_structured import (
    LlmEndpointConfig,
    StructuredAtomicExtractor,
    StructuredExtractionError,
    StructuredResponseCache,
)

from .schema import (
    CLAIM_SCHEMA_VERSION,
    claims_json_schema,
    normalize_claims,
    stable_claim_id,
)

PROMPT_VERSION = "llm_structured_caption_claims_v1"
RETRY_FORMAT_REMINDER = (
    "Return a complete valid JSON object matching the schema. Do not truncate."
)
SYSTEM_PROMPT = """Decompose exactly one generated news-image caption into atomic factual claims.

Rules:
1. Each claim must express exactly one factual assertion made by the caption.
2. Preserve the caption's factual meaning. Do not add, correct, verify, or rewrite information.
3. Do not infer missing subjects, entities, locations, times, relations, or background facts.
4. Keep named entities exactly as stated in the caption.
5. Split conjunctions only when they assert separable facts.
6. Do not output duplicate or near-duplicate claims.
7. A purely stylistic phrase that makes no factual assertion is not a claim.
8. If the caption contains no factual claim, return an empty claims list.

Return only the requested structured object."""


class StructuredClaimExtractor:
    """Caption-only extractor with content caching and current-input provenance."""

    method = PROMPT_VERSION

    def __init__(
        self,
        config: LlmEndpointConfig,
        cache_path: str | Path,
        post_json: Callable[[str, Mapping[str, str], Mapping[str, Any], float], Any]
        | None = None,
    ) -> None:
        if config.structured_output_mode not in {"json_schema", "json_object"}:
            raise ValueError("structured_output_mode must be json_schema or json_object")
        if config.model == "gpt-5.6-terra" and config.temperature is not None:
            raise ValueError("gpt-5.6-terra requires provider-default temperature")
        if config.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.config = config
        self.cache = StructuredResponseCache(cache_path)
        self._post_json = post_json or StructuredAtomicExtractor._urllib_post_json
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
    def _normalized_caption(caption: str) -> str:
        return " ".join(caption.split())

    def _cache_key(self, caption: str) -> str:
        material = {
            "prompt_version": PROMPT_VERSION,
            "schema_version": CLAIM_SCHEMA_VERSION,
            "model": self.config.model,
            "model_revision": self.config.model_revision,
            "api_url": self.config.api_url,
            "structured_output_mode": self.config.structured_output_mode,
            "temperature": (
                "provider_default"
                if self.config.temperature is None
                else self.config.temperature
            ),
            "caption": self._normalized_caption(caption),
        }
        return hashlib.sha256(
            json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _request_body(
        self, caption: str, *, include_retry_format_reminder: bool = False
    ) -> dict[str, Any]:
        if self.config.structured_output_mode == "json_schema":
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "atomic_caption_claims",
                    "strict": True,
                    "schema": claims_json_schema(),
                },
            }
        else:
            response_format = {"type": "json_object"}
        system = SYSTEM_PROMPT
        if include_retry_format_reminder:
            system += "\n\n" + RETRY_FORMAT_REMINDER
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": caption},
            ],
            "response_format": response_format,
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        return body

    @staticmethod
    def _content(response: Mapping[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise StructuredExtractionError(
                "missing chat-completions message content"
            ) from error
        if not isinstance(content, str):
            raise StructuredExtractionError("message content is not a string")
        return content

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

    @staticmethod
    def _response_snapshot(response: Mapping[str, Any]) -> str | None:
        for key in ("model_snapshot", "model_version", "snapshot", "version"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _materialize(
        self,
        content: Mapping[str, Any],
        *,
        caption: str,
        sample_id: str,
        caption_variant: str,
        cache_hit: bool,
    ) -> dict[str, Any]:
        claims = normalize_claims({"claims": content.get("claims")})
        return {
            "sample_id": sample_id,
            "caption_variant": caption_variant,
            "caption": caption,
            "extractor": PROMPT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "schema_version": CLAIM_SCHEMA_VERSION,
            "requested_model": self.config.model,
            "model_revision": self.config.model_revision,
            "response_model": content.get("response_model"),
            "response_snapshot": content.get("response_snapshot"),
            "response_id": content.get("response_id"),
            "cache_hit": cache_hit,
            "claims": [
                {
                    "claim_id": stable_claim_id(
                        sample_id, caption_variant, index, claim.text, claim.type
                    ),
                    "text": claim.text,
                    "type": claim.type,
                }
                for index, claim in enumerate(claims)
            ],
        }

    def extract(
        self, caption: str, *, sample_id: str, caption_variant: str
    ) -> dict[str, Any]:
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("caption must be non-empty")
        api_key = os.environ.get(self.config.api_key_env, "").strip()
        if not api_key:
            raise StructuredExtractionError(
                f"missing API credential in environment variable {self.config.api_key_env}"
            )
        cache_key = self._cache_key(caption)
        cached = self.cache.get_final(cache_key)
        if cached is not None:
            self._stats["cache_hit_count"] += 1
            return self._materialize(
                cached,
                caption=caption,
                sample_id=sample_id,
                caption_variant=caption_variant,
                cache_hit=True,
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        reminder = False
        for attempt in range(1, self.config.max_attempts + 1):
            response: Any = None
            body = self._request_body(
                caption, include_retry_format_reminder=reminder
            )
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
                if not isinstance(response, Mapping):
                    raise StructuredExtractionError("API response must be an object")
                self._record_usage(response)
                payload = json.loads(self._content(response))
                claims = normalize_claims(payload)
                content = {
                    "claims": [claim.to_dict() for claim in claims],
                    "response_model": response.get("model"),
                    "response_snapshot": self._response_snapshot(response),
                    "response_id": response.get("id"),
                }
                self.cache.put_attempt(cache_key, attempt, body, response=response)
                self.cache.put_final(
                    cache_key,
                    content,
                    content["response_model"],
                    content["response_id"],
                )
                self._stats["successful_request_count"] += 1
                return self._materialize(
                    content,
                    caption=caption,
                    sample_id=sample_id,
                    caption_variant=caption_variant,
                    cache_hit=False,
                )
            except (StructuredExtractionError, json.JSONDecodeError, TypeError, ValueError) as error:
                last_error = error
                self._stats["failed_request_count"] += 1
                if response is not None:
                    self._stats["malformed_response_count"] += 1
                self.cache.put_attempt(
                    cache_key, attempt, body, response=response, error=str(error)
                )
                if isinstance(error, json.JSONDecodeError):
                    reminder = True
        raise StructuredExtractionError(
            f"failed after {self.config.max_attempts} attempts: {last_error}"
        )


__all__ = ["PROMPT_VERSION", "StructuredClaimExtractor"]