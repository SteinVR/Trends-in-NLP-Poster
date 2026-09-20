"""Concrete Codex-backed providers for answering and schema-bound generation."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from src.common.runtime_logging import get_logger, log_event

_CANONICAL_UNANSWERABLE_FREE_TEXT = "There is no information on this question in the provided documents."

DEFAULT_CODEX_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CODEX_MODEL = "gpt-5.4-mini"
DEFAULT_PROVIDER_CONFIDENCE = 0.5
_MAX_RETRIES = 6
_RETRY_BACKOFF_SECONDS = (1.0, 3.0, 8.0, 15.0, 20.0, 30.0)
DEFAULT_MAX_OUTPUT_TOKENS = 2048
LOGGER = get_logger("providers.codex")
_SYSTEM_PROMPT_RESOURCE = ("resources", "prompts", "answering", "system.md")


def _load_structured_system_prompt() -> str:
    """Load the schema-bound answering system prompt from packaged resources."""

    template = resources.files("src").joinpath(*_SYSTEM_PROMPT_RESOURCE).read_text(encoding="utf-8")
    return template.replace(
        "{canonical_unanswerable_free_text}",
        _CANONICAL_UNANSWERABLE_FREE_TEXT,
    ).strip()


_STRUCTURED_SYSTEM_PROMPT = _load_structured_system_prompt()


@dataclass(slots=True)
class CodexAnsweringProvider:
    """HTTP provider that calls a Codex-compatible `/responses` endpoint for structured answering."""

    model: str = DEFAULT_CODEX_MODEL
    base_url: str = DEFAULT_CODEX_BASE_URL
    api_key: str | None = None
    timeout_seconds: float = 60.0
    user_agent: str = "agentic-rag-challenge/0.1.0"
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    transport: httpx.BaseTransport | None = None
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Execute `__post_init__`."""
        token = _resolve_provider_token(self.api_key)
        if not token:
            raise ValueError(
                "Codex provider token is missing. Set CODEX_OAUTH_TOKEN, OPENAI_API_KEY, or pass api_key explicitly."
            )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive.")

        self._client = _build_http_client(
            token=token,
            base_url=self.base_url,
            user_agent=self.user_agent,
            timeout_seconds=self.timeout_seconds,
            transport=self.transport,
        )

    def generate_structured(
        self,
        *,
        question: str,
        answer_type: str,
        evidence_pages: list[dict[str, Any]],
        answer_schema: type[BaseModel],
        reasoning_effort: str,
    ) -> dict[str, Any]:
        """Execute `generate_structured`."""
        started_at = time.perf_counter()
        normalized_effort = str(reasoning_effort).strip().lower() or "high"
        schema_name = f"{answer_type}_answer"
        schema_payload = _strict_openai_schema(answer_schema.model_json_schema())

        request_body = {
            "model": self.model,
            "reasoning": {"effort": normalized_effort},
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": _STRUCTURED_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _build_structured_user_prompt(
                                question=question,
                                answer_type=answer_type,
                                evidence_pages=evidence_pages,
                            ),
                        }
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema_payload,
                    "strict": True,
                }
            },
            "max_output_tokens": self.max_output_tokens,
        }
        last_exc: Exception | None = None
        for _attempt in range(_MAX_RETRIES + 1):
            response = _post_with_retry(self._client, "responses", request_body)
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Provider response must be a JSON object.")
            try:
                response_text = _extract_response_text(payload)
                structured = _parse_structured_payload(response_text)
            except ValueError as exc:
                last_exc = exc
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "provider.structured.parse_retry",
                    "Retrying due to malformed structured output.",
                    stage="answering",
                )
                time.sleep(1.0)
                continue
            # Normalize unanswerable outputs to match schema contracts before validation.
            if isinstance(structured, dict) and not structured.get("is_answerable", True):
                if answer_type == "free_text":
                    structured["final_answer"] = _CANONICAL_UNANSWERABLE_FREE_TEXT
                else:
                    structured["final_answer"] = None
            try:
                validated = answer_schema.model_validate(structured)
            except ValidationError as exc:
                last_exc = exc
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "provider.structured.validation_retry",
                    "Retrying due to schema validation failure.",
                    stage="answering",
                )
                time.sleep(1.0)
                continue
            break
        else:
            raise ValueError("Structured provider output failed after retries.") from last_exc

        result = dict(validated.model_dump())
        result["confidence"] = _coerce_confidence(structured.get("confidence", DEFAULT_PROVIDER_CONFIDENCE))
        log_event(
            LOGGER,
            logging.DEBUG,
            "provider.generate_structured.complete",
            "Structured provider response received.",
            stage="answering",
            model=self.model,
            answer_type=answer_type,
            evidence_page_count=len(evidence_pages),
            reasoning_effort=normalized_effort,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
        )
        return result

    def close(self) -> None:
        """Execute `close`."""
        self._client.close()

    def __enter__(self) -> "CodexAnsweringProvider":
        """Execute `__enter__`."""
        return self

    def __exit__(self, *_: object) -> None:
        """Execute `__exit__`."""
        self.close()


@dataclass(slots=True)
class CodexStructuredOutputProvider:
    """HTTP provider that requests schema-bound structured outputs."""

    model: str = DEFAULT_CODEX_MODEL
    base_url: str = DEFAULT_CODEX_BASE_URL
    api_key: str | None = None
    timeout_seconds: float = 60.0
    user_agent: str = "agentic-rag-challenge/0.1.0"
    transport: httpx.BaseTransport | None = None
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Execute `__post_init__`."""
        token = _resolve_provider_token(self.api_key)
        if not token:
            raise ValueError(
                "Codex provider token is missing. Set CODEX_OAUTH_TOKEN, OPENAI_API_KEY, or pass api_key explicitly."
            )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self._client = _build_http_client(
            token=token,
            base_url=self.base_url,
            user_agent=self.user_agent,
            timeout_seconds=self.timeout_seconds,
            transport=self.transport,
        )

    def generate_structured(
        self,
        *,
        schema_name: str,
        schema: dict[str, object],
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        reasoning_effort: str = "high",
    ) -> dict[str, Any]:
        """Execute `generate_structured`."""
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive.")
        request_body = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            "reasoning": {"effort": reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            "max_output_tokens": max_output_tokens,
        }

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            response = _post_with_retry(self._client, "responses", request_body)
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Provider response must be a JSON object.")
            try:
                response_text = _extract_response_text(payload)
                return _parse_structured_payload(response_text)
            except ValueError as exc:
                last_exc = exc
                if attempt >= _MAX_RETRIES:
                    break
                backoff = _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "provider.structured_output.parse_retry",
                    "Retrying structured output request due to malformed JSON payload.",
                    stage="provider",
                    attempt=attempt + 1,
                    backoff_seconds=backoff,
                    schema_name=schema_name,
                )
                time.sleep(backoff)

        raise ValueError("Provider structured payload is not valid JSON.") from last_exc

    def close(self) -> None:
        """Execute `close`."""
        self._client.close()

    def __enter__(self) -> "CodexStructuredOutputProvider":
        """Execute `__enter__`."""
        return self

    def __exit__(self, *_: object) -> None:
        """Execute `__exit__`."""
        self.close()


def _build_http_client(
    *,
    token: str,
    base_url: str,
    user_agent: str,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> httpx.Client:
    """Build http client."""
    return httpx.Client(
        base_url=base_url.rstrip("/") + "/",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        },
        timeout=httpx.Timeout(timeout_seconds),
        follow_redirects=True,
        transport=transport,
    )


def _resolve_provider_token(explicit_token: str | None) -> str:
    """Resolve provider token."""
    candidates = [
        explicit_token,
        os.getenv("CODEX_OAUTH_TOKEN"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("CODEX_API_KEY"),
    ]
    for candidate in candidates:
        token = (candidate or "").strip()
        if token:
            return token
    return ""


def _format_page_reference(page: dict[str, Any]) -> str:
    """Format page reference."""
    page_numbers = page.get("page_numbers")
    if isinstance(page_numbers, list):
        normalized_page_numbers: list[str] = []
        for page_number in page_numbers:
            try:
                parsed_page = int(page_number)
            except (TypeError, ValueError):
                continue
            if parsed_page > 0:
                normalized_page_numbers.append(str(parsed_page))
        return ",".join(normalized_page_numbers)

    page_number = page.get("page_number")
    if page_number is None:
        return ""
    try:
        parsed_page = int(page_number)
    except (TypeError, ValueError):
        return ""
    return str(parsed_page) if parsed_page > 0 else ""


def _strict_openai_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure a Pydantic-generated JSON schema is OpenAI strict-mode compatible."""
    schema = dict(schema)
    if "properties" in schema:
        schema["required"] = sorted(schema["properties"])
    return schema


def _build_structured_user_prompt(*, question: str, answer_type: str, evidence_pages: list[dict[str, Any]]) -> str:
    """Build structured user prompt."""
    evidence_lines: list[str] = []
    for index, page in enumerate(evidence_pages, start=1):
        doc_id = str(page.get("doc_id") or "").strip()
        page_ref = _format_page_reference(page)
        text = _normalize_whitespace(str(page.get("text") or ""))
        evidence_lines.append(f"[{index}] doc_id={doc_id}; pages={page_ref}; text={text}")

    evidence_blob = "\n".join(evidence_lines) if evidence_lines else "(no evidence)"
    return (
        f"Answer type: {_normalize_whitespace(answer_type)}\n"
        f"Question: {_normalize_whitespace(question)}\n"
        "Evidence:\n"
        f"{evidence_blob}"
    )


def _extract_response_text(payload: dict[str, Any]) -> str:
    """Extract response text."""
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = payload.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            # Skip reasoning items
            if item.get("type") == "reasoning":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                # Responses API uses "output_text" type with "text" key
                text = content_item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks)

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                text_chunks: list[str] = []
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        text_chunks.append(part["text"])
                if text_chunks:
                    return "\n".join(text_chunks)

    raise ValueError("Provider response does not contain readable text output.")


def _parse_structured_payload(response_text: str) -> dict[str, Any]:
    """Parse structured payload."""
    normalized = _normalize_whitespace(response_text)
    if not normalized:
        raise ValueError("Provider returned an empty structured payload.")
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ValueError("Provider structured payload must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Provider structured payload must be a JSON object.")
    return parsed


def _coerce_confidence(raw_confidence: Any) -> float:
    """Execute `_coerce_confidence`."""
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return DEFAULT_PROVIDER_CONFIDENCE
    if not math.isfinite(confidence):
        return DEFAULT_PROVIDER_CONFIDENCE
    if confidence < 0.0 or confidence > 1.0:
        return DEFAULT_PROVIDER_CONFIDENCE
    return confidence


def _normalize_whitespace(value: str) -> str:
    """Normalize whitespace."""
    return " ".join(value.split())


def _post_with_retry(
    client: httpx.Client,
    path: str,
    json_body: dict[str, Any],
    *,
    max_retries: int = _MAX_RETRIES,
) -> httpx.Response:
    """POST with retry on transient 5xx / connection errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.post(path, json=json_body)
            if response.status_code < 500:
                response.raise_for_status()
                return response
            # 5xx — retry
            last_exc = httpx.HTTPStatusError(
                f"Server error {response.status_code}",
                request=response.request,
                response=response,
            )
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            last_exc = exc
        if attempt < max_retries:
            backoff = _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]
            log_event(
                LOGGER,
                logging.WARNING,
                "provider.retry",
                "Retrying provider request after transient error.",
                stage="answering",
                attempt=attempt + 1,
                backoff_seconds=backoff,
            )
            time.sleep(backoff)
    raise last_exc  # type: ignore[misc]
