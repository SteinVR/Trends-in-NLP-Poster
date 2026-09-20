"""Provider contracts for answer generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel


class AnsweringProvider(Protocol):
    """Minimal contract for schema-bound answering generation."""

    def generate_structured(
        self,
        *,
        question: str,
        answer_type: str,
        evidence_pages: list[dict[str, Any]],
        answer_schema: type[BaseModel],
        reasoning_effort: str,
    ) -> Mapping[str, Any]:
        """Generate a schema-bound structured answer."""


class StructuredOutputProvider(Protocol):
    """Minimal contract for generic schema-bound structured generation."""

    def generate_structured(
        self,
        *,
        schema_name: str,
        schema: dict[str, object],
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        reasoning_effort: str = "high",
    ) -> Mapping[str, Any]:
        """Generate a structured JSON payload that matches the provided schema."""
