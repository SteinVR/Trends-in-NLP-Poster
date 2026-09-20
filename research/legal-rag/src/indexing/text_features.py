"""Shared lexical feature extraction helpers for indexing and retrieval."""

from __future__ import annotations

import re
from collections.abc import Iterable

_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&/-]{2,}\b")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})\b"
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def ordered_unique(values: Iterable[str]) -> list[str]:
    """Return non-empty values with stable order and duplicates removed."""

    return list(dict.fromkeys(value for value in values if value))


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms for lexical scoring."""

    return _TOKEN_RE.findall(text.lower())


def extract_entities(text: str, limit: int = 8) -> list[str]:
    """Extract upper-case entity-like spans from text."""

    entities = [match.group(0) for match in _ENTITY_RE.finditer(text)]
    return ordered_unique(entities)[:limit]


def extract_dates(text: str, limit: int = 6) -> list[str]:
    """Extract date-like spans from text."""

    dates = [match.group(0) for match in _DATE_RE.finditer(text)]
    return ordered_unique(dates)[:limit]
