"""Text normalization and scoring helpers for answering extraction."""

from __future__ import annotations

import re
import unicodedata

from .service_constants import (
    _ENTITY_LIST_ITEM_PATTERN,
    _LEGAL_ROLE_PATTERN,
    _NAME_PATTERN,
    _ROLE_ENTITY_EXTRACTOR,
    _SINGLE_TOKEN_ENTITY_EXCLUSIONS,
    _STOPWORDS,
    _UPPER_ENTITY_PATTERN,
)


def _keyword_tokens(text: str) -> set[str]:
    """Execute `_keyword_tokens`.

    Args:
        text: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 1 and token not in _STOPWORDS
    }


def _context_window(text: str, start: int, end: int, *, radius: int = 64) -> str:
    """Execute `_context_window`.

    Args:
        text: Input parameter.
        start: Input parameter.
        end: Input parameter.
        radius: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right]


def _sentence_for_span(text: str, start: int, end: int) -> str:
    """Execute `_sentence_for_span`.

    Args:
        text: Input parameter.
        start: Input parameter.
        end: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    sentence_start = text.rfind(".", 0, start)
    sentence_start = 0 if sentence_start == -1 else sentence_start + 1
    sentence_end = text.find(".", end)
    sentence_end = len(text) if sentence_end == -1 else sentence_end + 1
    return text[sentence_start:sentence_end]


def _score_context(*, context: str, question_tokens: set[str]) -> float:
    """Execute `_score_context`.

    Args:
        context: Input parameter.
        question_tokens: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    context_tokens = _keyword_tokens(context)
    if not question_tokens:
        return 0.0
    overlap = question_tokens & context_tokens
    return float(len(overlap))


def _sentences(text: str) -> list[str]:
    """Execute `_sentences`.

    Args:
        text: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    sentences = re.split(r"(?<=[.!?])\s+", _normalize_whitespace(text))
    return [sentence for sentence in sentences if sentence]


def _grounding_clauses(text: str) -> list[str]:
    """Execute `_grounding_clauses`.

    Args:
        text: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    normalized = _normalize_whitespace(text)
    fragments = re.split(r"(?i)\b(?:because|since|due to|as|therefore|thus|but)\b|[;:]", normalized)
    return [fragment.strip(" ,.") for fragment in fragments if fragment.strip(" ,.")]


def _clean_verbose_name_answer(raw: str) -> str:
    """Strip verbose explanatory suffixes from structured solver name answers.

    E.g. "SCT 295/2025 has the earlier issue date: 2025-12-10..." → "SCT 295/2025"
    """
    normalized = _normalize_whitespace(raw)
    # Try extracting a DIFC-style case reference at the start.
    case_match = re.match(
        r"^((?:CFI|SCT|ARB|DEC|TCD|ENF|CA)\s*\d{1,4}\s*/\s*\d{2,4})\b",
        normalized,
        re.IGNORECASE,
    )
    if case_match:
        return _normalize_whitespace(case_match.group(1))
    # Try extracting a "Claim No." style reference.
    claim_match = re.match(
        r"^(Claim\s+No\.?\s*[A-Z0-9/_-]+(?:\s*/\s*\d+)?)\b",
        normalized,
        re.IGNORECASE,
    )
    if claim_match:
        return _normalize_whitespace(claim_match.group(1))
    # Truncate at common verbose suffixes.
    for sep in (" has ", " is ", " was ", " (", " — "):
        idx = normalized.find(sep)
        if idx > 0:
            candidate = normalized[:idx].strip()
            if len(candidate) >= 3:
                return candidate
    return normalized


def _normalize_whitespace(value: str) -> str:
    """Normalize whitespace.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _normalize_entity_candidate(value: str) -> str:
    """Normalize entity candidate.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    normalized = _normalize_whitespace(value)
    return normalized.rstrip(".,;:")


def _entity_matches(text: str) -> list[tuple[str, int, int]]:
    """Execute `_entity_matches`.

    Args:
        text: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    matches: list[tuple[str, int, int]] = []
    seen_spans: set[tuple[int, int]] = set()
    for pattern in (_NAME_PATTERN, _UPPER_ENTITY_PATTERN):
        for match in pattern.finditer(text):
            candidate = _normalize_entity_candidate(match.group(0))
            if not candidate:
                continue
            span = (match.start(), match.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            matches.append((candidate, match.start(), match.end()))

    for role_match in _ROLE_ENTITY_EXTRACTOR.finditer(text):
        entity_block = role_match.group("entities")
        block_start = role_match.start("entities")
        for item in _ENTITY_LIST_ITEM_PATTERN.finditer(entity_block):
            candidate = _normalize_entity_candidate(item.group(0))
            if not _is_safe_single_token_entity(candidate=candidate, context=text, start=block_start + item.start()):
                continue
            span = (block_start + item.start(), block_start + item.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            matches.append((candidate, span[0], span[1]))

    matches.sort(key=lambda item: (item[1], item[2]))
    return matches


def _is_safe_single_token_entity(*, candidate: str, context: str, start: int) -> bool:
    """Return whether safe single token entity is true.

    Args:
        candidate: Input parameter.
        context: Input parameter.
        start: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    parts = candidate.split()
    if len(parts) > 1:
        return True
    token = parts[0].rstrip(".")
    if token in _SINGLE_TOKEN_ENTITY_EXCLUSIONS:
        return False
    if len(token) < 3:
        return False
    prefix = context[max(0, start - 48) : start]
    return bool(_LEGAL_ROLE_PATTERN.search(prefix))


__all__ = [name for name in globals() if not name.startswith("__")]
