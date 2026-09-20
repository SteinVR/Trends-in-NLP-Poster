"""Text and heading utilities used by index chunk builders."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from src.common.schemas import CanonicalPageRecord
from src.indexing.text_features import ordered_unique, tokenize_text

_PARAGRAPH_CLAUSE_RE = re.compile(
    r"^(?P<label>(?:\([A-Za-z0-9]+\)|\d+[A-Za-z]?(?:[.)])?|[A-Za-z](?:[.)])))\s+(?P<body>.+)$",
    re.DOTALL,
)
_BODY_START_CUES = {
    "a",
    "an",
    "the",
    "this",
    "these",
    "those",
    "any",
    "each",
    "every",
    "no",
    "such",
    "additional",
    "further",
    "supplemental",
    "all",
    "if",
    "when",
    "where",
    "unless",
}
_BODY_VERBS = {"is", "are", "must", "shall", "may", "can", "will", "remains", "means", "includes"}


@dataclass(slots=True)
class SplitHeadingParagraph:
    """Heading label and optional body remainder extracted from one paragraph."""

    kind: str
    label: str
    body_text: str | None = None


def extract_part_heading(text: str) -> str | None:
    """Extract first part heading from a page-level text payload."""

    return _extract_heading_by_kind(text, "part")


def extract_section_heading(text: str) -> str | None:
    """Extract first section heading from a page-level text payload."""

    return _extract_heading_by_kind(text, "section")


def extract_article_heading(text: str) -> str | None:
    """Extract first article heading from a page-level text payload."""

    return _extract_heading_by_kind(text, "article")


def clause_sources_for_text(text: str) -> list[tuple[str, str]]:
    """Split text into labeled clause sources used for clause and micro chunks."""

    paragraphs = content_paragraphs(text)
    lead_in_parts: list[str] = []
    numbered_clauses: list[tuple[str, str]] = []
    current_label: str | None = None
    current_parts: list[str] = []

    for paragraph in paragraphs:
        match = _PARAGRAPH_CLAUSE_RE.match(paragraph)
        if match:
            if current_label is None and lead_in_parts:
                numbered_clauses.append(("lead-in", "\n\n".join(lead_in_parts).strip()))
                lead_in_parts = []
            if current_label is not None and current_parts:
                numbered_clauses.append((current_label, "\n\n".join(current_parts).strip()))
            current_label = match.group("label").strip()
            current_parts = [paragraph.strip()]
            continue

        if current_label is not None:
            current_parts.append(paragraph.strip())
        else:
            lead_in_parts.append(paragraph.strip())

    if current_label is not None and current_parts:
        numbered_clauses.append((current_label, "\n\n".join(current_parts).strip()))

    if numbered_clauses:
        return numbered_clauses

    return [
        (str(index), paragraph)
        for index, paragraph in enumerate(paragraphs, start=1)
        if paragraph.strip()
    ]


def content_paragraphs(text: str) -> list[str]:
    """Return paragraph payload with heading-only lead-ins stripped."""

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    content: list[str] = []
    for paragraph in paragraphs:
        split_heading = split_heading_paragraph(paragraph)
        if split_heading is None:
            content.append(paragraph)
            continue
        if split_heading.body_text:
            content.append(split_heading.body_text)
    return content or paragraphs


def split_microchunks(
    text: str,
    *,
    token_chunk_size: int = 300,
    token_chunk_overlap: int = 50,
    token_counter: Callable[[str], int] | None = None,
) -> list[str]:
    """Split clause text into overlapping token-budgeted sentence chunks."""

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if not sentences:
        return [text.strip()]

    counter = token_counter or _count_tokens
    chunk_size = max(int(token_chunk_size), 1)
    overlap_target = max(int(token_chunk_overlap), 0)

    microchunks: list[str] = []
    start_index = 0
    while start_index < len(sentences):
        window: list[str] = []
        window_tokens = 0
        index = start_index
        while index < len(sentences):
            sentence = sentences[index]
            sentence_tokens = max(counter(sentence), 1)
            if window and (window_tokens + sentence_tokens) > chunk_size:
                break
            window.append(sentence)
            window_tokens += sentence_tokens
            index += 1
            if window_tokens >= chunk_size:
                break

        if not window:
            window = [sentences[start_index]]
            index = start_index + 1

        chunk = " ".join(window).strip()
        if chunk:
            microchunks.append(chunk)
        if index >= len(sentences):
            break

        overlap_tokens = 0
        overlap_count = 0
        for sentence in reversed(window):
            overlap_tokens += max(counter(sentence), 1)
            overlap_count += 1
            if overlap_tokens >= overlap_target:
                break

        next_start = index - overlap_count
        if next_start <= start_index:
            next_start = start_index + 1
        start_index = next_start

    return microchunks or [text.strip()]


def section_body_for_page(page: CanonicalPageRecord) -> str:
    """Generate section-body text representation for a single page."""

    return "\n\n".join(content_paragraphs(page.text)).strip()


def heading_path_for_block(page: CanonicalPageRecord, block: Any) -> list[str]:
    """Resolve heading path for a block, preferring block metadata values."""

    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, dict):
        heading_path = metadata.get("heading_path")
        if isinstance(heading_path, list):
            return ordered_unique(str(item).strip() for item in heading_path if str(item).strip())
    return list(page.heading_path)


def source_block_ids_for_block(page: CanonicalPageRecord, block: Any) -> list[str]:
    """Resolve block lineage IDs for a block, including inherited page IDs."""

    metadata = getattr(block, "metadata", None)
    candidate_ids: list[str] = []
    if isinstance(metadata, dict):
        source_block_ids = metadata.get("source_block_ids")
        if isinstance(source_block_ids, list):
            candidate_ids.extend(str(item).strip() for item in source_block_ids if str(item).strip())
        source_block_id = metadata.get("source_block_id")
        if source_block_id:
            candidate_ids.append(str(source_block_id).strip())
    block_id = getattr(block, "block_id", "")
    if block_id:
        candidate_ids.append(str(block_id).strip())
    candidate_ids.extend(page.source_block_ids)
    return ordered_unique(candidate_ids)


def _count_tokens(text: str) -> int:
    """Execute `_count_tokens`.

    Args:
        text: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    normalized = text.strip()
    if not normalized:
        return 0
    return len(tokenize_text(normalized))


def _normalize_heading_value(value: str) -> str:
    """Normalize heading value.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    normalized = " ".join(value.split()).strip()
    tokens = normalized.split(" ", 1)
    if not tokens:
        return normalized
    if len(tokens) == 1:
        return tokens[0].title()
    return f"{tokens[0].title()} {tokens[1]}"


def _extract_heading_by_kind(text: str, kind: str) -> str | None:
    """Extract heading by kind.

    Args:
        text: Input parameter.
        kind: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    for paragraph in [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]:
        split_heading = split_heading_paragraph(paragraph)
        if split_heading is None or split_heading.kind != kind:
            continue
        return split_heading.label
    return None


def split_heading_paragraph(paragraph: str) -> SplitHeadingParagraph | None:
    """Parse heading paragraph into heading label and optional body tail."""

    normalized = " ".join(paragraph.split()).strip()
    for kind, keyword in (("part", "part"), ("article", "article"), ("section", "section")):
        split_heading = _split_heading_keyword_paragraph(normalized, kind=kind, keyword=keyword)
        if split_heading is not None:
            return split_heading
    return None


def _split_heading_keyword_paragraph(
    paragraph: str,
    *,
    kind: str,
    keyword: str,
) -> SplitHeadingParagraph | None:
    """Execute `_split_heading_keyword_paragraph`.

    Args:
        paragraph: Input parameter.
        kind: Input parameter.
        keyword: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    match = re.match(rf"^(?P<prefix>{keyword}\s+\d+[A-Za-z0-9.-]*)\s+(?P<rest>.+)$", paragraph, flags=re.IGNORECASE)
    if not match:
        return None

    prefix = _normalize_heading_value(match.group("prefix"))
    rest_tokens = match.group("rest").split()
    if not rest_tokens:
        return SplitHeadingParagraph(kind=kind, label=prefix)

    heading_tokens: list[str] = []
    body_tokens: list[str] = []
    for index, _token in enumerate(rest_tokens):
        if index > 0 and _looks_like_body_start(rest_tokens[index:]):
            body_tokens = rest_tokens[index:]
            break
        heading_tokens.append(rest_tokens[index])

    label = prefix if not heading_tokens else _normalize_heading_value(f"{prefix} {' '.join(heading_tokens)}")
    body_text = " ".join(body_tokens).strip() or None
    return SplitHeadingParagraph(kind=kind, label=label, body_text=body_text)


def _looks_like_body_start(tokens: list[str]) -> bool:
    """Execute `_looks_like_body_start`.

    Args:
        tokens: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if not tokens:
        return False
    first = tokens[0].strip("()[]{}.,;:")
    if not first:
        return False

    lower_first = first.lower()
    if lower_first in _BODY_START_CUES:
        return True

    if len(tokens) >= 2:
        second = tokens[1].strip("()[]{}.,;:").lower()
        if second in _BODY_VERBS and first[:1].isupper() and not first.isupper():
            return True
    return False
