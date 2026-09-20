"""Evidence compression and support-page validation for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.common.schemas import QuestionRecord
from src.retrieval.page_lifter import PageLifter, extract_physical_pages


@dataclass(slots=True)
class SupportPageEvidenceCompressor:
    """Select a compact evidence set while preserving page-grounded support."""

    page_lifter: PageLifter = field(default_factory=PageLifter)
    max_chunks_per_page: int = 1

    def compress(
        self,
        question: QuestionRecord,
        reranked_chunks: list[dict[str, Any]],
        max_evidence: int,
    ) -> list[dict[str, Any]]:
        """Execute `compress`.

        Args:
            question: Input parameter.
            reranked_chunks: Input parameter.
            max_evidence: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        evidence_budget = max(int(max_evidence), 0)
        if evidence_budget == 0:
            return []

        valid_chunks: list[dict[str, Any]] = []
        valid_chunk_pages: list[set[int]] = []
        for chunk in self._sort_chunks(question=question, chunks=reranked_chunks):
            page_numbers = extract_physical_pages(chunk)
            if not page_numbers:
                continue
            valid_chunks.append(dict(chunk))
            valid_chunk_pages.append(set(page_numbers))

        if not valid_chunks:
            return []

        selected: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        lifted_pages = self.page_lifter.lift(valid_chunks)

        for lifted_page in lifted_pages:
            page_picks = 0
            for chunk, page_numbers in zip(valid_chunks, valid_chunk_pages):
                if chunk.get("doc_id") != lifted_page.doc_id:
                    continue
                if lifted_page.page_number not in page_numbers:
                    continue

                chunk_id = str(chunk.get("chunk_id") or "")
                if not chunk_id or chunk_id in seen_chunk_ids:
                    continue

                selected.append(dict(chunk))
                seen_chunk_ids.add(chunk_id)
                page_picks += 1
                if len(selected) >= evidence_budget or page_picks >= self.max_chunks_per_page:
                    break

            if len(selected) >= evidence_budget:
                return selected[:evidence_budget]

        for chunk in valid_chunks:
            chunk_id = str(chunk.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            selected.append(dict(chunk))
            seen_chunk_ids.add(chunk_id)
            if len(selected) >= evidence_budget:
                break

        return selected[:evidence_budget]

    def _sort_chunks(self, *, question: QuestionRecord, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute `_sort_chunks`.

        Args:
            question: Input parameter.
            chunks: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        page_priority_enabled = _is_issue_date_question(question)
        return sorted(
            (dict(chunk) for chunk in chunks),
            key=lambda chunk: (
                -_optional_score(chunk.get("rerank_score")),
                -(_chunk_type_priority(chunk) if page_priority_enabled else 0),
                -_score(chunk.get("retrieval_score")),
                str(chunk.get("chunk_id") or ""),
            ),
        )


def _is_issue_date_question(question: QuestionRecord) -> bool:
    """Return whether the question is an issue-date lookup."""

    if str(question.answer_type).strip().casefold() != "date":
        return False
    question_text = question.question.casefold()
    return "date of issue" in question_text or "issue date" in question_text


def _chunk_type_priority(chunk: dict[str, Any]) -> int:
    """Return chunk-family priority for issue-date extraction."""

    metadata = chunk.get("metadata")
    if isinstance(metadata, dict):
        chunk_family = str(metadata.get("chunk_family") or "").strip().casefold()
    else:
        chunk_family = ""
    if not chunk_family:
        chunk_family = str(chunk.get("chunk_type") or "").strip().casefold()
    priority_by_family = {
        "page": 4,
        "section": 3,
        "clause": 2,
        "table": 1,
        "microchunk": 0,
    }
    return priority_by_family.get(chunk_family, 0)


def _score(value: Any) -> float:
    """Execute `_score`.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_score(value: Any) -> float:
    """Execute `_optional_score`.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if value is None:
        return float("-inf")
    return _score(value)
