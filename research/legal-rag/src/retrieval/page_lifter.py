"""Chunk-to-page lifting utilities for retrieval grounding."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.evaluation.contracts import PageReference


def _source_value(source: Any, field_name: str) -> Any:
    """Execute `_source_value`.

    Args:
        source: Input parameter.
        field_name: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if isinstance(source, Mapping):
        return source.get(field_name)
    return getattr(source, field_name, None)


def extract_physical_pages(source: Any) -> list[int]:
    """Return sorted unique physical pages, or an empty list when the span is invalid."""

    raw_page_span = _source_value(source, "page_span")
    if not raw_page_span:
        return []

    pages: set[int] = set()
    for page in raw_page_span:
        try:
            normalized_page = int(page)
        except (TypeError, ValueError):
            return []
        if normalized_page <= 0:
            return []
        pages.add(normalized_page)

    return sorted(pages)


@dataclass(slots=True)
class LiftedPage:
    """Aggregated page-level support derived from retrieved chunks."""

    doc_id: str
    page_number: int
    chunk_ids: list[str] = field(default_factory=list)
    best_retrieval_score: float = 0.0
    best_rerank_score: float | None = None


@dataclass(slots=True)
class PageLifter:
    """Aggregate retrieved chunks into page-level grounding references."""

    def lift(self, chunks: list[Any]) -> list[LiftedPage]:
        """Execute `lift`.

        Args:
            chunks: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        pages_by_key: OrderedDict[tuple[str, int], LiftedPage] = OrderedDict()

        for chunk in chunks:
            doc_id = str(_source_value(chunk, "doc_id") or "").strip()
            chunk_id = str(_source_value(chunk, "chunk_id") or "").strip()
            if not doc_id or not chunk_id:
                continue

            retrieval_score = _coerce_score(_source_value(chunk, "retrieval_score"))
            rerank_score = _optional_score(_source_value(chunk, "rerank_score"))
            page_numbers = extract_physical_pages(chunk)
            if not page_numbers:
                continue

            for page_number in page_numbers:
                key = (doc_id, page_number)
                lifted = pages_by_key.setdefault(
                    key,
                    LiftedPage(doc_id=doc_id, page_number=page_number),
                )
                if chunk_id not in lifted.chunk_ids:
                    lifted.chunk_ids.append(chunk_id)
                lifted.best_retrieval_score = max(lifted.best_retrieval_score, retrieval_score)
                if rerank_score is not None:
                    current_best_rerank = lifted.best_rerank_score
                    if current_best_rerank is None:
                        lifted.best_rerank_score = rerank_score
                    else:
                        lifted.best_rerank_score = max(current_best_rerank, rerank_score)

        return sorted(
            pages_by_key.values(),
            key=lambda page: (
                -(page.best_rerank_score if page.best_rerank_score is not None else float("-inf")),
                -page.best_retrieval_score,
                page.doc_id,
                page.page_number,
            ),
        )

    def to_page_references(self, chunks: list[Any]) -> list[PageReference]:
        """Collapse retrieved chunks into deduplicated per-document page references."""

        pages_by_doc: OrderedDict[str, set[int]] = OrderedDict()
        for chunk in chunks:
            doc_id = str(_source_value(chunk, "doc_id") or "").strip()
            page_numbers = extract_physical_pages(chunk)
            if not doc_id or not page_numbers:
                continue

            pages_by_doc.setdefault(doc_id, set()).update(page_numbers)

        return [
            PageReference(doc_id=doc_id, page_numbers=sorted(page_numbers))
            for doc_id, page_numbers in pages_by_doc.items()
        ]


def _coerce_score(value: Any) -> float:
    """Execute `_coerce_score`.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_score(value: Any) -> float | None:
    """Execute `_optional_score`.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if value is None:
        return None
    return _coerce_score(value)
