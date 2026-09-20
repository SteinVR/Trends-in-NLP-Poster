"""Helper functions backing the answering service core."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING, Any

from src.answering.page_attribution import candidate_pages_from_chunks, flatten_page_references, prune_candidate_pages
from src.evaluation.contracts import AnswerType, PageReference
from src.retrieval.service import RetrievalResult

from .service_constants import (
    _AMOUNT_CUE_PATTERN,
    _DATE_OF_ISSUE_CUE_PATTERN,
    _ISSUE_DATE_QUERY_PATTERN,
    _TEMPORAL_CUE_PATTERN,
    _TITLE_PAGE_QUERY_PATTERN,
)
from .service_helpers_extractors import (
    _answers_equivalent,
    _extract_boolean,
    _extract_number,
    _extract_number_candidates,
    _group_page_references,
    _provider_evidence_page,
    _validate_free_text_grounding,
)
from .service_helpers_text import (
    _clean_verbose_name_answer,
    _context_window,
    _keyword_tokens,
    _normalize_whitespace,
    _sentences,
)
from .service_types import (
    DEFAULT_CONFIDENCE_THRESHOLD_BY_TYPE,
    DEFAULT_REASONING_EFFORT_BY_TYPE,
    AnswerResult,
    DeterministicExtractor,
    FreeTextGroundingValidationError,
    _PageCandidate,
)

if TYPE_CHECKING:
    from src.answering.page_attribution import PageAttributionConfig, PageAttributionTrace

del TYPE_CHECKING

__exported_symbols = (_clean_verbose_name_answer, _provider_evidence_page)


def _coerce_page(page: Mapping[str, Any]) -> _PageCandidate:
    """Execute `_coerce_page`."""
    page_number = int(page["page_number"])
    return _PageCandidate(
        doc_id=str(page["doc_id"]),
        page_numbers=(page_number,),
        text=str(page["text"]),
        include_page_numbers=False,
    )


def _coerce_retrieval_payload(
    retrieval_output: RetrievalResult | Iterable[Mapping[str, Any]],
) -> tuple[list[_PageCandidate], list[PageReference] | None]:
    """Execute `_coerce_retrieval_payload`."""
    if isinstance(retrieval_output, RetrievalResult):
        pages: list[_PageCandidate] = []
        for chunk in retrieval_output.evidence_chunks:
            page_numbers = tuple(sorted({int(page) for page in chunk.page_span if int(page) > 0}))
            if not page_numbers:
                continue
            pages.append(
                _PageCandidate(
                    doc_id=chunk.doc_id,
                    page_numbers=page_numbers,
                    text=chunk.text,
                    include_page_numbers=True,
                )
            )
        return pages, list(retrieval_output.retrieved_chunk_pages)

    if isinstance(retrieval_output, Mapping) or isinstance(retrieval_output, str | bytes):
        raise TypeError("Retriever must return a RetrievalResult or an iterable of page mappings.")

    return [_coerce_page(page) for page in retrieval_output], None


def _pass_a_pages_from_retrieval_payload(
    *,
    retrieval_output: RetrievalResult | Iterable[Mapping[str, Any]],
    pages: list[_PageCandidate],
    config: PageAttributionConfig,
) -> list[_PageCandidate]:
    """Execute `_pass_a_pages_from_retrieval_payload`."""
    if isinstance(retrieval_output, RetrievalResult):
        candidate_pages = candidate_pages_from_chunks(retrieval_output.evidence_chunks)
    else:
        candidate_pages = _candidate_pages_from_page_candidates(pages)
    pass_a_result = prune_candidate_pages(candidate_pages, config=config)
    selected_pairs = {(page.doc_id, page.page_number) for page in pass_a_result.selected_pages}
    filtered_pages = _filter_pages_by_pairs(pages, selected_pairs, require_full_span=True)
    if filtered_pages:
        return filtered_pages
    if not selected_pairs:
        return pages
    return _filter_pages_by_pairs(pages, selected_pairs)


def _effective_page_attribution_config(
    *,
    base_config: PageAttributionConfig,
    question_text: str,
) -> PageAttributionConfig:
    """Execute `_effective_page_attribution_config`."""
    if not base_config.suppress_title_pages:
        return base_config
    if not _question_requests_title_page(question_text):
        return base_config
    return replace(base_config, suppress_title_pages=False)


def _question_requests_title_page(question_text: str) -> bool:
    """Execute `_question_requests_title_page`."""
    return bool(_TITLE_PAGE_QUERY_PATTERN.search(question_text))


def _prefer_issue_date_relevant_pages(
    *,
    question_text: str,
    answer_type: AnswerType,
    pages: list[_PageCandidate],
    relevant_pages: list[int],
) -> list[int]:
    """Execute `_prefer_issue_date_relevant_pages`."""
    if answer_type != AnswerType.DATE:
        return relevant_pages
    if not _ISSUE_DATE_QUERY_PATTERN.search(question_text):
        return relevant_pages

    issue_date_pages: set[int] = set()
    for page in pages:
        if not _DATE_OF_ISSUE_CUE_PATTERN.search(page.text):
            continue
        issue_date_pages.update(page.page_numbers)

    if issue_date_pages:
        return sorted(issue_date_pages)
    return relevant_pages


def _candidate_pages_from_page_candidates(pages: Iterable[_PageCandidate]) -> list[Any]:
    """Execute `_candidate_pages_from_page_candidates`."""
    from src.answering.page_attribution import CandidatePage

    candidates_by_key: dict[tuple[str, int], Any] = {}
    ordered_candidates: list[CandidatePage] = []
    for page in pages:
        for page_number in page.page_numbers:
            key = (page.doc_id, page_number)
            if key in candidates_by_key:
                continue
            candidate = CandidatePage(
                doc_id=page.doc_id,
                page_number=page_number,
                text=page.text,
                chunk_ids=[f"{page.doc_id}:{page_number}"],
                best_retrieval_score=0.0,
                best_rerank_score=None,
            )
            candidates_by_key[key] = candidate
            ordered_candidates.append(candidate)
    return ordered_candidates


def _filter_pages_by_pairs(
    pages: Iterable[_PageCandidate],
    allowed_pairs: set[tuple[str, int]],
    *,
    require_full_span: bool = False,
) -> list[_PageCandidate]:
    """Execute `_filter_pages_by_pairs`."""
    if not allowed_pairs:
        return []
    filtered: list[_PageCandidate] = []
    for page in pages:
        page_pairs = {(page.doc_id, page_number) for page_number in page.page_numbers}
        if require_full_span and not page_pairs.issubset(allowed_pairs):
            continue
        overlap = tuple(page_number for page_number in page.page_numbers if (page.doc_id, page_number) in allowed_pairs)
        if not overlap:
            continue
        clipped_text = _clip_text_for_selected_span(
            text=page.text,
            source_span=page.page_numbers,
            selected_span=overlap,
        )
        filtered.append(
            _PageCandidate(
                doc_id=page.doc_id,
                page_numbers=overlap,
                text=clipped_text,
                include_page_numbers=page.include_page_numbers,
            )
        )
    return filtered


def _clip_text_for_selected_span(
    *,
    text: str,
    source_span: tuple[int, ...],
    selected_span: tuple[int, ...],
) -> str:
    """Execute `_clip_text_for_selected_span`."""
    normalized = _normalize_whitespace(text)
    if not normalized:
        return normalized
    if not source_span or source_span == selected_span:
        return normalized
    selected_runs = _contiguous_page_runs(selected_span)
    if len(selected_runs) == 1:
        return _clip_text_for_contiguous_span(
            text=normalized,
            source_span=source_span,
            selected_span=selected_runs[0],
            strict=False,
        )

    clipped_segments: list[str] = []
    seen_segments: set[str] = set()
    for selected_run in selected_runs:
        clipped_segment = _clip_text_for_contiguous_span(
            text=normalized,
            source_span=source_span,
            selected_span=selected_run,
            strict=True,
        )
        if not clipped_segment:
            continue
        segment_key = clipped_segment.casefold()
        if segment_key in seen_segments:
            continue
        seen_segments.add(segment_key)
        clipped_segments.append(clipped_segment)

    if not clipped_segments:
        return ""
    return _normalize_whitespace(" ".join(clipped_segments))


def _contiguous_page_runs(page_numbers: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Execute `_contiguous_page_runs`."""
    if not page_numbers:
        return []

    runs: list[tuple[int, ...]] = []
    current_run: list[int] = [int(page_numbers[0])]
    for page_number in page_numbers[1:]:
        normalized_page = int(page_number)
        if normalized_page == current_run[-1] + 1:
            current_run.append(normalized_page)
            continue
        runs.append(tuple(current_run))
        current_run = [normalized_page]
    runs.append(tuple(current_run))
    return runs


def _clip_text_for_contiguous_span(
    *,
    text: str,
    source_span: tuple[int, ...],
    selected_span: tuple[int, ...],
    strict: bool,
) -> str:
    """Execute `_clip_text_for_contiguous_span`."""
    selected_min = min(selected_span)
    selected_max = max(selected_span)
    prefix_pages = sum(1 for page_number in source_span if page_number < selected_min)
    suffix_pages = sum(1 for page_number in source_span if page_number > selected_max)
    if prefix_pages == 0 and suffix_pages == 0:
        return text

    sentences = _sentences(text)
    if len(sentences) <= 1:
        return "" if strict else text

    clipped = list(sentences)
    for _ in range(min(prefix_pages, max(len(clipped) - 1, 0))):
        clipped.pop(0)
    for _ in range(min(suffix_pages, max(len(clipped) - 1, 0))):
        clipped.pop()
    if not clipped:
        return "" if strict else text
    return _normalize_whitespace(" ".join(clipped))


def _empty_result(
    *,
    question_id: str,
    model_name: str | None = None,
    final_pages: list[PageReference] | None = None,
    page_trace: PageAttributionTrace | None = None,
) -> AnswerResult:
    """Execute `_empty_result`."""
    resolved_final_pages = _resolve_final_emitted_pages(
        final_pages=final_pages,
        page_trace=page_trace,
    )
    return AnswerResult(
        question_id=question_id,
        answer=None,
        confidence=0.0,
        evidence_pages=resolved_final_pages,
        page_trace=page_trace,
        model_name=model_name,
    )


def _resolve_final_emitted_pages(
    *,
    final_pages: list[PageReference] | None,
    page_trace: PageAttributionTrace | None,
) -> list[PageReference]:
    """Resolve final emitted pages."""
    if final_pages is not None:
        return _copy_page_references(final_pages)
    if page_trace is not None:
        return _copy_page_references(page_trace.final_emitted_pages)
    return []


def _copy_page_references(references: Iterable[PageReference]) -> list[PageReference]:
    """Execute `_copy_page_references`."""
    copied: list[PageReference] = []
    for reference in references:
        copied.append(
            PageReference(
                doc_id=reference.doc_id,
                page_numbers=[int(page_number) for page_number in reference.page_numbers],
            )
        )
    return copied


def _parse_iso_date_parts(value: str) -> tuple[int, int, int]:
    """Parse iso date parts."""
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    if match is None:
        raise ValueError("date must be ISO-8601 (YYYY-MM-DD).")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _resolve_provider_model_name(provider: Any) -> str | None:
    """Resolve provider model name."""
    model_name = getattr(provider, "model", None)
    if isinstance(model_name, str):
        normalized = model_name.strip()
        if normalized:
            return normalized
    return None


def _normalize_reasoning_effort_map(reasoning_effort_by_type: Mapping[str, str] | None) -> dict[str, str]:
    """Normalize reasoning effort map."""
    normalized = dict(DEFAULT_REASONING_EFFORT_BY_TYPE)
    if reasoning_effort_by_type is None:
        return normalized
    for answer_type, raw_effort in reasoning_effort_by_type.items():
        key = str(answer_type).strip()
        if key not in normalized:
            continue
        effort = str(raw_effort).strip().lower()
        if effort in {"low", "medium", "high"}:
            normalized[key] = effort
    return normalized


def _normalize_confidence_threshold_map(
    confidence_threshold_by_type: Mapping[str, float] | None,
) -> dict[str, float]:
    """Normalize confidence threshold map."""
    normalized = dict(DEFAULT_CONFIDENCE_THRESHOLD_BY_TYPE)
    if confidence_threshold_by_type is None:
        return normalized
    for answer_type, raw_threshold in confidence_threshold_by_type.items():
        key = str(answer_type).strip()
        if key not in normalized:
            continue
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            continue
        if 0.0 <= threshold <= 1.0 and math.isfinite(threshold):
            normalized[key] = threshold
    return normalized


def _coerce_confidence(raw_confidence: Any, *, default: float) -> float:
    """Execute `_coerce_confidence`."""
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(confidence):
        return default
    if confidence < 0.0 or confidence > 1.0:
        return default
    return confidence


def _normalize_relevant_page_list(raw_relevant_pages: Any) -> list[int]:
    """Normalize relevant page list."""
    if not isinstance(raw_relevant_pages, list):
        return []
    relevant_pages: list[int] = []
    for raw_page in raw_relevant_pages:
        try:
            page_number = int(raw_page)
        except (TypeError, ValueError):
            continue
        if page_number > 0:
            relevant_pages.append(page_number)
    return sorted(set(relevant_pages))


def _safe_relevant_page_pairs(
    *,
    pages: list[_PageCandidate],
    relevant_pages: list[int],
) -> set[tuple[str, int]]:
    """Execute `_safe_relevant_page_pairs`."""
    if not relevant_pages:
        return set()
    docs_by_page_number: dict[int, set[str]] = defaultdict(set)
    for page in pages:
        for page_number in page.page_numbers:
            docs_by_page_number[int(page_number)].add(page.doc_id)
    return {
        (next(iter(docs_by_page_number[page_number])), page_number)
        for page_number in set(relevant_pages)
        if len(docs_by_page_number.get(page_number, set())) == 1
    }


def _safe_relevant_evidence_pairs(
    *,
    pages: list[_PageCandidate],
    relevant_evidence_indices: list[int],
) -> set[tuple[str, int]]:
    """Map structured solver evidence indices to concrete (doc_id, page_number) pairs."""

    if not relevant_evidence_indices:
        return set()

    mapped_pairs: set[tuple[str, int]] = set()
    for evidence_index in relevant_evidence_indices:
        page_position = int(evidence_index) - 1
        if page_position < 0 or page_position >= len(pages):
            continue
        page = pages[page_position]
        for page_number in page.page_numbers:
            if int(page_number) > 0:
                mapped_pairs.add((page.doc_id, int(page_number)))
    return mapped_pairs


def _supported_page_pairs_for_free_text(
    *,
    answer: str,
    pages: list[_PageCandidate],
) -> set[tuple[str, int]]:
    """Execute `_supported_page_pairs_for_free_text`."""
    if not pages:
        return set()

    if _is_free_text_answer_supported(answer=answer, pages=pages):
        return flatten_page_references(_group_page_references(pages))

    for subset_size in range(1, len(pages)):
        for subset in combinations(pages, subset_size):
            if _is_free_text_answer_supported(answer=answer, pages=subset):
                return flatten_page_references(_group_page_references(subset))

    return set()


def _is_free_text_answer_supported(
    *,
    answer: str,
    pages: Iterable[_PageCandidate],
) -> bool:
    """Return whether free text answer supported is true."""
    try:
        _validate_free_text_grounding(answer=answer, pages=pages)
    except FreeTextGroundingValidationError:
        return False
    return True


def _passes_typed_guardrails(
    *,
    answer_type: AnswerType,
    question: str,
    proposed_answer: Any,
    pages: list[_PageCandidate],
) -> bool:
    """Execute `_passes_typed_guardrails`."""
    question_tokens = _keyword_tokens(question)
    if answer_type == AnswerType.BOOLEAN:
        return _matches_deterministic_evidence(
            extractor=_extract_boolean,
            question_tokens=question_tokens,
            pages=pages,
            proposed_answer=proposed_answer,
        )
    if answer_type == AnswerType.NUMBER:
        if _is_spurious_title_year_literal(proposed_answer=proposed_answer, pages=pages):
            return False
        return _matches_deterministic_evidence(
            extractor=_extract_number,
            question_tokens=question_tokens,
            pages=pages,
            proposed_answer=proposed_answer,
        )
    return True


def _matches_deterministic_evidence(
    *,
    extractor: DeterministicExtractor,
    question_tokens: set[str],
    pages: list[_PageCandidate],
    proposed_answer: Any,
) -> bool:
    """Execute `_matches_deterministic_evidence`."""
    seen_candidate = False
    for page in pages:
        solved = extractor(page=page, question_tokens=question_tokens)
        if solved is None:
            continue
        seen_candidate = True
        if _answers_equivalent(solved.answer, proposed_answer):
            return True
    return not seen_candidate


def _is_spurious_title_year_literal(*, proposed_answer: Any, pages: list[_PageCandidate]) -> bool:
    """Return whether spurious title year literal is true."""
    try:
        numeric_value = float(proposed_answer)
    except (TypeError, ValueError):
        return False

    if not numeric_value.is_integer():
        return False
    year = int(numeric_value)
    if not 1900 <= year <= 2100:
        return False

    for page in pages:
        for value, start, end in _extract_number_candidates(page.text):
            if int(value) != year:
                continue
            context = _context_window(page.text, start, end, radius=28).casefold()
            looks_like_case_literal = "case" in context or "case no" in context or "case number" in context
            has_amount_cue = bool(_AMOUNT_CUE_PATTERN.search(context))
            has_temporal_cue = bool(_TEMPORAL_CUE_PATTERN.search(context))
            if looks_like_case_literal and not has_amount_cue and not has_temporal_cue:
                return True
    return False


__all__ = [name for name in globals() if not name.startswith("__")]
