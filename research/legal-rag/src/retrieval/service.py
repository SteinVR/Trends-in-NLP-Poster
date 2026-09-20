"""Retrieval service contract for hybrid search, reranking, and evidence selection."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from src.common.runtime_logging import get_logger, log_event, questions_debug_enabled
from src.common.schemas import QuestionRecord
from src.evaluation.contracts import PageReference
from src.retrieval.evidence_compressor import SupportPageEvidenceCompressor
from src.retrieval.introspection import callable_accepts_keyword, metadata_field
from src.retrieval.page_lifter import PageLifter, extract_physical_pages

LOGGER = get_logger("retrieval.service")
_TITLE_PAGE_QUERY_PATTERN = re.compile(r"\b(?:title|cover)\s+page\b", re.IGNORECASE)
_LAW_NUMBER_QUERY_PATTERN = re.compile(r"\b(?:official\s+)?(?:difc\s+)?law\s+(?:no\.?|number)\b", re.IGNORECASE)
_LAW_IDENTIFIER_PATTERN = re.compile(r"\blaw\s+(?:no\.?|numbered)\b", re.IGNORECASE)
_ISSUE_DATE_QUERY_PATTERN = re.compile(r"\b(?:date\s+of\s+issue|issue\s+date)\b", re.IGNORECASE)
_MULTI_DOCUMENT_COVERAGE_PATTERN = re.compile(
    r"\b(?:across|between|both|all\s+documents?|every\s+document|full\s+case\s+files?|each\s+document)\b",
    re.IGNORECASE,
)
_ALL_DOCUMENTS_QUERY_PATTERN = re.compile(
    r"\b(?:all\s+documents?|every\s+document|full\s+case\s+files?|each\s+document)\b",
    re.IGNORECASE,
)
_STATUTE_MULTI_LAW_PATTERN = re.compile(
    r"\b(?:which\s+specific|which)\s+.*\blaws?\b.*\bamend(?:ed|s|ing)\b",
    re.IGNORECASE,
)
_CASE_ID_TOKEN_PATTERN = re.compile(r"\b\d{1,4}\s*/\s*\d{4}\b", re.IGNORECASE)
_DEFAULT_PAGE_MAP_CANDIDATE_PATHS = (
    Path("data/warmup/corpus/page_map.json"),
    Path("data/final/corpus/page_map.json"),
)


class HybridSearch(Protocol):
    """Search interface used by the retrieval service."""

    def search(
        self,
        question: QuestionRecord,
        top_k: int,
        *,
        query_metadata: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top candidate chunks for a question."""


class Reranker(Protocol):
    """Rerank interface used by the retrieval service."""

    def rerank(
        self,
        question: QuestionRecord,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return reranked chunks for a question."""


class EvidenceCompressor(Protocol):
    """Evidence compression interface used by the retrieval service."""

    def compress(
        self,
        question: QuestionRecord,
        reranked_chunks: list[dict[str, Any]],
        max_evidence: int,
    ) -> list[dict[str, Any]]:
        """Select the final evidence subset for a question."""


@dataclass(slots=True)
class RetrievedChunk:
    """Typed evidence chunk returned by the retrieval service."""

    chunk_id: str
    doc_id: str
    page_span: list[int]
    chunk_type: str
    text: str
    retrieval_score: float
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RetrievedChunk":
        """Build a retrieved chunk from a retrieval-stage payload."""

        page_span = extract_physical_pages(payload)
        if not page_span:
            raise ValueError("Retrieved chunks must reference at least one positive physical page.")

        chunk_id = str(payload["chunk_id"]).strip()
        doc_id = str(payload["doc_id"]).strip()
        if not chunk_id:
            raise ValueError("Retrieved chunks must include a non-empty chunk_id.")
        if not doc_id:
            raise ValueError("Retrieved chunks must include a non-empty doc_id.")

        rerank_score_raw = payload.get("rerank_score")

        return cls(
            chunk_id=chunk_id,
            doc_id=doc_id,
            page_span=page_span,
            chunk_type=str(payload["chunk_type"]),
            text=str(payload["text"]),
            retrieval_score=float(payload["retrieval_score"]),
            rerank_score=None if rerank_score_raw is None else float(rerank_score_raw),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class RetrievalResult:
    """Final retrieval output consumed by downstream answering."""

    question_id: str
    query: str
    answer_type: str
    candidate_count: int
    reranked_count: int
    evidence_chunks: list[RetrievedChunk] = field(default_factory=list)
    retrieved_chunk_pages: list[PageReference] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    is_unanswerable: bool = False


@dataclass(slots=True)
class RetrievalService:
    """Coordinates search, reranking, and final evidence selection."""

    hybrid_search: HybridSearch
    reranker: Reranker
    evidence_compressor: EvidenceCompressor = field(default_factory=SupportPageEvidenceCompressor)
    candidate_budget: int = 10
    rerank_budget: int = 5
    default_evidence_budget: int = 3
    evidence_budget_by_answer_type: dict[str, int] = field(default_factory=dict)
    question_class_budgets: dict[str, Any] = field(default_factory=dict)
    min_rerank_score: float = 0.0
    page_lifter: PageLifter = field(default_factory=PageLifter)
    parent_page_expansion_enabled: bool = False
    parent_page_expansion_limit: int = 0
    query_expansion_enabled: bool = False
    query_expansion_max_queries: int = 0
    query_expansion_case_only: bool = True
    page_map_paths: tuple[Path, ...] | None = None
    _page_count_map_cache: dict[str, int] | None = field(default=None, init=False, repr=False)

    def retrieve(
        self,
        question: QuestionRecord,
        *,
        query_metadata: Any | None = None,
    ) -> RetrievalResult:
        """Run the retrieval pipeline for a single question."""

        question_class = _derive_question_class(query_metadata)
        candidate_budget, rerank_budget, max_evidence, class_budget_applied = self._resolved_budgets(
            question=question,
            question_class=question_class,
        )
        search_questions = self._build_search_questions(
            question=question,
            query_metadata=query_metadata,
            question_class=question_class,
        )
        candidates = self._search_candidates(
            search_questions=search_questions,
            top_k=candidate_budget,
            query_metadata=query_metadata,
        )
        candidate_window = _copy_payload_window(candidates, limit=candidate_budget)
        candidate_count = len(candidate_window)
        if candidate_count == 0:
            return self._empty_result(question=question, candidate_count=0, reranked_count=0)

        reranked = self.reranker.rerank(
            question,
            _copy_payload_window(candidate_window, limit=candidate_budget),
            top_k=rerank_budget,
        )
        reranked_window = _copy_payload_window(reranked, limit=rerank_budget)
        reranked_count = len(reranked_window)
        if reranked_count == 0:
            return self._empty_result(
                question=question,
                candidate_count=candidate_count,
                reranked_count=0,
            )

        surviving_reranked = self._apply_rerank_threshold(reranked_window)
        if not surviving_reranked:
            return self._empty_result(
                question=question,
                candidate_count=candidate_count,
                reranked_count=reranked_count,
            )

        expanded_reranked, expanded_by_chunk_id = self._expand_parent_pages(surviving_reranked)

        if max_evidence == 0:
            return self._empty_result(
                question=question,
                candidate_count=candidate_count,
                reranked_count=reranked_count,
            )

        selected_payloads_raw = self.evidence_compressor.compress(
            question,
            [dict(item) for item in expanded_reranked],
            max_evidence=max_evidence,
        )
        selected_payloads = self._apply_expansion_to_selected_payloads(
            selected_payloads_raw[:max_evidence],
            expanded_by_chunk_id,
        )
        selected_payloads = _augment_multidoc_coverage_payloads(
            selected_payloads=selected_payloads,
            reranked_payloads=expanded_reranked,
            candidate_payloads=candidate_window,
            question=question,
            question_class=question_class,
            query_metadata=query_metadata,
            document_max_page=self._document_max_page,
        )
        selected_payloads = _filter_case_entity_aligned_payloads(
            selected_payloads=selected_payloads,
            reranked_payloads=expanded_reranked,
            candidate_payloads=candidate_window,
            question=question,
            question_class=question_class,
            query_metadata=query_metadata,
        )
        selected_payloads = _limit_single_law_statute_payloads_by_confidence_gap(
            selected_payloads=selected_payloads,
            question=question,
            question_class=question_class,
        )
        evidence_chunks = self._materialize_evidence(selected_payloads)
        if not evidence_chunks:
            return self._empty_result(
                question=question,
                candidate_count=candidate_count,
                reranked_count=reranked_count,
            )
        evidence_chunks = _expand_front_matter_evidence_for_case_queries(
            evidence_chunks=evidence_chunks,
            question=question,
            question_class=question_class,
            document_max_page=self._document_max_page,
        )

        retrieved_chunk_pages = self.page_lifter.to_page_references(evidence_chunks)
        if not retrieved_chunk_pages:
            return self._empty_result(
                question=question,
                candidate_count=candidate_count,
                reranked_count=reranked_count,
            )
        retrieved_chunk_pages = _expand_front_matter_pages_for_case_queries(
            references=retrieved_chunk_pages,
            question=question,
            question_class=question_class,
            document_max_page=self._document_max_page,
        )

        diagnostics: dict[str, Any] = {
            "chunk_family_participation": {
                "candidates": _chunk_family_participation(candidate_window),
                "evidence": _chunk_family_participation(selected_payloads),
            },
            "rerank_effect": {
                "applied": True,
                "candidate_count": candidate_count,
                "reranked_count": reranked_count,
            },
            "parent_page_expansion": {
                "enabled": bool(self.parent_page_expansion_enabled and self.parent_page_expansion_limit > 0),
                "applied": bool(expanded_by_chunk_id),
                "expanded_chunk_count": len(expanded_by_chunk_id),
            },
            "query_expansion": {
                "enabled": bool(self.query_expansion_enabled and self.query_expansion_max_queries > 0),
                "applied": len(search_questions) > 1,
                "query_count": len(search_questions),
            },
        }
        if query_metadata is not None:
            diagnostics["prefilter_usage"] = {
                "applied": _has_soft_prefilters(query_metadata),
                "question_class": question_class,
                "candidate_budget": candidate_budget,
                "rerank_budget": rerank_budget,
                "evidence_budget": max_evidence,
                "budget_source": "question_class" if class_budget_applied else "answer_type_or_default",
            }

        result = RetrievalResult(
            question_id=question.id,
            query=question.question,
            answer_type=question.answer_type,
            candidate_count=candidate_count,
            reranked_count=reranked_count,
            evidence_chunks=evidence_chunks,
            retrieved_chunk_pages=retrieved_chunk_pages,
            diagnostics=diagnostics,
            is_unanswerable=False,
        )
        if questions_debug_enabled():
            log_event(
                LOGGER,
                logging.DEBUG,
                "retrieval.complete",
                "Retrieval pipeline completed for question.",
                stage="retrieval",
                question_id=question.id,
                candidate_count=result.candidate_count,
                reranked_count=result.reranked_count,
                evidence_chunk_count=len(result.evidence_chunks),
                unanswerable=result.is_unanswerable,
            )
        return result

    def _apply_rerank_threshold(self, reranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop reranked candidates that do not clear the configured score threshold."""

        if self.min_rerank_score <= 0:
            return [dict(item) for item in reranked]

        surviving: list[dict[str, Any]] = []
        for item in reranked:
            rerank_score = item.get("rerank_score")
            if rerank_score is None:
                continue
            if float(rerank_score) >= self.min_rerank_score:
                surviving.append(dict(item))
        return surviving

    def _build_search_questions(
        self,
        *,
        question: QuestionRecord,
        query_metadata: Any | None,
        question_class: str,
    ) -> list[QuestionRecord]:
        """Build primary + expansion search questions for the hybrid search stage."""

        search_questions = [question]
        if not self.query_expansion_enabled:
            return search_questions
        expansion_limit = self._normalized_budget(self.query_expansion_max_queries)
        if expansion_limit == 0:
            return search_questions
        if self.query_expansion_case_only and question_class != "case":
            if not _allow_statute_law_number_expansion(question=question, question_class=question_class):
                return search_questions

        expansion_texts = _build_query_expansion_texts(
            question_text=question.question,
            query_metadata=query_metadata,
            limit=expansion_limit,
        )
        for expansion_text in expansion_texts:
            search_questions.append(
                question.model_copy(update={"question": expansion_text})
            )
        return search_questions

    def _search_candidates(
        self,
        *,
        search_questions: list[QuestionRecord],
        top_k: int,
        query_metadata: Any | None,
    ) -> list[dict[str, Any]]:
        """Run hybrid search for one or more query variants and merge unique candidates."""

        merged_by_key: dict[str, dict[str, Any]] = {}
        encounter_order: list[str] = []
        for query_index, search_question in enumerate(search_questions):
            metadata_for_search = query_metadata if query_index == 0 else None
            candidates = _hybrid_search_with_optional_query_metadata(
                self.hybrid_search,
                search_question,
                top_k=top_k,
                query_metadata=metadata_for_search,
            )
            for candidate in candidates:
                candidate_key = _candidate_identity(candidate)
                existing = merged_by_key.get(candidate_key)
                if existing is None:
                    merged_by_key[candidate_key] = dict(candidate)
                    encounter_order.append(candidate_key)
                    continue
                if _candidate_retrieval_score(candidate) > _candidate_retrieval_score(existing):
                    merged_by_key[candidate_key] = dict(candidate)

        merged_candidates = [merged_by_key[key] for key in encounter_order]
        merged_candidates.sort(
            key=lambda payload: (
                -_candidate_retrieval_score(payload),
                str(payload.get("chunk_id") or ""),
            )
        )
        return merged_candidates

    def _materialize_evidence(self, payloads: list[dict[str, Any]]) -> list[RetrievedChunk]:
        """Convert payloads into validated evidence chunks and fail on malformed entries."""

        evidence_chunks: list[RetrievedChunk] = []
        for index, payload in enumerate(payloads, start=1):
            try:
                evidence_chunks.append(RetrievedChunk.from_payload(payload))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed retrieval payload at position={index}; strict mode forbids silent drops."
                ) from exc
        return evidence_chunks

    def _expand_parent_pages(
        self,
        payloads: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        """Expand page spans to parent pages for the top reranked window when enabled."""

        expanded_payloads = [dict(item) for item in payloads]
        if not self.parent_page_expansion_enabled:
            return expanded_payloads, {}

        expansion_limit = self._normalized_budget(self.parent_page_expansion_limit)
        if expansion_limit == 0:
            return expanded_payloads, {}

        expanded_by_chunk_id: dict[str, dict[str, Any]] = {}
        for index, payload in enumerate(expanded_payloads[:expansion_limit]):
            parent_pages = _normalized_parent_pages(payload.get("parent_page_numbers"))
            if not parent_pages:
                continue
            current_pages = extract_physical_pages(payload)
            if current_pages == parent_pages:
                continue
            payload["page_span"] = list(parent_pages)
            metadata = dict(payload.get("metadata", {}))
            metadata["parent_page_numbers"] = list(parent_pages)
            payload["metadata"] = metadata
            expanded_payloads[index] = payload
            chunk_id = str(payload.get("chunk_id") or "").strip()
            if chunk_id:
                expanded_by_chunk_id[chunk_id] = dict(payload)

        return expanded_payloads, expanded_by_chunk_id

    def _apply_expansion_to_selected_payloads(
        self,
        selected_payloads: list[dict[str, Any]],
        expanded_by_chunk_id: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute `_apply_expansion_to_selected_payloads`.

        Args:
            selected_payloads: Input parameter.
            expanded_by_chunk_id: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        if not selected_payloads:
            return []
        if not expanded_by_chunk_id:
            return [dict(payload) for payload in selected_payloads]

        merged_payloads: list[dict[str, Any]] = []
        for selected in selected_payloads:
            payload = dict(selected)
            chunk_id = str(payload.get("chunk_id") or "").strip()
            expanded = expanded_by_chunk_id.get(chunk_id)
            if expanded is None:
                merged_payloads.append(payload)
                continue

            expanded_page_span = extract_physical_pages(expanded)
            if expanded_page_span:
                payload["page_span"] = expanded_page_span
            parent_pages = _normalized_parent_pages(expanded.get("parent_page_numbers"))
            if parent_pages:
                payload["parent_page_numbers"] = parent_pages
                metadata = dict(payload.get("metadata", {}))
                metadata["parent_page_numbers"] = parent_pages
                payload["metadata"] = metadata
            merged_payloads.append(payload)

        return merged_payloads

    def _empty_result(
        self,
        *,
        question: QuestionRecord,
        candidate_count: int,
        reranked_count: int,
    ) -> RetrievalResult:
        """Build an unanswerable retrieval result."""

        return RetrievalResult(
            question_id=question.id,
            query=question.question,
            answer_type=question.answer_type,
            candidate_count=candidate_count,
            reranked_count=reranked_count,
            evidence_chunks=[],
            retrieved_chunk_pages=[],
            diagnostics={},
            is_unanswerable=True,
        )

    def _normalized_budget(self, budget: int) -> int:
        """Convert caller-provided budgets into safe non-negative integers."""

        return max(int(budget), 0)

    def _resolved_budgets(self, *, question: QuestionRecord, question_class: str) -> tuple[int, int, int, bool]:
        """Execute `_resolved_budgets`.

        Args:
            question: Input parameter.
            question_class: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        question_class_budget = self.question_class_budgets.get(question_class)
        if question_class_budget is None:
            candidate_budget = self._normalized_budget(self.candidate_budget)
            rerank_budget = self._normalized_budget(self.rerank_budget)
            evidence_budget = self._normalized_budget(
                self.evidence_budget_by_answer_type.get(
                    question.answer_type,
                    self.default_evidence_budget,
                )
            )
            class_budget_applied = False
        else:
            candidate_budget = self._normalized_budget(
                metadata_field(question_class_budget, "candidate_budget", self.candidate_budget)
            )
            rerank_budget = self._normalized_budget(
                metadata_field(question_class_budget, "rerank_budget", self.rerank_budget)
            )
            evidence_budget = self._normalized_budget(
                metadata_field(
                    question_class_budget,
                    "evidence_budget",
                    self.evidence_budget_by_answer_type.get(question.answer_type, self.default_evidence_budget),
                )
            )
            class_budget_applied = True

        if _is_title_page_law_number_question(question):
            candidate_budget = max(candidate_budget, 12)
            rerank_budget = max(rerank_budget, 4)
            evidence_budget = max(evidence_budget, 2)

        if _question_requires_multidoc_coverage(question=question, question_class=question_class):
            if question_class == "cross_case":
                candidate_budget = max(candidate_budget, 30)
                rerank_budget = max(rerank_budget, 10)
                evidence_budget = max(evidence_budget, 6)
            elif question_class == "case":
                candidate_budget = max(candidate_budget, 18)
                rerank_budget = max(rerank_budget, 8)
                evidence_budget = max(evidence_budget, 5)
            elif question_class == "statute":
                candidate_budget = max(candidate_budget, 40)
                rerank_budget = max(rerank_budget, 12)
                evidence_budget = max(evidence_budget, 8)

        return candidate_budget, rerank_budget, evidence_budget, class_budget_applied

    def _document_max_page(self, doc_id: str) -> int | None:
        """Return known document page count from phase-scoped page-map metadata."""

        page_count_map = self._load_page_count_map()
        return page_count_map.get(doc_id)

    def _load_page_count_map(self) -> dict[str, int]:
        """Load doc_id -> page_count from configured page-map paths."""

        if self._page_count_map_cache is not None:
            return self._page_count_map_cache

        for path in self._resolved_page_map_paths():
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, list):
                continue

            mapping: dict[str, int] = {}
            for item in payload:
                if not isinstance(item, dict):
                    continue
                doc_id = str(item.get("doc_id") or "").strip()
                page_count = item.get("page_count")
                if not doc_id:
                    continue
                try:
                    normalized_count = int(page_count)
                except (TypeError, ValueError):
                    continue
                if normalized_count <= 0:
                    continue
                mapping[doc_id] = normalized_count
            self._page_count_map_cache = mapping
            return mapping

        self._page_count_map_cache = {}
        return self._page_count_map_cache

    def _resolved_page_map_paths(self) -> tuple[Path, ...]:
        """Resolve page-map paths without implicitly mixing warmup/final corpora."""

        if self.page_map_paths is not None:
            return tuple(Path(path) for path in self.page_map_paths)

        existing_default_paths = tuple(path for path in _DEFAULT_PAGE_MAP_CANDIDATE_PATHS if path.exists())
        if len(existing_default_paths) <= 1:
            return existing_default_paths

        log_event(
            LOGGER,
            logging.WARNING,
            "retrieval.page_map.ambiguous",
            "Multiple default page_map paths detected; skipping implicit selection to avoid corpus mixing.",
            stage="retrieval",
            candidate_paths=[str(path) for path in existing_default_paths],
        )
        return ()


def _is_title_page_law_number_question(question: QuestionRecord) -> bool:
    """Return whether question asks for an official law number from a title page."""

    if str(question.answer_type).strip().casefold() != "number":
        return False
    normalized_question = question.question.strip()
    if not _TITLE_PAGE_QUERY_PATTERN.search(normalized_question):
        return False
    return bool(_LAW_NUMBER_QUERY_PATTERN.search(normalized_question))


def _allow_statute_law_number_expansion(*, question: QuestionRecord, question_class: str) -> bool:
    """Allow non-case query expansion only for statute title/cover law-number prompts."""

    if question_class != "statute":
        return False
    return _is_title_page_law_number_question(question)


def _is_statute_multi_law_question(question: QuestionRecord) -> bool:
    """Return whether question explicitly asks across multiple amended laws."""

    return bool(_STATUTE_MULTI_LAW_PATTERN.search(question.question))


def _question_requires_multidoc_coverage(*, question: QuestionRecord, question_class: str) -> bool:
    """Return whether query benefits from coverage-first multi-document retrieval."""

    question_text = question.question
    if question_class in {"case", "cross_case"}:
        return bool(_MULTI_DOCUMENT_COVERAGE_PATTERN.search(question_text))
    if question_class == "statute":
        return _is_statute_multi_law_question(question)
    return False


def _augment_multidoc_coverage_payloads(
    *,
    selected_payloads: list[dict[str, Any]],
    reranked_payloads: list[dict[str, Any]],
    candidate_payloads: list[dict[str, Any]],
    question: QuestionRecord,
    question_class: str,
    query_metadata: Any | None,
    document_max_page: Callable[[str], int | None],
) -> list[dict[str, Any]]:
    """Inject front-matter coverage for additional documents in multi-doc questions."""

    if not selected_payloads:
        return []
    if not _question_requires_multidoc_coverage(question=question, question_class=question_class):
        return [dict(payload) for payload in selected_payloads]

    answer_type = str(question.answer_type).strip().casefold()
    if question_class == "cross_case":
        max_extra_docs = 4
    elif question_class == "case":
        max_extra_docs = 3
    elif question_class == "statute":
        max_extra_docs = 6
    else:
        max_extra_docs = 0

    augmented_payloads = [dict(payload) for payload in selected_payloads]
    covered_docs = {
        str(payload.get("doc_id") or "").strip()
        for payload in augmented_payloads
        if str(payload.get("doc_id") or "").strip()
    }
    force_probe_docs = 0
    if question_class == "cross_case" and _ALL_DOCUMENTS_QUERY_PATTERN.search(question.question):
        force_probe_docs = 1

    coverage_target_doc_count = _multidoc_coverage_target_doc_count(
        question=question,
        question_class=question_class,
        query_metadata=query_metadata,
    )
    if coverage_target_doc_count is not None:
        remaining_docs = max(coverage_target_doc_count - len(covered_docs), 0)
        max_extra_docs = min(max_extra_docs, remaining_docs)
        if force_probe_docs > 0:
            max_extra_docs = max(max_extra_docs, force_probe_docs)
        if max_extra_docs == 0:
            return augmented_payloads

    added_docs = 0
    for payload_pool, include_baseline_pages in (
        (reranked_payloads, True),
        (candidate_payloads, False),
    ):
        for payload in payload_pool:
            doc_id = str(payload.get("doc_id") or "").strip()
            if not doc_id or doc_id in covered_docs:
                continue
            coverage_pages = _coverage_pages_for_doc(
                payload=payload,
                doc_id=doc_id,
                question=question,
                question_class=question_class,
                answer_type=answer_type,
                include_baseline_pages=include_baseline_pages,
                document_max_page=document_max_page,
            )
            if not coverage_pages:
                continue

            enriched_payload = dict(payload)
            enriched_payload["page_span"] = coverage_pages
            metadata = dict(enriched_payload.get("metadata", {}))
            metadata["coverage_injected"] = True
            enriched_payload["metadata"] = metadata

            augmented_payloads.append(enriched_payload)
            covered_docs.add(doc_id)
            added_docs += 1
            if added_docs >= max_extra_docs:
                break
        if added_docs >= max_extra_docs:
            break

    return augmented_payloads


def _filter_case_entity_aligned_payloads(
    *,
    selected_payloads: list[dict[str, Any]],
    reranked_payloads: list[dict[str, Any]],
    candidate_payloads: list[dict[str, Any]],
    question: QuestionRecord,
    question_class: str,
    query_metadata: Any | None,
) -> list[dict[str, Any]]:
    """Drop cross-document noise when payload text disagrees with case entities.

    The filter is intentionally conservative: if entity alignment is missing or
    alignment would collapse expected cross-case coverage, it keeps the original
    selection to preserve recall.
    """

    if not selected_payloads:
        return []
    if question_class not in {"case", "cross_case"}:
        return [dict(payload) for payload in selected_payloads]

    case_tokens = _case_entity_tokens(query_metadata=query_metadata)
    if not case_tokens:
        return [dict(payload) for payload in selected_payloads]

    doc_token_hits = _case_token_hits_by_doc(
        payloads=[*candidate_payloads, *reranked_payloads, *selected_payloads],
        case_tokens=case_tokens,
    )
    aligned_docs = {doc_id for doc_id, hits in doc_token_hits.items() if hits}
    if not aligned_docs:
        return [dict(payload) for payload in selected_payloads]

    filtered_payloads = [
        dict(payload)
        for payload in selected_payloads
        if str(payload.get("doc_id") or "").strip() in aligned_docs
    ]
    if not filtered_payloads:
        return [dict(payload) for payload in selected_payloads]

    if question_class == "cross_case" and len(case_tokens) >= 2:
        covered_tokens = _covered_case_tokens(
            payloads=filtered_payloads,
            doc_token_hits=doc_token_hits,
        )
        if len(covered_tokens) < 2:
            return [dict(payload) for payload in selected_payloads]

    if not _ALL_DOCUMENTS_QUERY_PATTERN.search(question.question):
        filtered_payloads = _limit_case_docs_for_precision(
            payloads=filtered_payloads,
            question_class=question_class,
            case_tokens=case_tokens,
            doc_token_hits=doc_token_hits,
        )

    return filtered_payloads or [dict(payload) for payload in selected_payloads]


def _limit_case_docs_for_precision(
    *,
    payloads: list[dict[str, Any]],
    question_class: str,
    case_tokens: tuple[str, ...],
    doc_token_hits: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Constrain aligned documents for non-all-doc case prompts."""

    if not payloads:
        return []

    doc_scores = _best_payload_score_by_doc(payloads)
    ranked_docs = sorted(
        doc_scores,
        key=lambda doc_id: (doc_scores[doc_id][0], doc_scores[doc_id][1], doc_id),
        reverse=True,
    )
    if not ranked_docs:
        return [dict(payload) for payload in payloads]

    keep_docs: set[str]
    if question_class == "case":
        keep_docs = {ranked_docs[0]}
    elif question_class == "cross_case" and len(case_tokens) >= 2:
        keep_docs = set()
        for case_token in case_tokens:
            candidate_docs = [
                doc_id
                for doc_id in ranked_docs
                if case_token in doc_token_hits.get(doc_id, set())
            ]
            if candidate_docs:
                keep_docs.add(candidate_docs[0])
        if len(keep_docs) < 2:
            return [dict(payload) for payload in payloads]
    else:
        return [dict(payload) for payload in payloads]

    filtered_payloads = [
        dict(payload)
        for payload in payloads
        if str(payload.get("doc_id") or "").strip() in keep_docs
    ]
    return filtered_payloads or [dict(payload) for payload in payloads]


def _limit_single_law_statute_payloads_by_confidence_gap(
    *,
    selected_payloads: list[dict[str, Any]],
    question: QuestionRecord,
    question_class: str,
) -> list[dict[str, Any]]:
    """Consolidate statute docs only when top doc confidence is clearly dominant."""

    if not selected_payloads:
        return []
    if question_class != "statute":
        return [dict(payload) for payload in selected_payloads]
    if _is_statute_multi_law_question(question):
        return [dict(payload) for payload in selected_payloads]

    doc_scores = _best_payload_score_by_doc(selected_payloads)
    if len(doc_scores) < 2:
        return [dict(payload) for payload in selected_payloads]

    ranked_docs = sorted(
        doc_scores,
        key=lambda doc_id: (doc_scores[doc_id][0], doc_scores[doc_id][1], doc_id),
        reverse=True,
    )
    best_doc_id = ranked_docs[0]
    second_doc_id = ranked_docs[1]
    best_rerank, _best_retrieval = doc_scores[best_doc_id]
    second_rerank, _second_retrieval = doc_scores[second_doc_id]

    if best_rerank == float("-inf") or second_rerank == float("-inf"):
        return [dict(payload) for payload in selected_payloads]
    if best_rerank < 0.2:
        return [dict(payload) for payload in selected_payloads]
    if (best_rerank - second_rerank) < 0.25:
        return [dict(payload) for payload in selected_payloads]

    filtered_payloads = [
        dict(payload)
        for payload in selected_payloads
        if str(payload.get("doc_id") or "").strip() == best_doc_id
    ]
    return filtered_payloads or [dict(payload) for payload in selected_payloads]


def _best_payload_score_by_doc(payloads: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Return best (rerank, retrieval) score tuple per doc for payload ranking."""

    scores: dict[str, tuple[float, float]] = {}
    for payload in payloads:
        doc_id = str(payload.get("doc_id") or "").strip()
        if not doc_id:
            continue
        rerank_score_raw = payload.get("rerank_score")
        retrieval_score_raw = payload.get("retrieval_score")
        try:
            rerank_score = float(rerank_score_raw) if rerank_score_raw is not None else float("-inf")
        except (TypeError, ValueError):
            rerank_score = float("-inf")
        try:
            retrieval_score = float(retrieval_score_raw)
        except (TypeError, ValueError):
            retrieval_score = 0.0
        score = (rerank_score, retrieval_score)
        current = scores.get(doc_id)
        if current is None or score > current:
            scores[doc_id] = score
    return scores


def _covered_case_tokens(
    *,
    payloads: list[dict[str, Any]],
    doc_token_hits: dict[str, set[str]],
) -> set[str]:
    """Collect case tokens covered by a payload set."""

    covered_tokens: set[str] = set()
    for payload in payloads:
        doc_id = str(payload.get("doc_id") or "").strip()
        if not doc_id:
            continue
        covered_tokens.update(doc_token_hits.get(doc_id, set()))
    return covered_tokens


def _case_token_hits_by_doc(
    *,
    payloads: list[dict[str, Any]],
    case_tokens: tuple[str, ...],
) -> dict[str, set[str]]:
    """Map each doc_id to the set of matched case tokens from payload text."""

    hits_by_doc: dict[str, set[str]] = {}
    if not case_tokens:
        return hits_by_doc

    for payload in payloads:
        doc_id = str(payload.get("doc_id") or "").strip()
        if not doc_id:
            continue
        payload_hits = _payload_case_token_hits(payload=payload, case_tokens=case_tokens)
        if not payload_hits:
            continue
        hits_by_doc.setdefault(doc_id, set()).update(payload_hits)
    return hits_by_doc


def _payload_case_token_hits(
    *,
    payload: dict[str, Any],
    case_tokens: tuple[str, ...],
) -> set[str]:
    """Return case-id tokens present in payload text."""

    if not case_tokens:
        return set()
    payload_text = _compact_case_search_text(str(payload.get("text") or ""))
    if not payload_text:
        return set()
    return {case_token for case_token in case_tokens if case_token in payload_text}


def _case_entity_tokens(*, query_metadata: Any | None) -> tuple[str, ...]:
    """Extract normalized strong case-id tokens from query metadata."""

    legal_entities = metadata_field(query_metadata, "legal_entities", ())
    tokens: set[str] = set()
    for entity in legal_entities:
        if not bool(metadata_field(entity, "is_strong", True)):
            continue
        if str(metadata_field(entity, "entity_type", "")).strip() != "case_id":
            continue

        raw_value = _normalized_case_token(metadata_field(entity, "raw_value", ""))
        normalized_value = _normalized_case_token(metadata_field(entity, "normalized_value", ""))
        if raw_value:
            tokens.add(raw_value)
        if normalized_value:
            tokens.add(normalized_value)

    return tuple(sorted(tokens))


def _normalized_case_token(value: Any) -> str:
    """Normalize one case-id token for robust substring matching."""

    token_text = str(value or "").strip().casefold()
    if not token_text:
        return ""
    token_match = _CASE_ID_TOKEN_PATTERN.search(token_text)
    if token_match is not None:
        token_text = token_match.group(0)
    return _compact_case_search_text(token_text)


def _compact_case_search_text(value: str) -> str:
    """Lowercase and remove whitespace to stabilize case-id matching."""

    return "".join(str(value or "").casefold().split())


def _multidoc_coverage_target_doc_count(
    *,
    question: QuestionRecord,
    question_class: str,
    query_metadata: Any | None,
) -> int | None:
    """Estimate a recall-safe target for document coverage in multi-doc prompts."""

    if question_class == "cross_case":
        case_entity_count = _strong_legal_entity_count(query_metadata=query_metadata, entity_type="case_id")
        if case_entity_count >= 2:
            target = case_entity_count
            if _ALL_DOCUMENTS_QUERY_PATTERN.search(question.question):
                target += 1
            return min(max(target, 2), 5)
        return None

    if question_class == "case":
        case_entity_count = _strong_legal_entity_count(query_metadata=query_metadata, entity_type="case_id")
        if case_entity_count >= 1:
            target = 1
            if _ALL_DOCUMENTS_QUERY_PATTERN.search(question.question):
                target += 1
            return min(max(target, 1), 3)
        return None

    if question_class == "statute":
        statute_entity_count = _strong_legal_entity_count(query_metadata=query_metadata, entity_type="statute_id")
        if statute_entity_count >= 1:
            return min(max(statute_entity_count, 2), 6)
        return None

    return None


def _strong_legal_entity_count(*, query_metadata: Any | None, entity_type: str) -> int:
    """Count strong legal entities of one type from query metadata."""

    legal_entities = metadata_field(query_metadata, "legal_entities", ())
    count = 0
    for entity in legal_entities:
        if not bool(metadata_field(entity, "is_strong", True)):
            continue
        if str(metadata_field(entity, "entity_type", "")).strip() == entity_type:
            count += 1
    return count


def _coverage_pages_for_doc(
    *,
    payload: dict[str, Any],
    doc_id: str,
    question: QuestionRecord,
    question_class: str,
    answer_type: str,
    include_baseline_pages: bool,
    document_max_page: Callable[[str], int | None],
) -> list[int]:
    """Build recall-first page candidates for one additional coverage document."""

    baseline_pages = extract_physical_pages(payload) if include_baseline_pages else []
    page_candidates = {int(page) for page in baseline_pages if int(page) > 0}
    lowered_question = question.question.casefold()

    if question_class in {"case", "cross_case"}:
        if answer_type == "number" and "claim value" in lowered_question:
            page_candidates.add(3)
        else:
            page_candidates.add(1)
            if answer_type in {"boolean", "name", "names", "date", "free_text"}:
                page_candidates.add(2)
            if _ISSUE_DATE_QUERY_PATTERN.search(lowered_question):
                page_candidates.add(3)
    elif question_class == "statute" and _is_statute_multi_law_question(question):
        page_candidates.add(1)

    if not page_candidates:
        return []

    max_page = document_max_page(doc_id)
    normalized_pages = sorted(page for page in page_candidates if page > 0)
    if max_page is not None:
        normalized_pages = [page for page in normalized_pages if page <= max_page]
    return normalized_pages


def _copy_payload_window(payloads: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    """Return a defensive copy of payloads capped at the provided limit."""

    return [dict(item) for item in payloads[:limit]]


def _normalized_parent_pages(raw_value: Any) -> list[int]:
    """Execute `_normalized_parent_pages`.

    Args:
        raw_value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if not isinstance(raw_value, list):
        return []

    pages: set[int] = set()
    for page in raw_value:
        try:
            normalized = int(page)
        except (TypeError, ValueError):
            return []
        if normalized <= 0:
            return []
        pages.add(normalized)
    return sorted(pages)


def _chunk_family_participation(payloads: list[dict[str, Any]]) -> list[str]:
    """Execute `_chunk_family_participation`.

    Args:
        payloads: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    families: list[str] = []
    seen: set[str] = set()
    for payload in payloads:
        metadata = payload.get("metadata")
        family = None
        if isinstance(metadata, dict):
            family = metadata.get("chunk_family")
        if family is None:
            family = payload.get("chunk_type")
        normalized = str(family or "").strip()
        if not normalized or normalized in seen:
            continue
        families.append(normalized)
        seen.add(normalized)
    return families


def _has_soft_prefilters(query_metadata: Any | None) -> bool:
    """Return whether soft prefilters is present.

    Args:
        query_metadata: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    return bool(
        _metadata_sequence_length(metadata_field(query_metadata, "legal_entities", ()))
        or _metadata_sequence_length(metadata_field(query_metadata, "document_title_cues", ()))
    )


def _metadata_sequence_length(value: Any) -> int:
    """Execute `_metadata_sequence_length`.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    try:
        return len(value)
    except TypeError:
        return 0


def _derive_question_class(query_metadata: Any | None) -> str:
    """Execute `_derive_question_class`.

    Args:
        query_metadata: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if bool(metadata_field(query_metadata, "likely_multi_document", False)):
        return "cross_case"

    legal_entities = metadata_field(query_metadata, "legal_entities", ())
    case_entity_present = False
    statute_entity_present = False
    for entity in legal_entities:
        entity_type = str(metadata_field(entity, "entity_type", "")).strip()
        if not bool(metadata_field(entity, "is_strong", True)):
            continue
        if entity_type == "case_id":
            case_entity_present = True
        elif entity_type == "statute_id":
            statute_entity_present = True

    if case_entity_present:
        return "case"
    if statute_entity_present:
        return "statute"
    return "free_text"


def _build_query_expansion_texts(
    *,
    question_text: str,
    query_metadata: Any | None,
    limit: int,
) -> list[str]:
    """Build deterministic query-expansion strings from query metadata signals."""

    if limit <= 0:
        return []

    normalized_question = str(question_text or "").strip()
    normalized_question_key = normalized_question.casefold()

    expansion_candidates: list[str] = []
    legal_entities = metadata_field(query_metadata, "legal_entities", ())
    for entity in legal_entities:
        if not bool(metadata_field(entity, "is_strong", True)):
            continue
        entity_type = str(metadata_field(entity, "entity_type", "")).strip()
        raw_value = str(metadata_field(entity, "raw_value", "")).strip()
        normalized_value = str(metadata_field(entity, "normalized_value", "")).strip()

        if raw_value:
            expansion_candidates.append(raw_value)
            if entity_type == "case_id":
                expansion_candidates.append(f"case {raw_value}")
            elif entity_type == "statute_id":
                expansion_candidates.append(f"law {raw_value}")
        if normalized_value and normalized_value.casefold() != raw_value.casefold():
            expansion_candidates.append(normalized_value)

    for title_cue in metadata_field(query_metadata, "document_title_cues", ()):
        cue_text = str(title_cue or "").strip()
        if cue_text:
            expansion_candidates.append(cue_text)

    expansions: list[str] = []
    seen: set[str] = {normalized_question_key}
    for raw_expansion in expansion_candidates:
        expansion = str(raw_expansion or "").strip()
        if not expansion:
            continue
        expansion_key = expansion.casefold()
        if expansion_key in seen:
            continue
        seen.add(expansion_key)
        expansions.append(expansion)
        if len(expansions) >= limit:
            break
    return expansions


def _candidate_identity(payload: dict[str, Any]) -> str:
    """Build a stable identity key for merged candidate payloads."""

    chunk_id = str(payload.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id
    doc_id = str(payload.get("doc_id") or "").strip()
    page_span = ",".join(str(page) for page in extract_physical_pages(payload))
    chunk_type = str(payload.get("chunk_type") or "").strip()
    text = str(payload.get("text") or "").strip()
    return f"{doc_id}|{page_span}|{chunk_type}|{text[:120]}"


def _candidate_retrieval_score(payload: dict[str, Any]) -> float:
    """Return candidate retrieval score as float with safe fallback."""

    raw_score = payload.get("retrieval_score")
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return 0.0


def _hybrid_search_with_optional_query_metadata(
    hybrid_search: HybridSearch,
    question: QuestionRecord,
    *,
    top_k: int,
    query_metadata: Any | None,
) -> list[dict[str, Any]]:
    """Execute `_hybrid_search_with_optional_query_metadata`.

    Args:
        hybrid_search: Input parameter.
        question: Input parameter.
        top_k: Input parameter.
        query_metadata: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if callable_accepts_keyword(hybrid_search.search, "query_metadata"):
        return hybrid_search.search(
            question,
            top_k=top_k,
            query_metadata=query_metadata,
        )
    return hybrid_search.search(question, top_k=top_k)


def _expand_front_matter_pages_for_case_queries(
    *,
    references: list[PageReference],
    question: QuestionRecord,
    question_class: str,
    document_max_page: Callable[[str], int | None],
) -> list[PageReference]:
    """Add page two when case-like queries only retrieve a document front page."""

    expanded_references: list[PageReference] = []
    for reference in references:
        normalized_pages = _apply_query_page_hints(
            page_numbers=reference.page_numbers,
            doc_id=reference.doc_id,
            question=question,
            question_class=question_class,
            document_max_page=document_max_page,
        )
        expanded_references.append(
            PageReference(doc_id=reference.doc_id, page_numbers=normalized_pages)
        )
    return expanded_references


def _expand_front_matter_evidence_for_case_queries(
    *,
    evidence_chunks: list[RetrievedChunk],
    question: QuestionRecord,
    question_class: str,
    document_max_page: Callable[[str], int | None],
) -> list[RetrievedChunk]:
    """Propagate case front-page expansion into evidence chunk page spans."""

    expanded_chunks: list[RetrievedChunk] = []
    for chunk in evidence_chunks:
        expanded_page_span = _apply_query_page_hints(
            page_numbers=chunk.page_span,
            doc_id=chunk.doc_id,
            question=question,
            question_class=question_class,
            document_max_page=document_max_page,
        )
        if expanded_page_span != chunk.page_span:
            expanded_chunks.append(replace(chunk, page_span=expanded_page_span))
            continue
        expanded_chunks.append(chunk)
    return expanded_chunks


def _apply_query_page_hints(
    *,
    page_numbers: list[int],
    doc_id: str,
    question: QuestionRecord,
    question_class: str,
    document_max_page: Callable[[str], int | None],
) -> list[int]:
    """Apply narrow query-driven page hints for residual high-value miss patterns."""

    normalized_pages = sorted({int(page) for page in page_numbers if int(page) > 0})
    if not normalized_pages:
        return normalized_pages

    answer_type = str(question.answer_type).strip().casefold()
    question_text = question.question.casefold()
    max_page = document_max_page(doc_id)

    if question_class in {"case", "cross_case"} and answer_type in {"date", "name", "free_text"}:
        if normalized_pages == [1]:
            normalized_pages = [1, 2]

    if question_class in {"case", "cross_case"} and answer_type in {"date", "name"}:
        if _ISSUE_DATE_QUERY_PATTERN.search(question_text) and normalized_pages in ([1], [1, 2]):
            normalized_pages = [1, 2, 3]

    if question_class == "case" and answer_type == "date" and _ISSUE_DATE_QUERY_PATTERN.search(question_text):
        if 2 in normalized_pages:
            normalized_pages = [2]

    if question_class in {"case", "cross_case"} and answer_type == "number" and normalized_pages == [1]:
        normalized_pages = [1, 2, 3]

    if "cover page" in question_text or "title page" in question_text:
        normalized_pages.append(1)

    if question_class == "statute" and _LAW_IDENTIFIER_PATTERN.search(question_text):
        normalized_pages.append(1)

    if "last page" in question_text and max_page is not None:
        normalized_pages.append(max_page)

    if "conclusion section" in question_text and max_page is not None:
        normalized_pages.append(max_page)
        if max_page > 1:
            normalized_pages.append(max_page - 1)

    if (
        question_class == "statute"
        and answer_type == "free_text"
        and ("administer" in question_text or "responsible" in question_text)
    ):
        for candidate_page in (4, 5, 6):
            if max_page is None or candidate_page <= max_page:
                normalized_pages.append(candidate_page)

    # Precision-first narrowing for claim-value prompts: if page three is present,
    # keep it as the strongest anchor; otherwise keep the highest early-page cue.
    if question_class in {"case", "cross_case"} and answer_type == "number" and "claim value" in question_text:
        early_pages = [page for page in normalized_pages if page <= 3]
        if 3 in early_pages:
            normalized_pages = [3]
        elif early_pages:
            normalized_pages = [max(early_pages)]

    # Precision-first narrowing for statute administration prompts: discard tail
    # outliers when a focused 4-6 window is available.
    if (
        question_class == "statute"
        and answer_type == "free_text"
        and ("administer" in question_text or "responsible" in question_text)
    ):
        admin_window_pages = [page for page in normalized_pages if page in {4, 5, 6}]
        if admin_window_pages:
            normalized_pages = sorted(set(admin_window_pages))

    # For statute law-identifier prompts, keep front-range pages when page one is
    # present to suppress long-tail sections that frequently add precision noise.
    if (
        question_class == "statute"
        and answer_type in {"boolean", "number", "name"}
        and _LAW_IDENTIFIER_PATTERN.search(question_text)
        and 1 in normalized_pages
    ):
        front_range_pages = [page for page in normalized_pages if page <= 3]
        if front_range_pages:
            normalized_pages = sorted(set(front_range_pages))

    # Precision-first narrowing for conclusion-section prompts: keep only the
    # document tail pages if present.
    if "conclusion section" in question_text and max_page is not None:
        tail_pages = [page for page in normalized_pages if page in {max_page, max_page - 1}]
        if tail_pages:
            normalized_pages = sorted(set(page for page in tail_pages if page > 0))

    normalized = sorted({page for page in normalized_pages if page > 0})
    if max_page is not None:
        normalized = [page for page in normalized if page <= max_page]
    return normalized
