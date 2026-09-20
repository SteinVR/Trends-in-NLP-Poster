"""Cross-case decomposition for comparative multi-document questions (W4E).

Rule-based trigger, decomposition, and deterministic recomposition.
Disabled by default — controlled by ``DecompositionSettings.enabled``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.answering.page_attribution import PageAttributionTrace, build_page_trace
from src.answering.service import AnsweringService, AnswerResult, QueryMetadata
from src.common.schemas import QuestionRecord
from src.evaluation.contracts import PageReference

# ---------------------------------------------------------------------------
# Case reference extraction (captures full DIFC-style refs like "CFI 057/2025")
# ---------------------------------------------------------------------------

_FULL_CASE_REF = re.compile(
    r"\b(?P<ref>(?:CFI|SCT|ARB|DEC|TCD|ENF|CA)\s*\d{1,4}\s*/\s*\d{2,4})\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Comparison subject detection
# ---------------------------------------------------------------------------

_PARTY_SIGNAL = re.compile(
    r"(?:main\s+)?part(?:y|ies)|individual|company|entity|entities|person",
    re.IGNORECASE,
)
_JUDGE_SIGNAL = re.compile(r"judge|presid", re.IGNORECASE)
_DATE_SIGNAL = re.compile(r"(?:issue|filing|hearing)\s*date|issued\s+(?:first|earlier)|earlier", re.IGNORECASE)
_CLAIM_SIGNAL = re.compile(r"monetary\s+claim|higher|amount|claim", re.IGNORECASE)

_DECOMPOSABLE_TYPES = frozenset({"boolean", "name"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SubQuestionResult:
    """Answer for a single decomposed sub-question."""

    entity_id: str
    question: QuestionRecord
    answer_result: AnswerResult


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    """Full decomposition trace for telemetry."""

    triggered: bool
    original_question: QuestionRecord
    sub_results: tuple[SubQuestionResult, ...]
    final_answer_result: AnswerResult


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------


def extract_case_refs(text: str) -> list[str]:
    """Return unique case references found in *text*, preserving order."""
    seen: set[str] = set()
    refs: list[str] = []
    for m in _FULL_CASE_REF.finditer(text):
        normalized = re.sub(r"\s+", " ", m.group("ref")).strip().upper()
        if normalized not in seen:
            seen.add(normalized)
            refs.append(normalized)
    return refs


def should_decompose(
    question: QuestionRecord,
    query_metadata: QueryMetadata,
    *,
    enabled: bool = False,
) -> bool:
    """Return True when decomposition should be attempted."""
    if not enabled:
        return False
    if question.answer_type not in _DECOMPOSABLE_TYPES:
        return False
    case_refs = extract_case_refs(question.question)
    return len(case_refs) >= 2


# ---------------------------------------------------------------------------
# Decomposition — rule-based sub-question generation
# ---------------------------------------------------------------------------


def _detect_subject(question_text: str) -> str:
    """Detect what the cross-case question is comparing."""
    if _JUDGE_SIGNAL.search(question_text):
        return "judge"
    if _PARTY_SIGNAL.search(question_text):
        return "party"
    if _DATE_SIGNAL.search(question_text):
        return "date"
    if _CLAIM_SIGNAL.search(question_text):
        return "claim"
    return "generic"


_SUB_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    # subject -> answer_type -> (template, sub_answer_type)
    "party": {
        "boolean": ("Who are the main parties in case {case_ref}?", "names"),
    },
    "judge": {
        "boolean": ("Who are the judges in case {case_ref}?", "names"),
    },
    "date": {
        "name": ("What is the date of issue of the document in case {case_ref}?", "date"),
    },
    "claim": {
        "name": ("What is the monetary claim amount in case {case_ref}?", "number"),
    },
}


def decompose_question(
    question: QuestionRecord,
    query_metadata: QueryMetadata,
    *,
    max_subquestions: int = 4,
) -> list[tuple[str, QuestionRecord]]:
    """Split a cross-case question into per-case sub-questions.

    Returns list of ``(case_ref, sub_question)`` pairs.
    """
    case_refs = extract_case_refs(question.question)[:max_subquestions]
    if len(case_refs) < 2:
        return []

    subject = _detect_subject(question.question)
    answer_type = question.answer_type

    templates = _SUB_TEMPLATES.get(subject, {})
    entry = templates.get(answer_type)

    if entry is not None:
        template, sub_type = entry
    else:
        # No valid recomposer for this subject+answer_type — skip decomposition
        # so the caller falls back to baseline answering.  The generic fallback
        # ("Focus only on case ...") keeps the original answer_type which the
        # recomposer cannot handle correctly (e.g. boolean True/False converted
        # to empty entity sets → always False).
        return []

    result: list[tuple[str, QuestionRecord]] = []
    for ref in case_refs:
        sub_q = QuestionRecord(
            id=f"{question.id}__sub_{ref.replace(' ', '_').replace('/', '-')}",
            question=template.format(case_ref=ref),
            answer_type=sub_type,
        )
        result.append((ref, sub_q))
    return result


# ---------------------------------------------------------------------------
# Recomposition — deterministic
# ---------------------------------------------------------------------------


def _merge_evidence_pages(sub_results: list[SubQuestionResult]) -> list[PageReference]:
    """Deduplicated union of evidence pages from all sub-results."""
    seen: set[tuple[str, tuple[int, ...]]] = set()
    merged: list[PageReference] = []
    for sr in sub_results:
        references = list(sr.answer_result.evidence_pages)
        if not references and sr.answer_result.page_trace is not None:
            references = list(sr.answer_result.page_trace.pass_a_pages)
        for ref in references:
            key = (ref.doc_id, tuple(sorted(ref.page_numbers)))
            if key not in seen:
                seen.add(key)
                merged.append(ref)
    return merged


def _build_composite_trace(
    sub_results: list[SubQuestionResult],
    final_pages: list[PageReference],
) -> PageAttributionTrace:
    """Merge page traces from sub-results into one composite trace."""
    raw: list[PageReference] = []
    pass_a: list[PageReference] = []
    solver: list[PageReference] = []

    for sr in sub_results:
        trace = sr.answer_result.page_trace
        if trace is None:
            continue
        raw.extend(trace.raw_retrieved_chunk_pages)
        pass_a.extend(trace.pass_a_pages)
        solver.extend(trace.solver_reported_pages)

    return build_page_trace(
        raw_retrieved_chunk_pages=raw,
        pass_a_pages=pass_a,
        solver_reported_pages=solver,
        final_emitted_pages=final_pages,
        validation_action="decomposition_recomposed",
        collapsed_answer=False,
    )


def _recompose_boolean(
    original_question: QuestionRecord,
    sub_results: list[SubQuestionResult],
) -> Any:
    """Recompose boolean cross-case answer via set intersection.

    Each sub-answer should be a list of names/entities for one case.
    If any entity appears in all sub-answers → True, else False.
    """
    if not sub_results:
        return None

    entity_sets: list[set[str]] = []
    for sr in sub_results:
        answer = sr.answer_result.answer
        if answer is None:
            entity_sets.append(set())
        elif isinstance(answer, list):
            entity_sets.append({str(e).strip().lower() for e in answer if e})
        elif isinstance(answer, str):
            entity_sets.append({name.strip().lower() for name in answer.split(",") if name.strip()})
        else:
            entity_sets.append(set())

    if not entity_sets or all(not s for s in entity_sets):
        return False

    intersection = entity_sets[0]
    for s in entity_sets[1:]:
        intersection = intersection & s

    return bool(intersection)


def _recompose_name_comparison(
    original_question: QuestionRecord,
    sub_results: list[SubQuestionResult],
) -> Any:
    """Recompose name comparison (e.g. 'which case has earlier date').

    Returns the case reference that wins the comparison.
    """
    if len(sub_results) < 2:
        return None

    question_lower = original_question.question.lower()
    prefer_lower = any(w in question_lower for w in ("earlier", "first", "lowest", "smaller"))
    # prefer_higher when: "higher", "later", "largest", "greater"

    best_ref: str | None = None
    best_val: str | None = None

    for sr in sub_results:
        val = sr.answer_result.answer
        if val is None:
            continue
        val_str = str(val).strip()
        if not val_str:
            continue

        if best_val is None:
            best_ref = sr.entity_id
            best_val = val_str
            continue

        # Compare: try numeric first, then lexicographic (works for ISO dates)
        try:
            current_num = float(val_str.replace(",", ""))
            best_num = float(best_val.replace(",", ""))
            if prefer_lower:
                wins = current_num < best_num
            else:
                wins = current_num > best_num
        except ValueError:
            if prefer_lower:
                wins = val_str < best_val
            else:
                wins = val_str > best_val

        if wins:
            best_ref = sr.entity_id
            best_val = val_str

    return best_ref


def recompose_answer(
    original_question: QuestionRecord,
    sub_results: list[SubQuestionResult],
) -> AnswerResult:
    """Deterministic recomposition of sub-answers into a final answer."""
    merged_pages = _merge_evidence_pages(sub_results)
    composite_trace = _build_composite_trace(sub_results, merged_pages)

    if original_question.answer_type == "boolean":
        final_answer = _recompose_boolean(original_question, sub_results)
        confidence = 0.8 if final_answer is not None else 0.0
    elif original_question.answer_type == "name":
        final_answer = _recompose_name_comparison(original_question, sub_results)
        confidence = 0.8 if final_answer is not None else 0.0
    else:
        # Unexpected type — should not reach here due to trigger guard
        final_answer = None
        confidence = 0.0

    return AnswerResult(
        question_id=original_question.id,
        answer=final_answer,
        confidence=confidence,
        evidence_pages=merged_pages,
        page_trace=composite_trace,
        model_name="decomposition_w4e",
    )


# ---------------------------------------------------------------------------
# Orchestration — sequential execution
# ---------------------------------------------------------------------------


def answer_with_decomposition(
    question: QuestionRecord,
    query_metadata: QueryMetadata,
    answering_service: AnsweringService,
    *,
    max_subquestions: int = 4,
) -> AnswerResult:
    """Execute decomposed answering for a cross-case question."""
    sub_pairs = decompose_question(
        question,
        query_metadata,
        max_subquestions=max_subquestions,
    )

    if not sub_pairs:
        # Decomposition failed to produce sub-questions; fall back to baseline.
        return answering_service.answer_question(question, allow_decomposition=False)

    sub_results: list[SubQuestionResult] = []
    for case_ref, sub_q in sub_pairs:
        answer_result = answering_service.answer_question(sub_q, allow_decomposition=False)
        sub_results.append(
            SubQuestionResult(
                entity_id=case_ref,
                question=sub_q,
                answer_result=answer_result,
            )
        )

    return recompose_answer(original_question=question, sub_results=sub_results)
