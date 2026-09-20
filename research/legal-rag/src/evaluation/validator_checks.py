"""Shared heavy validation checks extracted from SubmissionValidator."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

from .contracts import AnswerType, UnanswerableRetrievalPolicy, ValidationIssue


def validate_answer_value(
    self: Any,
    answer_value: Any,
    *,
    answer_type: AnswerType,
    answer_index: int,
    question_id: str | None,
    answer_issue: Callable[..., ValidationIssue],
    is_number: Callable[[Any], bool],
) -> list[ValidationIssue]:
    """Validate one answer payload value against its declared answer type."""

    _answer_issue = answer_issue
    _is_number = is_number
    issues: list[ValidationIssue] = []
    deterministic_types = {
        AnswerType.NUMBER,
        AnswerType.BOOLEAN,
        AnswerType.NAME,
        AnswerType.NAMES,
        AnswerType.DATE,
    }

    if answer_value is None:
        if answer_type == AnswerType.FREE_TEXT:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="answer.null_not_allowed",
                    message="free_text answers must be non-null strings.",
                    field="answer",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
        return issues

    if answer_type == AnswerType.NUMBER:
        if not _is_number(answer_value):
            issues.append(
                _answer_issue(
                    code="answer.type_mismatch",
                    message="number answers must be JSON numbers.",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
        return issues

    if answer_type == AnswerType.BOOLEAN:
        if not isinstance(answer_value, bool):
            issues.append(
                _answer_issue(
                    code="answer.type_mismatch",
                    message="boolean answers must be JSON true/false.",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
        return issues

    if answer_type == AnswerType.NAME:
        if not isinstance(answer_value, str) or not answer_value.strip():
            issues.append(
                _answer_issue(
                    code="answer.type_mismatch",
                    message="name answers must be non-empty strings.",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
        return issues

    if answer_type == AnswerType.NAMES:
        if not isinstance(answer_value, list) or not answer_value:
            issues.append(
                _answer_issue(
                    code="answer.type_mismatch",
                    message="names answers must be non-empty arrays of strings.",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
            return issues
        if not all(isinstance(item, str) and item.strip() for item in answer_value):
            issues.append(
                _answer_issue(
                    code="answer.type_mismatch",
                    message="names answers must contain only non-empty strings.",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
        return issues

    if answer_type == AnswerType.DATE:
        if not isinstance(answer_value, str):
            issues.append(
                _answer_issue(
                    code="answer.type_mismatch",
                    message="date answers must be ISO 8601 strings (YYYY-MM-DD).",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
            return issues
        try:
            parsed = date.fromisoformat(answer_value)
        except ValueError:
            parsed = None
        if parsed is None or parsed.isoformat() != answer_value:
            issues.append(
                _answer_issue(
                    code="answer.date_invalid",
                    message="date answers must use exact YYYY-MM-DD format.",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
        return issues

    if answer_type == AnswerType.FREE_TEXT:
        if not isinstance(answer_value, str) or not answer_value.strip():
            issues.append(
                _answer_issue(
                    code="answer.type_mismatch",
                    message="free_text answers must be non-empty strings.",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
            return issues
        if len(answer_value) > 280:
            issues.append(
                _answer_issue(
                    code="answer.free_text_too_long",
                    message="free_text answers must be at most 280 characters.",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
        return issues

    if answer_type in deterministic_types:
        return issues

    issues.append(
        _answer_issue(
            code="answer.type_unknown",
            message=f"Unsupported answer_type '{answer_type}'.",
            question_id=question_id,
            answer_index=answer_index,
        )
    )
    return issues


def validate_retrieval(
    self: Any,
    retrieval: Any,
    *,
    answer_index: int,
    question_id: str | None,
) -> list[ValidationIssue]:
    """Validate retrieval telemetry payloads for doc/page grounding consistency."""

    issues: list[ValidationIssue] = []
    if not isinstance(retrieval, dict):
        issues.append(
            ValidationIssue(
                severity="error",
                code="telemetry.retrieval.invalid_type",
                message="telemetry.retrieval must be a JSON object.",
                field="telemetry.retrieval",
                question_id=question_id,
                answer_index=answer_index,
            )
        )
        return issues

    retrieved_chunk_pages = retrieval.get("retrieved_chunk_pages")
    if not isinstance(retrieved_chunk_pages, list):
        issues.append(
            ValidationIssue(
                severity="error",
                code="telemetry.retrieval.invalid_pages",
                message="telemetry.retrieval.retrieved_chunk_pages must be a JSON array.",
                field="telemetry.retrieval.retrieved_chunk_pages",
                question_id=question_id,
                answer_index=answer_index,
            )
        )
        return issues

    if not retrieved_chunk_pages:
        if self.policy.unanswerable_retrieval_policy == UnanswerableRetrievalPolicy.REQUIRE_NON_EMPTY:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="telemetry.retrieval.empty_disallowed",
                    message="retrieved_chunk_pages must be non-empty under the active validation policy.",
                    field="telemetry.retrieval.retrieved_chunk_pages",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
        elif self.policy.emit_ambiguity_warning:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="telemetry.retrieval.empty_allowed",
                    message=(
                        "retrieved_chunk_pages is empty. Allowed by local policy for unanswerable cases, "
                        "but the public docs are internally inconsistent."
                    ),
                    field="telemetry.retrieval.retrieved_chunk_pages",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
        return issues

    for item_index, item in enumerate(retrieved_chunk_pages):
        field_prefix = f"telemetry.retrieval.retrieved_chunk_pages[{item_index}]"
        if not isinstance(item, dict):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="telemetry.retrieval.entry_invalid_type",
                    message="Each retrieved_chunk_pages entry must be a JSON object.",
                    field=field_prefix,
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
            continue

        doc_id = item.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="telemetry.retrieval.doc_id_invalid",
                    message="doc_id must be a non-empty string.",
                    field=f"{field_prefix}.doc_id",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
            doc_id = None
        else:
            doc_id = doc_id.strip()
            if self.corpus_registry is not None and not self.corpus_registry.contains(doc_id):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="telemetry.retrieval.doc_id_unknown",
                        message=f"doc_id '{doc_id}' is not present in the supplied corpus directory.",
                        field=f"{field_prefix}.doc_id",
                        question_id=question_id,
                        answer_index=answer_index,
                    )
                )

        page_numbers = item.get("page_numbers")
        if not isinstance(page_numbers, list) or not page_numbers:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="telemetry.retrieval.page_numbers_invalid",
                    message="page_numbers must be a non-empty array of physical 1-based page numbers.",
                    field=f"{field_prefix}.page_numbers",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
            continue

        for page_index, page_number in enumerate(page_numbers):
            page_field = f"{field_prefix}.page_numbers[{page_index}]"
            if not isinstance(page_number, int) or isinstance(page_number, bool):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="telemetry.retrieval.page_number_invalid",
                        message="page_numbers must contain only integer physical page numbers.",
                        field=page_field,
                        question_id=question_id,
                        answer_index=answer_index,
                    )
                )
                continue
            if page_number < 1:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="telemetry.retrieval.page_number_non_positive",
                        message="page_numbers must be 1-based physical PDF page numbers.",
                        field=page_field,
                        question_id=question_id,
                        answer_index=answer_index,
                    )
                )
                continue
            if doc_id is None or self.corpus_registry is None:
                continue
            max_page = self.corpus_registry.page_count_for(doc_id)
            if max_page is not None and page_number > max_page:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="telemetry.retrieval.page_number_out_of_range",
                        message=(
                            f"page {page_number} is out of range for doc_id '{doc_id}' "
                            f"(max page {max_page})."
                        ),
                        field=page_field,
                        question_id=question_id,
                        answer_index=answer_index,
                    )
                )

    return issues
