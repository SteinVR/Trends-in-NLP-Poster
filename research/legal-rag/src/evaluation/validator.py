"""Local submission validator mirroring the public challenge contract."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fitz

from .contracts import (
    AnswerType,
    AnswerValidationReport,
    SubmissionValidationReport,
    UnanswerableRetrievalPolicy,
    ValidationIssue,
)
from .validator_checks import validate_answer_value, validate_retrieval


@dataclass(slots=True)
class ValidationPolicy:
    """Runtime validation policy toggles."""

    unanswerable_retrieval_policy: UnanswerableRetrievalPolicy = UnanswerableRetrievalPolicy.ALLOW_EMPTY
    require_full_question_coverage: bool = True
    emit_ambiguity_warning: bool = True

@dataclass(slots=True)
class CorpusRegistry:
    """Known corpus documents and their physical page counts."""

    page_counts: dict[str, int]

    @classmethod
    def from_pdf_directory(cls, pdf_dir: str | Path) -> "CorpusRegistry":
        """Execute `from_pdf_directory`."""
        root = Path(pdf_dir)
        if not root.exists():
            raise ValueError(f"Documents directory does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Documents path is not a directory: {root}")

        page_counts: dict[str, int] = {}
        for pdf_path in sorted(root.rglob("*.pdf")):
            with fitz.open(pdf_path) as document:
                page_counts[pdf_path.stem] = document.page_count
        return cls(page_counts=page_counts)

    def contains(self, doc_id: str) -> bool:
        """Execute `contains`."""
        return doc_id in self.page_counts

    def page_count_for(self, doc_id: str) -> int | None:
        """Execute `page_count_for`."""
        return self.page_counts.get(doc_id)

class SubmissionValidator:
    """Validate raw submission payloads against question types and local corpus metadata."""

    def __init__(
        self,
        *,
        question_types: dict[str, AnswerType],
        required_question_ids: set[str] | None = None,
        corpus_registry: CorpusRegistry | None = None,
        policy: ValidationPolicy | None = None,
    ) -> None:
        """Execute `__init__`."""
        self.question_types = question_types
        self.required_question_ids = required_question_ids or set(question_types)
        self.corpus_registry = corpus_registry
        self.policy = policy or ValidationPolicy()

    def validate(self, submission_payload: Any) -> SubmissionValidationReport:
        """Execute `validate`."""
        issues: list[ValidationIssue] = []
        answer_reports: list[AnswerValidationReport] = []
        malformed_answers = 0

        if not isinstance(submission_payload, dict):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="submission.invalid_type",
                    message="Submission payload must be a JSON object.",
                    field="$",
                )
            )
            return self._build_report(issues=issues, answer_reports=answer_reports, malformed_answers=malformed_answers)

        architecture_summary = submission_payload.get("architecture_summary")
        if architecture_summary is not None:
            if not isinstance(architecture_summary, str):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="submission.architecture_summary.invalid_type",
                        message="architecture_summary must be a string when present.",
                        field="architecture_summary",
                    )
                )
            elif len(architecture_summary) > 500:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="submission.architecture_summary.too_long",
                        message="architecture_summary must be at most 500 characters.",
                        field="architecture_summary",
                    )
                )

        raw_answers = submission_payload.get("answers")
        if not isinstance(raw_answers, list):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="submission.answers.invalid_type",
                    message="answers must be a JSON array.",
                    field="answers",
                )
            )
            return self._build_report(issues=issues, answer_reports=answer_reports, malformed_answers=malformed_answers)

        duplicate_ids, submitted_question_ids = _collect_question_id_sets(raw_answers)

        if self.policy.require_full_question_coverage:
            missing_question_ids = sorted(self.required_question_ids - submitted_question_ids)
            for question_id in missing_question_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="submission.answer_missing",
                        message=f"Missing answer for question_id '{question_id}'.",
                        question_id=question_id,
                    )
                )

        for answer_index, raw_answer in enumerate(raw_answers):
            answer_issues, question_id, answer_type = self._validate_answer(
                raw_answer,
                answer_index=answer_index,
                duplicate_ids=duplicate_ids,
            )
            telemetry_malformed = _has_error_with_prefix(answer_issues, "telemetry.")
            malformed_answers += int(telemetry_malformed)
            answer_reports.append(
                AnswerValidationReport(
                    answer_index=answer_index,
                    question_id=question_id,
                    answer_type=answer_type.value if answer_type is not None else None,
                    valid=not _has_error(answer_issues),
                    telemetry_malformed=telemetry_malformed,
                    issues=answer_issues,
                )
            )

        return self._build_report(issues=issues, answer_reports=answer_reports, malformed_answers=malformed_answers)

    def _validate_answer(
        self,
        raw_answer: Any,
        *,
        answer_index: int,
        duplicate_ids: set[str],
    ) -> tuple[list[ValidationIssue], str | None, AnswerType | None]:
        """Validate answer."""
        issues: list[ValidationIssue] = []
        if not isinstance(raw_answer, dict):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="submission.answer.invalid_type",
                    message="Each answers[] entry must be a JSON object.",
                    field=f"answers[{answer_index}]",
                    answer_index=answer_index,
                )
            )
            return issues, None, None

        question_id = raw_answer.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="submission.question_id.invalid",
                    message="question_id must be a non-empty string.",
                    field="question_id",
                    answer_index=answer_index,
                )
            )
            return issues, None, None

        if question_id != question_id.strip():
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="submission.question_id.whitespace",
                    message="question_id must match the dataset exactly and must not include surrounding whitespace.",
                    field="question_id",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
            return issues, question_id, None

        answer_type = self.question_types.get(question_id)
        if answer_type is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="submission.question_id.unknown",
                    message=f"question_id '{question_id}' is not present in the supplied question set.",
                    field="question_id",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
        elif question_id in duplicate_ids:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="submission.question_id.duplicate",
                    message=f"question_id '{question_id}' appears more than once in answers[].",
                    field="question_id",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )

        telemetry = raw_answer.get("telemetry")
        issues.extend(
            self._validate_telemetry(
                telemetry,
                answer_index=answer_index,
                question_id=question_id,
            )
        )

        if answer_type is not None:
            issues.extend(
                self._validate_answer_value(
                    raw_answer.get("answer"),
                    answer_type=answer_type,
                    answer_index=answer_index,
                    question_id=question_id,
                )
            )

        return issues, question_id, answer_type

    def _validate_answer_value(
        self,
        answer_value: Any,
        *,
        answer_type: AnswerType,
        answer_index: int,
        question_id: str | None,
    ) -> list[ValidationIssue]:
        """Validate a typed answer value using shared strict checks."""

        return validate_answer_value(
            self,
            answer_value,
            answer_type=answer_type,
            answer_index=answer_index,
            question_id=question_id,
            answer_issue=_answer_issue,
            is_number=_is_number,
        )

    def _validate_telemetry(
        self,
        telemetry: Any,
        *,
        answer_index: int,
        question_id: str | None,
    ) -> list[ValidationIssue]:
        """Validate telemetry."""
        issues: list[ValidationIssue] = []
        if not isinstance(telemetry, dict):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="telemetry.invalid_type",
                    message="telemetry must be a JSON object.",
                    field="telemetry",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
            return issues

        timing = telemetry.get("timing")
        issues.extend(self._validate_timing(timing, answer_index=answer_index, question_id=question_id))

        retrieval = telemetry.get("retrieval")
        issues.extend(self._validate_retrieval(retrieval, answer_index=answer_index, question_id=question_id))

        usage = telemetry.get("usage")
        issues.extend(self._validate_usage(usage, answer_index=answer_index, question_id=question_id))

        model_name = telemetry.get("model_name")
        if model_name is not None and (not isinstance(model_name, str) or not model_name.strip()):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="telemetry.model_name.invalid",
                    message="model_name must be a non-empty string when present.",
                    field="telemetry.model_name",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )

        return issues

    def _validate_timing(
        self,
        timing: Any,
        *,
        answer_index: int,
        question_id: str | None,
    ) -> list[ValidationIssue]:
        """Validate timing."""
        issues: list[ValidationIssue] = []
        if not isinstance(timing, dict):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="telemetry.timing.invalid_type",
                    message="telemetry.timing must be a JSON object.",
                    field="telemetry.timing",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
            return issues

        timing_values = {}
        for key in ("ttft_ms", "tpot_ms", "total_time_ms"):
            raw_value = timing.get(key)
            if not _is_number(raw_value):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="telemetry.timing.invalid_value",
                        message=f"telemetry.timing.{key} must be a non-negative number.",
                        field=f"telemetry.timing.{key}",
                        question_id=question_id,
                        answer_index=answer_index,
                    )
                )
                continue
            numeric_value = float(raw_value)
            timing_values[key] = numeric_value
            if numeric_value < 0:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="telemetry.timing.negative",
                        message=f"telemetry.timing.{key} must be non-negative.",
                        field=f"telemetry.timing.{key}",
                        question_id=question_id,
                        answer_index=answer_index,
                    )
                )

        if {"ttft_ms", "total_time_ms"} <= timing_values.keys():
            if timing_values["ttft_ms"] > timing_values["total_time_ms"]:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="telemetry.timing.ttft_gt_total",
                        message="telemetry.timing.ttft_ms must be less than or equal to total_time_ms.",
                        field="telemetry.timing.ttft_ms",
                        question_id=question_id,
                        answer_index=answer_index,
                    )
                )

        return issues

    def _validate_retrieval(
        self,
        retrieval: Any,
        *,
        answer_index: int,
        question_id: str | None,
    ) -> list[ValidationIssue]:
        """Validate retrieval telemetry payloads using shared strict checks."""

        return validate_retrieval(
            self,
            retrieval,
            answer_index=answer_index,
            question_id=question_id,
        )

    def _validate_usage(
        self,
        usage: Any,
        *,
        answer_index: int,
        question_id: str | None,
    ) -> list[ValidationIssue]:
        """Validate usage."""
        issues: list[ValidationIssue] = []
        if not isinstance(usage, dict):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="telemetry.usage.invalid_type",
                    message="telemetry.usage must be a JSON object.",
                    field="telemetry.usage",
                    question_id=question_id,
                    answer_index=answer_index,
                )
            )
            return issues

        for key in ("input_tokens", "output_tokens"):
            raw_value = usage.get(key)
            if not isinstance(raw_value, int) or isinstance(raw_value, bool):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="telemetry.usage.invalid_value",
                        message=f"telemetry.usage.{key} must be a non-negative integer.",
                        field=f"telemetry.usage.{key}",
                        question_id=question_id,
                        answer_index=answer_index,
                    )
                )
                continue
            if raw_value < 0:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="telemetry.usage.negative",
                        message=f"telemetry.usage.{key} must be non-negative.",
                        field=f"telemetry.usage.{key}",
                        question_id=question_id,
                        answer_index=answer_index,
                    )
                )

        return issues

    def _build_report(
        self,
        *,
        issues: list[ValidationIssue],
        answer_reports: list[AnswerValidationReport],
        malformed_answers: int,
    ) -> SubmissionValidationReport:
        """Build report."""
        error_count = sum(issue.severity == "error" for issue in issues)
        warning_count = sum(issue.severity == "warning" for issue in issues)
        for answer_report in answer_reports:
            error_count += sum(issue.severity == "error" for issue in answer_report.issues)
            warning_count += sum(issue.severity == "warning" for issue in answer_report.issues)

        return SubmissionValidationReport(
            valid=error_count == 0,
            expected_questions=len(self.question_types),
            evaluated_answers=len(answer_reports),
            malformed_answers=malformed_answers,
            error_count=error_count,
            warning_count=warning_count,
            unanswerable_retrieval_policy=self.policy.unanswerable_retrieval_policy,
            issues=issues,
            answers=answer_reports,
        )

def build_corpus_registry(pdf_dir: str | Path | None) -> CorpusRegistry | None:
    """Construct a corpus registry when local PDFs are available."""

    if pdf_dir is None:
        return None
    return CorpusRegistry.from_pdf_directory(pdf_dir)

def collect_answer_dicts(submission_payload: Any) -> list[dict[str, Any]]:
    """Return the raw answers[] objects from a submission payload."""

    if not isinstance(submission_payload, dict):
        return []
    raw_answers = submission_payload.get("answers")
    if not isinstance(raw_answers, list):
        return []
    return [answer for answer in raw_answers if isinstance(answer, dict)]

def find_answer_by_question_id(answer_dicts: Iterable[dict[str, Any]], question_id: str) -> dict[str, Any] | None:
    """Return the first answer object matching a question_id."""

    for answer in answer_dicts:
        raw_question_id = answer.get("question_id")
        if raw_question_id == question_id:
            return answer
    return None

def extract_retrieved_pairs(raw_answer: dict[str, Any] | None) -> set[tuple[str, int]]:
    """Extract valid-looking (doc_id, page_number) pairs from a raw answer payload."""

    if not isinstance(raw_answer, dict):
        return set()
    telemetry = raw_answer.get("telemetry")
    if not isinstance(telemetry, dict):
        return set()
    retrieval = telemetry.get("retrieval")
    if not isinstance(retrieval, dict):
        return set()
    retrieved_chunk_pages = retrieval.get("retrieved_chunk_pages")
    if not isinstance(retrieved_chunk_pages, list):
        return set()

    pairs: set[tuple[str, int]] = set()
    for item in retrieved_chunk_pages:
        if not isinstance(item, dict):
            continue
        doc_id = item.get("doc_id")
        page_numbers = item.get("page_numbers")
        if not isinstance(doc_id, str) or not doc_id.strip():
            continue
        if not isinstance(page_numbers, list):
            continue
        for page_number in page_numbers:
            if isinstance(page_number, int) and not isinstance(page_number, bool) and page_number > 0:
                pairs.add((doc_id.strip(), page_number))
    return pairs

def extract_ttft_ms(raw_answer: dict[str, Any] | None) -> float | None:
    """Extract ttft_ms from a raw answer payload when it is numeric."""

    if not isinstance(raw_answer, dict):
        return None
    telemetry = raw_answer.get("telemetry")
    if not isinstance(telemetry, dict):
        return None
    timing = telemetry.get("timing")
    if not isinstance(timing, dict):
        return None
    ttft_ms = timing.get("ttft_ms")
    if not _is_number(ttft_ms):
        return None
    return float(ttft_ms)

def _is_number(value: Any) -> bool:
    """Return whether number is true."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def _collect_question_id_sets(raw_answers: list[Any]) -> tuple[set[str], set[str]]:
    """Execute `_collect_question_id_sets`."""
    question_ids = [answer.get("question_id") for answer in raw_answers if isinstance(answer, dict)]
    normalized_question_ids = [
        question_id.strip() for question_id in question_ids if isinstance(question_id, str) and question_id.strip()
    ]
    duplicate_ids = {question_id for question_id, count in Counter(normalized_question_ids).items() if count > 1}
    submitted_question_ids = {
        question_id
        for question_id in question_ids
        if isinstance(question_id, str) and question_id and question_id == question_id.strip()
    }
    return duplicate_ids, submitted_question_ids

def _has_error(issues: Iterable[ValidationIssue]) -> bool:
    """Return whether error is present."""
    return any(issue.severity == "error" for issue in issues)

def _has_error_with_prefix(issues: Iterable[ValidationIssue], prefix: str) -> bool:
    """Return whether error with prefix is present."""
    return any(issue.severity == "error" and issue.code.startswith(prefix) for issue in issues)

def _answer_issue(
    *,
    code: str,
    message: str,
    question_id: str | None,
    answer_index: int,
) -> ValidationIssue:
    """Execute `_answer_issue`."""
    return ValidationIssue(
        severity="error",
        code=code,
        message=message,
        field="answer",
        question_id=question_id,
        answer_index=answer_index,
    )
