"""Pre-submit validation gates for packaged submission bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.contracts import (
    UnanswerableRetrievalPolicy,
    load_question_records,
    question_types_from_questions,
)
from src.evaluation.validator import (
    SubmissionValidationReport,
    SubmissionValidator,
    ValidationPolicy,
    build_corpus_registry,
)

MAX_CODE_ARCHIVE_BYTES = 25 * 1024 * 1024


class SubmitGate:
    """Validate a bundle before remote submission."""

    def __init__(
        self,
        *,
        max_code_archive_bytes: int = MAX_CODE_ARCHIVE_BYTES,
        unanswerable_retrieval_policy: UnanswerableRetrievalPolicy = UnanswerableRetrievalPolicy.ALLOW_EMPTY,
    ) -> None:
        """Execute `__init__`.

        Args:
            max_code_archive_bytes: Input parameter.
            unanswerable_retrieval_policy: Input parameter.

        Returns:
            None: This initializer does not return a value.
        """
        self.max_code_archive_bytes = int(max_code_archive_bytes)
        self.unanswerable_retrieval_policy = UnanswerableRetrievalPolicy(unanswerable_retrieval_policy)

    def validate(
        self,
        *,
        submission_payload: dict[str, Any],
        questions_path: str | Path,
        documents_dir: str | Path,
        code_archive_path: str | Path,
    ) -> SubmissionValidationReport:
        """Execute `validate`.

        Args:
            submission_payload: Input parameter.
            questions_path: Input parameter.
            documents_dir: Input parameter.
            code_archive_path: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        archive_path = Path(code_archive_path)
        _validate_code_archive_size(
            archive_path=archive_path,
            max_bytes=self.max_code_archive_bytes,
        )

        validator = _build_validator(
            questions_path=questions_path,
            documents_dir=documents_dir,
            unanswerable_retrieval_policy=self.unanswerable_retrieval_policy,
        )
        report = validator.validate(submission_payload)
        if report.valid:
            return report

        first_error_message = _first_error_message(report)
        if first_error_message is None:
            raise ValueError("Submission bundle failed validation.")
        raise ValueError(f"Submission bundle failed validation: {first_error_message}")


def validate_submission_bundle(
    *,
    submission_payload: dict[str, Any],
    questions_path: str | Path,
    documents_dir: str | Path,
    code_archive_path: str | Path,
    unanswerable_retrieval_policy: UnanswerableRetrievalPolicy = UnanswerableRetrievalPolicy.ALLOW_EMPTY,
) -> SubmissionValidationReport:
    """Validate submission payload + archive size against local rules."""

    gate = SubmitGate(unanswerable_retrieval_policy=unanswerable_retrieval_policy)
    return gate.validate(
        submission_payload=submission_payload,
        questions_path=questions_path,
        documents_dir=documents_dir,
        code_archive_path=code_archive_path,
    )


def _validate_code_archive_size(*, archive_path: Path, max_bytes: int) -> None:
    """Validate code archive size.

    Args:
        archive_path: Input parameter.
        max_bytes: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    if not archive_path.exists():
        raise ValueError(f"Code archive does not exist: {archive_path}")
    if not archive_path.is_file():
        raise ValueError(f"Code archive path is not a file: {archive_path}")
    if archive_path.stat().st_size > max_bytes:
        max_mb = max_bytes // (1024 * 1024)
        raise ValueError(f"Code archive size exceeds {max_mb} MB limit: {archive_path}")


def _build_validator(
    *,
    questions_path: str | Path,
    documents_dir: str | Path,
    unanswerable_retrieval_policy: UnanswerableRetrievalPolicy,
) -> SubmissionValidator:
    """Build validator.

    Args:
        questions_path: Input parameter.
        documents_dir: Input parameter.
        unanswerable_retrieval_policy: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    return SubmissionValidator(
        question_types=question_types_from_questions(load_question_records(questions_path)),
        corpus_registry=build_corpus_registry(documents_dir),
        policy=ValidationPolicy(
            unanswerable_retrieval_policy=UnanswerableRetrievalPolicy(unanswerable_retrieval_policy),
            require_full_question_coverage=True,
            emit_ambiguity_warning=True,
        ),
    )


def _first_error_message(report: SubmissionValidationReport) -> str | None:
    """Execute `_first_error_message`.

    Args:
        report: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    for issue in report.issues:
        if issue.severity == "error":
            return issue.message
    for answer_report in report.answers:
        for issue in answer_report.issues:
            if issue.severity == "error":
                return issue.message
    return None
