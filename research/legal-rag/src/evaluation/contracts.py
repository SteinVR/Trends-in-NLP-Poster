"""Contracts and report models for local evaluation and submission validation."""

from __future__ import annotations

import json
import unicodedata
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.common.schemas import QuestionRecord


class AnswerType(StrEnum):
    """Supported challenge answer types."""

    NUMBER = "number"
    BOOLEAN = "boolean"
    NAME = "name"
    NAMES = "names"
    DATE = "date"
    FREE_TEXT = "free_text"


class BenchmarkSourceType(StrEnum):
    """High-level source family for benchmark slicing."""

    STATUTE = "statute"
    CASE = "case"
    CROSS_CASE = "cross_case"


class BenchmarkDifficulty(StrEnum):
    """Manual difficulty label for benchmark slicing."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class UnanswerableRetrievalPolicy(StrEnum):
    """Policy for the documented empty/non-empty retrieval ambiguity."""

    ALLOW_EMPTY = "allow_empty"
    REQUIRE_NON_EMPTY = "require_non_empty"


class PageReference(BaseModel):
    """A document/page grounding reference."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    page_numbers: list[int] = Field(default_factory=list)

    @field_validator("doc_id")
    @classmethod
    def ensure_doc_id(cls, value: str) -> str:
        """Execute `ensure_doc_id`.

        Args:
            value: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("doc_id must be a non-empty string.")
        return normalized


class ReferenceAnswerRecord(BaseModel):
    """Gold/dev reference row used by the local scorer."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    answer_type: AnswerType
    answer: Any = None
    question: str | None = None
    gold_retrieval: list[PageReference] = Field(default_factory=list)
    source_type: BenchmarkSourceType | None = None
    difficulty: BenchmarkDifficulty | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("question_id")
    @classmethod
    def ensure_question_id(cls, value: str) -> str:
        """Execute `ensure_question_id`.

        Args:
            value: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("question_id must be a non-empty string.")
        return normalized


class AssistantScoreRecord(BaseModel):
    """Manual/proxy free-text score for a specific submission answer."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    assistant_score: float
    notes: str | None = None

    @field_validator("question_id")
    @classmethod
    def ensure_question_id(cls, value: str) -> str:
        """Execute `ensure_question_id`.

        Args:
            value: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("question_id must be a non-empty string.")
        return normalized

    @field_validator("assistant_score")
    @classmethod
    def ensure_score_range(cls, value: float) -> float:
        """Execute `ensure_score_range`.

        Args:
            value: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        if value < 0 or value > 1:
            raise ValueError("assistant_score must be between 0 and 1.")
        return value


class ValidationIssue(BaseModel):
    """Single validator/scorer issue."""

    model_config = ConfigDict(extra="forbid")

    severity: Literal["error", "warning"]
    code: str
    message: str
    field: str | None = None
    question_id: str | None = None
    answer_index: int | None = None


class AnswerValidationReport(BaseModel):
    """Validation result for a single answer payload."""

    model_config = ConfigDict(extra="forbid")

    answer_index: int
    question_id: str | None = None
    answer_type: str | None = None
    valid: bool
    telemetry_malformed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class SubmissionValidationReport(BaseModel):
    """Structured report returned by the submission validator."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    expected_questions: int
    evaluated_answers: int
    malformed_answers: int
    error_count: int
    warning_count: int
    unanswerable_retrieval_policy: UnanswerableRetrievalPolicy
    issues: list[ValidationIssue] = Field(default_factory=list)
    answers: list[AnswerValidationReport] = Field(default_factory=list)


class AnswerScoreBreakdown(BaseModel):
    """Per-question diagnostic breakdown."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    answer_type: AnswerType
    deterministic_score: float | None = None
    assistant_score: float | None = None
    grounding_score: float
    telemetry_factor: float
    ttft_ms: float | None = None
    ttft_factor: float
    notes: list[str] = Field(default_factory=list)


class ScoreMetrics(BaseModel):
    """Submission-level score aggregates."""

    model_config = ConfigDict(extra="forbid")

    deterministic: float
    assistant: float
    grounding: float
    telemetry: float
    ttft_ms: float | None = None
    ttft_multiplier: float
    total_score: float
    deterministic_questions: int
    free_text_questions: int
    scored_questions: int


class ScoreReport(BaseModel):
    """Structured scoring output."""

    model_config = ConfigDict(extra="forbid")

    metrics: ScoreMetrics
    validation: SubmissionValidationReport
    answer_scores: list[AnswerScoreBreakdown] = Field(default_factory=list)
    unscored_question_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def load_submission_payload(path: str | Path) -> Any:
    """Load a submission JSON payload without applying schema coercion."""

    submission_path = Path(path)
    with submission_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_question_records(path: str | Path) -> list[QuestionRecord]:
    """Load the platform question metadata file."""

    questions_path = Path(path)
    with questions_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Questions file must contain a JSON array: {questions_path}")

    return [QuestionRecord.model_validate(item) for item in payload]


def load_reference_records(path: str | Path) -> list[ReferenceAnswerRecord]:
    """Load reference answers from JSON or JSONL."""

    payload = _load_records_payload(path, primary_key="references")
    return [ReferenceAnswerRecord.model_validate(item) for item in payload]


def load_assistant_score_records(path: str | Path) -> list[AssistantScoreRecord]:
    """Load manual/proxy free-text scores from JSON or JSONL."""

    payload = _load_records_payload(path, primary_key="assistant_scores")
    return [AssistantScoreRecord.model_validate(item) for item in payload]


def normalize_text(value: str) -> str:
    """Normalize text for exact-name matching."""

    collapsed = " ".join(unicodedata.normalize("NFKC", value).split())
    return collapsed.casefold()


def flatten_page_references(references: list[PageReference]) -> set[tuple[str, int]]:
    """Flatten grouped page references into (doc_id, page_number) tuples."""

    flattened: set[tuple[str, int]] = set()
    for reference in references:
        for page_number in reference.page_numbers:
            flattened.add((reference.doc_id, page_number))
    return flattened


def question_types_from_questions(questions: list[QuestionRecord]) -> dict[str, AnswerType]:
    """Build a question_id -> answer_type mapping from platform question metadata."""

    question_types: dict[str, AnswerType] = {}
    for question in questions:
        answer_type = AnswerType(str(question.answer_type))
        if question.id in question_types:
            raise ValueError(f"Duplicate question_id in questions metadata: {question.id}")
        question_types[question.id] = answer_type
    return question_types


def question_types_from_references(references: list[ReferenceAnswerRecord]) -> dict[str, AnswerType]:
    """Build a question_id -> answer_type mapping from a reference subset."""

    question_types: dict[str, AnswerType] = {}
    for reference in references:
        if reference.question_id in question_types:
            raise ValueError(f"Duplicate question_id in references: {reference.question_id}")
        question_types[reference.question_id] = reference.answer_type
    return question_types


def _load_records_payload(path: str | Path, *, primary_key: str) -> list[dict[str, Any]]:
    """Load records payload.

    Args:
        path: Input parameter.
        primary_key: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    source_path = Path(path)
    if source_path.suffix.lower() == ".jsonl":
        with source_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        with source_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            rows = payload.get(primary_key)
            if rows is None and primary_key == "references":
                rows = payload.get("answers")
        else:
            rows = payload

    if not isinstance(rows, list):
        raise ValueError(f"Expected {source_path} to contain a JSON array or '{primary_key}' array.")

    return rows
