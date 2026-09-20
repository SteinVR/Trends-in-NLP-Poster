"""Answer routing and extraction-first solving."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.common.schemas import QuestionRecord
from src.evaluation.contracts import AnswerType, PageReference
from src.retrieval.service import RetrievalResult

from .service_constants import (
    _CASE_ID_PATTERN,
    _DIFC_CASE_REF_PATTERN,
    _DOCUMENT_TITLE_CUE_PATTERN,
    _MULTI_DOCUMENT_SIGNAL_PATTERN,
    _STATUTE_ID_PATTERN,
)

if TYPE_CHECKING:
    from src.answering.page_attribution import PageAttributionTrace

del TYPE_CHECKING

CANONICAL_UNANSWERABLE_FREE_TEXT_ANSWER = "There is no information on this question in the provided documents."
SCHEMA_FIRST_ANSWER_TYPES: frozenset[AnswerType] = frozenset(
    {
        AnswerType.BOOLEAN,
        AnswerType.NUMBER,
        AnswerType.NAME,
        AnswerType.NAMES,
        AnswerType.DATE,
        AnswerType.FREE_TEXT,
    }
)
DEFAULT_REASONING_EFFORT_BY_TYPE: dict[str, str] = {
    AnswerType.BOOLEAN.value: "high",
    AnswerType.NUMBER.value: "high",
    AnswerType.NAME.value: "high",
    AnswerType.NAMES.value: "high",
    AnswerType.DATE.value: "high",
    AnswerType.FREE_TEXT.value: "high",
}
DEFAULT_CONFIDENCE_THRESHOLD_BY_TYPE: dict[str, float] = {
    AnswerType.BOOLEAN.value: 0.7,
    AnswerType.NUMBER.value: 0.55,
    AnswerType.NAME.value: 0.75,
    AnswerType.NAMES.value: 0.75,
    AnswerType.DATE.value: 0.6,
    AnswerType.FREE_TEXT.value: 0.5,
}


class FreeTextGroundingValidationError(ValueError):
    """Raised when a generated free-text answer is unsupported by the evidence."""


def _normalize_whitespace(value: str) -> str:
    """Normalize Unicode whitespace and collapse multi-space sequences."""

    return " ".join(unicodedata.normalize("NFKC", value).split())


def _parse_iso_date_parts(value: str) -> tuple[int, int, int]:
    """Parse a strict ISO-8601 date string and return numeric components."""

    parsed_date = date.fromisoformat(value)
    return parsed_date.year, parsed_date.month, parsed_date.day


def _safe_iso_date(*, year: int, month: int, day: int) -> str | None:
    """Return normalized ISO-8601 date or `None` when the date is invalid."""

    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _copy_page_references(references: list[PageReference]) -> list[PageReference]:
    """Clone page reference payloads to avoid accidental mutation sharing."""

    return [
        PageReference(
            doc_id=reference.doc_id,
            page_numbers=list(reference.page_numbers),
        )
        for reference in references
    ]


class _StructuredAnswerBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_answerable: bool
    relevant_evidence_indices: list[int] = Field(default_factory=list)
    relevant_pages: list[int] = Field(default_factory=list)
    reasoning_summary: str
    confidence: float | None = None

    @staticmethod
    def _normalize_positive_integer_list(value: Any, *, field_name: str) -> list[int]:
        """Normalize positive integer lists used by structured page hints."""

        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be a list of integers.")
        normalized: list[int] = []
        for raw_item in value:
            try:
                item = int(raw_item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} must contain integers.") from exc
            if item > 0:
                normalized.append(item)
        return sorted(set(normalized))

    @field_validator("relevant_evidence_indices", mode="before")
    @classmethod
    def _normalize_relevant_evidence_indices(cls, value: Any) -> list[int]:
        """Normalize evidence indices selected by the structured solver."""

        return cls._normalize_positive_integer_list(value, field_name="relevant_evidence_indices")

    @field_validator("relevant_pages", mode="before")
    @classmethod
    def _normalize_relevant_pages(cls, value: Any) -> list[int]:
        """Normalize relevant pages."""
        return cls._normalize_positive_integer_list(value, field_name="relevant_pages")

    @field_validator("reasoning_summary")
    @classmethod
    def _validate_reasoning_summary(cls, value: str) -> str:
        """Validate reasoning summary."""
        normalized = _normalize_whitespace(str(value))
        if not normalized:
            raise ValueError("reasoning_summary must be non-empty.")
        return normalized

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: float | None) -> float | None:
        """Validate confidence."""
        if value is None:
            return value
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("confidence must be finite and between 0 and 1.")
        return value


class BooleanAnswerSchema(_StructuredAnswerBase):
    final_answer: bool | None

    @model_validator(mode="after")
    def _validate_answerable_contract(self) -> "BooleanAnswerSchema":
        """Validate answerable contract."""
        if self.is_answerable != (self.final_answer is not None):
            raise ValueError("boolean final_answer must be null iff is_answerable=false.")
        return self


class NumberAnswerSchema(_StructuredAnswerBase):
    final_answer: int | float | None

    @model_validator(mode="after")
    def _validate_answerable_contract(self) -> "NumberAnswerSchema":
        """Validate answerable contract."""
        if self.is_answerable != (self.final_answer is not None):
            raise ValueError("number final_answer must be null iff is_answerable=false.")
        return self


class NameAnswerSchema(_StructuredAnswerBase):
    final_answer: str | None

    @model_validator(mode="after")
    def _validate_answerable_contract(self) -> "NameAnswerSchema":
        """Validate answerable contract."""
        if self.is_answerable != (self.final_answer is not None):
            raise ValueError("name final_answer must be null iff is_answerable=false.")
        if self.final_answer is not None:
            normalized = _normalize_whitespace(self.final_answer)
            if not normalized:
                raise ValueError("name final_answer must be non-empty.")
            self.final_answer = normalized
        return self


class NamesAnswerSchema(_StructuredAnswerBase):
    final_answer: list[str] | None

    @model_validator(mode="after")
    def _validate_answerable_contract(self) -> "NamesAnswerSchema":
        """Validate answerable contract."""
        if self.is_answerable != (self.final_answer is not None):
            raise ValueError("names final_answer must be null iff is_answerable=false.")
        if self.final_answer is not None:
            normalized: list[str] = []
            seen: set[str] = set()
            for item in self.final_answer:
                candidate = _normalize_whitespace(str(item))
                if not candidate:
                    continue
                key = candidate.casefold()
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(candidate)
            if not normalized:
                raise ValueError("names final_answer must contain at least one non-empty name.")
            self.final_answer = normalized
        return self


class DateAnswerSchema(_StructuredAnswerBase):
    final_answer: str | None

    @model_validator(mode="after")
    def _validate_answerable_contract(self) -> "DateAnswerSchema":
        """Validate answerable contract."""
        if self.is_answerable != (self.final_answer is not None):
            raise ValueError("date final_answer must be null iff is_answerable=false.")
        if self.final_answer is not None:
            normalized = _normalize_whitespace(self.final_answer)
            year, month, day = _parse_iso_date_parts(normalized)
            if _safe_iso_date(year=year, month=month, day=day) is None:
                raise ValueError("date final_answer must be ISO-8601 (YYYY-MM-DD).")
            self.final_answer = normalized
        return self


class FreeTextAnswerSchema(_StructuredAnswerBase):
    final_answer: str = Field(min_length=1, max_length=280)

    @field_validator("final_answer", mode="before")
    @classmethod
    def _normalize_final_answer(cls, value: Any) -> str:
        """Normalize final answer text before field-level constraints are enforced."""
        if value is None:
            return ""
        return _normalize_whitespace(str(value))

    @model_validator(mode="after")
    def _validate_answerable_contract(self) -> "FreeTextAnswerSchema":
        """Validate answerable contract."""
        normalized = self.final_answer
        if not normalized:
            raise ValueError("free_text final_answer must be non-empty.")
        if len(normalized) > 280:
            raise ValueError("free_text final_answer must be at most 280 characters.")
        if not self.is_answerable and normalized != CANONICAL_UNANSWERABLE_FREE_TEXT_ANSWER:
            raise ValueError("free_text unanswerable output must be the canonical sentence.")
        self.final_answer = normalized
        return self


ANSWER_SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    AnswerType.BOOLEAN.value: BooleanAnswerSchema,
    AnswerType.NUMBER.value: NumberAnswerSchema,
    AnswerType.NAME.value: NameAnswerSchema,
    AnswerType.NAMES.value: NamesAnswerSchema,
    AnswerType.DATE.value: DateAnswerSchema,
    AnswerType.FREE_TEXT.value: FreeTextAnswerSchema,
}


@dataclass(frozen=True, slots=True)
class LegalEntityMetadata:
    entity_type: Literal["case_id", "statute_id"]
    raw_value: str
    normalized_value: str
    is_strong: bool


@dataclass(frozen=True, slots=True)
class QueryMetadata:
    legal_entities: tuple[LegalEntityMetadata, ...]
    document_title_cues: tuple[str, ...]
    likely_multi_document: bool


def extract_query_metadata(question_text: str) -> QueryMetadata:
    """Extract query metadata."""
    normalized_question = _normalize_whitespace(question_text)
    lowered_question = normalized_question.casefold()

    legal_entities: list[LegalEntityMetadata] = []
    seen_entities: set[tuple[str, str]] = set()

    for match in _CASE_ID_PATTERN.finditer(normalized_question):
        raw_value = _normalize_whitespace(match.group("case_id"))
        normalized_value = raw_value.replace(" ", "")
        entity_key = ("case_id", normalized_value.casefold())
        if entity_key in seen_entities:
            continue
        seen_entities.add(entity_key)
        legal_entities.append(
            LegalEntityMetadata(
                entity_type="case_id",
                raw_value=raw_value,
                normalized_value=normalized_value,
                is_strong=True,
            )
        )

    for match in _DIFC_CASE_REF_PATTERN.finditer(normalized_question):
        raw_value = _normalize_whitespace(match.group("case_id"))
        normalized_value = raw_value.replace(" ", "")
        entity_key = ("case_id", normalized_value.casefold())
        if entity_key in seen_entities:
            continue
        seen_entities.add(entity_key)
        legal_entities.append(
            LegalEntityMetadata(
                entity_type="case_id",
                raw_value=raw_value,
                normalized_value=normalized_value,
                is_strong=True,
            )
        )

    for match in _STATUTE_ID_PATTERN.finditer(normalized_question):
        raw_value = _normalize_whitespace(match.group("statute"))
        raw_value = re.sub(r"^(?:which|what|under|the)\s+", "", raw_value, flags=re.IGNORECASE)
        if not raw_value:
            continue
        normalized_value = raw_value.casefold()
        entity_key = ("statute_id", normalized_value)
        if entity_key in seen_entities:
            continue
        seen_entities.add(entity_key)
        legal_entities.append(
            LegalEntityMetadata(
                entity_type="statute_id",
                raw_value=raw_value,
                normalized_value=normalized_value,
                is_strong=True,
            )
        )

    document_title_cues: list[str] = []
    seen_titles: set[str] = set()
    for match in _DOCUMENT_TITLE_CUE_PATTERN.finditer(normalized_question):
        title = _normalize_whitespace(match.group("title"))
        key = title.casefold()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        document_title_cues.append(title)

    case_entity_count = sum(1 for entity in legal_entities if entity.entity_type == "case_id")
    likely_multi_document = (
        case_entity_count >= 2
        or bool(_MULTI_DOCUMENT_SIGNAL_PATTERN.search(lowered_question))
        or lowered_question.count(" and ") >= 2
    )

    return QueryMetadata(
        legal_entities=tuple(legal_entities),
        document_title_cues=tuple(document_title_cues),
        likely_multi_document=likely_multi_document,
    )


class Retriever(Protocol):
    """Retrieval interface expected by the answering service."""

    def retrieve(
        self,
        question: QuestionRecord,
        *,
        query_metadata: QueryMetadata | None = None,
    ) -> RetrievalResult | list[dict[str, Any]]:
        """Return candidate evidence pages or a typed retrieval result."""


@dataclass(slots=True)
class AnswerResult:
    """Structured answer returned by the answering service."""

    question_id: str
    answer: Any
    confidence: float
    evidence_pages: list[PageReference]
    page_trace: PageAttributionTrace | None = None
    model_name: str | None = None

    def __post_init__(self) -> None:
        # Keep one authoritative source for final pages: page_trace.final_emitted_pages.
        """Execute `__post_init__`."""
        if self.page_trace is not None:
            self.evidence_pages = _copy_page_references(self.page_trace.final_emitted_pages)
        else:
            self.evidence_pages = _copy_page_references(self.evidence_pages)


@dataclass(slots=True)
class _PageCandidate:
    doc_id: str
    page_numbers: tuple[int, ...]
    text: str
    include_page_numbers: bool = False

    @property
    def page_number(self) -> int:
        """Execute `page_number`."""
        return self.page_numbers[0]


@dataclass(slots=True)
class _SolvedAnswer:
    answer: Any
    supporting_pages: list[_PageCandidate]


DeterministicExtractor = Callable[..., _SolvedAnswer | None]

__all__ = [name for name in globals() if not name.startswith("__")]
