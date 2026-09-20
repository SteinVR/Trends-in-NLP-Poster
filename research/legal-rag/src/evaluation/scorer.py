"""Deterministic local scorer for challenge-style submissions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isclose
from typing import Any

from .contracts import (
    AnswerScoreBreakdown,
    AnswerType,
    AssistantScoreRecord,
    ReferenceAnswerRecord,
    ScoreMetrics,
    ScoreReport,
    SubmissionValidationReport,
    UnanswerableRetrievalPolicy,
    flatten_page_references,
    normalize_text,
    question_types_from_references,
)
from .validator import (
    SubmissionValidator,
    ValidationPolicy,
    build_corpus_registry,
    collect_answer_dicts,
    extract_retrieved_pairs,
    extract_ttft_ms,
    find_answer_by_question_id,
)

GROUNDING_BETA = 2.5
TTFT_PROXY_PENALTY_START_MS = 3_000
TTFT_PROXY_PENALTY_START_FACTOR = 0.99
TTFT_PROXY_PENALTY_FLOOR = 0.85
TTFT_PROXY_PENALTY_CAP_MS = 10_000
TTFT_PROXY_NOTE = (
    "TTFT > 3000 ms uses a local linear proxy because the public docs expose only a range "
    "(0.85-0.99), not the exact platform function."
)
MISSING_ASSISTANT_SCORE_NOTE = "No assistant_score provided for this free_text answer; defaulted to 0.0."
NO_DETERMINISTIC_NOTE = "Reference subset contains no deterministic questions; deterministic is forced to 0.0."
NO_FREE_TEXT_NOTE = "Reference subset contains no free_text questions; assistant is forced to 0.0."
PARTIAL_FREE_TEXT_SCORES_NOTE = (
    "Some free_text questions are missing manual/proxy assistant scores; those answers default to 0.0."
)


@dataclass(slots=True)
class ScoreOptions:
    """Runtime scorer options."""

    unanswerable_retrieval_policy: UnanswerableRetrievalPolicy = UnanswerableRetrievalPolicy.ALLOW_EMPTY
    documents_dir: str | None = None
    validation_question_types: dict[str, AnswerType] | None = None


def score_submission(
    submission_payload: Any,
    *,
    references: list[ReferenceAnswerRecord],
    assistant_scores: list[AssistantScoreRecord] | None = None,
    options: ScoreOptions | None = None,
) -> ScoreReport:
    """Score a raw submission against a reference subset."""

    options = options or ScoreOptions()
    corpus_registry = build_corpus_registry(options.documents_dir)
    reference_question_types = question_types_from_references(references)
    validation_question_types = options.validation_question_types or reference_question_types
    validation = SubmissionValidator(
        question_types=validation_question_types,
        required_question_ids=set(reference_question_types),
        corpus_registry=corpus_registry,
        policy=ValidationPolicy(
            unanswerable_retrieval_policy=options.unanswerable_retrieval_policy,
            require_full_question_coverage=True,
            emit_ambiguity_warning=True,
        ),
    ).validate(submission_payload)

    assistant_scores_by_question = {
        record.question_id: record.assistant_score for record in assistant_scores or []
    }
    answer_dicts = collect_answer_dicts(submission_payload)
    answer_scores: list[AnswerScoreBreakdown] = []
    notes: list[str] = [TTFT_PROXY_NOTE]

    deterministic_values: list[float] = []
    assistant_values: list[float] = []
    grounding_values: list[float] = []
    telemetry_values: list[float] = []
    ttft_values: list[float] = []
    ttft_ms_values: list[float] = []

    for reference in references:
        raw_answer = find_answer_by_question_id(answer_dicts, reference.question_id)
        answer_notes: list[str] = []

        deterministic_score: float | None = None
        if reference.answer_type != AnswerType.FREE_TEXT:
            deterministic_score = score_deterministic_answer(
                answer_type=reference.answer_type,
                predicted=raw_answer.get("answer") if raw_answer else None,
                expected=reference.answer,
            )
            deterministic_values.append(deterministic_score)

        assistant_score = None
        if reference.answer_type == AnswerType.FREE_TEXT:
            assistant_score = assistant_scores_by_question.get(reference.question_id)
            if assistant_score is None:
                assistant_score = 0.0
                answer_notes.append(MISSING_ASSISTANT_SCORE_NOTE)
            assistant_values.append(assistant_score)

        grounding_score = score_grounding(
            predicted_pairs=extract_retrieved_pairs(raw_answer),
            gold_pairs=flatten_page_references(reference.gold_retrieval),
        )
        grounding_values.append(grounding_score)

        telemetry_factor = telemetry_factor_for_answer(validation, question_id=reference.question_id)
        telemetry_values.append(telemetry_factor)

        ttft_ms = extract_ttft_ms(raw_answer)
        if ttft_ms is not None:
            ttft_ms_values.append(ttft_ms)
        ttft_factor = score_ttft(ttft_ms)
        ttft_values.append(ttft_factor)

        answer_scores.append(
            AnswerScoreBreakdown(
                question_id=reference.question_id,
                answer_type=reference.answer_type,
                deterministic_score=deterministic_score,
                assistant_score=assistant_score,
                grounding_score=grounding_score,
                telemetry_factor=telemetry_factor,
                ttft_ms=ttft_ms,
                ttft_factor=ttft_factor,
                notes=answer_notes,
            )
        )

    if not deterministic_values:
        notes.append(NO_DETERMINISTIC_NOTE)
    if not assistant_values:
        notes.append(NO_FREE_TEXT_NOTE)
    elif len(assistant_scores_by_question) < len(assistant_values):
        notes.append(PARTIAL_FREE_TEXT_SCORES_NOTE)

    deterministic_metric = _mean_or_zero(deterministic_values)
    assistant_metric = _mean_or_zero(assistant_values)
    grounding_metric = _mean_or_zero(grounding_values)
    telemetry_metric = _mean_or_zero(telemetry_values)
    ttft_metric = _mean_or_zero(ttft_values)
    mean_ttft_ms = _mean_or_none(ttft_ms_values)
    total_score = (
        (0.7 * deterministic_metric + 0.3 * assistant_metric)
        * grounding_metric
        * telemetry_metric
        * ttft_metric
    )

    unscored_question_ids = sorted(
        {
            answer.get("question_id")
            for answer in answer_dicts
            if isinstance(answer.get("question_id"), str)
            and answer.get("question_id") not in reference_question_types
        }
    )

    return ScoreReport(
        metrics=ScoreMetrics(
            deterministic=deterministic_metric,
            assistant=assistant_metric,
            grounding=grounding_metric,
            telemetry=telemetry_metric,
            ttft_ms=mean_ttft_ms,
            ttft_multiplier=ttft_metric,
            total_score=total_score,
            deterministic_questions=len(deterministic_values),
            free_text_questions=len(assistant_values),
            scored_questions=len(references),
        ),
        validation=validation,
        answer_scores=answer_scores,
        unscored_question_ids=unscored_question_ids,
        notes=notes,
    )


def score_deterministic_answer(*, answer_type: AnswerType, predicted: Any, expected: Any) -> float:
    """Score a deterministic answer according to the public rules."""

    if predicted is None or expected is None:
        return float(predicted is None and expected is None)

    if answer_type == AnswerType.NUMBER:
        if not _is_json_number(predicted) or not _is_json_number(expected):
            return 0.0
        expected_value = float(expected)
        predicted_value = float(predicted)
        tolerance = abs(expected_value) * 0.01
        if isclose(expected_value, 0.0):
            return float(isclose(predicted_value, expected_value, abs_tol=0.0))
        return float(abs(predicted_value - expected_value) <= tolerance)

    if answer_type == AnswerType.BOOLEAN:
        return float(isinstance(predicted, bool) and isinstance(expected, bool) and predicted == expected)

    if answer_type == AnswerType.NAME:
        if not isinstance(predicted, str) or not isinstance(expected, str):
            return 0.0
        return float(normalize_text(predicted) == normalize_text(expected))

    if answer_type == AnswerType.NAMES:
        if not isinstance(predicted, list) or not isinstance(expected, list):
            return 0.0
        if not all(isinstance(item, str) for item in predicted + expected):
            return 0.0
        predicted_set = {normalize_text(item) for item in predicted if normalize_text(item)}
        expected_set = {normalize_text(item) for item in expected if normalize_text(item)}
        if not predicted_set and not expected_set:
            return 1.0
        union = predicted_set | expected_set
        if not union:
            return 0.0
        return len(predicted_set & expected_set) / len(union)

    if answer_type == AnswerType.DATE:
        predicted_date = _parse_iso_date(predicted)
        expected_date = _parse_iso_date(expected)
        if predicted_date is None or expected_date is None:
            return 0.0
        return float(predicted_date == expected_date)

    return 0.0


def score_grounding(
    *,
    predicted_pairs: set[tuple[str, int]],
    gold_pairs: set[tuple[str, int]],
) -> float:
    """Score grounding via F-beta over flattened (doc_id, page_number) pairs."""

    if not predicted_pairs and not gold_pairs:
        return 1.0
    if not predicted_pairs or not gold_pairs:
        return 0.0

    overlap = len(predicted_pairs & gold_pairs)
    precision = overlap / len(predicted_pairs)
    recall = overlap / len(gold_pairs)
    if precision == 0 and recall == 0:
        return 0.0

    beta_sq = GROUNDING_BETA ** 2
    numerator = (1 + beta_sq) * precision * recall
    denominator = beta_sq * precision + recall
    if denominator == 0:
        return 0.0
    return numerator / denominator


def score_ttft(ttft_ms: float | None) -> float:
    """Score TTFT using the documented bins and a local >3s proxy."""

    if ttft_ms is None:
        return TTFT_PROXY_PENALTY_FLOOR
    if ttft_ms < 1_000:
        return 1.05
    if ttft_ms < 2_000:
        return 1.02
    if ttft_ms <= 3_000:
        return 1.0

    bounded_ms = min(ttft_ms, TTFT_PROXY_PENALTY_CAP_MS)
    span_ms = TTFT_PROXY_PENALTY_CAP_MS - TTFT_PROXY_PENALTY_START_MS
    decay = (bounded_ms - TTFT_PROXY_PENALTY_START_MS) / span_ms
    return TTFT_PROXY_PENALTY_START_FACTOR - decay * (
        TTFT_PROXY_PENALTY_START_FACTOR - TTFT_PROXY_PENALTY_FLOOR
    )


def telemetry_factor_for_answer(validation_report: SubmissionValidationReport, *, question_id: str) -> float:
    """Return the 0.9/1.0 telemetry factor from validator output."""

    for answer_report in validation_report.answers:
        if answer_report.question_id == question_id:
            return 0.9 if answer_report.telemetry_malformed else 1.0
    return 0.9


def _is_json_number(value: Any) -> bool:
    """Return whether json number is true.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mean_or_zero(values: list[float]) -> float:
    """Execute `_mean_or_zero`.

    Args:
        values: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_or_none(values: list[float]) -> float | None:
    """Execute `_mean_or_none`.

    Args:
        values: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if not values:
        return None
    return sum(values) / len(values)


def _parse_iso_date(value: Any) -> date | None:
    """Parse iso date.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    if parsed.isoformat() != value:
        return None
    return parsed
