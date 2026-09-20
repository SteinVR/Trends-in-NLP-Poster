"""Validation and finalization helpers for gold shard bundles."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.schemas import QuestionRecord

from .contracts import AnswerType, AssistantScoreRecord, ReferenceAnswerRecord
from .goldset_core import (
    ALLOWED_TAGS,
    DATE_PATTERN,
    UNANSWERABLE_FREE_TEXT_ANSWER,
    GoldShardBundle,
    StratumKey,
)


def validate_gold_shard_bundle(
    bundle: GoldShardBundle,
    *,
    questions_by_id: dict[str, QuestionRecord],
    require_assistant_scores: bool,
    require_reviewed_status: bool,
) -> None:
    """Validate a draft or reviewed shard bundle against the assigned questions."""

    if require_reviewed_status and bundle.status != "reviewed":
        raise ValueError(f"Expected bundle.status='reviewed', got {bundle.status!r}")

    assigned_ids = set(bundle.question_ids)
    if not assigned_ids:
        raise ValueError("GoldShardBundle.question_ids must not be empty.")

    unknown_ids = sorted(question_id for question_id in assigned_ids if question_id not in questions_by_id)
    if unknown_ids:
        raise ValueError("Bundle references unknown shard question_ids: " + ", ".join(unknown_ids))

    seen_reference_ids: set[str] = set()
    for reference in bundle.references:
        if reference.question_id not in assigned_ids:
            raise ValueError(
                f"Reference question_id={reference.question_id} is outside shard {bundle.shard_id} scope."
            )
        if reference.question_id in seen_reference_ids:
            raise ValueError(f"Duplicate reference question_id={reference.question_id}")
        seen_reference_ids.add(reference.question_id)
        validate_reference_answer_record(reference, expected_question=questions_by_id[reference.question_id])

    missing_reference_ids = sorted(assigned_ids - seen_reference_ids)
    if missing_reference_ids:
        raise ValueError(
            "Bundle does not cover every assigned question_id: " + ", ".join(missing_reference_ids)
        )

    free_text_ids = {
        question_id
        for question_id in assigned_ids
        if questions_by_id[question_id].answer_type == AnswerType.FREE_TEXT.value
    }
    score_ids: set[str] = set()
    for score in bundle.assistant_scores:
        if score.question_id not in free_text_ids:
            raise ValueError(
                f"assistant_score question_id={score.question_id} is not a free_text question "
                f"in shard {bundle.shard_id}."
            )
        if score.question_id in score_ids:
            raise ValueError(f"Duplicate assistant_score question_id={score.question_id}")
        score_ids.add(score.question_id)

    if require_assistant_scores and score_ids != free_text_ids:
        missing_score_ids = sorted(free_text_ids - score_ids)
        extra_score_ids = sorted(score_ids - free_text_ids)
        parts = []
        if missing_score_ids:
            parts.append("missing=" + ",".join(missing_score_ids))
        if extra_score_ids:
            parts.append("extra=" + ",".join(extra_score_ids))
        raise ValueError("assistant_scores coverage mismatch for free_text questions: " + "; ".join(parts))


def validate_reference_answer_record(reference: ReferenceAnswerRecord, *, expected_question: QuestionRecord) -> None:
    """Validate a reference row against the originating question and local gold rules."""

    expected_type = AnswerType(str(expected_question.answer_type))
    if reference.answer_type != expected_type:
        raise ValueError(
            f"answer_type mismatch for {reference.question_id}: "
            f"{reference.answer_type.value!r} != {expected_type.value!r}"
        )
    if reference.question is None or reference.question.strip() != expected_question.question.strip():
        raise ValueError(f"question text mismatch for {reference.question_id}")
    if not set(reference.tags) <= ALLOWED_TAGS:
        unknown_tags = sorted(set(reference.tags) - ALLOWED_TAGS)
        raise ValueError(f"Unsupported tags for {reference.question_id}: {', '.join(unknown_tags)}")

    if not reference.gold_retrieval:
        _validate_unanswerable_reference(reference)
    else:
        for page_reference in reference.gold_retrieval:
            for page_number in page_reference.page_numbers:
                if page_number <= 0:
                    raise ValueError(f"Non-positive page number in gold_retrieval for {reference.question_id}")
        _validate_answer_shape(reference)


def write_json(path: str | Path, payload: Any) -> None:
    """Serialize a payload to JSON with consistent formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _validate_answer_shape(reference: ReferenceAnswerRecord) -> None:
    """Validate answer shape.

    Args:
        reference: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    answer = reference.answer

    if reference.answer_type == AnswerType.NUMBER:
        if answer is None or isinstance(answer, bool) or not isinstance(answer, (int, float)):
            raise ValueError(f"number answer must be a JSON number for {reference.question_id}")
        return

    if reference.answer_type == AnswerType.BOOLEAN:
        if not isinstance(answer, bool):
            raise ValueError(f"boolean answer must be a JSON boolean for {reference.question_id}")
        return

    if reference.answer_type == AnswerType.NAME:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"name answer must be a non-empty string for {reference.question_id}")
        return

    if reference.answer_type == AnswerType.NAMES:
        if not isinstance(answer, list) or not answer or not all(
            isinstance(item, str) and item.strip() for item in answer
        ):
            raise ValueError(f"names answer must be a non-empty list[str] for {reference.question_id}")
        return

    if reference.answer_type == AnswerType.DATE:
        if not isinstance(answer, str) or not DATE_PATTERN.fullmatch(answer):
            raise ValueError(f"date answer must use YYYY-MM-DD for {reference.question_id}")
        return

    if reference.answer_type == AnswerType.FREE_TEXT:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"free_text answer must be a non-empty string for {reference.question_id}")
        if len(answer) > 280:
            raise ValueError(f"free_text answer must be <= 280 chars for {reference.question_id}")
        return

    raise ValueError(f"Unsupported answer_type={reference.answer_type.value!r}")


def _validate_unanswerable_reference(reference: ReferenceAnswerRecord) -> None:
    """Validate unanswerable reference.

    Args:
        reference: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    if reference.answer_type == AnswerType.FREE_TEXT:
        if reference.answer != UNANSWERABLE_FREE_TEXT_ANSWER:
            raise ValueError(
                "free_text unanswerable answers must use the canonical no-information sentence "
                f"for {reference.question_id}"
            )
        return
    if reference.answer is not None:
        raise ValueError(
            f"Deterministic unanswerable answers must be null when gold_retrieval=[] for {reference.question_id}"
        )


def _validate_unique_questions(questions: list[QuestionRecord]) -> None:
    """Validate unique questions.

    Args:
        questions: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    ids = [question.id for question in questions]
    texts = [question.question.strip() for question in questions]
    duplicate_ids = sorted(question_id for question_id, count in Counter(ids).items() if count > 1)
    duplicate_texts = sorted(text for text, count in Counter(texts).items() if count > 1)

    if duplicate_ids:
        raise ValueError("Duplicate question ids detected: " + ", ".join(duplicate_ids))
    if duplicate_texts:
        raise ValueError("Duplicate question texts detected: " + "; ".join(duplicate_texts))


def _distribute_counts_by_capacity(
    *,
    capacities: dict[StratumKey, int],
    total_count: int,
    minimum_per_group: int,
) -> dict[StratumKey, int]:
    """Execute `_distribute_counts_by_capacity`.

    Args:
        capacities: Input parameter.
        total_count: Input parameter.
        minimum_per_group: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    selected_counts = {group_key: 0 for group_key in capacities}
    if minimum_per_group:
        groups = [group_key for group_key, capacity in capacities.items() if capacity > 0]
        if total_count < len(groups) * minimum_per_group:
            raise ValueError("Not enough sample slots to satisfy minimum_per_group across all strata.")
        for group_key in groups:
            selected_counts[group_key] = minimum_per_group

    remaining = total_count - sum(selected_counts.values())
    while remaining > 0:
        remaining_before_distribution = remaining
        residual_capacities = {
            group_key: capacities[group_key] - selected_counts[group_key]
            for group_key in capacities
            if capacities[group_key] > selected_counts[group_key]
        }
        if not residual_capacities:
            raise ValueError("Ran out of residual capacity while distributing stratified counts.")

        total_residual_capacity = sum(residual_capacities.values())
        base_increments = {
            group_key: int(remaining_before_distribution * residual_capacity / total_residual_capacity)
            for group_key, residual_capacity in residual_capacities.items()
        }
        distributed = sum(base_increments.values())

        if distributed == 0:
            top_group_key = max(
                residual_capacities,
                key=lambda group_key: (residual_capacities[group_key], group_key),
            )
            selected_counts[top_group_key] += 1
            remaining -= 1
            continue

        for group_key, increment in base_increments.items():
            selected_counts[group_key] += increment

        remaining = remaining_before_distribution - distributed
        if remaining <= 0:
            break

        fractional_order = sorted(
            residual_capacities,
            key=lambda group_key: (
                remaining_before_distribution * residual_capacities[group_key] / total_residual_capacity
                - base_increments[group_key],
                residual_capacities[group_key],
                group_key,
            ),
            reverse=True,
        )
        for group_key in fractional_order:
            if remaining == 0:
                break
            if selected_counts[group_key] >= capacities[group_key]:
                continue
            selected_counts[group_key] += 1
            remaining -= 1

    return selected_counts


def _spread_select_questions(questions: list[QuestionRecord], sample_count: int) -> list[QuestionRecord]:
    """Execute `_spread_select_questions`.

    Args:
        questions: Input parameter.
        sample_count: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if sample_count == 0:
        return []
    if sample_count > len(questions):
        raise ValueError("Cannot select more questions than exist in a stratum.")
    if sample_count == len(questions):
        return list(questions)

    total_questions = len(questions)
    selected_indices = [
        ((2 * index + 1) * total_questions) // (2 * sample_count)
        for index in range(sample_count)
    ]
    return [questions[index] for index in selected_indices]


def _collect_reviewed_references_and_scores(
    *,
    reviewed_bundles: list[GoldShardBundle],
    questions_by_id: dict[str, QuestionRecord],
) -> tuple[dict[str, ReferenceAnswerRecord], dict[str, AssistantScoreRecord]]:
    """Execute `_collect_reviewed_references_and_scores`.

    Args:
        reviewed_bundles: Input parameter.
        questions_by_id: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if not reviewed_bundles:
        raise ValueError("At least one reviewed bundle is required.")

    combined_references: dict[str, ReferenceAnswerRecord] = {}
    combined_scores: dict[str, AssistantScoreRecord] = {}
    for bundle in reviewed_bundles:
        validate_gold_shard_bundle(
            bundle,
            questions_by_id=questions_by_id,
            require_assistant_scores=True,
            require_reviewed_status=True,
        )
        for reference in bundle.references:
            if reference.question_id in combined_references:
                raise ValueError(f"Duplicate reviewed reference for question_id={reference.question_id}")
            combined_references[reference.question_id] = reference
        for score in bundle.assistant_scores:
            if score.question_id in combined_scores:
                raise ValueError(f"Duplicate assistant_score for question_id={score.question_id}")
            combined_scores[score.question_id] = score
    return combined_references, combined_scores


def _free_text_question_ids(questions: list[QuestionRecord]) -> list[str]:
    """Execute `_free_text_question_ids`.

    Args:
        questions: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    return [question.id for question in questions if question.answer_type == AnswerType.FREE_TEXT.value]
