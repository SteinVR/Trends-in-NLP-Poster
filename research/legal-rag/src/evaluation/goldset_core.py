"""Helpers for gold-set sharding, review validation, and benchmark assembly."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.common.schemas import QuestionRecord

from .contracts import AnswerType, AssistantScoreRecord, ReferenceAnswerRecord, load_question_records
from .dev_benchmark import BenchmarkDataset

UNANSWERABLE_FREE_TEXT_ANSWER = "There is no information on this question in the provided documents."
ALLOWED_TAGS = {
    "comparative",
    "deterministic",
    "entity_overlap",
    "free_text",
    "multi_document",
    "multi_page",
    "negative",
    "single_document",
    "single_page",
    "table_lookup",
    "title_page",
}
CASE_ID_PATTERN = re.compile(r"\b[A-Z]{2,4}\s?\d{1,3}/\d{4}\b")
COMPARATIVE_HINTS = (
    " both ",
    " same ",
    " common ",
    " compare ",
    " across ",
    " between ",
)
NEGATIVE_HINTS = (
    " is there any ",
    " are there any ",
    " no information ",
    " not present ",
    " absent ",
)
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
StratumKey = tuple[str, str]


class GoldQuestionShard(BaseModel):
    """One deterministic question shard for manual annotation."""

    model_config = ConfigDict(extra="forbid")

    shard_id: str
    questions: list[QuestionRecord] = Field(default_factory=list)


class GoldShardManifestEntry(BaseModel):
    """Summary stats for one shard."""

    model_config = ConfigDict(extra="forbid")

    shard_id: str
    question_count: int
    question_ids: list[str] = Field(default_factory=list)
    answer_type_counts: dict[str, int] = Field(default_factory=dict)
    structural_bucket_counts: dict[str, int] = Field(default_factory=dict)


class GoldShardManifest(BaseModel):
    """Top-level manifest for a sharded gold workflow."""

    model_config = ConfigDict(extra="forbid")

    question_count: int
    shard_count: int
    entries: list[GoldShardManifestEntry] = Field(default_factory=list)


class GoldQuestionSetSummary(BaseModel):
    """Summary stats for one question set or sampled subset."""

    model_config = ConfigDict(extra="forbid")

    question_count: int
    question_ids: list[str] = Field(default_factory=list)
    answer_type_counts: dict[str, int] = Field(default_factory=dict)
    structural_bucket_counts: dict[str, int] = Field(default_factory=dict)


class GoldShardBundle(BaseModel):
    """Draft or reviewed shard output produced by annotators/reviewers."""

    model_config = ConfigDict(extra="forbid")

    shard_id: str
    status: Literal["draft", "reviewed"]
    question_ids: list[str] = Field(default_factory=list)
    references: list[ReferenceAnswerRecord] = Field(default_factory=list)
    assistant_scores: list[AssistantScoreRecord] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("question_ids")
    @classmethod
    def ensure_unique_question_ids(cls, value: list[str]) -> list[str]:
        """Execute `ensure_unique_question_ids`.

        Args:
            value: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        normalized = [item.strip() for item in value if item and item.strip()]
        if len(normalized) != len(set(normalized)):
            raise ValueError("GoldShardBundle.question_ids must be unique.")
        return normalized


def load_questions(path: str | Path) -> list[QuestionRecord]:
    """Load question records from disk."""

    return load_question_records(path)


def _validate_unique_questions(questions: list[QuestionRecord]) -> None:
    """Validate unique question ids/text via the validation module."""

    from .goldset_validation import _validate_unique_questions as validator

    validator(questions)


def _distribute_counts_by_capacity(
    *,
    capacities: dict[StratumKey, int],
    total_count: int,
    minimum_per_group: int,
) -> dict[StratumKey, int]:
    """Distribute per-stratum sample counts via the validation module."""

    from .goldset_validation import _distribute_counts_by_capacity as distributor

    return distributor(
        capacities=capacities,
        total_count=total_count,
        minimum_per_group=minimum_per_group,
    )


def _spread_select_questions(questions: list[QuestionRecord], sample_count: int) -> list[QuestionRecord]:
    """Select evenly spread questions from a stratum via the validation module."""

    from .goldset_validation import _spread_select_questions as spread_selector

    return spread_selector(questions, sample_count)


def _collect_reviewed_references_and_scores(
    *,
    reviewed_bundles: list[GoldShardBundle],
    questions_by_id: dict[str, QuestionRecord],
) -> tuple[dict[str, ReferenceAnswerRecord], dict[str, AssistantScoreRecord]]:
    """Collect reviewed references/scores via the validation module."""

    from .goldset_validation import _collect_reviewed_references_and_scores as collector

    return collector(reviewed_bundles=reviewed_bundles, questions_by_id=questions_by_id)


def _free_text_question_ids(questions: list[QuestionRecord]) -> list[str]:
    """Return free-text question ids via the validation module."""

    from .goldset_validation import _free_text_question_ids as resolver

    return resolver(questions)


def validate_reference_answer_record(reference: ReferenceAnswerRecord, *, expected_question: QuestionRecord) -> None:
    """Validate one reference answer via the validation module."""

    from .goldset_validation import validate_reference_answer_record as validator

    validator(reference, expected_question=expected_question)


def structural_bucket(question: QuestionRecord) -> str:
    """Infer a coarse structural bucket for shard balancing."""

    text = f" {question.question.strip().lower()} "
    case_matches = CASE_ID_PATTERN.findall(question.question)
    if len(case_matches) >= 2 or any(hint in text for hint in COMPARATIVE_HINTS):
        return "comparative_multi_document"
    if any(hint in text for hint in NEGATIVE_HINTS):
        return "negative_unanswerable"
    if question.answer_type == AnswerType.FREE_TEXT.value:
        return "free_text"
    return "standard"


def split_questions_stratified(
    questions: list[QuestionRecord],
    *,
    shard_count: int,
) -> list[GoldQuestionShard]:
    """Split questions into deterministic stratified shards."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")

    _validate_unique_questions(questions)
    if len(questions) % shard_count != 0:
        raise ValueError(
            f"Question count {len(questions)} is not evenly divisible by shard_count={shard_count}."
        )

    grouped: dict[StratumKey, list[QuestionRecord]] = defaultdict(list)
    for question in questions:
        grouped[(question.answer_type, structural_bucket(question))].append(question)

    shard_states = [
        {
            "shard_id": f"part_{index + 1:02d}",
            "questions": [],
            "answer_type_counts": Counter(),
            "bucket_counts": Counter(),
        }
        for index in range(shard_count)
    ]

    for answer_type, bucket in sorted(grouped):
        for question in sorted(grouped[(answer_type, bucket)], key=lambda item: item.id):
            target_state = min(
                shard_states,
                key=lambda state: (
                    len(state["questions"]),
                    state["answer_type_counts"][answer_type],
                    state["bucket_counts"][bucket],
                    state["shard_id"],
                ),
            )
            target_state["questions"].append(question)
            target_state["answer_type_counts"][answer_type] += 1
            target_state["bucket_counts"][bucket] += 1

    shards = [
        GoldQuestionShard(
            shard_id=str(state["shard_id"]),
            questions=list(state["questions"]),
        )
        for state in shard_states
    ]
    _validate_unique_questions([question for shard in shards for question in shard.questions])

    expected_size = len(questions) // shard_count
    sizes = {len(shard.questions) for shard in shards}
    if sizes != {expected_size}:
        raise ValueError(f"Expected evenly sized shards of {expected_size}, got sizes={sorted(sizes)}")

    return shards


def sample_questions_stratified(
    questions: list[QuestionRecord],
    *,
    sample_count: int,
) -> list[QuestionRecord]:
    """Select a deterministic stratified subset from a larger question pool."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")

    _validate_unique_questions(questions)
    if sample_count > len(questions):
        raise ValueError(
            f"sample_count={sample_count} cannot exceed total question count {len(questions)}."
        )
    if sample_count == len(questions):
        return list(questions)

    grouped: dict[StratumKey, list[QuestionRecord]] = defaultdict(list)
    for question in questions:
        grouped[(question.answer_type, structural_bucket(question))].append(question)

    group_keys = sorted(grouped)
    minimum_per_group = 1 if sample_count >= len(group_keys) else 0
    selected_counts = _distribute_counts_by_capacity(
        capacities={group_key: len(grouped[group_key]) for group_key in group_keys},
        total_count=sample_count,
        minimum_per_group=minimum_per_group,
    )

    selected_ids: set[str] = set()
    for group_key in group_keys:
        selected_questions = _spread_select_questions(
            sorted(grouped[group_key], key=lambda item: item.id),
            selected_counts[group_key],
        )
        selected_ids.update(question.id for question in selected_questions)

    sampled_questions = [question for question in questions if question.id in selected_ids]
    if len(sampled_questions) != sample_count:
        raise ValueError(
            f"Expected sampled subset of {sample_count} questions, got {len(sampled_questions)}."
        )
    _validate_unique_questions(sampled_questions)
    return sampled_questions


def build_shard_manifest(shards: list[GoldQuestionShard]) -> GoldShardManifest:
    """Build a summary manifest from generated shards."""

    entries = []
    for shard in shards:
        answer_type_counts = Counter(question.answer_type for question in shard.questions)
        bucket_counts = Counter(structural_bucket(question) for question in shard.questions)
        entries.append(
            GoldShardManifestEntry(
                shard_id=shard.shard_id,
                question_count=len(shard.questions),
                question_ids=[question.id for question in shard.questions],
                answer_type_counts=dict(sorted(answer_type_counts.items())),
                structural_bucket_counts=dict(sorted(bucket_counts.items())),
            )
        )

    return GoldShardManifest(
        question_count=sum(entry.question_count for entry in entries),
        shard_count=len(shards),
        entries=entries,
    )


def build_question_set_summary(questions: list[QuestionRecord]) -> GoldQuestionSetSummary:
    """Build a summary for one question set."""

    _validate_unique_questions(questions)
    answer_type_counts = Counter(question.answer_type for question in questions)
    bucket_counts = Counter(structural_bucket(question) for question in questions)
    return GoldQuestionSetSummary(
        question_count=len(questions),
        question_ids=[question.id for question in questions],
        answer_type_counts=dict(sorted(answer_type_counts.items())),
        structural_bucket_counts=dict(sorted(bucket_counts.items())),
    )


def build_shard_scaffold(shard: GoldQuestionShard, *, status: Literal["draft", "reviewed"]) -> GoldShardBundle:
    """Create an empty shard bundle scaffold."""

    return GoldShardBundle(
        shard_id=shard.shard_id,
        status=status,
        question_ids=[question.id for question in shard.questions],
    )


def load_gold_shard_bundle(path: str | Path) -> GoldShardBundle:
    """Load one shard bundle from disk."""

    bundle_path = Path(path)
    with bundle_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return GoldShardBundle.model_validate(payload)


def merge_reviewed_shards(
    *,
    reviewed_bundles: list[GoldShardBundle],
    questions: list[QuestionRecord],
    benchmark_id: str,
    title: str | None = None,
    notes: list[str] | None = None,
    include_assistant_scores: bool = False,
) -> BenchmarkDataset:
    """Merge reviewed shard bundles into one benchmark dataset."""

    questions_by_id = {question.id: question for question in questions}
    ordered_ids = [question.id for question in questions]
    combined_references, combined_scores = _collect_reviewed_references_and_scores(
        reviewed_bundles=reviewed_bundles,
        questions_by_id=questions_by_id,
    )

    missing_reference_ids = [question_id for question_id in ordered_ids if question_id not in combined_references]
    if missing_reference_ids:
        raise ValueError(
            "Reviewed bundles do not cover every question. Missing question_ids: "
            + ", ".join(missing_reference_ids)
        )

    free_text_ids = _free_text_question_ids(questions)
    missing_score_ids = [question_id for question_id in free_text_ids if question_id not in combined_scores]
    if missing_score_ids:
        raise ValueError(
            "Reviewed bundles are missing assistant_scores for free_text question_ids: "
            + ", ".join(missing_score_ids)
        )

    references = [combined_references[question_id] for question_id in ordered_ids]
    assistant_scores = [combined_scores[question_id] for question_id in free_text_ids]

    return BenchmarkDataset(
        benchmark_id=benchmark_id,
        title=title or benchmark_id,
        references=references,
        assistant_scores=assistant_scores if include_assistant_scores else [],
        notes=notes or [],
    )


def collect_reviewed_assistant_scores(
    *,
    reviewed_bundles: list[GoldShardBundle],
    questions: list[QuestionRecord],
) -> list[AssistantScoreRecord]:
    """Collect ordered free-text assistant scores from reviewed bundles."""

    questions_by_id = {question.id: question for question in questions}
    free_text_ids = _free_text_question_ids(questions)
    _, combined_scores = _collect_reviewed_references_and_scores(
        reviewed_bundles=reviewed_bundles,
        questions_by_id=questions_by_id,
    )

    missing_score_ids = [question_id for question_id in free_text_ids if question_id not in combined_scores]
    if missing_score_ids:
        raise ValueError(
            "Reviewed bundles are missing assistant_scores for free_text question_ids: "
            + ", ".join(missing_score_ids)
        )

    return [combined_scores[question_id] for question_id in free_text_ids]


def combine_question_sets(question_sets: list[list[QuestionRecord]]) -> list[QuestionRecord]:
    """Concatenate question sets while preserving input order and uniqueness."""

    combined_questions = [question for question_set in question_sets for question in question_set]
    _validate_unique_questions(combined_questions)
    return combined_questions


def combine_benchmark_datasets(
    *,
    questions: list[QuestionRecord],
    benchmarks: list[BenchmarkDataset],
    benchmark_id: str,
    title: str | None = None,
    notes: list[str] | None = None,
) -> BenchmarkDataset:
    """Combine multiple benchmark datasets into one reference-only benchmark."""

    if not benchmarks:
        raise ValueError("At least one benchmark dataset is required.")

    questions_by_id = {question.id: question for question in questions}
    ordered_ids = [question.id for question in questions]
    combined_references: dict[str, ReferenceAnswerRecord] = {}

    for dataset in benchmarks:
        for reference in dataset.references:
            if reference.question_id not in questions_by_id:
                raise ValueError(
                    f"Benchmark reference question_id={reference.question_id} is not present in combined questions."
                )
            if reference.question_id in combined_references:
                raise ValueError(f"Duplicate benchmark reference for question_id={reference.question_id}")
            validate_reference_answer_record(reference, expected_question=questions_by_id[reference.question_id])
            combined_references[reference.question_id] = reference

    missing_reference_ids = [question_id for question_id in ordered_ids if question_id not in combined_references]
    if missing_reference_ids:
        raise ValueError(
            "Combined benchmark references do not cover every question. Missing question_ids: "
            + ", ".join(missing_reference_ids)
        )

    return BenchmarkDataset(
        benchmark_id=benchmark_id,
        title=title or benchmark_id,
        references=[combined_references[question_id] for question_id in ordered_ids],
        assistant_scores=[],
        notes=notes or [],
    )


def combine_assistant_score_sets(
    *,
    questions: list[QuestionRecord],
    assistant_score_sets: list[list[AssistantScoreRecord]],
) -> list[AssistantScoreRecord]:
    """Combine assistant score collections with full free-text coverage checks."""

    if not assistant_score_sets:
        raise ValueError("At least one assistant score collection is required.")

    questions_by_id = {question.id: question for question in questions}
    free_text_ids = _free_text_question_ids(questions)
    combined_scores: dict[str, AssistantScoreRecord] = {}

    for score_set in assistant_score_sets:
        for score in score_set:
            question = questions_by_id.get(score.question_id)
            if question is None:
                raise ValueError(
                    f"assistant_score question_id={score.question_id} is not present in combined questions."
                )
            if question.answer_type != AnswerType.FREE_TEXT.value:
                raise ValueError(
                    f"assistant_score question_id={score.question_id} is not a free_text question."
                )
            if score.question_id in combined_scores:
                raise ValueError(f"Duplicate assistant_score for question_id={score.question_id}")
            combined_scores[score.question_id] = score

    missing_score_ids = [question_id for question_id in free_text_ids if question_id not in combined_scores]
    if missing_score_ids:
        raise ValueError(
            "Combined assistant_scores do not cover every free_text question. Missing question_ids: "
            + ", ".join(missing_score_ids)
        )

    return [combined_scores[question_id] for question_id in free_text_ids]


def collect_referenced_doc_ids(references: list[ReferenceAnswerRecord]) -> list[str]:
    """Collect the unique doc_ids referenced by a set of gold answers."""

    doc_ids = {page_reference.doc_id for reference in references for page_reference in reference.gold_retrieval}
    return sorted(doc_ids)
