"""Benchmark loading, execution, and comparison operations."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from ..contracts import (
    AnswerType,
    AssistantScoreRecord,
    ReferenceAnswerRecord,
    ScoreReport,
    UnanswerableRetrievalPolicy,
    load_assistant_score_records,
    load_question_records,
    load_submission_payload,
    question_types_from_questions,
)
from ..scorer import ScoreOptions, score_submission
from .models import (
    BenchmarkComparisonReport,
    BenchmarkConsistencyReport,
    BenchmarkDataset,
    BenchmarkDeltaReport,
    BenchmarkMetricRange,
    BenchmarkQuestionConsistency,
    BenchmarkQuestionMetadata,
    BenchmarkRunReport,
    BenchmarkSliceReport,
    MetricDelta,
)


def load_benchmark_dataset(path: str | Path) -> BenchmarkDataset:
    """Load a benchmark dataset from JSON."""

    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark dataset must be a JSON object: {dataset_path}")
    return BenchmarkDataset.model_validate(payload)


def load_benchmark_bundle(path: str | Path) -> BenchmarkDataset:
    """Backward-compatible alias for benchmark dataset loading."""

    return load_benchmark_dataset(path)


def load_benchmark_metadata_records(path: str | Path) -> list[BenchmarkQuestionMetadata]:
    """Load benchmark metadata rows from JSON or JSONL."""

    source_path = Path(path)
    if source_path.suffix.lower() == ".jsonl":
        with source_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        with source_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("metadata") if isinstance(payload, dict) else payload

    if not isinstance(rows, list):
        raise ValueError(f"Expected {source_path} to contain a JSON array or 'metadata' array.")
    return [BenchmarkQuestionMetadata.model_validate(row) for row in rows]


def load_benchmark_run_report(path: str | Path) -> BenchmarkRunReport:
    """Load a serialized benchmark run report."""

    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark run report must be a JSON object: {report_path}")
    return BenchmarkRunReport.model_validate(payload)


def load_benchmark_summary_report(path: str | Path) -> BenchmarkRunReport:
    """Backward-compatible alias for benchmark run reports."""

    return load_benchmark_run_report(path)


def build_benchmark_dataset_from_legacy(
    *,
    benchmark_name: str,
    references: list[ReferenceAnswerRecord],
    metadata: list[BenchmarkQuestionMetadata],
    assistant_scores: list[AssistantScoreRecord] | None = None,
    title: str | None = None,
    description: str | None = None,
    notes: list[str] | None = None,
) -> BenchmarkDataset:
    """Build a structured benchmark dataset from legacy reference + metadata inputs."""

    metadata_by_question = {record.question_id: record for record in metadata}
    reference_ids = {reference.question_id for reference in references}
    unknown_metadata_ids = sorted(set(metadata_by_question) - reference_ids)
    if unknown_metadata_ids:
        raise ValueError(
            "Benchmark metadata contains unknown question_id values: "
            + ", ".join(unknown_metadata_ids)
        )

    enriched_references = [
        reference.model_copy(update=_reference_overlay(reference, metadata_by_question.get(reference.question_id)))
        for reference in references
    ]

    return BenchmarkDataset(
        benchmark_id=benchmark_name,
        title=title or benchmark_name,
        description=description,
        references=enriched_references,
        assistant_scores=assistant_scores or [],
        notes=notes or [],
    )


def load_assistant_scores_or_none(path: str | Path | None) -> list[AssistantScoreRecord] | None:
    """Load assistant score rows only when a path is provided."""

    if path is None:
        return None
    return load_assistant_score_records(path)


def build_benchmark_report(
    submission_payload: Any,
    *,
    benchmark_name: str,
    references: list[ReferenceAnswerRecord],
    questions: list[Any],
    metadata: list[BenchmarkQuestionMetadata],
    assistant_scores: list[AssistantScoreRecord] | None = None,
    options: ScoreOptions | None = None,
) -> BenchmarkRunReport:
    """Build a benchmark report by overlaying metadata onto references."""

    resolved_options = options or ScoreOptions()
    if resolved_options.validation_question_types is None:
        resolved_options = ScoreOptions(
            unanswerable_retrieval_policy=resolved_options.unanswerable_retrieval_policy,
            documents_dir=resolved_options.documents_dir,
            validation_question_types=question_types_from_questions(questions),
        )

    return run_benchmark(
        submission_payload,
        dataset=build_benchmark_dataset_from_legacy(
            benchmark_name=benchmark_name,
            references=references,
            metadata=metadata,
            assistant_scores=assistant_scores,
        ),
        assistant_scores=assistant_scores,
        options=resolved_options,
    )


def run_benchmark(
    submission_payload: Any,
    *,
    dataset: BenchmarkDataset,
    assistant_scores: list[AssistantScoreRecord] | None = None,
    options: ScoreOptions | None = None,
) -> BenchmarkRunReport:
    """Score a submission against a benchmark dataset and derive slices."""

    resolved_options = options or ScoreOptions()
    resolved_assistant_scores = assistant_scores if assistant_scores is not None else dataset.assistant_scores
    overall = score_submission(
        submission_payload,
        references=dataset.references,
        assistant_scores=resolved_assistant_scores,
        options=resolved_options,
    )

    slices: list[BenchmarkSliceReport] = []
    for slice_id, label, slice_references in build_default_slices(dataset.references):
        question_ids = {reference.question_id for reference in slice_references}
        slices.append(
            BenchmarkSliceReport(
                slice_id=slice_id,
                label=label,
                question_ids=sorted(question_ids),
                question_count=len(slice_references),
                score=score_submission(
                    submission_payload,
                    references=slice_references,
                    assistant_scores=_filter_assistant_scores(
                        resolved_assistant_scores,
                        question_ids=question_ids,
                    ),
                    options=resolved_options,
                ),
            )
        )

    return BenchmarkRunReport(
        benchmark_id=dataset.benchmark_id,
        title=dataset.title,
        reference_count=len(dataset.references),
        assistant_score_count=len(resolved_assistant_scores),
        overall=overall,
        slices=slices,
        notes=dataset.notes,
    )


def benchmark_submission(
    *,
    submission_path: str | Path,
    benchmark_path: str | Path,
    questions_path: str | Path,
    documents_dir: str | Path,
    assistant_scores_path: str | Path | None = None,
    run_label: str | None = None,
    config_version: str | None = None,
    submission_uuid: str | None = None,
    unanswerable_retrieval_policy: UnanswerableRetrievalPolicy = UnanswerableRetrievalPolicy.ALLOW_EMPTY,
) -> BenchmarkRunReport:
    """Load inputs from disk and run the benchmark."""

    submission_payload = load_submission_payload(submission_path)
    dataset = load_benchmark_dataset(benchmark_path)
    assistant_scores = load_assistant_scores_or_none(assistant_scores_path)
    validation_question_types = question_types_from_questions(load_question_records(questions_path))
    report = run_benchmark(
        submission_payload,
        dataset=dataset,
        assistant_scores=assistant_scores,
        options=ScoreOptions(
            unanswerable_retrieval_policy=unanswerable_retrieval_policy,
            documents_dir=str(documents_dir),
            validation_question_types=validation_question_types,
        ),
    )
    report.notes.extend(
        [
            value
            for value in (
                f"run_label={run_label}" if run_label else None,
                f"config_version={config_version}" if config_version else None,
                f"submission_uuid={submission_uuid}" if submission_uuid else None,
            )
            if value is not None
        ]
    )
    return report


def compare_benchmark_runs(
    baseline: BenchmarkRunReport,
    candidate: BenchmarkRunReport,
) -> BenchmarkComparisonReport:
    """Compare two reports generated from the same benchmark dataset."""

    if baseline.benchmark_id != candidate.benchmark_id:
        raise ValueError(
            "Cannot compare benchmark runs with different benchmark_id values: "
            f"{baseline.benchmark_id!r} != {candidate.benchmark_id!r}"
        )

    baseline_slices = {slice_report.slice_id: slice_report for slice_report in baseline.slices}
    candidate_slices = {slice_report.slice_id: slice_report for slice_report in candidate.slices}
    common_slice_ids = sorted(set(baseline_slices) & set(candidate_slices))
    question_deltas = _question_total_deltas(baseline.overall, candidate.overall)

    return BenchmarkComparisonReport(
        baseline_label="baseline",
        candidate_label="candidate",
        benchmark_id=baseline.benchmark_id,
        overall=_build_delta_report(
            slice_id="overall",
            label="Overall",
            question_count=candidate.reference_count,
            baseline=baseline.overall,
            candidate=candidate.overall,
        ),
        slices=[
            _build_delta_report(
                slice_id=slice_id,
                label=candidate_slices[slice_id].label,
                question_count=candidate_slices[slice_id].question_count,
                baseline=baseline_slices[slice_id].score,
                candidate=candidate_slices[slice_id].score,
            )
            for slice_id in common_slice_ids
        ],
        missing_in_baseline=sorted(set(candidate_slices) - set(baseline_slices)),
        missing_in_candidate=sorted(set(baseline_slices) - set(candidate_slices)),
        regressions=[question_id for question_id, delta in question_deltas.items() if delta < 0],
    )


def compare_benchmark_reports(
    baseline: BenchmarkRunReport,
    candidate: BenchmarkRunReport,
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
) -> BenchmarkComparisonReport:
    """Backward-compatible comparison helper with explicit labels."""

    report = compare_benchmark_runs(baseline, candidate)
    report.baseline_label = baseline_label
    report.candidate_label = candidate_label
    return report


def analyze_run_consistency(
    reports: list[BenchmarkRunReport],
    *,
    instability_threshold: float = 0.05,
) -> BenchmarkConsistencyReport:
    """Summarize per-question variance across repeated benchmark runs."""

    if len(reports) < 2:
        raise ValueError("At least two benchmark reports are required for consistency analysis.")

    benchmark_id = reports[0].benchmark_id
    expected_question_ids = _report_question_ids(reports[0])
    for report in reports[1:]:
        if report.benchmark_id != benchmark_id:
            raise ValueError("All benchmark reports must come from the same benchmark_id.")
        if _report_question_ids(report) != expected_question_ids:
            raise ValueError("All benchmark reports must contain the same question set.")

    per_question_scores: dict[str, list[float]] = {question_id: [] for question_id in expected_question_ids}
    per_question_types: dict[str, AnswerType] = {}
    for report in reports:
        for answer_score in report.overall.answer_scores:
            per_question_scores[answer_score.question_id].append(_answer_total_score(answer_score))
            per_question_types[answer_score.question_id] = answer_score.answer_type

    question_consistency: list[BenchmarkQuestionConsistency] = []
    unstable_questions: list[BenchmarkQuestionConsistency] = []
    for question_id in sorted(per_question_scores):
        values = per_question_scores[question_id]
        std_total_score = pstdev(values)
        unstable = std_total_score > instability_threshold
        record = BenchmarkQuestionConsistency(
            question_id=question_id,
            answer_type=per_question_types[question_id],
            runs=len(values),
            mean_total_score=mean(values),
            std_total_score=std_total_score,
            min_total_score=min(values),
            max_total_score=max(values),
            unstable=unstable,
        )
        question_consistency.append(record)
        if unstable:
            unstable_questions.append(record)

    metric_ranges = {
        metric_name: BenchmarkMetricRange(
            minimum=min(values),
            maximum=max(values),
            spread=max(values) - min(values),
        )
        for metric_name, values in {
            "deterministic": [report.overall.metrics.deterministic for report in reports],
            "assistant": [report.overall.metrics.assistant for report in reports],
            "grounding": [report.overall.metrics.grounding for report in reports],
            "telemetry": [report.overall.metrics.telemetry for report in reports],
            "ttft_multiplier": [report.overall.metrics.ttft_multiplier for report in reports],
            "total_score": [report.overall.metrics.total_score for report in reports],
        }.items()
    }

    return BenchmarkConsistencyReport(
        benchmark_id=benchmark_id,
        run_count=len(reports),
        instability_threshold=instability_threshold,
        unstable_questions=unstable_questions,
        question_consistency=question_consistency,
        metric_ranges=metric_ranges,
    )


def analyze_benchmark_consistency(
    reports: list[BenchmarkRunReport],
    *,
    instability_threshold: float = 0.05,
) -> BenchmarkConsistencyReport:
    """Backward-compatible alias for repeated-run consistency analysis."""

    return analyze_run_consistency(reports, instability_threshold=instability_threshold)


def build_default_slices(
    references: list[ReferenceAnswerRecord],
) -> list[tuple[str, str, list[ReferenceAnswerRecord]]]:
    """Build deterministic default slices from reference metadata."""

    grouped: dict[str, tuple[str, list[ReferenceAnswerRecord]]] = {}

    def add(slice_id: str, label: str, reference: ReferenceAnswerRecord) -> None:
        """Execute `add`.

        Args:
            slice_id: Input parameter.
            label: Input parameter.
            reference: Input parameter.

        Returns:
            None: This function does not return a value.
        """
        bucket = grouped.setdefault(slice_id, (label, []))
        bucket[1].append(reference)

    for reference in references:
        add(f"answer_type:{reference.answer_type.value}", f"Answer Type: {reference.answer_type.value}", reference)

        if reference.source_type is not None:
            add(f"source_type:{reference.source_type.value}", f"Source Type: {reference.source_type.value}", reference)

        if reference.difficulty is not None:
            add(f"difficulty:{reference.difficulty.value}", f"Difficulty: {reference.difficulty.value}", reference)

        doc_count = len({page.doc_id for page in reference.gold_retrieval})
        add(f"doc_count:{doc_count}", f"Doc Count: {doc_count}", reference)
        add(f"doc_count:{_doc_count_bucket(doc_count)}", f"Doc Count Bucket: {_doc_count_bucket(doc_count)}", reference)

        for tag in reference.tags:
            add(f"tag:{tag}", f"Tag: {tag}", reference)

    return [(slice_id, label, rows) for slice_id, (label, rows) in sorted(grouped.items())]


def _reference_overlay(
    reference: ReferenceAnswerRecord,
    metadata: BenchmarkQuestionMetadata | None,
) -> dict[str, Any]:
    """Execute `_reference_overlay`.

    Args:
        reference: Input parameter.
        metadata: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if metadata is None:
        return {}

    derived_doc_count = len({page.doc_id for page in reference.gold_retrieval})
    if metadata.doc_count is not None and metadata.doc_count != derived_doc_count:
        raise ValueError(
            f"Metadata doc_count mismatch for {reference.question_id}: "
            f"{metadata.doc_count} != {derived_doc_count}"
        )

    return {
        "source_type": metadata.source_type,
        "difficulty": metadata.difficulty,
        "tags": metadata.tags,
        "notes": metadata.notes or reference.notes,
    }


def _filter_assistant_scores(
    assistant_scores: list[AssistantScoreRecord] | None,
    *,
    question_ids: set[str],
) -> list[AssistantScoreRecord] | None:
    """Execute `_filter_assistant_scores`.

    Args:
        assistant_scores: Input parameter.
        question_ids: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if assistant_scores is None:
        return None
    return [record for record in assistant_scores if record.question_id in question_ids]


def _build_delta_report(
    *,
    slice_id: str,
    label: str,
    question_count: int,
    baseline: ScoreReport,
    candidate: ScoreReport,
) -> BenchmarkDeltaReport:
    """Build delta report.

    Args:
        slice_id: Input parameter.
        label: Input parameter.
        question_count: Input parameter.
        baseline: Input parameter.
        candidate: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    return BenchmarkDeltaReport(
        slice_id=slice_id,
        label=label,
        question_count=question_count,
        baseline_total_score=baseline.metrics.total_score,
        candidate_total_score=candidate.metrics.total_score,
        delta=MetricDelta(
            deterministic=candidate.metrics.deterministic - baseline.metrics.deterministic,
            assistant=candidate.metrics.assistant - baseline.metrics.assistant,
            grounding=candidate.metrics.grounding - baseline.metrics.grounding,
            telemetry=candidate.metrics.telemetry - baseline.metrics.telemetry,
            ttft_multiplier=candidate.metrics.ttft_multiplier - baseline.metrics.ttft_multiplier,
            total_score=candidate.metrics.total_score - baseline.metrics.total_score,
        ),
    )


def _question_total_deltas(baseline: ScoreReport, candidate: ScoreReport) -> dict[str, float]:
    """Execute `_question_total_deltas`.

    Args:
        baseline: Input parameter.
        candidate: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    baseline_scores = {row.question_id: _answer_total_score(row) for row in baseline.answer_scores}
    candidate_scores = {row.question_id: _answer_total_score(row) for row in candidate.answer_scores}
    return {
        question_id: candidate_scores[question_id] - baseline_scores[question_id]
        for question_id in sorted(baseline_scores)
        if question_id in candidate_scores
    }


def _report_question_ids(report: BenchmarkRunReport) -> list[str]:
    """Execute `_report_question_ids`.

    Args:
        report: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    return sorted(answer_score.question_id for answer_score in report.overall.answer_scores)


def _answer_total_score(answer_score: Any) -> float:
    """Execute `_answer_total_score`.

    Args:
        answer_score: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    base_score = (
        answer_score.deterministic_score
        if answer_score.deterministic_score is not None
        else answer_score.assistant_score or 0.0
    )
    return base_score * answer_score.grounding_score * answer_score.telemetry_factor * answer_score.ttft_factor


def _doc_count_bucket(doc_count: int) -> str:
    """Execute `_doc_count_bucket`.

    Args:
        doc_count: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if doc_count <= 1:
        return "single"
    return "multi"
