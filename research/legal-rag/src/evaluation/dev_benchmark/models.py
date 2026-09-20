"""Local dev benchmark dataset, slicing, and regression helpers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts import (
    AnswerType,
    AssistantScoreRecord,
    BenchmarkDifficulty,
    BenchmarkSourceType,
    ReferenceAnswerRecord,
    ScoreMetrics,
    ScoreReport,
)


class BenchmarkQuestionMetadata(BaseModel):
    """Question-level metadata used for benchmark slicing."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    source_type: BenchmarkSourceType | None = None
    difficulty: BenchmarkDifficulty | None = None
    doc_count: int | None = Field(default=None, ge=1)
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

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        """Normalize tags.

        Args:
            value: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        normalized = [item.strip() for item in value if item and item.strip()]
        return list(dict.fromkeys(normalized))


BenchmarkMetadataRecord = BenchmarkQuestionMetadata
BenchmarkQuestionDifficulty = BenchmarkDifficulty


class BenchmarkDataset(BaseModel):
    """Score-compatible gold dataset plus optional manual assistant scores."""

    model_config = ConfigDict(extra="forbid")

    benchmark_id: str
    title: str | None = None
    description: str | None = None
    references: list[ReferenceAnswerRecord] = Field(default_factory=list)
    assistant_scores: list[AssistantScoreRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("benchmark_id")
    @classmethod
    def ensure_benchmark_id(cls, value: str) -> str:
        """Execute `ensure_benchmark_id`.

        Args:
            value: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("benchmark_id must be a non-empty string.")
        return normalized


class BenchmarkSliceReport(BaseModel):
    """Aggregate metrics for a single benchmark slice."""

    model_config = ConfigDict(extra="forbid")

    slice_id: str
    label: str
    question_ids: list[str] = Field(default_factory=list)
    question_count: int
    score: ScoreReport

    @property
    def metrics(self) -> ScoreMetrics:
        """Execute `metrics`.

        Returns:
            Any: The computed result of the function.
        """
        return self.score.metrics

    @property
    def dimension(self) -> str:
        """Execute `dimension`.

        Returns:
            Any: The computed result of the function.
        """
        return self.slice_id.split(":", maxsplit=1)[0]

    @property
    def value(self) -> str:
        """Execute `value`.

        Returns:
            Any: The computed result of the function.
        """
        parts = self.slice_id.split(":", maxsplit=1)
        return parts[1] if len(parts) == 2 else self.slice_id


class BenchmarkRunReport(BaseModel):
    """Structured benchmark output for one submission."""

    model_config = ConfigDict(extra="forbid")

    benchmark_id: str
    title: str | None = None
    reference_count: int
    assistant_score_count: int
    overall: ScoreReport
    slices: list[BenchmarkSliceReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


BenchmarkSummaryReport = BenchmarkRunReport


class MetricDelta(BaseModel):
    """Delta summary for benchmark metrics."""

    model_config = ConfigDict(extra="forbid")

    deterministic: float
    assistant: float
    grounding: float
    telemetry: float
    ttft_multiplier: float
    total_score: float


class BenchmarkDeltaReport(BaseModel):
    """Delta view for a specific overall or slice report."""

    model_config = ConfigDict(extra="forbid")

    slice_id: str
    label: str
    question_count: int
    baseline_total_score: float
    candidate_total_score: float
    delta: MetricDelta


class BenchmarkComparisonReport(BaseModel):
    """Comparison between two benchmark run reports."""

    model_config = ConfigDict(extra="forbid")

    baseline_label: str
    candidate_label: str
    benchmark_id: str
    overall: BenchmarkDeltaReport
    slices: list[BenchmarkDeltaReport] = Field(default_factory=list)
    missing_in_baseline: list[str] = Field(default_factory=list)
    missing_in_candidate: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)


class BenchmarkMetricRange(BaseModel):
    """Min/max/spread for one benchmark metric across repeated runs."""

    model_config = ConfigDict(extra="forbid")

    minimum: float
    maximum: float
    spread: float


class BenchmarkQuestionConsistency(BaseModel):
    """Repeated-run stability summary for one question."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    answer_type: AnswerType
    runs: int
    mean_total_score: float
    std_total_score: float
    min_total_score: float
    max_total_score: float
    unstable: bool


class BenchmarkConsistencyReport(BaseModel):
    """Stability summary across repeated benchmark reports."""

    model_config = ConfigDict(extra="forbid")

    benchmark_id: str | None = None
    run_count: int
    instability_threshold: float
    unstable_questions: list[BenchmarkQuestionConsistency]
    question_consistency: list[BenchmarkQuestionConsistency]
    metric_ranges: dict[str, BenchmarkMetricRange]

    @property
    def report_count(self) -> int:
        """Execute `report_count`.

        Returns:
            Any: The computed result of the function.
        """
        return self.run_count
