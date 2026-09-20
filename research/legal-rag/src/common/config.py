"""Configuration loading from YAML with environment-aware overrides."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config_overrides import apply_env_overrides, parse_page_validation_mode
from .schemas import Phase

DEFAULT_BASE_URL = "https://platform.agentic-challenge.ai/api/v1"
DEFAULT_ENV_FILE = Path(".env")
ENV_VAR_PATTERN = re.compile(r"\$(\w+)|\$\{([^}]+)\}")
SUPPORTED_ANSWER_TYPES = frozenset({"boolean", "number", "name", "names", "date", "free_text"})
DEFAULT_REASONING_EFFORT_BY_TYPE: dict[str, str] = {
    "boolean": "high",
    "number": "high",
    "name": "high",
    "names": "high",
    "date": "high",
    "free_text": "high",
}
DEFAULT_CONFIDENCE_THRESHOLD_BY_TYPE: dict[str, float] = {
    "boolean": 0.7,
    "number": 0.55,
    "name": 0.75,
    "names": 0.75,
    "date": 0.6,
    "free_text": 0.5,
}


class ConfigError(ValueError):
    """Raised when the project configuration is invalid or incomplete."""


class ApiSettings(BaseModel):
    """Remote API settings."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = "COMPETITION_API_KEY"
    api_key: str | None = Field(default=None, exclude=True, repr=False)
    timeout_seconds: float = 120.0
    user_agent: str = "agentic-rag-challenge/0.1.0"

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        """Normalize base url."""
        return value.rstrip("/")

    @field_validator("timeout_seconds")
    @classmethod
    def ensure_positive_timeout(cls, value: float) -> float:
        """Execute `ensure_positive_timeout`."""
        if value <= 0:
            raise ValueError("api.timeout_seconds must be positive.")
        return value


class StorageSettings(BaseModel):
    """Phase-local storage layout."""

    model_config = ConfigDict(extra="forbid")

    root_dir: Path = Path("data")
    questions_dirname: str = "questions"
    questions_filename: str = "questions.json"
    documents_dirname: str = "documents"
    documents_archive_name: str = "documents.zip"
    extracted_documents_dirname: str = "pdfs"
    manifests_dirname: str = "manifests"
    manifest_filename: str = "sync_manifest.json"
    corpus_dirname: str = "corpus"
    corpus_filename: str = "corpus.jsonl"
    page_map_filename: str = "page_map.json"
    corpus_manifest_filename: str = "corpus_manifest.json"
    debug_dirname: str = "debug"


class DownloadSettings(BaseModel):
    """Download-time behavior for the local sync flow."""

    model_config = ConfigDict(extra="forbid")

    extract_documents: bool = True
    overwrite: bool = False
    chunk_size_bytes: int = 1_048_576

    @field_validator("chunk_size_bytes")
    @classmethod
    def ensure_positive_chunk_size(cls, value: int) -> int:
        """Execute `ensure_positive_chunk_size`."""
        if value <= 0:
            raise ValueError("download.chunk_size_bytes must be positive.")
        return value


class EvaluationSettings(BaseModel):
    """Local evaluator/runtime policy toggles."""

    model_config = ConfigDict(extra="forbid")

    unanswerable_retrieval_policy: Literal["allow_empty", "require_non_empty"] = "allow_empty"


class IngestionSettings(BaseModel):
    """Configurable policies for the corpus parsing stage."""

    model_config = ConfigDict(extra="forbid")

    enable_ocr: bool = True
    ocr_unavailable_policy: Literal["error", "degraded"] = "error"
    ocr_device: str = "gpu:0"
    ocr_enable_hpi: bool = False
    ocr_enable_mkldnn: bool = False
    ocr_cpu_threads: int = 8
    ocr_model_root_dir: Path = Field(default_factory=lambda: Path.home() / ".paddlex" / "official_models")
    ocr_text_detection_model_dir: Path | None = None
    ocr_text_recognition_model_dir: Path | None = None
    ocr_render_dpi: int = 96
    ocr_text_detection_limit_side_len: int = 736
    ocr_text_recognition_batch_size: int = 1
    extractable_text_min_chars: int = 50
    suspicious_text_max_chars: int = 200
    large_image_area_ratio: float = 0.25
    scan_image_area_ratio: float = 0.70
    text_completeness_target_chars: int = 1500
    encoding_bad_char_threshold: float = 0.05
    enable_semantic_table_blocks: bool = False
    opendataloader_table_method: Literal["default", "cluster"] = "cluster"
    opendataloader_reading_order: Literal["off", "xycut"] = "xycut"
    opendataloader_hybrid_backend: Literal["off", "docling-fast"] = "docling-fast"
    opendataloader_hybrid_mode: Literal["auto", "full"] = "auto"
    opendataloader_hybrid_url: str | None = None
    opendataloader_hybrid_timeout_ms: int = 30_000
    fail_fast_on_document_error: bool = True

    @field_validator(
        "extractable_text_min_chars",
        "suspicious_text_max_chars",
        "text_completeness_target_chars",
        "ocr_cpu_threads",
        "ocr_render_dpi",
        "ocr_text_detection_limit_side_len",
        "ocr_text_recognition_batch_size",
    )
    @classmethod
    def ensure_positive_int(cls, value: int) -> int:
        """Execute `ensure_positive_int`."""
        if value <= 0:
            raise ValueError("ingestion integer thresholds must be positive.")
        return value

    @field_validator("opendataloader_hybrid_timeout_ms")
    @classmethod
    def ensure_positive_timeout_ms(cls, value: int) -> int:
        """Execute `ensure_positive_timeout_ms`."""
        if value <= 0:
            raise ValueError("ingestion opendataloader_hybrid_timeout_ms must be positive.")
        return value

    @field_validator("large_image_area_ratio", "scan_image_area_ratio", "encoding_bad_char_threshold")
    @classmethod
    def ensure_ratio(cls, value: float) -> float:
        """Execute `ensure_ratio`."""
        if not 0 <= value <= 1:
            raise ValueError("ingestion ratio thresholds must be between 0 and 1.")
        return value

    @field_validator("ocr_device")
    @classmethod
    def ensure_non_empty_ocr_device(cls, value: str) -> str:
        """Execute `ensure_non_empty_ocr_device`."""
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("ingestion.ocr_device must be non-empty.")
        return normalized

    @field_validator(
        "ocr_model_root_dir",
        "ocr_text_detection_model_dir",
        "ocr_text_recognition_model_dir",
        mode="before",
    )
    @classmethod
    def normalize_optional_paths(cls, value: Any) -> Any:
        """Normalize optional paths."""
        if value is None or value == "":
            return None
        return Path(str(value)).expanduser()


class AnsweringSettings(BaseModel):
    """Schema-first answering defaults and confidence controls."""

    model_config = ConfigDict(extra="forbid")

    reasoning_effort_by_type: dict[str, Literal["low", "medium", "high"]] = Field(
        default_factory=lambda: dict(DEFAULT_REASONING_EFFORT_BY_TYPE)
    )
    confidence_threshold_by_type: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_CONFIDENCE_THRESHOLD_BY_TYPE)
    )

    @field_validator("reasoning_effort_by_type")
    @classmethod
    def ensure_reasoning_effort_coverage(
        cls,
        value: dict[str, Literal["low", "medium", "high"]],
    ) -> dict[str, Literal["low", "medium", "high"]]:
        """Execute `ensure_reasoning_effort_coverage`."""
        missing = SUPPORTED_ANSWER_TYPES - set(value)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"answering.reasoning_effort_by_type missing keys: {missing_list}")
        return value

    @field_validator("confidence_threshold_by_type")
    @classmethod
    def ensure_threshold_coverage_and_range(cls, value: dict[str, float]) -> dict[str, float]:
        """Execute `ensure_threshold_coverage_and_range`."""
        missing = SUPPORTED_ANSWER_TYPES - set(value)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"answering.confidence_threshold_by_type missing keys: {missing_list}")
        for answer_type, threshold in value.items():
            if answer_type not in SUPPORTED_ANSWER_TYPES:
                continue
            if not 0 <= threshold <= 1:
                raise ValueError(f"answering.confidence_threshold_by_type[{answer_type!r}] must be between 0 and 1.")
        return value


class PageAttributionSettings(BaseModel):
    """Configurable two-pass page attribution behavior."""

    model_config = ConfigDict(extra="forbid")

    pass_a_enabled: bool = True
    suppress_title_pages: bool = True
    suppress_repeated_boilerplate: bool = True
    validation_mode: Literal["degrade_first", "strict_suppress"] = "strict_suppress"
    emit_empty_pages_for_no_answer: bool = False
    allow_solver_page_narrowing: bool = False

    @field_validator("validation_mode", mode="before")
    @classmethod
    def normalize_validation_mode(cls, value: Any) -> Any:
        """Normalize validation mode."""
        if isinstance(value, str):
            return parse_page_validation_mode(value)
        return value


class IndexingSettings(BaseModel):
    """Chunking and indexing controls."""

    model_config = ConfigDict(extra="forbid")

    token_chunk_size: int = 300
    token_chunk_overlap: int = 50
    enabled_chunk_families: list[str] = Field(
        default_factory=lambda: ["page", "section", "clause", "microchunk", "table"]
    )

    @field_validator("token_chunk_size")
    @classmethod
    def ensure_positive_chunk_size(cls, value: int) -> int:
        """Execute `ensure_positive_chunk_size`."""
        if value <= 0:
            raise ValueError("indexing.token_chunk_size must be positive.")
        return value

    @field_validator("token_chunk_overlap")
    @classmethod
    def ensure_non_negative_chunk_overlap(cls, value: int) -> int:
        """Execute `ensure_non_negative_chunk_overlap`."""
        if value < 0:
            raise ValueError("indexing.token_chunk_overlap must be non-negative.")
        return value

    @field_validator("enabled_chunk_families")
    @classmethod
    def ensure_non_empty_chunk_families(cls, value: list[str]) -> list[str]:
        """Execute `ensure_non_empty_chunk_families`."""
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            name = str(item).strip()
            if not name or name in seen:
                continue
            normalized.append(name)
            seen.add(name)
        if not normalized:
            raise ValueError("indexing.enabled_chunk_families must include at least one family.")
        return normalized


class RetrievalSettings(BaseModel):
    """Retrieval-stage budgets and optional context expansion controls."""

    model_config = ConfigDict(extra="forbid")

    candidate_budget: int = 10
    rerank_budget: int = 5
    default_evidence_budget: int = 3
    evidence_budget_by_answer_type: dict[str, int] = Field(default_factory=dict)
    question_class_budgets: dict[str, "QuestionClassBudgetSettings"] = Field(default_factory=dict)
    min_rerank_score: float = 0.0
    parent_page_expansion_enabled: bool = False
    parent_page_expansion_limit: int = 0
    query_expansion_enabled: bool = False
    query_expansion_max_queries: int = 0
    query_expansion_case_only: bool = True

    @field_validator(
        "candidate_budget",
        "rerank_budget",
        "default_evidence_budget",
        "parent_page_expansion_limit",
        "query_expansion_max_queries",
    )
    @classmethod
    def ensure_non_negative_int(cls, value: int) -> int:
        """Execute `ensure_non_negative_int`."""
        if value < 0:
            raise ValueError("retrieval integer settings must be non-negative.")
        return value

    @field_validator("min_rerank_score")
    @classmethod
    def ensure_non_negative_score(cls, value: float) -> float:
        """Execute `ensure_non_negative_score`."""
        if value < 0:
            raise ValueError("retrieval.min_rerank_score must be non-negative.")
        return value

    @field_validator("evidence_budget_by_answer_type")
    @classmethod
    def ensure_non_negative_budgets(cls, value: dict[str, int]) -> dict[str, int]:
        """Execute `ensure_non_negative_budgets`."""
        normalized: dict[str, int] = {}
        for answer_type, budget in value.items():
            key = str(answer_type).strip()
            if not key:
                continue
            budget_value = int(budget)
            if budget_value < 0:
                raise ValueError("retrieval.evidence_budget_by_answer_type budgets must be non-negative.")
            normalized[key] = budget_value
        return normalized

    @field_validator("question_class_budgets")
    @classmethod
    def ensure_normalized_question_class_budgets(
        cls,
        value: dict[str, "QuestionClassBudgetSettings"],
    ) -> dict[str, "QuestionClassBudgetSettings"]:
        """Execute `ensure_normalized_question_class_budgets`."""
        normalized: dict[str, QuestionClassBudgetSettings] = {}
        for question_class, budget in value.items():
            key = str(question_class).strip()
            if not key:
                continue
            normalized[key] = budget
        return normalized


class QuestionClassBudgetSettings(BaseModel):
    """Question-class-specific retrieval budgets."""

    model_config = ConfigDict(extra="forbid")

    candidate_budget: int
    rerank_budget: int
    evidence_budget: int

    @field_validator("candidate_budget", "rerank_budget", "evidence_budget")
    @classmethod
    def ensure_non_negative_budget(cls, value: int) -> int:
        """Execute `ensure_non_negative_budget`."""
        if value < 0:
            raise ValueError("question-class budgets must be non-negative.")
        return value


class LoggingSettings(BaseModel):
    """Runtime logging configuration."""

    model_config = ConfigDict(extra="forbid")

    directory: Path = Path("logs/project/runtime")
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    console: bool = True
    jsonl: bool = True
    debug_documents: bool = False
    debug_questions: bool = False

    @field_validator("level", mode="before")
    @classmethod
    def normalize_level(cls, value: Any) -> Any:
        """Normalize level."""
        if isinstance(value, str):
            return value.strip().upper()
        return value


class DecompositionSettings(BaseModel):
    """Cross-case decomposition controls (W4E)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_subquestions: int = 4


class RuntimeSettings(BaseModel):
    """Execution-time runtime requirements for production flows."""

    model_config = ConfigDict(extra="forbid")

    require_gpu: bool = False


class AppConfig(BaseModel):
    """Complete configuration for the W1A foundation layer."""

    model_config = ConfigDict(extra="forbid")

    version: str = "v001"
    phase: Phase = Phase.WARMUP
    api: ApiSettings = Field(default_factory=ApiSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    download: DownloadSettings = Field(default_factory=DownloadSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    answering: AnsweringSettings = Field(default_factory=AnsweringSettings)
    page_attribution: PageAttributionSettings = Field(default_factory=PageAttributionSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    decomposition: DecompositionSettings = Field(default_factory=DecompositionSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    def resolve_api_key(self) -> str:
        """Return the competition API key from the configured environment variable."""

        api_key = (self.api.api_key or "").strip()
        if not api_key:
            api_key = os.getenv(self.api.api_key_env, "").strip()
        if not api_key and self.api.api_key_env != "EVAL_API_KEY":
            api_key = os.getenv("EVAL_API_KEY", "").strip()
        if not api_key:
            raise ConfigError(
                f"Competition API key is not set. Export {self.api.api_key_env} or EVAL_API_KEY, "
                "or add it to your .env file."
            )
        return api_key


def default_config_path(module_file: str | Path | None = None) -> Path:
    """Execute `default_config_path`."""
    common_module = Path(module_file).resolve() if module_file is not None else Path(__file__).resolve()
    package_root = common_module.parents[1]
    repo_root = package_root.parent
    repo_config = repo_root / "configs" / "baseline" / "v001" / "config.yaml"
    if repo_config.exists():
        return repo_config

    return package_root / "resources" / "configs" / "baseline" / "v001" / "config.yaml"


DEFAULT_CONFIG_PATH = default_config_path()


def load_app_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    env_file: str | Path = DEFAULT_ENV_FILE,
    phase_override: str | Phase | None = None,
) -> AppConfig:
    """Load YAML config, apply environment expansion, and then merge env overrides."""

    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    file_env = _read_env_file(env_file)
    effective_env = _effective_env(file_env)
    raw_data = _read_yaml(path)
    config = AppConfig.model_validate(_expand_env_vars(raw_data, effective_env))
    config = apply_env_overrides(config, effective_env)
    config = _apply_resolved_api_key(config, effective_env)

    if phase_override is not None:
        config = config.model_copy(update={"phase": Phase(str(phase_override).lower())})

    return config


def _read_yaml(path: Path) -> dict[str, Any]:
    """Execute `_read_yaml`."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, Mapping):
        raise ConfigError(f"Config file must contain a YAML mapping at the root: {path}")

    return dict(loaded)


def _read_env_file(env_file: str | Path) -> dict[str, str]:
    """Execute `_read_env_file`."""
    env_path = Path(env_file)
    if not env_path.exists():
        return {}

    return {key: value for key, value in dotenv_values(env_path).items() if value is not None}


def _effective_env(file_env: Mapping[str, str]) -> dict[str, str]:
    """Execute `_effective_env`."""
    env = dict(file_env)
    env.update(os.environ)
    return env


def _expand_env_vars(value: Any, env: Mapping[str, str]) -> Any:
    """Execute `_expand_env_vars`."""
    if isinstance(value, str):
        return ENV_VAR_PATTERN.sub(_build_env_replacer(env), value)
    if isinstance(value, list):
        return [_expand_env_vars(item, env) for item in value]
    if isinstance(value, Mapping):
        return {key: _expand_env_vars(item, env) for key, item in value.items()}
    return value


def _build_env_replacer(env: Mapping[str, str]):
    """Build env replacer."""

    def replace(match: re.Match[str]) -> str:
        """Execute `replace`."""
        variable_name = match.group(1) or match.group(2) or ""
        return env.get(variable_name, match.group(0))

    return replace


def _apply_resolved_api_key(config: AppConfig, env: Mapping[str, str]) -> AppConfig:
    """Execute `_apply_resolved_api_key`."""
    api_key = env.get(config.api.api_key_env, "").strip()
    if not api_key and config.api.api_key_env != "EVAL_API_KEY":
        api_key = env.get("EVAL_API_KEY", "").strip()
    if not api_key:
        return config

    return config.model_copy(update={"api": config.api.model_copy(update={"api_key": api_key})})


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    env_file: str | Path = DEFAULT_ENV_FILE,
    phase_override: str | Phase | None = None,
) -> AppConfig:
    """Load config."""
    return load_app_config(config_path, env_file=env_file, phase_override=phase_override)


def resolve_api_key(config: AppConfig) -> str:
    """Resolve api key."""
    return config.resolve_api_key()


def materialize_phase_paths(config: AppConfig, *, create: bool = False):
    """Execute `materialize_phase_paths`."""
    from src.submission.sync import PhasePaths

    paths = PhasePaths.from_config(config)
    if create:
        paths.ensure_directories()
    return paths
