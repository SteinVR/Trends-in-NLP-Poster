"""End-to-end submission pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import src.indexing.service as indexing_service
from src.answering.page_attribution import PageAttributionConfig
from src.answering.service import (
    AnsweringService,
    AnswerResult,
)
from src.common.config import AppConfig
from src.common.runtime_logging import get_logger, log_event, questions_debug_enabled
from src.common.schemas import QuestionRecord
from src.evaluation.contracts import (
    UnanswerableRetrievalPolicy,
    load_question_records,
)
from src.indexing.service import HybridIndexService, IndexPaths
from src.ingestion.service import CorpusParseService, CorpusPaths
from src.providers.codex_provider import CodexAnsweringProvider
from src.retrieval.hybrid_search import QdrantHybridSearchBackend, RRFHybridSearch
from src.retrieval.reranker import QwenReranker
from src.retrieval.service import RetrievalService
from src.submission.packager import SubmissionPackager, build_submission_payload
from src.submission.pipeline_utils import (
    _architecture_summary,
    _assert_required_gpu_runtime,
    _CachingRetriever,
    _close_quietly,
    _config_aligned_to_documents_dir,
    _corpus_artifacts_complete,
    _elapsed_milliseconds,
    _estimate_input_tokens,
    _estimate_output_tokens,
    _index_artifacts_complete,
    _LazyProvider,
    _project_root,
    _supports_api_key_argument,
    _validate_locally_before_packaging,
    _write_trace_artifacts,
    format_build_submission_result,
)
from src.submission.submit_gate import MAX_CODE_ARCHIVE_BYTES, SubmitGate
from src.submission.sync import PhasePaths
from src.submission.telemetry_packer import build_answer_telemetry

ANSWERING_MODEL = "gpt-5.4-mini"
LOGGER = get_logger("submission.pipeline")
QA_CHECKPOINT_VERSION = 1
CODE_ARCHIVE_INCLUDE_PATHS = (
    "src",
    "configs",
    "scripts",
    "tools",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
)

__all__ = [
    "BuildSubmissionResult",
    "BuildSubmissionService",
    "format_build_submission_result",
]


@dataclass(slots=True)
class BuildSubmissionResult:
    """Summary for a build-submission pipeline run."""

    command: str
    phase: str
    output_dir: Path
    submission_path: Path
    code_archive_path: Path
    trace_manifest_path: Path
    answer_count: int
    reused_corpus: bool
    reused_index: bool


class BuildSubmissionService:
    """Coordinate W3A/W3B/W3C components into a validated submission bundle."""

    def __init__(
        self,
        config: AppConfig,
        *,
        env_file: Path | None = None,
        provider_class: type = CodexAnsweringProvider,
        max_code_archive_bytes: int = MAX_CODE_ARCHIVE_BYTES,
        provider_api_key: str | None = None,
        clock: Any | None = None,
    ) -> None:
        """Execute `__init__`.

        Args:
            config: Input parameter.
            env_file: Input parameter.
            provider_class: Input parameter.
            max_code_archive_bytes: Input parameter.
            provider_api_key: Input parameter.
            clock: Input parameter.

        Returns:
            None: This initializer does not return a value.
        """
        self.config = config
        self.env_file = env_file or Path(".env")
        self.provider_class = provider_class
        self.answering_model = ANSWERING_MODEL
        self.provider_api_key = provider_api_key
        self.unanswerable_retrieval_policy = UnanswerableRetrievalPolicy(
            config.evaluation.unanswerable_retrieval_policy
        )
        self.submit_gate = SubmitGate(
            max_code_archive_bytes=max_code_archive_bytes,
            unanswerable_retrieval_policy=self.unanswerable_retrieval_policy,
        )
        self.packager = SubmissionPackager(max_code_archive_bytes=max_code_archive_bytes)
        self._clock = clock or time.perf_counter

    def build_submission(
        self,
        *,
        questions_path: str | Path,
        documents_dir: str | Path,
        output_dir: str | Path,
        resume_from_checkpoint: bool = False,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 1,
    ) -> BuildSubmissionResult:
        """Build submission.

        Args:
            questions_path: Input parameter.
            documents_dir: Input parameter.
            output_dir: Input parameter.
            resume_from_checkpoint: Whether to reuse saved QA progress when available.
            checkpoint_path: Optional checkpoint JSON path.
            checkpoint_every: Persist checkpoint every N newly processed questions.

        Returns:
            Any: The computed result of the function.
        """
        resolved_questions_path = Path(questions_path)
        resolved_documents_dir = Path(documents_dir)
        resolved_output_dir = Path(output_dir)
        resolved_checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else resolved_output_dir / "qa_checkpoint.json"
        )
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be >= 1.")
        log_event(
            LOGGER,
            logging.INFO,
            "build_submission.start",
            "Starting submission build.",
            stage="submission",
            questions_path=resolved_questions_path,
            documents_dir=resolved_documents_dir,
            output_dir=resolved_output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            checkpoint_path=resolved_checkpoint_path,
            checkpoint_every=checkpoint_every,
        )

        if not resolved_questions_path.exists():
            raise ValueError(f"Questions file does not exist: {resolved_questions_path}")
        if not resolved_documents_dir.exists():
            raise ValueError(f"Documents directory does not exist: {resolved_documents_dir}")

        runtime_config = _config_aligned_to_documents_dir(self.config, resolved_documents_dir)
        _assert_required_gpu_runtime(runtime_config)
        phase_paths = PhasePaths.from_config(runtime_config)
        if phase_paths.documents_extract_dir.resolve() != resolved_documents_dir.resolve():
            raise ValueError(
                "build-submission expects --documents-dir to match the phase-local documents directory "
                f"({phase_paths.documents_extract_dir})."
            )

        corpus_started = time.perf_counter()
        reused_corpus = self._ensure_corpus(runtime_config, resolved_documents_dir)
        corpus_ms = round((time.perf_counter() - corpus_started) * 1000, 1)
        log_event(
            LOGGER,
            logging.INFO,
            "build_submission.stage.corpus",
            "Corpus stage completed.",
            stage="submission",
            reused=reused_corpus,
            duration_ms=corpus_ms,
        )

        index_started = time.perf_counter()
        reused_index = self._ensure_index(runtime_config)
        index_ms = round((time.perf_counter() - index_started) * 1000, 1)
        log_event(
            LOGGER,
            logging.INFO,
            "build_submission.stage.index",
            "Index stage completed.",
            stage="submission",
            reused=reused_index,
            duration_ms=index_ms,
        )

        questions = load_question_records(resolved_questions_path)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        log_event(
            LOGGER,
            logging.INFO,
            "build_submission.qa.start",
            "Starting QA pipeline.",
            stage="submission",
            question_count=len(questions),
            reused_corpus=reused_corpus,
            reused_index=reused_index,
        )
        qa_started = time.perf_counter()
        answers, traces = self._run_qa_pipeline(
            runtime_config,
            questions,
            checkpoint_path=resolved_checkpoint_path,
            resume_from_checkpoint=resume_from_checkpoint,
            checkpoint_every=checkpoint_every,
        )
        qa_ms = round((time.perf_counter() - qa_started) * 1000, 1)
        log_event(
            LOGGER,
            logging.INFO,
            "build_submission.stage.qa",
            "QA pipeline completed.",
            stage="submission",
            question_count=len(questions),
            answer_count=len(answers),
            duration_ms=qa_ms,
        )

        submission_payload = build_submission_payload(
            architecture_summary=_architecture_summary(),
            answers=answers,
        )
        _validate_locally_before_packaging(
            submission_payload=submission_payload,
            questions_path=resolved_questions_path,
            documents_dir=resolved_documents_dir,
            unanswerable_retrieval_policy=self.unanswerable_retrieval_policy,
        )

        submission_path = self.packager.write_submission_payload(
            submission_payload=submission_payload,
            output_dir=resolved_output_dir,
        )
        trace_manifest_path = _write_trace_artifacts(
            output_dir=resolved_output_dir,
            phase=runtime_config.phase.value,
            traces=traces,
        )
        code_archive_path = self.packager.create_code_archive(
            output_path=resolved_output_dir / "code_archive.zip",
            project_root=_project_root(),
            excluded_paths={resolved_output_dir.resolve()},
            include_paths=CODE_ARCHIVE_INCLUDE_PATHS,
        )

        self.submit_gate.validate(
            submission_payload=submission_payload,
            questions_path=resolved_questions_path,
            documents_dir=resolved_documents_dir,
            code_archive_path=code_archive_path,
        )

        result = BuildSubmissionResult(
            command="build-submission",
            phase=runtime_config.phase.value,
            output_dir=resolved_output_dir,
            submission_path=submission_path,
            code_archive_path=code_archive_path,
            trace_manifest_path=trace_manifest_path,
            answer_count=len(answers),
            reused_corpus=reused_corpus,
            reused_index=reused_index,
        )
        log_event(
            LOGGER,
            logging.INFO,
            "build_submission.complete",
            "Submission build completed.",
            stage="submission",
            answer_count=result.answer_count,
            submission_path=result.submission_path,
            code_archive_path=result.code_archive_path,
            trace_manifest_path=result.trace_manifest_path,
            reused_corpus=result.reused_corpus,
            reused_index=result.reused_index,
        )
        return result

    def _ensure_corpus(self, config: AppConfig, documents_dir: Path) -> bool:
        """Execute `_ensure_corpus`.

        Args:
            config: Input parameter.
            documents_dir: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        corpus_paths = CorpusPaths.from_config(config)
        if _corpus_artifacts_complete(corpus_paths=corpus_paths, documents_dir=documents_dir):
            log_event(
                LOGGER,
                logging.INFO,
                "build_submission.corpus.reuse",
                "Reused existing corpus artifacts.",
                stage="submission",
                corpus_manifest_path=corpus_paths.corpus_manifest_path,
            )
            return True

        service = CorpusParseService(config, env_file=self.env_file)
        # Force overwrite when our integrity check fails to avoid parse-corpus internal "skip" cache reuse.
        service.parse_corpus(dry_run=False, overwrite=True)
        log_event(
            LOGGER,
            logging.INFO,
            "build_submission.corpus.rebuild",
            "Rebuilt corpus artifacts for submission build.",
            stage="submission",
            corpus_manifest_path=corpus_paths.corpus_manifest_path,
        )
        return False

    def _ensure_index(self, config: AppConfig) -> bool:
        """Execute `_ensure_index`.

        Args:
            config: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        index_paths = IndexPaths.from_config(config)
        if _index_artifacts_complete(index_paths=index_paths):
            log_event(
                LOGGER,
                logging.INFO,
                "build_submission.index.reuse",
                "Reused existing index artifacts.",
                stage="submission",
                index_manifest_path=index_paths.manifest_path,
            )
            return True

        service = HybridIndexService(config)
        # Force overwrite when integrity checks fail to prevent stale sparse-state reuse.
        service.build_indices(dry_run=False, overwrite=True)
        if not _index_artifacts_complete(index_paths=index_paths):
            raise RuntimeError(
                "Index rebuild completed but required artifacts are still incomplete. "
                "Refusing to continue with potentially dense-only or partial index state."
            )
        log_event(
            LOGGER,
            logging.INFO,
            "build_submission.index.rebuild",
            "Rebuilt index artifacts for submission build.",
            stage="submission",
            index_manifest_path=index_paths.manifest_path,
        )
        return False

    def _run_qa_pipeline(
        self,
        config: AppConfig,
        questions: list[QuestionRecord],
        *,
        checkpoint_path: Path | None = None,
        resume_from_checkpoint: bool = False,
        checkpoint_every: int = 1,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run qa pipeline.

        Args:
            config: Input parameter.
            questions: Input parameter.
            checkpoint_path: Optional path to QA checkpoint JSON.
            resume_from_checkpoint: Whether checkpoint resume is enabled.
            checkpoint_every: Persist checkpoint every N newly processed questions.

        Returns:
            Any: The computed result of the function.
        """
        index_paths = IndexPaths.from_config(config)
        dense_embedder = indexing_service.build_query_embedder(config)
        backend = QdrantHybridSearchBackend(
            index_dir=index_paths.index_dir,
            dense_embedder=dense_embedder,
            enable_sparse_compat=True,
        )

        provider_model = self.answering_model
        provider = _LazyProvider(
            factory=self._build_provider,
            model=provider_model,
        )
        retrieval_settings = config.retrieval

        retrieval_service = RetrievalService(
            hybrid_search=RRFHybridSearch(backend=backend),
            reranker=QwenReranker(),
            candidate_budget=retrieval_settings.candidate_budget,
            rerank_budget=retrieval_settings.rerank_budget,
            default_evidence_budget=retrieval_settings.default_evidence_budget,
            evidence_budget_by_answer_type=dict(retrieval_settings.evidence_budget_by_answer_type),
            question_class_budgets=dict(retrieval_settings.question_class_budgets),
            min_rerank_score=retrieval_settings.min_rerank_score,
            parent_page_expansion_enabled=retrieval_settings.parent_page_expansion_enabled,
            parent_page_expansion_limit=retrieval_settings.parent_page_expansion_limit,
            query_expansion_enabled=retrieval_settings.query_expansion_enabled,
            query_expansion_max_queries=retrieval_settings.query_expansion_max_queries,
            query_expansion_case_only=retrieval_settings.query_expansion_case_only,
            page_map_paths=(index_paths.page_map_path,),
        )
        retriever = _CachingRetriever(retrieval_service)
        answering_service = AnsweringService(
            retriever=retriever,
            provider=provider,
            reasoning_effort_by_type=config.answering.reasoning_effort_by_type,
            confidence_threshold_by_type=config.answering.confidence_threshold_by_type,
            page_attribution_config=PageAttributionConfig(**config.page_attribution.model_dump()),
        )

        question_order = [question.id for question in questions]
        question_set_digest = self._question_set_digest(questions)
        answers_by_question_id: dict[str, dict[str, Any]] = {}
        traces_by_question_id: dict[str, dict[str, Any]] = {}
        resumed_answer_count = 0
        newly_processed_count = 0
        pending_since_checkpoint = 0

        if checkpoint_path is not None and resume_from_checkpoint:
            answers_by_question_id, traces_by_question_id = self._load_qa_checkpoint(
                checkpoint_path=checkpoint_path,
                question_set_digest=question_set_digest,
            )
            resumed_answer_count = len(answers_by_question_id)
            if resumed_answer_count > 0:
                log_event(
                    LOGGER,
                    logging.INFO,
                    "build_submission.qa.resume",
                    "Resuming QA from checkpoint.",
                    stage="submission",
                    checkpoint_path=checkpoint_path,
                    resumed_answer_count=resumed_answer_count,
                    total_question_count=len(questions),
                )

        total_questions = len(questions)

        try:
            for index, question in enumerate(questions, start=1):
                if question.id in answers_by_question_id:
                    continue

                question_trace_id = uuid4().hex
                question_started_at = self._clock()

                answer_result = self._answer_question(
                    question=question,
                    answering_service=answering_service,
                )
                total_time_ms = _elapsed_milliseconds(
                    started_at=question_started_at,
                    ended_at=self._clock(),
                )

                retrieval_result = retriever.consume_latest(question)
                default_model_name = provider_model
                model_name = answer_result.model_name or default_model_name
                input_tokens = _estimate_input_tokens(question=question, retrieval_result=retrieval_result)
                output_tokens = _estimate_output_tokens(answer_result)

                telemetry = build_answer_telemetry(
                    retrieval_result=retrieval_result,
                    answer_result=answer_result,
                    model_name=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_time_ms=total_time_ms,
                    ttft_ms=total_time_ms,
                    tpot_ms=0.0,
                )

                answers_by_question_id[question.id] = {
                    "question_id": question.id,
                    "answer": answer_result.answer,
                    "telemetry": telemetry,
                }
                traces_by_question_id[question.id] = {
                    "trace_id": question_trace_id,
                    "question_id": question.id,
                    "answer_type": question.answer_type,
                    "model_name": model_name,
                    "candidate_count": retrieval_result.candidate_count,
                    "reranked_count": retrieval_result.reranked_count,
                    "evidence_chunk_count": len(retrieval_result.evidence_chunks),
                    "retrieved_chunk_pages": [
                        {
                            "doc_id": str(reference["doc_id"]),
                            "page_numbers": [int(page_number) for page_number in reference["page_numbers"]],
                        }
                        for reference in telemetry["retrieval"]["retrieved_chunk_pages"]
                    ],
                }
                newly_processed_count += 1
                pending_since_checkpoint += 1

                if checkpoint_path is not None and pending_since_checkpoint >= checkpoint_every:
                    self._save_qa_checkpoint(
                        checkpoint_path=checkpoint_path,
                        question_set_digest=question_set_digest,
                        question_order=question_order,
                        answers_by_question_id=answers_by_question_id,
                        traces_by_question_id=traces_by_question_id,
                    )
                    pending_since_checkpoint = 0

                if questions_debug_enabled():
                    log_event(
                        LOGGER,
                        logging.DEBUG,
                        "build_submission.question.complete",
                        "Question processed in QA pipeline.",
                        stage="submission",
                        question_id=question.id,
                        answer_type=question.answer_type,
                        candidate_count=retrieval_result.candidate_count,
                        reranked_count=retrieval_result.reranked_count,
                        evidence_chunk_count=len(retrieval_result.evidence_chunks),
                        duration_ms=total_time_ms,
                    )
                if index % 10 == 0 or index == total_questions:
                    log_event(
                        LOGGER,
                        logging.INFO,
                        "build_submission.qa.progress",
                        "QA pipeline progress updated.",
                        stage="submission",
                        processed_question_count=index,
                        total_question_count=total_questions,
                        answered_question_count=len(answers_by_question_id),
                        resumed_answer_count=resumed_answer_count,
                        newly_processed_count=newly_processed_count,
                    )
            if checkpoint_path is not None:
                self._save_qa_checkpoint(
                    checkpoint_path=checkpoint_path,
                    question_set_digest=question_set_digest,
                    question_order=question_order,
                    answers_by_question_id=answers_by_question_id,
                    traces_by_question_id=traces_by_question_id,
                )
        finally:
            _close_quietly(provider)
            _close_quietly(backend)

        answers: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        missing_question_ids: list[str] = []
        for question_id in question_order:
            answer = answers_by_question_id.get(question_id)
            trace = traces_by_question_id.get(question_id)
            if answer is None or trace is None:
                missing_question_ids.append(question_id)
                continue
            answers.append(answer)
            traces.append(trace)

        if missing_question_ids:
            raise ValueError(
                "QA checkpoint state is incomplete for the active question set; "
                f"missing {len(missing_question_ids)} question_ids."
            )

        return answers, traces

    def _question_set_digest(self, questions: list[QuestionRecord]) -> str:
        """Build a stable digest used to validate checkpoint compatibility."""
        serialized = json.dumps([question.id for question in questions], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _load_qa_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        question_set_digest: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        """Load QA checkpoint payload if it exists.

        Args:
            checkpoint_path: Checkpoint file path.
            question_set_digest: Digest of the currently requested question set.

        Returns:
            tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]: Answers and traces keyed by question_id.
        """
        if not checkpoint_path.exists():
            return {}, {}

        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")
        if payload.get("version") != QA_CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported checkpoint version in {checkpoint_path}: {payload.get('version')!r}")
        if payload.get("question_set_digest") != question_set_digest:
            raise ValueError("Checkpoint question set mismatch; refusing to resume with incompatible questions.")

        raw_answers = payload.get("answers_by_question_id")
        raw_traces = payload.get("traces_by_question_id")
        if not isinstance(raw_answers, dict) or not isinstance(raw_traces, dict):
            raise ValueError(f"Malformed checkpoint content: {checkpoint_path}")

        answers_by_question_id: dict[str, dict[str, Any]] = {}
        for question_id, answer in raw_answers.items():
            if isinstance(question_id, str) and isinstance(answer, dict):
                answers_by_question_id[question_id] = answer

        traces_by_question_id: dict[str, dict[str, Any]] = {}
        for question_id, trace in raw_traces.items():
            if isinstance(question_id, str) and isinstance(trace, dict):
                traces_by_question_id[question_id] = trace

        return answers_by_question_id, traces_by_question_id

    def _save_qa_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        question_set_digest: str,
        question_order: list[str],
        answers_by_question_id: dict[str, dict[str, Any]],
        traces_by_question_id: dict[str, dict[str, Any]],
    ) -> None:
        """Persist current QA progress atomically."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_answers = {
            question_id: answers_by_question_id[question_id]
            for question_id in question_order
            if question_id in answers_by_question_id
        }
        ordered_traces = {
            question_id: traces_by_question_id[question_id]
            for question_id in question_order
            if question_id in traces_by_question_id
        }
        payload = {
            "version": QA_CHECKPOINT_VERSION,
            "question_set_digest": question_set_digest,
            "saved_at": datetime.now(tz=UTC).isoformat(),
            "answer_count": len(ordered_answers),
            "trace_count": len(ordered_traces),
            "answers_by_question_id": ordered_answers,
            "traces_by_question_id": ordered_traces,
        }
        temp_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(checkpoint_path)

    def _answer_question(
        self,
        *,
        question: QuestionRecord,
        answering_service: AnsweringService,
    ) -> AnswerResult:
        """Execute `_answer_question`.

        Args:
            question: Input parameter.
            answering_service: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        return answering_service.answer_question(question, allow_decomposition=False)

    def _build_provider(self, model: str) -> Any:
        """Build provider.

        Args:
            model: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        provider_kwargs: dict[str, Any] = {"model": model}
        if self.provider_api_key and _supports_api_key_argument(self.provider_class):
            provider_kwargs["api_key"] = self.provider_api_key
        return self.provider_class(**provider_kwargs)
