"""Submission pipeline helpers shared by build orchestration."""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import fitz
from qdrant_client import QdrantClient

from src.answering.service import AnswerResult
from src.common.config import AppConfig
from src.common.file_hash import sha256_for_file
from src.common.schemas import CorpusParseManifest, ParseStatus, Phase, QuestionRecord
from src.evaluation.contracts import UnanswerableRetrievalPolicy, load_question_records, question_types_from_questions
from src.evaluation.validator import (
    SubmissionValidationReport,
    SubmissionValidator,
    ValidationPolicy,
    build_corpus_registry,
)
from src.indexing.service import IndexPaths
from src.ingestion.service import CorpusPaths
from src.retrieval.service import RetrievalResult, RetrievalService


class _BuildSubmissionResultLike(Protocol):
    """Protocol for formatting build-submission summaries."""

    command: str
    phase: str
    answer_count: int
    reused_corpus: bool
    reused_index: bool
    output_dir: Path
    submission_path: Path
    code_archive_path: Path
    trace_manifest_path: Path


def format_build_submission_result(result: _BuildSubmissionResultLike) -> str:
    """Render a user-facing summary for build-submission."""

    lines = [
        f"command: {result.command}",
        f"phase: {result.phase}",
        f"answers: {result.answer_count}",
        f"reused_corpus: {str(result.reused_corpus).lower()}",
        f"reused_index: {str(result.reused_index).lower()}",
        f"output_dir: {result.output_dir}",
        f"submission_path: {result.submission_path}",
        f"code_archive_path: {result.code_archive_path}",
        f"trace_manifest_path: {result.trace_manifest_path}",
    ]
    return "\n".join(lines)


@dataclass(slots=True)
class _CachingRetriever:
    delegate: RetrievalService
    _latest_question_id: str | None = field(default=None, init=False)
    _latest_result: RetrievalResult | None = field(default=None, init=False)

    def retrieve(
        self,
        question: QuestionRecord,
        *,
        query_metadata: Any | None = None,
    ) -> RetrievalResult:
        """Execute `retrieve`."""
        result = self.delegate.retrieve(question, query_metadata=query_metadata)
        self._latest_question_id = question.id
        self._latest_result = result
        return result

    def consume_latest(
        self,
        question: QuestionRecord,
        *,
        query_metadata: Any | None = None,
    ) -> RetrievalResult:
        """Execute `consume_latest`."""
        if self._latest_question_id == question.id and self._latest_result is not None:
            result = self._latest_result
            self._latest_question_id = None
            self._latest_result = None
            return result
        return self.delegate.retrieve(question, query_metadata=query_metadata)

    def discard_latest(self) -> None:
        """Drop cached retrieval state without triggering a new search."""
        self._latest_question_id = None
        self._latest_result = None


@dataclass(slots=True)
class _LazyProvider:
    factory: Any
    model: str
    _delegate: Any | None = field(default=None, init=False, repr=False)

    def generate_structured(self, **kwargs: Any) -> Any:
        """Execute `generate_structured`."""
        provider = self._provider()
        structured_method = getattr(provider, "generate_structured", None)
        if not callable(structured_method):
            raise AttributeError("Provider does not implement generate_structured.")
        return structured_method(**kwargs)

    def close(self) -> None:
        """Execute `close`."""
        if self._delegate is None:
            return
        _close_quietly(self._delegate)
        self._delegate = None

    def _provider(self) -> Any:
        """Execute `_provider`."""
        if self._delegate is None:
            self._delegate = self.factory(self.model)
        return self._delegate


def _validate_locally_before_packaging(
    *,
    submission_payload: dict[str, Any],
    questions_path: Path,
    documents_dir: Path,
    unanswerable_retrieval_policy: UnanswerableRetrievalPolicy,
) -> SubmissionValidationReport:
    """Validate locally before packaging."""
    validator = SubmissionValidator(
        question_types=question_types_from_questions(load_question_records(questions_path)),
        corpus_registry=build_corpus_registry(documents_dir),
        policy=ValidationPolicy(
            unanswerable_retrieval_policy=unanswerable_retrieval_policy,
            require_full_question_coverage=True,
            emit_ambiguity_warning=True,
        ),
    )
    report = validator.validate(submission_payload)
    if report.valid:
        return report

    first_error_message = _first_validation_error_message(report)
    if first_error_message is None:
        raise ValueError("Local validation failed before packaging.")
    raise ValueError(f"Local validation failed before packaging: {first_error_message}")


def _first_validation_error_message(report: SubmissionValidationReport) -> str | None:
    """Execute `_first_validation_error_message`."""
    for issue in report.issues:
        if issue.severity == "error":
            return issue.message
    for answer_report in report.answers:
        for issue in answer_report.issues:
            if issue.severity == "error":
                return issue.message
    return None


def _estimate_input_tokens(*, question: QuestionRecord, retrieval_result: RetrievalResult) -> int:
    """Execute `_estimate_input_tokens`."""
    total = _estimate_tokens_from_text(question.question)
    for chunk in retrieval_result.evidence_chunks:
        total += _estimate_tokens_from_text(chunk.text)
    return total


def _estimate_output_tokens(answer_result: AnswerResult) -> int:
    """Execute `_estimate_output_tokens`."""
    if answer_result.answer is None:
        return 0
    return _estimate_tokens_from_text(str(answer_result.answer))


def _estimate_tokens_from_text(text: str) -> int:
    """Execute `_estimate_tokens_from_text`."""
    return len(re.findall(r"[A-Za-z0-9]+", text))


def _elapsed_milliseconds(*, started_at: float, ended_at: float) -> float:
    """Execute `_elapsed_milliseconds`."""
    return max((float(ended_at) - float(started_at)) * 1000.0, 0.0)


def _corpus_artifacts_complete(*, corpus_paths: CorpusPaths, documents_dir: Path) -> bool:
    """Execute `_corpus_artifacts_complete`."""
    required_paths = [
        corpus_paths.corpus_path,
        corpus_paths.page_map_path,
        corpus_paths.corpus_manifest_path,
        corpus_paths.debug_dir,
    ]
    if any(not path.exists() for path in required_paths):
        return False

    try:
        manifest = CorpusParseManifest.model_validate_json(
            corpus_paths.corpus_manifest_path.read_text(encoding="utf-8")
        )
    except Exception:
        return False

    current_pdf_paths = sorted(path for path in documents_dir.rglob("*.pdf") if path.is_file())
    current_doc_ids = {path.stem for path in current_pdf_paths}
    manifest_doc_ids = {record.doc_id for record in manifest.records}
    if manifest_doc_ids != current_doc_ids:
        return False

    manifest_by_doc_id = {record.doc_id: record for record in manifest.records}
    for pdf_path in current_pdf_paths:
        doc_id = pdf_path.stem
        manifest_record = manifest_by_doc_id.get(doc_id)
        if manifest_record is None:
            return False

        if manifest_record.sha256 != sha256_for_file(pdf_path):
            return False

        if manifest_record.parse_status is ParseStatus.FAILED:
            continue

        try:
            with fitz.open(pdf_path) as document:
                page_count = int(document.page_count)
        except Exception:
            return False
        if int(manifest_record.page_count) != page_count:
            return False

    required_debug_docs = {
        record.doc_id
        for record in manifest.records
        if record.parse_status is not ParseStatus.FAILED and record.doc_id in current_doc_ids
    }
    return all((corpus_paths.debug_dir / f"{doc_id}.md").exists() for doc_id in required_debug_docs)


def _index_artifacts_complete(*, index_paths: IndexPaths) -> bool:
    """Execute `_index_artifacts_complete`."""
    required_paths = [
        index_paths.qdrant_dir,
        index_paths.page_parent_map_path,
        index_paths.manifest_path,
        index_paths.sparse_encoder_state_path,
    ]
    if any(not path.exists() for path in required_paths):
        return False

    if not any(index_paths.qdrant_dir.iterdir()):
        return False

    try:
        manifest_payload = json.loads(index_paths.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest_payload, dict):
            return False

        expected_source = _index_source_artifacts(index_paths)
        if expected_source is None:
            return False
        if manifest_payload.get("source_artifacts") != expected_source:
            return False

        chunk_count = manifest_payload.get("chunk_count")
        if not isinstance(chunk_count, int) or chunk_count <= 0:
            return False

        sparse_state_manifest = manifest_payload.get("sparse_encoder_state")
        if not isinstance(sparse_state_manifest, dict):
            return False
        if sparse_state_manifest.get("path") != index_paths.sparse_encoder_state_path.name:
            return False
        sparse_state_sha = sparse_state_manifest.get("sha256")
        if not isinstance(sparse_state_sha, str) or not sparse_state_sha:
            return False
        if sparse_state_sha != sha256_for_file(index_paths.sparse_encoder_state_path):
            return False

        client = QdrantClient(path=str(index_paths.qdrant_dir))
        try:
            collections = client.get_collections().collections
            if len(collections) != 1:
                return False
            collection_info = client.get_collection(collections[0].name)
            if collection_info.points_count == 0:
                return False
            if collection_info.points_count != chunk_count:
                return False
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

        page_parent_map_payload = json.loads(index_paths.page_parent_map_path.read_text(encoding="utf-8"))
        if not isinstance(page_parent_map_payload, dict):
            return False
        if len(page_parent_map_payload) != chunk_count:
            return False
    except Exception:
        return False

    return True


def _index_source_artifacts(index_paths: IndexPaths) -> dict[str, Any] | None:
    """Execute `_index_source_artifacts`."""
    source_paths = [
        index_paths.corpus_path,
        index_paths.page_map_path,
        index_paths.corpus_manifest_path,
    ]
    if any(not path.exists() for path in source_paths):
        return None

    return {
        "corpus_path": str(index_paths.corpus_path),
        "corpus_sha256": sha256_for_file(index_paths.corpus_path),
        "page_map_path": str(index_paths.page_map_path),
        "page_map_sha256": sha256_for_file(index_paths.page_map_path),
        "corpus_manifest_path": str(index_paths.corpus_manifest_path),
        "corpus_manifest_sha256": sha256_for_file(index_paths.corpus_manifest_path),
    }


def _assert_required_gpu_runtime(config: AppConfig) -> None:
    """Execute `_assert_required_gpu_runtime`."""
    if not config.runtime.require_gpu:
        return

    errors: list[str] = []
    try:
        import torch
    except Exception as exc:
        errors.append(f"torch unavailable: {exc}")
    else:
        if not torch.cuda.is_available():
            errors.append("torch CUDA runtime is not available for retrieval models")

    if config.ingestion.enable_ocr and config.ingestion.ocr_device.strip().lower().startswith("gpu"):
        try:
            import paddle
        except Exception as exc:
            errors.append(f"paddle unavailable: {exc}")
        else:
            if not paddle.device.is_compiled_with_cuda():
                errors.append("paddle is not compiled with CUDA for OCR")
            elif int(paddle.device.cuda.device_count()) <= 0:
                errors.append("paddle does not see a CUDA device for OCR")

    if errors:
        raise RuntimeError("GPU runtime contract not satisfied for build-submission: " + "; ".join(errors))


def _config_aligned_to_documents_dir(config: AppConfig, documents_dir: Path) -> AppConfig:
    """Execute `_config_aligned_to_documents_dir`."""
    resolved_documents_dir = documents_dir.resolve()
    extracted_name = config.storage.extracted_documents_dirname
    documents_name = config.storage.documents_dirname

    if resolved_documents_dir.name != extracted_name:
        return config
    if resolved_documents_dir.parent.name != documents_name:
        return config

    phase_name = resolved_documents_dir.parent.parent.name
    try:
        phase = Phase(phase_name)
    except ValueError:
        return config

    root_dir = resolved_documents_dir.parent.parent.parent
    return config.model_copy(
        update={
            "phase": phase,
            "storage": config.storage.model_copy(update={"root_dir": root_dir}),
        }
    )


def _write_trace_artifacts(*, output_dir: Path, phase: str, traces: list[dict[str, Any]]) -> Path:
    """Execute `_write_trace_artifacts`."""
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace_files: list[dict[str, str]] = []
    for entry in traces:
        question_id = str(entry["question_id"])
        trace_path = trace_dir / _trace_filename(question_id)
        with trace_path.open("w", encoding="utf-8") as handle:
            json.dump(entry, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        trace_files.append(
            {
                "question_id": question_id,
                "trace_id": str(entry["trace_id"]),
                "path": str(trace_path.relative_to(output_dir)),
            }
        )

    manifest_path = output_dir / "trace_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_trace_id": uuid4().hex,
                "phase": phase,
                "generated_at": datetime.now(tz=UTC).isoformat(),
                "trace_count": len(trace_files),
                "traces": trace_files,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    return manifest_path


def _trace_filename(question_id: str) -> str:
    """Execute `_trace_filename`."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", question_id).strip("._")
    if not normalized:
        normalized = "question"
    safe_prefix = normalized[:48]
    digest = sha256(question_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe_prefix}__{digest}.json"


def _architecture_summary() -> str:
    """Execute `_architecture_summary`."""
    summary = (
        "Pipeline builds/reuses phase-local corpus and hybrid index artifacts, runs hybrid retrieval with reranking, "
        "routes questions through deterministic and free-text answering, packs telemetry, validates locally, and "
        "ships submission plus code archive."
    )
    return summary[:500]


def _project_root() -> Path:
    """Execute `_project_root`."""
    return Path(__file__).resolve().parents[2]


def _supports_api_key_argument(provider_class: type[Any]) -> bool:
    """Execute `_supports_api_key_argument`."""
    try:
        signature = inspect.signature(provider_class)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "api_key":
            return True
    return False


def _synthetic_retrieval_from_decomposition(
    question: QuestionRecord,
    answer_result: AnswerResult,
) -> RetrievalResult:
    """Build a lightweight RetrievalResult from a decomposed answer's page trace."""
    return RetrievalResult(
        question_id=question.id,
        query=question.question,
        answer_type=question.answer_type,
        candidate_count=0,
        reranked_count=0,
        evidence_chunks=[],
        retrieved_chunk_pages=list(answer_result.evidence_pages),
        diagnostics={"decomposed": True},
        is_unanswerable=answer_result.answer is None,
    )


def _close_quietly(resource: Any) -> None:
    """Execute `_close_quietly`."""
    close = getattr(resource, "close", None)
    if callable(close):
        close()
