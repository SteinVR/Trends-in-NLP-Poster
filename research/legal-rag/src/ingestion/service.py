"""Phase-aware corpus parsing orchestration."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence

from dotenv import dotenv_values

from src.common.config import AppConfig
from src.common.runtime_logging import documents_debug_enabled, get_logger, log_event
from src.common.schemas import (
    CanonicalPageRecord,
    CorpusParseManifest,
    CorpusParseManifestRecord,
    NormalizedStructuralBlock,
    PageMapRecord,
    ParsedDocument,
    ParseStatus,
    TableBlock,
)
from src.ingestion.corpus_builder import CorpusBuilder, build_failed_manifest_record
from src.ingestion.ocr_engine import build_ocr_engine
from src.ingestion.pdf_parser import DocumentParser
from src.ingestion.sidecars import load_raw_parse_sidecar, write_raw_parse_sidecar
from src.ingestion.structural_normalizer import normalize_structural_sidecar
from src.ingestion.table_serializer import serialize_document_tables
from src.providers import CodexStructuredOutputProvider, StructuredOutputProvider
from src.submission.sync import PhasePaths

LOGGER = get_logger("ingestion.service")


@dataclass(slots=True)
class CorpusPaths:
    """Resolved local filesystem layout for corpus parsing artifacts."""

    documents_dir: Path
    corpus_dir: Path
    corpus_path: Path
    page_map_path: Path
    corpus_manifest_path: Path
    debug_dir: Path
    raw_parse_dir: Path

    @classmethod
    def from_config(cls, config: AppConfig) -> "CorpusPaths":
        """Execute `from_config`.

        Args:
            config: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        phase_paths = PhasePaths.from_config(config)
        corpus_dir = phase_paths.phase_dir / config.storage.corpus_dirname
        return cls(
            documents_dir=phase_paths.documents_extract_dir,
            corpus_dir=corpus_dir,
            corpus_path=corpus_dir / config.storage.corpus_filename,
            page_map_path=corpus_dir / config.storage.page_map_filename,
            corpus_manifest_path=corpus_dir / config.storage.corpus_manifest_filename,
            debug_dir=corpus_dir / config.storage.debug_dirname,
            raw_parse_dir=corpus_dir / "raw_parse",
        )

    def ensure_directories(self) -> None:
        """Execute `ensure_directories`.

        Returns:
            None: This function does not return a value.
        """
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.raw_parse_dir.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ParseCorpusResult:
    """User-facing summary for parse-corpus CLI actions."""

    command: str
    phase: str
    dry_run: bool
    manifest_path: Path
    planned_paths: list[Path] = field(default_factory=list)
    written_paths: list[Path] = field(default_factory=list)
    skipped_paths: list[Path] = field(default_factory=list)
    parsed_document_count: int | None = None
    parsed_page_count: int | None = None
    failed_document_count: int | None = None


class DocumentParseFailureError(ValueError):
    """Fail-fast parse error wrapped into a CLI-handled exception type."""

    def __init__(self, *, doc_id: str, pdf_path: Path, cause: Exception) -> None:
        """Execute `__init__`.

        Args:
            doc_id: Input parameter.
            pdf_path: Input parameter.
            cause: Input parameter.

        Returns:
            None: This initializer does not return a value.
        """
        super().__init__(
            f"Document parse failed for '{doc_id}' ({pdf_path.name}): {type(cause).__name__}: {cause}"
        )
        self.doc_id = doc_id
        self.pdf_path = pdf_path
        self.cause = cause


class CorpusParseService:
    """Orchestrates phase-local PDF parsing into canonical corpus artifacts."""

    def __init__(self, config: AppConfig, *, env_file: Path | None = None) -> None:
        """Execute `__init__`.

        Args:
            config: Input parameter.
            env_file: Input parameter.

        Returns:
            None: This initializer does not return a value.
        """
        self.config = config
        self.env_file = env_file or Path(".env")
        self.paths = CorpusPaths.from_config(config)
        self.ocr_engine = build_ocr_engine(config.ingestion)
        self.parser = DocumentParser(config, ocr_engine=self.ocr_engine)
        self.builder = CorpusBuilder(config)
        self._structured_provider: StructuredOutputProvider | None = None

    def parse_corpus(
        self,
        *,
        dry_run: bool = False,
        overwrite: bool | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> ParseCorpusResult:
        """Parse corpus.

        Args:
            dry_run: Input parameter.
            overwrite: Input parameter.
            doc_ids: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        overwrite = self.config.download.overwrite if overwrite is None else overwrite
        planned_paths = self._artifact_paths()
        log_event(
            LOGGER,
            logging.INFO,
            "parse.start",
            "Starting corpus parse.",
            stage="ingestion",
            dry_run=dry_run,
            overwrite=overwrite,
            documents_dir=self.paths.documents_dir,
            selected_doc_ids=list(doc_ids or []),
        )
        result = ParseCorpusResult(
            command="parse-corpus",
            phase=self.config.phase.value,
            dry_run=dry_run,
            manifest_path=self.paths.corpus_manifest_path,
            planned_paths=planned_paths,
        )
        if dry_run:
            log_event(
                LOGGER,
                logging.INFO,
                "parse.dry_run",
                "Corpus parse dry-run completed.",
                stage="ingestion",
                planned_paths=planned_paths,
            )
            return result

        if not self.paths.documents_dir.exists():
            raise ValueError(
                f"Documents directory does not exist: {self.paths.documents_dir}. "
                "Run sync-phase first or pass a config with extracted PDFs."
            )

        selected_doc_ids = {doc_id for doc_id in (doc_ids or []) if doc_id}
        pdf_paths = sorted(self.paths.documents_dir.rglob("*.pdf"))
        if selected_doc_ids:
            pdf_paths = [path for path in pdf_paths if path.stem in selected_doc_ids]
        if not pdf_paths:
            raise ValueError(f"No PDFs found for parsing in {self.paths.documents_dir}")

        current_doc_ids = {path.stem for path in pdf_paths}
        if not overwrite and self._artifact_set_complete(current_doc_ids):
            result.skipped_paths.extend(self._artifact_paths())
            log_event(
                LOGGER,
                logging.INFO,
                "parse.skip_cache",
                "Skipped corpus parse because artifacts are complete.",
                stage="ingestion",
                document_count=len(current_doc_ids),
                skipped_paths=result.skipped_paths,
            )
            return result

        self.paths.ensure_directories()

        successful_documents: list[ParsedDocument] = []
        serialized_tables_by_doc: dict[str, dict[int, list[TableBlock]]] = {}
        structural_blocks_by_doc: dict[str, dict[int, list[NormalizedStructuralBlock]]] = {}
        manifest_records: list[CorpusParseManifestRecord] = []
        failed_document_count = 0

        for pdf_path in pdf_paths:
            try:
                if documents_debug_enabled():
                    log_event(
                        LOGGER,
                        logging.DEBUG,
                        "parse.document.start",
                        "Parsing document.",
                        stage="ingestion",
                        doc_id=pdf_path.stem,
                        pdf_path=pdf_path,
                    )
                document = self.parser.parse_document(pdf_path)
                successful_documents.append(document)
                serialized_tables_by_doc[document.doc_id] = serialize_document_tables(
                    document,
                    enable_semantic_blocks=self.config.ingestion.enable_semantic_table_blocks,
                    structured_provider=self._get_structured_provider(),
                )
                self._write_raw_parse_sidecars(document)
                structural_blocks = self._structural_blocks_for_document(document)
                if structural_blocks:
                    structural_blocks_by_doc[document.doc_id] = structural_blocks
                if documents_debug_enabled():
                    log_event(
                        LOGGER,
                        logging.DEBUG,
                        "parse.document.complete",
                        "Parsed document.",
                        stage="ingestion",
                        doc_id=document.doc_id,
                        page_count=document.page_count,
                        parse_status=document.parse_status.value,
                    )
            except Exception as exc:
                failed_document_count += 1
                manifest_records.append(build_failed_manifest_record(pdf_path, str(exc)))
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "parse.document.failed",
                    "Document parsing failed.",
                    stage="ingestion",
                    doc_id=pdf_path.stem,
                    pdf_path=pdf_path,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                raise DocumentParseFailureError(
                    doc_id=pdf_path.stem,
                    pdf_path=pdf_path,
                    cause=exc,
                ) from exc

        build_result = self.builder.build(
            phase=self.config.phase.value,
            documents=successful_documents,
            serialized_tables_by_doc=serialized_tables_by_doc,
            structural_blocks_by_doc=structural_blocks_by_doc,
        )
        page_records = list(build_result.corpus_records)
        page_map_records = list(build_result.page_map_records)
        success_manifest_records = list(build_result.manifest.records)
        debug_markdown = dict(build_result.debug_markdown)

        manifest = CorpusParseManifest(
            phase=self.config.phase,
            parsed_at=datetime.now(tz=UTC),
            document_count=len(success_manifest_records) + failed_document_count,
            page_count=len(page_records),
            failed_document_count=failed_document_count,
            records=[*success_manifest_records, *manifest_records],
        )

        self._write_jsonl(self.paths.corpus_path, page_records)
        self._write_json(self.paths.page_map_path, page_map_records)
        self._write_json(self.paths.corpus_manifest_path, manifest)
        self._write_debug_files(debug_markdown)

        result.written_paths.extend(self._artifact_paths())
        result.parsed_document_count = len(success_manifest_records)
        result.parsed_page_count = len(page_records)
        result.failed_document_count = failed_document_count
        log_event(
            LOGGER,
            logging.INFO,
            "parse.complete",
            "Corpus parse completed.",
            stage="ingestion",
            parsed_document_count=result.parsed_document_count,
            parsed_page_count=result.parsed_page_count,
            failed_document_count=result.failed_document_count,
            written_paths=result.written_paths,
        )
        return result

    def _artifact_paths(self) -> list[Path]:
        """Execute `_artifact_paths`.

        Returns:
            Any: The computed result of the function.
        """
        return [
            self.paths.corpus_path,
            self.paths.page_map_path,
            self.paths.corpus_manifest_path,
            self.paths.debug_dir,
            self.paths.raw_parse_dir,
        ]

    def _artifact_set_complete(self, current_doc_ids: set[str]) -> bool:
        """Execute `_artifact_set_complete`.

        Args:
            current_doc_ids: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        if any(not path.exists() for path in self._artifact_paths()):
            return False

        try:
            manifest = CorpusParseManifest.model_validate_json(
                self.paths.corpus_manifest_path.read_text(encoding="utf-8")
            )
        except Exception:
            return False

        manifest_doc_ids = {record.doc_id for record in manifest.records}
        if manifest_doc_ids != current_doc_ids:
            return False

        required_doc_ids = {
            record.doc_id
            for record in manifest.records
            if record.parse_status is not ParseStatus.FAILED
            and record.doc_id in current_doc_ids
        }
        return all(self._document_artifacts_complete(doc_id) for doc_id in required_doc_ids)

    def _write_jsonl(self, path: Path, records: Iterable[CanonicalPageRecord]) -> None:
        """Execute `_write_jsonl`.

        Args:
            path: Input parameter.
            records: Input parameter.

        Returns:
            None: This function does not return a value.
        """
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
                handle.write("\n")

    def _write_json(self, path: Path, payload: CorpusParseManifest | list[PageMapRecord]) -> None:
        """Execute `_write_json`.

        Args:
            path: Input parameter.
            payload: Input parameter.

        Returns:
            None: This function does not return a value.
        """
        with path.open("w", encoding="utf-8") as handle:
            if isinstance(payload, CorpusParseManifest):
                serialized = payload.model_dump(mode="json")
            else:
                serialized = [item.model_dump(mode="json") for item in payload]
            json.dump(serialized, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _write_debug_files(self, debug_markdown: dict[str, str]) -> None:
        """Execute `_write_debug_files`.

        Args:
            debug_markdown: Input parameter.

        Returns:
            None: This function does not return a value.
        """
        for existing_file in self.paths.debug_dir.glob("*.md"):
            existing_file.unlink()
        for doc_id, markdown in debug_markdown.items():
            target = self.paths.debug_dir / f"{doc_id}.md"
            with target.open("w", encoding="utf-8") as handle:
                handle.write(markdown)
                if not markdown.endswith("\n"):
                    handle.write("\n")

    def _write_raw_parse_sidecars(self, document: ParsedDocument) -> None:
        """Execute `_write_raw_parse_sidecars`.

        Args:
            document: Input parameter.

        Returns:
            None: This function does not return a value.
        """
        if not document.raw_parse_sidecars:
            return
        for parser_name, sidecar in document.raw_parse_sidecars.items():
            write_raw_parse_sidecar(self.paths.raw_parse_dir / parser_name, sidecar)

    def _document_artifacts_complete(self, doc_id: str) -> bool:
        """Execute `_document_artifacts_complete`.

        Args:
            doc_id: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        debug_path = self.paths.debug_dir / f"{doc_id}.md"
        pymupdf_sidecar_path = self.paths.raw_parse_dir / "pymupdf" / f"{doc_id}.json"
        if not debug_path.exists() or not pymupdf_sidecar_path.exists():
            return False

        try:
            load_raw_parse_sidecar(pymupdf_sidecar_path)
        except Exception:
            return False

        return (self.paths.raw_parse_dir / "opendataloader" / f"{doc_id}.json").exists()

    def _build_structured_provider(self) -> StructuredOutputProvider | None:
        """Build structured provider.

        Returns:
            Any: The computed result of the function.
        """
        if not self.config.ingestion.enable_semantic_table_blocks:
            return None

        try:
            return CodexStructuredOutputProvider(api_key=_resolve_provider_api_key(self.env_file))
        except ValueError as exc:
            log_event(
                LOGGER,
                logging.ERROR,
                "parse.semantic_provider_unavailable",
                "Semantic table blocks are enabled but no structured provider is available. Failing parse.",
                stage="ingestion",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise ValueError(
                "Semantic table blocks require a configured structured provider. "
                "Set CODEX_OAUTH_TOKEN or OPENAI_API_KEY before parsing."
            ) from exc

    def _get_structured_provider(self) -> StructuredOutputProvider | None:
        """Return structured provider.

        Returns:
            Any: The computed result of the function.
        """
        if not self.config.ingestion.enable_semantic_table_blocks:
            return None
        if self._structured_provider is None:
            self._structured_provider = self._build_structured_provider()
        return self._structured_provider

    def _structural_blocks_for_document(
        self,
        document: ParsedDocument,
    ) -> dict[int, list[NormalizedStructuralBlock]]:
        """Execute `_structural_blocks_for_document`.

        Args:
            document: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        sidecar = document.raw_parse_sidecars.get("opendataloader")
        if sidecar is None:
            return {}
        by_page: dict[int, list[NormalizedStructuralBlock]] = {}
        for block in normalize_structural_sidecar(sidecar):
            by_page.setdefault(block.page_number, []).append(block)
        return by_page


def format_parse_result(result: ParseCorpusResult) -> str:
    """Return a human-readable parse-corpus CLI summary."""

    lines = [
        f"command: {result.command}",
        f"phase: {result.phase}",
        f"dry_run: {str(result.dry_run).lower()}",
        f"manifest: {result.manifest_path}",
    ]
    if result.parsed_document_count is not None:
        lines.append(f"parsed_document_count: {result.parsed_document_count}")
    if result.parsed_page_count is not None:
        lines.append(f"parsed_page_count: {result.parsed_page_count}")
    if result.failed_document_count is not None:
        lines.append(f"failed_document_count: {result.failed_document_count}")
    if result.planned_paths:
        lines.append("planned_paths:")
        lines.extend(f"  - {path}" for path in result.planned_paths)
    if result.written_paths:
        lines.append("written_paths:")
        lines.extend(f"  - {path}" for path in result.written_paths)
    if result.skipped_paths:
        lines.append("skipped_paths:")
        lines.extend(f"  - {path}" for path in result.skipped_paths)
    return "\n".join(lines)


def _resolve_provider_api_key(env_file: Path) -> str | None:
    """Resolve provider api key.

    Args:
        env_file: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    env_values = dotenv_values(env_file) if env_file.exists() else {}
    for key in ("CODEX_OAUTH_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"):
        # Process env takes precedence per key.
        from_env = os.getenv(key)
        if from_env is not None:
            token = from_env.strip()
            if token:
                return token
            continue
        value = env_values.get(key)
        if value:
            token = str(value).strip()
            if token:
                return token
    return None
