"""Phase-aware local storage and sync orchestration."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

from src.common.config import AppConfig
from src.common.file_hash import sha256_for_file
from src.common.runtime_logging import get_logger, log_event
from src.common.schemas import ArtifactMetadata, PhaseSyncManifest, QuestionRecord
from src.submission.api_client import CompetitionApiClient

LOGGER = get_logger("submission.sync")


def _resolve_phase_dir(*, root_dir: Path, phase_value: str) -> Path:
    """Resolve phase directory with compatibility fallback for `final` -> `full`.

    Args:
        root_dir: Base storage directory.
        phase_value: Configured phase value (`warmup` or `final`).

    Returns:
        Path: Existing phase directory when found; otherwise the canonical phase path.
    """
    canonical_dir = root_dir / phase_value
    if canonical_dir.exists():
        return canonical_dir
    if phase_value == "final":
        legacy_full_dir = root_dir / "full"
        if legacy_full_dir.exists():
            return legacy_full_dir
    return canonical_dir


@dataclass(slots=True)
class PhasePaths:
    """Resolved local filesystem layout for a phase."""

    phase_dir: Path
    questions_dir: Path
    questions_path: Path
    documents_dir: Path
    documents_archive_path: Path
    documents_extract_dir: Path
    corpus_dir: Path
    corpus_path: Path
    page_map_path: Path
    corpus_manifest_path: Path
    debug_dir: Path
    manifests_dir: Path
    manifest_path: Path

    @classmethod
    def from_config(cls, config: AppConfig) -> "PhasePaths":
        """Execute `from_config`.

        Args:
            config: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        phase_dir = _resolve_phase_dir(root_dir=config.storage.root_dir, phase_value=config.phase.value)
        questions_dir = phase_dir / config.storage.questions_dirname
        documents_dir = phase_dir / config.storage.documents_dirname
        corpus_dir = phase_dir / config.storage.corpus_dirname
        manifests_dir = phase_dir / config.storage.manifests_dirname

        return cls(
            phase_dir=phase_dir,
            questions_dir=questions_dir,
            questions_path=questions_dir / config.storage.questions_filename,
            documents_dir=documents_dir,
            documents_archive_path=documents_dir / config.storage.documents_archive_name,
            documents_extract_dir=documents_dir / config.storage.extracted_documents_dirname,
            corpus_dir=corpus_dir,
            corpus_path=corpus_dir / config.storage.corpus_filename,
            page_map_path=corpus_dir / config.storage.page_map_filename,
            corpus_manifest_path=corpus_dir / config.storage.corpus_manifest_filename,
            debug_dir=corpus_dir / config.storage.debug_dirname,
            manifests_dir=manifests_dir,
            manifest_path=manifests_dir / config.storage.manifest_filename,
        )

    def ensure_directories(self) -> None:
        """Execute `ensure_directories`.

        Returns:
            None: This function does not return a value.
        """
        self.questions_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def ensure_corpus_directories(self) -> None:
        """Execute `ensure_corpus_directories`.

        Returns:
            None: This function does not return a value.
        """
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class CommandResult:
    """User-facing summary for a CLI action."""

    command: str
    phase: str
    dry_run: bool
    manifest_path: Path
    planned_paths: list[Path] = field(default_factory=list)
    written_paths: list[Path] = field(default_factory=list)
    skipped_paths: list[Path] = field(default_factory=list)
    questions_count: int | None = None
    extracted_pdf_count: int | None = None
    parsed_document_count: int | None = None
    parsed_page_count: int | None = None
    failed_document_count: int | None = None


class PhaseSyncService:
    """Orchestrates local phase-isolated download flows."""

    def __init__(self, config: AppConfig) -> None:
        """Execute `__init__`.

        Args:
            config: Input parameter.

        Returns:
            None: This initializer does not return a value.
        """
        self.config = config
        self.paths = PhasePaths.from_config(config)

    def download_questions(
        self,
        *,
        client: CompetitionApiClient | None,
        dry_run: bool = False,
        overwrite: bool | None = None,
    ) -> CommandResult:
        """Execute `download_questions`.

        Args:
            client: Input parameter.
            dry_run: Input parameter.
            overwrite: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        overwrite = self.config.download.overwrite if overwrite is None else overwrite
        log_event(
            LOGGER,
            logging.INFO,
            "sync.questions.start",
            "Starting questions sync.",
            stage="sync",
            dry_run=dry_run,
            overwrite=overwrite,
            target_path=self.paths.questions_path,
        )

        result = CommandResult(
            command="download-questions",
            phase=self.config.phase.value,
            dry_run=dry_run,
            manifest_path=self.paths.manifest_path,
            planned_paths=[self.paths.questions_path, self.paths.manifest_path],
        )

        if dry_run:
            log_event(
                LOGGER,
                logging.INFO,
                "sync.questions.dry_run",
                "Questions sync dry-run completed.",
                stage="sync",
                planned_paths=result.planned_paths,
            )
            return result

        self.paths.ensure_directories()

        if self.paths.questions_path.exists() and not overwrite:
            result.skipped_paths.append(self.paths.questions_path)
            manifest = self._write_manifest(["download-questions", "reuse-questions"])
            result.written_paths.append(self.paths.manifest_path)
            result.questions_count = manifest.questions_count
            log_event(
                LOGGER,
                logging.INFO,
                "sync.questions.reuse",
                "Reused existing questions file.",
                stage="sync",
                questions_count=manifest.questions_count,
                skipped_path=self.paths.questions_path,
            )
            return result

        if client is None:
            raise ValueError("A live API client is required unless dry_run=True.")

        questions = client.download_questions()
        self._write_questions_file(questions)

        manifest = self._write_manifest(["download-questions"])
        result.written_paths.extend([self.paths.questions_path, self.paths.manifest_path])
        result.questions_count = manifest.questions_count
        log_event(
            LOGGER,
            logging.INFO,
            "sync.questions.complete",
            "Questions sync completed.",
            stage="sync",
            questions_count=manifest.questions_count,
            written_paths=result.written_paths,
        )
        return result

    def download_documents(
        self,
        *,
        client: CompetitionApiClient | None,
        dry_run: bool = False,
        overwrite: bool | None = None,
        extract_documents: bool | None = None,
    ) -> CommandResult:
        """Execute `download_documents`.

        Args:
            client: Input parameter.
            dry_run: Input parameter.
            overwrite: Input parameter.
            extract_documents: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        overwrite = self.config.download.overwrite if overwrite is None else overwrite
        extract_documents = (
            self.config.download.extract_documents if extract_documents is None else extract_documents
        )
        log_event(
            LOGGER,
            logging.INFO,
            "sync.documents.start",
            "Starting documents sync.",
            stage="sync",
            dry_run=dry_run,
            overwrite=overwrite,
            extract_documents=extract_documents,
            archive_path=self.paths.documents_archive_path,
            extract_dir=self.paths.documents_extract_dir,
        )

        planned_paths = [self.paths.documents_archive_path, self.paths.manifest_path]
        if extract_documents:
            planned_paths.append(self.paths.documents_extract_dir)

        result = CommandResult(
            command="download-documents",
            phase=self.config.phase.value,
            dry_run=dry_run,
            manifest_path=self.paths.manifest_path,
            planned_paths=planned_paths,
        )

        if dry_run:
            log_event(
                LOGGER,
                logging.INFO,
                "sync.documents.dry_run",
                "Documents sync dry-run completed.",
                stage="sync",
                planned_paths=result.planned_paths,
            )
            return result

        self.paths.ensure_directories()

        archive_downloaded = False
        if self.paths.documents_archive_path.exists() and not overwrite:
            result.skipped_paths.append(self.paths.documents_archive_path)
            log_event(
                LOGGER,
                logging.INFO,
                "sync.documents.reuse_archive",
                "Reused existing documents archive.",
                stage="sync",
                archive_path=self.paths.documents_archive_path,
            )
        else:
            if client is None:
                raise ValueError("A live API client is required unless dry_run=True.")
            with self.paths.documents_archive_path.open("wb") as archive_handle:
                client.stream_documents(
                    archive_handle,
                    chunk_size=self.config.download.chunk_size_bytes,
                )
            archive_downloaded = True
            result.written_paths.append(self.paths.documents_archive_path)

        extracted_pdf_count = None
        if extract_documents:
            extracted_pdf_count = self._extract_documents(overwrite=overwrite or archive_downloaded)
            result.written_paths.append(self.paths.documents_extract_dir)
        elif self.paths.documents_extract_dir.exists():
            shutil.rmtree(self.paths.documents_extract_dir)

        manifest = self._write_manifest(["download-documents"])
        result.written_paths.append(self.paths.manifest_path)
        result.extracted_pdf_count = (
            extracted_pdf_count if extracted_pdf_count is not None else manifest.extracted_pdf_count
        )
        log_event(
            LOGGER,
            logging.INFO,
            "sync.documents.complete",
            "Documents sync completed.",
            stage="sync",
            extracted_pdf_count=result.extracted_pdf_count,
            written_paths=result.written_paths,
            skipped_paths=result.skipped_paths,
        )
        return result

    def sync_phase(
        self,
        *,
        client: CompetitionApiClient | None,
        dry_run: bool = False,
        overwrite: bool | None = None,
        extract_documents: bool | None = None,
    ) -> CommandResult:
        """Execute `sync_phase`.

        Args:
            client: Input parameter.
            dry_run: Input parameter.
            overwrite: Input parameter.
            extract_documents: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        log_event(
            LOGGER,
            logging.INFO,
            "sync.phase.start",
            "Starting phase sync.",
            stage="sync",
            dry_run=dry_run,
            overwrite=overwrite,
            extract_documents=extract_documents,
        )
        questions_result = self.download_questions(client=client, dry_run=dry_run, overwrite=overwrite)
        documents_result = self.download_documents(
            client=client,
            dry_run=dry_run,
            overwrite=overwrite,
            extract_documents=extract_documents,
        )

        written_paths = _dedupe_paths([*questions_result.written_paths, *documents_result.written_paths])
        skipped_paths = _dedupe_paths([*questions_result.skipped_paths, *documents_result.skipped_paths])
        planned_paths = _dedupe_paths([*questions_result.planned_paths, *documents_result.planned_paths])

        result = CommandResult(
            command="sync",
            phase=self.config.phase.value,
            dry_run=dry_run,
            manifest_path=self.paths.manifest_path,
            planned_paths=planned_paths,
            written_paths=written_paths,
            skipped_paths=skipped_paths,
            questions_count=questions_result.questions_count,
            extracted_pdf_count=documents_result.extracted_pdf_count,
        )

        if not dry_run:
            manifest = self._write_manifest(["download-questions", "download-documents", "sync"])
            if self.paths.manifest_path not in result.written_paths:
                result.written_paths.append(self.paths.manifest_path)
            result.questions_count = manifest.questions_count
            result.extracted_pdf_count = manifest.extracted_pdf_count

        log_event(
            LOGGER,
            logging.INFO,
            "sync.phase.complete",
            "Phase sync completed.",
            stage="sync",
            questions_count=result.questions_count,
            extracted_pdf_count=result.extracted_pdf_count,
            written_paths=result.written_paths,
            skipped_paths=result.skipped_paths,
        )
        return result

    def _write_questions_file(self, questions: list[QuestionRecord]) -> None:
        """Execute `_write_questions_file`.

        Args:
            questions: Input parameter.

        Returns:
            None: This function does not return a value.
        """
        payload = [question.model_dump(mode="json") for question in questions]
        with self.paths.questions_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _extract_documents(self, *, overwrite: bool) -> int:
        """Extract documents.

        Args:
            overwrite: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        if overwrite and self.paths.documents_extract_dir.exists():
            shutil.rmtree(self.paths.documents_extract_dir)

        self.paths.documents_extract_dir.mkdir(parents=True, exist_ok=True)

        with ZipFile(self.paths.documents_archive_path) as archive:
            pdf_count = 0
            target_root = self.paths.documents_extract_dir.resolve()
            for member in archive.infolist():
                destination = (self.paths.documents_extract_dir / member.filename).resolve()
                try:
                    destination.relative_to(target_root)
                except ValueError as exc:
                    raise ValueError(f"Unsafe archive member path: {member.filename}") from exc
                archive.extract(member, path=self.paths.documents_extract_dir)
                if not member.is_dir() and Path(member.filename).suffix.lower() == ".pdf":
                    pdf_count += 1

        return pdf_count

    def _write_manifest(self, operations: list[str]) -> PhaseSyncManifest:
        """Execute `_write_manifest`.

        Args:
            operations: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        manifest = self._build_manifest(operations=operations)
        with self.paths.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return manifest

    def _build_manifest(self, *, operations: list[str]) -> PhaseSyncManifest:
        """Build manifest.

        Args:
            operations: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        questions_count = self._count_questions()
        extracted_pdf_count = self._count_pdfs()
        artifacts: dict[str, ArtifactMetadata] = {}

        if self.paths.questions_path.exists():
            artifacts["questions_json"] = _build_file_metadata(self.paths.questions_path)

        if self.paths.documents_archive_path.exists():
            artifacts["documents_zip"] = _build_file_metadata(self.paths.documents_archive_path)

        if self.paths.documents_extract_dir.exists():
            artifacts["extracted_pdfs"] = _build_directory_metadata(self.paths.documents_extract_dir, "*.pdf")

        notes = [
            (
                "Phase selection is local-only for download storage because /questions and /documents "
                "do not accept a phase parameter."
            ),
        ]

        return PhaseSyncManifest(
            phase=self.config.phase,
            synced_at=datetime.now(tz=UTC),
            base_url=self.config.api.base_url,
            api_key_env=self.config.api.api_key_env,
            operations=operations,
            questions_count=questions_count,
            extracted_pdf_count=extracted_pdf_count,
            artifacts=artifacts,
            notes=notes,
        )

    def _count_questions(self) -> int | None:
        """Execute `_count_questions`.

        Returns:
            Any: The computed result of the function.
        """
        if not self.paths.questions_path.exists():
            return None

        with self.paths.questions_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, list):
            return None

        return len(payload)

    def _count_pdfs(self) -> int | None:
        """Execute `_count_pdfs`.

        Returns:
            Any: The computed result of the function.
        """
        if not self.paths.documents_extract_dir.exists():
            return None
        return sum(1 for path in self.paths.documents_extract_dir.rglob("*.pdf") if path.is_file())


def _build_file_metadata(path: Path) -> ArtifactMetadata:
    """Build file metadata.

    Args:
        path: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    stat = path.stat()
    return ArtifactMetadata(
        kind="file",
        relative_path=str(path),
        sha256=sha256_for_file(path),
        size_bytes=stat.st_size,
        updated_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
    )


def _build_directory_metadata(path: Path, pattern: str) -> ArtifactMetadata:
    """Build directory metadata.

    Args:
        path: Input parameter.
        pattern: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    files = [candidate for candidate in path.rglob(pattern) if candidate.is_file()]
    updated_at = None
    if files:
        latest_mtime = max(file_path.stat().st_mtime for file_path in files)
        updated_at = datetime.fromtimestamp(latest_mtime, tz=UTC)

    return ArtifactMetadata(
        kind="directory",
        relative_path=str(path),
        file_count=len(files),
        updated_at=updated_at,
    )


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    """Execute `_dedupe_paths`.

    Args:
        paths: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped
