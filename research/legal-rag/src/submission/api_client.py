"""Thin httpx-based client for the competition API."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from zipfile import ZipFile

import httpx

from src.common.config import AppConfig
from src.common.runtime_logging import get_logger, log_event
from src.common.schemas import QuestionRecord

LOGGER = get_logger("submission.api_client")


@dataclass(frozen=True)
class DocumentsDownloadResult:
    archive_path: Path
    extract_dir: Path
    document_files: list[str]


class CompetitionApiClient:
    """Minimal client for downloading phase assets and handling submissions."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 120.0,
        user_agent: str = "agentic-rag-challenge/0.1.0",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Execute `__init__`.

        Args:
            api_key: Input parameter.
            base_url: Input parameter.
            timeout_seconds: Input parameter.
            user_agent: Input parameter.
            transport: Input parameter.

        Returns:
            None: This initializer does not return a value.
        """
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "X-API-Key": api_key,
                "User-Agent": user_agent,
                "Accept": "application/json, application/zip",
            },
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
            transport=transport,
        )

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> "CompetitionApiClient":
        """Execute `from_config`.

        Args:
            config: Input parameter.
            transport: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        return cls(
            api_key=config.resolve_api_key(),
            base_url=config.api.base_url,
            timeout_seconds=config.api.timeout_seconds,
            user_agent=config.api.user_agent,
            transport=transport,
        )

    def close(self) -> None:
        """Execute `close`.

        Returns:
            None: This function does not return a value.
        """
        self._client.close()

    def __enter__(self) -> "CompetitionApiClient":
        """Execute `__enter__`.

        Returns:
            Any: The computed result of the function.
        """
        return self

    def __exit__(self, *_: object) -> None:
        """Execute `__exit__`.

        Args:
            *_: Input parameter.

        Returns:
            None: This function does not return a value.
        """
        self.close()

    def download_questions(self, target_path: str | Path | None = None) -> list[QuestionRecord]:
        """Execute `download_questions`.

        Args:
            target_path: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        log_event(
            LOGGER,
            logging.INFO,
            "api.questions.start",
            "Downloading questions from competition API.",
            stage="sync",
            target_path=target_path,
        )
        response = self._client.get("questions")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Expected /questions to return a JSON list.")
        questions = [QuestionRecord.model_validate(item) for item in payload]
        if target_path is not None:
            self._write_json(Path(target_path), [question.model_dump(mode="json") for question in questions])
        log_event(
            LOGGER,
            logging.INFO,
            "api.questions.complete",
            "Downloaded questions from competition API.",
            stage="sync",
            question_count=len(questions),
            target_path=target_path,
        )
        return questions

    def stream_documents(self, destination: BinaryIO, *, chunk_size: int = 1_048_576) -> None:
        """Execute `stream_documents`.

        Args:
            destination: Input parameter.
            chunk_size: Input parameter.

        Returns:
            None: This function does not return a value.
        """
        total_bytes = 0
        log_event(
            LOGGER,
            logging.INFO,
            "api.documents.start",
            "Streaming documents archive from competition API.",
            stage="sync",
            chunk_size=chunk_size,
        )
        with self._client.stream("GET", "documents") as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(chunk_size=chunk_size):
                if chunk:
                    destination.write(chunk)
                    total_bytes += len(chunk)
        log_event(
            LOGGER,
            logging.INFO,
            "api.documents.complete",
            "Documents archive stream completed.",
            stage="sync",
            bytes_downloaded=total_bytes,
        )

    def download_documents(self, zip_path: str | Path, extract_dir: str | Path) -> DocumentsDownloadResult:
        """Execute `download_documents`.

        Args:
            zip_path: Input parameter.
            extract_dir: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        archive_path = Path(zip_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with archive_path.open("wb") as archive_handle:
            self.stream_documents(archive_handle)

        documents_dir = Path(extract_dir)
        self._extract_archive(archive_path, documents_dir)

        result = DocumentsDownloadResult(
            archive_path=archive_path,
            extract_dir=documents_dir,
            document_files=sorted(str(path.relative_to(documents_dir)) for path in documents_dir.rglob("*.pdf")),
        )
        log_event(
            LOGGER,
            logging.INFO,
            "api.documents.extracted",
            "Extracted downloaded document archive.",
            stage="sync",
            archive_path=archive_path,
            extract_dir=documents_dir,
            extracted_pdf_count=len(result.document_files),
        )
        return result

    def submit_submission(self, submission_path: str | Path, code_archive_path: str | Path) -> dict[str, Any]:
        """Execute `submit_submission`.

        Args:
            submission_path: Input parameter.
            code_archive_path: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        submission = Path(submission_path)
        code_archive = Path(code_archive_path)
        started_at = time.perf_counter()
        log_event(
            LOGGER,
            logging.INFO,
            "api.submit.start",
            "Submitting bundle to competition API.",
            stage="submission",
            submission_path=submission,
            code_archive_path=code_archive,
        )

        with submission.open("rb") as submission_handle, code_archive.open("rb") as archive_handle:
            response = self._client.post(
                "submissions",
                files={
                    "file": (submission.name, submission_handle, "application/json"),
                    "code_archive": (code_archive.name, archive_handle, "application/zip"),
                },
            )

        response.raise_for_status()
        payload = response.json()
        duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
        log_event(
            LOGGER,
            logging.INFO,
            "api.submit.complete",
            "Submission accepted by competition API.",
            stage="submission",
            submission_uuid=payload.get("uuid"),
            status=payload.get("status"),
            duration_ms=duration_ms,
        )
        return payload

    def get_submission_status(self, submission_uuid: str) -> dict[str, Any]:
        """Return submission status.

        Args:
            submission_uuid: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        log_event(
            LOGGER,
            logging.INFO,
            "api.status.start",
            "Fetching submission status from competition API.",
            stage="submission",
            submission_uuid=submission_uuid,
        )
        response = self._client.get(f"submissions/{submission_uuid}/status")
        response.raise_for_status()
        payload = response.json()
        log_event(
            LOGGER,
            logging.INFO,
            "api.status.complete",
            "Fetched submission status from competition API.",
            stage="submission",
            submission_uuid=submission_uuid,
            status=payload.get("status"),
            review_status=payload.get("review_status"),
        )
        return payload

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        """Execute `_write_json`.

        Args:
            path: Input parameter.
            payload: Input parameter.

        Returns:
            None: This function does not return a value.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _extract_archive(archive_path: Path, extract_dir: Path) -> None:
        """Extract archive.

        Args:
            archive_path: Input parameter.
            extract_dir: Input parameter.

        Returns:
            None: This function does not return a value.
        """
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with ZipFile(archive_path) as archive:
            target_root = extract_dir.resolve()
            for member in archive.infolist():
                destination = (extract_dir / member.filename).resolve()
                try:
                    destination.relative_to(target_root)
                except ValueError as exc:
                    raise ValueError(f"Unsafe archive member path: {member.filename}") from exc
                archive.extract(member, path=extract_dir)
