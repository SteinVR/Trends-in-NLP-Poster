"""Submission payload and code-archive packaging utilities."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from src.submission.submit_gate import MAX_CODE_ARCHIVE_BYTES

_EXCLUDED_ROOT_NAMES = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".apm",
    "__pycache__",
    "data",
    "logs",
}
_SECRET_FILE_NAMES = {".env"}
_SECRET_FILE_PREFIXES = (".env.",)


@dataclass(slots=True)
class SubmissionPackager:
    """Create phase output artifacts for submission and code archive."""

    max_code_archive_bytes: int = MAX_CODE_ARCHIVE_BYTES

    def write_submission_payload(self, *, submission_payload: dict[str, Any], output_dir: str | Path) -> Path:
        """Execute `write_submission_payload`.

        Args:
            submission_payload: Input parameter.
            output_dir: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        submission_path = target_dir / "submission.json"
        with submission_path.open("w", encoding="utf-8") as handle:
            json.dump(submission_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return submission_path

    def create_code_archive(
        self,
        *,
        output_path: str | Path,
        project_root: str | Path,
        excluded_paths: set[Path] | None = None,
        include_paths: Iterable[str | Path] | None = None,
    ) -> Path:
        """Execute `create_code_archive`.

        Args:
            output_path: Input parameter.
            project_root: Input parameter.
            excluded_paths: Input parameter.
            include_paths: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        archive_path = Path(output_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        root = Path(project_root).resolve()
        normalized_include_paths = _normalize_include_paths(root=root, include_paths=include_paths)

        excluded = {path.resolve() for path in (excluded_paths or set())}
        excluded.add(archive_path.resolve())

        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            for file_path in _iter_archive_files(
                root=root,
                excluded_paths=excluded,
                include_paths=normalized_include_paths,
            ):
                archive.write(file_path, arcname=file_path.relative_to(root))

        _validate_archive_size(
            archive_path=archive_path,
            max_code_archive_bytes=self.max_code_archive_bytes,
        )

        return archive_path


def build_submission_payload(*, architecture_summary: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the final submission JSON payload."""

    return {
        "architecture_summary": architecture_summary,
        "answers": list(answers),
    }


def _is_excluded(path: Path, *, root: Path, excluded_paths: set[Path]) -> bool:
    """Return whether excluded is true.

    Args:
        path: Input parameter.
        root: Input parameter.
        excluded_paths: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if any(path == excluded or _is_relative_to(path, excluded) for excluded in excluded_paths):
        return True

    if _is_secret_file(path.name):
        return True

    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        # Symlink-resolved targets may point outside the project root.
        return True
    if not relative_parts:
        return False

    return relative_parts[0] in _EXCLUDED_ROOT_NAMES


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether relative to is true.

    Args:
        path: Input parameter.
        parent: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _iter_archive_files(
    *,
    root: Path,
    excluded_paths: set[Path],
    include_paths: list[Path] | None,
):
    """Execute `_iter_archive_files`.

    Args:
        root: Input parameter.
        excluded_paths: Input parameter.
        include_paths: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    seen_files: set[Path] = set()

    if include_paths is None:
        candidates = sorted(root.rglob("*"))
    else:
        candidates = []
        for include_path in include_paths:
            if include_path.is_file():
                candidates.append(include_path)
                continue
            if include_path.is_dir():
                candidates.extend(sorted(include_path.rglob("*")))

    for file_path in candidates:
        if not file_path.is_file():
            continue
        resolved_file = file_path.resolve()
        if resolved_file in seen_files:
            continue
        seen_files.add(resolved_file)
        if _is_excluded(resolved_file, root=root, excluded_paths=excluded_paths):
            continue
        yield resolved_file


def _is_secret_file(name: str) -> bool:
    """Return whether secret file is true.

    Args:
        name: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if name in _SECRET_FILE_NAMES:
        return True
    return any(name.startswith(prefix) for prefix in _SECRET_FILE_PREFIXES)


def _normalize_include_paths(*, root: Path, include_paths: Iterable[str | Path] | None) -> list[Path] | None:
    """Normalize include paths.

    Args:
        root: Input parameter.
        include_paths: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if include_paths is None:
        return None

    normalized_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for include_path in include_paths:
        candidate = Path(include_path)
        resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if not _is_relative_to(resolved, root):
            continue
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        normalized_paths.append(resolved)
    return sorted(normalized_paths)


def _validate_archive_size(*, archive_path: Path, max_code_archive_bytes: int) -> None:
    """Validate archive size.

    Args:
        archive_path: Input parameter.
        max_code_archive_bytes: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    if archive_path.stat().st_size > max_code_archive_bytes:
        max_mb = max_code_archive_bytes // (1024 * 1024)
        raise ValueError(f"Code archive size exceeds {max_mb} MB limit: {archive_path}")
