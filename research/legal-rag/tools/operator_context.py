"""Resolve config-aware operator paths for numbered shell scripts."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from dotenv import dotenv_values

from src.common.config import load_app_config
from src.evaluation.contracts import AnswerType, load_question_records
from src.submission.sync import PhasePaths

SCHEMA_FIRST_PROVIDER_TYPES = frozenset(answer_type.value for answer_type in AnswerType)


def _resolve_provider_api_key_source(env_file: Path) -> str:
    env_values = dotenv_values(env_file) if env_file.exists() else {}
    for key in ("CODEX_OAUTH_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"):
        process_value = os.environ.get(key, "").strip()
        if process_value:
            return key
        file_value = str(env_values.get(key) or "").strip()
        if file_value:
            return key
    return ""


def _question_set_requires_provider(questions_path: Path) -> bool:
    if not questions_path.exists():
        return False
    try:
        questions = load_question_records(questions_path)
    except (OSError, ValueError):
        return False
    return any(str(question.answer_type) in SCHEMA_FIRST_PROVIDER_TYPES for question in questions)


def _build_context(*, config_path: Path, env_file: Path, phase_override: str | None) -> dict[str, str]:
    config = load_app_config(config_path, env_file=env_file, phase_override=phase_override)
    storage = config.storage
    phase = config.phase.value

    phase_paths = PhasePaths.from_config(config)
    phase_dir = phase_paths.phase_dir
    questions_path = phase_paths.questions_path
    documents_dir = phase_paths.documents_dir
    documents_extract_dir = phase_paths.documents_extract_dir
    corpus_dir = phase_paths.corpus_dir
    index_dir = phase_dir / "index"
    submission_dir = phase_dir / "submission"

    provider_key_source = _resolve_provider_api_key_source(env_file)
    question_set_requires_provider = _question_set_requires_provider(questions_path)

    return {
        "PHASE": phase,
        "PHASE_DIR": str(phase_dir),
        "QUESTIONS_PATH": str(questions_path),
        "DOCUMENTS_DIR": str(documents_extract_dir),
        "DOCUMENTS_ARCHIVE_PATH": str(documents_dir / storage.documents_archive_name),
        "CORPUS_DIR": str(corpus_dir),
        "CORPUS_PATH": str(corpus_dir / storage.corpus_filename),
        "PAGE_MAP_PATH": str(corpus_dir / storage.page_map_filename),
        "CORPUS_MANIFEST_PATH": str(corpus_dir / storage.corpus_manifest_filename),
        "DEBUG_DIR": str(corpus_dir / storage.debug_dirname),
        "INDEX_DIR": str(index_dir),
        "INDEX_MANIFEST_PATH": str(index_dir / "index_manifest.json"),
        "PAGE_PARENT_MAP_PATH": str(index_dir / "page_parent_map.json"),
        "SPARSE_ENCODER_STATE_PATH": str(index_dir / "sparse_encoder_state.json"),
        "QDRANT_DIR": str(index_dir / "qdrant"),
        "SUBMISSION_DIR": str(submission_dir),
        "SUBMISSION_PATH": str(submission_dir / "submission.json"),
        "TRACE_MANIFEST_PATH": str(submission_dir / "trace_manifest.json"),
        "TRACES_DIR": str(submission_dir / "traces"),
        "QUESTION_SET_REQUIRES_PROVIDER": "1" if question_set_requires_provider else "0",
        "PROVIDER_API_KEY_PRESENT": "1" if provider_key_source else "0",
        "PROVIDER_API_KEY_SOURCE": provider_key_source,
    }


def _emit_shell(context: dict[str, str]) -> str:
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in context.items())


def _emit_json(context: dict[str, str]) -> str:
    import json

    return json.dumps(context, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve config-aware operator paths for shell scripts.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--env-file", default=".env", help="Path to the env file.")
    parser.add_argument("--phase", help="Optional phase override.")
    parser.add_argument(
        "--format",
        choices=("shell", "json"),
        default="shell",
        help="Output format for resolved values.",
    )
    args = parser.parse_args()

    context = _build_context(
        config_path=Path(args.config),
        env_file=Path(args.env_file),
        phase_override=args.phase,
    )
    if args.format == "json":
        print(_emit_json(context))
    else:
        print(_emit_shell(context))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
