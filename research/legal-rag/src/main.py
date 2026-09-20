"""CLI entrypoint for configuration-driven phase sync commands."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Sequence

import httpx

from src.common.config import DEFAULT_CONFIG_PATH, AppConfig, ConfigError, load_app_config
from src.common.runtime_logging import (
    configure_runtime_logging,
    get_logger,
    log_event,
    runtime_log_path,
    shutdown_runtime_logging,
)
from src.main_handlers import _run_command
from src.main_parser_helpers import (
    _add_analyze_benchmark_consistency_command,
    _add_benchmark_summary_command,
    _add_build_indices_command,
    _add_build_submission_command,
    _add_compare_benchmark_runs_command,
    _add_download_documents_command,
    _add_download_questions_command,
    _add_parse_corpus_command,
    _add_score_submission_command,
    _add_submission_status_command,
    _add_submit_command,
    _add_sync_command,
    _add_validate_submission_command,
)
from src.submission.api_client import CompetitionApiClient

ClientFactory = Callable[[AppConfig], CompetitionApiClient]
LOGGER = get_logger("main")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser and register all subcommands."""

    parser = argparse.ArgumentParser(description="Agentic RAG Challenge CLI")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file.",
    )
    parser.add_argument("--env-file", default=".env", help="Path to the env file used for API credentials.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_download_questions_command(subparsers)
    _add_download_documents_command(subparsers)
    _add_sync_command(subparsers)
    _add_parse_corpus_command(subparsers)
    _add_build_indices_command(subparsers)
    _add_build_submission_command(subparsers)
    _add_submit_command(subparsers)
    _add_submission_status_command(subparsers)
    _add_validate_submission_command(subparsers)
    _add_score_submission_command(subparsers)
    _add_benchmark_summary_command(subparsers)
    _add_compare_benchmark_runs_command(subparsers)
    _add_analyze_benchmark_consistency_command(subparsers)

    return parser


def main(argv: Sequence[str] | None = None, client_factory: ClientFactory | None = None) -> int:
    """Run one CLI command and return the process-style exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    result = ""

    try:
        phase_override = getattr(args, "phase", None)
        config = load_app_config(Path(args.config), env_file=Path(args.env_file), phase_override=phase_override)
        configure_runtime_logging(config, command=args.command, phase=config.phase.value)
        log_event(
            LOGGER,
            logging.INFO,
            "command.start",
            "Starting CLI command.",
            stage="cli",
            command_name=args.command,
            config_path=Path(args.config),
            env_file=Path(args.env_file),
        )
        result = _run_command(args, config, client_factory or CompetitionApiClient.from_config)
    except (ConfigError, ValueError, OSError, httpx.HTTPError) as exc:
        if runtime_log_path() is not None:
            log_event(
                LOGGER,
                logging.ERROR,
                "command.error",
                "CLI command failed.",
                stage="cli",
                command_name=getattr(args, "command", None),
                error_type=type(exc).__name__,
                error_message=str(exc),
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        parser.exit(status=1, message=f"error: {exc}\n")
        return 1
    else:
        log_event(
            LOGGER,
            logging.INFO,
            "command.success",
            "CLI command completed.",
            stage="cli",
            command_name=args.command,
            output_length=len(result),
            log_path=str(runtime_log_path()) if runtime_log_path() is not None else None,
        )
        print(result)
        return 0
    finally:
        shutdown_runtime_logging()


if __name__ == "__main__":
    raise SystemExit(main())
