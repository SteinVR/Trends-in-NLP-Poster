"""CLI parser subcommand registration helpers."""

from __future__ import annotations

import argparse

from src.common.schemas import Phase


def _add_download_questions_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Execute `_add_download_questions_command`.

    Args:
        subparsers: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    parser = subparsers.add_parser("download-questions", help="Download questions for the declared local phase.")
    parser.add_argument("--phase", choices=[phase.value for phase in Phase], help="Override the config phase.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and paths without making API calls.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing local questions JSON.")


def _add_download_documents_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Execute `_add_download_documents_command`.

    Args:
        subparsers: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    parser = subparsers.add_parser("download-documents", help="Download the documents ZIP for the declared phase.")
    parser.add_argument("--phase", choices=[phase.value for phase in Phase], help="Override the config phase.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and paths without making API calls.")
    parser.add_argument("--force", action="store_true", help="Overwrite the local documents ZIP and extracted PDFs.")
    parser.add_argument(
        "--extract",
        dest="extract_documents",
        action="store_true",
        default=None,
        help="Extract PDFs after downloading the ZIP.",
    )
    parser.add_argument(
        "--no-extract",
        dest="extract_documents",
        action="store_false",
        help="Keep only the ZIP archive without extracting PDFs.",
    )


def _add_sync_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Execute `_add_sync_command`.

    Args:
        subparsers: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    for command_name in ("sync", "sync-phase"):
        parser = subparsers.add_parser(command_name, help="Download questions and documents for the declared phase.")
        parser.add_argument("--phase", choices=[phase.value for phase in Phase], help="Override the config phase.")
        parser.add_argument("--dry-run", action="store_true", help="Resolve config and paths without making API calls.")
        parser.add_argument("--force", action="store_true", help="Overwrite existing local download artifacts.")
        parser.add_argument(
            "--extract",
            dest="extract_documents",
            action="store_true",
            default=None,
            help="Extract PDFs after downloading the ZIP.",
        )
        parser.add_argument(
            "--no-extract",
            dest="extract_documents",
            action="store_false",
            help="Keep only the ZIP archive without extracting PDFs.",
        )


def _add_submit_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Execute `_add_submit_command`.

    Args:
        subparsers: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    parser = subparsers.add_parser("submit", help="Submit submission.json and code archive to the platform.")
    parser.add_argument("--submission-path", required=True)
    parser.add_argument("--code-archive-path", required=True)


def _add_submission_status_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Execute `_add_submission_status_command`.

    Args:
        subparsers: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    parser = subparsers.add_parser("submission-status", help="Query submission processing status.")
    parser.add_argument("--submission-uuid", required=True)


def _add_parse_corpus_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Execute `_add_parse_corpus_command`.

    Args:
        subparsers: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    parser = subparsers.add_parser("parse-corpus", help="Parse extracted PDFs into canonical page-level corpus files.")
    parser.add_argument("--phase", choices=[phase.value for phase in Phase], help="Override the config phase.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve phase-local parse paths without writing files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing corpus artifacts.")
    parser.add_argument(
        "--doc-id",
        dest="doc_ids",
        action="append",
        help="Optional document ID filter. Pass multiple times to parse a subset.",
    )


def _add_build_indices_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Execute `_add_build_indices_command`.

    Args:
        subparsers: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    parser = subparsers.add_parser("build-indices", help="Build phase-local hybrid retrieval indices.")
    parser.add_argument("--phase", choices=[phase.value for phase in Phase], help="Override the config phase.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve index artifact paths without writing files.")
    parser.add_argument("--force", action="store_true", help="Rebuild index artifacts even when cache is valid.")


def _add_build_submission_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Execute `_add_build_submission_command`.

    Args:
        subparsers: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    parser = subparsers.add_parser(
        "build-submission",
        help="Run the full pipeline and create submission.json + code_archive.zip.",
    )
    parser.add_argument("--phase", choices=[phase.value for phase in Phase], help="Override the config phase.")
    parser.add_argument(
        "--questions-path",
        required=True,
        help="Path to questions.json used for pipeline execution and local validation.",
    )
    parser.add_argument(
        "--documents-dir",
        required=True,
        help="Path to extracted PDFs used by corpus/index/pipeline execution.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for submission artifacts.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume QA from a saved checkpoint when available.",
    )
    parser.add_argument(
        "--checkpoint-path",
        help=(
            "Optional path to the QA checkpoint JSON. "
            "Defaults to <output-dir>/qa_checkpoint.json."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Persist QA checkpoint every N processed questions (default: 1).",
    )


def _add_validate_submission_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Execute `_add_validate_submission_command`.

    Args:
        subparsers: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    parser = subparsers.add_parser("validate-submission", help="Validate a submission.json against local rules.")
    parser.add_argument("--phase", choices=[phase.value for phase in Phase], help="Override the config phase.")
    parser.add_argument("--submission-path", required=True)
    parser.add_argument("--questions-path", help="Path to questions.json. Defaults to the phase-local questions file.")
    parser.add_argument(
        "--documents-dir",
        help=(
            "Path to extracted PDFs. Defaults to the phase-local documents/pdfs directory "
            "when it exists."
        ),
    )
    parser.add_argument("--output-path", help="Optional JSON report destination.")


def _add_score_submission_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Execute `_add_score_submission_command`.

    Args:
        subparsers: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    parser = subparsers.add_parser("score-submission", help="Score a submission against a local reference set.")
    parser.add_argument("--submission-path", required=True)
    parser.add_argument("--reference-path", required=True, help="Path to reference answers JSON/JSONL.")
    parser.add_argument(
        "--questions-path",
        help="Path to questions.json used to validate raw extra answers. Defaults to the phase-local questions file.",
    )
    parser.add_argument(
        "--assistant-scores-path",
        help="Optional manual/proxy free_text scores JSON/JSONL for this submission.",
    )
    parser.add_argument(
        "--documents-dir",
        help=(
            "Path to extracted PDFs used for doc/page validation. Defaults to the phase-local "
            "documents/pdfs directory when it exists."
        ),
    )
    parser.add_argument("--output-path", help="Optional JSON report destination.")


def _add_benchmark_summary_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Execute `_add_benchmark_summary_command`.

    Args:
        subparsers: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    parser = subparsers.add_parser(
        "benchmark-summary",
        help="Run a submission against a structured dev benchmark dataset and compute slice metrics.",
    )
    parser.add_argument("--phase", choices=[phase.value for phase in Phase], help="Override the config phase.")
    parser.add_argument("--submission-path", required=True)
    parser.add_argument("--benchmark-path", help="Path to benchmark dataset JSON.")
    parser.add_argument("--reference-path", help="Path to reference answers JSON/JSONL.")
    parser.add_argument("--metadata-path", help="Path to benchmark metadata JSON/JSONL.")
    parser.add_argument("--benchmark-name", help="Logical benchmark name used in the report.")
    parser.add_argument("--questions-path", help="Path to questions.json. Defaults to the phase-local questions file.")
    parser.add_argument("--assistant-scores-path", help="Optional free_text assistant scores JSON/JSONL.")
    parser.add_argument(
        "--documents-dir",
        help=(
            "Path to extracted PDFs used for doc/page validation. Defaults to the phase-local "
            "documents/pdfs directory when it exists."
        ),
    )
    parser.add_argument("--label", help="Optional run label stored in the benchmark report.")
    parser.add_argument("--config-version", help="Optional config version stored in the benchmark report.")
    parser.add_argument("--submission-uuid", help="Optional submission UUID stored in the benchmark report.")
    parser.add_argument("--output-path", help="Optional JSON report destination.")


def _add_compare_benchmark_runs_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Execute `_add_compare_benchmark_runs_command`.

    Args:
        subparsers: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    parser = subparsers.add_parser(
        "compare-benchmark-runs",
        help="Compare two serialized benchmark run reports.",
    )
    parser.add_argument("--baseline-path", required=True)
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--baseline-label")
    parser.add_argument("--candidate-label")
    parser.add_argument("--output-path", help="Optional JSON report destination.")


def _add_analyze_benchmark_consistency_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Execute `_add_analyze_benchmark_consistency_command`.

    Args:
        subparsers: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    parser = subparsers.add_parser(
        "analyze-benchmark-consistency",
        help="Analyze repeated benchmark runs for unstable questions.",
    )
    parser.add_argument(
        "--report-path",
        dest="report_paths",
        action="append",
        required=True,
        help="Path to a benchmark run JSON. Pass multiple times for repeated runs.",
    )
    parser.add_argument("--instability-threshold", type=float, default=0.05)
    parser.add_argument("--output-path", help="Optional JSON report destination.")
