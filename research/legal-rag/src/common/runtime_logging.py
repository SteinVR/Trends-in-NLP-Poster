"""Shared runtime logging configuration and structured event helpers."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from src.common.config import AppConfig

LOGGER_NAMESPACE = "agentic_rag"
_RUNTIME_CONTEXT: ContextVar[RuntimeLogContext | None] = ContextVar("runtime_log_context", default=None)


@dataclass(frozen=True, slots=True)
class RuntimeLogContext:
    """Resolved runtime logging context for one CLI invocation."""

    run_id: str
    command: str
    phase: str
    log_path: Path | None
    level: str
    debug_documents: bool
    debug_questions: bool


class _JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        """Execute `format`.

        Args:
            record: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "log"),
            "message": record.getMessage(),
            "component": _component_name(record.name),
            "stage": getattr(record, "stage", "runtime"),
        }
        context = _RUNTIME_CONTEXT.get()
        if context is not None:
            payload.update(
                {
                    "run_id": context.run_id,
                    "command": context.command,
                    "phase": context.phase,
                }
            )

        event_context = getattr(record, "context", None)
        if isinstance(event_context, Mapping):
            payload.update(_normalize_value(dict(event_context)))

        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__
            payload["error"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        """Execute `format`.

        Args:
            record: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        component = _component_name(record.name)
        event = getattr(record, "event", "log")
        message = record.getMessage()
        fragments = [f"[{timestamp}]", f"[{record.levelname}]", f"[{component}:{event}]", message]

        event_context = getattr(record, "context", None)
        if isinstance(event_context, Mapping):
            for key in ("question_id", "doc_id", "duration_ms", "log_path"):
                value = event_context.get(key)
                if value is None:
                    continue
                fragments.append(f"{key}={value}")

        if record.exc_info:
            fragments.append(self.formatException(record.exc_info))

        return " ".join(str(fragment) for fragment in fragments if fragment)


def configure_runtime_logging(config: AppConfig, *, command: str, phase: str) -> RuntimeLogContext:
    """Configure the shared project logger for one CLI run."""

    settings = config.logging
    effective_level_name = "DEBUG" if (settings.debug_documents or settings.debug_questions) else settings.level
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid4().hex
    log_path: Path | None = None
    if settings.jsonl:
        log_dir = settings.directory / phase
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{timestamp}_{_safe_filename(command)}_{run_id}.jsonl"

    project_logger = logging.getLogger(LOGGER_NAMESPACE)
    _reset_logger(project_logger)
    project_logger.setLevel(getattr(logging, effective_level_name))
    project_logger.propagate = False

    if settings.console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(_ConsoleFormatter())
        project_logger.addHandler(console_handler)

    if settings.jsonl:
        assert log_path is not None
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(_JsonlFormatter())
        project_logger.addHandler(file_handler)

    context = RuntimeLogContext(
        run_id=run_id,
        command=command,
        phase=phase,
        log_path=log_path,
        level=effective_level_name,
        debug_documents=settings.debug_documents,
        debug_questions=settings.debug_questions,
    )
    _RUNTIME_CONTEXT.set(context)
    log_event(
        get_logger("runtime"),
        logging.INFO,
        "runtime.configured",
        "Runtime logging configured.",
        stage="runtime",
        log_path=str(log_path) if log_path is not None else None,
        configured_level=effective_level_name,
        console=settings.console,
        jsonl=settings.jsonl,
    )
    return context


def shutdown_runtime_logging() -> None:
    """Close project logger handlers and clear the current runtime context."""

    project_logger = logging.getLogger(LOGGER_NAMESPACE)
    _reset_logger(project_logger)
    _RUNTIME_CONTEXT.set(None)


def get_logger(component: str) -> logging.Logger:
    """Return a project-scoped logger for a specific component."""

    normalized = component.strip(".")
    if not normalized:
        return logging.getLogger(LOGGER_NAMESPACE)
    return logging.getLogger(f"{LOGGER_NAMESPACE}.{normalized}")


def get_runtime_context() -> RuntimeLogContext | None:
    """Return runtime context.

    Returns:
        Any: The computed result of the function.
    """
    return _RUNTIME_CONTEXT.get()


def runtime_log_path() -> Path | None:
    """Execute `runtime_log_path`.

    Returns:
        Any: The computed result of the function.
    """
    context = get_runtime_context()
    return None if context is None else context.log_path


def questions_debug_enabled() -> bool:
    """Execute `questions_debug_enabled`.

    Returns:
        Any: The computed result of the function.
    """
    context = get_runtime_context()
    return bool(context and context.debug_questions)


def documents_debug_enabled() -> bool:
    """Execute `documents_debug_enabled`.

    Returns:
        Any: The computed result of the function.
    """
    context = get_runtime_context()
    return bool(context and context.debug_documents)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    *,
    stage: str,
    exc_info: Any | None = None,
    **context: Any,
) -> None:
    """Emit one structured runtime event."""

    logger.log(
        level,
        message,
        extra={
            "event": event,
            "stage": stage,
            "context": _normalize_value(context),
        },
        exc_info=exc_info,
    )


def _component_name(logger_name: str) -> str:
    """Execute `_component_name`.

    Args:
        logger_name: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if logger_name == LOGGER_NAMESPACE:
        return "main"
    prefix = f"{LOGGER_NAMESPACE}."
    if logger_name.startswith(prefix):
        return logger_name[len(prefix) :]
    return logger_name


def _normalize_value(value: Any) -> Any:
    """Normalize value.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    return value


def _reset_logger(logger: logging.Logger) -> None:
    """Execute `_reset_logger`.

    Args:
        logger: Input parameter.

    Returns:
        None: This function does not return a value.
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _safe_filename(value: str) -> str:
    """Execute `_safe_filename`.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    return "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)
