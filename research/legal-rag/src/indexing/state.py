"""Validation helpers for persisted indexing sidecar state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.indexing.embedders import BM25SparseEncoder


def load_page_parent_map(
    path: Path,
    *,
    expected_chunk_count: int,
) -> dict[str, dict[str, Any]] | None:
    """Load and validate page-parent map sidecar by chunk cardinality and schema."""

    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = _normalize_page_parent_map_payload(raw_payload)
    if len(normalized) != expected_chunk_count:
        return None

    for chunk_id, entry in normalized.items():
        if not isinstance(chunk_id, str) or not chunk_id:
            return None
        if not isinstance(entry.get("doc_id"), str) or not entry["doc_id"]:
            return None
        page_span = _page_span_from_entry(entry)
        if not page_span or not all(isinstance(page_number, int) and page_number >= 1 for page_number in page_span):
            return None
    return normalized


def load_sparse_encoder_state(path: Path) -> dict[str, Any] | None:
    """Load sparse-encoder state and validate strict restore contract."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        BM25SparseEncoder.from_state(payload)
    except Exception as exc:
        raise RuntimeError("Sparse encoder state is unreadable or invalid in strict mode.") from exc
    return payload


def _normalize_page_parent_map_payload(raw_payload: Any) -> dict[str, dict[str, Any]]:
    """Normalize page parent map payload.

    Args:
        raw_payload: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    if isinstance(raw_payload, dict):
        normalized: dict[str, dict[str, Any]] = {}
        for chunk_id, entry in raw_payload.items():
            if not isinstance(entry, dict):
                raise ValueError("page_parent_map values must be objects")
            normalized[chunk_id] = dict(entry)
            normalized[chunk_id].setdefault("chunk_id", chunk_id)
        return normalized

    if isinstance(raw_payload, list):
        normalized = {}
        for entry in raw_payload:
            if not isinstance(entry, dict):
                raise ValueError("page_parent_map entries must be objects")
            chunk_id = entry.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("page_parent_map entries must include a non-empty chunk_id")
            normalized[chunk_id] = dict(entry)
        return normalized

    raise ValueError("page_parent_map must be a JSON object or array")


def _page_span_from_entry(entry: dict[str, Any]) -> list[int] | None:
    """Execute `_page_span_from_entry`.

    Args:
        entry: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    for key in ("page_span", "page_numbers", "parent_page_numbers"):
        value = entry.get(key)
        if isinstance(value, list):
            return value
    return None
