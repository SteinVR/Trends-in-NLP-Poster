"""Execute and adapt opendataloader sidecars for routed structural parses."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable

from src.common.config import AppConfig
from src.common.schemas import ParserRoute, RawParseNode, RawParseSidecar, RawParseTable

_NON_ALNUM_PATTERN = re.compile(r"[^0-9A-Za-z._-]+")
_HEADING_LEVEL_PATTERN = re.compile(r"\d+")


class OpenDataLoaderExecutionError(RuntimeError):
    """Raised when a routed opendataloader execution cannot produce JSON output."""


@lru_cache(maxsize=1)
def opendataloader_convert_available() -> bool:
    """Return whether opendataloader convert is importable and callable in this runtime."""

    try:
        _load_convert_callable()
    except OpenDataLoaderExecutionError:
        return False
    return True


@dataclass(slots=True)
class _ExtractionState:
    doc_id: str
    nodes: list[RawParseNode] = field(default_factory=list)
    tables: list[RawParseTable] = field(default_factory=list)
    table_aliases: dict[str, str] = field(default_factory=dict)
    used_node_ids: set[str] = field(default_factory=set)
    used_table_ids: set[str] = field(default_factory=set)
    node_counter: Iterable[int] = field(default_factory=lambda: count(1))
    table_counter: Iterable[int] = field(default_factory=lambda: count(1))


def build_opendataloader_sidecar(
    *,
    pdf_path: str | Path,
    page_geometry: dict[int, list[float]],
    parser_route: ParserRoute,
    config: AppConfig,
    route_decisions: list[str] | None = None,
) -> RawParseSidecar:
    """Execute opendataloader for one routed path and adapt JSON to sidecar schema."""

    source_path = Path(pdf_path)
    try:
        with TemporaryDirectory(prefix=f"opendataloader-{source_path.stem}-") as tmp_dir:
            output_dir = Path(tmp_dir)
            _run_opendataloader_convert(
                pdf_path=source_path,
                output_dir=output_dir,
                parser_route=parser_route,
                config=config,
            )
            payload_path = _resolve_output_payload(output_dir, source_path)
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exercised through unit tests via synthetic failures
        raise OpenDataLoaderExecutionError(
            f"Failed to execute opendataloader for {source_path.name} on route {parser_route.value}: {exc}"
        ) from exc

    nodes, tables = _extract_nodes_and_tables(payload, source_path.stem)
    decisions = list(route_decisions or [f"{parser_route.value}_selected"])
    if "opendataloader_convert_executed" not in decisions:
        decisions.append("opendataloader_convert_executed")
    page_count = int(payload.get("number of pages") or len(page_geometry))

    return RawParseSidecar(
        doc_id=source_path.stem,
        parser_name="opendataloader",
        parser_route=parser_route,
        parser_provenance=_route_provenance(parser_route),
        page_count=max(page_count, 0),
        page_geometry=page_geometry,
        diagnostics={
            "adapter_mode": "executed",
            "table_method": config.ingestion.opendataloader_table_method,
            "reading_order": config.ingestion.opendataloader_reading_order,
            "hybrid_backend": (
                config.ingestion.opendataloader_hybrid_backend if parser_route is ParserRoute.HYBRID else None
            ),
            "hybrid_mode": config.ingestion.opendataloader_hybrid_mode if parser_route is ParserRoute.HYBRID else None,
            "node_count": len(nodes),
            "table_count": len(tables),
        },
        route_decisions=decisions,
        nodes=nodes,
        tables=tables,
    )


def route_provenance(parser_route: ParserRoute) -> str:
    """Expose parser provenance value for one routed structural parse."""

    return _route_provenance(parser_route)


def _route_provenance(parser_route: ParserRoute) -> str:
    """Execute `_route_provenance`."""
    route_suffixes = {
        ParserRoute.TAGGED: "tagged",
        ParserRoute.HYBRID: "hybrid",
        ParserRoute.LOCAL_ONLY: "local_only",
    }
    return f"opendataloader:{route_suffixes[parser_route]}"


def _run_opendataloader_convert(
    *,
    pdf_path: Path,
    output_dir: Path,
    parser_route: ParserRoute,
    config: AppConfig,
) -> None:
    """Run opendataloader convert."""
    convert = _load_convert_callable()
    kwargs: dict[str, Any] = {
        "input_path": [str(pdf_path)],
        "output_dir": str(output_dir),
        "format": "json",
        "quiet": True,
        "table_method": config.ingestion.opendataloader_table_method,
        "reading_order": config.ingestion.opendataloader_reading_order,
    }
    if parser_route is ParserRoute.TAGGED:
        kwargs["use_struct_tree"] = True
    elif parser_route is ParserRoute.HYBRID:
        kwargs["hybrid"] = config.ingestion.opendataloader_hybrid_backend
        kwargs["hybrid_mode"] = config.ingestion.opendataloader_hybrid_mode
        if config.ingestion.opendataloader_hybrid_url:
            kwargs["hybrid_url"] = config.ingestion.opendataloader_hybrid_url
        kwargs["hybrid_timeout"] = str(config.ingestion.opendataloader_hybrid_timeout_ms)
        kwargs["hybrid_fallback"] = False
    elif parser_route is not ParserRoute.LOCAL_ONLY:
        raise ValueError(f"Unsupported parser route for opendataloader execution: {parser_route!s}")

    # Force headless Java execution to keep structural parsing deterministic in
    # CI/sandbox runtimes without an X11 display server.
    previous_java_options = os.environ.get("JAVA_TOOL_OPTIONS")
    options = (previous_java_options or "").strip()
    if "-Djava.awt.headless=true" not in options:
        options = f"{options} -Djava.awt.headless=true".strip()
    os.environ["JAVA_TOOL_OPTIONS"] = options
    try:
        convert(**kwargs)
    finally:
        if previous_java_options is None:
            os.environ.pop("JAVA_TOOL_OPTIONS", None)
        else:
            os.environ["JAVA_TOOL_OPTIONS"] = previous_java_options


def _load_convert_callable() -> Callable[..., None]:
    """Load convert callable."""
    module_spec = importlib.util.find_spec("opendataloader_pdf")
    if module_spec is None:
        raise OpenDataLoaderExecutionError("opendataloader_pdf package is not importable.")

    module = importlib.import_module("opendataloader_pdf")
    convert = getattr(module, "convert", None)
    if not callable(convert):
        raise OpenDataLoaderExecutionError("opendataloader_pdf.convert is unavailable.")
    return convert


def _resolve_output_payload(output_dir: Path, pdf_path: Path) -> Path:
    """Resolve output payload."""
    json_paths = sorted(output_dir.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON output produced under {output_dir}.")

    preferred_names = {f"{pdf_path.stem}.json", f"{pdf_path.name}.json"}
    for candidate in json_paths:
        if candidate.name in preferred_names:
            return candidate

    stem_matches = [candidate for candidate in json_paths if candidate.stem == pdf_path.stem]
    if len(stem_matches) == 1:
        return stem_matches[0]

    if len(json_paths) == 1:
        return json_paths[0]

    candidate_names = ", ".join(path.name for path in json_paths)
    raise FileNotFoundError(
        f"Unable to resolve JSON output for {pdf_path.name}. Candidates: {candidate_names}"
    )


def _extract_nodes_and_tables(payload: dict[str, Any], doc_id: str) -> tuple[list[RawParseNode], list[RawParseTable]]:
    """Extract nodes and tables."""
    state = _ExtractionState(doc_id=doc_id)
    for element in _as_element_list(payload.get("kids")):
        _visit_element(element, state)
    _resolve_table_aliases(state)
    return state.nodes, state.tables


def _visit_element(element: dict[str, Any], state: _ExtractionState) -> None:
    """Execute `_visit_element`."""
    element_type = str(element.get("type", "")).strip().lower()
    if not element_type:
        for child in _iter_child_elements(element):
            _visit_element(child, state)
        return

    if element_type == "table":
        _add_table_element(element, state)
        return

    if element_type in {"table row", "table cell"}:
        for child in _iter_child_elements(element):
            _visit_element(child, state)
        return

    kind = element_type.replace(" ", "_")
    text = _extract_text(element, element_type)
    if text or kind in {"heading", "caption", "list_item", "paragraph", "list"}:
        node = RawParseNode(
            node_id=_build_node_id(element, state),
            kind=kind,
            page_number=_page_number(element),
            text=text,
            bbox=_bbox(element),
            heading_level=_heading_level(element) if kind == "heading" else None,
            linked_content_id=_linked_content_id(element),
        )
        state.nodes.append(node)

    for child in _iter_child_elements(element):
        _visit_element(child, state)


def _add_table_element(element: dict[str, Any], state: _ExtractionState) -> None:
    """Execute `_add_table_element`."""
    table_id = _build_table_id(element, state)
    raw_id = _element_id_token(element)
    if raw_id is not None:
        state.table_aliases[raw_id] = table_id

    rows = _extract_table_rows(element)
    table_text = _rows_to_markdown(rows)
    linked_content_id = _linked_content_id(element)
    page_number = _page_number(element)

    state.nodes.append(
        RawParseNode(
            node_id=f"{table_id}-node",
            kind="table",
            page_number=page_number,
            text=table_text,
            bbox=_bbox(element),
            linked_content_id=linked_content_id,
        )
    )
    state.tables.append(
        RawParseTable(
            table_id=table_id,
            page_number=page_number,
            bbox=_bbox(element),
            raw_json=element,
            raw_html=_raw_html(element),
            linked_context_block_id=linked_content_id,
        )
    )


def _resolve_table_aliases(state: _ExtractionState) -> None:
    """Resolve table aliases."""
    for node in state.nodes:
        if node.linked_content_id and node.linked_content_id in state.table_aliases:
            node.linked_content_id = state.table_aliases[node.linked_content_id]
    for table in state.tables:
        if table.linked_context_block_id and table.linked_context_block_id in state.table_aliases:
            table.linked_context_block_id = state.table_aliases[table.linked_context_block_id]


def _extract_text(element: dict[str, Any], element_type: str) -> str:
    """Extract text."""
    content = element.get("content")
    if isinstance(content, str):
        normalized = _normalize_text(content)
        if normalized:
            return normalized

    if element_type == "list":
        list_items = [
            _normalize_text(str(item.get("content", "")))
            for item in _as_element_list(element.get("list items"))
            if isinstance(item.get("content"), str)
        ]
        return "\n".join(item for item in list_items if item)

    return ""


def _extract_table_rows(element: dict[str, Any]) -> list[list[str]]:
    """Extract table rows."""
    rows: list[list[str]] = []
    for row in _as_element_list(element.get("rows")):
        cells = _as_element_list(row.get("cells"))
        if not cells:
            continue
        rows.append([_cell_text(cell) for cell in cells])
    if rows:
        return rows

    for row in _as_element_list(element.get("kids")):
        if str(row.get("type", "")).strip().lower() != "table row":
            continue
        cells = _as_element_list(row.get("cells")) or _as_element_list(row.get("kids"))
        if cells:
            rows.append([_cell_text(cell) for cell in cells])
    return rows


def _cell_text(cell: dict[str, Any]) -> str:
    """Execute `_cell_text`."""
    content = cell.get("content")
    if isinstance(content, str):
        normalized = _normalize_text(content)
        if normalized:
            return normalized

    fragments: list[str] = []
    _collect_text_fragments(cell, fragments)
    return _normalize_text(" ".join(fragments))


def _collect_text_fragments(element: dict[str, Any], sink: list[str]) -> None:
    """Execute `_collect_text_fragments`."""
    content = element.get("content")
    if isinstance(content, str):
        normalized = _normalize_text(content)
        if normalized:
            sink.append(normalized)

    for child in _iter_child_elements(element):
        _collect_text_fragments(child, sink)


def _iter_child_elements(element: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute `_iter_child_elements`."""
    children: list[dict[str, Any]] = []
    children.extend(_as_element_list(element.get("kids")))
    children.extend(_as_element_list(element.get("list items")))

    for row in _as_element_list(element.get("rows")):
        children.append(row)
        children.extend(_as_element_list(row.get("cells")))
    children.extend(_as_element_list(element.get("cells")))
    return children


def _heading_level(element: dict[str, Any]) -> int | None:
    """Execute `_heading_level`."""
    value = element.get("heading level")
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0:
        return int(value)
    if isinstance(value, str):
        match = _HEADING_LEVEL_PATTERN.search(value)
        if match:
            return int(match.group(0))
        if value.strip().lower() in {"title", "doctitle"}:
            return 1

    level = element.get("level")
    if isinstance(level, str):
        match = _HEADING_LEVEL_PATTERN.search(level)
        if match:
            return int(match.group(0))
    return None


def _page_number(element: dict[str, Any]) -> int:
    """Execute `_page_number`."""
    value = element.get("page number")
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0:
        return int(value)
    return 1


def _bbox(element: dict[str, Any]) -> list[float] | None:
    """Execute `_bbox`."""
    value = element.get("bounding box")
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    except (TypeError, ValueError):
        return None


def _raw_html(element: dict[str, Any]) -> str | None:
    """Execute `_raw_html`."""
    for key in ("raw html", "raw_html", "html"):
        value = element.get(key)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
    return None


def _linked_content_id(element: dict[str, Any]) -> str | None:
    """Execute `_linked_content_id`."""
    value = element.get("linked content id")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _build_node_id(element: dict[str, Any], state: _ExtractionState) -> str:
    """Build node id."""
    token = _element_id_token(element)
    if token is not None:
        candidate = _ensure_prefixed_id(state.doc_id, "n", token)
        if candidate not in state.used_node_ids:
            state.used_node_ids.add(candidate)
            return candidate
    return _claim_auto_id(state.doc_id, "n-auto-", state.used_node_ids, state.node_counter)


def _build_table_id(element: dict[str, Any], state: _ExtractionState) -> str:
    """Build table id."""
    token = _element_id_token(element)
    if token is not None:
        candidate = _ensure_prefixed_id(state.doc_id, "t", token)
        if candidate not in state.used_table_ids:
            state.used_table_ids.add(candidate)
            return candidate
    return _claim_auto_id(state.doc_id, "t-auto-", state.used_table_ids, state.table_counter)


def _element_id_token(element: dict[str, Any]) -> str | None:
    """Execute `_element_id_token`."""
    value = element.get("id")
    if value is None:
        return None
    normalized = _NON_ALNUM_PATTERN.sub("-", str(value)).strip("-")
    return normalized or None


def _ensure_prefixed_id(doc_id: str, marker: str, token: str) -> str:
    """Execute `_ensure_prefixed_id`."""
    return f"{doc_id}-{marker}{token}"


def _claim_auto_id(doc_id: str, prefix: str, used: set[str], counter: Iterable[int]) -> str:
    """Execute `_claim_auto_id`."""
    while True:
        candidate = f"{doc_id}-{prefix}{next(counter)}"
        if candidate not in used:
            used.add(candidate)
            return candidate


def _as_element_list(value: Any) -> list[dict[str, Any]]:
    """Execute `_as_element_list`."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _normalize_text(value: str) -> str:
    """Normalize text."""
    return " ".join(value.split())


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """Execute `_rows_to_markdown`."""
    if not rows:
        return ""
    width = max((len(row) for row in rows), default=0)
    if width <= 0:
        return ""
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    header = "| " + " | ".join(padded_rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(row) + " |" for row in padded_rows[1:]]
    return "\n".join([header, separator, *body]) if body else "\n".join([header, separator])
