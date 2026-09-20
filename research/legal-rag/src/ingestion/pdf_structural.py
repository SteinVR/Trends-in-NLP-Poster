"""Structural sidecar and route helpers for PDF parsing."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import fitz

from src.common.config import AppConfig
from src.common.schemas import (
    ParsedPage,
    ParserRoute,
    RawParseNode,
    RawParseSidecar,
    RawParseTable,
)

from .opendataloader_adapter import (
    OpenDataLoaderExecutionError,
    build_opendataloader_sidecar,
    opendataloader_convert_available,
    route_provenance,
)
from .parser_routing import DocumentStructuralProfile, ParserAvailability, resolve_parser_route

LOGGER = logging.getLogger(__name__)

def _build_pymupdf_sidecar(
    *,
    doc_id: str,
    pages: list[ParsedPage],
    page_geometry: dict[int, list[float]],
    parser_route: ParserRoute,
) -> RawParseSidecar:
    """Build pymupdf sidecar.

    Args:
        doc_id: Input parameter.
        pages: Input parameter.
        page_geometry: Input parameter.
        parser_route: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    nodes: list[RawParseNode] = []
    tables: list[RawParseTable] = []
    for page in pages:
        for block_index, text_block in enumerate(page.text_blocks, start=1):
            nodes.append(
                RawParseNode(
                    node_id=f"{doc_id}-p{page.page_number}-b{block_index}",
                    kind="paragraph",
                    page_number=page.page_number,
                    text=text_block,
                    bbox=None,
                )
            )
        for table in page.tables:
            tables.append(
                RawParseTable(
                    table_id=table.table_id,
                    page_number=table.page_number,
                    bbox=None,
                    raw_json={"rows": table.rows} if table.rows else None,
                    raw_html=table.raw_html,
                    linked_context_block_id=None,
                )
            )

    return RawParseSidecar(
        doc_id=doc_id,
        parser_name="pymupdf",
        parser_route=parser_route,
        parser_provenance="pymupdf:native",
        page_count=len(pages),
        page_geometry=page_geometry,
        diagnostics={},
        route_decisions=[f"{parser_route.value}_selected"],
        nodes=nodes,
        tables=tables,
    )


def _build_raw_parse_sidecars(
    *,
    pdf_path: Path,
    doc_id: str,
    pages: list[ParsedPage],
    page_geometry: dict[int, list[float]],
    parser_route: ParserRoute,
    config: AppConfig,
) -> tuple[str, dict[str, RawParseSidecar]]:
    """Build raw parse sidecars.

    Args:
        pdf_path: Input parameter.
        doc_id: Input parameter.
        pages: Input parameter.
        page_geometry: Input parameter.
        parser_route: Input parameter.
        config: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    route_decisions: list[str] = [f"{parser_route.value}_selected"]

    try:
        structural_sidecar = build_opendataloader_sidecar(
            pdf_path=pdf_path,
            page_geometry=page_geometry,
            parser_route=parser_route,
            config=config,
            route_decisions=route_decisions,
        )
    except OpenDataLoaderExecutionError as exc:
        LOGGER.error(
            "Structural opendataloader route %s failed for %s.",
            parser_route.value,
            pdf_path.name,
        )
        raise OpenDataLoaderExecutionError(
            f"Structural parser route {parser_route.value} failed for {pdf_path.name}; "
            "OpenDataLoader failure is fatal."
        ) from exc

    pymupdf_sidecar = _build_pymupdf_sidecar(
        doc_id=doc_id,
        pages=pages,
        page_geometry=page_geometry,
        parser_route=parser_route,
    )
    return route_provenance(structural_sidecar.parser_route), {
        "pymupdf": pymupdf_sidecar,
        "opendataloader": structural_sidecar,
    }


def _resolve_document_route(document: fitz.Document, pages: list[ParsedPage], config: AppConfig) -> ParserRoute:
    """Resolve document route.

    Args:
        document: Input parameter.
        pages: Input parameter.
        config: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    profile = DocumentStructuralProfile(
        has_struct_tree=_has_struct_tree(document),
        born_digital=_is_born_digital(pages, config),
        layout_complex=_is_layout_complex(pages),
    )
    availability = _parser_availability(config)
    return resolve_parser_route(profile, availability)


def _parser_availability(config: AppConfig) -> ParserAvailability:
    """Execute `_parser_availability`.

    Args:
        config: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    opendataloader_available = _is_opendataloader_available()
    normalized_hybrid_url = (config.ingestion.opendataloader_hybrid_url or "").strip()
    hybrid_url_configured = config.ingestion.opendataloader_hybrid_url is None or bool(normalized_hybrid_url)
    hybrid_backend_healthy = (
        opendataloader_available
        and config.ingestion.opendataloader_hybrid_backend != "off"
        and hybrid_url_configured
    )
    return ParserAvailability(
        opendataloader_available=opendataloader_available,
        hybrid_backend_healthy=hybrid_backend_healthy,
    )


def _is_opendataloader_available() -> bool:
    # Compatibility probe for existing instrumentation/tests; actual capability is
    # decided by whether the convert callable is importable and executable.
    """Return whether opendataloader available is true.

    Returns:
        Any: The computed result of the function.
    """
    return shutil.which("java") is not None and opendataloader_convert_available()


def _has_struct_tree(document: fitz.Document) -> bool:
    """Return whether struct tree is present.

    Args:
        document: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    try:
        catalog_xref = document.pdf_catalog()
        key_type, value = document.xref_get_key(catalog_xref, "StructTreeRoot")
    except Exception as exc:
        raise RuntimeError("Failed to inspect PDF StructTreeRoot during parser routing.") from exc
    return key_type != "null" and bool(str(value).strip())


def _is_born_digital(pages: list[ParsedPage], config: AppConfig) -> bool:
    """Return whether born digital is true.

    Args:
        pages: Input parameter.
        config: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    native_chars = sum(len((page.native_text or "").strip()) for page in pages)
    return native_chars >= config.ingestion.extractable_text_min_chars


def _is_layout_complex(pages: list[ParsedPage]) -> bool:
    """Return whether layout complex is true.

    Args:
        pages: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    return any(
        page.signals.table_count > 0 or page.signals.drawing_count > 0 or page.signals.image_block_count > 0
        for page in pages
    )
