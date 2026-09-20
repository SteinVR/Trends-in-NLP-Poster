"""Shared typed models for the foundation and ingestion layers."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _empty_page_signal() -> "PageSignal":
    """Execute `_empty_page_signal`.

    Returns:
        Any: The computed result of the function.
    """
    return PageSignal(
        text_char_count=0,
        word_count=0,
        image_block_count=0,
        max_image_area_ratio=0.0,
        drawing_count=0,
        table_count=0,
        legal_marker_count=0,
        bad_char_ratio=0.0,
    )


class Phase(StrEnum):
    """Competition phases supported by the local workspace."""

    WARMUP = "warmup"
    WARMUP_OCR = "warmup-ocr"
    FINAL = "final"


class TriageLabel(StrEnum):
    """Rule-based routing label for a parsed page."""

    NATIVE_CLEAN = "native_clean"
    SCAN = "scan"
    SUSPICIOUS = "suspicious"


class SourceMode(StrEnum):
    """Source used for the final page text."""

    NATIVE = "native"
    OCR = "ocr"
    HYBRID = "hybrid"


class ParseStatus(StrEnum):
    """Parse completion state for a page or document."""

    PARSED = "parsed"
    DEGRADED = "degraded"
    FAILED = "failed"


class ParserRoute(StrEnum):
    """Selected parser route for structural ingestion."""

    TAGGED = "tagged"
    HYBRID = "hybrid"
    LOCAL_ONLY = "local_only"


class ContentBlockType(StrEnum):
    """Retrieval-facing block type inside a canonical page record."""

    TEXT = "text"
    TABLE = "table"


class QuestionRecord(BaseModel):
    """Question payload returned by the competition API."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    answer_type: str


class ArtifactMetadata(BaseModel):
    """Persisted metadata for a local phase artifact."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["file", "directory"]
    relative_path: str
    sha256: str | None = None
    size_bytes: int | None = None
    file_count: int | None = None
    updated_at: datetime | None = None


class PhaseSyncManifest(BaseModel):
    """Manifest describing the current local state for a phase sync."""

    model_config = ConfigDict(extra="forbid")

    phase: Phase
    phase_isolation: Literal["local-only"] = "local-only"
    synced_at: datetime
    base_url: str
    api_key_env: str
    operations: list[str] = Field(default_factory=list)
    questions_count: int | None = None
    extracted_pdf_count: int | None = None
    artifacts: dict[str, ArtifactMetadata] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class SerializedTableBlock(BaseModel):
    """Self-contained retrieval block derived from a raw table."""

    model_config = ConfigDict(extra="forbid")

    block_id: str
    text: str
    source_page_number: int = Field(ge=1)
    row_anchor: str | None = None
    column_context: list[str] = Field(default_factory=list)
    table_caption: str | None = None
    footnotes: list[str] = Field(default_factory=list)
    legal_context: list[str] = Field(default_factory=list)


class SemanticTableBlock(BaseModel):
    """Semantic retrieval block produced from table serialization."""

    model_config = ConfigDict(extra="forbid")

    block_id: str
    text: str
    source_page_number: int = Field(ge=1)


class TableBlock(BaseModel):
    """Structured table representation preserved in the canonical corpus."""

    model_config = ConfigDict(extra="forbid")

    table_id: str
    page_span: list[int] = Field(default_factory=list)
    raw_markdown: str | None = None
    raw_html: str | None = None
    header_signature: list[str] = Field(default_factory=list)
    serialized_blocks: list[SerializedTableBlock] = Field(default_factory=list)
    semantic_blocks: list[SemanticTableBlock] = Field(default_factory=list)


class RawTableRecord(BaseModel):
    """Raw table extracted from a page before retrieval-friendly serialization."""

    model_config = ConfigDict(extra="forbid")

    table_id: str
    page_number: int = Field(ge=1)
    rows: list[list[str]] = Field(default_factory=list)
    row_page_numbers: list[int] = Field(default_factory=list)
    has_header: bool = False
    raw_markdown: str | None = None
    raw_html: str | None = None
    header_signature: list[str] = Field(default_factory=list)
    caption: str | None = None
    footnotes: list[str] = Field(default_factory=list)
    preceding_context: list[str] = Field(default_factory=list)
    following_context: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _compat_aliases(cls, value: Any) -> Any:
        """Execute `_compat_aliases`.

        Args:
            value: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        if not isinstance(value, dict):
            return value

        payload = dict(value)
        if "rows" not in payload and "cells" in payload:
            payload["rows"] = payload.pop("cells")
        if "preceding_context" not in payload and "context_before" in payload:
            payload["preceding_context"] = payload.pop("context_before")
        if "following_context" not in payload and "context_after" in payload:
            payload["following_context"] = payload.pop("context_after")
        return payload

    @property
    def cells(self) -> list[list[str]]:
        """Execute `cells`.

        Returns:
            Any: The computed result of the function.
        """
        return self.rows

    @property
    def context_before(self) -> list[str]:
        """Execute `context_before`.

        Returns:
            Any: The computed result of the function.
        """
        return self.preceding_context

    @property
    def context_after(self) -> list[str]:
        """Execute `context_after`.

        Returns:
            Any: The computed result of the function.
        """
        return self.following_context


class RawParseNode(BaseModel):
    """Typed structural node preserved in parser sidecars."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    kind: str
    page_number: int = Field(ge=1)
    text: str
    bbox: list[float] | None = None
    heading_level: int | None = None
    linked_content_id: str | None = None


class RawParseTable(BaseModel):
    """Raw table object preserved in parser sidecars."""

    model_config = ConfigDict(extra="forbid")

    table_id: str
    page_number: int = Field(ge=1)
    bbox: list[float] | None = None
    raw_json: dict[str, Any] | None = None
    raw_html: str | None = None
    linked_context_block_id: str | None = None


class RawParseSidecar(BaseModel):
    """Sidecar artifact preserving parser-native structural outputs."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    parser_name: str
    parser_route: ParserRoute
    parser_provenance: str
    page_count: int = Field(ge=0)
    page_geometry: dict[int, list[float]] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    route_decisions: list[str] = Field(default_factory=list)
    nodes: list[RawParseNode] = Field(default_factory=list)
    tables: list[RawParseTable] = Field(default_factory=list)


class NormalizedStructuralBlock(BaseModel):
    """Normalized structural block consumed by canonical corpus assembly."""

    model_config = ConfigDict(extra="forbid")

    source_block_id: str
    page_number: int = Field(ge=1)
    text: str
    heading_path: list[str] = Field(default_factory=list)
    block_roles: list[str] = Field(default_factory=list)
    linked_content_id: str | None = None


class PageSignal(BaseModel):
    """Low-level diagnostic signals collected during parsing."""

    model_config = ConfigDict(extra="forbid")

    text_char_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    image_block_count: int = Field(ge=0)
    max_image_area_ratio: float = Field(ge=0.0, le=1.0)
    drawing_count: int = Field(ge=0)
    table_count: int = Field(ge=0)
    legal_marker_count: int = Field(ge=0)
    bad_char_ratio: float = Field(ge=0.0, le=1.0)


class ParsedPage(BaseModel):
    """Intermediate parsed-page representation used before corpus assembly."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    page_number: int = Field(ge=1)
    document_title: str | None = None
    document_family: str | None = None
    triage_label: TriageLabel
    source_mode: SourceMode
    parse_status: ParseStatus = ParseStatus.PARSED
    text: str
    native_text: str | None = None
    ocr_text: str | None = None
    text_blocks: list[str] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0)
    quality_flags: list[str] = Field(default_factory=list)
    signals: PageSignal = Field(default_factory=_empty_page_signal)
    resolution_notes: list[str] = Field(default_factory=list)
    tables: list[RawTableRecord] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _compat_aliases(cls, value: Any) -> Any:
        """Execute `_compat_aliases`.

        Args:
            value: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        if not isinstance(value, dict):
            return value

        payload = dict(value)
        if "tables" not in payload and "table_candidates" in payload:
            payload["tables"] = payload.pop("table_candidates")
        return payload

    @property
    def table_candidates(self) -> list[RawTableRecord]:
        """Execute `table_candidates`.

        Returns:
            Any: The computed result of the function.
        """
        return self.tables


class ParsedDocument(BaseModel):
    """Intermediate parsed-document representation used before corpus assembly."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    filename: str
    sha256: str
    page_count: int = Field(ge=0)
    document_title: str | None = None
    document_family: str | None = None
    parse_status: ParseStatus = ParseStatus.PARSED
    failure_reason: str | None = None
    parser_provenance: str | None = None
    raw_parse_sidecars: dict[str, RawParseSidecar] = Field(default_factory=dict)
    pages: list[ParsedPage] = Field(default_factory=list)


class ContentBlock(BaseModel):
    """Retrieval-facing block stored on a page."""

    model_config = ConfigDict(extra="forbid")

    block_id: str
    type: ContentBlockType
    text: str
    page_number: int = Field(ge=1)
    table_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalPageRecord(BaseModel):
    """Canonical page-level record emitted by the ingestion stage."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    page_number: int = Field(ge=1)
    document_title: str | None = None
    document_family: str | None = None
    triage_label: TriageLabel
    source_mode: SourceMode
    parse_status: ParseStatus = ParseStatus.PARSED
    quality_score: float = Field(ge=0.0, le=1.0)
    quality_flags: list[str] = Field(default_factory=list)
    parser_provenance: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    block_roles: list[str] = Field(default_factory=list)
    source_block_ids: list[str] = Field(default_factory=list)
    text: str
    blocks: list[ContentBlock] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)


class PageMapRecord(BaseModel):
    """Document-level page map summary for downstream lookups."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    filename: str
    document_title: str | None = None
    document_family: str | None = None
    page_count: int = Field(ge=0)
    page_numbers: list[int] = Field(default_factory=list)


class CorpusParseManifestRecord(BaseModel):
    """Document-level parse outcome recorded in the corpus manifest."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    filename: str
    sha256: str
    page_count: int = Field(ge=0)
    parse_status: ParseStatus
    failure_reason: str | None = None


class CorpusParseManifest(BaseModel):
    """Top-level manifest for a parse-corpus run."""

    model_config = ConfigDict(extra="forbid")

    phase: Phase
    parsed_at: datetime
    document_count: int = Field(ge=0)
    page_count: int = Field(ge=0)
    failed_document_count: int = Field(ge=0)
    records: list[CorpusParseManifestRecord] = Field(default_factory=list)
