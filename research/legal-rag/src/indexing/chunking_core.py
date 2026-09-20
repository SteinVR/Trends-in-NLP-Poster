"""Structure-aware chunk construction for hybrid indexing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.common.schemas import CanonicalPageRecord, ContentBlockType
from src.indexing.chunking_text import (
    clause_sources_for_text,
    extract_article_heading,
    extract_part_heading,
    extract_section_heading,
    heading_path_for_block,
    section_body_for_page,
    source_block_ids_for_block,
    split_microchunks,
)
from src.indexing.text_features import extract_dates, extract_entities, ordered_unique, tokenize_text

MAX_BM25_TERMS_METADATA = 32


@dataclass(slots=True)
class HeadingContext:
    """Rolling heading context preserved across continuation pages."""

    part_label: str | None = None
    article_label: str | None = None
    section_label_value: str | None = None

    def merged_with(self, text: str) -> "HeadingContext":
        """Return heading context updated by explicit headings from the current page."""

        explicit_part = extract_part_heading(text)
        explicit_article = extract_article_heading(text)
        explicit_section = extract_section_heading(text)
        if explicit_part and explicit_part != self.part_label and explicit_article is None and explicit_section is None:
            return HeadingContext(part_label=explicit_part, article_label=None, section_label_value=None)
        if explicit_article and explicit_article != self.article_label and explicit_section is None:
            return HeadingContext(
                part_label=explicit_part or self.part_label,
                article_label=explicit_article,
                section_label_value=None,
            )
        return HeadingContext(
            part_label=explicit_part or self.part_label,
            article_label=explicit_article or self.article_label,
            section_label_value=explicit_section or self.section_label_value,
        )

    def neighboring_headings(self) -> list[str]:
        """Return ordered non-empty heading lineage for the current page."""

        headings: list[str] = []
        if self.part_label:
            headings.append(self.part_label)
        if self.article_label:
            headings.append(self.article_label)
        if self.section_label_value:
            headings.append(self.section_label_value)
        return headings

    def section_label(self) -> str:
        """Return strongest available section-like heading label."""

        if self.section_label_value:
            return self.section_label_value
        if self.article_label:
            return self.article_label
        if self.part_label:
            return self.part_label
        return ""


@dataclass(slots=True)
class PageChunkContext:
    """Per-page heading context used to derive higher-level chunks."""

    page: CanonicalPageRecord
    headings: HeadingContext


@dataclass(slots=True)
class IndexChunk:
    """Intermediate indexing unit persisted into Qdrant."""

    chunk_id: str
    doc_id: str
    page_span: list[int]
    chunk_type: str
    text: str
    section: str
    clause: str | None
    neighboring_headings: list[str]
    entities: list[str]
    dates: list[str]
    token_count: int
    bm25_terms: list[str]
    document_title: str | None = None
    document_family: str | None = None
    parser_provenance: str | None = None
    heading_path: list[str] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    parent_page_numbers: list[int] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        """Serialize chunk payload stored in the vector index."""

        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "page_span": list(self.page_span),
            "parent_page_numbers": list(self.parent_page_numbers or self.page_span),
            "chunk_type": self.chunk_type,
            "text": self.text,
            "section": self.section,
            "clause": self.clause,
            "neighboring_headings": list(self.neighboring_headings),
            "entities": list(self.entities),
            "dates": list(self.dates),
            "token_count": self.token_count,
            "bm25_terms": list(self.bm25_terms),
            "document_title": self.document_title,
            "document_family": self.document_family,
            "parser_provenance": self.parser_provenance,
            "heading_path": list(self.heading_path),
            "source_block_ids": list(self.source_block_ids),
        }


def build_index_chunks(
    pages: Iterable[CanonicalPageRecord],
    *,
    enabled_chunk_families: set[str] | None = None,
    token_chunk_size: int = 300,
    token_chunk_overlap: int = 50,
) -> list[IndexChunk]:
    """Derive structure-aware hybrid indexing chunks from canonical pages."""

    families = {
        str(name).strip()
        for name in (enabled_chunk_families or {"page", "section", "clause", "microchunk", "table"})
        if str(name).strip()
    }
    if not families:
        return []

    chunks: list[IndexChunk] = []
    pages_by_doc: dict[str, list[CanonicalPageRecord]] = defaultdict(list)
    for page in pages:
        pages_by_doc[page.doc_id].append(page)

    for doc_id in sorted(pages_by_doc):
        page_contexts: list[PageChunkContext] = []
        last_headings = HeadingContext()
        for page in sorted(pages_by_doc[doc_id], key=lambda item: item.page_number):
            current_headings = last_headings.merged_with(page.text)
            last_headings = current_headings
            page_contexts.append(PageChunkContext(page=page, headings=current_headings))

        parent_page_numbers_by_page = _parent_page_numbers_by_page(page_contexts)
        for page_context in page_contexts:
            page = page_context.page
            neighboring_headings = page_context.headings.neighboring_headings()
            section_title = page_context.headings.section_label()
            page_prefix = f"{page.doc_id}-p{page.page_number}"
            parent_page_numbers = parent_page_numbers_by_page.get(page.page_number, [page.page_number])

            if "page" in families:
                chunks.append(
                    _make_chunk(
                        chunk_id=f"{page_prefix}-page",
                        doc_id=page.doc_id,
                        page_span=[page.page_number],
                        parent_page_numbers=parent_page_numbers,
                        chunk_type="page",
                        text=page.text,
                        section=section_title,
                        clause=None,
                        neighboring_headings=neighboring_headings,
                        document_title=page.document_title,
                        document_family=page.document_family,
                        parser_provenance=page.parser_provenance,
                        heading_path=page.heading_path,
                        source_block_ids=page.source_block_ids,
                    )
                )

            if "clause" in families or "microchunk" in families:
                chunks.extend(
                    _clause_chunks_for_page(
                        page,
                        page_context.headings,
                        parent_page_numbers=parent_page_numbers,
                        include_clause_chunks="clause" in families,
                        include_microchunks="microchunk" in families,
                        token_chunk_size=token_chunk_size,
                        token_chunk_overlap=token_chunk_overlap,
                    )
                )

            if "table" in families:
                for table_block in page.blocks:
                    if table_block.type is not ContentBlockType.TABLE:
                        continue
                    chunks.append(
                        _make_chunk(
                            chunk_id=table_block.block_id,
                            doc_id=page.doc_id,
                            page_span=[page.page_number],
                            parent_page_numbers=parent_page_numbers,
                            chunk_type="table",
                            text=table_block.text,
                            section=section_title,
                            clause=table_block.metadata.get("row_anchor"),
                            neighboring_headings=neighboring_headings,
                            document_title=page.document_title,
                            document_family=page.document_family,
                            parser_provenance=page.parser_provenance,
                            heading_path=heading_path_for_block(page, table_block),
                            source_block_ids=source_block_ids_for_block(page, table_block),
                        )
                    )

        if "section" in families:
            chunks.extend(_section_chunks_for_doc(page_contexts))

    return chunks


def _clause_chunks_for_page(
    page: CanonicalPageRecord,
    headings: HeadingContext,
    *,
    parent_page_numbers: list[int],
    include_clause_chunks: bool,
    include_microchunks: bool,
    token_chunk_size: int,
    token_chunk_overlap: int,
) -> list[IndexChunk]:
    """Execute `_clause_chunks_for_page`.

    Args:
        page: Input parameter.
        headings: Input parameter.
        parent_page_numbers: Input parameter.
        include_clause_chunks: Input parameter.
        include_microchunks: Input parameter.
        token_chunk_size: Input parameter.
        token_chunk_overlap: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    clause_sources = clause_sources_for_text(page.text)
    chunks: list[IndexChunk] = []
    neighboring_headings = headings.neighboring_headings() or [f"Page {page.page_number}"]
    section_title = headings.section_label()

    for clause_index, (clause_label, clause_text) in enumerate(clause_sources, start=1):
        clause_name = clause_label
        clause_chunk_id = f"{page.doc_id}-p{page.page_number}-clause-{clause_index}"
        if include_clause_chunks:
            chunks.append(
                _make_chunk(
                    chunk_id=clause_chunk_id,
                    doc_id=page.doc_id,
                    page_span=[page.page_number],
                    parent_page_numbers=parent_page_numbers,
                    chunk_type="clause",
                    text=clause_text,
                    section=section_title,
                    clause=clause_name,
                    neighboring_headings=neighboring_headings,
                    document_title=page.document_title,
                    document_family=page.document_family,
                    parser_provenance=page.parser_provenance,
                    heading_path=page.heading_path,
                    source_block_ids=page.source_block_ids,
                )
            )

        if include_microchunks:
            for micro_index, micro_text in enumerate(
                split_microchunks(
                    clause_text,
                    token_chunk_size=token_chunk_size,
                    token_chunk_overlap=token_chunk_overlap,
                ),
                start=1,
            ):
                chunks.append(
                    _make_chunk(
                        chunk_id=f"{clause_chunk_id}-micro-{micro_index}",
                        doc_id=page.doc_id,
                        page_span=[page.page_number],
                        parent_page_numbers=parent_page_numbers,
                        chunk_type="microchunk",
                        text=micro_text,
                        section=section_title,
                        clause=clause_name,
                        neighboring_headings=neighboring_headings,
                        document_title=page.document_title,
                        document_family=page.document_family,
                        parser_provenance=page.parser_provenance,
                        heading_path=page.heading_path,
                        source_block_ids=page.source_block_ids,
                    )
                )
    return chunks


def _section_chunks_for_doc(page_contexts: list[PageChunkContext]) -> list[IndexChunk]:
    """Execute `_section_chunks_for_doc`.

    Args:
        page_contexts: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    section_chunks: list[IndexChunk] = []
    current_group: list[PageChunkContext] = []
    current_key: tuple[str, tuple[str, ...]] | None = None

    for page_context in page_contexts:
        section_label = page_context.headings.section_label()
        neighboring_headings = tuple(page_context.headings.neighboring_headings())
        if not section_label or not neighboring_headings:
            if current_group:
                section_chunks.append(_build_section_chunk(current_group, len(section_chunks) + 1))
                current_group = []
                current_key = None
            continue

        group_key = (section_label, neighboring_headings)
        if current_key is None or group_key == current_key:
            current_group.append(page_context)
            current_key = group_key
            continue

        section_chunks.append(_build_section_chunk(current_group, len(section_chunks) + 1))
        current_group = [page_context]
        current_key = group_key

    if current_group:
        section_chunks.append(_build_section_chunk(current_group, len(section_chunks) + 1))

    return section_chunks


def _parent_page_numbers_by_page(page_contexts: list[PageChunkContext]) -> dict[int, list[int]]:
    """Execute `_parent_page_numbers_by_page`.

    Args:
        page_contexts: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    parent_pages_by_page: dict[int, list[int]] = {}
    current_group: list[PageChunkContext] = []
    current_key: tuple[str, tuple[str, ...]] | None = None

    def flush_group() -> None:
        """Execute `flush_group`.

        Returns:
            None: This function does not return a value.
        """
        if not current_group:
            return
        group_pages = [page_context.page.page_number for page_context in current_group]
        for page_context in current_group:
            parent_pages_by_page[page_context.page.page_number] = list(group_pages)

    for page_context in page_contexts:
        section_label = page_context.headings.section_label()
        neighboring_headings = tuple(page_context.headings.neighboring_headings())
        if not section_label or not neighboring_headings:
            flush_group()
            current_group = []
            current_key = None
            parent_pages_by_page[page_context.page.page_number] = [page_context.page.page_number]
            continue

        group_key = (section_label, neighboring_headings)
        if current_key is None or group_key == current_key:
            current_group.append(page_context)
            current_key = group_key
            continue

        flush_group()
        current_group = [page_context]
        current_key = group_key

    flush_group()
    return parent_pages_by_page


def _build_section_chunk(page_contexts: list[PageChunkContext], section_index: int) -> IndexChunk:
    """Build section chunk.

    Args:
        page_contexts: Input parameter.
        section_index: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    first_page = page_contexts[0].page
    headings = page_contexts[0].headings
    section_text = "\n\n".join(section_body_for_page(page_context.page) for page_context in page_contexts).strip()
    page_span = [page_context.page.page_number for page_context in page_contexts]
    return _make_chunk(
        chunk_id=f"{first_page.doc_id}-section-{section_index}",
        doc_id=first_page.doc_id,
        page_span=page_span,
        parent_page_numbers=page_span,
        chunk_type="section",
        text=section_text or "\n\n".join(page_context.page.text.strip() for page_context in page_contexts).strip(),
        section=headings.section_label(),
        clause=None,
        neighboring_headings=headings.neighboring_headings(),
        document_title=first_page.document_title,
        document_family=first_page.document_family,
        parser_provenance=first_page.parser_provenance,
        heading_path=_combined_heading_path(page_contexts),
        source_block_ids=_combined_source_block_ids(page_contexts),
    )


def _make_chunk(
    *,
    chunk_id: str,
    doc_id: str,
    page_span: list[int],
    parent_page_numbers: list[int] | None = None,
    chunk_type: str,
    text: str,
    section: str,
    clause: str | None,
    neighboring_headings: list[str],
    document_title: str | None,
    document_family: str | None,
    parser_provenance: str | None = None,
    heading_path: list[str] | None = None,
    source_block_ids: list[str] | None = None,
) -> IndexChunk:
    """Execute `_make_chunk`.

    Args:
        chunk_id: Input parameter.
        doc_id: Input parameter.
        page_span: Input parameter.
        parent_page_numbers: Input parameter.
        chunk_type: Input parameter.
        text: Input parameter.
        section: Input parameter.
        clause: Input parameter.
        neighboring_headings: Input parameter.
        document_title: Input parameter.
        document_family: Input parameter.
        parser_provenance: Input parameter.
        heading_path: Input parameter.
        source_block_ids: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    normalized_text = text.strip()
    tokens = tokenize_text(normalized_text)
    return IndexChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        page_span=list(page_span),
        parent_page_numbers=list(parent_page_numbers or page_span),
        chunk_type=chunk_type,
        text=normalized_text,
        section=section,
        clause=clause,
        neighboring_headings=ordered_unique(neighboring_headings),
        entities=extract_entities(normalized_text),
        dates=extract_dates(normalized_text),
        token_count=len(tokens),
        bm25_terms=ordered_unique(tokens)[:MAX_BM25_TERMS_METADATA],
        document_title=document_title,
        document_family=document_family,
        parser_provenance=parser_provenance,
        heading_path=ordered_unique(heading_path or []),
        source_block_ids=ordered_unique(source_block_ids or []),
    )


def _combined_heading_path(page_contexts: list[PageChunkContext]) -> list[str]:
    """Execute `_combined_heading_path`.

    Args:
        page_contexts: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    combined: list[str] = []
    for page_context in page_contexts:
        combined.extend(page_context.page.heading_path)
    return ordered_unique(combined)


def _combined_source_block_ids(page_contexts: list[PageChunkContext]) -> list[str]:
    """Execute `_combined_source_block_ids`.

    Args:
        page_contexts: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    combined: list[str] = []
    for page_context in page_contexts:
        combined.extend(page_context.page.source_block_ids)
    return ordered_unique(combined)
