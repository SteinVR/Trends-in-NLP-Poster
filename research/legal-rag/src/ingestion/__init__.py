"""Corpus parsing stage exports."""

from .corpus_builder import (
    CorpusArtifacts,
    CorpusBuilder,
    CorpusBuildResult,
    build_corpus_artifacts,
    render_debug_markdown,
)
from .ocr_engine import (
    OCREngine,
    OcrEngine,
    OcrPageResult,
    OcrTableExtraction,
    OCRUnavailableError,
    PaddleOcrAdapter,
    build_ocr_engine,
)
from .opendataloader_adapter import build_opendataloader_sidecar, route_provenance
from .page_triage import (
    PageSignals,
    QualityBreakdown,
    classify_page,
    compute_quality_score,
    quality_score_and_flags,
    score_page_quality,
    triage_page,
)
from .parser_routing import (
    DocumentStructuralProfile,
    ParserAvailability,
    ParserRoutingError,
    RouteBatch,
    plan_route_batches,
    resolve_parser_route,
)
from .pdf_parser import (
    DocumentParser,
    DocumentParseResult,
    PageParseResult,
    ParsedDocument,
    ParsedPage,
    ParsedTableCandidate,
    TableCandidate,
    parse_pdf_document,
)
from .service import CorpusParseService, ParseCorpusResult, format_parse_result
from .sidecars import load_raw_parse_sidecar, write_raw_parse_sidecar
from .structural_normalizer import normalize_structural_sidecar
from .table_serializer import (
    PageTableAssignments,
    TableSerializer,
    detect_tables,
    extract_document_tables,
    merge_adjacent_tables,
    serialize_document_tables,
    serialize_table,
)

__all__ = [
    "CorpusArtifacts",
    "CorpusBuildResult",
    "CorpusBuilder",
    "CorpusParseService",
    "DocumentParseResult",
    "DocumentStructuralProfile",
    "DocumentParser",
    "OCREngine",
    "OCRUnavailableError",
    "OcrEngine",
    "OcrPageResult",
    "OcrTableExtraction",
    "build_opendataloader_sidecar",
    "PageParseResult",
    "PageSignals",
    "PageTableAssignments",
    "ParserAvailability",
    "ParserRoutingError",
    "PaddleOcrAdapter",
    "ParsedDocument",
    "ParsedPage",
    "ParsedTableCandidate",
    "ParseCorpusResult",
    "QualityBreakdown",
    "RouteBatch",
    "TableCandidate",
    "TableSerializer",
    "build_corpus_artifacts",
    "build_ocr_engine",
    "classify_page",
    "compute_quality_score",
    "detect_tables",
    "extract_document_tables",
    "format_parse_result",
    "merge_adjacent_tables",
    "load_raw_parse_sidecar",
    "normalize_structural_sidecar",
    "parse_pdf_document",
    "plan_route_batches",
    "quality_score_and_flags",
    "resolve_parser_route",
    "render_debug_markdown",
    "route_provenance",
    "score_page_quality",
    "serialize_document_tables",
    "serialize_table",
    "triage_page",
    "write_raw_parse_sidecar",
]
