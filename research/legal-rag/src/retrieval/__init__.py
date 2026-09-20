"""Public retrieval interfaces and baseline implementations for the retrieval stage."""

from src.retrieval.evidence_compressor import SupportPageEvidenceCompressor
from src.retrieval.hybrid_search import QdrantHybridSearchBackend, RRFHybridSearch
from src.retrieval.page_lifter import LiftedPage, PageLifter, extract_physical_pages
from src.retrieval.reranker import QwenReranker, TransformersQwenRerankerBackend
from src.retrieval.service import RetrievalResult, RetrievalService, RetrievedChunk

__all__ = (
    "LiftedPage",
    "PageLifter",
    "QdrantHybridSearchBackend",
    "QwenReranker",
    "RRFHybridSearch",
    "RetrievalResult",
    "RetrievalService",
    "RetrievedChunk",
    "SupportPageEvidenceCompressor",
    "TransformersQwenRerankerBackend",
    "extract_physical_pages",
)
