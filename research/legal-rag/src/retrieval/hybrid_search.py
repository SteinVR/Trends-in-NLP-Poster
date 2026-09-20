"""Scale-safe rank-fusion hybrid search for retrieval."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from qdrant_client import QdrantClient, models

from src.common.runtime_logging import get_logger, log_event, questions_debug_enabled
from src.common.schemas import QuestionRecord
from src.indexing.service import BM25SparseEncoder
from src.retrieval.introspection import callable_accepts_keyword, metadata_field

LOGGER = get_logger("retrieval.hybrid_search")


class SearchBackend(Protocol):
    """Backend interface that yields raw retrieval candidates."""

    def search(
        self,
        question: QuestionRecord,
        top_k: int,
        *,
        query_metadata: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Return raw candidates for a question."""


class DenseEmbedder(Protocol):
    """Minimal dense query embedder contract used by retrieval backends."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode text inputs as dense vectors."""


@dataclass(slots=True)
class QdrantHybridSearchBackend:
    """Dense-first Qdrant retrieval backend over persisted W3A artifacts."""

    index_dir: Path
    dense_embedder: DenseEmbedder
    collection_name: str | None = None
    candidate_multiplier: int = 3
    enable_sparse_compat: bool = False
    _client: QdrantClient | None = field(default=None, init=False, repr=False)
    _resolved_collection_name: str | None = field(default=None, init=False, repr=False)
    _sparse_encoder: BM25SparseEncoder | None = field(default=None, init=False, repr=False)
    _is_loaded: bool = field(default=False, init=False, repr=False)

    def search(
        self,
        question: QuestionRecord,
        top_k: int,
        *,
        query_metadata: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Execute `search`."""
        result_limit = max(int(top_k), 0)
        if result_limit == 0:
            return []

        self._ensure_loaded()
        assert self._client is not None
        assert self._resolved_collection_name is not None

        query_vectors = self.dense_embedder.encode([question.question])
        if len(query_vectors) != 1:
            raise ValueError("Dense embedder must return exactly one query vector.")
        dense_query = [float(value) for value in query_vectors[0]]

        dense_limit = max(result_limit * max(self.candidate_multiplier, 1), result_limit)
        dense_hits = self._client.query_points(
            collection_name=self._resolved_collection_name,
            query=dense_query,
            using="dense",
            limit=dense_limit,
            with_payload=True,
            with_vectors=False,
        ).points

        sparse_hits: list[models.ScoredPoint] = []
        if self._sparse_encoder is not None:
            sparse_hits = self._client.query_points(
                collection_name=self._resolved_collection_name,
                query=self._sparse_encoder.encode(question.question),
                using="sparse",
                limit=dense_limit,
                with_payload=True,
                with_vectors=False,
            ).points

        merged = _merge_scored_hits(dense_hits=dense_hits, sparse_hits=sparse_hits)
        merged = _apply_query_metadata_boosts(merged, query_metadata=query_metadata)
        merged.sort(
            key=lambda candidate: (
                -candidate["retrieval_score"],
                str(candidate["chunk_id"]),
            )
        )
        window = merged[:result_limit]
        if questions_debug_enabled():
            log_event(
                LOGGER,
                logging.DEBUG,
                "retrieval.search.complete",
                "Hybrid search completed for question.",
                stage="retrieval",
                question_id=question.id,
                dense_hit_count=len(dense_hits),
                sparse_hit_count=len(sparse_hits),
                candidate_count=len(window),
            )
        return window

    def close(self) -> None:
        """Execute `close`."""
        if self._client is None:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        self._client = None
        self._resolved_collection_name = None
        self._sparse_encoder = None
        self._is_loaded = False

    def _ensure_loaded(self) -> None:
        """Execute `_ensure_loaded`."""
        if self._is_loaded:
            return

        manifest = _read_index_manifest(self.index_dir)
        self._resolved_collection_name = self.collection_name or str(manifest["collection_name"])
        qdrant_dir = self.index_dir / "qdrant"
        if not qdrant_dir.exists():
            raise ValueError(f"Qdrant index directory does not exist: {qdrant_dir}")

        self._client = QdrantClient(path=str(qdrant_dir))
        if self.enable_sparse_compat:
            sparse_state_path = _resolve_sparse_state_path(self.index_dir, manifest)
            if sparse_state_path.exists():
                sparse_state_payload = json.loads(sparse_state_path.read_text(encoding="utf-8"))
                self._sparse_encoder = BM25SparseEncoder.from_state(sparse_state_payload)
        self._is_loaded = True
        log_event(
            LOGGER,
            logging.INFO,
            "retrieval.backend.loaded",
            "Loaded persisted retrieval backend.",
            stage="retrieval",
            index_dir=self.index_dir,
            collection_name=self._resolved_collection_name,
            sparse_enabled=self._sparse_encoder is not None,
        )


@dataclass(slots=True)
class RRFHybridSearch:
    """Fuse dense and sparse rankings using reciprocal rank fusion."""

    backend: SearchBackend
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    def search(
        self,
        question: QuestionRecord,
        top_k: int,
        *,
        query_metadata: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Execute `search`."""
        result_limit = max(int(top_k), 0)
        if result_limit == 0:
            return []

        raw_candidates = [
            dict(candidate)
            for candidate in _backend_search_with_optional_query_metadata(
                self.backend,
                question,
                top_k=top_k,
                query_metadata=query_metadata,
            )
        ]
        if not raw_candidates:
            return []

        dense_ranks = self._rank_candidates(raw_candidates, primary_key="dense_score", fallback_key="retrieval_score")
        sparse_ranks = self._rank_candidates(raw_candidates, primary_key="sparse_score")
        ranked_candidates = [
            self._normalize_candidate(candidate, dense_ranks, sparse_ranks)
            for candidate in raw_candidates
        ]
        ranked_candidates.sort(
            key=lambda candidate: (
                -candidate["retrieval_score"],
                candidate["best_rank"],
                candidate["rank_sum"],
                candidate["chunk_id"],
            ),
        )
        return ranked_candidates[:result_limit]

    def _normalize_candidate(
        self,
        candidate: dict[str, Any],
        dense_ranks: dict[str, int],
        sparse_ranks: dict[str, int],
    ) -> dict[str, Any]:
        """Normalize candidate."""
        normalized = dict(candidate)
        chunk_id = str(candidate.get("chunk_id") or "")
        dense_rank = dense_ranks.get(chunk_id)
        sparse_rank = sparse_ranks.get(chunk_id)
        ranks = [rank for rank in (dense_rank, sparse_rank) if rank is not None]

        normalized["dense_score"] = _score(candidate.get("dense_score"), fallback=candidate.get("retrieval_score"))
        normalized["sparse_score"] = _optional_score(candidate.get("sparse_score"))
        normalized["dense_rank"] = dense_rank
        normalized["sparse_rank"] = sparse_rank
        normalized["retrieval_score"] = self._rrf_score(dense_rank, self.dense_weight) + self._rrf_score(
            sparse_rank,
            self.sparse_weight,
        )
        normalized["best_rank"] = min(ranks) if ranks else 10**9
        normalized["rank_sum"] = sum(ranks) if ranks else 10**9
        return normalized

    def _rank_candidates(
        self,
        candidates: list[dict[str, Any]],
        *,
        primary_key: str,
        fallback_key: str | None = None,
    ) -> dict[str, int]:
        """Execute `_rank_candidates`."""
        scored_candidates = []
        for candidate in candidates:
            chunk_id = str(candidate.get("chunk_id") or "")
            score = _optional_score(candidate.get(primary_key))
            if score is None and fallback_key is not None:
                score = _optional_score(candidate.get(fallback_key))
            if not chunk_id or score is None:
                continue
            scored_candidates.append((chunk_id, score))

        scored_candidates.sort(key=lambda item: (-item[1], item[0]))
        return {
            chunk_id: rank
            for rank, (chunk_id, _score_value) in enumerate(scored_candidates, start=1)
        }

    def _rrf_score(self, rank: int | None, weight: float) -> float:
        """Execute `_rrf_score`."""
        if rank is None:
            return 0.0
        return float(weight) / (self.rrf_k + rank)


def _score(value: Any, *, fallback: Any = None) -> float:
    """Execute `_score`."""
    if value is None:
        value = fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_score(value: Any) -> float | None:
    """Execute `_optional_score`."""
    if value is None:
        return None
    return _score(value)


def _read_index_manifest(index_dir: Path) -> dict[str, Any]:
    """Execute `_read_index_manifest`."""
    manifest_path = index_dir / "index_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Index manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Index manifest must be a JSON object.")
    collection_name = payload.get("collection_name")
    if not isinstance(collection_name, str) or not collection_name:
        raise ValueError("Index manifest must include a non-empty collection_name.")
    return payload


def _resolve_sparse_state_path(index_dir: Path, manifest: dict[str, Any]) -> Path:
    """Resolve sparse state path."""
    sparse_state = manifest.get("sparse_encoder_state")
    if isinstance(sparse_state, dict):
        state_path_raw = sparse_state.get("path")
        if isinstance(state_path_raw, str) and state_path_raw.strip():
            state_path = Path(state_path_raw)
            if state_path.is_absolute():
                return state_path
            manifest_relative_path = index_dir / state_path
            if manifest_relative_path.exists() or not state_path.exists():
                return manifest_relative_path
            if state_path.exists():
                return state_path
    return index_dir / "sparse_encoder_state.json"


def _merge_scored_hits(
    *,
    dense_hits: list[models.ScoredPoint],
    sparse_hits: list[models.ScoredPoint],
) -> list[dict[str, Any]]:
    """Execute `_merge_scored_hits`."""
    merged: dict[str, dict[str, Any]] = {}

    for hit in dense_hits:
        payload = dict(hit.payload or {})
        chunk_id = _extract_chunk_id(payload, hit.id)
        candidate = merged.get(chunk_id)
        if candidate is None:
            candidate = _candidate_from_payload(chunk_id, payload)
            merged[chunk_id] = candidate
        candidate["dense_score"] = float(hit.score)

    for hit in sparse_hits:
        payload = dict(hit.payload or {})
        chunk_id = _extract_chunk_id(payload, hit.id)
        candidate = merged.get(chunk_id)
        if candidate is None:
            candidate = _candidate_from_payload(chunk_id, payload)
            merged[chunk_id] = candidate
        candidate["sparse_score"] = float(hit.score)

    for candidate in merged.values():
        dense_score = _optional_score(candidate.get("dense_score"))
        sparse_score = _optional_score(candidate.get("sparse_score"))
        if dense_score is None and sparse_score is None:
            candidate["retrieval_score"] = 0.0
            continue
        candidate["retrieval_score"] = (dense_score or 0.0) + (sparse_score or 0.0)

    return list(merged.values())


def _extract_chunk_id(payload: dict[str, Any], point_id: Any) -> str:
    """Extract chunk id."""
    chunk_id = str(payload.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id
    return str(point_id)


def _candidate_from_payload(chunk_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute `_candidate_from_payload`."""
    page_span = payload.get("page_span")
    if not isinstance(page_span, list) or not page_span:
        page_span = payload.get("page_numbers")
    if not isinstance(page_span, list):
        page_span = []

    normalized_page_span: list[int] = []
    for page_number in page_span:
        try:
            parsed = int(page_number)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            normalized_page_span.append(parsed)

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    parent_page_numbers = payload.get("parent_page_numbers")
    if parent_page_numbers is None:
        parent_page_numbers = metadata.get("parent_page_numbers")

    return {
        "chunk_id": chunk_id,
        "doc_id": str(payload.get("doc_id") or ""),
        "page_span": normalized_page_span,
        "parent_page_numbers": _normalized_page_numbers(parent_page_numbers),
        "chunk_type": str(payload.get("chunk_type") or "unknown"),
        "text": str(payload.get("text") or ""),
        "section": str(payload.get("section") or ""),
        "clause": str(payload.get("clause") or ""),
        "neighboring_headings": _string_list(payload.get("neighboring_headings")),
        "entities": _string_list(payload.get("entities")),
        "document_title": str(payload.get("document_title") or ""),
        "document_family": str(payload.get("document_family") or ""),
        "metadata": metadata,
        "dense_score": None,
        "sparse_score": None,
        "retrieval_score": 0.0,
        "metadata_boost": 0.0,
    }


def _normalized_page_numbers(raw_page_numbers: Any) -> list[int]:
    """Execute `_normalized_page_numbers`."""
    if not isinstance(raw_page_numbers, list):
        return []

    normalized: list[int] = []
    for page_number in raw_page_numbers:
        try:
            parsed = int(page_number)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            normalized.append(parsed)
    return normalized


def _apply_query_metadata_boosts(
    candidates: list[dict[str, Any]],
    *,
    query_metadata: Any | None,
) -> list[dict[str, Any]]:
    """Execute `_apply_query_metadata_boosts`."""
    legal_entities = _query_metadata_signals(query_metadata, field_name="legal_entities")
    title_cues = _query_metadata_signals(query_metadata, field_name="document_title_cues")
    if not legal_entities and not title_cues:
        return [dict(candidate) for candidate in candidates]

    boosted_candidates: list[dict[str, Any]] = []
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        normalized_candidate_signals = _candidate_signals(candidate)

        entity_match_count = sum(
            1
            for signal in legal_entities
            if _signals_overlap(normalized_candidate_signals, signal)
        )
        title_match_count = sum(
            1
            for signal in title_cues
            if _signals_overlap(normalized_candidate_signals, signal)
        )

        metadata_boost = min((0.18 * entity_match_count) + (0.12 * title_match_count), 0.72)
        candidate["metadata_boost"] = metadata_boost
        candidate["retrieval_score"] = _score(candidate.get("retrieval_score")) + metadata_boost
        boosted_candidates.append(candidate)
    return boosted_candidates


def _query_metadata_signals(query_metadata: Any, *, field_name: str) -> tuple[str, ...]:
    """Execute `_query_metadata_signals`."""
    raw_values = metadata_field(query_metadata, field_name, ())
    normalized: list[str] = []

    for raw_value in raw_values:
        if field_name == "legal_entities":
            raw_value = metadata_field(raw_value, "normalized_value", "")
        signal = _normalize_signal(raw_value)
        if signal:
            normalized.append(signal)
    return tuple(dict.fromkeys(normalized))


def _candidate_signals(candidate: dict[str, Any]) -> tuple[str, ...]:
    """Execute `_candidate_signals`."""
    raw_signals: list[Any] = [
        candidate.get("doc_id"),
        candidate.get("document_title"),
        candidate.get("text"),
        candidate.get("section"),
        candidate.get("clause"),
        *candidate.get("neighboring_headings", []),
        *candidate.get("entities", []),
    ]
    normalized = [_normalize_signal(value) for value in raw_signals]
    return tuple(signal for signal in dict.fromkeys(normalized) if signal)


def _signals_overlap(candidate_signals: tuple[str, ...], target_signal: str) -> bool:
    """Execute `_signals_overlap`."""
    return any(
        target_signal in candidate_signal or candidate_signal in target_signal
        for candidate_signal in candidate_signals
        if candidate_signal
    )


def _normalize_signal(value: Any) -> str:
    """Normalize signal."""
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _string_list(value: Any) -> list[str]:
    """Execute `_string_list`."""
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _backend_search_with_optional_query_metadata(
    backend: SearchBackend,
    question: QuestionRecord,
    *,
    top_k: int,
    query_metadata: Any | None,
) -> list[dict[str, Any]]:
    """Execute `_backend_search_with_optional_query_metadata`."""
    if callable_accepts_keyword(backend.search, "query_metadata"):
        return backend.search(
            question,
            top_k=top_k,
            query_metadata=query_metadata,
        )
    return backend.search(question, top_k=top_k)
