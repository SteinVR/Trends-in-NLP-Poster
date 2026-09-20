"""Hybrid indexing service for canonical legal corpora."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from src.common.config import AppConfig
from src.common.file_hash import sha256_for_file
from src.common.runtime_logging import get_logger, log_event
from src.common.schemas import CanonicalPageRecord, CorpusParseManifest, PageMapRecord
from src.indexing.chunking_core import IndexChunk, build_index_chunks
from src.indexing.embedders import (
    BM25SparseEncoder,
    build_dense_embedder,
    build_query_embedder,
    dense_encoder_contract,
)
from src.indexing.state import load_page_parent_map
from src.indexing.text_features import tokenize_text
from src.submission.sync import PhasePaths

LOGGER = get_logger("indexing.service")
__all__ = [
    "BM25SparseEncoder",
    "BuildIndexResult",
    "HybridIndexService",
    "IndexPaths",
    "build_query_embedder",
    "format_build_index_result",
]


@dataclass(slots=True)
class IndexPaths:
    """Resolved filesystem layout for indexing outputs."""

    corpus_path: Path
    page_map_path: Path
    corpus_manifest_path: Path
    index_dir: Path
    qdrant_dir: Path
    page_parent_map_path: Path
    manifest_path: Path
    sparse_encoder_state_path: Path

    @classmethod
    def from_config(cls, config: AppConfig) -> "IndexPaths":
        """Resolve canonical paths for index artifacts from runtime config."""

        phase_paths = PhasePaths.from_config(config)
        index_dir = phase_paths.phase_dir / "index"
        return cls(
            corpus_path=phase_paths.corpus_path,
            page_map_path=phase_paths.page_map_path,
            corpus_manifest_path=phase_paths.corpus_manifest_path,
            index_dir=index_dir,
            qdrant_dir=index_dir / "qdrant",
            page_parent_map_path=index_dir / "page_parent_map.json",
            manifest_path=index_dir / "index_manifest.json",
            sparse_encoder_state_path=index_dir / "sparse_encoder_state.json",
        )

    def ensure_directories(self) -> None:
        """Create parent directories required for writing indexing artifacts."""

        self.index_dir.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class BuildIndexResult:
    """User-facing summary for a build-indices action."""

    command: str
    phase: str
    dry_run: bool
    manifest_path: Path
    planned_paths: list[Path] = field(default_factory=list)
    written_paths: list[Path] = field(default_factory=list)
    skipped_paths: list[Path] = field(default_factory=list)
    indexed_document_count: int | None = None
    indexed_page_count: int | None = None
    chunk_count: int | None = None


class HybridIndexService:
    """Build phase-local hybrid indices from canonical corpus artifacts."""

    collection_name = "legal_hybrid_index"

    def __init__(self, config: AppConfig) -> None:
        """Initialize the service with runtime configuration and path layout."""

        self.config = config
        self.paths = IndexPaths.from_config(config)

    def build_indices(
        self,
        *,
        dry_run: bool = False,
        overwrite: bool | None = None,
    ) -> BuildIndexResult:
        """Build dense+sparse retrieval artifacts for the configured phase."""

        overwrite = self.config.download.overwrite if overwrite is None else overwrite
        planned_paths = [*self._artifact_paths(), *self._optional_artifact_paths()]
        log_event(
            LOGGER,
            logging.INFO,
            "index.start",
            "Starting index build.",
            stage="indexing",
            dry_run=dry_run,
            overwrite=overwrite,
            corpus_path=self.paths.corpus_path,
            index_dir=self.paths.index_dir,
        )
        result = BuildIndexResult(
            command="build-indices",
            phase=self.config.phase.value,
            dry_run=dry_run,
            manifest_path=self.paths.manifest_path,
            planned_paths=planned_paths,
        )

        if dry_run:
            log_event(
                LOGGER,
                logging.INFO,
                "index.dry_run",
                "Index build dry-run completed.",
                stage="indexing",
                planned_paths=planned_paths,
            )
            return result

        corpus_pages, page_map_records, corpus_manifest = self._load_corpus_inputs()
        self._validate_page_attribution(corpus_pages, page_map_records)
        source_manifest = self._build_source_manifest(corpus_pages, page_map_records, corpus_manifest)
        result.indexed_document_count = source_manifest["indexed_document_count"]
        result.indexed_page_count = source_manifest["indexed_page_count"]

        if not overwrite:
            existing_manifest = self._load_existing_manifest(source_manifest)
            if existing_manifest is not None:
                result.skipped_paths.extend(path for path in planned_paths if path.exists())
                result.chunk_count = existing_manifest.get("chunk_count")
                log_event(
                    LOGGER,
                    logging.INFO,
                    "index.skip_cache",
                    "Skipped index build because artifacts are up to date.",
                    stage="indexing",
                    chunk_count=result.chunk_count,
                    skipped_paths=result.skipped_paths,
                )
                return result

        enabled_chunk_families = set(self.config.indexing.enabled_chunk_families)
        chunks = build_index_chunks(
            corpus_pages,
            enabled_chunk_families=enabled_chunk_families,
            token_chunk_size=self.config.indexing.token_chunk_size,
            token_chunk_overlap=self.config.indexing.token_chunk_overlap,
        )
        dense_embedder = build_dense_embedder(self.config)
        embedding_started_at = time.perf_counter()
        dense_vectors = dense_embedder.encode([chunk.text for chunk in chunks])
        embedding_duration_ms = round((time.perf_counter() - embedding_started_at) * 1000, 3)
        if len(dense_vectors) != len(chunks):
            raise ValueError("Dense embedder returned a mismatched number of vectors for indexing chunks.")
        sparse_encoder = BM25SparseEncoder([chunk.text for chunk in chunks])

        self.paths.ensure_directories()
        if self.paths.qdrant_dir.exists():
            shutil.rmtree(self.paths.qdrant_dir)
        self.paths.qdrant_dir.mkdir(parents=True, exist_ok=True)

        page_parent_map = self._write_qdrant_index(chunks, dense_vectors, sparse_encoder)
        manifest_payload = {
            **source_manifest,
            "command": "build-indices",
            "phase": self.config.phase.value,
            "collection_name": self.collection_name,
            "chunk_count": len(chunks),
            "built_at": datetime.now(tz=UTC).isoformat(),
            "chunking": {
                "levels": sorted(enabled_chunk_families),
                "carry_heading_context": True,
                "unnumbered_clause_strategy": "paragraph-synthesis",
                "token_chunk_size": self.config.indexing.token_chunk_size,
                "token_chunk_overlap": self.config.indexing.token_chunk_overlap,
            },
            "dense_encoder": dense_encoder_contract(dense_embedder),
            "sparse_encoder": sparse_encoder.metadata(),
            "sparse_encoder_state": {
                "path": self.paths.sparse_encoder_state_path.name,
            },
        }

        self._write_json(self.paths.page_parent_map_path, page_parent_map)
        self._write_json(self.paths.sparse_encoder_state_path, sparse_encoder.export_state())
        manifest_payload["sparse_encoder_state"]["sha256"] = sha256_for_file(self.paths.sparse_encoder_state_path)
        self._write_json(self.paths.manifest_path, manifest_payload)

        result.written_paths.extend(planned_paths)
        result.chunk_count = len(chunks)
        log_event(
            LOGGER,
            logging.INFO,
            "index.complete",
            "Index build completed.",
            stage="indexing",
            indexed_document_count=result.indexed_document_count,
            indexed_page_count=result.indexed_page_count,
            chunk_count=result.chunk_count,
            embedding_duration_ms=embedding_duration_ms,
            dense_model=getattr(dense_embedder, "model_name", None),
            written_paths=result.written_paths,
        )
        return result

    def _artifact_paths(self) -> list[Path]:
        """Return required artifact paths for a valid built index."""

        return [
            self.paths.qdrant_dir,
            self.paths.page_parent_map_path,
            self.paths.manifest_path,
        ]

    def _optional_artifact_paths(self) -> list[Path]:
        """Return optional sidecar artifact paths produced by index building."""

        return [self.paths.sparse_encoder_state_path]

    def _load_corpus_inputs(
        self,
    ) -> tuple[list[CanonicalPageRecord], list[PageMapRecord], CorpusParseManifest]:
        """Load corpus/page-map/manifest artifacts required for indexing."""

        required_paths = [
            self.paths.corpus_path,
            self.paths.page_map_path,
            self.paths.corpus_manifest_path,
        ]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            raise ValueError(
                "Indexing requires existing corpus artifacts before running: "
                f"missing {', '.join(str(path) for path in missing_paths)}. "
                "Run parse-corpus first."
            )

        corpus_pages: list[CanonicalPageRecord] = []
        with self.paths.corpus_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                corpus_pages.append(CanonicalPageRecord.model_validate_json(line))

        page_map_payload = json.loads(self.paths.page_map_path.read_text(encoding="utf-8"))
        page_map_records = [PageMapRecord.model_validate(record) for record in page_map_payload]
        corpus_manifest = CorpusParseManifest.model_validate_json(
            self.paths.corpus_manifest_path.read_text(encoding="utf-8")
        )
        return corpus_pages, page_map_records, corpus_manifest

    def _validate_page_attribution(
        self,
        corpus_pages: list[CanonicalPageRecord],
        page_map_records: list[PageMapRecord],
    ) -> None:
        """Validate corpus page map alignment to prevent attribution drift."""

        corpus_page_numbers: dict[str, list[int]] = {}
        for page in corpus_pages:
            corpus_page_numbers.setdefault(page.doc_id, []).append(page.page_number)

        page_map_by_doc = {record.doc_id: sorted(record.page_numbers) for record in page_map_records}
        corpus_by_doc = {doc_id: sorted(page_numbers) for doc_id, page_numbers in corpus_page_numbers.items()}
        if corpus_by_doc != page_map_by_doc:
            raise ValueError(
                "Corpus page attribution does not match the page map; "
                "rebuild parse-corpus inputs before build-indices."
            )

    def _build_source_manifest(
        self,
        corpus_pages: list[CanonicalPageRecord],
        page_map_records: list[PageMapRecord],
        corpus_manifest: CorpusParseManifest,
    ) -> dict[str, Any]:
        """Build input artifact digest payload included in index manifest."""

        return {
            "source_artifacts": {
                "corpus_path": str(self.paths.corpus_path),
                "corpus_sha256": sha256_for_file(self.paths.corpus_path),
                "page_map_path": str(self.paths.page_map_path),
                "page_map_sha256": sha256_for_file(self.paths.page_map_path),
                "corpus_manifest_path": str(self.paths.corpus_manifest_path),
                "corpus_manifest_sha256": sha256_for_file(self.paths.corpus_manifest_path),
            },
            "indexed_document_count": len(page_map_records),
            "indexed_page_count": len(corpus_pages),
            "parse_document_count": corpus_manifest.document_count,
            "parse_page_count": corpus_manifest.page_count,
        }

    def _load_existing_manifest(self, source_manifest: dict[str, Any]) -> dict[str, Any] | None:
        """Validate existing index artifacts and return manifest when cache is reusable."""

        if any(not path.exists() for path in self._artifact_paths()):
            return None

        client: QdrantClient | None = None
        try:
            manifest_payload = json.loads(self.paths.manifest_path.read_text(encoding="utf-8"))
            if manifest_payload.get("source_artifacts") != source_manifest["source_artifacts"]:
                return None

            client = QdrantClient(path=str(self.paths.qdrant_dir))
            collections = client.get_collections().collections
            if len(collections) != 1:
                return None
            collection_name = collections[0].name
            collection_info = client.get_collection(collection_name)
            chunk_count = manifest_payload.get("chunk_count")
            if not isinstance(chunk_count, int) or chunk_count <= 0:
                return None
            if collection_info.points_count == 0:
                return None
            if collection_info.points_count != chunk_count:
                return None
            if load_page_parent_map(path=self.paths.page_parent_map_path, expected_chunk_count=chunk_count) is None:
                return None
            return manifest_payload
        except Exception as exc:
            raise RuntimeError(
                "Existing index artifacts failed integrity validation in strict mode."
            ) from exc
        finally:
            if client is not None:
                _close_client(client)

    def _write_qdrant_index(
        self,
        chunks: list[IndexChunk],
        dense_vectors: list[list[float]],
        sparse_encoder: BM25SparseEncoder,
    ) -> dict[str, dict[str, Any]]:
        """Write vector collection and return page-parent map sidecar payload."""

        if not chunks:
            raise ValueError("No index chunks were generated from the canonical corpus.")

        client = QdrantClient(path=str(self.paths.qdrant_dir))
        try:
            if client.collection_exists(self.collection_name):
                client.delete_collection(self.collection_name)
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=len(dense_vectors[0]),
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )

            points: list[models.PointStruct] = []
            page_parent_map: dict[str, dict[str, Any]] = {}
            for chunk, dense_vector in zip(chunks, dense_vectors, strict=True):
                points.append(
                    models.PointStruct(
                        id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                        vector={
                            "dense": dense_vector,
                            "sparse": sparse_encoder.encode_tokens(tokenize_text(chunk.text)),
                        },
                        payload=chunk.payload(),
                    )
                )
                page_parent_map[chunk.chunk_id] = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "page_span": list(chunk.page_span),
                    "page_numbers": list(chunk.page_span),
                    "parent_page_numbers": list(chunk.parent_page_numbers or chunk.page_span),
                    "chunk_type": chunk.chunk_type,
                    "section": chunk.section,
                    "clause": chunk.clause,
                    "neighboring_headings": list(chunk.neighboring_headings),
                    "parser_provenance": chunk.parser_provenance,
                    "heading_path": list(chunk.heading_path),
                    "source_block_ids": list(chunk.source_block_ids),
                }

            client.upsert(collection_name=self.collection_name, points=points, wait=True)
            return page_parent_map
        finally:
            _close_client(client)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        """Write pretty JSON payload with trailing newline."""

        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def format_build_index_result(result: BuildIndexResult) -> str:
    """Return a human-readable build-indices summary."""

    lines = [
        f"command: {result.command}",
        f"phase: {result.phase}",
        f"dry_run: {str(result.dry_run).lower()}",
        f"manifest: {result.manifest_path}",
    ]
    if result.indexed_document_count is not None:
        lines.append(f"indexed_document_count: {result.indexed_document_count}")
    if result.indexed_page_count is not None:
        lines.append(f"indexed_page_count: {result.indexed_page_count}")
    if result.chunk_count is not None:
        lines.append(f"chunk_count: {result.chunk_count}")
    if result.planned_paths:
        lines.append("planned_paths:")
        lines.extend(f"  - {path}" for path in result.planned_paths)
    if result.written_paths:
        lines.append("written_paths:")
        lines.extend(f"  - {path}" for path in result.written_paths)
    if result.skipped_paths:
        lines.append("skipped_paths:")
        lines.extend(f"  - {path}" for path in result.skipped_paths)
    return "\n".join(lines)


def _close_client(client: QdrantClient) -> None:
    """Close Qdrant client if the runtime exposes a close method."""

    close = getattr(client, "close", None)
    if callable(close):
        close()
