"""Dense and sparse encoder implementations for hybrid indexing."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

from huggingface_hub import snapshot_download
from qdrant_client import models

from src.common.config import AppConfig
from src.indexing.text_features import tokenize_text

DEFAULT_QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QWEN3_PROMPT_NAME = "document"
DEFAULT_EMBED_BATCH_SIZE = 16
BM25_K1 = 1.5
BM25_B = 0.75
LOGGER = logging.getLogger(__name__)


class DenseEmbedder:
    """Dense embedding protocol for the indexing service."""

    model_name: str
    prompt_name: str | None
    normalize_embeddings: bool

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into dense float vectors."""

        raise NotImplementedError


class Qwen3DenseEmbedder(DenseEmbedder):
    """Local Qwen3 embedding path used by the indexing baseline."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_QWEN3_EMBEDDING_MODEL,
        prompt_name: str | None = DEFAULT_QWEN3_PROMPT_NAME,
        normalize_embeddings: bool = True,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        local_files_only: bool = True,
        device: str | None = None,
    ) -> None:
        """Execute `__init__`.

        Args:
            model_name: Input parameter.
            prompt_name: Input parameter.
            normalize_embeddings: Input parameter.
            batch_size: Input parameter.
            local_files_only: Input parameter.
            device: Input parameter.

        Returns:
            None: This initializer does not return a value.
        """
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self.device = device
        self._model: Any | None = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts with SentenceTransformers in strict local-files mode."""

        if not texts:
            return []

        model = self._load_model()
        vectors = self._encode_with_adaptive_batch(model, texts)

        if hasattr(vectors, "tolist"):
            return [[float(value) for value in row] for row in vectors.tolist()]
        return [[float(value) for value in row] for row in vectors]

    def _encode_with_adaptive_batch(self, model: Any, texts: list[str]) -> Any:
        """Encode texts and reduce batch size on CUDA OOM until successful."""

        batch_size = max(int(self.batch_size), 1)
        while True:
            encode_kwargs: dict[str, Any] = {
                "batch_size": batch_size,
                "convert_to_numpy": True,
                "normalize_embeddings": self.normalize_embeddings,
                "show_progress_bar": False,
            }
            if self.prompt_name:
                encode_kwargs["prompt_name"] = self.prompt_name

            try:
                try:
                    return model.encode(texts, **encode_kwargs)
                except TypeError:
                    encode_kwargs.pop("prompt_name", None)
                    return model.encode(texts, **encode_kwargs)
            except Exception as exc:
                if not _is_torch_cuda_oom(exc) or batch_size == 1:
                    raise
                next_batch_size = max(batch_size // 2, 1)
                LOGGER.warning(
                    "Dense embedding hit CUDA OOM at batch_size=%d; retrying with batch_size=%d.",
                    batch_size,
                    next_batch_size,
                )
                _try_release_torch_cuda_cache()
                batch_size = next_batch_size

    def _load_model(self) -> Any:
        """Load and cache SentenceTransformer with strict runtime checks."""

        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required to build dense indices with Qwen3-Embedding-0.6B."
            ) from exc

        runtime_device = self._resolve_runtime_device()
        model_source = self.model_name
        if self.local_files_only:
            try:
                model_source = snapshot_download(
                    repo_id=self.model_name,
                    local_files_only=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Qwen3 embedding model '{self.model_name}' is not available in local Hugging Face cache. "
                    "Populate the local snapshot first (offline mode is enabled)."
                ) from exc

        try:
            self._model = SentenceTransformer(
                model_source,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
                device=runtime_device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Qwen3 embedding model '{self.model_name}' could not be resolved from the current local "
                "runtime/cache. Query-time retrieval initializes SentenceTransformer with local_files_only=True, "
                "but the tokenizer/model loader still raised an exception. Inspect the chained exception for the "
                "real cause instead of assuming the local snapshot is missing."
            ) from exc
        return self._model

    def _resolve_runtime_device(self) -> str:
        """Resolve the preferred runtime device for sentence-transformers."""

        if self.device:
            normalized = str(self.device).strip()
            if not normalized.lower().startswith("cuda"):
                raise RuntimeError(
                    f"Qwen3 embedding model requires a CUDA device, got device={normalized!r}."
                )
            return normalized
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "Qwen3 embedding model requires CUDA, but torch runtime is unavailable."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen3 embedding model requires CUDA, but torch.cuda.is_available() is False."
            )
        return "cuda"


class BM25SparseEncoder:
    """BM25 Okapi sparse encoder with retrieval-facing metadata."""

    def __init__(
        self,
        texts: list[str],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        """Execute `__init__`.

        Args:
            texts: Input parameter.
            k1: Input parameter.
            b: Input parameter.

        Returns:
            None: This initializer does not return a value.
        """
        self.k1 = k1
        self.b = b
        tokenized_texts = [tokenize_text(text) for text in texts]
        self.document_count = len(tokenized_texts)
        self.document_lengths = [max(len(tokens), 1) for tokens in tokenized_texts]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths) if self.document_lengths else 1.0
        )

        document_frequency: Counter[str] = Counter()
        for tokens in tokenized_texts:
            document_frequency.update(set(tokens))
        if not document_frequency:
            document_frequency["_empty"] = 1

        self.term_index = {term: index for index, term in enumerate(sorted(document_frequency))}
        self.inverse_document_frequency = {
            term: math.log(1.0 + ((self.document_count - frequency + 0.5) / (frequency + 0.5)))
            for term, frequency in document_frequency.items()
        }

    def encode(self, text: str) -> models.SparseVector:
        """Encode raw text into a BM25 sparse vector."""

        return self.encode_tokens(tokenize_text(text))

    def encode_tokens(self, tokens: list[str]) -> models.SparseVector:
        """Encode pre-tokenized text into a BM25 sparse vector."""

        frequencies = Counter(tokens or ["_empty"])
        document_length = max(len(tokens), 1)
        weighted_terms: list[tuple[int, float]] = []

        for term, frequency in frequencies.items():
            term_id = self.term_index.get(term)
            if term_id is None:
                continue
            idf = self.inverse_document_frequency.get(term, 0.0)
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * (document_length / self.average_document_length)
            )
            weight = idf * ((frequency * (self.k1 + 1.0)) / denominator)
            weighted_terms.append((term_id, float(weight)))

        weighted_terms.sort(key=lambda item: item[0])
        if not weighted_terms:
            weighted_terms = [(0, 0.0)]
        return models.SparseVector(
            indices=[index for index, _ in weighted_terms],
            values=[value for _, value in weighted_terms],
        )

    def metadata(self) -> dict[str, Any]:
        """Return sparse-encoder metadata stored in the index manifest."""

        return {
            "scheme": "bm25_okapi",
            "tokenizer": "regex-lowercase",
            "k1": self.k1,
            "b": self.b,
            "average_document_length": self.average_document_length,
            "document_count": self.document_count,
            "vocabulary_size": len(self.term_index),
        }

    def export_state(self) -> dict[str, Any]:
        """Serialize sparse encoder state for retrieval reuse."""

        return {
            "version": 1,
            "metadata": self.metadata(),
            "term_index": dict(self.term_index),
            "inverse_document_frequency": dict(self.inverse_document_frequency),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "BM25SparseEncoder":
        """Restore sparse encoder from serialized state payload."""

        metadata = state.get("metadata")
        term_index_raw = state.get("term_index")
        idf_raw = state.get("inverse_document_frequency")
        if not isinstance(metadata, Mapping):
            raise ValueError("sparse encoder state must include metadata.")
        if not isinstance(term_index_raw, Mapping) or not term_index_raw:
            raise ValueError("sparse encoder state must include term_index mapping.")
        if not isinstance(idf_raw, Mapping) or not idf_raw:
            raise ValueError("sparse encoder state must include inverse_document_frequency mapping.")

        encoder = cls.__new__(cls)
        encoder.k1 = float(metadata.get("k1", BM25_K1))
        encoder.b = float(metadata.get("b", BM25_B))
        encoder.document_count = int(metadata.get("document_count", 1))
        encoder.average_document_length = float(metadata.get("average_document_length", 1.0))
        encoder.document_lengths = [1] * max(encoder.document_count, 1)
        encoder.term_index = {str(term): int(index) for term, index in term_index_raw.items()}
        encoder.inverse_document_frequency = {
            str(term): float(weight) for term, weight in idf_raw.items() if str(term) in encoder.term_index
        }
        if not encoder.inverse_document_frequency:
            raise ValueError("sparse encoder inverse_document_frequency cannot be empty.")
        return encoder


def build_dense_embedder(_settings: AppConfig) -> DenseEmbedder:
    """Build the dense embedder used for index construction."""

    return Qwen3DenseEmbedder()


def build_query_embedder(_settings: AppConfig) -> DenseEmbedder:
    """Build the dense embedder used for query-time retrieval."""

    return Qwen3DenseEmbedder(prompt_name="query")


def dense_encoder_contract(embedder: DenseEmbedder | None = None) -> dict[str, Any]:
    """Return the dense-encoder contract that retrieval can rely on."""

    model_name = getattr(embedder, "model_name", DEFAULT_QWEN3_EMBEDDING_MODEL)
    prompt_name = getattr(embedder, "prompt_name", DEFAULT_QWEN3_PROMPT_NAME)
    normalize_embeddings = getattr(embedder, "normalize_embeddings", True)
    return {
        "provider": "sentence_transformers",
        "family": "qwen3_embedding",
        "model_name": model_name,
        "prompt_name": prompt_name,
        "normalize_embeddings": normalize_embeddings,
        "local_files_only": True,
    }


def _is_torch_cuda_oom(exc: Exception) -> bool:
    """Return whether the raised exception indicates CUDA out-of-memory."""

    message = str(exc).lower()
    if "cuda out of memory" in message or "outofmemoryerror" in message:
        return True
    try:
        import torch

        return isinstance(exc, torch.OutOfMemoryError)
    except Exception:
        return False


def _try_release_torch_cuda_cache() -> None:
    """Best-effort CUDA cache release after OOM before retrying."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return
