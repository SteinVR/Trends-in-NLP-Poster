"""Qwen3-oriented reranking behavior for retrieval candidates."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from huggingface_hub import snapshot_download

from src.common.runtime_logging import get_logger, log_event, questions_debug_enabled
from src.common.schemas import QuestionRecord

LOGGER = get_logger("retrieval.reranker")

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
DEFAULT_QWEN3_RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
DEFAULT_QWEN3_RERANK_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".'
)
DEFAULT_QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "for",
        "how",
        "in",
        "is",
        "of",
        "or",
        "the",
        "to",
        "what",
        "which",
        "who",
    }
)


class QwenRerankerBackend(Protocol):
    """Backend interface for scoring query/document pairs with a reranker model."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one rerank score per candidate document."""


class CandidateReranker(Protocol):
    """Fallback interface matching the retrieval-service reranker contract."""

    def rerank(
        self,
        question: QuestionRecord,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return reranked candidates."""


@dataclass(slots=True)
class TransformersQwenRerankerBackend:
    """Lazy Transformers backend for `Qwen/Qwen3-Reranker-0.6B`.

    The official Qwen reranker checkpoint is a causal LM. Rerank scores come from the
    next-token probability of answering "yes" versus "no" for a formatted query/document prompt.
    """

    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    device: str | None = None
    batch_size: int = 8
    max_length: int = 8192
    instruction: str | None = DEFAULT_QWEN3_RERANK_INSTRUCTION
    system_prompt: str = DEFAULT_QWEN3_RERANK_SYSTEM_PROMPT
    trust_remote_code: bool = True
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _runtime_device: str | None = field(default=None, init=False, repr=False)
    _token_false_id: int | None = field(default=None, init=False, repr=False)
    _token_true_id: int | None = field(default=None, init=False, repr=False)
    _prefix_token_ids: list[int] | None = field(default=None, init=False, repr=False)
    _suffix_token_ids: list[int] | None = field(default=None, init=False, repr=False)

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Execute `score`.

        Args:
            query: Input parameter.
            documents: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        if not documents:
            return []

        self._ensure_loaded()

        import torch

        scores: list[float] = []
        base_batch_size = max(int(self.batch_size), 1)
        current_batch_size = base_batch_size
        position = 0
        while position < len(documents):
            batch_documents = documents[position : position + current_batch_size]
            if not batch_documents:
                break
            try:
                model_inputs = self._tokenize_pairs(
                    [self._format_pair(query, document) for document in batch_documents]
                )
                with torch.no_grad():
                    outputs = self._model(**model_inputs)
                    batch_logits = outputs.logits[:, -1, :]
                    true_vector = batch_logits[:, self._token_true_id]
                    false_vector = batch_logits[:, self._token_false_id]
                    rerank_logits = torch.stack([false_vector, true_vector], dim=1)
                    batch_scores = torch.nn.functional.log_softmax(rerank_logits, dim=1)[:, 1].exp()

                scores.extend(float(score) for score in batch_scores.detach().float().cpu().tolist())
                position += len(batch_documents)
                if current_batch_size < base_batch_size:
                    current_batch_size = min(base_batch_size, current_batch_size * 2)
            except Exception as exc:
                if not _is_cuda_oom_error(exc):
                    raise
                if current_batch_size == 1:
                    raise RuntimeError(
                        "Qwen reranker failed with CUDA OOM even at batch_size=1."
                    ) from exc
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)

        return scores

    def _ensure_loaded(self) -> None:
        """Execute `_ensure_loaded`.

        Returns:
            None: This function does not return a value.
        """
        if (
            self._model is not None
            and self._tokenizer is not None
            and self._runtime_device is not None
            and self._token_true_id is not None
            and self._token_false_id is not None
            and self._prefix_token_ids is not None
            and self._suffix_token_ids is not None
        ):
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        runtime_device = self.device or "cuda"
        if not str(runtime_device).lower().startswith("cuda"):
            raise RuntimeError(
                f"Qwen reranker requires a CUDA device, got device={runtime_device!r}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen reranker requires CUDA, but torch.cuda.is_available() is False."
            )
        model_path = snapshot_download(repo_id=self.model_name, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )
        model.to(runtime_device)
        model.eval()

        token_false_id = _single_token_id(tokenizer, "no")
        token_true_id = _single_token_id(tokenizer, "yes")
        prefix = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n"
        prefix_token_ids = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_token_ids = tokenizer.encode(DEFAULT_QWEN3_RERANK_SUFFIX, add_special_tokens=False)

        self._tokenizer = tokenizer
        self._model = model
        self._runtime_device = runtime_device
        self._token_false_id = token_false_id
        self._token_true_id = token_true_id
        self._prefix_token_ids = list(prefix_token_ids)
        self._suffix_token_ids = list(suffix_token_ids)

    def _format_pair(self, query: str, document: str) -> str:
        """Format pair.

        Args:
            query: Input parameter.
            document: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        instruction = self.instruction or DEFAULT_QWEN3_RERANK_INSTRUCTION
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"

    def _tokenize_pairs(self, pairs: list[str]) -> dict[str, Any]:
        """Execute `_tokenize_pairs`.

        Args:
            pairs: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        assert self._tokenizer is not None
        assert self._runtime_device is not None
        assert self._prefix_token_ids is not None
        assert self._suffix_token_ids is not None

        content_max_length = max(self.max_length - len(self._prefix_token_ids) - len(self._suffix_token_ids), 1)
        tokenized = self._tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=content_max_length,
        )
        for index, input_ids in enumerate(tokenized["input_ids"]):
            tokenized["input_ids"][index] = self._prefix_token_ids + list(input_ids) + self._suffix_token_ids

        padded_inputs = self._tokenizer.pad(
            tokenized,
            padding=True,
            return_tensors="pt",
        )
        return {key: value.to(self._runtime_device) for key, value in padded_inputs.items()}


@dataclass(slots=True)
class QwenReranker:
    """Primary reranker that routes scoring through a Qwen3 model backend."""

    backend: QwenRerankerBackend = field(default_factory=TransformersQwenRerankerBackend)
    fallback_reranker: CandidateReranker | None = None

    def rerank(
        self,
        question: QuestionRecord,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Execute `rerank`.

        Args:
            question: Input parameter.
            candidates: Input parameter.
            top_k: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        result_limit = max(int(top_k), 0)
        if result_limit == 0 or not candidates:
            return []

        try:
            scores = self.backend.score(
                question.question,
                [str(candidate.get("text") or "") for candidate in candidates],
            )
            if len(scores) != len(candidates):
                raise ValueError("Qwen reranker backend returned a mismatched number of scores.")
            reranked = [
                {
                    **dict(candidate),
                    "rerank_score": float(score),
                }
                for candidate, score in zip(candidates, scores, strict=True)
            ]
        except Exception as exc:
            log_event(
                LOGGER,
                logging.ERROR,
                "retrieval.rerank.failed",
                "Qwen reranker failed; strict mode forbids lexical fallback.",
                stage="retrieval",
                question_id=question.id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise RuntimeError(
                "Reranker backend failed in strict mode. Resolve model/runtime issue instead of using fallback."
            ) from exc

        reranked.sort(
            key=lambda candidate: (
                -candidate["rerank_score"],
                -_score(candidate.get("retrieval_score")),
                str(candidate.get("chunk_id") or ""),
            ),
        )
        window = reranked[:result_limit]
        if questions_debug_enabled():
            log_event(
                LOGGER,
                logging.DEBUG,
                "retrieval.rerank.complete",
                "Reranking completed for question.",
                stage="retrieval",
                question_id=question.id,
                candidate_count=len(candidates),
                reranked_count=len(window),
            )
        return window


@dataclass(slots=True)
class LexicalFallbackReranker:
    """Explicit lightweight fallback for environments without a model backend."""

    lexical_weight: float = 0.7
    retrieval_weight: float = 0.3
    stopwords: frozenset[str] = field(default_factory=lambda: DEFAULT_STOPWORDS)

    def rerank(
        self,
        question: QuestionRecord,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Execute `rerank`.

        Args:
            question: Input parameter.
            candidates: Input parameter.
            top_k: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        result_limit = max(int(top_k), 0)
        query_tokens = self._tokenize(question.question)
        reranked: list[dict[str, Any]] = []

        for candidate in candidates:
            normalized = dict(candidate)
            retrieval_score = _score(candidate.get("retrieval_score"))
            overlap_score = self._overlap_score(query_tokens, self._tokenize(str(candidate.get("text") or "")))
            normalized["rerank_score"] = (overlap_score * self.lexical_weight) + (
                retrieval_score * self.retrieval_weight
            )
            reranked.append(normalized)

        reranked.sort(
            key=lambda candidate: (
                -candidate["rerank_score"],
                -_score(candidate.get("retrieval_score")),
                str(candidate.get("chunk_id") or ""),
            ),
        )
        return reranked[:result_limit]

    def _tokenize(self, text: str) -> set[str]:
        """Execute `_tokenize`.

        Args:
            text: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        return {
            token
            for token in TOKEN_PATTERN.findall(text.lower())
            if token and token not in self.stopwords
        }

    def _overlap_score(self, query_tokens: set[str], candidate_tokens: set[str]) -> float:
        """Execute `_overlap_score`.

        Args:
            query_tokens: Input parameter.
            candidate_tokens: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        if not query_tokens or not candidate_tokens:
            return 0.0
        overlap = len(query_tokens & candidate_tokens)
        return overlap / len(query_tokens)


def _single_token_id(tokenizer: Any, text: str) -> int:
    """Execute `_single_token_id`.

    Args:
        tokenizer: Input parameter.
        text: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    input_ids = getattr(tokenizer(text, add_special_tokens=False), "input_ids", None)
    if not input_ids or len(input_ids) != 1:
        raise RuntimeError(f"Expected '{text}' to map to a single token for Qwen reranker scoring.")
    return int(input_ids[0])


def _is_cuda_oom_error(error: Exception) -> bool:
    """Return whether the error is a CUDA out-of-memory failure."""

    if type(error).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(error).casefold()


def _score(value: Any) -> float:
    """Execute `_score`.

    Args:
        value: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
