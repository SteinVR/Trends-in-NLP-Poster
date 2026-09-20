"""Telemetry packaging helpers for submission answers."""

from __future__ import annotations

from typing import Any

from src.answering.page_attribution import PageAttributionTrace
from src.answering.service import AnswerResult
from src.evaluation.contracts import PageReference
from src.retrieval.service import RetrievalResult


def build_answer_telemetry(
    *,
    retrieval_result: RetrievalResult,
    answer_result: AnswerResult,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    total_time_ms: float,
    ttft_ms: float | None = None,
    tpot_ms: float = 0.0,
) -> dict[str, Any]:
    """Build telemetry payload for one answer.

    Final grounding pages must come from the answering page trace.
    For non-streaming generation, `ttft_ms` falls back to `total_time_ms`.
    """

    resolved_ttft_ms = float(total_time_ms) if ttft_ms is None else float(ttft_ms)
    page_trace = _require_page_trace(answer_result)
    retrieved_chunk_pages = [
        _serialize_page_reference(reference) for reference in page_trace.final_emitted_pages
    ]

    return {
        "timing": {
            "ttft_ms": resolved_ttft_ms,
            "tpot_ms": float(tpot_ms),
            "total_time_ms": float(total_time_ms),
        },
        "retrieval": _build_retrieval_telemetry(
            retrieved_chunk_pages=retrieved_chunk_pages,
            retrieval_result=retrieval_result,
            page_trace=page_trace,
        ),
        "usage": {
            "input_tokens": max(int(input_tokens), 0),
            "output_tokens": max(int(output_tokens), 0),
        },
        "model_name": str(model_name),
    }


def pack_answer_telemetry(
    *,
    retrieval_result: RetrievalResult,
    answer_result: AnswerResult,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    total_time_ms: float,
    ttft_ms: float | None = None,
    tpot_ms: float = 0.0,
) -> dict[str, Any]:
    """Backward-compatible alias for telemetry packaging."""

    return build_answer_telemetry(
        retrieval_result=retrieval_result,
        answer_result=answer_result,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_time_ms=total_time_ms,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
    )


class TelemetryPacker:
    """OO wrapper for telemetry packing."""

    def build_answer_telemetry(
        self,
        *,
        retrieval_result: RetrievalResult,
        answer_result: AnswerResult,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        total_time_ms: float,
        ttft_ms: float | None = None,
        tpot_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Build answer telemetry.

        Args:
            retrieval_result: Input parameter.
            answer_result: Input parameter.
            model_name: Input parameter.
            input_tokens: Input parameter.
            output_tokens: Input parameter.
            total_time_ms: Input parameter.
            ttft_ms: Input parameter.
            tpot_ms: Input parameter.

        Returns:
            Any: The computed result of the function.
        """
        return build_answer_telemetry(
            retrieval_result=retrieval_result,
            answer_result=answer_result,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_time_ms=total_time_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
        )

    pack_answer = build_answer_telemetry
    pack_answer_telemetry = build_answer_telemetry


def _require_page_trace(answer_result: AnswerResult) -> PageAttributionTrace:
    """Return the required page trace or fail the submission contract."""

    if answer_result.page_trace is None:
        raise ValueError("AnswerResult.page_trace is required to build submission telemetry.")
    return answer_result.page_trace


def _serialize_page_reference(reference: PageReference) -> dict[str, Any]:
    """Execute `_serialize_page_reference`.

    Args:
        reference: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    return {
        "doc_id": reference.doc_id,
        "page_numbers": list(reference.page_numbers),
    }


def _build_retrieval_telemetry(
    *,
    retrieved_chunk_pages: list[dict[str, Any]],
    retrieval_result: RetrievalResult,
    page_trace: PageAttributionTrace,
) -> dict[str, Any]:
    """Build retrieval telemetry.

    Args:
        retrieved_chunk_pages: Input parameter.
        retrieval_result: Input parameter.
        page_trace: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    payload: dict[str, Any] = {"retrieved_chunk_pages": retrieved_chunk_pages}
    payload["raw_retrieved_chunk_pages"] = [
        _serialize_page_reference(reference) for reference in page_trace.raw_retrieved_chunk_pages
    ]
    payload["pass_a_pages"] = [
        _serialize_page_reference(reference) for reference in page_trace.pass_a_pages
    ]
    payload["solver_reported_pages"] = [
        _serialize_page_reference(reference) for reference in page_trace.solver_reported_pages
    ]
    final_count = sum(len(reference.page_numbers) for reference in page_trace.final_emitted_pages)
    payload["page_validation"] = {
        "action": page_trace.validation_action,
        "collapsed_answer": page_trace.collapsed_answer,
        "selection_source": page_trace.selection_source,
        "llm_selected_count": int(page_trace.llm_selected_count),
        "final_count": int(final_count),
    }
    if retrieval_result.diagnostics:
        payload["diagnostics"] = dict(retrieval_result.diagnostics)
    return payload
