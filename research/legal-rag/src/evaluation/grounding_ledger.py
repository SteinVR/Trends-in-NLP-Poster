"""Grounding layer ledger builders for recall-first run analysis."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.contracts import PageReference, ReferenceAnswerRecord, flatten_page_references
from src.evaluation.scorer import GROUNDING_BETA, score_grounding

_FAILURE_BUCKETS = ("raw_miss", "pass_a_miss", "final_miss")
_TOP_FAILURE_BUCKETS = set(_FAILURE_BUCKETS)
_LEDGER_FIELDNAMES = [
    "run_id",
    "question_id",
    "answer_type",
    "gold_pages",
    "raw_pages",
    "pass_a_pages",
    "final_pages",
    "raw_hit_pages",
    "pass_a_hit_pages",
    "final_hit_pages",
    "raw_miss_pages",
    "pass_a_miss_pages",
    "final_miss_pages",
    "raw_recall",
    "pass_a_recall",
    "final_precision",
    "final_grounding",
    "failure_bucket",
]
_SUMMARY_FIELDNAMES = [
    "run_id",
    "question_count",
    "gold_empty_count",
    "raw_miss_count",
    "pass_a_miss_count",
    "final_miss_count",
    "hit_count",
    "raw_miss_rate",
    "pass_a_miss_rate",
    "final_miss_rate",
    "mean_raw_recall",
    "mean_pass_a_recall",
    "mean_final_grounding",
]
_TOP_FAILURE_FIELDNAMES = [
    "question_id",
    "answer_type",
    "run_count",
    "total_failures",
    "raw_miss_count",
    "pass_a_miss_count",
    "final_miss_count",
    "dominant_bucket",
]


@dataclass(slots=True, frozen=True)
class GroundingLedgerRow:
    """Per-question grounding analysis row."""

    run_id: str
    question_id: str
    answer_type: str
    gold_pages: str
    raw_pages: str
    pass_a_pages: str
    final_pages: str
    raw_hit_pages: str
    pass_a_hit_pages: str
    final_hit_pages: str
    raw_miss_pages: str
    pass_a_miss_pages: str
    final_miss_pages: str
    raw_recall: float
    pass_a_recall: float
    final_precision: float
    final_grounding: float
    failure_bucket: str


@dataclass(slots=True, frozen=True)
class GroundingSummaryRow:
    """Run-level summary row for bucket and recall metrics."""

    run_id: str
    question_count: int
    gold_empty_count: int
    raw_miss_count: int
    pass_a_miss_count: int
    final_miss_count: int
    hit_count: int
    raw_miss_rate: float
    pass_a_miss_rate: float
    final_miss_rate: float
    mean_raw_recall: float
    mean_pass_a_recall: float
    mean_final_grounding: float


def build_grounding_ledger_rows(
    *,
    run_id: str,
    submission_payload: Any,
    references: list[ReferenceAnswerRecord],
) -> list[GroundingLedgerRow]:
    """Build per-question grounding ledger rows from submission telemetry and benchmark references."""

    answers_by_question_id = _answers_by_question_id(submission_payload)
    rows: list[GroundingLedgerRow] = []
    for reference in references:
        answer_payload = answers_by_question_id.get(reference.question_id, {})
        gold_pairs = flatten_page_references(reference.gold_retrieval)
        retrieval_payload = _retrieval_telemetry(answer_payload)

        final_pairs = _pairs_from_raw_page_refs(retrieval_payload.get("retrieved_chunk_pages"))
        raw_pairs = _pairs_from_raw_page_refs(
            retrieval_payload.get("raw_retrieved_chunk_pages"),
            fallback=final_pairs,
        )
        pass_a_pairs = _pairs_from_raw_page_refs(
            retrieval_payload.get("pass_a_pages"),
            fallback=raw_pairs,
        )

        raw_hit_pairs = gold_pairs & raw_pairs
        pass_a_hit_pairs = gold_pairs & pass_a_pairs
        final_hit_pairs = gold_pairs & final_pairs

        raw_miss_pairs = gold_pairs - raw_pairs
        pass_a_miss_pairs = gold_pairs - pass_a_pairs
        final_miss_pairs = gold_pairs - final_pairs

        row = GroundingLedgerRow(
            run_id=run_id,
            question_id=reference.question_id,
            answer_type=str(reference.answer_type.value),
            gold_pages=_pairs_to_compact_string(gold_pairs),
            raw_pages=_pairs_to_compact_string(raw_pairs),
            pass_a_pages=_pairs_to_compact_string(pass_a_pairs),
            final_pages=_pairs_to_compact_string(final_pairs),
            raw_hit_pages=_pairs_to_compact_string(raw_hit_pairs),
            pass_a_hit_pages=_pairs_to_compact_string(pass_a_hit_pairs),
            final_hit_pages=_pairs_to_compact_string(final_hit_pairs),
            raw_miss_pages=_pairs_to_compact_string(raw_miss_pairs),
            pass_a_miss_pages=_pairs_to_compact_string(pass_a_miss_pairs),
            final_miss_pages=_pairs_to_compact_string(final_miss_pairs),
            raw_recall=_score_recall(predicted_pairs=raw_pairs, gold_pairs=gold_pairs),
            pass_a_recall=_score_recall(predicted_pairs=pass_a_pairs, gold_pairs=gold_pairs),
            final_precision=_score_precision(predicted_pairs=final_pairs, gold_pairs=gold_pairs),
            final_grounding=score_grounding(predicted_pairs=final_pairs, gold_pairs=gold_pairs),
            failure_bucket=_classify_failure_bucket(
                gold_pairs=gold_pairs,
                raw_pairs=raw_pairs,
                pass_a_pairs=pass_a_pairs,
                final_pairs=final_pairs,
            ),
        )
        rows.append(row)

    return rows


def build_grounding_summary(*, run_id: str, rows: list[GroundingLedgerRow]) -> GroundingSummaryRow:
    """Compute run-level summary metrics from per-question ledger rows."""

    question_count = len(rows)
    bucket_counts = {bucket: 0 for bucket in (*_FAILURE_BUCKETS, "hit", "gold_empty")}
    for row in rows:
        bucket_counts[row.failure_bucket] = bucket_counts.get(row.failure_bucket, 0) + 1

    eligible_rows = [row for row in rows if row.failure_bucket != "gold_empty"]
    eligible_count = len(eligible_rows)

    return GroundingSummaryRow(
        run_id=run_id,
        question_count=question_count,
        gold_empty_count=bucket_counts.get("gold_empty", 0),
        raw_miss_count=bucket_counts.get("raw_miss", 0),
        pass_a_miss_count=bucket_counts.get("pass_a_miss", 0),
        final_miss_count=bucket_counts.get("final_miss", 0),
        hit_count=bucket_counts.get("hit", 0),
        raw_miss_rate=_safe_ratio(bucket_counts.get("raw_miss", 0), eligible_count),
        pass_a_miss_rate=_safe_ratio(bucket_counts.get("pass_a_miss", 0), eligible_count),
        final_miss_rate=_safe_ratio(bucket_counts.get("final_miss", 0), eligible_count),
        mean_raw_recall=_safe_mean([row.raw_recall for row in eligible_rows]),
        mean_pass_a_recall=_safe_mean([row.pass_a_recall for row in eligible_rows]),
        mean_final_grounding=_safe_mean([row.final_grounding for row in eligible_rows]),
    )


def build_grounding_summary_payload(
    *,
    summary: GroundingSummaryRow,
    rows: list[GroundingLedgerRow],
    ledger_jsonl_path: Path,
    ledger_csv_path: Path,
    summary_runs_tsv_path: Path,
    top_failures_tsv_path: Path,
) -> dict[str, Any]:
    """Build extended JSON summary payload for one grounding run."""

    eligible_rows = [row for row in rows if row.failure_bucket != "gold_empty"]
    raw_precision_values: list[float] = []
    pass_a_precision_values: list[float] = []
    raw_grounding_values: list[float] = []
    pass_a_grounding_values: list[float] = []
    final_recall_values: list[float] = []

    for row in eligible_rows:
        gold_pairs = _pairs_from_compact_string(row.gold_pages)
        raw_pairs = _pairs_from_compact_string(row.raw_pages)
        pass_a_pairs = _pairs_from_compact_string(row.pass_a_pages)
        final_pairs = _pairs_from_compact_string(row.final_pages)

        raw_precision_values.append(_score_precision(predicted_pairs=raw_pairs, gold_pairs=gold_pairs))
        pass_a_precision_values.append(_score_precision(predicted_pairs=pass_a_pairs, gold_pairs=gold_pairs))
        raw_grounding_values.append(score_grounding(predicted_pairs=raw_pairs, gold_pairs=gold_pairs))
        pass_a_grounding_values.append(score_grounding(predicted_pairs=pass_a_pairs, gold_pairs=gold_pairs))
        final_recall_values.append(_score_recall(predicted_pairs=final_pairs, gold_pairs=gold_pairs))

    return {
        "run_id": summary.run_id,
        "question_count": summary.question_count,
        "eligible_question_count": len(eligible_rows),
        "gold_empty_count": summary.gold_empty_count,
        "raw_miss_count": summary.raw_miss_count,
        "pass_a_miss_count": summary.pass_a_miss_count,
        "final_miss_count": summary.final_miss_count,
        "hit_count": summary.hit_count,
        "raw_miss_rate": summary.raw_miss_rate,
        "pass_a_miss_rate": summary.pass_a_miss_rate,
        "final_miss_rate": summary.final_miss_rate,
        "mean_raw_recall": summary.mean_raw_recall,
        "mean_pass_a_recall": summary.mean_pass_a_recall,
        "mean_final_recall": _safe_mean(final_recall_values),
        "mean_raw_precision": _safe_mean(raw_precision_values),
        "mean_pass_a_precision": _safe_mean(pass_a_precision_values),
        "mean_final_precision": _safe_mean([row.final_precision for row in eligible_rows]),
        "mean_raw_grounding": _safe_mean(raw_grounding_values),
        "mean_pass_a_grounding": _safe_mean(pass_a_grounding_values),
        "mean_final_grounding": summary.mean_final_grounding,
        "grounding_beta": GROUNDING_BETA,
        "ledger_jsonl": str(ledger_jsonl_path),
        "ledger_csv": str(ledger_csv_path),
        "summary_runs_tsv": str(summary_runs_tsv_path),
        "top_failures_tsv": str(top_failures_tsv_path),
    }


def build_top_failures_from_ledger_dir(*, ledger_dir: Path, top_n: int) -> list[dict[str, str | int]]:
    """Aggregate top persistent failure question IDs across all available run ledgers."""

    bucket_counts_by_question: dict[str, dict[str, int]] = {}
    run_ids_by_question: dict[str, set[str]] = {}
    answer_type_by_question: dict[str, str] = {}

    for ledger_path in sorted(ledger_dir.glob("*.csv")):
        for row in _read_csv_rows(ledger_path):
            question_id = str(row.get("question_id") or "").strip()
            failure_bucket = str(row.get("failure_bucket") or "").strip()
            run_id = str(row.get("run_id") or "").strip()
            answer_type = str(row.get("answer_type") or "").strip()
            if not question_id or not run_id:
                continue
            if failure_bucket not in _TOP_FAILURE_BUCKETS:
                continue

            bucket_counts = bucket_counts_by_question.setdefault(
                question_id,
                {bucket: 0 for bucket in _FAILURE_BUCKETS},
            )
            bucket_counts[failure_bucket] += 1
            run_ids_by_question.setdefault(question_id, set()).add(run_id)
            if answer_type:
                answer_type_by_question.setdefault(question_id, answer_type)

    ranked_rows: list[dict[str, str | int]] = []
    for question_id, bucket_counts in bucket_counts_by_question.items():
        total_failures = sum(bucket_counts.values())
        dominant_bucket = max(
            _FAILURE_BUCKETS,
            key=lambda bucket: (bucket_counts[bucket], bucket),
        )
        ranked_rows.append(
            {
                "question_id": question_id,
                "answer_type": answer_type_by_question.get(question_id, ""),
                "run_count": len(run_ids_by_question.get(question_id, set())),
                "total_failures": total_failures,
                "raw_miss_count": bucket_counts["raw_miss"],
                "pass_a_miss_count": bucket_counts["pass_a_miss"],
                "final_miss_count": bucket_counts["final_miss"],
                "dominant_bucket": dominant_bucket,
            }
        )

    ranked_rows.sort(
        key=lambda row: (
            -int(row["run_count"]),
            -int(row["total_failures"]),
            str(row["question_id"]),
        )
    )
    return ranked_rows[: max(top_n, 0)]


def build_top_failures_from_runs_dir(*, runs_dir: Path, top_n: int) -> list[dict[str, str | int]]:
    """Aggregate top persistent failure question IDs across structured run bundles."""

    bucket_counts_by_question: dict[str, dict[str, int]] = {}
    run_ids_by_question: dict[str, set[str]] = {}
    answer_type_by_question: dict[str, str] = {}

    for ledger_path in sorted(runs_dir.glob("*/grounding/ledger.csv")):
        for row in _read_csv_rows(ledger_path):
            question_id = str(row.get("question_id") or "").strip()
            failure_bucket = str(row.get("failure_bucket") or "").strip()
            run_id = str(row.get("run_id") or "").strip()
            answer_type = str(row.get("answer_type") or "").strip()
            if not question_id or not run_id:
                continue
            if failure_bucket not in _TOP_FAILURE_BUCKETS:
                continue

            bucket_counts = bucket_counts_by_question.setdefault(
                question_id,
                {bucket: 0 for bucket in _FAILURE_BUCKETS},
            )
            bucket_counts[failure_bucket] += 1
            run_ids_by_question.setdefault(question_id, set()).add(run_id)
            if answer_type:
                answer_type_by_question.setdefault(question_id, answer_type)

    ranked_rows: list[dict[str, str | int]] = []
    for question_id, bucket_counts in bucket_counts_by_question.items():
        total_failures = sum(bucket_counts.values())
        dominant_bucket = max(
            _FAILURE_BUCKETS,
            key=lambda bucket: (bucket_counts[bucket], bucket),
        )
        ranked_rows.append(
            {
                "question_id": question_id,
                "answer_type": answer_type_by_question.get(question_id, ""),
                "run_count": len(run_ids_by_question.get(question_id, set())),
                "total_failures": total_failures,
                "raw_miss_count": bucket_counts["raw_miss"],
                "pass_a_miss_count": bucket_counts["pass_a_miss"],
                "final_miss_count": bucket_counts["final_miss"],
                "dominant_bucket": dominant_bucket,
            }
        )

    ranked_rows.sort(
        key=lambda row: (
            -int(row["run_count"]),
            -int(row["total_failures"]),
            str(row["question_id"]),
        )
    )
    return ranked_rows[: max(top_n, 0)]


def write_grounding_ledger_jsonl(*, path: Path, rows: list[GroundingLedgerRow]) -> None:
    """Serialize per-question rows as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_row_to_dict(row), ensure_ascii=True))
            handle.write("\n")


def write_grounding_ledger_csv(*, path: Path, rows: list[GroundingLedgerRow]) -> None:
    """Serialize per-question rows as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_LEDGER_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_dict(row))


def upsert_summary_row(*, path: Path, summary: GroundingSummaryRow) -> None:
    """Insert or replace one run summary row in a TSV table."""

    existing_rows = _read_tsv_rows(path) if path.exists() else []
    filtered_rows = [row for row in existing_rows if str(row.get("run_id") or "") != summary.run_id]
    filtered_rows.append(_summary_to_dict(summary))
    filtered_rows.sort(key=lambda row: str(row.get("run_id") or ""))
    _write_tsv_rows(path=path, fieldnames=_SUMMARY_FIELDNAMES, rows=filtered_rows)


def write_top_failures_tsv(*, path: Path, rows: list[dict[str, str | int]]) -> None:
    """Write aggregated top failures table as TSV."""

    _write_tsv_rows(path=path, fieldnames=_TOP_FAILURE_FIELDNAMES, rows=rows)


def _answers_by_question_id(submission_payload: Any) -> dict[str, dict[str, Any]]:
    """Build question_id -> answer object mapping from a raw submission payload."""

    if not isinstance(submission_payload, dict):
        return {}
    answers = submission_payload.get("answers")
    if not isinstance(answers, list):
        return {}

    answer_map: dict[str, dict[str, Any]] = {}
    for item in answers:
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("question_id") or "").strip()
        if not question_id:
            continue
        answer_map.setdefault(question_id, item)
    return answer_map


def _retrieval_telemetry(answer_payload: dict[str, Any]) -> dict[str, Any]:
    """Extract telemetry.retrieval payload from one answer object."""

    telemetry = answer_payload.get("telemetry")
    if not isinstance(telemetry, dict):
        return {}
    retrieval = telemetry.get("retrieval")
    if not isinstance(retrieval, dict):
        return {}
    return retrieval


def _pairs_from_raw_page_refs(
    raw_references: Any,
    *,
    fallback: set[tuple[str, int]] | None = None,
) -> set[tuple[str, int]]:
    """Flatten page references from raw telemetry payloads into `(doc_id, page_number)` pairs."""

    parsed = _parse_page_references(raw_references)
    if parsed:
        return flatten_page_references(parsed)
    return set(fallback or ())


def _parse_page_references(raw_references: Any) -> list[PageReference]:
    """Parse raw page reference payloads into typed `PageReference` records."""

    if not isinstance(raw_references, list):
        return []

    parsed: list[PageReference] = []
    for item in raw_references:
        if not isinstance(item, dict):
            continue
        try:
            parsed.append(PageReference.model_validate(item))
        except Exception:
            continue
    return parsed


def _pairs_to_compact_string(pairs: set[tuple[str, int]]) -> str:
    """Serialize page-pair sets into a compact, deterministic string representation."""

    if not pairs:
        return ""
    ordered = sorted(pairs, key=lambda pair: (pair[0], pair[1]))
    return "|".join(f"{doc_id}:{page_number}" for doc_id, page_number in ordered)


def _pairs_from_compact_string(raw_pairs: str) -> set[tuple[str, int]]:
    """Parse compact `doc_id:page|...` strings back into pair sets."""

    normalized = str(raw_pairs or "").strip()
    if not normalized:
        return set()

    pairs: set[tuple[str, int]] = set()
    for item in normalized.split("|"):
        doc_id, separator, page_number = item.partition(":")
        if not separator:
            continue
        doc_id = doc_id.strip()
        page_number = page_number.strip()
        if not doc_id or not page_number:
            continue
        try:
            pairs.add((doc_id, int(page_number)))
        except ValueError:
            continue
    return pairs


def _score_recall(*, predicted_pairs: set[tuple[str, int]], gold_pairs: set[tuple[str, int]]) -> float:
    """Compute recall on flattened page pairs with empty-gold compatibility."""

    if not gold_pairs:
        return 1.0
    return len(predicted_pairs & gold_pairs) / len(gold_pairs)


def _score_precision(*, predicted_pairs: set[tuple[str, int]], gold_pairs: set[tuple[str, int]]) -> float:
    """Compute precision on flattened page pairs with empty-set compatibility."""

    if not predicted_pairs:
        return 1.0 if not gold_pairs else 0.0
    return len(predicted_pairs & gold_pairs) / len(predicted_pairs)


def _classify_failure_bucket(
    *,
    gold_pairs: set[tuple[str, int]],
    raw_pairs: set[tuple[str, int]],
    pass_a_pairs: set[tuple[str, int]],
    final_pairs: set[tuple[str, int]],
) -> str:
    """Classify one question into the recall-first grounding failure bucket chain."""

    if not gold_pairs:
        return "gold_empty"
    if gold_pairs & final_pairs:
        return "hit"
    if not gold_pairs & raw_pairs:
        return "raw_miss"
    if not gold_pairs & pass_a_pairs:
        return "pass_a_miss"
    return "final_miss"


def _row_to_dict(row: GroundingLedgerRow) -> dict[str, str]:
    """Convert a `GroundingLedgerRow` to CSV/JSONL-friendly string dictionary."""

    return {
        "run_id": row.run_id,
        "question_id": row.question_id,
        "answer_type": row.answer_type,
        "gold_pages": row.gold_pages,
        "raw_pages": row.raw_pages,
        "pass_a_pages": row.pass_a_pages,
        "final_pages": row.final_pages,
        "raw_hit_pages": row.raw_hit_pages,
        "pass_a_hit_pages": row.pass_a_hit_pages,
        "final_hit_pages": row.final_hit_pages,
        "raw_miss_pages": row.raw_miss_pages,
        "pass_a_miss_pages": row.pass_a_miss_pages,
        "final_miss_pages": row.final_miss_pages,
        "raw_recall": f"{row.raw_recall:.12f}",
        "pass_a_recall": f"{row.pass_a_recall:.12f}",
        "final_precision": f"{row.final_precision:.12f}",
        "final_grounding": f"{row.final_grounding:.12f}",
        "failure_bucket": row.failure_bucket,
    }


def _summary_to_dict(row: GroundingSummaryRow) -> dict[str, str]:
    """Convert a `GroundingSummaryRow` to a TSV-ready dictionary."""

    return {
        "run_id": row.run_id,
        "question_count": str(row.question_count),
        "gold_empty_count": str(row.gold_empty_count),
        "raw_miss_count": str(row.raw_miss_count),
        "pass_a_miss_count": str(row.pass_a_miss_count),
        "final_miss_count": str(row.final_miss_count),
        "hit_count": str(row.hit_count),
        "raw_miss_rate": f"{row.raw_miss_rate:.12f}",
        "pass_a_miss_rate": f"{row.pass_a_miss_rate:.12f}",
        "final_miss_rate": f"{row.final_miss_rate:.12f}",
        "mean_raw_recall": f"{row.mean_raw_recall:.12f}",
        "mean_pass_a_recall": f"{row.mean_pass_a_recall:.12f}",
        "mean_final_grounding": f"{row.mean_final_grounding:.12f}",
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a safe ratio with zero-denominator fallback."""

    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _safe_mean(values: list[float]) -> float:
    """Return arithmetic mean with empty-list fallback."""

    if not values:
        return 0.0
    return sum(values) / len(values)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows from disk with empty-file safety."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    """Read TSV rows from disk with empty-file safety."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def _write_tsv_rows(*, path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows to a TSV file, creating parent directories when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            normalized = {name: row.get(name, "") for name in fieldnames}
            writer.writerow(normalized)
