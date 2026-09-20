"""Page triage and quality scoring for the corpus parsing stage."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.config import IngestionSettings
from src.common.schemas import PageSignal, TriageLabel

LEGAL_MARKERS = (
    "article",
    "section",
    "clause",
    "law",
    "court",
    "judgment",
    "order",
    "decision",
    "regulation",
)


@dataclass(slots=True)
class QualityBreakdown:
    """Deterministic quality score plus component details."""

    score: float
    text_completeness: float
    encoding_cleanliness: float
    legal_marker_score: float
    flags: list[str] = field(default_factory=list)


def estimate_bad_char_ratio(text: str) -> float:
    """Estimate encoding noise by replacement and control characters."""

    if not text:
        return 0.0
    bad_chars = sum(1 for char in text if char == "\ufffd" or (ord(char) < 32 and char not in "\n\t\r"))
    return bad_chars / len(text)


def count_legal_markers(text: str) -> int:
    """Return the number of matched legal markers in the text."""

    return len(_legal_marker_hits(text))


def triage_page(signals: PageSignal, settings: IngestionSettings) -> TriageLabel:
    """Choose the page routing label using deterministic thresholds."""

    if signals.text_char_count < settings.extractable_text_min_chars:
        return TriageLabel.SCAN

    if (
        signals.max_image_area_ratio >= settings.scan_image_area_ratio
        and signals.text_char_count < settings.suspicious_text_max_chars
    ):
        return TriageLabel.SCAN

    if (
        signals.text_char_count <= settings.suspicious_text_max_chars
        or signals.bad_char_ratio >= settings.encoding_bad_char_threshold
        or (
            signals.max_image_area_ratio >= settings.large_image_area_ratio
            and signals.text_char_count < settings.text_completeness_target_chars
        )
    ):
        return TriageLabel.SUSPICIOUS

    return TriageLabel.NATIVE_CLEAN


def classify_page(signals: PageSignal, settings: IngestionSettings) -> tuple[TriageLabel, list[str]]:
    """Compatibility wrapper returning triage label plus flags."""

    breakdown = score_page_quality(signals, settings)
    return triage_page(signals, settings), breakdown.flags

def score_page_quality(signals: PageSignal, settings: IngestionSettings) -> QualityBreakdown:
    """Score text quality using the W2C heuristic weighting."""

    text_completeness = min(signals.text_char_count / settings.text_completeness_target_chars, 1.0)
    encoding_cleanliness = 1.0 - min(
        signals.bad_char_ratio / max(settings.encoding_bad_char_threshold, 1e-6),
        1.0,
    )
    legal_marker_score = min(signals.legal_marker_count / 3.0, 1.0)

    score = round(
        (0.5 * text_completeness) + (0.3 * encoding_cleanliness) + (0.2 * legal_marker_score),
        4,
    )

    flags: list[str] = []
    if signals.text_char_count < settings.extractable_text_min_chars:
        flags.append("low_text")
    if signals.bad_char_ratio >= settings.encoding_bad_char_threshold:
        flags.append("bad_encoding")
    if signals.max_image_area_ratio >= settings.large_image_area_ratio:
        flags.append("large_image")
    if signals.table_count > 0:
        flags.append("has_tables")
    if signals.legal_marker_count == 0:
        flags.append("missing_legal_markers")

    return QualityBreakdown(
        score=score,
        text_completeness=round(text_completeness, 4),
        encoding_cleanliness=round(encoding_cleanliness, 4),
        legal_marker_score=round(legal_marker_score, 4),
        flags=flags,
    )


def compute_quality_score(signals: PageSignal, settings: IngestionSettings) -> float:
    """Compatibility wrapper returning only the score."""

    return score_page_quality(signals, settings).score


def quality_score_and_flags(signals: PageSignal, settings: IngestionSettings) -> tuple[float, list[str]]:
    """Compatibility wrapper returning only score and flags."""

    breakdown = score_page_quality(signals, settings)
    return breakdown.score, breakdown.flags


def _legal_marker_hits(text: str) -> list[str]:
    """Execute `_legal_marker_hits`.

    Args:
        text: Input parameter.

    Returns:
        Any: The computed result of the function.
    """
    lowered = text.lower()
    return [marker for marker in LEGAL_MARKERS if marker in lowered]


PageSignals = PageSignal
