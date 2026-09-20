"""Deterministic extractors and grounding helpers for answering service."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from datetime import date
from typing import Any

from src.evaluation.contracts import AnswerType, PageReference

from .service_constants import (
    _AMOUNT_CUE_PATTERN,
    _BOOLEAN_NEGATIVE,
    _BOOLEAN_NO_MODAL_PATTERN,
    _BOOLEAN_NO_PERMISSION_PATTERN,
    _CONTRACT_ENTITY_PATTERN,
    _DATE_OF_ISSUE_CUE_PATTERN,
    _DATE_PATTERN,
    _DURATION_QUESTION_HINTS,
    _ISO_DATE_PATTERN,
    _LAW_NUMBER_PATTERN,
    _LAW_NUMBER_QUERY_PATTERN,
    _LEGAL_OUTCOME_PATTERNS,
    _MONTH_FIRST_DATE_PATTERN,
    _MONTHS,
    _NUMBER_PATTERN,
    _OBLIGATION_QUESTION_HINTS,
    _PERMISSION_QUESTION_HINTS,
    _SINGLE_TOKEN_ENTITY_EXCLUSIONS,
    _TEMPORAL_CUE_PATTERN,
    _WORD_NUMBER_TOKENS,
    _WORD_NUMBER_VALUES,
)
from .service_helpers_text import (
    _context_window,
    _entity_matches,
    _grounding_clauses,
    _keyword_tokens,
    _normalize_entity_candidate,
    _normalize_whitespace,
    _score_context,
    _sentence_for_span,
    _sentences,
)
from .service_types import (
    DeterministicExtractor,
    FreeTextGroundingValidationError,
    _PageCandidate,
    _SolvedAnswer,
)


def _answers_equivalent(left: Any, right: Any) -> bool:
    """Execute `_answers_equivalent`."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    except (TypeError, ValueError):
        pass
    return _answer_key(left) == _answer_key(right)


def _parse_numeric_value(raw_value: str) -> float:
    """Parse numeric value."""
    numeric_text = raw_value.split()[-1]
    return float(numeric_text.replace(",", ""))


def _extract_number(
    *,
    page: _PageCandidate,
    question_tokens: set[str],
    prefer_law_number: bool = False,
) -> _SolvedAnswer | None:
    """Extract number."""
    if prefer_law_number:
        law_number = _extract_law_number_literal(page.text)
        if law_number is not None:
            return _SolvedAnswer(answer=law_number, supporting_pages=[page])
        return None

    scored: list[tuple[float, float]] = []
    duration_intent = _is_duration_question(question_tokens)
    question_numeric_literals = {
        float(int(token))
        for token in question_tokens
        if token.isdigit()
    }
    for value, start, end in _extract_number_candidates(page.text):
        sentence = _sentence_for_span(page.text, start, end)
        local_context = _context_window(page.text, start, end, radius=24)
        prefix = page.text[max(0, start - 20) : start]
        suffix = page.text[end : min(len(page.text), end + 16)]
        previous_char = page.text[start - 1] if start > 0 else ""
        next_char = page.text[end] if end < len(page.text) else ""
        score = (2.0 * _score_context(context=local_context, question_tokens=question_tokens)) + _score_context(
            context=sentence,
            question_tokens=question_tokens,
        )
        has_amount_cue = bool(_AMOUNT_CUE_PATTERN.search(local_context))
        has_temporal_cue = bool(_TEMPORAL_CUE_PATTERN.search(local_context))
        has_duration_unit_suffix = bool(
            re.match(
                r"^\s*(?:business\s+)?(?:day|days|week|weeks|month|months|year|years|hour|hours)\b",
                suffix,
                re.IGNORECASE,
            )
        )
        looks_like_legal_reference_number = bool(
            re.search(r"\b(?:article|section|clause)\s+\d+(?:\s*\(\d+\))?\b", local_context, re.IGNORECASE)
        )
        is_parenthetical_marker = previous_char == "(" and next_char == ")"
        if has_amount_cue:
            score += 1.5
        if has_temporal_cue:
            score += 1.5
        if duration_intent and has_temporal_cue:
            score += 2.0
        if duration_intent and has_duration_unit_suffix:
            score += 3.0
        if (
            duration_intent
            and looks_like_legal_reference_number
            and not has_temporal_cue
            and not has_duration_unit_suffix
        ):
            continue
        if duration_intent and is_parenthetical_marker and not has_temporal_cue and not has_duration_unit_suffix:
            continue
        if duration_intent and has_amount_cue and not has_temporal_cue:
            score -= 2.0
        if duration_intent and _AMOUNT_CUE_PATTERN.search(prefix):
            score -= 3.0
        if not duration_intent and has_temporal_cue:
            score -= 1.0
        if score <= 0:
            continue
        scored.append((score, value))

    non_question_scored = [
        (score, candidate_value)
        for score, candidate_value in scored
        if candidate_value not in question_numeric_literals
    ]
    value = _unique_best_value(non_question_scored if non_question_scored else scored)
    if value is None:
        return None

    answer: int | float
    if value.is_integer():
        answer = int(value)
    else:
        answer = value
    return _SolvedAnswer(answer=answer, supporting_pages=[page])


def _is_law_number_question(question_text: str) -> bool:
    """Return whether law number question is true."""
    normalized = question_text.strip().lower()
    if not _LAW_NUMBER_QUERY_PATTERN.search(normalized):
        return False
    if re.search(r"\bhow\s+many\b", normalized):
        return False
    if _TEMPORAL_CUE_PATTERN.search(normalized):
        return False
    return bool(re.search(r"\b(?:what|which)\b", normalized))


def _extract_law_number_literal(text: str) -> int | float | None:
    """Extract law number literal."""
    match = _LAW_NUMBER_PATTERN.search(text)
    if match is None:
        return None
    try:
        value = float(match.group("number"))
    except (TypeError, ValueError):
        return None
    return int(value) if value.is_integer() else value


def _extract_boolean(*, page: _PageCandidate, question_tokens: set[str]) -> _SolvedAnswer | None:
    """Extract boolean."""
    candidates: list[tuple[float, bool]] = []
    permission_question = bool(question_tokens & _PERMISSION_QUESTION_HINTS)
    obligation_question = bool(question_tokens & _OBLIGATION_QUESTION_HINTS)
    if not permission_question and not obligation_question:
        permission_question = True
        obligation_question = True
    for sentence in _sentences(page.text):
        score = _score_context(context=sentence, question_tokens=question_tokens)
        if score <= 0:
            continue
        if permission_question and _contains_permission_negation(sentence):
            candidates.append((score + 1.0, False))
            continue
        if obligation_question and _contains_boolean_phrase(sentence, ("must not", "shall not")):
            candidates.append((score + 1.0, False))
            continue
        if permission_question and _contains_boolean_phrase(sentence, ("permitted", "allowed", "may", "can")):
            candidates.append((score, True))
            continue
        if obligation_question and _contains_boolean_phrase(sentence, ("must", "shall")):
            candidates.append((score, True))

    answer = _unique_best_value(candidates)
    if answer is None:
        return None

    return _SolvedAnswer(answer=answer, supporting_pages=[page])


def _extract_name(*, page: _PageCandidate, question_tokens: set[str]) -> _SolvedAnswer | None:
    """Extract name."""
    candidates = _name_candidates(page=page, question_tokens=question_tokens)
    answer = _unique_best_value(candidates)
    if answer is None:
        return None
    return _SolvedAnswer(answer=answer, supporting_pages=[page])


def _extract_names(*, page: _PageCandidate, question_tokens: set[str]) -> _SolvedAnswer | None:
    """Extract names."""
    role = _target_role_from_question_tokens(question_tokens)
    listed_role_query = _is_listed_role_query(question_tokens)
    if role is not None:
        role_labeled_names = _extract_role_labeled_entities(
            text=page.text,
            role=role,
            listed_only=listed_role_query,
        )
        if role_labeled_names:
            return _SolvedAnswer(answer=role_labeled_names, supporting_pages=[page])
        if listed_role_query:
            return None

    candidates = _name_candidates(page=page, question_tokens=question_tokens)
    if not candidates:
        return None

    ordered_names: list[str] = []
    seen: set[str] = set()
    for _, candidate in candidates:
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered_names.append(candidate)

    if not ordered_names:
        return None

    return _SolvedAnswer(answer=ordered_names, supporting_pages=[page])


def _extract_date(*, page: _PageCandidate, question_tokens: set[str]) -> _SolvedAnswer | None:
    """Extract date."""
    issue_date_intent = "date" in question_tokens and "issue" in question_tokens
    if issue_date_intent:
        scored_issue: list[tuple[float, str]] = []
        for iso_value, start, end in _extract_date_candidates(page.text):
            context = _context_window(page.text, start, end, radius=64)
            score = _score_context(context=context, question_tokens=question_tokens)
            if _DATE_OF_ISSUE_CUE_PATTERN.search(context):
                score += 6.0
            if start <= 240:
                score += 2.0
            text_length = max(len(page.text), 1)
            score += max(0.0, 1.0 - (float(start) / float(text_length)))
            scored_issue.append((score, iso_value))

        issue_answer = _unique_best_value(scored_issue)
        if issue_answer is not None:
            return _SolvedAnswer(answer=issue_answer, supporting_pages=[page])

    scored: list[tuple[float, str]] = []
    for iso_value, start, end in _extract_date_candidates(page.text):
        context = _context_window(page.text, start, end)
        scored.append((_score_context(context=context, question_tokens=question_tokens), iso_value))

    answer = _unique_best_value(scored)
    if answer is None:
        return None

    return _SolvedAnswer(answer=answer, supporting_pages=[page])


def _name_candidates(*, page: _PageCandidate, question_tokens: set[str]) -> list[tuple[float, str]]:
    """Execute `_name_candidates`."""
    candidates: list[tuple[float, str]] = []
    for candidate, start, end in _entity_matches(page.text):
        context = _context_window(page.text, start, end)
        score = _score_context(context=context, question_tokens=question_tokens)
        if score <= 0:
            continue
        candidates.append((score, candidate))
    return candidates


def _target_role_from_question_tokens(question_tokens: set[str]) -> str | None:
    """Return target legal role for role-specific names queries."""

    if "claimant" in question_tokens or "claimants" in question_tokens:
        return "claimant"
    if "defendant" in question_tokens or "defendants" in question_tokens:
        return "defendant"
    if "respondent" in question_tokens or "respondents" in question_tokens:
        return "respondent"
    if "appellant" in question_tokens or "appellants" in question_tokens:
        return "appellant"
    if "judge" in question_tokens or "judges" in question_tokens:
        return "judge"
    return None


def _is_listed_role_query(question_tokens: set[str]) -> bool:
    """Return whether query asks for explicitly listed role entities."""

    listing_tokens = {"listed", "caption", "header", "title", "cover"}
    return bool(question_tokens.intersection(listing_tokens))


def _extract_role_labeled_entities(
    *,
    text: str,
    role: str,
    listed_only: bool,
) -> list[str]:
    """Extract role-labeled entities with caption-first precision."""

    role_pattern = _role_pattern(role)
    if role_pattern is None:
        return []

    candidates: list[str] = []
    seen: set[str] = set()
    caption_pattern = _caption_role_pattern(role_pattern)
    for match in caption_pattern.finditer(text):
        candidate = _normalize_role_entity_candidate(match.group("entity"))
        if not _is_valid_role_entity(candidate):
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    if candidates or listed_only:
        return candidates

    sentence_pattern = _sentence_role_pattern(role_pattern)
    for match in sentence_pattern.finditer(text):
        candidate = _normalize_role_entity_candidate(match.group("entity"))
        if not _is_valid_role_entity(candidate):
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    return candidates


def _role_pattern(role: str) -> str | None:
    """Return a regex fragment matching role labels on legal caption lines."""

    if role == "claimant":
        return r"claimant(?!['’]s)(?:/[a-z]+)?"
    if role == "defendant":
        return r"defendant(?!['’]s)(?:/[a-z]+)?"
    if role == "respondent":
        return r"respondent(?!['’]s)(?:/[a-z]+)?"
    if role == "appellant":
        return r"appellant(?!['’]s)(?:/[a-z]+)?"
    if role == "judge":
        return r"judge(?:\s+at\s+first\s+instance)?"
    return None


def _caption_role_pattern(role_pattern: str) -> re.Pattern[str]:
    """Build caption-style entity-before-role pattern."""

    return re.compile(
        rf"\b(?P<entity>[A-Z][A-Za-z0-9&'/-]{{2,}}(?:\s+[A-Z][A-Za-z0-9&'/-]{{2,}}){{0,4}})\s+"
        rf"(?i:{role_pattern})\b",
    )


def _sentence_role_pattern(role_pattern: str) -> re.Pattern[str]:
    """Build sentence-style role-is-entity pattern."""

    return re.compile(
        rf"\b(?i:(?:the\s+)?{role_pattern})\b\s*(?::|is|was|are|were)\s+"
        rf"(?P<entity>[A-Z][A-Za-z0-9&'/-]{{1,}}(?:\s+[A-Z][A-Za-z0-9&'/-]{{1,}}){{0,4}})\b",
    )


def _is_valid_role_entity(candidate: str) -> bool:
    """Return whether extracted role entity candidate is usable."""

    normalized = candidate.strip().rstrip(".")
    if not normalized:
        return False
    parts = normalized.split()
    if len(parts) == 1 and normalized in _SINGLE_TOKEN_ENTITY_EXCLUSIONS:
        return False
    if len(parts) > 6:
        return False
    return True


def _normalize_role_entity_candidate(raw_candidate: str) -> str:
    """Normalize role entity and trim caption boilerplate prefixes."""

    normalized = _normalize_entity_candidate(raw_candidate)
    lowered = normalized.casefold()
    between_marker = " between "
    if between_marker in lowered:
        index = lowered.rfind(between_marker)
        normalized = normalized[index + len(between_marker) :].strip()
    return _normalize_entity_candidate(normalized)


def _validate_free_text_grounding(*, answer: str, pages: Iterable[_PageCandidate]) -> None:
    """Validate free text grounding."""
    evidence_text = " ".join(page.text for page in pages)
    evidence_tokens = _keyword_tokens(evidence_text)
    evidence_numbers = _extract_numeric_values(evidence_text)
    answer_numbers = _extract_numeric_values(answer)
    unsupported_numbers = answer_numbers - evidence_numbers
    if unsupported_numbers:
        raise FreeTextGroundingValidationError(
            "free_text answer contradicts the evidence with unsupported numbers."
        )

    evidence_dates = set()
    for page in pages:
        evidence_dates.update(_extract_dates_from_text(page.text))

    answer_dates = _extract_dates_from_text(answer)
    missing_dates = answer_dates - evidence_dates
    if missing_dates:
        raise FreeTextGroundingValidationError(
            "free_text answer contradicts the evidence with unsupported dates."
        )

    evidence_names = _extract_names_from_text(evidence_text)
    answer_names = _extract_names_from_text(answer)
    unsupported_names = answer_names - evidence_names
    if unsupported_names:
        raise FreeTextGroundingValidationError(
            "free_text answer contradicts the evidence with unsupported names."
        )

    evidence_outcomes = _extract_outcomes(evidence_text)
    answer_outcomes = _extract_outcomes(answer)
    unsupported_outcomes = answer_outcomes - evidence_outcomes
    if unsupported_outcomes:
        raise FreeTextGroundingValidationError(
            "free_text answer contradicts the evidence with unsupported outcomes."
        )

    _validate_fact_polarity(answer=answer, evidence_text=evidence_text)

    for clause in _grounding_clauses(answer):
        clause_tokens = _keyword_tokens(clause)
        if not clause_tokens:
            continue
        unsupported_tokens = clause_tokens - evidence_tokens
        if len(unsupported_tokens) >= 2:
            raise FreeTextGroundingValidationError(
                "free_text answer contradicts the evidence with unsupported rationale or clauses."
            )


def _validate_provider_confidence(raw_confidence: Any) -> float:
    """Validate provider confidence."""
    confidence = float(raw_confidence)
    if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        raise ValueError("Provider confidence must be finite and between 0.0 and 1.0.")
    return confidence


def _extract_numeric_values(text: str) -> set[float]:
    """Extract numeric values."""
    return {value for value, _, _ in _extract_number_candidates(text)}


def _extract_names_from_text(text: str) -> set[str]:
    """Extract names from text."""
    return {candidate.casefold() for candidate, _, _ in _entity_matches(text)}


def _extract_outcomes(text: str) -> set[str]:
    """Extract outcomes."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    outcomes: set[str] = set()
    for label, pattern in _LEGAL_OUTCOME_PATTERNS.items():
        if pattern.search(normalized):
            outcomes.add(label)
    return outcomes


def _extract_dates_from_text(text: str) -> set[str]:
    """Extract dates from text."""
    return {iso_value for iso_value, _, _ in _extract_date_candidates(text)}


def _extract_date_candidates(text: str) -> list[tuple[str, int, int]]:
    """Extract date candidates."""
    candidates: list[tuple[str, int, int]] = []
    for match in _ISO_DATE_PATTERN.finditer(text):
        iso_value = _safe_iso_date(
            year=int(match.group(1)),
            month=int(match.group(2)),
            day=int(match.group(3)),
        )
        if iso_value is not None:
            candidates.append((iso_value, match.start(), match.end()))
    for match in _DATE_PATTERN.finditer(text):
        iso_value = _safe_iso_date_from_textual_match(match)
        if iso_value is None:
            continue
        candidates.append((iso_value, match.start(), match.end()))
    for match in _MONTH_FIRST_DATE_PATTERN.finditer(text):
        iso_value = _safe_iso_date_from_textual_match(match)
        if iso_value is None:
            continue
        candidates.append((iso_value, match.start(), match.end()))
    return candidates


def _extract_written_number_values(text: str) -> set[float]:
    """Extract written number values."""
    return {value for value, _, _ in _extract_written_number_candidates(text)}


def _extract_number_candidates(text: str) -> list[tuple[float, int, int]]:
    """Extract number candidates."""
    candidates: list[tuple[float, int, int]] = []
    for match in _NUMBER_PATTERN.finditer(text):
        raw_value = match.group(0).strip()
        try:
            candidates.append((_parse_numeric_value(raw_value), match.start(), match.end()))
        except ValueError:
            continue
    candidates.extend(_extract_written_number_candidates(text))
    candidates.sort(key=lambda item: (item[1], item[2]))
    return candidates


def _extract_written_number_candidates(text: str) -> list[tuple[float, int, int]]:
    """Extract written number candidates."""
    candidates: list[tuple[float, int, int]] = []
    word_matches = list(re.finditer(r"[A-Za-z]+(?:-[A-Za-z]+)?", text))
    index = 0
    while index < len(word_matches):
        token = word_matches[index].group(0).casefold()
        if token not in _WORD_NUMBER_TOKENS and _hyphenated_number_tokens(token) is None:
            index += 1
            continue

        start = word_matches[index].start()
        end = word_matches[index].end()
        sequence: list[str] = []
        cursor = index
        while cursor < len(word_matches):
            current = word_matches[cursor].group(0).casefold()
            if cursor > index and not re.fullmatch(r"[\s-]+", text[end : word_matches[cursor].start()]):
                break
            expanded = _hyphenated_number_tokens(current)
            if current in _WORD_NUMBER_TOKENS:
                sequence.append(current)
                end = word_matches[cursor].end()
                cursor += 1
                continue
            if expanded is not None:
                sequence.extend(expanded)
                end = word_matches[cursor].end()
                cursor += 1
                continue
            break

        parsed = _parse_written_number_sequence(sequence)
        if parsed is not None:
            candidates.append((float(parsed), start, end))
            index = cursor
            continue
        index += 1

    return candidates


def _hyphenated_number_tokens(token: str) -> list[str] | None:
    """Execute `_hyphenated_number_tokens`."""
    parts = token.split("-")
    if len(parts) < 2 or any(part not in _WORD_NUMBER_TOKENS for part in parts):
        return None
    return parts


def _parse_written_number_sequence(tokens: list[str]) -> int | None:
    """Parse written number sequence."""
    total = 0
    current = 0
    seen_number = False
    for token in tokens:
        if token == "and":
            continue
        if token in _WORD_NUMBER_VALUES:
            current += _WORD_NUMBER_VALUES[token]
            seen_number = True
            continue
        if token == "hundred":
            if current == 0:
                current = 1
            current *= 100
            seen_number = True
            continue
        if token == "thousand":
            if current == 0:
                current = 1
            total += current * 1000
            current = 0
            seen_number = True
            continue
        return None
    if not seen_number:
        return None
    return total + current


def _safe_iso_date(*, year: int, month: int, day: int) -> str | None:
    """Execute `_safe_iso_date`."""
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _safe_iso_date_from_textual_match(match: re.Match[str]) -> str | None:
    """Execute `_safe_iso_date_from_textual_match`."""
    return _safe_iso_date(
        year=int(match.group("year")),
        month=_MONTHS[match.group("month").casefold()],
        day=int(match.group("day")),
    )


def _to_iso_date(*, day: int, month_name: str, year: int) -> str:
    """Execute `_to_iso_date`."""
    month = _MONTHS[month_name.casefold()]
    return date(year, month, day).isoformat()


def _validate_fact_polarity(*, answer: str, evidence_text: str) -> None:
    """Validate fact polarity."""
    answer_states = _extract_fact_states(answer)
    evidence_states = _extract_fact_states(evidence_text)
    for label, answer_values in answer_states.items():
        evidence_values = evidence_states.get(label)
        if not evidence_values:
            continue
        if answer_values.isdisjoint(evidence_values):
            raise FreeTextGroundingValidationError(
                "free_text answer contradicts the evidence with unsupported polarity or state."
            )


def _extract_fact_states(text: str) -> dict[str, set[str]]:
    """Extract fact states."""
    facts: dict[str, set[str]] = defaultdict(set)
    for sentence in _sentences(text):
        lowered = sentence.casefold()
        for label, pattern in _LEGAL_OUTCOME_PATTERNS.items():
            for match in pattern.finditer(lowered):
                facts[label].add("negated" if _match_is_negated(lowered, match.start()) else "affirmed")
        for state in ("valid", "void"):
            for match in re.finditer(rf"\b{state}\b", lowered):
                if not _CONTRACT_ENTITY_PATTERN.search(_context_window(lowered, match.start(), match.end(), radius=24)):
                    continue
                facts["contract_status"].add(f"not_{state}" if _match_is_negated(lowered, match.start()) else state)
    return facts


def _match_is_negated(text: str, match_start: int) -> bool:
    """Execute `_match_is_negated`."""
    prefix = text[max(0, match_start - 24) : match_start]
    return bool(
        re.search(
            r"(?:did\s+not|didn't|was\s+not|were\s+not|is\s+not|are\s+not|not|never)\s*$",
            prefix,
        )
    )


def _group_page_references(pages: Iterable[_PageCandidate]) -> list[PageReference]:
    """Execute `_group_page_references`."""
    grouped: dict[str, set[int]] = defaultdict(set)
    for page in pages:
        grouped[page.doc_id].update(page.page_numbers)

    return [
        PageReference(doc_id=doc_id, page_numbers=sorted(page_numbers))
        for doc_id, page_numbers in sorted(grouped.items())
    ]


def _dedupe_pages(pages: Iterable[_PageCandidate]) -> list[_PageCandidate]:
    """Execute `_dedupe_pages`."""
    seen: set[tuple[str, tuple[int, ...]]] = set()
    deduped: list[_PageCandidate] = []
    for page in pages:
        key = (page.doc_id, page.page_numbers)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(page)
    return deduped


def _provider_evidence_page(page: _PageCandidate) -> dict[str, Any]:
    """Execute `_provider_evidence_page`."""
    return {
        "doc_id": page.doc_id,
        "page_number": page.page_number,
        "page_numbers": list(page.page_numbers),
        "text": page.text,
    }


def _contains_permission_negation(text: str) -> bool:
    """Execute `_contains_permission_negation`."""
    return (
        _contains_boolean_phrase(text, _BOOLEAN_NEGATIVE)
        or bool(_BOOLEAN_NO_PERMISSION_PATTERN.search(text))
        or bool(_BOOLEAN_NO_MODAL_PATTERN.search(text))
    )


def _answer_key(answer: Any) -> str:
    """Execute `_answer_key`."""
    if isinstance(answer, list):
        normalized_items = sorted(
            _normalize_whitespace(item).casefold() if isinstance(item, str) else str(item)
            for item in answer
        )
        return "|".join(normalized_items)
    if isinstance(answer, str):
        return _normalize_whitespace(answer).casefold()
    return str(answer)


def _is_duration_question(question_tokens: set[str]) -> bool:
    """Return whether duration question is true."""
    return bool(question_tokens.intersection(_DURATION_QUESTION_HINTS))


def _contains_boolean_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    """Execute `_contains_boolean_phrase`."""
    return any(re.search(_phrase_pattern(phrase), text, re.IGNORECASE) for phrase in phrases)


def _phrase_pattern(phrase: str) -> str:
    """Execute `_phrase_pattern`."""
    tokens = [re.escape(token) for token in phrase.split()]
    return r"\b" + r"\s+".join(tokens) + r"\b"


def _get_deterministic_extractor(answer_type: AnswerType) -> DeterministicExtractor:
    """Return deterministic extractor."""
    extractor = _DETERMINISTIC_EXTRACTORS.get(answer_type)
    if extractor is None:
        raise ValueError(f"Unsupported answer_type: {answer_type.value}")
    return extractor


def _unique_best_value(scored_values: Iterable[tuple[float, Any]]) -> Any | None:
    """Execute `_unique_best_value`."""
    scored = list(scored_values)
    if not scored:
        return None

    best_score = max(score for score, _ in scored)
    best_values = {value for score, value in scored if score == best_score}
    if len(best_values) != 1:
        return None

    return next(iter(best_values))


_DETERMINISTIC_EXTRACTORS: dict[AnswerType, DeterministicExtractor] = {
    AnswerType.NUMBER: _extract_number,
    AnswerType.BOOLEAN: _extract_boolean,
    AnswerType.NAME: _extract_name,
    AnswerType.NAMES: _extract_names,
    AnswerType.DATE: _extract_date,
}

__all__ = [name for name in globals() if not name.startswith("__")]
