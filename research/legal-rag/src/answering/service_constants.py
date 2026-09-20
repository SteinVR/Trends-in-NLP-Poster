"""Shared answering constants and regex patterns."""

from __future__ import annotations

import re

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "who",
}
_NUMBER_PATTERN = re.compile(r"(?<!\w)(?:[A-Z]{2,5}\s*)?[-+]?\d[\d,]*(?:\.\d+)?(?!\w)")
_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})\s+(?P<month>"
    + "|".join(_MONTHS)
    + r")\s+(?P<year>\d{4})(?!\d)",
    re.IGNORECASE,
)
_MONTH_FIRST_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<month>"
    + "|".join(_MONTHS)
    + r")\s+(?P<day>\d{1,2})(?:,\s*|\s+)(?P<year>\d{4})(?!\d)",
    re.IGNORECASE,
)
_ISO_DATE_PATTERN = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_UPPER_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Z0-9&'/-]*(?:\.?\s+[A-Z][A-Z0-9&'/-]*)+\.?\b")
_ENTITY_LIST_ITEM_PATTERN = re.compile(
    r"\b(?:[A-Z]{2,}[A-Z0-9&'/-]*|[A-Z][a-z][A-Za-z0-9&'/-]*)(?:\.?\s+[A-Z][A-Za-z0-9&'/-]*)*\.?\b"
)
_BOOLEAN_NEGATIVE = (
    "not permitted",
    "not allowed",
    "prohibited",
    "forbidden",
    "must not",
    "shall not",
    "cannot",
    "may not",
)
_BOOLEAN_POSITIVE = (
    "permitted",
    "allowed",
    "may",
    "can",
    "shall",
    "must",
)
_PERMISSION_QUESTION_HINTS = frozenset(
    {
        "allow",
        "allowed",
        "allows",
        "can",
        "forbid",
        "forbidden",
        "forbids",
        "may",
        "permit",
        "permitted",
        "permits",
        "prohibit",
        "prohibited",
        "prohibits",
    }
)
_OBLIGATION_QUESTION_HINTS = frozenset(
    {
        "must",
        "obligation",
        "obligations",
        "obliged",
        "require",
        "required",
        "requires",
        "shall",
    }
)
_BOOLEAN_NO_PERMISSION_PATTERN = re.compile(
    r"\bno\s+\w+(?:\s+\w+){0,4}\s+(?:is|are|was|were)\s+(?:permitted|allowed)\b",
    re.IGNORECASE,
)
_BOOLEAN_NO_MODAL_PATTERN = re.compile(
    r"\bno\s+\w+(?:\s+\w+){0,4}\s+(?:may|can|shall|must)\b",
    re.IGNORECASE,
)
_TEMPORAL_UNITS = ("day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours")
_DURATION_QUESTION_HINTS = frozenset(_TEMPORAL_UNITS) | {
    "deadline",
    "duration",
    "period",
    "time",
    "within",
    "long",
    "response",
    "payment",
    "payable",
}
_AMOUNT_CUE_PATTERN = re.compile(r"\b(?:aed|usd|eur|fine|amount|price|cost|penalty|fee)\b", re.IGNORECASE)
_TEMPORAL_CUE_PATTERN = re.compile(
    r"\b(?:days?|weeks?|months?|years?|hours?|deadline|within|after|before)\b",
    re.IGNORECASE,
)
_LAW_NUMBER_PATTERN = re.compile(
    r"\b(?:difc\s+)?law\s+(?:no\.?|number)\s*(?P<number>\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_LAW_NUMBER_QUERY_PATTERN = re.compile(
    r"\b(?:official\s+)?(?:difc\s+)?law\s+(?:no\.?|number)\b",
    re.IGNORECASE,
)
_LEGAL_OUTCOME_PATTERNS = {
    "dismiss": re.compile(r"\bdismiss(?:ed|es|ing)?\b"),
    "grant": re.compile(r"\bgrant(?:ed|s|ing)?\b"),
    "deny": re.compile(r"\bden(?:y|ied|ies|ying)\b"),
    "allow": re.compile(r"\ballow(?:ed|s|ing)?\b"),
    "prohibit": re.compile(r"\bprohibit(?:ed|s|ing)?\b"),
    "terminate": re.compile(r"\bterminat(?:e|ed|es|ing)\b"),
    "order": re.compile(r"\border(?:ed|s|ing)?\b"),
    "pay": re.compile(r"\bpay(?:able|ment|ments|ing|s)?\b"),
    "liable": re.compile(r"\bliab(?:ility|le)\b"),
}
_WORD_NUMBER_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_WORD_NUMBER_SCALES = {
    "hundred": 100,
    "thousand": 1000,
}
_WORD_NUMBER_TOKENS = frozenset(_WORD_NUMBER_VALUES) | frozenset(_WORD_NUMBER_SCALES) | {"and"}
_CONTRACT_ENTITY_PATTERN = re.compile(r"\b(?:contract|agreement)\b")
_LEGAL_ROLE_PATTERN = re.compile(
    r"\b(?:respondent|claimant|appellant|petitioner|defendant|plaintiff|party|parties|seller|buyer|"
    r"landlord|tenant|employer|employee|liquidator|lessor|lessee|borrower|lender|guarantor|"
    r"insured|insurer|contractor|subcontractor)\b",
    re.IGNORECASE,
)
_ROLE_ENTITY_EXTRACTOR = re.compile(
    r"\b(?:the\s+)?(?:respondent|claimant|appellant|petitioner|defendant|plaintiff|party|parties|seller|buyer|"
    r"landlord|tenant|employer|employee|liquidator|lessor|lessee|borrower|lender|guarantor|"
    r"insured|insurer|contractor|subcontractor)\b"
    r"\s*(?::|(?:is|was|are|were|means|named|identified\s+as|shall\s+be))\s*"
    r"(?P<entities>[^.;:\n]+)",
    re.IGNORECASE,
)
_SINGLE_TOKEN_ENTITY_EXCLUSIONS = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "Schedule",
    "Article",
    "Section",
    "Clause",
    "Agreement",
    "Contract",
}
_CASE_ID_PATTERN = re.compile(
    r"\bcase\s*(?:no\.?|number)?\s*(?P<case_id>\d{1,4}\s*/\s*\d{2,4})\b",
    re.IGNORECASE,
)
_DIFC_CASE_REF_PATTERN = re.compile(
    r"\b(?:CFI|SCT|ARB|DEC|TCD|ENF|CA)\s*(?P<case_id>\d{1,4}\s*/\s*\d{2,4})\b",
    re.IGNORECASE,
)
_STATUTE_ID_PATTERN = re.compile(
    r"\b(?P<statute>(?:[A-Z]{2,}\s+)?[A-Za-z]+(?:\s+[A-Za-z]+){0,3}\s+Law)\b"
)
_DOCUMENT_TITLE_CUE_PATTERN = re.compile(
    r"\b(?P<title>[A-Z][A-Za-z0-9'&/-]*(?:\s+[A-Z][A-Za-z0-9'&/-]*)*\s+"
    r"(?:Agreement|Contract|Deed|Order|Policy|Protocol|Arrangement|Lease))\b"
)
_MULTI_DOCUMENT_SIGNAL_PATTERN = re.compile(
    r"\b(across|between|both|related\s+matters?|related\s+cases?|multiple|versus|vs\.?)\b",
    re.IGNORECASE,
)
_TITLE_PAGE_QUERY_PATTERN = re.compile(r"\b(?:title|cover)\s+page\b", re.IGNORECASE)
_ISSUE_DATE_QUERY_PATTERN = re.compile(r"\b(?:date\s+of\s+issue|issue\s+date)\b", re.IGNORECASE)
_DATE_OF_ISSUE_CUE_PATTERN = re.compile(r"\bdate\s+of\s+issue\b", re.IGNORECASE)


__all__ = [name for name in globals() if not name.startswith("__")]
