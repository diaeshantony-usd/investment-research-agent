"""
memory/tickers.py
=================
Lightweight, offline detection of *which company a question is about*.

The conversation layer needs to know two things about every incoming question:

1. Does it name a company? (``extract_tickers``)
2. Does it refer back to something said earlier, e.g. "What about its margins?"
   (``is_follow_up`` / ``needs_company_reference``)

Both checks are rule-based on purpose: they run on every turn, must be instant and must
never fail because a network call or an LLM is unavailable. Validating that a ticker
really exists (``NVDAA`` -> "did you mean NVDA?") is left to the market-data tools
downstream.

Example:
    >>> extract_tickers("Compare $AMD with Intel")
    ['AMD', 'INTC']
    >>> is_follow_up("What about its debt?")
    True

Owner: Diaesh Antony
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Final

__all__ = [
    "COMPANY_TICKERS",
    "extract_tickers",
    "is_follow_up",
    "needs_company_reference",
]

# Upper-case tokens that match the ticker shape but are finance or English words.
# A token in this set is only treated as a ticker when written explicitly as ``$TOKEN``.
# fmt: off
_NOT_TICKERS: Final[frozenset[str]] = frozenset({
    "A", "I", "AI", "AM", "AN", "AND", "ARE", "AS", "AT", "BE", "BY", "DO", "FOR", "GO",
    "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON", "OR", "SO", "TO", "UP",
    "US", "WE", "THE", "WHAT", "HOW", "WHY", "WHO", "VS", "ETC", "USA", "USD", "EUR",
    "CEO", "CFO", "CTO", "EPS", "PE", "P", "E", "PEG", "ROE", "ROA", "ROI", "EBIT",
    "EBITDA", "FCF", "YOY", "QOQ", "TTM", "YTD", "Q", "Q1", "Q2", "Q3", "Q4", "FY",
    "GDP", "CPI", "PCE", "FED", "FOMC", "SEC", "ETF", "IPO", "ESG", "API", "LLM", "NEWS",
    "SMA", "EMA", "RSI", "MACD", "ATH", "EV", "PS", "PB", "DCF", "WACC", "M", "B", "K",
})
# fmt: on

# Company names users commonly type instead of the symbol. Read-only so that callers
# cannot mutate shared module state by accident.
COMPANY_TICKERS: Final = MappingProxyType(
    {
        "apple": "AAPL",
        "nvidia": "NVDA",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "alphabet": "GOOGL",
        "google": "GOOGL",
        "meta": "META",
        "facebook": "META",
        "tesla": "TSLA",
        "netflix": "NFLX",
        "amd": "AMD",
        "advanced micro devices": "AMD",
        "intel": "INTC",
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "exxon": "XOM",
        "exxonmobil": "XOM",
        "broadcom": "AVGO",
        "salesforce": "CRM",
        "oracle": "ORCL",
        "walmart": "WMT",
        "berkshire": "BRK-B",
    }
)

# 1-5 capital letters with an optional class suffix (BRK.B / BRK-B), optionally
# prefixed by "$". The look-arounds stop matches inside longer words.
_TICKER_PATTERN: Final = re.compile(r"(?<![\w$])\$?([A-Z]{1,5}(?:[.\-][A-Z]{1,3})?)(?!\w)")

# One alternation over all company names, longest first so that
# "advanced micro devices" wins over any shorter overlapping name.
_COMPANY_PATTERN: Final = re.compile(
    r"\b("
    + "|".join(re.escape(n) for n in sorted(COMPANY_TICKERS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

# Words that signal the question depends on earlier context.
_FOLLOW_UP_PATTERN: Final = re.compile(
    r"\b(it|its|it's|they|their|them|this|that|these|those|the company|the stock|"
    r"same|again|also|too|earlier|previous|previously|above|you said|what about|"
    r"how about|and the|compared?|versus)\b",
    re.IGNORECASE,
)

# A stricter subset: references that only make sense if a company is already known.
_COMPANY_REFERENCE_PATTERN: Final = re.compile(
    r"\b(its|their|the company|the stock|this stock|that stock)\b",
    re.IGNORECASE,
)


def extract_tickers(text: str | None) -> list[str]:
    """Return the tickers mentioned in ``text``, in order of appearance.

    Recognises explicit symbols (``$NVDA``), bare upper-case symbols (``AAPL``,
    ``(NVDA)``, ``BRK.B``) and the company names listed in ``COMPANY_TICKERS``.
    Share-class dots are normalised to dashes (``BRK.B`` -> ``BRK-B``) to match the
    symbol format used by the market-data tools.

    Args:
        text: Free-text question from the user. ``None`` is treated as empty.

    Returns:
        Unique tickers in the order they first appear; empty when none are found.
    """
    if not text:
        return []

    # Collect (position, symbol) pairs from both detectors, then sort by position so
    # that "Compare AMD and Intel" yields ["AMD", "INTC"] regardless of detector order.
    found: list[tuple[int, str]] = []
    for match in _TICKER_PATTERN.finditer(text):
        symbol = match.group(1).replace(".", "-")
        is_explicit = match.group(0).startswith("$")
        if is_explicit or symbol not in _NOT_TICKERS:
            found.append((match.start(), symbol))

    found.extend(
        (match.start(), COMPANY_TICKERS[match.group(1).lower()])
        for match in _COMPANY_PATTERN.finditer(text)
    )

    # dict.fromkeys keeps the first occurrence of each symbol and preserves order.
    return list(dict.fromkeys(symbol for _, symbol in sorted(found)))


def is_follow_up(text: str | None) -> bool:
    """Return True when the question refers back to earlier conversation.

    Args:
        text: Free-text question from the user.

    Returns:
        True if the question contains a pronoun or phrase such as "its", "that" or
        "what about" that depends on previous turns.
    """
    if not text:
        return False
    return _FOLLOW_UP_PATTERN.search(text) is not None


def needs_company_reference(text: str | None) -> bool:
    """Return True when the question cannot be answered without a known company.

    "What about its debt?" needs a company; "What about interest rates?" does not.

    Args:
        text: Free-text question from the user.

    Returns:
        True if the question points at "its", "the company", "the stock" and so on.
    """
    if not text:
        return False
    return _COMPANY_REFERENCE_PATTERN.search(text) is not None
