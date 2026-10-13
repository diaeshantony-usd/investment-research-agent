"""
memory/tickers.py
=================
Detection of *which company a question is about*.

The conversation layer needs to know two things about every incoming question:

1. Does it name a company?
   * ``extract_tickers`` - instant and offline: symbols such as ``$NVDA``, ``AAPL``,
     ``BRK.B``.
   * ``resolve_tickers`` - ``extract_tickers`` plus Yahoo Finance search: confirms typed
     symbols exist (``NVDAA`` -> "did you mean NVDA?") and finds companies by name
     ("How is Costco doing?", "tell me about nvidia"). No company names are hard-coded.
2. Does it refer back to something said earlier, e.g. "What about its margins?"
   (``is_follow_up`` / ``needs_company_reference``)

Yahoo lookups are cached for the life of the process and "fail open": if Yahoo cannot
be reached, typed symbols are accepted as before and name lookup is skipped, so a
network problem never blocks a question.

Example:
    >>> extract_tickers("Compare $AMD with INTC")
    ['AMD', 'INTC']
    >>> is_follow_up("What about its debt?")
    True

Owner: Diaesh Antony
"""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from research_agent.config import logger

__all__ = [
    "TickerResolution",
    "extract_tickers",
    "is_follow_up",
    "needs_company_reference",
    "resolve_tickers",
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
    "GPU", "CPU", "TPU", "NPU", "SAAS", "CAPEX", "OPEX", "COGS", "EBITA", "ARR", "IR", "PR",
})
# fmt: on

# 1-5 capital letters with an optional class suffix (BRK.B / BRK-B), optionally
# prefixed by "$". The look-arounds stop matches inside longer words.
_TICKER_PATTERN: Final = re.compile(r"(?<![\w$])\$?([A-Z]{1,5}(?:[.\-][A-Z]{1,3})?)(?!\w)")

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

    Recognises explicit symbols (``$NVDA``) and bare upper-case symbols (``AAPL``,
    ``(NVDA)``, ``BRK.B``). Company names are resolved by ``resolve_tickers`` through
    Yahoo Finance search. Share-class dots are normalised to dashes (``BRK.B`` ->
    ``BRK-B``) to match the symbol format used by the market-data tools.

    Args:
        text: Free-text question from the user. ``None`` is treated as empty.

    Returns:
        Unique tickers in the order they first appear; empty when none are found.
    """
    if not text:
        return []

    symbols = []
    for match in _TICKER_PATTERN.finditer(text):
        symbol = match.group(1).replace(".", "-")
        is_explicit = match.group(0).startswith("$")
        if is_explicit or symbol not in _NOT_TICKERS:
            symbols.append(symbol)
    # dict.fromkeys keeps the first occurrence of each symbol and preserves order.
    return list(dict.fromkeys(symbols))


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


# ---------------------------------------------------------------------------
# Yahoo Finance lookups (company names and ticker validation)
# ---------------------------------------------------------------------------

# Yahoo quote types we treat as researchable securities.
_STOCK_TYPES: Final = frozenset({"EQUITY"})
_SYMBOL_TYPES: Final = frozenset({"EQUITY", "ETF"})
# Shorter symbols are not "corrected": removing a letter from "AB" is a guess, not a fix.
_MIN_TYPO_LENGTH: Final = 3

# A run of 1-4 capitalised words, e.g. "Costco" or "Home Depot".
_NAME_CANDIDATE: Final = re.compile(r"\b[A-Z][\w&'.\-]*(?:\s+[A-Z][\w&'.\-]*){0,3}")
_POSSESSIVE: Final = re.compile("['\u2019]s$", re.IGNORECASE)
# Up to 4 words following a cue word: "about nvidia", "of costco", "analyse home depot".
# (Captured inside a look-ahead so overlapping phrases are all found:
# "are the margins of home depot" yields both "the margins of home" and "home depot".)
_CUE_PHRASE: Final = re.compile(
    r"\b(?:about|of|for|on|at|analy[sz]e|research|evaluate|check|compare|is|are|was|and|"
    r"vs|versus|with|to|than)\s+"
    r"(?=([\w&'.\-]+(?:\s+[\w&'.\-]+){0,3}))",
    re.IGNORECASE,
)

# Capitalised words that start or decorate questions, or are finance vocabulary, and so
# must never be searched as company names.
# fmt: off
_NOT_NAMES: Final[frozenset[str]] = frozenset({
    "a", "an", "the", "and", "or", "of", "for", "in", "on", "to", "vs", "versus", "with",
    "i", "me", "my", "we", "you", "it", "its", "is", "are", "was", "were", "be", "do", "does",
    "did", "can", "could", "should", "would", "will", "what", "how", "why", "when", "where",
    "which", "who", "whose", "tell", "show", "give", "explain", "analyse", "analyze",
    "compare", "evaluate", "research", "look", "check", "find", "please", "now", "then",
    "also", "about", "has", "have", "any", "latest", "recent", "new", "today",
    "yesterday", "this", "that", "these", "those", "q1", "q2", "q3", "q4", "fy",
    "earnings", "revenue", "margin", "margins", "profit", "price", "prices", "stock",
    "stocks", "share", "shares", "market", "markets", "return", "returns", "equity", "debt",
    "cash", "flow", "ratio", "interest", "rate", "rates", "inflation", "dividend",
    "valuation", "growth", "risk", "risks", "news", "report", "guidance", "outlook",
    "fed", "economy", "gdp", "cpi", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "hello", "hi", "thanks",
    "doing", "going", "performing", "trading", "looking", "data", "financials", "info",
    "information", "details", "company", "companies", "first", "second", "third", "last",
    "previous", "next", "other", "same", "operating", "gross", "net", "free", "total",
    "between", "difference", "against", "than", "from", "into", "over", "under", "up",
    "down", "so", "far", "much", "many", "more", "most", "less", "good", "bad", "buy",
    "sell", "hold", "invest", "investing", "investment", "quarter", "quarterly", "year",
    "annual", "week", "month", "day", "time", "concept", "meaning", "term", "terms",
    "i'm", "im", "researching", "researched", "before", "after", "earlier", "demand",
    "supply", "sales", "rising", "falling", "rise", "fall", "higher", "lower", "strong",
    "weak", "sector", "industry",
})
# fmt: on


@dataclass(frozen=True)
class TickerResolution:
    """Result of ``resolve_tickers``.

    Attributes:
        tickers: Confirmed tickers, in the order they appear in the question.
        unknown: A typed symbol Yahoo does not recognise (when no ticker was confirmed).
        suggestion: Yahoo's closest match for ``unknown``, if any (e.g. NVDAA -> NVDA).
    """

    tickers: tuple[str, ...] = ()
    unknown: str | None = None
    suggestion: str | None = None


# Yahoo exchange codes of over-the-counter markets (not exchange listings).
_OTC_EXCHANGES: Final = frozenset({"PNK", "OQB", "OQX", "OEM", "OBB", "OTC", "OPR"})


@functools.lru_cache(maxsize=512)
def _yahoo_search(query: str, fuzzy: bool = False) -> tuple[dict[str, Any], ...]:
    """Yahoo Finance symbol search (the site's search box), cached per process.

    Raises:
        Exception: Any network / Yahoo error (not cached, so a later call retries).
    """
    import yfinance as yf  # local import: heavy dependency, only needed on a lookup

    search = yf.Search(
        query,
        max_results=5,
        news_count=0,
        lists_count=0,
        recommended=0,
        include_cb=False,
        enable_fuzzy_query=fuzzy,
        raise_errors=True,
    )
    quotes = [
        {
            "symbol": str(quote.get("symbol") or "").upper(),
            "name": quote.get("shortname") or quote.get("longname") or "",
            "type": quote.get("quoteType") or "",
            "exchange": str(quote.get("exchange") or "").upper(),
        }
        for quote in search.quotes or []
    ]
    # US exchange listings first ("XOM" before "XOM.NE"): the system covers US equities.
    return tuple(sorted(quotes, key=lambda q: not _is_us_listing(q)))


def _is_us_listing(quote: Mapping[str, Any]) -> bool:
    """True for a symbol listed on a US exchange (not a foreign or OTC listing)."""
    return "." not in quote["symbol"] and quote.get("exchange", "") not in _OTC_EXCHANGES


def _search_or_none(query: str, fuzzy: bool = False) -> tuple[dict[str, Any], ...] | None:
    """``_yahoo_search`` that returns None (instead of raising) when Yahoo is unavailable."""
    try:
        return _yahoo_search(query, fuzzy)
    except Exception as exc:  # noqa: BLE001 - lookups fail open; the question still runs
        logger.warning("Yahoo Finance search failed for %r: %s", query, exc)
        return None


def _symbol_exists(symbol: str) -> bool | None:
    """True/False if Yahoo knows ``symbol``; None if Yahoo could not be reached."""
    quotes = _search_or_none(symbol)
    if quotes is None:
        return None
    return any(q["symbol"] == symbol and q["type"] in _SYMBOL_TYPES for q in quotes)


def _is_stock_symbol(symbol: str) -> bool:
    """True if Yahoo lists ``symbol`` exactly, as a stock."""
    return any(
        q["symbol"] == symbol and q["type"] in _STOCK_TYPES for q in _search_or_none(symbol) or ()
    )


def _closest_symbol(symbol: str) -> str | None:
    """Best guess at the stock a mistyped symbol was meant to be.

    1. Yahoo's fuzzy search.
    2. If that finds nothing, the symbol with one letter removed - the most common
       typo is an extra or doubled letter ("NVDAA" -> "NVDA").
    """
    for quote in _search_or_none(symbol, fuzzy=True) or ():
        if quote["type"] in _STOCK_TYPES and quote["symbol"] and quote["symbol"] != symbol:
            return quote["symbol"]
    if len(symbol) < _MIN_TYPO_LENGTH:
        return None
    variants = dict.fromkeys(symbol[:i] + symbol[i + 1 :] for i in range(len(symbol)))
    for variant in variants:
        if _is_stock_symbol(variant):
            return variant
    return None


def _content_runs(phrase: str) -> list[str]:
    """Split a phrase into runs of up to 3 consecutive non-filler words."""
    runs: list[str] = []
    current: list[str] = []
    for raw in phrase.split():
        # Drop surrounding punctuation and a possessive "'s" ("Costco's" -> "Costco").
        word = _POSSESSIVE.sub("", raw.strip('.,;:!?"()')).strip("'\u2019")
        if not word or word.lower() in _NOT_NAMES or word.isupper():
            # Filler words end a run; all-caps words are symbols, handled elsewhere.
            if current:
                runs.append(" ".join(current))
            current = []
            continue
        current.append(word)
        if len(current) == 3:
            runs.append(" ".join(current))
            current = []
    if current:
        runs.append(" ".join(current))
    return [run for run in runs if len(run) >= 3]


def _name_candidates(text: str) -> list[str]:
    """Phrases that could be company names.

    * Capitalised phrases anywhere ("Costco", "Home Depot").
    * Lowercase phrases right after a cue word ("tell me about nvidia",
      "financials of costco"), so lower-case questions work too.
    """
    candidates: list[str] = []
    phrases = [m.group(0) for m in _NAME_CANDIDATE.finditer(text)]
    phrases += [m.group(1) for m in _CUE_PHRASE.finditer(text)]
    for phrase in phrases:
        for run in _content_runs(phrase):
            if run.lower() not in {c.lower() for c in candidates}:
                candidates.append(run)
    # Drop fragments of longer candidates ("home" when "home depot" is present), so a
    # partial name cannot resolve to a different company.
    lowered = [c.lower() for c in candidates]
    return [
        candidate
        for candidate, low in zip(candidates, lowered, strict=True)
        if not any(low != other and f" {low} " in f" {other} " for other in lowered)
    ]


def _normalise_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _name_matches(candidate: str, quote: dict[str, Any]) -> bool:
    """True if ``candidate`` is the company's ticker or the start of its name.

    Guards against Yahoo's loose matching: "operating margin" must not resolve to an
    unrelated company. "nvidia" matches "NVIDIA Corporation", "home depot" matches
    "The Home Depot, Inc.".
    """
    if candidate.upper() == quote["symbol"]:
        return True
    words = _normalise_words(candidate)
    name = _normalise_words(quote["name"])
    if name[:1] == ["the"]:
        name = name[1:]
    return bool(words) and name[: len(words)] == words


def _lookup_company(name: str) -> str | None:
    """Ticker of the first US-listed stock in Yahoo's results whose name starts with ``name``.

    Foreign and OTC listings are skipped: a place name such as "Tokyo" in "What's the
    weather in Tokyo?" must not resolve to Tokyo Electron (8035.T).
    """
    for quote in _search_or_none(name) or ():
        if (
            quote["type"] in _STOCK_TYPES
            and quote["symbol"]
            and _is_us_listing(quote)
            and _name_matches(name, quote)
        ):
            return quote["symbol"]
    return None


def resolve_tickers(text: str | None) -> TickerResolution:
    """Find the companies a question is about, confirming them with Yahoo Finance.

    1. Typed symbols (``extract_tickers``) are confirmed with Yahoo; unconfirmed ones
       are dropped. If Yahoo cannot be reached, typed symbols are accepted unchecked.
    2. If nothing was confirmed, likely company names are looked up by name
       ("How is Costco doing?" -> COST, "tell me about nvidia" -> NVDA). A result is
       only accepted when the company's name starts with the searched words.
    3. If still nothing, an unrecognised typed symbol is reported with Yahoo's closest
       match, so the caller can ask "did you mean ...?" instead of running research.

    Args:
        text: Free-text question from the user.

    Returns:
        The confirmed tickers, or the unknown symbol and a suggestion.
    """
    if not text:
        return TickerResolution()

    confirmed: list[str] = []
    unknown: str | None = None
    for symbol in extract_tickers(text):
        exists = _symbol_exists(symbol)
        if exists is None or exists:  # None = Yahoo unreachable -> accept (fail open)
            confirmed.append(symbol)
        elif unknown is None:
            unknown = symbol

    if not confirmed:
        for name in _name_candidates(text):
            found = _lookup_company(name)
            if found and found not in confirmed:
                logger.info("Resolved company name %r to %s via Yahoo Finance", name, found)
                confirmed.append(found)

    if confirmed:
        return TickerResolution(tickers=tuple(confirmed))
    if unknown:
        return TickerResolution(unknown=unknown, suggestion=_closest_symbol(unknown))
    return TickerResolution()
