"""
tools/news.py
=============
News tools for live headlines about a stock ticker, time-windowed:

- ``get_news_data``       - a rolling lookback window (last N days)
- ``get_news_by_period``  - an explicit date range (from_date, to_date)

Three layers of fallback, in order, matching the design doc's data-sources
table:

1. NewsAPI.org's ``/v2/everything`` (primary; full-text search by company name)
2. yfinance's ``Ticker.news`` (used when NewsAPI has no key, fails, or
   returns zero results; filtered down to the requested date window)
3. The last cached response for that exact query/window (used when both
   live sources fail; see ``tools/cache.py``)

Both tools raise no exception to the caller: on exhausted fallbacks, a bad
request or zero results, they return a JSON payload with an
``error``/``message`` field, so the calling agent can report the gap
instead of fabricating news.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import yfinance as yf

from research_agent.agent_framework import tool
from research_agent.config import NEWS_API_KEY, logger
from research_agent.tools.cache import ONE_HOUR, cached
from research_agent.tools.yahoo_finance import get_ticker_details

_MAX_ARTICLES = 10

# Strips a trailing legal suffix ("NVIDIA Corporation" -> "NVIDIA", "Apple
# Inc." -> "Apple") so the NewsAPI query reads like prose, not a filing name.
_LEGAL_SUFFIX_RE = re.compile(
    r"\s*,?\s*(?:&|and)?\s*\b(Corporation|Corp|Incorporated|Inc|Company|Co|Holdings?|Group|"
    r"Limited|Ltd|PLC|LLC|L\.P\.)\.?\s*$",
    re.IGNORECASE,
)


def _simplify_company_name(name: str) -> str:
    """Repeatedly strip trailing legal suffixes, e.g. 'X Holdings, Inc.' -> 'X'."""
    simplified = name.strip()
    while True:
        stripped = _LEGAL_SUFFIX_RE.sub("", simplified).strip()
        if stripped == simplified:
            return simplified or name
        simplified = stripped


def _search_term(ticker: str) -> str:
    """Resolve a ticker to the search term NewsAPI is most likely to match on.

    NewsAPI's full-text search matches prose far better on a company name
    than a bare ticker symbol ("NVIDIA" appears in articles; "NVDA" rarely
    does). The name is looked up live via ``get_ticker_details`` (already
    cached, with its own Alpha Vantage fallback) instead of a maintained
    ticker->name table, so any ticker works, not just a hardcoded few.
    """
    clean = (ticker or "").strip().upper()
    try:
        details = get_ticker_details.func(clean)
    except Exception as exc:  # noqa: BLE001 - any lookup failure just falls back to the raw ticker
        logger.warning(
            "Could not resolve a company name for '%s' (%s); searching the ticker as-is",
            clean,
            exc,
        )
        return clean

    name = details.get("name")
    return _simplify_company_name(name) if name else clean


def _client() -> Any:
    """Build a ``NewsApiClient``.

    Raises:
        ValueError: If ``NEWS_API_KEY`` is not configured.
    """
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY is not set. Add it to .env to enable live news search.")

    from newsapi import NewsApiClient  # local import: optional/heavy dependency

    return NewsApiClient(api_key=NEWS_API_KEY)


def _validate_date_range(from_date: str, to_date: str) -> None:
    """Raises ``ValueError`` if either date is malformed or ``from_date`` is after ``to_date``."""
    try:
        start = datetime.strptime(from_date, "%Y-%m-%d").date()
        end = datetime.strptime(to_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Dates must be in 'YYYY-MM-DD' format: {exc}") from exc
    if start > end:
        raise ValueError(f"from_date ({from_date}) must not be after to_date ({to_date})")


def _format_articles(raw_articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim a NewsAPI ``articles`` response down to the fields the agents need."""
    return [
        {
            "headline": article.get("title"),
            "source": (article.get("source") or {}).get("name"),
            "date": (article.get("publishedAt") or "")[:10],
            "url": article.get("url"),
            "summary": article.get("description"),
        }
        for article in raw_articles[:_MAX_ARTICLES]
    ]


def _fetch_from_newsapi(query: str, from_date: str, to_date: str) -> list[dict[str, Any]]:
    """Primary source: NewsAPI.org's ``/v2/everything`` endpoint.

    Raises:
        ValueError: If the API key is missing, the request fails, NewsAPI
            reports an error status, or it returns zero articles (so the
            caller moves on to the next fallback rather than reporting
            "no news" when another source might still have some).
    """
    client = _client()
    try:
        response = client.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language="en",
            sort_by="publishedAt",
            page_size=_MAX_ARTICLES,
        )
    except Exception as exc:
        raise ValueError(f"NewsAPI request failed for '{query}': {exc}") from exc

    if response.get("status") != "ok":
        raise ValueError(f"NewsAPI error for '{query}': {response.get('message', response)}")

    articles = _format_articles(response.get("articles", []))
    if not articles:
        raise ValueError(f"NewsAPI returned zero articles for '{query}'")
    return articles


def _parse_article_date(raw: Any) -> datetime | None:
    """Best-effort ISO-8601 parse; returns ``None`` instead of raising on a bad value."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _fetch_from_yfinance(ticker: str, from_date: str, to_date: str) -> list[dict[str, Any]]:
    """Fallback source: yfinance's own news feed for a ticker, filtered to the date window.

    yfinance only exposes recent news (no arbitrary historical search), so
    this mainly helps ``get_news_data``'s rolling window; an old
    ``get_news_by_period`` range will usually find nothing here either, and
    the caller falls through to a cached response instead.

    Raises:
        ValueError: If the request fails, or there is no news in range.
    """
    try:
        raw_articles = yf.Ticker(ticker).news or []
    except Exception as exc:
        raise ValueError(f"yfinance news request failed for '{ticker}': {exc}") from exc

    if not raw_articles:
        raise ValueError(f"yfinance returned no news for '{ticker}'")

    window_start = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    window_end = datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(
        days=1
    )

    articles = []
    for item in raw_articles:
        # Newer yfinance nests article fields under "content"; older versions do not.
        content = item.get("content", item)
        published = _parse_article_date(content.get("pubDate") or content.get("displayTime"))
        if published is not None and not (window_start <= published < window_end):
            continue

        provider = content.get("provider") or {}
        link = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        articles.append(
            {
                "headline": content.get("title"),
                "source": provider.get("displayName") or "Yahoo Finance",
                "date": str(published.date()) if published else None,
                "url": link.get("url") if isinstance(link, dict) else None,
                "summary": content.get("summary"),
            }
        )
        if len(articles) >= _MAX_ARTICLES:
            break

    if not articles:
        raise ValueError(f"yfinance had no news for '{ticker}' between {from_date} and {to_date}")
    return articles


@cached("news_articles", ttl_seconds=ONE_HOUR)
def _fetch_articles(ticker: str, query: str, from_date: str, to_date: str) -> list[dict[str, Any]]:
    """Fetch articles for a ticker/date-window: NewsAPI primary, yfinance fallback.

    Cached by (ticker, query, from_date, to_date): a failure of both live
    sources is never cached as a success, and if that happens the ``cached``
    decorator itself serves the last cached result for this exact window -
    a third fallback layer - before finally propagating the error.

    Raises:
        ValueError: If both NewsAPI and yfinance fail or return nothing, and
            there is no cached fallback either.
    """
    errors = []
    try:
        return _fetch_from_newsapi(query, from_date, to_date)
    except ValueError as exc:
        errors.append(f"NewsAPI: {exc}")

    try:
        return _fetch_from_yfinance(ticker, from_date, to_date)
    except ValueError as exc:
        errors.append(f"yfinance: {exc}")

    raise ValueError(f"All news sources failed for '{ticker}': " + "; ".join(errors))


@tool(
    name="get_news_data",
    description=(
        "Retrieves recent news headlines, sources, dates, URLs and summaries for a stock "
        "ticker from NewsAPI.org, over a rolling lookback window (default: last 7 days)."
    ),
)
def get_news_data(ticker: str, days: int = 7) -> str:
    """Returns news headlines for a ticker over a rolling lookback window.

    Args:
        ticker: Stock symbol (e.g. 'NVDA', 'AAPL').
        days: How many days back to search. NewsAPI's free developer tier
            only covers roughly the last month.

    Returns:
        JSON with the ticker, the search window and an ``articles`` list; an
        ``error`` field when the key is missing or the request failed, or a
        ``message`` field when the request succeeded with zero articles.
    """
    clean_ticker = (ticker or "").strip().upper()
    query = _search_term(clean_ticker)
    today = datetime.now(timezone.utc).date()
    from_date = (today - timedelta(days=max(days, 1))).isoformat()
    to_date = today.isoformat()

    logger.info(
        "Executing get_news_data for %s (query=%r, %s to %s)",
        clean_ticker,
        query,
        from_date,
        to_date,
    )
    return _run_search(clean_ticker, query, from_date, to_date, days=days)


@tool(
    name="get_news_by_period",
    description=(
        "Retrieves news headlines for a stock ticker from NewsAPI.org within an explicit "
        "date range (from_date, to_date, both 'YYYY-MM-DD'), instead of a rolling window."
    ),
)
def get_news_by_period(ticker: str, from_date: str, to_date: str) -> str:
    """Returns news headlines for a ticker within an explicit date range.

    Args:
        ticker: Stock symbol (e.g. 'NVDA', 'AAPL').
        from_date: Start date, inclusive, as 'YYYY-MM-DD'.
        to_date: End date, inclusive, as 'YYYY-MM-DD'.

    Returns:
        JSON with the ticker, the requested period and an ``articles`` list;
        an ``error`` field on a bad date range, missing key or failed
        request, or a ``message`` field when there are zero articles.
    """
    clean_ticker = (ticker or "").strip().upper()
    query = _search_term(clean_ticker)

    try:
        _validate_date_range(from_date, to_date)
    except ValueError as exc:
        logger.warning("get_news_by_period rejected dates for %s: %s", clean_ticker, exc)
        return json.dumps({"ticker": clean_ticker, "articles": [], "error": str(exc)})

    logger.info(
        "Executing get_news_by_period for %s (query=%r, %s to %s)",
        clean_ticker,
        query,
        from_date,
        to_date,
    )
    return _run_search(clean_ticker, query, from_date, to_date)


def _run_search(
    ticker: str, query: str, from_date: str, to_date: str, days: int | None = None
) -> str:
    """Shared tail end of both tools: fetch, then shape the JSON response."""
    try:
        articles = _fetch_articles(ticker, query, from_date, to_date)
    except ValueError as exc:
        logger.warning("News search failed for %s: %s", ticker, exc)
        return json.dumps(
            {"ticker": ticker, "from": from_date, "to": to_date, "articles": [], "error": str(exc)}
        )

    if not articles:
        window = f"the last {days} day(s)" if days else f"{from_date} to {to_date}"
        return json.dumps(
            {
                "ticker": ticker,
                "from": from_date,
                "to": to_date,
                "articles": [],
                "message": f"No news found for '{ticker}' in {window}.",
            }
        )

    return json.dumps(
        {"ticker": ticker, "from": from_date, "to": to_date, "articles": articles}, indent=2
    )
