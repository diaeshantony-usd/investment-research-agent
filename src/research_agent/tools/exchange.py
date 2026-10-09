"""
tools/exchange.py
==================
Tool for listing actively traded stock ticker symbols on a US exchange.

Sourced from Nasdaq Trader's public symbol directory
(https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs), which is free,
requires no API key, and flags delisted/test symbols so they can be
excluded - this is what "active" means here: currently listed and not a
test issue.

The downloaded directory is cached to ``data/cache/`` for a day (see
``tools/cache.py``), so repeated calls and offline re-runs don't re-fetch a
multi-thousand-row file every time.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
import requests

from research_agent.agent_framework import tool
from research_agent.config import logger
from research_agent.tools.cache import ONE_DAY, cached

_NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
_OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# otherlisted.txt's single-letter exchange codes.
_EXCHANGE_CODES: dict[str, str] = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}

_KNOWN_EXCHANGES = frozenset({"NASDAQ", *_EXCHANGE_CODES.values()})
_DEFAULT_LIMIT = 50


@cached("exchange_symbol_directory", ttl_seconds=ONE_DAY)
def _download_symbol_directory(url: str) -> list[dict[str, Any]]:
    """Download and parse one of Nasdaq Trader's pipe-delimited symbol directories.

    Cached for a day (listings barely change intraday) and returned as
    records rather than a DataFrame, so the cache entry is plain JSON.
    """
    logger.info("Downloading exchange symbol directory from %s", url)
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    # The last line is a trailer ("File Creation Time: ..."), not a data row.
    lines = [
        line for line in response.text.splitlines() if not line.startswith("File Creation Time")
    ]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
    return df.to_dict("records")


def _load_nasdaq_listed() -> pd.DataFrame:
    """Active (non-test) NASDAQ-listed symbols, normalised to the common schema."""
    df = pd.DataFrame(_download_symbol_directory(_NASDAQ_LISTED_URL))
    active = df[(df["Test Issue"] == "N") & (df["Financial Status"] == "N")]
    return pd.DataFrame(
        {
            "ticker": active["Symbol"],
            "name": active["Security Name"],
            "exchange": "NASDAQ",
            "is_etf": active["ETF"] == "Y",
        }
    )


def _load_other_listed(exchange_filter: str | None) -> pd.DataFrame:
    """Active (non-test) NYSE/NYSE American/NYSE Arca/Cboe BZX/IEX symbols."""
    df = pd.DataFrame(_download_symbol_directory(_OTHER_LISTED_URL))
    active = df[df["Test Issue"] == "N"]
    exchange_names = active["Exchange"].map(_EXCHANGE_CODES).fillna(active["Exchange"])

    result = pd.DataFrame(
        {
            "ticker": active["ACT Symbol"],
            "name": active["Security Name"],
            "exchange": exchange_names,
            "is_etf": active["ETF"] == "Y",
        }
    )
    if exchange_filter:
        result = result[result["exchange"].str.upper() == exchange_filter.upper()]
    return result


def _load_listings(exchange: str) -> pd.DataFrame:
    """Load and combine the symbol directories relevant to ``exchange`` ('ALL' for every one)."""
    frames = []
    if exchange in ("NASDAQ", "ALL"):
        frames.append(_load_nasdaq_listed())
    if exchange != "NASDAQ":
        frames.append(_load_other_listed(None if exchange == "ALL" else exchange))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@tool(
    name="get_active_tickers",
    description=(
        "Lists actively traded stock ticker symbols on a US exchange (NASDAQ, NYSE, NYSE "
        "American, NYSE Arca, Cboe BZX, IEX, or 'ALL'), sourced from Nasdaq Trader's public "
        "symbol directory. Optionally filter by a ticker/company-name search term."
    ),
)
def get_active_tickers(
    exchange: str = "NASDAQ", query: str = "", limit: int = _DEFAULT_LIMIT
) -> str:
    """Return active ticker symbols for a US exchange.

    Args:
        exchange: One of 'NASDAQ', 'NYSE', 'NYSE American', 'NYSE Arca',
            'Cboe BZX', 'IEX', or 'ALL' for every exchange in the directory.
        query: Optional case-insensitive substring to filter by ticker or
            company name (e.g. 'bank', 'NVD').
        limit: Maximum number of rows to return; the directories list
            several thousand symbols in total.

    Returns:
        JSON with the matched ``count`` and a ``tickers`` list of
        ``{ticker, name, exchange, is_etf}`` (capped at ``limit``), or an
        ``error`` field if the symbol directory could not be downloaded or
        ``exchange`` is not recognised.
    """
    exchange_clean = (exchange or "NASDAQ").strip().upper()
    logger.info(
        "Executing get_active_tickers exchange=%s query=%r limit=%s",
        exchange_clean,
        query,
        limit,
    )

    if exchange_clean != "ALL" and exchange_clean not in {e.upper() for e in _KNOWN_EXCHANGES}:
        known = sorted(_KNOWN_EXCHANGES)
        return json.dumps(
            {
                "exchange": exchange_clean,
                "tickers": [],
                "error": f"Unknown exchange '{exchange_clean}'. Known: {known} or 'ALL'.",
            }
        )

    try:
        listings = _load_listings(exchange_clean)
    except requests.RequestException as exc:
        logger.warning("get_active_tickers failed to download symbol directory: %s", exc)
        return json.dumps(
            {
                "exchange": exchange_clean,
                "tickers": [],
                "error": f"Could not download exchange symbol directory: {exc}",
            }
        )

    if query:
        mask = listings["ticker"].str.contains(query, case=False, na=False) | listings[
            "name"
        ].str.contains(query, case=False, na=False)
        listings = listings[mask]

    total = len(listings)
    rows = listings.head(max(1, limit)).to_dict("records")
    tickers: list[dict[str, Any]] = [
        {
            "ticker": row["ticker"],
            "name": row["name"],
            "exchange": row["exchange"],
            "is_etf": bool(row["is_etf"]),
        }
        for row in rows
    ]

    return json.dumps({"exchange": exchange_clean, "count": total, "tickers": tickers}, indent=2)
