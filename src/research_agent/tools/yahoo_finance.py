"""
tools/yahoo_finance.py
=======================
Yahoo Finance tools, backed by ``yfinance``, for live ticker data.

Each public function is decorated with ``@tool`` so agents can call it
directly:

- ``get_ticker_details``      - company profile: name, sector, exchange, market cap, ratios
- ``get_historical_data``     - OHLCV candles for one ticker via ``Ticker.history()``
- ``download_price_history``  - bulk OHLCV for one or more tickers via ``yf.download()``
- ``get_financial_statements``- annual/quarterly income statement, balance sheet, cash flow

``get_ticker_details`` and ``get_financial_statements`` fall back to Alpha
Vantage (``tools/alpha_vantage.py``) when yfinance raises and
``ALPHA_VANTAGE_KEY`` is configured; the returned dict's ``source`` field
says which one actually answered. Every function still raises ``ValueError``
when neither source has data; ``Tool.execute()`` catches that and reports a
clear error back to the agent instead of crashing the run.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import yfinance as yf

from research_agent.agent_framework import tool
from research_agent.config import logger
from research_agent.tools import alpha_vantage
from research_agent.tools.cache import ONE_DAY, ONE_HOUR, SIX_HOURS, cached


def get_ticker(ticker: str) -> yf.Ticker:
    """Return a ``yfinance.Ticker`` for a cleaned, upper-cased symbol.

    Args:
        ticker: Stock symbol (e.g. 'nvda', ' AAPL ').

    Returns:
        A ``yfinance.Ticker`` instance.

    Raises:
        ValueError: If ``ticker`` is blank.
    """
    clean = (ticker or "").strip().upper()
    if not clean:
        raise ValueError("ticker must not be empty")
    return yf.Ticker(clean)


def _get_ticker_details_yfinance(ticker: str) -> dict[str, Any]:
    """Yfinance's half of ``get_ticker_details``; raises ``ValueError`` on failure."""
    t = get_ticker(ticker)
    logger.info("yfinance: fetching ticker details for %s", ticker)

    try:
        info = t.get_info()
    except Exception as exc:
        raise ValueError(f"No ticker details found for '{ticker}': {exc}") from exc

    has_price = info.get("currentPrice") is not None or info.get("regularMarketPrice") is not None
    if not info or not has_price:
        raise ValueError(f"No ticker details found for '{ticker}'")

    return {
        "ticker": info.get("symbol", ticker),
        "name": info.get("longName") or info.get("shortName"),
        "exchange": info.get("exchange"),
        "currency": info.get("currency"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "employees": info.get("fullTimeEmployees"),
        "market_cap": info.get("marketCap"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "previous_close": info.get("previousClose"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "beta": info.get("beta"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield": info.get("dividendYield"),
    }


@tool(
    name="get_ticker_details",
    description=(
        "Retrieves a company's live profile - name, exchange, sector, industry, market cap, "
        "current price, 52-week range, beta and P/E ratios - from Yahoo Finance, falling "
        "back to Alpha Vantage if Yahoo Finance has no data for the ticker."
    ),
)
@cached("get_ticker_details", ttl_seconds=ONE_HOUR)
def get_ticker_details(ticker: str) -> dict[str, Any]:
    """Return company profile and key ratios for a ticker.

    Tries yfinance first; if it raises and ``ALPHA_VANTAGE_KEY`` is set,
    falls back to Alpha Vantage's OVERVIEW/GLOBAL_QUOTE endpoints.

    Args:
        ticker: Stock symbol (e.g. 'NVDA').

    Returns:
        Name, exchange, currency, sector, industry, market cap, current
        price, 52-week range, beta, P/E ratios, and a ``source`` field
        naming which provider actually answered.

    Raises:
        ValueError: If neither yfinance nor Alpha Vantage has a profile for
            this ticker.
    """
    clean = (ticker or "").strip().upper()

    try:
        details = _get_ticker_details_yfinance(clean)
    except Exception as yf_exc:
        if not alpha_vantage.is_configured():
            raise ValueError(f"No ticker details found for '{clean}': {yf_exc}") from yf_exc
        try:
            details = alpha_vantage.get_ticker_details(clean)
        except Exception as av_exc:
            raise ValueError(
                f"No ticker details found for '{clean}': yfinance error: {yf_exc}; "
                f"Alpha Vantage fallback error: {av_exc}"
            ) from av_exc
        else:
            details["source"] = "alpha_vantage"
            logger.warning(
                "yfinance failed for %s (%s); served Alpha Vantage fallback", clean, yf_exc
            )
    else:
        details["source"] = "yfinance"
    return details


@tool(
    name="get_historical_data",
    description=(
        "Retrieves daily/weekly/monthly OHLCV price candles for one stock ticker from Yahoo "
        "Finance over a lookback window, e.g. period='6mo', interval='1d'."
    ),
)
@cached("get_historical_data", ttl_seconds=SIX_HOURS)
def get_historical_data(
    ticker: str, period: str = "6mo", interval: str = "1d"
) -> list[dict[str, Any]]:
    """Return OHLCV candles for one ticker via ``Ticker.history()``.

    Args:
        ticker: Stock symbol (e.g. 'NVDA').
        period: Lookback window, e.g. '1mo', '6mo', '1y', 'max'.
        interval: Candle size, e.g. '1d', '1wk', '1mo'.

    Returns:
        Rows of ``{date, open, high, low, close, volume}``, oldest first.

    Raises:
        ValueError: If no price history is returned for this ticker/period.
    """
    clean = (ticker or "").strip().upper()
    t = get_ticker(clean)
    logger.info(
        "yfinance: fetching history for %s period=%s interval=%s", clean, period, interval
    )

    history = t.history(period=period, interval=interval)
    if history.empty:
        raise ValueError(f"No historical data found for '{clean}' (period={period!r})")

    return _frame_to_candles(history.reset_index())


@tool(
    name="download_price_history",
    description=(
        "Downloads OHLCV price history for one or more tickers from Yahoo Finance in a "
        "single batch call. Pass tickers as a single symbol, a comma/space-separated "
        "string (e.g. 'NVDA, AAPL'), or a list of symbols."
    ),
)
@cached("download_price_history", ttl_seconds=SIX_HOURS)
def download_price_history(
    tickers: str | list[str], period: str = "6mo", interval: str = "1d"
) -> dict[str, list[dict[str, Any]]]:
    """Bulk OHLCV download for one or more tickers via ``yf.download()``.

    Args:
        tickers: A single symbol, a comma/space-separated string of symbols,
            or a list of symbols.
        period: Lookback window, e.g. '1mo', '6mo', '1y', 'max'.
        interval: Candle size, e.g. '1d', '1wk', '1mo'.

    Returns:
        ``{ticker: [candles...]}`` for every requested symbol, in the same
        ``{date, open, high, low, close, volume}`` shape as
        :func:`get_historical_data`.

    Raises:
        ValueError: If no tickers are given, or the download returns nothing.
    """
    symbols = re.split(r"[,\s]+", tickers) if isinstance(tickers, str) else list(tickers)
    clean_symbols = [s.strip().upper() for s in symbols if s and s.strip()]
    if not clean_symbols:
        raise ValueError("At least one ticker must be provided")

    logger.info(
        "yfinance: downloading price history for %s period=%s interval=%s",
        clean_symbols,
        period,
        interval,
    )
    data = yf.download(
        clean_symbols,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"yf.download() returned no data for {clean_symbols}")

    is_multi = isinstance(data.columns, pd.MultiIndex)
    result: dict[str, list[dict[str, Any]]] = {}
    for symbol in clean_symbols:
        frame = data[symbol] if is_multi else data
        frame = frame.dropna(how="all").reset_index()
        result[symbol] = _frame_to_candles(frame)
    return result


def _get_financial_statements_yfinance(ticker: str) -> dict[str, Any]:
    """Yfinance's half of ``get_financial_statements``; raises ``ValueError`` on failure."""
    t = get_ticker(ticker)
    logger.info("yfinance: fetching financial statements for %s", ticker)

    statements = {
        "ticker": ticker,
        "annual_income_statement": _statement_to_records(t.financials),
        "quarterly_income_statement": _statement_to_records(t.quarterly_financials),
        "annual_balance_sheet": _statement_to_records(t.balance_sheet),
        "annual_cash_flow": _statement_to_records(t.cashflow),
    }
    if not any(statements[k] for k in statements if k != "ticker"):
        raise ValueError(f"No financial statements found for '{ticker}'")
    return statements


@tool(
    name="get_financial_statements",
    description=(
        "Retrieves annual and quarterly income statement, balance sheet and cash flow line "
        "items for a ticker from Yahoo Finance, falling back to Alpha Vantage if Yahoo "
        "Finance has no statements for the ticker."
    ),
)
@cached("get_financial_statements", ttl_seconds=ONE_DAY)
def get_financial_statements(ticker: str) -> dict[str, Any]:
    """Return annual and quarterly financial statements for a ticker.

    Tries yfinance first; if it raises and ``ALPHA_VANTAGE_KEY`` is set,
    falls back to Alpha Vantage's INCOME_STATEMENT/BALANCE_SHEET/CASH_FLOW
    endpoints.

    Args:
        ticker: Stock symbol (e.g. 'NVDA').

    Returns:
        ``annual_income_statement``, ``quarterly_income_statement``,
        ``annual_balance_sheet`` and ``annual_cash_flow`` (each a list of
        per-period line-item dicts), plus a ``source`` field naming which
        provider actually answered. Alpha Vantage has no quarterly balance
        sheet/cash flow equivalent wired up, so those stay annual-only
        regardless of source.

    Raises:
        ValueError: If neither yfinance nor Alpha Vantage has statements for
            this ticker.
    """
    clean = (ticker or "").strip().upper()

    try:
        statements = _get_financial_statements_yfinance(clean)
    except Exception as yf_exc:
        if not alpha_vantage.is_configured():
            raise ValueError(f"No financial statements found for '{clean}': {yf_exc}") from yf_exc
        try:
            statements = alpha_vantage.get_financial_statements(clean)
        except Exception as av_exc:
            raise ValueError(
                f"No financial statements found for '{clean}': yfinance error: {yf_exc}; "
                f"Alpha Vantage fallback error: {av_exc}"
            ) from av_exc
        else:
            statements["source"] = "alpha_vantage"
            logger.warning(
                "yfinance failed for %s (%s); served Alpha Vantage fallback", clean, yf_exc
            )
    else:
        statements["source"] = "yfinance"
    return statements


def _frame_to_candles(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a yfinance OHLCV frame (with a Date/Datetime column) to candle dicts."""
    date_col = "Date" if "Date" in frame.columns else "Datetime"
    candles = []
    for row in frame.to_dict("records"):
        if pd.isna(row.get("Close")):
            continue
        candles.append(
            {
                "date": str(row[date_col]),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]) if not pd.isna(row.get("Volume")) else None,
            }
        )
    return candles


def _statement_to_records(statement: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a yfinance financial-statement frame (line items x periods) to per-period dicts."""
    if statement is None or statement.empty:
        return []
    records = []
    for period, column in statement.items():
        record: dict[str, Any] = {
            "period": str(period.date()) if hasattr(period, "date") else str(period)
        }
        for line_item, value in column.items():
            record[str(line_item)] = None if pd.isna(value) else float(value)
        records.append(record)
    return records
