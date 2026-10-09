"""
tools/alpha_vantage.py
=======================
Alpha Vantage REST client, used only as a fallback when yfinance has no data
for a ticker (rate-limited, delisted, or otherwise unavailable).

Not exposed as agent-facing ``@tool`` functions on its own -
``get_ticker_details`` and ``get_financial_statements`` in
``tools/yahoo_finance.py`` call into this module only after their primary
yfinance call fails, and only when ``ALPHA_VANTAGE_KEY`` is configured.

Free-tier quirks handled here:

- A missing/invalid key or an exhausted daily quota comes back as HTTP 200
  with a ``"Note"`` or ``"Information"`` field instead of a real error.
- Every field in the JSON response is a string; numeric fields are coerced
  to ``float`` so the shape matches the yfinance-backed functions exactly.
"""

from __future__ import annotations

from typing import Any

import requests

from research_agent.config import ALPHA_VANTAGE_KEY, logger

_BASE_URL = "https://www.alphavantage.co/query"


def is_configured() -> bool:
    """True if ``ALPHA_VANTAGE_KEY`` is set, so a fallback attempt is worth making."""
    return bool(ALPHA_VANTAGE_KEY)


def _safe_float(value: Any) -> float | None:
    if value in (None, "None", "-", ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get(params: dict[str, str]) -> dict[str, Any]:
    """GET one Alpha Vantage endpoint and return its JSON.

    Raises:
        ValueError: If the key is missing, the request fails, or Alpha
            Vantage reports a rate-limit/invalid-input message instead of
            real data.
    """
    if not ALPHA_VANTAGE_KEY:
        raise ValueError("ALPHA_VANTAGE_KEY is not set")

    try:
        response = requests.get(
            _BASE_URL, params={**params, "apikey": ALPHA_VANTAGE_KEY}, timeout=15
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise ValueError(f"Alpha Vantage request failed: {exc}") from exc

    if "Note" in data or "Information" in data:
        notice = data.get("Note") or data.get("Information")
        raise ValueError(f"Alpha Vantage rate limit/notice: {notice}")
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    return data


def get_ticker_details(ticker: str) -> dict[str, Any]:
    """Company profile and live quote via the ``OVERVIEW`` and ``GLOBAL_QUOTE`` endpoints.

    Args:
        ticker: Stock symbol, already cleaned/upper-cased by the caller.

    Returns:
        The same field shape as ``yahoo_finance.get_ticker_details``.

    Raises:
        ValueError: If Alpha Vantage has no overview for this ticker.
    """
    logger.info("Alpha Vantage: fetching ticker details for %s", ticker)

    overview = _get({"function": "OVERVIEW", "symbol": ticker})
    if not overview or "Symbol" not in overview:
        raise ValueError(f"Alpha Vantage has no overview for '{ticker}'")

    current_price = None
    previous_close = None
    try:
        quote = _get({"function": "GLOBAL_QUOTE", "symbol": ticker}).get("Global Quote", {})
        current_price = _safe_float(quote.get("05. price"))
        previous_close = _safe_float(quote.get("08. previous close"))
    except ValueError:
        pass  # the live quote is a bonus; OVERVIEW's fields are the primary payload

    return {
        "ticker": overview.get("Symbol", ticker),
        "name": overview.get("Name"),
        "exchange": overview.get("Exchange"),
        "currency": overview.get("Currency"),
        "sector": overview.get("Sector"),
        "industry": overview.get("Industry"),
        "website": None,
        "employees": None,
        "market_cap": _safe_float(overview.get("MarketCapitalization")),
        "current_price": current_price,
        "previous_close": previous_close,
        "fifty_two_week_high": _safe_float(overview.get("52WeekHigh")),
        "fifty_two_week_low": _safe_float(overview.get("52WeekLow")),
        "beta": _safe_float(overview.get("Beta")),
        "trailing_pe": _safe_float(overview.get("TrailingPE") or overview.get("PERatio")),
        "forward_pe": _safe_float(overview.get("ForwardPE")),
        "dividend_yield": _safe_float(overview.get("DividendYield")),
    }


def _reports_to_records(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Convert one Alpha Vantage statement payload's report list to per-period dicts."""
    records = []
    for report in payload.get(key) or []:
        record: dict[str, Any] = {"period": report.get("fiscalDateEnding")}
        for field, value in report.items():
            if field == "fiscalDateEnding":
                continue
            record[field] = value if field == "reportedCurrency" else _safe_float(value)
        records.append(record)
    return records


def get_financial_statements(ticker: str) -> dict[str, Any]:
    """Income statement, balance sheet and cash flow via Alpha Vantage.

    Args:
        ticker: Stock symbol, already cleaned/upper-cased by the caller.

    Returns:
        The same field shape as ``yahoo_finance.get_financial_statements``.

    Raises:
        ValueError: If none of the three statements return any reports.
    """
    logger.info("Alpha Vantage: fetching financial statements for %s", ticker)

    income = _get({"function": "INCOME_STATEMENT", "symbol": ticker})
    balance = _get({"function": "BALANCE_SHEET", "symbol": ticker})
    cashflow = _get({"function": "CASH_FLOW", "symbol": ticker})

    statements = {
        "ticker": ticker,
        "annual_income_statement": _reports_to_records(income, "annualReports"),
        "quarterly_income_statement": _reports_to_records(income, "quarterlyReports"),
        "annual_balance_sheet": _reports_to_records(balance, "annualReports"),
        "annual_cash_flow": _reports_to_records(cashflow, "annualReports"),
    }
    if not any(statements[k] for k in statements if k != "ticker"):
        raise ValueError(f"Alpha Vantage has no financial statements for '{ticker}'")
    return statements
