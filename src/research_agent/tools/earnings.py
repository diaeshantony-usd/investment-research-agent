"""
tools/earnings.py
=================
Tool for fetching quarterly earnings figures, revenue, and fundamentals.

Composes the live Yahoo Finance tools in ``tools/yahoo_finance.py`` rather
than keeping its own fixed ticker list: quarterly revenue/gross
margin/operating and net income come from ``get_financial_statements``,
forward P/E from ``get_ticker_details``, and free cash flow from the
ticker's own quarterly cash flow statement. Both reused tools already fall
back to Alpha Vantage internally when yfinance has nothing for a ticker.

EPS actual/estimate/surprise comes from ``Ticker.get_earnings_dates()``;
when that's empty and ``ALPHA_VANTAGE_KEY`` is configured, falls back to
Alpha Vantage's ``EARNINGS`` endpoint (``tools/alpha_vantage.py``) for EPS
only - Alpha Vantage has no quarterly cash flow equivalent wired up, so
free cash flow stays ``None`` in that case.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from research_agent.agent_framework import tool
from research_agent.config import logger
from research_agent.tools import alpha_vantage
from research_agent.tools.cache import ONE_DAY, cached
from research_agent.tools.yahoo_finance import (
    get_financial_statements,
    get_ticker,
    get_ticker_details,
)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_reported_eps(ticker: str) -> dict[str, Any] | None:
    """Most recent quarter with an actual (not just estimated) EPS, or ``None``."""
    try:
        dates = get_ticker(ticker).get_earnings_dates(limit=12)
    except Exception as exc:  # noqa: BLE001 - yfinance raises assorted errors; treat as no data
        logger.info("yfinance: no earnings dates for %s: %s", ticker, exc)
        return None
    if dates is None or dates.empty or "Reported EPS" not in dates.columns:
        return None

    reported = dates.dropna(subset=["Reported EPS"])
    if reported.empty:
        return None

    row = reported.iloc[0]
    eps_actual = _safe_float(row.get("Reported EPS"))
    eps_estimate = _safe_float(row.get("EPS Estimate"))
    eps_surprise_pct = (
        (eps_actual - eps_estimate) / abs(eps_estimate) * 100
        if eps_actual is not None and eps_estimate not in (None, 0)
        else None
    )
    return {
        "quarter": str(reported.index[0].date()),
        "eps_actual": eps_actual,
        "eps_estimate": eps_estimate,
        "eps_surprise_pct": eps_surprise_pct,
    }


def _free_cash_flow(ticker: str) -> float | None:
    """Latest quarterly operating cash flow minus capital expenditure, or ``None``."""
    try:
        cashflow = get_ticker(ticker).quarterly_cashflow
    except Exception as exc:  # noqa: BLE001 - yfinance raises assorted errors; treat as no data
        logger.info("yfinance: no quarterly cash flow for %s: %s", ticker, exc)
        return None
    if cashflow is None or cashflow.empty:
        return None

    column = cashflow.iloc[:, 0]
    ocf = _safe_float(
        column.get("Operating Cash Flow")
        if "Operating Cash Flow" in column.index
        else column.get("Total Cash From Operating Activities")
    )
    capex = _safe_float(column.get("Capital Expenditure"))
    if ocf is None or capex is None:
        return None
    return ocf - abs(capex)


@tool(
    name="get_earnings_data",
    description=(
        "Retrieves the most recently reported quarter's revenue, gross margin, operating "
        "and net income, free cash flow, forward P/E, and EPS actual/estimate/surprise for "
        "a stock ticker, live from Yahoo Finance (falling back to Alpha Vantage for EPS if "
        "needed)."
    ),
)
@cached("get_earnings_data", ttl_seconds=ONE_DAY)
def get_earnings_data(ticker: str) -> dict[str, Any]:
    """Returns the latest quarterly earnings figures for a given ticker symbol.

    Args:
        ticker: Stock symbol (e.g. 'NVDA', 'AAPL').

    Returns:
        Latest quarter's revenue, gross margin %, operating/net income, free
        cash flow, forward P/E, and EPS actual/estimate/surprise %, plus a
        ``source`` field naming which provider answered the EPS figures.

    Raises:
        ValueError: If no revenue or EPS data can be found for this ticker
            from either provider.
    """
    clean = (ticker or "").strip().upper()
    if not clean:
        raise ValueError("ticker must not be empty")
    logger.info("Executing get_earnings_data for ticker: %s", clean)

    quarter = revenue = gross_margin_pct = operating_income = net_income = None
    try:
        statements = get_financial_statements.func(clean)
        quarterly = statements.get("quarterly_income_statement") or []
        if quarterly:
            latest = quarterly[0]
            quarter = latest.get("period")
            revenue = latest.get("Total Revenue")
            gross_profit = latest.get("Gross Profit")
            operating_income = latest.get("Operating Income")
            net_income = latest.get("Net Income")
            if revenue and gross_profit is not None:
                gross_margin_pct = round(gross_profit / revenue * 100, 2)
    except ValueError as exc:
        logger.info("No quarterly income statement for %s: %s", clean, exc)

    forward_pe = None
    try:
        forward_pe = get_ticker_details.func(clean).get("forward_pe")
    except ValueError as exc:
        logger.info("No ticker details for %s: %s", clean, exc)

    free_cash_flow = _free_cash_flow(clean)

    eps = _latest_reported_eps(clean)
    source = "yfinance"
    if eps is None and alpha_vantage.is_configured():
        try:
            eps = alpha_vantage.get_earnings(clean)
        except ValueError as exc:
            logger.warning("Alpha Vantage EPS fallback failed for %s: %s", clean, exc)
        else:
            source = "alpha_vantage"

    if revenue is None and eps is None:
        raise ValueError(f"No earnings data available for '{clean}'")

    eps = eps or {}
    eps_surprise_pct = eps.get("eps_surprise_pct")
    return {
        "ticker": clean,
        "quarter": quarter or eps.get("quarter"),
        "revenue": revenue,
        "gross_margin_pct": gross_margin_pct,
        "operating_income": operating_income,
        "net_income": net_income,
        "free_cash_flow": free_cash_flow,
        "forward_pe": forward_pe,
        "eps_actual": eps.get("eps_actual"),
        "eps_estimate": eps.get("eps_estimate"),
        "eps_surprise_pct": round(eps_surprise_pct, 2) if eps_surprise_pct is not None else None,
        "source": source,
    }
