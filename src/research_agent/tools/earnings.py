"""
tools/earnings.py
=================
Tool for fetching quarterly earnings figures, revenue, and fundamentals.

Composes the live Yahoo Finance tools in ``tools/yahoo_finance.py`` rather
than keeping its own fixed ticker list: quarterly revenue/margins/income
and the latest annual free cash flow come from ``get_financial_statements``,
forward P/E and market cap from ``get_ticker_details``. Both reused tools
already fall back to Alpha Vantage internally when yfinance has nothing
for a ticker.

EPS history comes from ``Ticker.earnings_history``; when that's empty and
``ALPHA_VANTAGE_KEY`` is configured, ``_get_eps_history`` falls back to
Alpha Vantage's ``EARNINGS`` endpoint (``tools/alpha_vantage.py``).

Every external call degrades gracefully: a failed/missing source is
listed by field name in the response's ``unavailable`` list rather than
raising, except a total failure to find any quarterly statements, which
is reported as a single ``error`` field with no other figures - there is
nothing derivable without at least one quarter.
"""

from __future__ import annotations

from datetime import datetime, timedelta
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

# A quarter is treated as "about one year earlier" within this many days of
# an exact 365-day gap, to tolerate real-world fiscal-calendar drift.
_YOY_TOLERANCE_DAYS = 45


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


def _format_money(value: float | None) -> str | None:
    """Scale a raw dollar amount to the nearest T/B/M/K suffix, e.g. ``"$46.00B"``."""
    if value is None:
        return None
    magnitude = abs(value)
    if magnitude >= 1e12:
        scaled, suffix = value / 1e12, "T"
    elif magnitude >= 1e9:
        scaled, suffix = value / 1e9, "B"
    elif magnitude >= 1e6:
        scaled, suffix = value / 1e6, "M"
    elif magnitude >= 1e3:
        scaled, suffix = value / 1e3, "K"
    else:
        scaled, suffix = value, ""
    return f"${scaled:.2f}{suffix}"


def _growth_pct(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return round((current - previous) / previous * 100, 2)


def _parse_period(period: Any) -> datetime | None:
    if not period:
        return None
    try:
        return datetime.strptime(str(period)[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _sorted_quarters(quarters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Quarterly records newest-first, regardless of the order a provider returned them in."""
    return sorted(
        quarters, key=lambda q: _parse_period(q.get("period")) or datetime.min, reverse=True
    )


def _select_latest_quarter(sorted_quarters: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The newest quarter that actually has figures.

    Some providers list a not-yet-reported quarter before the one with real
    numbers (seen live for AMZN); skip any quarter with no revenue.
    """
    for quarter in sorted_quarters:
        if quarter.get("Total Revenue") is not None:
            return quarter
    return None


def _select_prior_quarter(
    sorted_quarters: list[dict[str, Any]], latest: dict[str, Any]
) -> dict[str, Any] | None:
    """The quarter immediately before ``latest``, for quarter-over-quarter growth."""
    latest_date = _parse_period(latest.get("period"))
    if latest_date is None:
        return None
    for quarter in sorted_quarters:
        quarter_date = _parse_period(quarter.get("period"))
        if quarter_date is not None and quarter_date < latest_date:
            return quarter
    return None


def _select_yoy_quarter(
    sorted_quarters: list[dict[str, Any]], latest: dict[str, Any]
) -> dict[str, Any] | None:
    """The quarter closest to one year before ``latest``, within ``_YOY_TOLERANCE_DAYS``."""
    latest_date = _parse_period(latest.get("period"))
    if latest_date is None:
        return None
    target = latest_date - timedelta(days=365)

    best, best_diff = None, None
    for quarter in sorted_quarters:
        if quarter is latest:
            continue
        quarter_date = _parse_period(quarter.get("period"))
        if quarter_date is None:
            continue
        diff = abs((quarter_date - target).days)
        if diff <= _YOY_TOLERANCE_DAYS and (best_diff is None or diff < best_diff):
            best, best_diff = quarter, diff
    return best


def _free_cash_flow_annual(statements: dict[str, Any]) -> tuple[float | None, str | None]:
    """Latest annual free cash flow (and its fiscal year end), or ``(None, None)``.

    Prefers yfinance's own ``"Free Cash Flow"`` line item; falls back to
    Alpha Vantage's shape (operating cash flow minus capital expenditure)
    when that line item isn't present.
    """
    records = statements.get("annual_cash_flow") or []
    if not records:
        return None, None

    latest = max(records, key=lambda r: _parse_period(r.get("period")) or datetime.min)
    fiscal_year_end = latest.get("period")

    free_cash_flow = latest.get("Free Cash Flow")
    if free_cash_flow is not None:
        return free_cash_flow, fiscal_year_end

    operating_cash_flow = latest.get("operatingCashflow")
    capital_expenditure = latest.get("capitalExpenditures")
    if operating_cash_flow is not None and capital_expenditure is not None:
        return operating_cash_flow - abs(capital_expenditure), fiscal_year_end
    return None, fiscal_year_end


def _eps_history_yfinance(ticker: str) -> list[dict[str, Any]]:
    """Quarterly EPS actual/estimate/surprise, newest first, via ``Ticker.earnings_history``.

    Raises:
        ValueError: If yfinance has no earnings history for this ticker.
    """
    frame = getattr(get_ticker(ticker), "earnings_history", None)
    if frame is None or frame.empty:
        raise ValueError(f"No EPS history found for '{ticker}'")

    history = []
    for period, row in frame.sort_index(ascending=False).iterrows():
        surprise = _safe_float(row.get("surprisePercent"))
        history.append(
            {
                "period": str(period.date()) if hasattr(period, "date") else str(period),
                "eps_actual": _safe_float(row.get("epsActual")),
                "eps_estimate": _safe_float(row.get("epsEstimate")),
                "surprise_pct": round(surprise * 100, 2) if surprise is not None else None,
            }
        )
    return history


@cached("get_eps_history", ttl_seconds=ONE_DAY)
def _get_eps_history(ticker: str) -> list[dict[str, Any]]:
    """EPS history via yfinance, falling back to Alpha Vantage if that's empty.

    Raises:
        ValueError: If neither provider has EPS history for this ticker.
    """
    try:
        return _eps_history_yfinance(ticker)
    except Exception as exc:
        logger.info("yfinance EPS history failed for %s: %s", ticker, exc)
        if not alpha_vantage.is_configured():
            raise
        return alpha_vantage.get_earnings_history(ticker)


@tool(
    name="get_earnings_data",
    description=(
        "Retrieves the most recently reported quarter's revenue, margins, YoY/QoQ revenue "
        "growth, annual free cash flow, forward P/E, market cap and EPS history for a stock "
        "ticker, live from Yahoo Finance (falling back to Alpha Vantage where needed)."
    ),
)
def get_earnings_data(ticker: str) -> dict[str, Any]:
    """Returns the latest quarterly earnings figures and trends for a given ticker symbol.

    Args:
        ticker: Stock symbol (e.g. 'NVDA', 'AAPL').

    Returns:
        Latest quarter's revenue, margins, YoY/QoQ revenue growth, annual
        free cash flow, forward P/E, market cap, EPS actual/surprise and
        history, plus an ``unavailable`` list naming any field that
        couldn't be determined. A ticker with no quarterly statements at
        all returns only ``ticker`` and ``error``.
    """
    clean = (ticker or "").strip().upper()
    if not clean:
        return {"ticker": clean, "error": "ticker must not be empty"}
    logger.info("Executing get_earnings_data for ticker: %s", clean)

    try:
        statements = get_financial_statements.func(clean)
    except Exception as exc:  # noqa: BLE001 - any statements-provider failure is a clean miss
        logger.warning("No financial statements for %s: %s", clean, exc)
        return {"ticker": clean, "error": f"No earnings data available for '{clean}': {exc}"}

    quarters = _sorted_quarters(statements.get("quarterly_income_statement") or [])
    latest = _select_latest_quarter(quarters)
    if latest is None:
        return {"ticker": clean, "error": f"No quarterly figures available for '{clean}'"}

    revenue = latest.get("Total Revenue")
    gross_profit = latest.get("Gross Profit")
    operating_income = latest.get("Operating Income")
    net_income = latest.get("Net Income")

    unavailable: list[str] = []

    prior_qoq = _select_prior_quarter(quarters, latest)
    prior_yoy = _select_yoy_quarter(quarters, latest)
    qoq_revenue = prior_qoq.get("Total Revenue") if prior_qoq else None
    yoy_revenue = prior_yoy.get("Total Revenue") if prior_yoy else None
    revenue_growth_qoq_pct = _growth_pct(revenue, qoq_revenue)
    revenue_growth_yoy_pct = _growth_pct(revenue, yoy_revenue)
    if revenue_growth_yoy_pct is None:
        unavailable.append("revenue_growth_yoy_pct")

    free_cash_flow, fiscal_year_end = _free_cash_flow_annual(statements)

    forward_pe = market_cap = None
    try:
        details = get_ticker_details.func(clean)
    except Exception as exc:  # noqa: BLE001 - ticker-details failure degrades gracefully
        logger.info("No ticker details for %s: %s", clean, exc)
        unavailable.append("forward_pe")
    else:
        forward_pe = details.get("forward_pe")
        market_cap = details.get("market_cap")

    eps_actual = eps_quarter = eps_surprise_pct = None
    eps_history: list[dict[str, Any]] = []
    try:
        eps_history = _get_eps_history(clean)
    except Exception as exc:  # noqa: BLE001 - EPS-history failure degrades gracefully
        logger.info("No EPS history for %s: %s", clean, exc)
        unavailable.extend(["eps_surprise_pct", "eps_history"])
    else:
        if eps_history:
            latest_eps = eps_history[0]
            eps_actual = latest_eps.get("eps_actual")
            eps_quarter = latest_eps.get("period")
            eps_surprise_pct = latest_eps.get("surprise_pct")
        else:
            unavailable.extend(["eps_surprise_pct", "eps_history"])

    return {
        "ticker": clean,
        "quarter_end": latest.get("period"),
        "revenue": _format_money(revenue),
        "gross_margin_pct": (
            round(gross_profit / revenue * 100, 2) if revenue and gross_profit is not None else None
        ),
        "operating_margin_pct": (
            round(operating_income / revenue * 100, 2)
            if revenue and operating_income is not None
            else None
        ),
        "net_margin_pct": (
            round(net_income / revenue * 100, 2) if revenue and net_income is not None else None
        ),
        "revenue_growth_yoy_pct": revenue_growth_yoy_pct,
        "revenue_growth_qoq_pct": revenue_growth_qoq_pct,
        "free_cash_flow_annual": _format_money(free_cash_flow),
        "fiscal_year_end": fiscal_year_end,
        "forward_pe": forward_pe,
        "market_cap": _format_money(market_cap),
        "eps_actual": eps_actual,
        "eps_quarter": eps_quarter,
        "eps_surprise_pct": eps_surprise_pct,
        "eps_history": eps_history,
        "unavailable": unavailable,
    }
