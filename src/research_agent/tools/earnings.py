"""
tools/earnings.py
=================
``get_earnings_data``: a one-call earnings snapshot for the Earnings analyst, built from
live data instead of a fixed ticker list.

It combines three sources that already exist in the tools package:

* ``get_financial_statements`` (``tools/yahoo_finance.py``) - quarterly income
  statement and annual cash flow, with an Alpha Vantage fallback.
* ``get_ticker_details`` (``tools/yahoo_finance.py``) - P/E ratios and market cap.
* Reported vs. estimated EPS for recent quarters (yfinance, falling back to Alpha
  Vantage's EARNINGS endpoint) - for the EPS surprise.

From these it derives the figures analysts quote first: revenue and its growth, gross /
operating / net margins, EPS and the EPS surprise, free cash flow and valuation. Every
number in the output comes straight from a data provider or from a documented formula
on provider data (e.g. ``gross_margin_pct = gross_profit / revenue``), so the Critic can
trace it. All provider calls are cached in ``data/cache/`` (see ``tools/cache.py``).

Missing inputs are reported as ``null`` and listed in ``unavailable`` rather than
guessed. If no statements can be retrieved at all, the tool returns an ``error`` field.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

from research_agent.agent_framework import tool
from research_agent.config import logger
from research_agent.tools import alpha_vantage
from research_agent.tools.cache import SIX_HOURS, cached
from research_agent.tools.yahoo_finance import (
    get_financial_statements,
    get_ticker,
    get_ticker_details,
)

__all__ = ["get_earnings_data"]

# Statement line items have different names in yfinance and Alpha Vantage; each metric
# lists every alias we accept, in priority order.
_REVENUE = ("Total Revenue", "Operating Revenue", "totalRevenue")
_GROSS_PROFIT = ("Gross Profit", "grossProfit")
_OPERATING_INCOME = ("Operating Income", "operatingIncome")
_NET_INCOME = ("Net Income", "Net Income Common Stockholders", "netIncome")
_DILUTED_EPS = ("Diluted EPS", "Basic EPS")
_FREE_CASH_FLOW = ("Free Cash Flow",)
_OPERATING_CASH_FLOW = ("Operating Cash Flow", "operatingCashflow")
_CAPEX = ("Capital Expenditure", "capitalExpenditures")

# A "same quarter last year" period ends 330-400 days before the latest quarter.
_YEAR_AGO_DAYS = (330, 400)
# How many recent EPS reports to include, so the analyst can see a trend.
_EPS_HISTORY_QUARTERS = 4


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _pick(record: Mapping[str, Any] | None, aliases: Iterable[str]) -> float | None:
    """Return the first non-null numeric value among ``aliases`` in ``record``.

    ``record`` may be a dict or a pandas row, so it is compared with ``None`` explicitly
    (a pandas Series has no single truth value).
    """
    if record is None:
        return None
    for name in aliases:
        value = record.get(name)
        if isinstance(value, (int, float)) and not math.isnan(value):
            return float(value)
    return None


def _ratio_pct(numerator: float | None, denominator: float | None) -> float | None:
    """``numerator / denominator`` as a percentage rounded to 2 dp, or None."""
    if numerator is None or not denominator:
        return None
    return round(numerator / denominator * 100, 2)


def _growth_pct(current: float | None, previous: float | None) -> float | None:
    """Percentage change from ``previous`` to ``current``, or None if not computable."""
    if current is None or not previous:
        return None
    return round((current - previous) / abs(previous) * 100, 2)


def _money(value: float | None) -> str | None:
    """Format a dollar amount as ``$38.20B`` / ``$512.40M`` for readable briefs."""
    if value is None:
        return None
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if magnitude >= threshold:
            return f"{sign}${magnitude / threshold:.2f}{suffix}"
    return f"{sign}${magnitude:,.2f}"


def _sorted_by_period(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return statement records newest first (providers differ in their ordering)."""
    return sorted(records, key=lambda r: str(r.get("period") or ""), reverse=True)


# ---------------------------------------------------------------------------
# EPS surprise
# ---------------------------------------------------------------------------


def _eps_history_yfinance(ticker: str) -> list[dict[str, Any]]:
    """Recent quarters' reported vs. estimated EPS from yfinance's earnings history."""
    frame = get_ticker(ticker).earnings_history
    if frame is None or frame.empty:
        raise ValueError(f"yfinance has no earnings history for '{ticker}'")

    quarters = []
    for period, row in frame.iterrows():
        surprise = row.get("surprisePercent")
        quarters.append(
            {
                "period": str(period.date()) if hasattr(period, "date") else str(period),
                "eps_actual": _pick(row, ("epsActual",)),
                "eps_estimate": _pick(row, ("epsEstimate",)),
                # yfinance reports the surprise as a fraction (0.0625 = 6.25%).
                "surprise_pct": (
                    round(float(surprise) * 100, 2)
                    if isinstance(surprise, (int, float)) and not math.isnan(surprise)
                    else None
                ),
            }
        )
    return _sorted_by_period(quarters)


@cached("eps_history", ttl_seconds=SIX_HOURS)
def _get_eps_history(ticker: str) -> list[dict[str, Any]]:
    """EPS reported vs. estimate, newest first; yfinance with an Alpha Vantage fallback.

    Raises:
        ValueError: If neither provider has earnings history for the ticker.
    """
    try:
        return _eps_history_yfinance(ticker)
    except Exception as yf_exc:
        if not alpha_vantage.is_configured():
            raise ValueError(f"No EPS history for '{ticker}': {yf_exc}") from yf_exc
        logger.warning(
            "yfinance EPS history failed for %s (%s); trying Alpha Vantage", ticker, yf_exc
        )
        return alpha_vantage.get_earnings_history(ticker)


# ---------------------------------------------------------------------------
# Snapshot assembly
# ---------------------------------------------------------------------------


def _period_date(record: Mapping[str, Any]) -> date | None:
    """Parse a statement record's ``period`` (``YYYY-MM-DD``) as a date, if possible."""
    try:
        return date.fromisoformat(str(record.get("period"))[:10])
    except ValueError:
        return None


def _year_ago_quarter(
    latest: Mapping[str, Any], quarters: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Find the quarter that ended roughly one year before ``latest``, matched by date."""
    latest_date = _period_date(latest)
    if latest_date is None:
        return None
    low, high = _YEAR_AGO_DAYS
    for quarter in quarters:
        quarter_date = _period_date(quarter)
        if quarter_date and low <= (latest_date - quarter_date).days <= high:
            return quarter
    return None


def _income_metrics(quarters: list[dict[str, Any]]) -> dict[str, Any]:
    """Latest-quarter income figures, margins and growth from quarterly statements.

    Providers sometimes list the newest quarter before its figures are filled in, so
    only quarters that report revenue are considered.
    """
    quarters = [q for q in quarters if _pick(q, _REVENUE) is not None]
    latest = quarters[0] if quarters else None
    prior_quarter = quarters[1] if len(quarters) > 1 else None
    year_ago = _year_ago_quarter(latest, quarters[1:]) if latest else None

    revenue = _pick(latest, _REVENUE)
    gross_profit = _pick(latest, _GROSS_PROFIT)
    operating_income = _pick(latest, _OPERATING_INCOME)
    net_income = _pick(latest, _NET_INCOME)

    return {
        "quarter_end": latest.get("period") if latest else None,
        "revenue": _money(revenue),
        "revenue_growth_yoy_pct": _growth_pct(revenue, _pick(year_ago, _REVENUE)),
        "revenue_growth_qoq_pct": _growth_pct(revenue, _pick(prior_quarter, _REVENUE)),
        "gross_profit": _money(gross_profit),
        "gross_margin_pct": _ratio_pct(gross_profit, revenue),
        "operating_income": _money(operating_income),
        "operating_margin_pct": _ratio_pct(operating_income, revenue),
        "net_income": _money(net_income),
        "net_margin_pct": _ratio_pct(net_income, revenue),
        "diluted_eps": _pick(latest, _DILUTED_EPS),
    }


def _free_cash_flow(annual_cash_flow: list[dict[str, Any]]) -> dict[str, Any]:
    """Latest fiscal year's free cash flow (reported, or operating cash flow - capex)."""
    latest = annual_cash_flow[0] if annual_cash_flow else None
    fcf = _pick(latest, _FREE_CASH_FLOW)
    if fcf is None:
        operating = _pick(latest, _OPERATING_CASH_FLOW)
        capex = _pick(latest, _CAPEX)
        if operating is not None and capex is not None:
            # yfinance reports capex as a negative outflow, Alpha Vantage as positive.
            fcf = operating - abs(capex)
    return {
        "free_cash_flow_annual": _money(fcf),
        "fiscal_year_end": latest.get("period") if latest else None,
    }


@tool(
    name="get_earnings_data",
    description=(
        "Retrieves the latest quarterly earnings for a stock ticker from live data: revenue "
        "and its year-over-year / quarter-over-quarter growth, gross / operating / net "
        "margins, diluted EPS, EPS vs. analyst estimate (surprise) for recent quarters, "
        "annual free cash flow, P/E ratios and market cap."
    ),
)
def get_earnings_data(ticker: str) -> str:
    """Return a live earnings snapshot for ``ticker`` as JSON.

    Args:
        ticker: Stock symbol (e.g. 'NVDA', 'AAPL').

    Returns:
        JSON with latest-quarter income metrics, margins, growth, EPS surprise history,
        free cash flow and valuation, plus ``sources`` and an ``unavailable`` list of
        fields that could not be retrieved. Returns an ``error`` field when no financial
        statements exist for the ticker.
    """
    clean_ticker = (ticker or "").strip().upper()
    logger.info("Executing get_earnings_data for ticker: %s", clean_ticker)

    # Financial statements are required; without them there is no earnings snapshot.
    try:
        statements = get_financial_statements.func(clean_ticker)
    except ValueError as exc:
        logger.warning("No earnings data for %s: %s", clean_ticker, exc)
        return json.dumps(
            {
                "ticker": clean_ticker,
                "error": f"No earnings data available for '{clean_ticker}': {exc}",
            }
        )

    quarters = _sorted_by_period(statements.get("quarterly_income_statement") or [])
    annual_cash_flow = _sorted_by_period(statements.get("annual_cash_flow") or [])
    sources = {"statements": statements.get("source")}

    payload: dict[str, Any] = {"ticker": clean_ticker}
    payload.update(_income_metrics(quarters))
    payload.update(_free_cash_flow(annual_cash_flow))

    # Valuation and EPS surprise are optional extras: a failure here should not hide
    # the statement figures that were retrieved successfully.
    try:
        details = get_ticker_details.func(clean_ticker)
    except ValueError as exc:
        logger.warning("Valuation data unavailable for %s: %s", clean_ticker, exc)
        details = {}
    payload["trailing_pe"] = details.get("trailing_pe")
    payload["forward_pe"] = details.get("forward_pe")
    payload["market_cap"] = _money(details.get("market_cap"))
    sources["valuation"] = details.get("source")

    try:
        eps_history = _get_eps_history(clean_ticker)[:_EPS_HISTORY_QUARTERS]
    except ValueError as exc:
        logger.warning("EPS history unavailable for %s: %s", clean_ticker, exc)
        eps_history = []
    latest_eps = eps_history[0] if eps_history else {}
    # EPS can be reported before the statements for that quarter are published, so the
    # EPS quarter is stated explicitly instead of being assumed to equal quarter_end.
    payload["eps_quarter"] = latest_eps.get("period")
    payload["eps_actual"] = latest_eps.get("eps_actual")
    payload["eps_estimate"] = latest_eps.get("eps_estimate")
    payload["eps_surprise_pct"] = latest_eps.get("surprise_pct")
    payload["eps_history"] = eps_history

    payload["sources"] = sources
    # Tell the analyst explicitly which figures are missing so it does not fill gaps.
    payload["unavailable"] = sorted(
        key for key, value in payload.items() if value is None or value == []
    )
    return json.dumps(payload, indent=2, default=str)
