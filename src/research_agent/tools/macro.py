"""
tools/macro.py
==============
Tool for retrieving macroeconomic indicators, treasury yields, and inflation
metrics, backed by the FRED (Federal Reserve Economic Data) API via the
``fredapi`` package.

Each indicator is fetched (and cached, see ``tools/cache.py``) independently
per FRED series, so one series being temporarily unavailable doesn't block
the others. Inflation indicators (CPI, core PCE) are reported as a
year-over-year percent change rather than the raw index level, matching how
they're normally quoted.
"""

from __future__ import annotations

import json
from typing import Any

from research_agent.agent_framework import tool
from research_agent.config import FRED_API_KEY, logger
from research_agent.tools.cache import ONE_DAY, cached

# FRED series ID behind each indicator this tool reports.
_SERIES: dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",  # Federal Funds Effective Rate, %
    "us_10y_treasury": "DGS10",  # 10-Year Treasury Constant Maturity Rate, %
    "us_2y_treasury": "DGS2",  # 2-Year Treasury Constant Maturity Rate, %
    "cpi_yoy": "CPIAUCSL",  # CPI for All Urban Consumers (reported as YoY % change)
    "core_pce_yoy": "PCEPILFE",  # Core PCE Price Index (reported as YoY % change)
    "gdp_growth_annualized": "A191RL1Q225SBEA",  # Real GDP, % change, annualized
}

# Series reported as a year-over-year percent change rather than their raw level.
_YOY_SERIES: frozenset[str] = frozenset({"cpi_yoy", "core_pce_yoy"})


def _client() -> Any:
    """Build a ``fredapi.Fred`` client.

    Raises:
        ValueError: If ``FRED_API_KEY`` is not configured.
    """
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY is not set. Add it to .env to enable live macro data.")

    from fredapi import Fred  # local import: optional/heavy dependency

    return Fred(api_key=FRED_API_KEY)


def _latest_value(fred: Any, series_id: str) -> tuple[Any, float]:
    """Return ``(date, value)`` of a series' most recent non-missing observation."""
    series = fred.get_series(series_id).dropna()
    if series.empty:
        raise ValueError(f"FRED returned no observations for series '{series_id}'")
    return series.index[-1], float(series.iloc[-1])


def _latest_yoy_pct_change(fred: Any, series_id: str) -> tuple[Any, float]:
    """Return ``(date, year-over-year % change)`` for a monthly series.

    Compares the latest observation to the one ~12 months earlier in the
    series (13 monthly points back, since the latest point is index -1).
    """
    series = fred.get_series(series_id).dropna()
    if len(series) < 13:
        raise ValueError(f"Not enough history on '{series_id}' to compute a year-over-year change")

    latest_date, latest_value = series.index[-1], float(series.iloc[-1])
    year_ago_value = float(series.iloc[-13])
    if year_ago_value == 0:
        raise ValueError(
            f"Cannot compute a year-over-year change for '{series_id}': base value is 0"
        )

    pct_change = (latest_value - year_ago_value) / year_ago_value * 100
    return latest_date, pct_change


@cached("macro_indicator", ttl_seconds=ONE_DAY)
def _fetch_indicator(category: str) -> dict[str, Any]:
    """Fetch one macro indicator by its internal category name.

    Raises:
        ValueError: If ``category`` is unknown, the key is missing, or the
            FRED request fails.
    """
    series_id = _SERIES.get(category)
    if series_id is None:
        raise ValueError(f"Unknown macro category '{category}'. Known: {sorted(_SERIES)}")

    logger.info("FRED: fetching %s (series=%s)", category, series_id)
    fred = _client()
    try:
        if category in _YOY_SERIES:
            date, value = _latest_yoy_pct_change(fred, series_id)
        else:
            date, value = _latest_value(fred, series_id)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"FRED request failed for '{series_id}': {exc}") from exc

    return {
        "series_id": series_id,
        "date": str(date.date()) if hasattr(date, "date") else str(date),
        "value": round(value, 2),
    }


@tool(
    name="get_macro_indicators",
    description=(
        "Retrieves live US macroeconomic indicators from the FRED API: fed funds rate, "
        "2-year and 10-year treasury yields, CPI and core PCE inflation (year-over-year), "
        "and real GDP growth (annualized). Pass one category name or 'all'."
    ),
)
def get_macro_indicators(category: str = "all") -> str:
    """Returns macroeconomic indicators and monetary policy backdrop from FRED.

    Args:
        category: One of 'fed_funds_rate', 'us_10y_treasury', 'us_2y_treasury',
            'cpi_yoy', 'core_pce_yoy', 'gdp_growth_annualized', or 'all'.
            Unknown values return every indicator, so a loosely worded
            request still gets data.

    Returns:
        JSON keyed by indicator name, each value ``{series_id, date,
        value}``; a top-level ``error`` field if every requested indicator
        failed, or a ``_partial_errors`` field if only some did.
    """
    logger.info("Executing get_macro_indicators [category=%s]", category)
    categories = [category] if category in _SERIES else list(_SERIES)

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for cat in categories:
        try:
            results[cat] = _fetch_indicator(cat)
        except ValueError as exc:  # noqa: PERF203 - each series must fail independently
            errors[cat] = str(exc)

    if not results:
        logger.warning("get_macro_indicators failed for %s: %s", categories, errors)
        return json.dumps({"category": category, "error": f"FRED request(s) failed: {errors}"})

    if errors:
        results["_partial_errors"] = errors

    return json.dumps(results, indent=2)
