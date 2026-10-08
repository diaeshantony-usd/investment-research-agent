"""
tools/market.py
===============
Tool for retrieving market technicals, price action, moving averages, and indicators.
"""

from __future__ import annotations

import json
from typing import Any

from research_agent.agent_framework import tool
from research_agent.config import logger
from research_agent.tools.yahoo_finance import get_ticker_details

_MARKET_DATABASE: dict[str, dict[str, Any]] = {
    "NVDA": {
        "current_price": "$138.50",
        "52_week_high": "$140.76",
        "52_week_low": "$45.11",
        "50_day_sma": "$126.80",
        "200_day_sma": "$102.40",
        "rsi_14": "61.4",
        "beta": "1.68",
        "trend": "Bullish Uptrend",
    },
    "AAPL": {
        "current_price": "$224.20",
        "52_week_high": "$237.23",
        "52_week_low": "$164.08",
        "50_day_sma": "$218.40",
        "200_day_sma": "$195.10",
        "rsi_14": "54.8",
        "beta": "1.02",
        "trend": "Moderate Uptrend",
    },
    "MSFT": {
        "current_price": "$428.90",
        "52_week_high": "$468.35",
        "52_week_low": "$327.00",
        "50_day_sma": "$435.10",
        "200_day_sma": "$412.50",
        "rsi_14": "48.2",
        "beta": "1.15",
        "trend": "Consolidation",
    },
}


@tool(
    name="get_market_technicals",
    description=(
        "Retrieves technical price action, moving averages (SMA 50/200), RSI, and trend metrics."
    ),
)
def get_market_technicals(ticker: str) -> str:
    """Returns price action and technical indicators for a given stock ticker.

    Args:
        ticker: Stock symbol (e.g. 'NVDA', 'AAPL').

    Returns:
        JSON with price, 52-week range, moving averages, RSI, beta and trend, or
        an ``error`` field when the ticker is not covered.
    """
    clean_ticker = (ticker or "").strip().upper()
    logger.info("Executing get_market_technicals for ticker: %s", clean_ticker)

    data = _MARKET_DATABASE.get(clean_ticker)
    if data:
        payload = {"ticker": clean_ticker, **data}
        return json.dumps(payload, indent=2)

    logger.warning("No market technicals found for ticker: %s", clean_ticker)
    return json.dumps(
        {
            "ticker": clean_ticker,
            "error": f"No technical data available for '{clean_ticker}'",
        }
    )

@tool(
    name="get_stock_price",
    description=(
        "Fetches the live current price, previous close, 52-week range and market cap for "
        "a stock ticker from Yahoo Finance."
    ),
)
def get_stock_price(ticker: str) -> str:
    """Returns the live current price and basic quote details for a ticker.

    Args:
        ticker: Stock symbol (e.g. 'NVDA', 'AAPL').

    Returns:
        JSON with current price, previous close, 52-week range and market
        cap, or an ``error`` field when Yahoo Finance has no data for the
        ticker.
    """
    clean_ticker = (ticker or "").strip().upper()
    logger.info("Executing get_stock_price for ticker: %s", clean_ticker)

    try:
        details = get_ticker_details.func(clean_ticker)
    except ValueError as exc:
        logger.warning("get_stock_price failed for %s: %s", clean_ticker, exc)
        return json.dumps({"ticker": clean_ticker, "error": str(exc)})

    return json.dumps(
        {
            "ticker": details["ticker"],
            "name": details["name"],
            "currency": details["currency"],
            "current_price": details["current_price"],
            "previous_close": details["previous_close"],
            "fifty_two_week_high": details["fifty_two_week_high"],
            "fifty_two_week_low": details["fifty_two_week_low"],
            "market_cap": details["market_cap"],
        },
        indent=2,
    )
