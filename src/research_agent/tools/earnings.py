"""
tools/earnings.py
=================
Tool for fetching quarterly earnings figures, revenue, and fundamentals.
"""

from __future__ import annotations

import json
from typing import Any

from research_agent.agent_framework import tool
from research_agent.config import logger

# Local database / cache for resilient offline demonstration
_EARNINGS_DATABASE: dict[str, dict[str, Any]] = {
    "NVDA": {
        "quarter": "Q2 2026",
        "revenue": "$38.2 Billion",
        "gross_margin": "75.4%",
        "operating_income": "$22.8 Billion",
        "net_income": "$19.3 Billion",
        "forward_pe": "34.2",
        "free_cash_flow": "$14.5 Billion",
        "eps_actual": "$0.68",
        "eps_estimate": "$0.64",
        "eps_surprise": "+6.25%",
    },
    "AAPL": {
        "quarter": "Q3 2026",
        "revenue": "$85.8 Billion",
        "gross_margin": "46.3%",
        "operating_income": "$25.4 Billion",
        "net_income": "$21.4 Billion",
        "forward_pe": "29.8",
        "free_cash_flow": "$28.1 Billion",
        "eps_actual": "$1.40",
        "eps_estimate": "$1.35",
        "eps_surprise": "+3.70%",
    },
    "MSFT": {
        "quarter": "Q4 2026",
        "revenue": "$64.7 Billion",
        "gross_margin": "69.8%",
        "operating_income": "$27.9 Billion",
        "net_income": "$22.0 Billion",
        "forward_pe": "31.5",
        "free_cash_flow": "$23.3 Billion",
        "eps_actual": "$2.95",
        "eps_estimate": "$2.93",
        "eps_surprise": "+0.68%",
    },
}


@tool(
    name="get_earnings_data",
    description="Retrieves quarterly earnings figures, revenue, margins, and cash flow for a stock ticker.",
)
def get_earnings_data(ticker: str) -> str:
    """Returns quarterly earnings figures for a given ticker symbol.

    Args:
        ticker: Stock symbol (e.g. 'NVDA', 'AAPL').
    """
    clean_ticker = (ticker or "").strip().upper()
    logger.info(f"Executing get_earnings_data for ticker: {clean_ticker}")

    data = _EARNINGS_DATABASE.get(clean_ticker)
    if data:
        payload = {"ticker": clean_ticker, **data}
        return json.dumps(payload, indent=2)

    logger.warning(f"No earnings data found for ticker: {clean_ticker}")
    return json.dumps(
        {
            "ticker": clean_ticker,
            "error": f"No earnings data available for '{clean_ticker}'",
        }
    )
