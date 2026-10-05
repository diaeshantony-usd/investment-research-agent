"""
tools/news.py
=============
Tool for retrieving recent news headlines, sentiment, and catalyst events.
"""

from __future__ import annotations

import json
from typing import Any

from research_agent.agent_framework import tool
from research_agent.config import logger

_NEWS_DATABASE: dict[str, list[dict[str, Any]]] = {
    "NVDA": [
        {
            "headline": "NVIDIA Blackwell Architecture Ramp Exceeds Supply Chain Expectations",
            "source": "Bloomberg",
            "date": "2026-09-28",
            "sentiment": "Bullish",
            "summary": "Hyperscalers expand capex allocations for next-gen AI accelerators.",
        },
        {
            "headline": "Export Controls Under Scrutiny Ahead of Global Trade Talks",
            "source": "Reuters",
            "date": "2026-09-24",
            "sentiment": "Neutral/Caution",
            "summary": "Policy makers review advanced semiconductor shipment thresholds.",
        },
        {
            "headline": "Data Center Networking Revenue Expands with Quantum InfiniBand Adoption",
            "source": "Financial Times",
            "date": "2026-09-18",
            "sentiment": "Bullish",
            "summary": "Enterprise clusters adopt full-stack accelerated computing platforms.",
        },
    ],
    "AAPL": [
        {
            "headline": "Apple Intelligence Rollout Drives Solid iPhone Upgrade Cycle",
            "source": "Wall Street Journal",
            "date": "2026-09-26",
            "sentiment": "Bullish",
            "summary": "Early holiday pre-orders show strong mix toward Pro tier devices.",
        },
    ],
}


@tool(
    name="get_news_data",
    description=(
        "Retrieves recent news headlines, publication dates, and catalyst summaries for a stock."
    ),
)
def get_news_data(ticker: str) -> str:
    """Returns curated news headlines and sentiment signals for a given ticker.

    Args:
        ticker: Stock symbol (e.g. 'NVDA', 'AAPL').

    Returns:
        JSON with an ``articles`` list (headline, source, date, sentiment,
        summary); the list is empty when no news is available.
    """
    clean_ticker = (ticker or "").strip().upper()
    logger.info("Executing get_news_data for ticker: %s", clean_ticker)

    articles = _NEWS_DATABASE.get(clean_ticker)
    if articles:
        return json.dumps({"ticker": clean_ticker, "articles": articles}, indent=2)

    logger.warning("No news found for ticker: %s", clean_ticker)
    return json.dumps(
        {
            "ticker": clean_ticker,
            "articles": [],
            "message": f"No recent news found for '{clean_ticker}'",
        }
    )
