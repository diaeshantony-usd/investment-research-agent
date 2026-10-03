"""
tools/filings.py
================
Tool for querying SEC EDGAR filings (10-K and 10-Q) excerpts and risk factors.
"""

from __future__ import annotations

import json
from typing import Any

from research_agent.agent_framework import tool
from research_agent.config import logger

_FILINGS_DATABASE: dict[str, dict[str, Any]] = {
    "NVDA": {
        "form": "10-Q",
        "period_ended": "2026-07-28",
        "risk_factors": "Key risks include customer concentration among top tier CSPs, geopolitical export license policies, and supply chain packaging constraints.",
        "segment_performance": "Compute & Networking revenue totaled $34.1B (+162% YoY), driven by Data Center architecture platforms.",
    },
    "AAPL": {
        "form": "10-Q",
        "period_ended": "2026-06-29",
        "risk_factors": "Risks include regulatory scrutiny of App Store fee structures, intense regional smartphone competition, and global FX volatility.",
        "segment_performance": "Services reached an all-time revenue record of $24.2B with paid subscriptions exceeding 1 billion.",
    },
}


@tool(
    name="search_sec_filings",
    description="Searches SEC EDGAR 10-K and 10-Q filing disclosures, risk factors, and segment performance.",
)
def search_sec_filings(ticker: str, section: str = "all") -> str:
    """Returns curated SEC filing extracts for a company.

    Args:
        ticker: Stock symbol (e.g. 'NVDA', 'AAPL').
        section: Filing section ('risk_factors', 'segment_performance', or 'all').
    """
    clean_ticker = (ticker or "").strip().upper()
    logger.info(f"Executing search_sec_filings for ticker: {clean_ticker} [section={section}]")

    data = _FILINGS_DATABASE.get(clean_ticker)
    if not data:
        return json.dumps({"error": f"No SEC filings found for '{clean_ticker}'"})

    if section in data:
        return json.dumps({section: data[section]}, indent=2)

    return json.dumps(data, indent=2)
