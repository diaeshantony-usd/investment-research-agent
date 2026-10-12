"""
tools package
=============
Exports all domain data collection tools for specialist agents.
"""

from research_agent.tools.earnings import get_earnings_data
from research_agent.tools.filings import search_sec_filings
from research_agent.tools.macro import get_macro_indicators
from research_agent.tools.market import get_market_technicals
from research_agent.tools.news import get_news_data
from research_agent.tools.yahoo_finance import lookup_ticker

ALL_TOOLS = [
    get_earnings_data,
    get_market_technicals,
    get_news_data,
    get_macro_indicators,
    search_sec_filings,
    lookup_ticker,
]

__all__ = [
    "ALL_TOOLS",
    "get_earnings_data",
    "get_macro_indicators",
    "get_market_technicals",
    "get_news_data",
    "lookup_ticker",
    "search_sec_filings",
]
