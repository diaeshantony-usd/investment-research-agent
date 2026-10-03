"""
agents/market_analyst.py
========================
Market Analyst Agent: Gathers price action, technical indicators, and momentum signals.
"""

from __future__ import annotations

from typing import Any

from research_agent.agent_framework import Agent
from research_agent.tools.macro import get_macro_indicators
from research_agent.tools.market import get_market_technicals

MARKET_ANALYST_ROLE = (
    "You are the Chartered Market Technician (CMT) and Macro Context Specialist. "
    "Your mission is to evaluate price action, trend momentum, and macroeconomic backdrop using "
    "get_market_technicals and get_macro_indicators.\n\n"
    "Analysis Framework:\n"
    "1. Trend Structure: Evaluate current price relative to 50-day and 200-day simple moving averages (SMA) to confirm trend direction.\n"
    "2. Momentum Oscillators: Interpret 14-day RSI to identify overbought (>70) or oversold (<30) conditions and divergence.\n"
    "3. Volatility & Risk: Review stock beta and trading range (52-week high/low).\n"
    "4. Macroeconomic Environment: Incorporate prevailing interest rate regimes (Fed funds), 10-year Treasury yields, and inflation trends."
)


def market_analyst_agent(
    model: str | Any = None,
    llm: Any = None,
    verbose: bool = False,
) -> Agent:
    """Creates the MarketAnalyst agent equipped with market technicals and macro tools."""
    return Agent(
        name="MarketAnalyst",
        role=MARKET_ANALYST_ROLE,
        tools=[get_market_technicals, get_macro_indicators],
        model=model,
        llm=llm,
        verbose=verbose,
    )
