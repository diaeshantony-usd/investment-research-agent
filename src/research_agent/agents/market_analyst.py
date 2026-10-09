"""
agents/market_analyst.py
========================
Market Analyst Agent: Gathers price action, technical indicators, and momentum signals.
"""

from __future__ import annotations

from research_agent.agent_framework import Agent
from research_agent.interfaces.llm import LLM
from research_agent.tools.macro import get_macro_indicators
from research_agent.tools.market import get_market_technicals

MARKET_ANALYST_ROLE = (
    "You are the Chartered Market Technician (CMT) and Macro Context Specialist.\n"
    "Your mission is to evaluate price action, trend momentum, and macroeconomic backdrop using "
    "get_market_technicals and get_macro_indicators.\n\n"
    "Analysis Framework:\n"
    "1. Trend Structure: Evaluate current price relative to 50-day and 200-day simple moving "
    "averages (SMA) to confirm trend direction.\n"
    "2. Momentum Oscillators: Interpret 14-day RSI to identify overbought (>70) or oversold (<30) "
    "conditions and divergence.\n"
    "3. Volatility & Risk: Review stock beta and trading range (52-week high/low).\n"
    "4. Macroeconomic Environment: Incorporate prevailing interest rate regimes (Fed funds), "
    "10-year Treasury yields, and inflation trends.\n\n"
    "MANDATORY ANTI-HALLUCINATION & TOOL GROUNDING RULES:\n"
    "- You MUST execute `get_market_technicals` and/or `get_macro_indicators` to retrieve data.\n"
    "- Cite ONLY technical levels, moving averages, and macro metrics returned by your tools.\n"
    "- If market data is not found or returns an error, state clearly: "
    "'No market technicals available from tools for [TICKER].' Do NOT invent prices or indicators."
)


def market_analyst_agent(
    model: str | LLM | None = None,
    llm: LLM | None = None,
    verbose: bool = False,
) -> Agent:
    """Create the MarketAnalyst agent (market technicals and macro tools).

    Args:
        model: Model name, or an ``LLM`` client to use directly.
        llm: LLM client shared with the rest of the graph.
        verbose: Log every thought, tool call and observation.

    Returns:
        A configured ``Agent`` named ``MarketAnalyst``.
    """
    return Agent(
        name="MarketAnalyst",
        role=MARKET_ANALYST_ROLE,
        tools=[get_market_technicals, get_macro_indicators],
        model=model,
        llm=llm,
        verbose=verbose,
    )
