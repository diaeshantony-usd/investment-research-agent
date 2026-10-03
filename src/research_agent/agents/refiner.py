"""
agents/refiner.py
=================
Refiner Agent: Polishes and corrects research briefs based on Critic feedback.
"""

from __future__ import annotations

from typing import Any

from research_agent.agent_framework import Agent
from research_agent.tools.earnings import get_earnings_data
from research_agent.tools.filings import search_sec_filings
from research_agent.tools.macro import get_macro_indicators
from research_agent.tools.market import get_market_technicals
from research_agent.tools.news import get_news_data

REFINER_ROLE = (
    "You are the Senior Copy Editor and Lead Revision Specialist. "
    "Your duty is to polish and refine draft research briefs that have undergone audit by the Critic.\n\n"
    "Refinement Directives:\n"
    "1. Target Flagged Issues: Address each deficiency identified in the Critic audit report point-by-point.\n"
    "2. Metric Precision: Verify all numbers against original specialist observations or use tools to fetch missing data.\n"
    "3. Structural Enhancement: Enhance document headers, bullet formatting, and clarity without altering verified facts.\n"
    "4. Maintain Balance: Ensure bull and bear sections maintain analytical equilibrium.\n"
    "5. Output directly the complete, final polished Markdown brief. Do NOT include any change summaries, notes, or meta-commentary."
)


def refiner_agent(
    model: str | Any = None,
    llm: Any = None,
    tools: list[Any] | None = None,
    verbose: bool = False,
) -> Agent:
    """Creates the Refiner agent equipped with tools to fetch and verify missing data."""
    default_tools = [
        get_earnings_data,
        get_market_technicals,
        get_news_data,
        search_sec_filings,
        get_macro_indicators,
    ]
    return Agent(
        name="BriefRefiner",
        role=REFINER_ROLE,
        tools=tools or default_tools,
        model=model,
        llm=llm,
        verbose=verbose,
    )
