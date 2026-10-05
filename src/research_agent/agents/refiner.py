"""
agents/refiner.py
=================
Refiner Agent: Polishes and corrects research briefs based on Critic feedback.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from research_agent.agent_framework import Agent, Tool
from research_agent.interfaces.llm import LLM
from research_agent.tools.earnings import get_earnings_data
from research_agent.tools.filings import search_sec_filings
from research_agent.tools.macro import get_macro_indicators
from research_agent.tools.market import get_market_technicals
from research_agent.tools.news import get_news_data

REFINER_ROLE = (
    "You are the Senior Copy Editor and Lead Revision Specialist. "
    "Your duty is to polish and refine draft research briefs that have undergone audit by the "
    "Critic.\n\n"
    "Refinement Directives:\n"
    "1. Target Flagged Issues: Address each deficiency identified in the Critic audit report "
    "point-by-point.\n"
    "2. Metric Precision: Verify all numbers against original specialist observations or use "
    "tools to fetch missing data.\n"
    "3. Structural Enhancement: Enhance document headers, bullet formatting, and clarity without "
    "altering verified facts.\n"
    "4. Maintain Balance: Ensure bull and bear sections maintain analytical equilibrium.\n"
    "5. Output directly the complete, final polished Markdown brief. Do NOT include any change "
    "summaries, notes, or meta-commentary."
)


def refiner_agent(
    model: str | LLM | None = None,
    llm: LLM | None = None,
    tools: Sequence[Tool | Callable[..., Any]] | None = None,
    verbose: bool = False,
) -> Agent:
    """Create the Refiner agent that fixes the issues raised by the Critic.

    By default it can call every data tool to fetch figures that are missing or wrong.

    Args:
        model: Model name, or an ``LLM`` client to use directly.
        llm: LLM client shared with the rest of the graph.
        tools: Tools to use instead of the default data tools.
        verbose: Log every thought, tool call and observation.

    Returns:
        A configured ``Agent`` named ``BriefRefiner``.
    """
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
