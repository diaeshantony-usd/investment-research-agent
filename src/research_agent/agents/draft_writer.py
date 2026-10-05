"""
agents/draft_writer.py
======================
Draft Writer Agent: Synthesizes specialist findings into a cohesive, professional research brief.
"""

from __future__ import annotations

from research_agent.agent_framework import Agent
from research_agent.interfaces.llm import LLM

DRAFT_WRITER_ROLE = (
    "You are the Senior Research Editor and Brief Synthesizer. "
    "Your primary responsibility is to consolidate disparate specialist findings (fundamentals, "
    "technicals, news, macro) "
    "into an institutional-grade, publication-ready equity research brief formatted in clean "
    "Markdown.\n\n"
    "Required Document Structure:\n"
    "# Research Brief: [Company Ticker]\n"
    "## Executive Summary\n"
    "- Core investment thesis and high-level stance (Bullish / Neutral / Bearish).\n"
    "## 1. Fundamental & Earnings Performance\n"
    "- Exact revenue, gross margins, operating income, free cash flow, and valuation multiples.\n"
    "## 2. Technical Profile & Market Momentum\n"
    "- Price action relative to 50/200-day SMAs, 14-day RSI momentum, and key support levels.\n"
    "## 3. Catalysts, Macro Context & Risk Analysis\n"
    "- Sourced news catalysts, supply chain or regulatory risks, and interest rate sensitivity.\n"
    "## 4. Investment Stance & Outlook\n"
    "- Clear balanced summary comparing upside drivers against downside risks."
)


def draft_writer_agent(
    model: str | LLM | None = None,
    llm: LLM | None = None,
    verbose: bool = False,
) -> Agent:
    """Create the DraftWriter agent that turns specialist findings into a brief.

    Args:
        model: Model name, or an ``LLM`` client to use directly.
        llm: LLM client shared with the rest of the graph.
        verbose: Log every thought, tool call and observation.

    Returns:
        A configured ``Agent`` named ``DraftWriter``.
    """
    return Agent(
        name="DraftWriter",
        role=DRAFT_WRITER_ROLE,
        model=model,
        llm=llm,
        verbose=verbose,
    )
