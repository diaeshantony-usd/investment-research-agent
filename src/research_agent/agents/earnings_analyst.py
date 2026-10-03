"""
agents/earnings_analyst.py
==========================
Earnings Analyst Agent: Gathers and analyzes quarterly earnings, fundamentals, and filings.
"""

from __future__ import annotations

from typing import Any

from research_agent.agent_framework import Agent
from research_agent.tools.earnings import get_earnings_data
from research_agent.tools.filings import search_sec_filings

EARNINGS_ANALYST_ROLE = (
    "You are the Senior Fundamentals and Earnings Analyst on an institutional equity research desk. "
    "Your mandate is to perform exhaustive forensic analysis of corporate financial health using get_earnings_data "
    "and search_sec_filings.\n\n"
    "Core Responsibilities:\n"
    "1. Core Financial Metrics: Extract quarterly revenue, gross margins, operating income, net margin, and free cash flow.\n"
    "2. Valuation Multiples: Contextualize forward P/E, EV/EBITDA, and revenue multiples against historical baselines.\n"
    "3. SEC Disclosures: Review 10-Q and 10-K disclosures for segment performance highlights and balance sheet risks.\n"
    "4. Rigorous Grounding: Every metric cited must strictly match raw tool data without hallucination or extrapolation."
)


def earnings_analyst_agent(
    model: str | Any = None,
    llm: Any = None,
    verbose: bool = False,
) -> Agent:
    """Creates the EarningsAnalyst agent equipped with earnings and filings tools."""
    return Agent(
        name="EarningsAnalyst",
        role=EARNINGS_ANALYST_ROLE,
        tools=[get_earnings_data, search_sec_filings],
        model=model,
        llm=llm,
        verbose=verbose,
    )
