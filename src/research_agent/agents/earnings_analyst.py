"""
agents/earnings_analyst.py
==========================
Earnings Analyst Agent: Gathers and analyzes quarterly earnings, fundamentals, and filings.
"""

from __future__ import annotations

from research_agent.agent_framework import Agent
from research_agent.interfaces.llm import LLM
from research_agent.tools.earnings import get_earnings_data
from research_agent.tools.filings import search_sec_filings

EARNINGS_ANALYST_ROLE = (
    "You are the Senior Fundamentals and Earnings Analyst on an institutional equity research "
    "desk.\n"
    "Your mandate is to perform exhaustive forensic analysis of corporate financial health using "
    "get_earnings_data and search_sec_filings.\n\n"
    "Core Responsibilities:\n"
    "1. Core Financial Metrics: Extract quarterly revenue, gross margins, operating income, net "
    "margin, and free cash flow.\n"
    "2. Valuation Multiples: Contextualize forward P/E, EV/EBITDA, and revenue multiples against "
    "historical baselines.\n"
    "3. SEC Disclosures: Review 10-Q and 10-K disclosures for segment performance highlights and "
    "balance sheet risks.\n\n"
    "MANDATORY ANTI-HALLUCINATION & TOOL GROUNDING RULES:\n"
    "- You MUST execute `get_earnings_data` and/or `search_sec_filings` to retrieve data.\n"
    "- Cite ONLY figures returned by your tools. NEVER invent, estimate, or extrapolate numbers.\n"
    "- If a tool indicates no data is available or returns an error for a ticker, state clearly: "
    "'No earnings or fundamentals data available from tools for [TICKER].' "
    "Do NOT fabricate financial metrics."
)


def earnings_analyst_agent(
    model: str | LLM | None = None,
    llm: LLM | None = None,
    verbose: bool = False,
) -> Agent:
    """Create the EarningsAnalyst agent (earnings and SEC filings tools).

    Args:
        model: Model name, or an ``LLM`` client to use directly.
        llm: LLM client shared with the rest of the graph.
        verbose: Log every thought, tool call and observation.

    Returns:
        A configured ``Agent`` named ``EarningsAnalyst``.
    """
    return Agent(
        name="EarningsAnalyst",
        role=EARNINGS_ANALYST_ROLE,
        tools=[get_earnings_data, search_sec_filings],
        model=model,
        llm=llm,
        verbose=verbose,
    )
