"""
agents/router.py
================
Router / Orchestrator Agent: ReAct supervisor coordinating specialist sub-agents.
"""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.agent_framework import Agent, OrchestratorAgent, Tool
from research_agent.agents.earnings_analyst import earnings_analyst_agent
from research_agent.agents.market_analyst import market_analyst_agent
from research_agent.agents.news_analyst import news_analyst_agent
from research_agent.config import DEFAULT_MODEL
from research_agent.interfaces.llm import LLM

ROUTER_ROLE = (
    "You are the Lead Research Orchestrator and Dynamic Dispatcher on an institutional "
    "equity research desk.\n"
    "Your objective is to inspect the user's research inquiry and plan, and dynamically "
    "coordinate specialist research sub-agents using your delegated tools.\n\n"
    "Delegation Protocol:\n"
    "1. Targeted Domain Inquiries: If the user explicitly asks about specific areas "
    "(e.g., only earnings/margins, only price/technicals, or only news/catalysts), "
    "invoke ONLY the matching specialist agent(s).\n"
    "2. Comprehensive Research: If the user provides a general company inquiry without "
    "sub-domain constraints (e.g., 'Analyze NVDA', 'How is Tesla?'), invoke ALL THREE "
    "specialist agents (earnings, market, and news).\n"
    "3. Multi-Domain Requests: If the user asks for two domains (e.g., earnings and news), "
    "invoke both.\n"
    "4. Synthesis: Consolidate the findings from all invoked specialist sub-agents into a "
    "structured, publication-ready research memo."
)


ROUTER_DESCRIPTION = "Decomposing research brief and dispatching specialist analyst team"


def router_agent(
    model: str | LLM | None = None,
    llm: LLM | None = None,
    subagents: Sequence[Agent] | None = None,
    tools: Sequence[Tool] | None = None,
    verbose: bool = True,
) -> OrchestratorAgent:
    """Creates the ReAct OrchestratorAgent acting as dynamic router for specialist sub-agents.

    Args:
        model: Model name, or an ``LLM`` client to use directly.
        llm: Shared LLM client.
        subagents: Specialist sub-agents to register as callable tools. If None, instantiates
            the default specialist suite (EarningsAnalyst, MarketAnalyst, NewsAnalyst).
        tools: Optional direct utility tools.
        verbose: Enable detailed logging during execution.

    Returns:
        A configured ``OrchestratorAgent`` named ``ResearchRouter``.
    """
    chosen_llm = llm if isinstance(llm, LLM) else None
    chosen_model = model if isinstance(model, str) else DEFAULT_MODEL

    if subagents is None:
        subagents = [
            earnings_analyst_agent(llm=chosen_llm, model=chosen_model, verbose=verbose),
            market_analyst_agent(llm=chosen_llm, model=chosen_model, verbose=verbose),
            news_analyst_agent(llm=chosen_llm, model=chosen_model, verbose=verbose),
        ]

    return OrchestratorAgent(
        name="ResearchRouter",
        role=ROUTER_ROLE,
        description=ROUTER_DESCRIPTION,
        subagents=subagents,
        tools=tools,
        llm=chosen_llm,
        model=chosen_model,
        verbose=verbose,
    )
