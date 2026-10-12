"""
agents/news_analyst.py
======================
News Analyst Agent: Gathers recent headlines, sentiment signals, and catalyst events.

The agent runs the five-stage prompt chain in ``workflows/news_chain.py``
(ingest -> preprocess -> classify -> extract -> summarize) rather than a single
free-form prompt, so every bullet in its digest cites a fetched article.
"""

from __future__ import annotations

from research_agent.agent_framework import Agent
from research_agent.interfaces.llm import LLM
from research_agent.tools.news import get_news_data
from research_agent.workflows.news_chain import NewsChainAgent

NEWS_ANALYST_ROLE = (
    "You are the Financial News and Catalyst Analyst.\n"
    "You produce a cited digest of the last 30 days of news for one company through a "
    "five-stage chain: ingest headlines with get_news_data, clean and de-duplicate them, "
    "classify each by topic and sentiment, extract entities and figures that appear in "
    "the article text, and summarize the five most important developments.\n\n"
    "MANDATORY ANTI-HALLUCINATION & TOOL GROUNDING RULES:\n"
    "- Report ONLY news items returned by the tool. NEVER invent headlines, articles, or dates.\n"
    "- Every bullet must cite the supporting article id, e.g. [A3].\n"
    "- If no news is found or the ticker is not covered, state clearly: "
    "'No news data available from tools for [TICKER].' Do NOT fabricate news."
)


NEWS_ANALYST_DESCRIPTION = "Gathering recent headlines, sentiment signals, and catalyst events"


def news_analyst_agent(
    model: str | LLM | None = None,
    llm: LLM | None = None,
    verbose: bool = False,
) -> Agent:
    """Create the NewsAnalyst agent, which runs the five-stage news prompt chain.

    Args:
        model: Model name, or an ``LLM`` client to use directly.
        llm: LLM client shared with the rest of the graph.
        verbose: Log every thought, tool call and observation.

    Returns:
        A configured ``Agent`` named ``NewsAnalyst``.
    """
    return NewsChainAgent(
        name="NewsAnalyst",
        role=NEWS_ANALYST_ROLE,
        description=NEWS_ANALYST_DESCRIPTION,
        news_tool=get_news_data,
        model=model,
        llm=llm,
        verbose=verbose,
    )
