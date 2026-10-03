"""
agents/news_analyst.py
======================
News Analyst Agent: Gathers recent headlines, sentiment signals, and catalyst events.
"""

from __future__ import annotations

from typing import Any

from research_agent.agent_framework import Agent
from research_agent.tools.news import get_news_data

NEWS_ANALYST_ROLE = (
    "You are the Financial News and Catalyst Analyst. "
    "Your role is to harvest, synthesize, and filter high-impact media reporting and material corporate events "
    "using get_news_data.\n\n"
    "Analysis Protocol:\n"
    "1. Catalyst Identification: Pinpoint major catalysts including earnings surprises, product roadmaps, and executive moves.\n"
    "2. Regulatory & Geopolitical Scrutiny: Monitor antitrust probes, export controls, and regulatory challenges.\n"
    "3. Sentiment Classification: Classify recent coverage into Bullish, Neutral, or Bearish tone.\n"
    "4. Sourced Digest: Present findings in concise, cited bullet points detailing source and date."
)


def news_analyst_agent(
    model: str | Any = None,
    llm: Any = None,
    verbose: bool = False,
) -> Agent:
    """Creates the NewsAnalyst agent equipped with news retrieval tools."""
    return Agent(
        name="NewsAnalyst",
        role=NEWS_ANALYST_ROLE,
        tools=[get_news_data],
        model=model,
        llm=llm,
        verbose=verbose,
    )
