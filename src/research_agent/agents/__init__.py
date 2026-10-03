"""
agents package
==============
Exports all specialized and supervisory agents in the investment research pipeline.
"""

from research_agent.agents.critic import CRITIC_ROLE, critic_agent
from research_agent.agents.draft_writer import DRAFT_WRITER_ROLE, draft_writer_agent
from research_agent.agents.earnings_analyst import (
    EARNINGS_ANALYST_ROLE,
    earnings_analyst_agent,
)
from research_agent.agents.market_analyst import (
    MARKET_ANALYST_ROLE,
    market_analyst_agent,
)
from research_agent.agents.news_analyst import (
    NEWS_ANALYST_ROLE,
    news_analyst_agent,
)
from research_agent.agents.planner import PLANNER_ROLE, planner_agent
from research_agent.agents.refiner import REFINER_ROLE, refiner_agent

__all__ = [
    "CRITIC_ROLE",
    "DRAFT_WRITER_ROLE",
    "EARNINGS_ANALYST_ROLE",
    "MARKET_ANALYST_ROLE",
    "NEWS_ANALYST_ROLE",
    "PLANNER_ROLE",
    "REFINER_ROLE",
    "critic_agent",
    "draft_writer_agent",
    "earnings_analyst_agent",
    "market_analyst_agent",
    "news_analyst_agent",
    "planner_agent",
    "refiner_agent",
]
