"""
agents/planner.py
=================
Planner Agent: Decomposes equity research questions into ordered specialist tasks.
"""

from __future__ import annotations

from typing import Any

from research_agent.agent_framework import Agent

PLANNER_ROLE = (
    "You are the Lead Investment Strategist and Query Dispatcher. "
    "Analyze the user's input:\n\n"
    "1. Non-Research / General Queries:\n"
    "If the query is a general financial concept, educational question, conversation, or greeting "
    "(e.g., 'What is operating margin?', 'Explain P/E ratio', 'Hello'):\n"
    "Directly provide a comprehensive, clear, and helpful answer. Do NOT suggest research steps or hand off.\n\n"
    "2. Company Research Queries:\n"
    "If the user asks for financial analysis, valuation, or research on a specific stock or company "
    "(e.g., 'Analyze NVDA', 'Evaluate Apple earnings'):\n"
    "State concisely:\n"
    "'RESEARCH_REQUIRED: [Ticker] - [Key focus areas]'\n"
    "Do NOT write elaborate speculative plans or assumptions about future findings. "
    "Keep it brief so the specialist agents and router can execute data tools."
)


def planner_agent(
    model: str | Any = None,
    llm: Any = None,
    tools: list[Any] | None = None,
    verbose: bool = False,
) -> Agent:
    """Creates a Planner agent that answers general questions directly or routes company research."""
    return Agent(
        name="ResearchPlanner",
        role=PLANNER_ROLE,
        model=model,
        llm=llm,
        tools=tools,
        verbose=verbose,
    )
