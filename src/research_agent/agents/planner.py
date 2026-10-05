"""
agents/planner.py
=================
Planner Agent: Decomposes equity research questions into ordered specialist tasks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from research_agent.agent_framework import Agent, Tool
from research_agent.interfaces.llm import LLM

PLANNER_ROLE = (
    "You are the Lead Investment Strategist and Query Dispatcher. "
    "Analyze the user's input:\n\n"
    "1. Non-Research / General Queries:\n"
    "If the query is a general financial concept, educational question, conversation, or greeting "
    "(e.g., 'What is operating margin?', 'Explain P/E ratio', 'Hello'):\n"
    "Directly provide a comprehensive, clear, and helpful answer. Do NOT suggest research steps "
    "or hand off.\n\n"
    "2. Company Research Queries:\n"
    "If the user asks for financial analysis, valuation, or research on a specific stock or "
    "company "
    "(e.g., 'Analyze NVDA', 'Evaluate Apple earnings'):\n"
    "State concisely:\n"
    "'RESEARCH_REQUIRED: [Ticker] - [Key focus areas]'\n"
    "Do NOT write elaborate speculative plans or assumptions about future findings. "
    "Keep it brief so the specialist agents and router can execute data tools."
)


def planner_agent(
    model: str | LLM | None = None,
    llm: LLM | None = None,
    tools: Sequence[Tool | Callable[..., Any]] | None = None,
    verbose: bool = False,
) -> Agent:
    """Create the Planner agent (graph entry point).

    It answers general finance questions directly. For company research it replies
    ``RESEARCH_REQUIRED: <ticker> - <focus areas>``, which routes the graph to the
    specialist analysts.

    Args:
        model: Model name, or an ``LLM`` client to use directly.
        llm: LLM client shared with the rest of the graph.
        tools: Tools the planner may call (the graph passes memory tools).
        verbose: Log every thought, tool call and observation.

    Returns:
        A configured ``Agent`` named ``ResearchPlanner``.
    """
    return Agent(
        name="ResearchPlanner",
        role=PLANNER_ROLE,
        model=model,
        llm=llm,
        tools=tools,
        verbose=verbose,
    )
