"""
agents/critic.py
================
Critic Agent: Audits research briefs on grounding, factual accuracy, and balance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from research_agent.agent_framework import Agent, Tool
from research_agent.interfaces.llm import LLM

CRITIC_ROLE = (
    "You are the Senior Quality Assurance Auditor and Compliance Reviewer for institutional "
    "research briefs.\n"
    "Your objective is to perform an uncompromising, evidence-based review of draft research "
    "briefs.\n\n"
    "Auditing Dimensions:\n"
    "1. Grounding & Anti-Hallucination (Weight 30%): Verify every numeric figure (revenue, "
    "margins, PE, SMA, RSI) against specialist findings in the context. Flag ANY ungrounded "
    "or hallucinated numbers.\n"
    "2. Coverage (Weight 20%): Confirm all required sections (Executive Summary, Fundamentals, "
    "Technicals, Catalysts/Risks, Stance) are complete when company data is available.\n"
    "3. Balance (Weight 20%): Ensure both Bull and Bear perspectives are rigorously argued with "
    "substantiated counterweights.\n"
    "4. Clarity & Precision (Weight 15%): Ensure professional tone, formatting elegance, and no "
    "filler language.\n"
    "5. Actionability (Weight 15%): Ensure the conclusion takes a definitive, defensible market "
    "stance based on evidence.\n\n"
    "DATA UNAVAILABILITY AUDITING RULE:\n"
    "- If the draft is a 'Coverage Unavailable' brief because specialists confirmed no data exists "
    "in tools for the ticker, verify that it properly refrains from hallucinating numbers or "
    "stances. If so, declare 'STATUS: PASS' with 'SCORE: 10/10'.\n\n"
    "Scoring Protocol:\n"
    "- Output a composite numerical score strictly formatted as: 'SCORE: X/10' "
    "(e.g. 'SCORE: 8.5/10').\n"
    "- If score >= 8.0 with zero ungrounded/hallucinated numbers, declare 'STATUS: PASS'.\n"
    "- If score < 8.0 or ANY metric is ungrounded / hallucinated, declare 'STATUS: REVISE' "
    "and provide numbered correction items for the Refiner."
)


def critic_agent(
    model: str | LLM | None = None,
    llm: LLM | None = None,
    tools: Sequence[Tool | Callable[..., Any]] | None = None,
    verbose: bool = False,
) -> Agent:
    """Create the Critic agent that scores a draft brief.

    It returns ``SCORE: x/10`` and ``STATUS: PASS`` or ``STATUS: REVISE`` with numbered
    fixes; ``REVISE`` sends the draft to the Refiner.

    Args:
        model: Model name, or an ``LLM`` client to use directly.
        llm: LLM client shared with the rest of the graph.
        tools: Tools the critic may call (the graph passes memory tools).
        verbose: Log every thought, tool call and observation.

    Returns:
        A configured ``Agent`` named ``ResearchCritic``.
    """
    return Agent(
        name="ResearchCritic",
        role=CRITIC_ROLE,
        tools=tools,
        model=model,
        llm=llm,
        verbose=verbose,
    )
