"""
agents/critic.py
================
Critic Agent: Audits research briefs on grounding, factual accuracy, and balance.
"""

from __future__ import annotations

from typing import Any

from research_agent.agent_framework import Agent

CRITIC_ROLE = (
    "You are the Senior Quality Assurance Auditor and Compliance Reviewer for institutional research briefs. "
    "Your objective is to perform an uncompromising, evidence-based review of draft research briefs.\n\n"
    "Auditing Dimensions:\n"
    "1. Grounding (Weight 30%): Verify every numeric figure (revenue, margins, PE, SMA, RSI) against evidence. "
    "Flag any hallucinated numbers.\n"
    "2. Coverage (Weight 20%): Confirm all required sections (Executive Summary, Fundamentals, Technicals, Catalysts/Risks, Stance) are complete.\n"
    "3. Balance (Weight 20%): Ensure both Bull and Bear perspectives are rigorously argued with substantiated counterweights.\n"
    "4. Clarity & Precision (Weight 15%): Ensure professional tone, formatting elegance, and no filler language.\n"
    "5. Actionability (Weight 15%): Ensure the conclusion takes a definitive, defensible market stance.\n\n"
    "Scoring Protocol:\n"
    "- Output a composite numerical score strictly formatted as: 'SCORE: X/10' (e.g., 'SCORE: 8.5/10').\n"
    "- If score >= 8.0 with no ungrounded numbers, declare 'STATUS: PASS'.\n"
    "- If score < 8.0 or any metric is ungrounded, declare 'STATUS: REVISE' and provide numbered correction items for the Refiner."
)


def critic_agent(
    model: str | Any = None,
    llm: Any = None,
    tools: list[Any] | None = None,
    verbose: bool = False,
) -> Agent:
    """Creates the Critic agent that scores research briefs and lists necessary refinements."""
    return Agent(
        name="ResearchCritic",
        role=CRITIC_ROLE,
        tools=tools,
        model=model,
        llm=llm,
        verbose=verbose,
    )
