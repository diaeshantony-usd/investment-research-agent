"""
workflows/graph.py
==================
Builds the compiled multi-agent research workflow:

    planner -> (specialists in parallel) -> draft_writer -> critic <-> refiner -> release

Graph wiring: N L N Sai Krishna Akula. Critic / Refiner review loop
(``workflows/review_loop.py``): Diaesh Antony.
"""

from __future__ import annotations

from typing import Any

from research_agent.agent_framework import (
    END,
    CompiledStateGraph,
    GraphMemory,
    GraphState,
    StateGraph,
)
from research_agent.agents import (
    draft_writer_agent,
    earnings_analyst_agent,
    market_analyst_agent,
    news_analyst_agent,
    planner_agent,
    refiner_agent,
)
from research_agent.config import logger
from research_agent.workflows.review_loop import make_review_nodes

# Phrases that only a model answering from its own pre-trained knowledge would use - a
# cached, tool-sourced brief never contains them. Seen live: the Planner's Rule 4
# ("answer from entity_store") is a plain instruction, not an enforced constraint, so a
# model can still drift into pre-trained recall instead of quoting the cached brief.
# This catches that drift and forces a live research dispatch instead of trusting it.
_STALE_KNOWLEDGE_MARKERS = (
    "knowledge cutoff",
    "as of my last update",
    "as of my training",
    "my training data",
    "i don't have access to real-time",
    "i do not have access to real-time",
    "i recommend checking the latest",
    "check the latest financial reports",
)


def _looks_like_stale_pretrained_recall(plan_text: str) -> bool:
    """True if a non-RESEARCH_REQUIRED Planner answer shows signs of pre-trained recall."""
    lowered = plan_text.lower()
    return any(marker in lowered for marker in _STALE_KNOWLEDGE_MARKERS)


def route_planner(state: GraphState) -> str:
    """Run the specialists when the planner asked for research, or its "from memory"
    answer looks like ungrounded pre-trained recall rather than the cached brief.
    """
    plan = state.agents.get("ResearchPlanner", "")
    if "RESEARCH_REQUIRED" in plan.upper():
        return "specialists"
    if _looks_like_stale_pretrained_recall(plan):
        logger.warning(
            "Planner's memory-recall answer looks ungrounded (stale-knowledge phrasing "
            "found); forcing a live research dispatch instead of trusting it."
        )
        return "specialists"
    return END


def build_research_workflow(llm: Any, memory: GraphMemory) -> CompiledStateGraph:
    """Build and compile the research agent graph.

    Flow::

        planner --(RESEARCH_REQUIRED)--> specialists (earnings, market, news in parallel)
                --> draft_writer --> critic --(revise)--> refiner --> critic ...
                                     critic --(pass / stop rule)--> release --> END
        planner --(otherwise)--> END

    The critic checks the brief against the raw tool data and the refiner fixes what it
    flags; stopping rules and release are described in ``workflows/review_loop.py``.

    Args:
        llm: Chat model client shared by every agent.
        memory: Checkpoint store for finished runs.

    Returns:
        The compiled graph. Call ``invoke(payload)`` to run it.
    """
    planner = planner_agent(llm=llm, tools=memory.as_tools())
    earnings_analyst = earnings_analyst_agent(llm=llm)
    market_analyst = market_analyst_agent(llm=llm)
    news_analyst = news_analyst_agent(llm=llm)
    draft_writer = draft_writer_agent(llm=llm)
    critic_node, refiner_node, release_node, route_review = make_review_nodes(
        llm=llm, refiner=refiner_agent(llm=llm)
    )

    workflow = StateGraph(state_schema=GraphState, memory=memory, verbose=False)
    workflow.add_node("planner", planner)
    workflow.add_node("specialists", [earnings_analyst, market_analyst, news_analyst])
    workflow.add_node("draft_writer", draft_writer)
    workflow.add_node("critic", critic_node)
    workflow.add_node("refiner", refiner_node)
    workflow.add_node("release", release_node)
    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        decider=route_planner,
        route_map={"specialists": "specialists", END: END},
    )
    workflow.add_edge("specialists", "draft_writer")
    workflow.add_edge("draft_writer", "critic")

    workflow.add_conditional_edges(
        "critic",
        decider=route_review,
        route_map={"refine": "refiner", "release": "release", "default": "release"},
    )
    workflow.add_edge("refiner", "critic")
    workflow.add_edge("release", END)
    return workflow.compile()
