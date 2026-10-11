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
from research_agent.workflows.review_loop import make_review_nodes


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

    def route_planner(state: GraphState) -> str:
        """Run the specialists only when the planner decided research is needed."""
        plan = state.agents.get("ResearchPlanner", "").upper()
        if "RESEARCH_REQUIRED" in plan:
            return "specialists"
        return END

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
