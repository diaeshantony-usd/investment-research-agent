"""
workflows/graph.py
==================
Builds the compiled multi-agent research workflow:

    planner -> (specialists in parallel) -> draft_writer -> critic <-> refiner

Author of the graph wiring: N L N Sai Krishna Akula (moved here from
``workflows/session.py`` with routing behaviour unchanged, so that ``ResearchSession``
can be developed separately).
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
    critic_agent,
    draft_writer_agent,
    earnings_analyst_agent,
    market_analyst_agent,
    news_analyst_agent,
    planner_agent,
    refiner_agent,
)

# The critic asks for another revision when its verdict contains any of these markers.
_REVISION_MARKERS = ("REVISE", "FAIL", "NEEDS WORK", "UNVERIFIED", "DEFICIENCY")

# Upper bound on refiner -> critic loops, so a strict critic cannot loop forever.
_MAX_REFINEMENTS = 2


def build_research_workflow(llm: Any, memory: GraphMemory) -> CompiledStateGraph:
    """Build and compile the research agent graph.

    Flow::

        planner --(RESEARCH_REQUIRED)--> specialists (earnings, market, news in parallel)
                --> draft_writer --> critic --(needs work)--> refiner --> critic ...
        planner --(otherwise)--> END              critic --(pass)--> END

    Args:
        llm: Chat model client shared by every agent.
        memory: Checkpoint store; the planner and critic also receive its tools.

    Returns:
        The compiled graph. Call ``invoke(payload)`` to run it.
    """
    planner = planner_agent(llm=llm, tools=memory.as_tools())
    earnings_analyst = earnings_analyst_agent(llm=llm)
    market_analyst = market_analyst_agent(llm=llm)
    news_analyst = news_analyst_agent(llm=llm)
    draft_writer = draft_writer_agent(llm=llm)
    critic = critic_agent(llm=llm, tools=memory.as_tools())
    refiner = refiner_agent(llm=llm)

    workflow = StateGraph(state_schema=GraphState, memory=memory, verbose=False)
    workflow.add_node("planner", planner)
    workflow.add_node("specialists", [earnings_analyst, market_analyst, news_analyst])
    workflow.add_node("draft_writer", draft_writer)
    workflow.add_node("critic", critic)
    workflow.add_node("refiner", refiner)
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

    def route_critique(state: GraphState) -> str:
        """Send the draft back to the refiner until it passes or the loop limit is hit."""
        refine_count = getattr(state, "refine_count", 0)
        critique_text = state.agents.get("ResearchCritic", "").upper()
        needs_work = any(marker in critique_text for marker in _REVISION_MARKERS)
        if needs_work and refine_count < _MAX_REFINEMENTS:
            state.refine_count = refine_count + 1
            return "refine"
        return "pass"

    workflow.add_conditional_edges(
        "critic",
        decider=route_critique,
        route_map={"refine": "refiner", "pass": END, "default": END},
    )
    workflow.add_edge("refiner", "critic")
    return workflow.compile()
