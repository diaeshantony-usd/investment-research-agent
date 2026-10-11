"""
test_workflows_graph.py
========================
Tests for ``workflows/graph.py``'s planner routing, specifically the guardrail added
after a live bug report: the Planner can answer a "recall from memory" question with
pre-trained knowledge instead of the cached, tool-sourced brief (e.g. "As of my
knowledge cutoff in December 2023..."), and nothing used to catch it since that path
skips the Critic/Refiner review loop entirely. ``route_planner`` now detects that
stale-knowledge phrasing and forces a live research dispatch instead of trusting it.
"""

from __future__ import annotations

import json
from typing import Any

from research_agent.agent_framework import GraphMemory, GraphState
from research_agent.interfaces import LLM
from research_agent.workflows.graph import (
    _looks_like_stale_pretrained_recall,
    build_research_workflow,
    route_planner,
)


class TestLooksLikeStalePretrainedRecall:
    def test_detects_knowledge_cutoff_phrasing(self):
        text = (
            "Microsoft Corporation (MSFT)... As of my knowledge cutoff in December 2023, "
            "Microsoft's revenue was generated primarily through..."
        )
        assert _looks_like_stale_pretrained_recall(text) is True

    def test_detects_recommend_checking_latest_phrasing(self):
        text = "For a more up-to-date analysis, I recommend checking the latest financial reports."
        assert _looks_like_stale_pretrained_recall(text) is True

    def test_grounded_recall_is_not_flagged(self):
        text = (
            "Based on the cached brief, MSFT's Q4 revenue was $64.7 Billion with a forward "
            "P/E of 31.5."
        )
        assert _looks_like_stale_pretrained_recall(text) is False

    def test_case_insensitive(self):
        assert _looks_like_stale_pretrained_recall("AS OF MY KNOWLEDGE CUTOFF...") is True


class TestRoutePlanner:
    def test_research_required_routes_to_specialists(self):
        state = GraphState()
        state.agents["ResearchPlanner"] = "RESEARCH_REQUIRED: MSFT - evaluate"
        assert route_planner(state) == "specialists"

    def test_grounded_memory_recall_ends_workflow(self):
        state = GraphState()
        state.agents["ResearchPlanner"] = "Per the cached brief, MSFT's revenue was $64.7B."
        assert route_planner(state) == "__end__"

    def test_stale_pretrained_recall_is_forced_to_specialists(self):
        state = GraphState()
        state.agents["ResearchPlanner"] = (
            "Microsoft Corporation (MSFT) is a multinational technology company... As of my "
            "knowledge cutoff in December 2023, Microsoft's revenue was generated primarily "
            "through its Productivity and Business Processes segment..."
        )
        assert route_planner(state) == "specialists"


class ScriptedLLM(LLM):
    """Replays scripted ``(content, tool_calls)`` pairs in order."""

    def __init__(self, responses: list[tuple[str, list[dict[str, Any]]]]) -> None:
        self.model = "mock"
        self.responses = responses
        self.idx = 0

    def chat(self, messages, tools=None):
        content, calls = self.responses[min(self.idx, len(self.responses) - 1)]
        self.idx += 1
        return content, calls, {"role": "assistant", "content": content}


def _stub_no_news(monkeypatch) -> None:
    """Make get_news_data return no articles, so the News analyst's deterministic
    chain exits before its own LLM calls, keeping the scripted response queue aligned.
    """
    from research_agent.tools.news import get_news_data

    monkeypatch.setattr(
        get_news_data,
        "func",
        lambda ticker, days=7: json.dumps({"ticker": ticker, "articles": [], "message": "none"}),
    )


def test_stale_pretrained_recall_triggers_a_real_research_run(monkeypatch):
    """End-to-end reproduction of the reported MSFT bug: the Planner tries to answer a
    'recall from memory' question with pre-trained knowledge and no RESEARCH_REQUIRED
    marker. Before the fix this text became the final answer unchecked; now the
    workflow detects it and runs the specialists for real instead.
    """
    _stub_no_news(monkeypatch)
    llm = ScriptedLLM(
        [
            (
                "Microsoft Corporation (MSFT) is a multinational technology company... As of "
                "my knowledge cutoff in December 2023, Microsoft's revenue was generated "
                "primarily through its Productivity and Business Processes segment...",
                [],
            ),  # Planner: no RESEARCH_REQUIRED, but stale pre-trained phrasing
            ("No earnings tools needed for this summary.", []),  # EarningsAnalyst
            ("No market tools needed for this summary.", []),  # MarketAnalyst
            ("# Research Brief: MSFT\nGrounded, tool-backed content.", []),  # DraftWriter
            ("SCORE: 9/10. STATUS: PASS.", []),  # Critic
        ]
    )
    mem = GraphMemory()
    app = build_research_workflow(llm=llm, memory=mem)
    state = app.invoke({"question": "how is MSFT doing?"})

    executed_nodes = [s["node"] for s in state.trajectory]
    assert "specialists" in executed_nodes
    assert "draft_writer" in executed_nodes
    assert "knowledge cutoff" not in state.output.lower()
