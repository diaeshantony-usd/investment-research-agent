"""
tests/test_router.py
====================
Unit and integration tests for OrchestratorAgent / Router ReAct delegation and workflow execution.
"""

from __future__ import annotations

import json
from typing import Any

from research_agent.agent_framework import GraphMemory, OrchestratorAgent, Tool
from research_agent.agents.earnings_analyst import earnings_analyst_agent
from research_agent.agents.router import router_agent
from research_agent.interfaces import LLM
from research_agent.workflows.graph import build_research_workflow


class MockRouterTestLLM(LLM):
    """Deterministic LLM for router orchestrator testing."""

    def __init__(
        self,
        responses: list[tuple[str, list[dict[str, Any]]]] | None = None,
        model: str = "mock-router-model",
    ):
        self.model = model
        self.responses = responses or []
        self.response_idx = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: dict[str, Tool] | None = None,
    ) -> tuple[str, list[dict[str, Any]], Any]:
        if self.response_idx < len(self.responses):
            resp = self.responses[self.response_idx]
            self.response_idx += 1
            return resp[0], resp[1], None

        last_msg = messages[-1].get("content", "") if messages else ""
        return f"Handled: {last_msg[:40]}", [], None

    def _extract_tool_callables(self, tools: dict[str, Tool] | None) -> list[Any]:
        if not tools:
            return []
        return [t.func for t in tools.values()]


class TestRouterOrchestratorAgent:
    """Verifies Router OrchestratorAgent initialization, tools, and ReAct delegation."""

    def test_router_agent_properties(self):
        router = router_agent(verbose=False)
        assert isinstance(router, OrchestratorAgent)
        assert router.name == "ResearchRouter"
        assert "delegate_to_earningsanalyst" in router.tools
        assert "delegate_to_marketanalyst" in router.tools
        assert "delegate_to_newsanalyst" in router.tools

    def test_orchestrator_delegates_to_specialist_tool(self, monkeypatch):
        from research_agent.tools.earnings import get_earnings_data

        monkeypatch.setattr(
            get_earnings_data,
            "func",
            lambda ticker: json.dumps({"ticker": ticker, "revenue_b": 26.0, "net_margin": 55.0}),
        )

        llm = MockRouterTestLLM(
            responses=[
                (
                    "I will check earnings for NVDA.",
                    [
                        {
                            "name": "delegate_to_earningsanalyst",
                            "args": {"task_description": "Analyze NVDA Q2 earnings"},
                        }
                    ],
                ),
                (
                    "Checking earnings data...",
                    [{"name": "get_earnings_data", "args": {"ticker": "NVDA"}}],
                ),
                ("NVDA Q2 revenue was $26.0B with 55% net margin.", []),
                ("Based on EarningsAnalyst findings: NVDA revenue $26.0B, net margin 55%.", []),
            ]
        )

        earnings = earnings_analyst_agent(llm=llm, verbose=False)
        router = router_agent(llm=llm, subagents=[earnings], verbose=False)
        resp = router.run("What were NVDA quarterly earnings?")

        assert "NVDA revenue $26.0B" in str(resp.content)
        tool_steps = [s for s in resp.trace if s.step_type == "tool_call"]
        assert any(s.content.get("tool") == "delegate_to_earningsanalyst" for s in tool_steps)


class TestDynamicWorkflowRouting:
    """Verifies that compiled workflow executes through the orchestrator node."""

    def test_workflow_runs_with_orchestrator_node(self, monkeypatch):
        from research_agent.tools.earnings import get_earnings_data

        monkeypatch.setattr(
            get_earnings_data,
            "func",
            lambda ticker: json.dumps({"ticker": ticker, "revenue_b": 26.0, "net_margin": 55.0}),
        )

        llm = MockRouterTestLLM(
            responses=[
                ('{"research_required": true, "ticker": "NVDA"}', []),  # Planner
                (
                    "Delegating to EarningsAnalyst...",
                    [
                        {
                            "name": "delegate_to_earningsanalyst",
                            "args": {"task_description": "Analyze NVDA earnings"},
                        }
                    ],
                ),  # Orchestrator turn 1
                (
                    "Fetching data...",
                    [{"name": "get_earnings_data", "args": {"ticker": "NVDA"}}],
                ),  # EarningsAnalyst turn 1
                ("NVDA revenue was $26.0B with 55% margin.", []),  # EarningsAnalyst turn 2
                (
                    "Synthesized fundamentals: NVDA revenue $26.0B with 55% margin.",
                    [],
                ),  # Orchestrator turn 2
                (
                    "# Research Brief: NVDA\n\n## Fundamentals\nRevenue $26.0B with 55% margin.\n\n"
                    "## Executive Summary\nStrong fundamentals.\n\n"
                    "## Technicals\nTechnicals not examined in this targeted request.\n\n"
                    "## Catalysts & Risks\nRisks noted.\n\n## Stance\nPositive stance.",
                    [],
                ),  # DraftWriter
            ]
        )

        mem = GraphMemory()
        app = build_research_workflow(llm=llm, memory=mem)
        state = app.invoke({"question": "What were NVDA quarterly earnings and gross margin?"})

        executed_nodes = [s["node"] for s in state.trajectory]
        assert "planner" in executed_nodes
        assert "orchestrator" in executed_nodes
        assert executed_nodes[-1] == "release"
        assert state.output != ""
