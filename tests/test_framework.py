"""
test_framework.py
=================
Comprehensive unit tests for agent_framework.py:
- Tool and @tool decorator
- Agent execution loop and tool calling
- GraphState blackboard schema
- StateGraph builder, conditional routing, cyclic loops, and CompiledStateGraph runtime
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from research_agent.agent_framework import (
    END,
    Agent,
    GraphState,
    StateGraph,
    Tool,
    display_trajectory,
    tool,
)
from research_agent.interfaces import LLM


class MockFrameworkLLM(LLM):
    """Deterministic LLM for testing framework mechanisms without network calls."""

    def __init__(
        self,
        responses: list[tuple[str, list[dict[str, Any]]]] | None = None,
        model: str = "mock-model",
    ):
        self.model = model
        self.responses = responses or []
        self.response_idx = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]], Any]:
        if self.response_idx < len(self.responses):
            resp = self.responses[self.response_idx]
            self.response_idx += 1
            return resp[0], resp[1], None

        last_msg = messages[-1].get("content", "") if messages else ""
        return f"Handled: {last_msg[:40]}", [], None

    def _extract_tool_callables(self, tools: dict[str, Tool] | None) -> list[Callable]:
        if not tools:
            return []
        return [t.func for t in tools.values()]


class TestAgentFramework:
    """Verifies core classes and orchestration mechanics in agent_framework.py."""

    def test_tool_class_and_decorator(self):
        def sample_add(a: int, b: int) -> int:
            """Adds two integers."""
            return a + b

        t1 = Tool(name="add", description="Add numbers", func=sample_add)
        assert t1.name == "add"
        assert t1.description == "Add numbers"
        assert t1.execute(a=2, b=3) == "5"

        @tool(name="multiply", description="Multiply numbers")
        def sample_mul(a: int, b: int) -> int:
            return a * b

        assert isinstance(sample_mul, Tool)
        assert sample_mul.name == "multiply"
        assert sample_mul.execute(a=3, b=4) == "12"

    def test_agent_run_direct_answer(self):
        llm = MockFrameworkLLM(responses=[("Direct response from agent", [])])
        agent = Agent(name="TestAgent", role="Tester", llm=llm)
        resp = agent.run("Hello test")

        assert resp.agent_name == "TestAgent"
        assert resp.content == "Direct response from agent"
        assert any(step.step_type == "final_answer" for step in resp.trace)

    def test_agent_tool_calling_loop(self):
        @tool(name="get_stock_price", description="Fetch stock price")
        def get_stock_price(ticker: str) -> str:
            return f"${ticker.upper()}: 120.50"

        llm = MockFrameworkLLM(
            responses=[
                # Turn 1: model requests tool call
                (
                    "Checking price...",
                    [{"name": "get_stock_price", "args": {"ticker": "NVDA"}}],
                ),
                # Turn 2: model returns final answer
                ("NVDA is currently trading at $120.50.", []),
            ]
        )
        agent = Agent(name="PriceAgent", role="Price Checker", tools=[get_stock_price], llm=llm)
        resp = agent.run("What is NVDA price?")

        assert "120.50" in resp.content
        trace_types = [t.step_type for t in resp.trace]
        assert "tool_call" in trace_types
        assert "tool_result" in trace_types
        assert "final_answer" in trace_types

    def test_graph_state_schema(self):
        state = GraphState(
            question="What is operating margin?",
            thread_id="test_thread_1",
            refine_count=1,
        )
        assert state.question == "What is operating margin?"
        assert state.thread_id == "test_thread_1"
        assert state.refine_count == 1
        assert isinstance(state.agents, dict)
        assert isinstance(state.tools, dict)
        assert isinstance(state.trajectory, list)

        # Dictionary access compatibility
        state["custom_key"] = "custom_val"
        assert state["custom_key"] == "custom_val"
        assert state.get("non_existent", "fallback") == "fallback"

    def test_stategraph_linear_pipeline(self):
        llm = MockFrameworkLLM(
            responses=[
                ("Analysis completed.", []),
                ("Brief finalized.", []),
            ]
        )
        analyst = Agent(name="Analyst", role="Analyst", llm=llm)
        writer = Agent(name="Writer", role="Writer", llm=llm)

        graph = StateGraph(state_schema=GraphState)
        graph.add_node("analyst", analyst)
        graph.add_node("writer", writer)
        graph.set_entry_point("analyst")
        graph.add_edge("analyst", "writer")
        graph.add_edge("writer", END)

        app = graph.compile()
        state = app.invoke({"question": "Analyze company"})

        assert len(state.trajectory) == 2
        assert state.trajectory[0]["node"] == "analyst"
        assert state.trajectory[1]["node"] == "writer"
        assert state.output == "Brief finalized."

    def test_stategraph_conditional_routing(self):
        llm = MockFrameworkLLM(responses=[("Non-research answer", [])])
        planner = Agent(name="Planner", role="Router", llm=llm)

        graph = StateGraph(state_schema=GraphState)
        graph.add_node("planner", planner)
        graph.set_entry_point("planner")

        def route_fn(state: GraphState) -> str:
            ans = state.agents.get("Planner", "")
            return "specialist" if "RESEARCH" in ans else END

        graph.add_conditional_edges(
            "planner", decider=route_fn, route_map={"specialist": "specialist", END: END}
        )
        app = graph.compile()

        state = app.invoke({"question": "General question"})
        assert len(state.trajectory) == 1
        assert state.trajectory[0]["node"] == "planner"
        assert state.output == "Non-research answer"

    def test_stategraph_cyclic_loop(self):
        # Evaluator-Optimizer loop: Writer -> Critic -> Refiner -> Critic -> END
        llm = MockFrameworkLLM(
            responses=[
                ("Draft v1", []),
                ("NEEDS WORK: missing data", []),
                ("Draft v2 (refined)", []),
                ("PASS: verified", []),
            ]
        )
        writer = Agent(name="Writer", role="Drafting", llm=llm)
        critic = Agent(name="Critic", role="Auditing", llm=llm)
        refiner = Agent(name="Refiner", role="Refining", llm=llm)

        graph = StateGraph(state_schema=GraphState)
        graph.add_node("writer", writer)
        graph.add_node("critic", critic)
        graph.add_node("refiner", refiner)
        graph.set_entry_point("writer")
        graph.add_edge("writer", "critic")

        def route_critique(state: GraphState) -> str:
            refine_count = getattr(state, "refine_count", 0)
            critique = state.agents.get("Critic", "").upper()
            if "NEEDS WORK" in critique and refine_count < 1:
                state.refine_count = refine_count + 1
                return "refine"
            return "pass"

        graph.add_conditional_edges(
            "critic",
            decider=route_critique,
            route_map={"refine": "refiner", "pass": END},
        )
        graph.add_edge("refiner", "critic")

        app = graph.compile()
        state = app.invoke({"question": "Write research brief"})

        # Trajectory sequence: writer -> critic -> refiner -> critic
        nodes = [s["node"] for s in state.trajectory]
        assert nodes == ["writer", "critic", "refiner", "critic"]
        assert state.output == "Draft v2 (refined)"

    def test_display_trajectory_table(self):
        state = GraphState(
            trajectory=[
                {
                    "step": 1,
                    "node": "planner",
                    "type": "agent",
                    "executor": "ResearchPlanner",
                    "tools": [],
                    "summary": "Direct answer",
                }
            ]
        )
        ascii_table = display_trajectory(state, as_markdown=False)
        assert ascii_table is not None
        assert "EXECUTION TRAJECTORY TABLE" in ascii_table
        assert "ResearchPlanner" in ascii_table

    def test_agent_system_prompt_grounding_directive(self):
        tool = Tool(name="fetch_data", description="Fetches data", func=lambda: "data")
        agent = Agent(name="DataAnalyst", tools=[tool], llm=MockFrameworkLLM())
        sys_prompt = agent._format_system_prompt()
        assert "MANDATORY TOOL USE & GROUNDING DIRECTIVE" in sys_prompt
        assert "fetch_data" in sys_prompt

    def test_planner_role_guardrails(self):
        from research_agent.agents.planner import PLANNER_ROLE

        assert "CRITICAL DOMAIN GUARDRAILS" in PLANNER_ROLE
        assert "2+2" in PLANNER_ROLE
        assert "RESEARCH_REQUIRED" in PLANNER_ROLE
        assert "cannot answer non-financial questions" in PLANNER_ROLE

    def test_draft_writer_role_data_availability_gate(self):
        from research_agent.agents.draft_writer import DRAFT_WRITER_ROLE

        assert "DATA AVAILABILITY GATE" in DRAFT_WRITER_ROLE
        assert "Data Unavailable" in DRAFT_WRITER_ROLE
        assert "MUST NOT issue an investment stance" in DRAFT_WRITER_ROLE

    def test_workflow_data_unavailable_passes_critic(self):
        from research_agent.agent_framework import GraphMemory
        from research_agent.workflows.graph import build_research_workflow

        # Workflow where DraftWriter outputs Coverage Unavailable and Critic passes it
        llm = MockFrameworkLLM(
            responses=[
                ("RESEARCH_REQUIRED: TSLA - evaluate", []),  # Planner
                ("No earnings data for TSLA", []),  # EarningsAnalyst
                ("No market technicals for TSLA", []),  # MarketAnalyst
                ("No news for TSLA", []),  # NewsAnalyst
                ("# Research Brief: TSLA - Coverage Unavailable", []),  # DraftWriter
                ("SCORE: 10/10. STATUS: PASS - Data unavailable confirmed.", []),  # Critic
            ]
        )
        mem = GraphMemory()
        app = build_research_workflow(llm=llm, memory=mem)
        state = app.invoke({"question": "how is tesla?"})

        executed_nodes = [s["node"] for s in state.trajectory]
        assert "planner" in executed_nodes
        assert "specialists" in executed_nodes
        assert "draft_writer" in executed_nodes
        assert "critic" in executed_nodes
        assert "refiner" not in executed_nodes
        assert "Coverage Unavailable" in state.output

    def test_workflow_uncovered_ticker_with_tools_passes_critic(self):
        from research_agent.agent_framework import GraphMemory
        from research_agent.workflows.graph import build_research_workflow

        # Specialists invoke real tools for TSLA which return no data / errors
        llm = MockFrameworkLLM(
            responses=[
                ("RESEARCH_REQUIRED: TSLA - evaluate", []),  # Planner
                (
                    "Checking earnings...",
                    [{"name": "get_earnings_data", "args": {"ticker": "TSLA"}}],
                ),
                ("No earnings found for TSLA.", []),
                (
                    "Checking market...",
                    [{"name": "get_market_technicals", "args": {"ticker": "TSLA"}}],
                ),
                ("No technicals found for TSLA.", []),
                ("Checking news...", [{"name": "get_news_data", "args": {"ticker": "TSLA"}}]),
                ("No news found for TSLA.", []),
                ("# Research Brief: TSLA - Data Unavailable", []),  # DraftWriter
                ("SCORE: 10/10. STATUS: PASS - Verified no data.", []),  # Critic
            ]
        )
        mem = GraphMemory()
        app = build_research_workflow(llm=llm, memory=mem)
        state = app.invoke({"question": "how is tesla?"})

        executed_nodes = [s["node"] for s in state.trajectory]
        assert "planner" in executed_nodes
        assert "specialists" in executed_nodes
        assert "draft_writer" in executed_nodes
        assert "critic" in executed_nodes
        assert "refiner" not in executed_nodes
        assert "Data Unavailable" in state.output


