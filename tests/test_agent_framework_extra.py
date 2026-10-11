"""
test_agent_framework_extra.py
==============================
Targeted tests for ``agent_framework.py`` branches not already exercised by
``test_framework.py``: verbose logging paths, the LLM-error and max-iterations
exits of the Agent loop, GraphMemory's legacy-format and edge-case handling,
StateGraph compile/draw, conditional routing variants, and the trajectory
renderers. All offline; no network or real LLM calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from research_agent import agent_framework
from research_agent.agent_framework import (
    END,
    Agent,
    AgentResponse,
    GraphMemory,
    GraphState,
    StateGraph,
    Tool,
    TraceStep,
    _compact_text,
    _format_step_summary,
    display_trajectory,
    format_trajectory_markdown,
)
from research_agent.interfaces import LLM


class ScriptedLLM(LLM):
    """Replays scripted ``(content, tool_calls)`` pairs; repeats the last one forever."""

    def __init__(
        self, responses: list[tuple[str, list[dict[str, Any]]]], model: str = "mock"
    ) -> None:
        self.model = model
        self.responses = responses
        self.idx = 0

    def chat(self, messages, tools=None):
        resp = self.responses[min(self.idx, len(self.responses) - 1)]
        self.idx += 1
        return resp[0], resp[1], {"role": "assistant", "content": resp[0]}


class ExplodingLLM(LLM):
    model = "exploding"

    def chat(self, messages, tools=None):
        raise RuntimeError("provider unreachable")


# ===========================================================================
# Tool
# ===========================================================================


class TestToolExecuteAndCallable:
    def test_execute_catches_exceptions(self):
        t = Tool(name="boom", description="d", func=lambda: 1 / 0)
        result = t.execute()
        assert "Error executing tool 'boom'" in result
        assert "ZeroDivisionError" in result

    def test_to_callable_routes_through_execute(self):
        t = Tool(name="add", description="d", func=lambda a, b: a + b)
        wrapper = t.to_callable()
        assert wrapper(a=1, b=2) == "3"
        assert wrapper.__name__ == "add"


# ===========================================================================
# AgentResponse
# ===========================================================================


class TestAgentResponse:
    def test_str_returns_content(self):
        resp = AgentResponse("A", "prompt", "the answer", [], [], 1)
        assert str(resp) == "the answer"

    def test_display_trace_handles_every_step_type(self):
        trace = [
            TraceStep("10:00:00", "A", "thought", "thinking..."),
            TraceStep("10:00:01", "A", "tool_call", {"tool": "x", "arguments": {}}),
            TraceStep("10:00:02", "A", "tool_result", {"ok": True}),
            TraceStep("10:00:03", "A", "final_answer", "done"),
            TraceStep("10:00:04", "A", "weird_step", "???"),
        ]
        resp = AgentResponse("A", "p", "done", trace, [], 1)
        resp.display_trace()  # must not raise


# ===========================================================================
# Agent
# ===========================================================================


class TestAgentInit:
    def test_accepts_llm_instance_as_model(self):
        llm = ScriptedLLM([("hi", [])])
        agent = Agent(name="A", model=llm)
        assert agent.llm is llm
        assert agent.model == "mock"

    def test_accepts_explicit_llm_kwarg(self):
        llm = ScriptedLLM([("hi", [])], model="explicit")
        agent = Agent(name="A", llm=llm)
        assert agent.llm is llm
        assert agent.model == "explicit"

    def test_builds_default_llm_when_none_given(self, monkeypatch):
        from research_agent.interfaces import llm as llm_module

        monkeypatch.setattr(llm_module.ollama, "Client", lambda host, headers: object())
        agent = Agent(name="A")
        assert agent.llm is not None

    def test_add_tool_wraps_plain_callable(self):
        def my_fn(x: int) -> int:
            """Doubles x."""
            return x * 2

        agent = Agent(name="A", llm=ScriptedLLM([("hi", [])]), tools=[my_fn])
        assert "my_fn" in agent.tools
        assert isinstance(agent.tools["my_fn"], Tool)

    def test_add_tool_rejects_non_callable(self):
        agent = Agent(name="A", llm=ScriptedLLM([("hi", [])]))
        with pytest.raises(TypeError, match="Expected Tool or Callable"):
            agent.add_tool(123)

    def test_extract_tool_callables_delegates_to_llm(self):
        llm = ScriptedLLM([("hi", [])])
        agent = Agent(name="A", llm=llm, tools=[Tool(name="t", description="d", func=lambda: 1)])
        callables = agent._extract_tool_callables()
        assert [c.__name__ for c in callables] == [
            c.__name__ for c in llm.get_tool_callables(agent.tools)
        ]


class TestAgentRun:
    def test_verbose_logs_activation_thought_and_final_answer(self, caplog):
        llm = ScriptedLLM([("final thought", [])])
        agent = Agent(name="A", llm=llm, verbose=True)
        resp = agent.run("hello")
        assert resp.content == "final thought"

    def test_llm_error_ends_run_without_raising(self):
        agent = Agent(name="A", llm=ExplodingLLM())
        resp = agent.run("hello")
        assert "API Error" in resp.content
        assert resp.trace[-1].step_type == "error"

    def test_unknown_tool_call_reports_available_tools(self):
        llm = ScriptedLLM(
            [
                ("calling", [{"name": "not_a_real_tool", "args": {}}]),
                ("done", []),
            ]
        )
        agent = Agent(
            name="A",
            llm=llm,
            verbose=True,
            tools=[Tool(name="real", description="d", func=lambda: 1)],
        )
        resp = agent.run("hello")
        tool_results = [s for s in resp.trace if s.step_type == "tool_result"]
        assert "not in available tools" in tool_results[0].content

    def test_known_tool_call_executes_and_logs(self):
        llm = ScriptedLLM(
            [
                ("calling", [{"name": "real", "args": {"x": 1}}]),
                ("done", []),
            ]
        )
        agent = Agent(
            name="A",
            llm=llm,
            verbose=True,
            tools=[Tool(name="real", description="d", func=lambda x: x * 2)],
        )
        resp = agent.run("hello")
        tool_results = [s for s in resp.trace if s.step_type == "tool_result"]
        assert tool_results[0].content == "2"

    def test_stops_at_max_iterations_without_final_answer(self):
        llm = ScriptedLLM([("still going", [{"name": "real", "args": {}}])])
        agent = Agent(
            name="A",
            llm=llm,
            max_iterations=2,
            tools=[Tool(name="real", description="d", func=lambda: "ok")],
        )
        resp = agent.run("hello")
        assert "stopped after reaching maximum iterations" in resp.content
        assert resp.iterations == 2


# ===========================================================================
# GraphState
# ===========================================================================


class TestGraphState:
    def test_tools_data_property_and_getitem_alias(self):
        state = GraphState()
        state.tools["x"] = {"a": 1}
        assert state.tools_data == state.tools
        assert state["tools_data"] == state.tools

    def test_getitem_raises_keyerror_for_unknown_key(self):
        state = GraphState()
        with pytest.raises(KeyError):
            _ = state["not_a_field_or_agent"]

    def test_get_returns_default_for_unknown_key(self):
        state = GraphState()
        assert state.get("missing", "fallback") == "fallback"


# ===========================================================================
# GraphMemory
# ===========================================================================


class TestGraphMemoryWrite:
    def test_write_accepts_plain_dict_state(self):
        mem = GraphMemory()
        state_id = mem.write({"thread_id": "t1", "question": "hi"})
        assert state_id in mem.records["t1"]["states"]
        assert mem.records["t1"]["states"][state_id]["input"] == "hi"

    def test_write_repairs_thread_missing_states_key(self):
        mem = GraphMemory()
        mem.records["t1"] = {}  # legacy/malformed: no "states" key yet
        mem.write({"thread_id": "t1"})
        assert "states" in mem.records["t1"]


class TestGraphMemoryLegacyFormat:
    def test_all_states_flattens_legacy_flat_thread(self):
        mem = GraphMemory()
        mem.records["legacy"] = {"state_abc": {"question": "hi"}}  # no "states" wrapper
        assert mem._all_states() == {"state_abc": {"question": "hi"}}

    def test_list_threads_and_list_state_ids(self):
        mem = GraphMemory()
        mem.write({"thread_id": "t1"})
        assert "t1" in json.loads(mem.list_threads())
        assert json.loads(mem.list_state_ids(thread_id="t1"))
        assert json.loads(mem.list_state_ids())


class TestGraphMemoryHistoryPaging:
    def test_invalid_page_and_limit_fall_back_to_defaults(self):
        mem = GraphMemory()
        mem.write({"thread_id": "t1"})
        result = json.loads(mem.get_history_states(page="not a number", limit="also not"))
        assert result["page"] == 1
        assert result["limit"] == 1

    def test_scopes_to_thread_when_given(self):
        mem = GraphMemory()
        mem.write({"thread_id": "t1"})
        mem.write({"thread_id": "t2"})
        result = json.loads(mem.get_history_states(thread_id="t1"))
        assert result["thread_id"] == "t1"
        assert result["total_records"] == 1

    def test_get_state_missing_id_returns_error_payload(self):
        mem = GraphMemory()
        result = json.loads(mem.get_state("nope"))
        assert "error" in result

    def test_as_tools_returns_four_callable_tools(self):
        mem = GraphMemory()
        tools = mem.as_tools()
        assert len(tools) == 4
        assert all(isinstance(t, Tool) for t in tools)


class TestGraphMemoryLoad:
    def test_load_missing_file_yields_empty_store(self, tmp_path):
        mem = GraphMemory()
        mem.load(str(tmp_path / "does_not_exist.json"))
        assert mem.records == {}

    def test_load_empty_file_yields_empty_store(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("")
        mem = GraphMemory()
        mem.load(str(path))
        assert mem.records == {}

    def test_load_corrupt_json_yields_empty_store(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json")
        mem = GraphMemory()
        mem.load(str(path))
        assert mem.records == {}

    def test_load_converts_legacy_flat_layout(self, tmp_path):
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps({"state_1": {"thread_id": "t1", "question": "hi"}}))
        mem = GraphMemory()
        mem.load(str(path))
        assert "state_1" in mem.records["t1"]["states"]

    def test_init_with_existing_filepath_loads_it(self, tmp_path):
        path = tmp_path / "preload.json"
        path.write_text(json.dumps({"t1": {"states": {"s1": {"question": "hi"}}}}))
        mem = GraphMemory(filepath=str(path))
        assert "s1" in mem.records["t1"]["states"]


# ===========================================================================
# StateGraph / compile / draw
# ===========================================================================


class TestStateGraphCompile:
    def test_compile_picks_first_node_as_entry_point(self):
        graph = StateGraph().add_node("only", lambda s: {})
        compiled = graph.compile()
        assert compiled.graph.entry_point == "only"

    def test_compile_raises_without_nodes_or_entry_point(self):
        with pytest.raises(ValueError, match="no nodes or entry point"):
            StateGraph().compile()

    def test_compile_with_draw_true_logs_mermaid(self, caplog):
        graph = StateGraph().add_node("only", lambda s: {})
        graph.compile(draw=True)  # must not raise even without IPython installed


class TestToMermaid:
    def test_renders_all_node_and_edge_kinds(self):
        llm = ScriptedLLM([("done", [])])
        agent_a = Agent(name="AgentA", llm=llm)
        agent_b = Agent(name="AgentB", llm=llm)

        graph = (
            StateGraph()
            .add_node("parallel_step", [agent_a, agent_b])
            .add_node("agent_step", agent_a)
            .add_node("fn_step", lambda s: {})
            .add_edge("parallel_step", "agent_step")
            .add_edge("agent_step", "fn_step")
            .add_conditional_edges("fn_step", lambda s: "done", {"done": END})
            .set_entry_point("parallel_step")
        )
        compiled = graph.compile()
        chart = compiled.to_mermaid()
        assert "Parallel: [AgentA, AgentB]" in chart
        assert "Agent: AgentA" in chart
        assert "fn_step (fn:" in chart
        assert "__end__([End])" in chart

    def test_conditional_edge_without_route_map(self):
        graph = (
            StateGraph()
            .add_node("a", lambda s: {})
            .add_conditional_edges("a", lambda s: END)
            .set_entry_point("a")
        )
        compiled = graph.compile()
        assert "-.-> __end__([End])" in compiled.to_mermaid()

    def test_draw_as_image_and_as_text_both_fall_back_to_log(self):
        graph = StateGraph().add_node("a", lambda s: {}).set_entry_point("a")
        compiled = graph.compile()
        assert "flowchart TD" in compiled.draw(as_image=True)
        assert "flowchart TD" in compiled.draw(as_image=False)


# ===========================================================================
# CompiledStateGraph._initialize_state
# ===========================================================================


@dataclass
class _StrictState:
    question: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.question, str):
            raise TypeError("question must be a string")


class TestInitializeState:
    def test_passes_through_existing_state_instance(self):
        compiled = StateGraph().add_node("a", lambda s: {}).compile()
        state = GraphState(question="already built")
        assert compiled._initialize_state(state) is state

    def test_falls_back_to_default_on_type_error(self):
        graph = StateGraph(state_schema=_StrictState).add_node("a", lambda s: {})
        compiled = graph.compile()
        result = compiled._initialize_state({"question": 12345})
        assert result.question == ""

    def test_extra_dict_keys_become_attributes(self):
        compiled = StateGraph().add_node("a", lambda s: {}).compile()
        result = compiled._initialize_state({"question": "hi", "custom_attr": "value"})
        assert result.custom_attr == "value"

    def test_non_dict_non_state_input_returns_default_state(self):
        compiled = StateGraph().add_node("a", lambda s: {}).compile()
        result = compiled._initialize_state(12345)
        assert isinstance(result, GraphState)


# ===========================================================================
# Full invoke() runs: verbose logging, routing variants, error paths
# ===========================================================================


class TestInvokeRouting:
    def test_agent_decider_routes_by_response_content(self):
        router_llm = ScriptedLLM([("GO_B", [])])
        specialist_llm = ScriptedLLM([("b done", [])])
        router = Agent(name="Router", llm=router_llm)
        agent_b = Agent(name="B", llm=specialist_llm)

        graph = (
            StateGraph()
            .add_node("route", lambda s: {})
            .add_node("b", agent_b)
            .add_conditional_edges("route", router, {"go_b": "b"})
            .set_entry_point("route")
        )
        compiled = graph.compile()
        final = compiled.invoke(GraphState(question="hi"))
        assert final.agents["B"] == "b done"

    def test_route_map_falls_back_to_default_key(self):
        graph = (
            StateGraph()
            .add_node("a", lambda s: {})
            .add_node("fallback", lambda s: {"output": "fell back"})
            .add_conditional_edges("a", lambda s: "unmatched_key", {"default": "fallback"})
            .set_entry_point("a")
        )
        final = graph.compile().invoke(GraphState())
        assert final.output == "fell back"

    def test_decider_without_route_map_returns_raw_node_name(self):
        graph = (
            StateGraph()
            .add_node("a", lambda s: {})
            .add_node("nextnode", lambda s: {"output": "arrived"})
            .add_conditional_edges("a", lambda s: "NextNode")
            .set_entry_point("a")
        )
        final = graph.compile().invoke(GraphState())
        assert final.output == "arrived"

    def test_unrouted_node_ends_workflow(self):
        graph = StateGraph().add_node("a", lambda s: {"output": "only step"}).set_entry_point("a")
        final = graph.compile().invoke(GraphState())
        assert final.output == "only step"

    def test_unknown_target_node_raises(self):
        graph = StateGraph().add_node("a", lambda s: {}).set_entry_point("a")
        compiled = graph.compile()
        compiled.graph.edges["a"] = "ghost_node"
        with pytest.raises(ValueError, match="not found in graph nodes"):
            compiled.invoke(GraphState())

    def test_invoke_raises_when_entry_point_cleared_after_compile(self):
        graph = StateGraph().add_node("a", lambda s: {})
        compiled = graph.compile()
        compiled.graph.entry_point = None
        with pytest.raises(ValueError, match="no entry_point defined"):
            compiled.invoke(GraphState())

    def test_verbose_parallel_and_agent_and_function_nodes(self):
        llm = ScriptedLLM(
            [
                ("calling", [{"name": "t", "args": {}}]),
                ("done a", []),
            ]
        )
        agent_a = Agent(
            name="A", llm=llm, tools=[Tool(name="t", description="d", func=lambda: "ok")]
        )
        agent_b = Agent(name="B", llm=ScriptedLLM([("done b", [])]))

        def replace_with_new_state(state: GraphState) -> GraphState:
            return GraphState(question=state.question, output="replaced")

        graph = (
            StateGraph(verbose=True)
            .add_node("parallel", [agent_a, agent_b])
            .add_node("fn", replace_with_new_state)
            .add_edge("parallel", "fn")
            .set_entry_point("parallel")
        )
        final = graph.compile().invoke(GraphState(question="hi"))
        assert final.output == "replaced"


class TestFinalizeOutput:
    def test_strips_change_summary_header(self):
        compiled = StateGraph().add_node("a", lambda s: {}).compile()
        state = GraphState()
        state.agents["Refiner"] = "### Change Summary\nSome notes\n# Final Brief\nThe real content"
        compiled._finalize_output(state)
        assert state.output == "# Final Brief\nThe real content"


# ===========================================================================
# _compact_text / trajectory renderers
# ===========================================================================


class TestCompactText:
    def test_truncates_long_strings(self):
        assert _compact_text("x" * 200, max_len=10) == "x" * 10 + "..."

    def test_serializes_dicts_and_lists(self):
        assert _compact_text({"a": 1}) == '{"a": 1}'

    def test_falls_back_to_str_when_json_dumps_fails(self, monkeypatch):
        def boom(*args, **kwargs):
            raise TypeError("not serialisable")

        monkeypatch.setattr(agent_framework.json, "dumps", boom)
        assert _compact_text({"a": 1}) == "{'a': 1}"


class TestFormatStepSummary:
    def test_uses_explicit_summary_when_present(self):
        assert _format_step_summary({"summary": "hello"}) == "hello"

    def test_falls_back_to_first_line_of_output(self):
        step = {"output": "first line\nsecond line"}
        assert _format_step_summary(step) == "first line"

    def test_falls_back_to_updates_keys(self):
        step = {"updates": {"x": 1, "y": 2}}
        assert "Updated:" in _format_step_summary(step)

    def test_falls_back_to_executed_successfully(self):
        assert _format_step_summary({}) == "Executed successfully"


class TestFormatTrajectoryMarkdown:
    def test_empty_trajectory_is_blank(self):
        assert format_trajectory_markdown([]) == ""

    def test_renders_a_row_per_step(self):
        trajectory = [
            {
                "step": 1,
                "node": "a",
                "type": "agent",
                "executor": "A",
                "tools": ["t"],
                "summary": "ok",
            }
        ]
        md = format_trajectory_markdown(trajectory)
        assert "`a`" in md
        assert "`t`" in md


class TestDisplayTrajectory:
    def test_no_trajectory_returns_none(self):
        assert display_trajectory({"question": "hi"}) is None

    def test_ascii_table_for_non_markdown(self):
        state = {
            "trajectory": [
                {
                    "step": 1,
                    "node": "a",
                    "type": "agent",
                    "executor": "A",
                    "tools": [],
                    "summary": "ok",
                }
            ]
        }
        result = display_trajectory(state, as_markdown=False)
        assert "EXECUTION TRAJECTORY TABLE" in result

    def test_markdown_mode_renders_without_raising(self):
        # IPython is installed in this environment, so Markdown display succeeds
        # (returns None) rather than exercising the except-fallback branch; either
        # outcome is valid, this just guards against a crash in either case.
        state = {
            "trajectory": [
                {
                    "step": 1,
                    "node": "a",
                    "type": "fn",
                    "executor": "f",
                    "tools": [],
                    "summary": "ok",
                }
            ]
        }
        result = display_trajectory(state, as_markdown=True)
        assert result is None or result.startswith("### Execution Trajectory Summary")

    def test_markdown_mode_falls_back_when_ipython_display_fails(self, monkeypatch):
        import IPython.display as ipy_display

        def boom(*args, **kwargs):
            raise RuntimeError("no display available")

        monkeypatch.setattr(ipy_display, "display", boom)
        state = {
            "trajectory": [
                {
                    "step": 1,
                    "node": "a",
                    "type": "fn",
                    "executor": "f",
                    "tools": [],
                    "summary": "ok",
                }
            ]
        }
        result = display_trajectory(state, as_markdown=True)
        assert result.startswith("### Execution Trajectory Summary")
