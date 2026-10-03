"""
test_memory.py
==============
Unit tests for threaded GraphMemory checkpoint persistence and tool endpoints.
"""

from __future__ import annotations

import json

from research_agent.agent_framework import GraphMemory, GraphState


class TestThreadedGraphMemory:
    """Verifies thread-scoped storage, state isolation, pagination, and tool interfaces."""

    def test_write_and_structure(self, temp_memory_file: str):
        memory = GraphMemory(filepath=temp_memory_file)
        state_nvda = GraphState(question="Analyze NVDA", thread_id="nvda_thread")
        uid_nvda = memory.write(state_nvda)

        state_aapl = GraphState(question="Analyze AAPL", thread_id="aapl_thread")
        uid_aapl = memory.write(state_aapl)

        # Verify disk JSON structure
        with open(temp_memory_file, encoding="utf-8") as f:
            disk_data = json.load(f)

        assert "nvda_thread" in disk_data
        assert "states" in disk_data["nvda_thread"]
        assert uid_nvda in disk_data["nvda_thread"]["states"]
        assert disk_data["nvda_thread"]["states"][uid_nvda]["question"] == "Analyze NVDA"

        assert "aapl_thread" in disk_data
        assert uid_aapl in disk_data["aapl_thread"]["states"]

    def test_get_state(self, temp_memory: GraphMemory):
        state = GraphState(question="Explain P/E", thread_id="concept_thread")
        uid = temp_memory.write(state)

        # Scoped retrieval
        res_scoped = json.loads(temp_memory.get_state(uid, thread_id="concept_thread"))
        assert res_scoped["state_unique_id"] == uid
        assert res_scoped["question"] == "Explain P/E"

        # Global lookup across threads
        res_global = json.loads(temp_memory.get_state(uid))
        assert res_global["state_unique_id"] == uid

        # Missing lookup
        res_missing = json.loads(temp_memory.get_state("non_existent_id"))
        assert "error" in res_missing

    def test_get_history_states_pagination(self, temp_memory: GraphMemory):
        for i in range(5):
            st = GraphState(question=f"Q{i}", thread_id="page_thread")
            temp_memory.write(st)

        # Retrieve first page
        page1 = json.loads(temp_memory.get_history_states(page=1, limit=2, thread_id="page_thread"))
        assert page1["page"] == 1
        assert page1["limit"] == 2
        assert page1["total_records"] == 5
        assert page1["total_pages"] == 3
        assert len(page1["records"]) == 2
        assert page1["has_next_page"] is True

    def test_list_threads_and_ids(self, temp_memory: GraphMemory):
        temp_memory.write(GraphState(question="Q1", thread_id="thread_alpha"))
        temp_memory.write(GraphState(question="Q2", thread_id="thread_beta"))

        threads = json.loads(temp_memory.list_threads())
        assert "thread_alpha" in threads
        assert "thread_beta" in threads

        alpha_ids = json.loads(temp_memory.list_state_ids(thread_id="thread_alpha"))
        assert len(alpha_ids) == 1

    def test_as_tools_integration(self, temp_memory: GraphMemory):
        tools = temp_memory.as_tools()
        tool_names = [t.name for t in tools]
        assert "get_history_states" in tool_names
        assert "get_state" in tool_names
        assert "list_state_ids" in tool_names
        assert "list_threads" in tool_names

        # Execute as tool
        list_tool = next(t for t in tools if t.name == "list_threads")
        res = list_tool.func()
        assert isinstance(res, str)
