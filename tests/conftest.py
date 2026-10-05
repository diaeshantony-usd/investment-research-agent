"""
conftest.py
===========
Shared test fixtures and deterministic MockLLM for unit testing research agents and workflows.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from research_agent.agent_framework import GraphMemory, Tool
from research_agent.interfaces import LLM


class MockLLM(LLM):
    """Deterministic, fast mock LLM for offline testing without calling live model servers."""

    def __init__(self, model: str = "mock-model"):
        self.model = model
        self.responses: list[tuple[str, list[dict[str, Any]]]] = []
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
        return f"Mock response for input: {last_msg[:60]}", [], None

    def _extract_tool_callables(self, tools: dict[str, Tool] | None) -> list[Callable]:
        if not tools:
            return []
        return [t.func for t in tools.values()]


@pytest.fixture
def mock_llm() -> MockLLM:
    """Provides a fresh MockLLM instance."""
    return MockLLM()


@pytest.fixture
def temp_memory_file():
    """Provide an isolated temporary JSON file path.

    Uses ``tempfile`` rather than ``tmp_path`` to avoid Windows temp-permission issues.
    """
    td = tempfile.mkdtemp(prefix="agent_mem_test_")
    filepath = Path(td) / "test_memory.json"
    yield str(filepath)
    shutil.rmtree(td, ignore_errors=True)


@pytest.fixture
def temp_memory(temp_memory_file: str) -> GraphMemory:
    """Provides an isolated GraphMemory instance with automatic cleanup."""
    return GraphMemory(filepath=temp_memory_file)
