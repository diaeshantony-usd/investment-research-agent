"""
research_agent package
======================
Institutional-grade multi-agent equity research system built with StateGraph and Ollama.
"""

from research_agent.agent_framework import (
    END,
    Agent,
    AgentResponse,
    CompiledStateGraph,
    GraphMemory,
    GraphState,
    StateGraph,
    Tool,
    TraceStep,
    tool,
)
from research_agent.config import (
    CACHE_DIR,
    DATA_DIR,
    LOGS_DIR,
    MEMORY_DIR,
    logger,
    setup_logging,
)
from research_agent.interfaces import (
    LLM,
    LLMInterface,
    OllamaLLM,
    OpenAILLM,
    get_llm,
)
from research_agent.workflows import ResearchSession

__version__ = "0.1.0"

__all__ = [
    "CACHE_DIR",
    "DATA_DIR",
    "END",
    "LLM",
    "LOGS_DIR",
    "MEMORY_DIR",
    "Agent",
    "AgentResponse",
    "CompiledStateGraph",
    "GraphMemory",
    "GraphState",
    "LLMInterface",
    "OllamaLLM",
    "OpenAILLM",
    "ResearchSession",
    "StateGraph",
    "Tool",
    "TraceStep",
    "get_llm",
    "logger",
    "setup_logging",
    "tool",
]
