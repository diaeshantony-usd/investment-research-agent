"""
interfaces package
==================
Exposes LLM interfaces and multi-provider clients (Ollama and OpenAI).
"""

from research_agent.interfaces.llm import (
    LLM,
    LLMInterface,
    OllamaLLM,
    OpenAILLM,
    get_llm,
)

__all__ = [
    "LLM",
    "LLMInterface",
    "OllamaLLM",
    "OpenAILLM",
    "get_llm",
]
