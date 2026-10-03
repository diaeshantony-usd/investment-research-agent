"""
interfaces/llm.py
=================
Abstract LLM interface with multi-provider implementations (Ollama and OpenAI).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import ollama
import openai

from research_agent.config import (
    DEFAULT_API_KEY,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OPENAI_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    logger,
)


class LLM(ABC):
    """Abstract interface defining the standardized contract for LLM backends."""

    def _extract_tool_callables(self, tools: dict[str, Any]) -> Any:
        """Transforms Tool objects into the LLM-specific tool callable or schema format."""
        callables = []
        for t in tools.values():
            if hasattr(t, "to_callable"):
                callables.append(t.to_callable())
            elif callable(t):
                callables.append(t)
        return callables

    def get_tool_callables(self, tools: dict[str, Any]) -> Any:
        """Public alias for extracting tool callables/schemas."""
        return self._extract_tool_callables(tools)

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]], Any]:
        """Sends conversation history and optional tools to the model.

        Returns:
            tuple containing:
            - content: Generated text string.
            - tool_calls: Parsed tool calls [{'name': str, 'args': dict}].
            - raw_message: Underlying message object from provider SDK.
        """


# Backward compatibility alias
LLMInterface = LLM


class OllamaLLM(LLM):
    """Concrete LLM client connecting to Ollama Cloud or local Ollama instances."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model or DEFAULT_MODEL
        self.host = host or DEFAULT_OLLAMA_HOST
        self.api_key = api_key or DEFAULT_API_KEY

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.client = ollama.Client(host=self.host, headers=headers)
        logger.info(f"Initialized OllamaLLM client [model={self.model}, host={self.host}]")

    def _extract_tool_callables(self, tools: dict[str, Any]) -> list[Callable]:
        """Extracts callables formatted for the Ollama SDK tool-calling interface."""
        callables = []
        for t in tools.values():
            if hasattr(t, "to_callable"):
                callables.append(t.to_callable())
            elif callable(t):
                callables.append(t)
        return callables

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]], Any]:
        """Executes chat completion with Ollama."""
        callables = self._extract_tool_callables(tools) if tools else None

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if callables:
            kwargs["tools"] = callables

        try:
            response = self.client.chat(**kwargs)
            msg = response.message

            content = msg.content or ""
            parsed_tool_calls: list[dict[str, Any]] = []
            raw_calls = getattr(msg, "tool_calls", None) or []

            for call in raw_calls:
                parsed_tool_calls.append(
                    {
                        "name": call.function.name,
                        "args": call.function.arguments or {},
                    }
                )

            return content, parsed_tool_calls, msg

        except Exception as exc:
            logger.error(
                f"Error communicating with Ollama model '{self.model}': {exc}",
                exc_info=True,
            )
            raise


class OpenAILLM(LLM):
    """Concrete LLM client connecting to OpenAI or compatible endpoints."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or DEFAULT_OPENAI_MODEL
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url or OPENAI_BASE_URL

        client_kwargs: dict[str, Any] = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = openai.OpenAI(**client_kwargs)
        logger.info(f"Initialized OpenAILLM client [model={self.model}, base_url={self.base_url}]")

    def _extract_tool_callables(self, tools: dict[str, Any]) -> list[dict[str, Any]]:
        """Converts tool objects into standard OpenAI function calling JSON schemas."""
        openai_tools = []
        for tool_name, tool_obj in tools.items():
            func = getattr(tool_obj, "func", tool_obj)
            desc = getattr(tool_obj, "description", "") or (
                func.__doc__ if hasattr(func, "__doc__") else ""
            )

            # If tool has explicit parameters schema
            params = getattr(tool_obj, "parameters", None)
            if not params:
                params = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": desc or f"Execute {tool_name}",
                        "parameters": params,
                    },
                }
            )
        return openai_tools

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]], Any]:
        """Executes chat completion with OpenAI."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = self._extract_tool_callables(tools)

        try:
            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message

            content = msg.content or ""
            parsed_tool_calls: list[dict[str, Any]] = []

            raw_calls = getattr(msg, "tool_calls", None) or []
            for call in raw_calls:
                fn_args: dict[str, Any] = {}
                raw_args = getattr(call.function, "arguments", "")
                if isinstance(raw_args, str) and raw_args.strip():
                    try:
                        fn_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        fn_args = {"raw_input": raw_args}
                elif isinstance(raw_args, dict):
                    fn_args = raw_args

                parsed_tool_calls.append(
                    {
                        "name": call.function.name,
                        "args": fn_args,
                    }
                )

            return content, parsed_tool_calls, msg

        except Exception as exc:
            logger.error(
                f"Error communicating with OpenAI model '{self.model}': {exc}",
                exc_info=True,
            )
            raise


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    host: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLM:
    """Factory helper to instantiate the configured LLM client.

    Supports providers: 'ollama' and 'openai'.
    """
    selected_provider = (provider or DEFAULT_LLM_PROVIDER).lower()

    if selected_provider == "openai":
        return OpenAILLM(model=model, api_key=api_key, base_url=base_url)
    elif selected_provider == "ollama":
        return OllamaLLM(model=model, host=host, api_key=api_key)
    else:
        raise ValueError(
            f"Unsupported LLM provider '{selected_provider}'. Choose 'ollama' or 'openai'."
        )
