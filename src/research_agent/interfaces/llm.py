"""
interfaces/llm.py
=================
Abstract LLM interface with multi-provider implementations (Ollama and OpenAI).

Every client returns the same normalised shape from ``chat`` so agents never depend on
a provider SDK directly::

    content, tool_calls, raw_message = llm.chat(messages, tools)
    # tool_calls == [{"name": "get_earnings_data", "args": {"ticker": "NVDA"}}]

Author: N L N Sai Krishna Akula
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

__all__ = ["LLM", "LLMInterface", "OllamaLLM", "OpenAILLM", "get_llm"]


class LLM(ABC):
    """Abstract interface defining the standardized contract for LLM backends."""

    def _extract_tool_callables(self, tools: dict[str, Any]) -> Any:
        """Convert ``Tool`` objects into the provider's tool format.

        The default returns plain callables (what the Ollama SDK expects); providers
        that need JSON schemas override this.
        """
        callables: list[Callable[..., Any]] = []
        for t in tools.values():
            if hasattr(t, "to_callable"):
                callables.append(t.to_callable())
            elif callable(t):
                callables.append(t)
        return callables

    def get_tool_callables(self, tools: dict[str, Any]) -> Any:
        """Return ``tools`` in the provider's format (public wrapper)."""
        return self._extract_tool_callables(tools)

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]], Any]:
        """Send the conversation and optional tools to the model.

        Args:
            messages: Chat history in OpenAI-style ``{"role", "content"}`` format.
            tools: Tools the model may call, keyed by name.

        Returns:
            tuple containing:
            - content: Generated text string.
            - tool_calls: Parsed tool calls [{'name': str, 'args': dict}].
            - raw_message: Underlying message object from provider SDK.
        """


# Backward-compatible alias for code written against the earlier class name.
LLMInterface = LLM


class OllamaLLM(LLM):
    """Concrete LLM client connecting to Ollama Cloud or local Ollama instances."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Create an Ollama client.

        Args:
            model: Model name. Defaults to ``DEFAULT_MODEL``.
            host: Ollama server URL. Defaults to ``OLLAMA_HOST``.
            api_key: Bearer token for Ollama Cloud. Defaults to ``OLLAMA_API_KEY``;
                not needed for a local Ollama server.
        """
        self.model = model or DEFAULT_MODEL
        self.host = host or DEFAULT_OLLAMA_HOST
        self.api_key = api_key or DEFAULT_API_KEY

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.client = ollama.Client(host=self.host, headers=headers)
        logger.info("Initialized OllamaLLM client [model=%s, host=%s]", self.model, self.host)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]], Any]:
        """Run one chat completion against Ollama.

        Raises:
            Exception: Any SDK or network error, after logging it.
        """
        callables = self._extract_tool_callables(tools) if tools else None

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if callables:
            kwargs["tools"] = callables

        try:
            response = self.client.chat(**kwargs)

        except Exception:
            logger.exception("Error communicating with Ollama model '%s'", self.model)
            raise

        msg = response.message
        content = msg.content or ""
        raw_calls = getattr(msg, "tool_calls", None) or []
        parsed_tool_calls = [
            {"name": call.function.name, "args": call.function.arguments or {}}
            for call in raw_calls
        ]
        return content, parsed_tool_calls, msg


class OpenAILLM(LLM):
    """Concrete LLM client connecting to OpenAI or compatible endpoints."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Create an OpenAI (or OpenAI-compatible) client.

        Args:
            model: Model name. Defaults to ``DEFAULT_OPENAI_MODEL``.
            api_key: API key. Defaults to ``OPENAI_API_KEY``.
            base_url: Alternative endpoint for OpenAI-compatible servers.
        """
        self.model = model or DEFAULT_OPENAI_MODEL
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url or OPENAI_BASE_URL

        client_kwargs: dict[str, Any] = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = openai.OpenAI(**client_kwargs)
        logger.info(
            "Initialized OpenAILLM client [model=%s, base_url=%s]", self.model, self.base_url
        )

    def _extract_tool_callables(self, tools: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert tools into OpenAI function-calling JSON schemas."""
        openai_tools: list[dict[str, Any]] = []
        for tool_name, tool_obj in tools.items():
            func = getattr(tool_obj, "func", tool_obj)
            desc = getattr(tool_obj, "description", "") or (
                func.__doc__ if hasattr(func, "__doc__") else ""
            )

            # Use the tool's explicit JSON schema when present; otherwise declare an
            # argument-free function.
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
        """Run one chat completion against the OpenAI API.

        Tool-call arguments arrive as a JSON string and are decoded; malformed JSON is
        passed through as ``{"raw_input": ...}`` so the agent can still see it.

        Raises:
            Exception: Any SDK or network error, after logging it.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = self._extract_tool_callables(tools)

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception:
            logger.exception("Error communicating with OpenAI model '%s'", self.model)
            raise

        msg = response.choices[0].message
        content = msg.content or ""
        raw_calls = getattr(msg, "tool_calls", None) or []
        parsed_tool_calls = [
            {"name": call.function.name, "args": _decode_tool_arguments(call.function)}
            for call in raw_calls
        ]
        return content, parsed_tool_calls, msg


def _decode_tool_arguments(function: Any) -> dict[str, Any]:
    """Return an OpenAI tool call's arguments as a dict, tolerating malformed JSON."""
    raw_args = getattr(function, "arguments", "")
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {"raw_input": raw_args}
    return {}


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    host: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLM:
    """Create the LLM client for ``provider`` (default: ``DEFAULT_LLM_PROVIDER``).

    Args:
        provider: ``"ollama"`` or ``"openai"`` (case-insensitive).
        model: Model name override.
        host: Ollama host override (ignored for OpenAI).
        api_key: API key override.
        base_url: OpenAI-compatible endpoint override (ignored for Ollama).

    Returns:
        A ready-to-use ``LLM`` client.

    Raises:
        ValueError: If the provider is not supported.
    """
    selected_provider = (provider or DEFAULT_LLM_PROVIDER).lower()

    if selected_provider == "openai":
        return OpenAILLM(model=model, api_key=api_key, base_url=base_url)
    if selected_provider == "ollama":
        return OllamaLLM(model=model, host=host, api_key=api_key)
    raise ValueError(
        f"Unsupported LLM provider '{selected_provider}'. Choose 'ollama' or 'openai'."
    )
