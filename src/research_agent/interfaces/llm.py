"""
interfaces/llm.py
=================
Abstract LLM interface with multi-provider implementations (Ollama and OpenAI), plus a
fallback client that retries a failed Ollama request on Groq.

Every client returns the same normalised shape from ``chat`` so agents never depend on
a provider SDK directly::

    content, tool_calls, raw_message = llm.chat(messages, tools)
    # tool_calls == [{"name": "get_earnings_data", "args": {"ticker": "NVDA"}}]

Author: N L N Sai Krishna Akula
"""

from __future__ import annotations

import inspect
import json
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import ollama
import openai
from ollama._utils import convert_function_to_tool

from research_agent.config import (
    DEFAULT_API_KEY,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OPENAI_MODEL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    LLM_BACKUP_MAX_RETRIES,
    LLM_FALLBACK_COOLDOWN_SECONDS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    logger,
)

__all__ = ["LLM", "FallbackLLM", "LLMInterface", "OllamaLLM", "OpenAILLM", "get_llm"]


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


# --------------------------------------------------------------------------- messages
# Agents keep each provider's raw reply message in their history (an Ollama ``Message``, an
# OpenAI ``ChatCompletionMessage`` or a plain dict). When a request moves to another
# provider, the history is converted to the format that provider accepts.


def _field(message: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a dict or an SDK message object."""
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def _tool_calls(message: Any) -> list[tuple[str | None, str, Any]]:
    """``(call id, function name, arguments)`` for each tool call in a message."""
    calls = []
    for call in _field(message, "tool_calls") or []:
        function = _field(call, "function")
        calls.append(
            (_field(call, "id"), _field(function, "name", ""), _field(function, "arguments"))
        )
    return calls


def _to_openai_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert a mixed chat history to OpenAI chat-completions messages.

    OpenAI-compatible APIs require every ``tool`` message to name the ``tool_call_id`` it
    answers, and every tool call to be answered. The agent loop appends tool results in
    call order without ids, so ids are assigned (or reused) and matched up here.
    """
    out: list[dict[str, Any]] = []
    pending: list[str] = []  # ids of tool calls still waiting for their result

    def close_pending() -> None:
        while pending:
            out.append({"role": "tool", "tool_call_id": pending.pop(0), "content": "(no result)"})

    for message in messages:
        role = _field(message, "role", "user")
        content = _field(message, "content") or ""
        if role == "tool":
            call_id = _field(message, "tool_call_id") or (pending.pop(0) if pending else None)
            if call_id is None:  # a result without a call: keep the text as context
                out.append({"role": "user", "content": f"Tool result: {content}"})
            else:
                if call_id in pending:
                    pending.remove(call_id)
                out.append({"role": "tool", "tool_call_id": call_id, "content": str(content)})
            continue
        close_pending()
        entry: dict[str, Any] = {"role": role, "content": content}
        calls = _tool_calls(message) if role == "assistant" else []
        if calls:
            entry["tool_calls"] = []
            for i, (given_id, name, args) in enumerate(calls):
                call_id = given_id or f"call_{len(out)}_{i}"
                arguments = args if isinstance(args, str) else json.dumps(args or {})
                entry["tool_calls"].append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
                pending.append(call_id)
        out.append(entry)
    close_pending()
    return out


def _to_ollama_messages(messages: list[Any]) -> list[Any]:
    """Convert OpenAI reply objects in a history to dicts the Ollama SDK accepts.

    Dicts and Ollama ``Message`` objects are passed through unchanged.
    """
    out: list[Any] = []
    for message in messages:
        if isinstance(message, dict | ollama.Message):
            out.append(message)
            continue
        entry: dict[str, Any] = {
            "role": _field(message, "role", "assistant"),
            "content": _field(message, "content") or "",
        }
        calls = _tool_calls(message)
        if calls:
            entry["tool_calls"] = [
                {"function": {"name": name, "arguments": _decode_arguments(args)}}
                for _, name, args in calls
            ]
        out.append(entry)
    return out


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
            "messages": _to_ollama_messages(messages),
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
        max_retries: int | None = None,
    ) -> None:
        """Create an OpenAI (or OpenAI-compatible) client.

        Args:
            model: Model name. Defaults to ``DEFAULT_OPENAI_MODEL``.
            api_key: API key. Defaults to ``OPENAI_API_KEY``.
            base_url: Alternative endpoint for OpenAI-compatible servers.
            max_retries: Retries for rate-limited or failed requests (SDK default: 2).
        """
        self.model = model or DEFAULT_OPENAI_MODEL
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url or OPENAI_BASE_URL

        client_kwargs: dict[str, Any] = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries

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

            # Use the tool's explicit JSON schema when present; otherwise derive it from
            # the function signature, so the model sees the real argument names.
            params = getattr(tool_obj, "parameters", None) or _schema_from_signature(func)

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
            "messages": _to_openai_messages(messages),
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


def _schema_from_signature(func: Any) -> dict[str, Any]:
    """JSON schema for a function's arguments, built from its signature and type hints.

    Arguments with a default value are optional. A function whose signature cannot be read
    is declared without arguments.
    """
    empty: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    if not callable(func):
        return empty
    try:
        schema = convert_function_to_tool(func).model_dump(exclude_none=True)
        params = schema["function"]["parameters"]
        signature = inspect.signature(func)
    except Exception:  # noqa: BLE001 - an unreadable signature just means "no arguments"
        return empty
    params["required"] = [
        name
        for name in params.get("required", [])
        if name in signature.parameters
        and signature.parameters[name].default is inspect.Parameter.empty
    ]
    return params


def _decode_tool_arguments(function: Any) -> dict[str, Any]:
    """Return an OpenAI tool call's arguments as a dict, tolerating malformed JSON."""
    return _decode_arguments(getattr(function, "arguments", ""))


def _decode_arguments(raw_args: Any) -> dict[str, Any]:
    """Decode tool-call arguments given as a dict or a JSON string."""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {"raw_input": raw_args}
    return {}


class FallbackLLM(LLM):
    """Send each request to ``primary``; if it fails, retry it on ``backup``.

    After a failure the backup is used for ``cooldown`` seconds before the primary is tried
    again, so a provider that is over its usage limit is not called (and waited on) for
    every request. If the backup fails too, its error is raised.
    """

    def __init__(self, primary: LLM, backup: LLM, cooldown: float | None = None) -> None:
        self.primary = primary
        self.backup = backup
        self.cooldown = LLM_FALLBACK_COOLDOWN_SECONDS if cooldown is None else cooldown
        self.model = getattr(primary, "model", None)
        self._primary_down_until = 0.0
        self._lock = threading.Lock()

    def _primary_available(self) -> bool:
        with self._lock:
            return time.monotonic() >= self._primary_down_until

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]], Any]:
        """Run the request on the primary, or on the backup when the primary fails."""
        if self._primary_available():
            try:
                return self.primary.chat(messages, tools)
            except Exception as exc:  # noqa: BLE001 - any provider error moves to the backup
                with self._lock:
                    self._primary_down_until = time.monotonic() + self.cooldown
                logger.warning(
                    "Primary LLM '%s' failed (%s); switching to backup '%s' for %ss",
                    getattr(self.primary, "model", "?"),
                    _short_error(exc),
                    getattr(self.backup, "model", "?"),
                    self.cooldown,
                )
        return self.backup.chat(messages, tools)


def _short_error(exc: Exception, limit: int = 160) -> str:
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _groq_model_for(model: str | None) -> str:
    """The Groq name of an Ollama model: ``gpt-oss:120b`` -> ``openai/gpt-oss-120b``."""
    if model and model.startswith("gpt-oss:"):
        return "openai/gpt-oss-" + model.split(":", 1)[1]
    return GROQ_MODEL


def _groq_llm(model: str | None = None) -> OpenAILLM:
    return OpenAILLM(
        model=_groq_model_for(model),
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        max_retries=LLM_BACKUP_MAX_RETRIES,
    )


def _backup_llm(model: str | None) -> LLM | None:
    """The Groq backup client, or None when no ``GROQ_API_KEY`` is configured."""
    return _groq_llm(model) if GROQ_API_KEY else None


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    host: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLM:
    """Create the LLM client for ``provider`` (default: ``DEFAULT_LLM_PROVIDER``).

    Args:
        provider: ``"ollama"``, ``"openai"`` or ``"groq"`` (case-insensitive). When it is
            not given, ``DEFAULT_LLM_PROVIDER`` is used and, if that is Ollama and a
            ``GROQ_API_KEY`` is configured, a failed request is retried on Groq.
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
    if selected_provider == "groq":
        return _groq_llm(model)
    if selected_provider == "ollama":
        client = OllamaLLM(model=model, host=host, api_key=api_key)
        # The automatic backup applies to the default provider only; an explicitly
        # requested provider is used as is.
        backup = _backup_llm(model) if provider is None else None
        return FallbackLLM(client, backup) if backup else client
    raise ValueError(
        f"Unsupported LLM provider '{selected_provider}'. Choose 'ollama', 'openai' or 'groq'."
    )
