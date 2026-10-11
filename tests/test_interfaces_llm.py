"""
test_interfaces_llm.py
=======================
Offline tests for the LLM provider clients. Each underlying SDK client (``ollama.Client``,
``openai.OpenAI``) is replaced with a fake, so these tests check request shaping and
response normalisation without network access.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from research_agent.interfaces import llm as llm_module
from research_agent.interfaces.llm import LLM, OllamaLLM, OpenAILLM, get_llm


class DummyTool:
    """Minimal stand-in for ``agent_framework.Tool``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"Does {name}"
        self.parameters = {"type": "object", "properties": {}, "required": []}

    def to_callable(self):
        def _callable():
            return "ok"

        _callable.__name__ = self.name
        return _callable


class TestExtractToolCallables:
    def test_uses_to_callable_when_present(self):
        class Probe(LLM):
            def chat(self, messages, tools=None):
                raise NotImplementedError

        callables = Probe()._extract_tool_callables({"t": DummyTool("t")})
        assert len(callables) == 1
        assert callables[0]() == "ok"

    def test_plain_callables_pass_through(self):
        class Probe(LLM):
            def chat(self, messages, tools=None):
                raise NotImplementedError

        def plain():
            return "plain"

        callables = Probe()._extract_tool_callables({"p": plain})
        assert callables == [plain]

    def test_skips_non_callable_non_tool_values(self):
        class Probe(LLM):
            def chat(self, messages, tools=None):
                raise NotImplementedError

        assert Probe()._extract_tool_callables({"x": 42}) == []

    def test_get_tool_callables_is_the_public_wrapper(self):
        class Probe(LLM):
            def chat(self, messages, tools=None):
                raise NotImplementedError

        assert Probe().get_tool_callables({}) == []


class FakeFunction:
    def __init__(self, name: str, arguments: Any = None) -> None:
        self.name = name
        self.arguments = arguments


class FakeOllamaCall:
    def __init__(self, name: str, arguments: dict) -> None:
        self.function = FakeFunction(name, arguments)


class FakeOllamaClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.last_kwargs: dict | None = None

    def chat(self, **kwargs):
        self.last_kwargs = kwargs
        if self._error:
            raise self._error
        return self._response


class TestOllamaLLM:
    def test_init_sets_up_client_and_defaults(self, monkeypatch):
        created = {}

        class FakeClientCtor:
            def __init__(self, host, headers):
                created["host"] = host
                created["headers"] = headers

        monkeypatch.setattr(llm_module.ollama, "Client", FakeClientCtor)
        client = OllamaLLM(model="my-model", host="http://x", api_key="key123")
        assert client.model == "my-model"
        assert created["host"] == "http://x"
        assert created["headers"] == {"Authorization": "Bearer key123"}

    def test_chat_without_tools(self, monkeypatch):
        fake = FakeOllamaClient(
            response=SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))
        )
        monkeypatch.setattr(llm_module.ollama, "Client", lambda host, headers: fake)
        client = OllamaLLM(model="m", api_key="k")
        content, calls, _raw = client.chat([{"role": "user", "content": "hello"}])
        assert content == "hi"
        assert calls == []
        assert "tools" not in fake.last_kwargs

    def test_chat_with_tools_parses_tool_calls(self, monkeypatch):
        message = SimpleNamespace(
            content="",
            tool_calls=[FakeOllamaCall("get_earnings_data", {"ticker": "NVDA"})],
        )
        fake = FakeOllamaClient(response=SimpleNamespace(message=message))
        monkeypatch.setattr(llm_module.ollama, "Client", lambda host, headers: fake)
        client = OllamaLLM(model="m", api_key="k")
        _content, calls, _raw = client.chat(
            [{"role": "user", "content": "earnings?"}], tools={"t": DummyTool("get_earnings_data")}
        )
        assert calls == [{"name": "get_earnings_data", "args": {"ticker": "NVDA"}}]
        assert "tools" in fake.last_kwargs

    def test_chat_logs_and_reraises_on_sdk_error(self, monkeypatch):
        fake = FakeOllamaClient(error=RuntimeError("boom"))
        monkeypatch.setattr(llm_module.ollama, "Client", lambda host, headers: fake)
        client = OllamaLLM(model="m", api_key="k")
        with pytest.raises(RuntimeError, match="boom"):
            client.chat([{"role": "user", "content": "hi"}])


class FakeOpenAIFunction:
    def __init__(self, name: str, arguments) -> None:
        self.name = name
        self.arguments = arguments


class FakeOpenAIToolCall:
    def __init__(self, name: str, arguments) -> None:
        self.function = FakeOpenAIFunction(name, arguments)


class FakeOpenAIClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.last_kwargs: dict | None = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._error:
            raise self._error
        return self._response


class TestOpenAILLM:
    def test_init_passes_api_key_and_base_url(self, monkeypatch):
        created = {}

        class FakeOpenAICtor:
            def __init__(self, **kwargs):
                created.update(kwargs)

        monkeypatch.setattr(llm_module.openai, "OpenAI", FakeOpenAICtor)
        OpenAILLM(model="gpt", api_key="sk-1", base_url="http://local")
        assert created == {"api_key": "sk-1", "base_url": "http://local"}

    def test_extract_tool_callables_builds_json_schema(self):
        client = OpenAILLM.__new__(OpenAILLM)  # skip __init__'s client construction
        schemas = client._extract_tool_callables(
            {"get_earnings_data": DummyTool("get_earnings_data")}
        )
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "get_earnings_data"
        assert schemas[0]["function"]["parameters"]["type"] == "object"

    def test_extract_tool_callables_defaults_params_when_missing(self):
        bare = SimpleNamespace(description="", func=lambda: None)
        client = OpenAILLM.__new__(OpenAILLM)
        schemas = client._extract_tool_callables({"bare": bare})
        assert schemas[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def test_chat_parses_json_tool_arguments(self, monkeypatch):
        message = SimpleNamespace(
            content="",
            tool_calls=[FakeOpenAIToolCall("get_earnings_data", '{"ticker": "NVDA"}')],
        )
        fake = FakeOpenAIClient(
            response=SimpleNamespace(choices=[SimpleNamespace(message=message)])
        )
        monkeypatch.setattr(llm_module.openai, "OpenAI", lambda **kwargs: fake)
        client = OpenAILLM(model="gpt", api_key="sk-1")
        _content, calls, _raw = client.chat([{"role": "user", "content": "hi"}])
        assert calls == [{"name": "get_earnings_data", "args": {"ticker": "NVDA"}}]

    def test_chat_without_tools_omits_tools_kwarg(self, monkeypatch):
        message = SimpleNamespace(content="hi", tool_calls=None)
        fake = FakeOpenAIClient(
            response=SimpleNamespace(choices=[SimpleNamespace(message=message)])
        )
        monkeypatch.setattr(llm_module.openai, "OpenAI", lambda **kwargs: fake)
        client = OpenAILLM(model="gpt", api_key="sk-1")
        content, calls, _ = client.chat([{"role": "user", "content": "hi"}])
        assert content == "hi"
        assert calls == []
        assert "tools" not in fake.last_kwargs

    def test_chat_logs_and_reraises_on_sdk_error(self, monkeypatch):
        fake = FakeOpenAIClient(error=RuntimeError("down"))
        monkeypatch.setattr(llm_module.openai, "OpenAI", lambda **kwargs: fake)
        client = OpenAILLM(model="gpt", api_key="sk-1")
        with pytest.raises(RuntimeError, match="down"):
            client.chat([{"role": "user", "content": "hi"}])


class TestDecodeToolArguments:
    def test_dict_passes_through(self):
        assert llm_module._decode_tool_arguments(FakeOpenAIFunction("x", {"a": 1})) == {"a": 1}

    def test_valid_json_string_is_parsed(self):
        assert llm_module._decode_tool_arguments(FakeOpenAIFunction("x", '{"a": 1}')) == {"a": 1}

    def test_malformed_json_falls_back_to_raw_input(self):
        result = llm_module._decode_tool_arguments(FakeOpenAIFunction("x", "{not json"))
        assert result == {"raw_input": "{not json"}

    def test_blank_or_missing_arguments_return_empty_dict(self):
        assert llm_module._decode_tool_arguments(FakeOpenAIFunction("x", "")) == {}
        assert llm_module._decode_tool_arguments(SimpleNamespace()) == {}


class TestGetLlm:
    def test_defaults_to_ollama(self, monkeypatch):
        monkeypatch.setattr(llm_module.ollama, "Client", lambda host, headers: object())
        client = get_llm(provider="ollama", model="m", api_key="k")
        assert isinstance(client, OllamaLLM)

    def test_openai_provider(self, monkeypatch):
        monkeypatch.setattr(llm_module.openai, "OpenAI", lambda **kwargs: object())
        client = get_llm(provider="OpenAI", model="gpt", api_key="sk-1")
        assert isinstance(client, OpenAILLM)

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm(provider="not-a-provider")
