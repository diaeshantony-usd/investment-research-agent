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

import ollama
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

    def test_schema_is_built_from_the_function_signature(self):
        def delegate(task_description: str, top_k: int = 5) -> str:
            """Delegate a task."""
            return task_description

        client = OpenAILLM.__new__(OpenAILLM)
        params = client._extract_tool_callables({"delegate": delegate})[0]["function"]["parameters"]
        assert params["properties"]["task_description"]["type"] == "string"
        assert params["properties"]["top_k"]["type"] == "integer"
        assert params["required"] == ["task_description"]  # top_k has a default

    def test_agent_delegation_tool_exposes_task_description(self):
        from research_agent.agent_framework import Agent

        agent = Agent.__new__(Agent)
        agent.name, agent.role = "EarningsAnalyst", "earnings"
        tool = agent.as_tool()
        client = OpenAILLM.__new__(OpenAILLM)
        params = client._extract_tool_callables({tool.name: tool})[0]["function"]["parameters"]
        assert params["required"] == ["task_description"]

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


# --------------------------------------------------------------------------- Groq backup


class TestMessageConversion:
    def _history(self):
        assistant = ollama.Message(
            role="assistant",
            content="",
            tool_calls=[
                ollama.Message.ToolCall(
                    function=ollama.Message.ToolCall.Function(
                        name="get_price", arguments={"ticker": "NVDA"}
                    )
                ),
                ollama.Message.ToolCall(
                    function=ollama.Message.ToolCall.Function(name="get_news", arguments={})
                ),
            ],
        )
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Analyse NVDA"},
            assistant,
            {"role": "tool", "content": "182.4"},
            {"role": "tool", "content": "headlines"},
        ]

    def test_ollama_history_becomes_valid_openai_messages(self):
        out = llm_module._to_openai_messages(self._history())
        calls = out[2]["tool_calls"]
        assert [c["function"]["name"] for c in calls] == ["get_price", "get_news"]
        assert calls[0]["function"]["arguments"] == '{"ticker": "NVDA"}'
        # each tool result answers the call it belongs to, in order
        assert out[3] == {"role": "tool", "tool_call_id": calls[0]["id"], "content": "182.4"}
        assert out[4]["tool_call_id"] == calls[1]["id"]

    def test_unanswered_calls_get_a_placeholder_result(self):
        history = [*self._history()[:4], {"role": "user", "content": "go on"}]
        out = llm_module._to_openai_messages(history)
        assert out[4]["role"] == "tool"
        assert out[4]["content"] == "(no result)"
        assert out[5] == {"role": "user", "content": "go on"}

    def test_tool_result_without_a_call_is_kept_as_text(self):
        out = llm_module._to_openai_messages([{"role": "tool", "content": "x"}])
        assert out == [{"role": "user", "content": "Tool result: x"}]

    def test_openai_reply_objects_become_ollama_dicts(self):
        reply = SimpleNamespace(
            role="assistant",
            content="",
            tool_calls=[
                SimpleNamespace(
                    id="c1", function=SimpleNamespace(name="get_price", arguments='{"t": 1}')
                )
            ],
        )
        plain = {"role": "user", "content": "hi"}
        out = llm_module._to_ollama_messages([plain, reply])
        assert out[0] is plain
        assert out[1] == {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_price", "arguments": {"t": 1}}}],
        }

    def test_openai_client_sends_converted_messages(self, monkeypatch):
        fake = FakeOpenAIClient(
            response=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))]
            )
        )
        monkeypatch.setattr(llm_module.openai, "OpenAI", lambda **_: fake)
        OpenAILLM(model="m", api_key="k").chat(self._history())
        sent = fake.last_kwargs["messages"]
        assert sent[3]["tool_call_id"] == sent[2]["tool_calls"][0]["id"]


class ScriptedLLM(LLM):
    def __init__(self, model, error=None):
        self.model = model
        self.error = error
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.error:
            raise self.error
        return f"from {self.model}", [], None


class TestFallbackLLM:
    def test_uses_primary_while_it_works(self):
        primary, backup = ScriptedLLM("ollama"), ScriptedLLM("groq")
        client = llm_module.FallbackLLM(primary, backup, cooldown=60)
        assert client.chat([])[0] == "from ollama"
        assert backup.calls == 0

    def test_failure_moves_the_request_to_the_backup_and_stays_there(self):
        primary = ScriptedLLM("ollama", error=RuntimeError("429 usage limit"))
        backup = ScriptedLLM("groq")
        client = llm_module.FallbackLLM(primary, backup, cooldown=60)
        assert client.chat([])[0] == "from groq"
        assert client.chat([])[0] == "from groq"
        assert primary.calls == 1  # not retried during the cooldown

    def test_primary_is_tried_again_after_the_cooldown(self):
        primary = ScriptedLLM("ollama", error=RuntimeError("down"))
        client = llm_module.FallbackLLM(primary, ScriptedLLM("groq"), cooldown=0)
        client.chat([])
        primary.error = None
        assert client.chat([])[0] == "from ollama"

    def test_backup_error_is_raised_when_both_fail(self):
        client = llm_module.FallbackLLM(
            ScriptedLLM("ollama", error=RuntimeError("a")),
            ScriptedLLM("groq", error=RuntimeError("b")),
        )
        with pytest.raises(RuntimeError, match="b"):
            client.chat([])


class TestGroqWiring:
    def test_model_names_map_to_groq(self):
        assert llm_module._groq_model_for("gpt-oss:120b") == "openai/gpt-oss-120b"
        assert llm_module._groq_model_for("gpt-oss:20b") == "openai/gpt-oss-20b"
        assert llm_module._groq_model_for("llama3") == llm_module.GROQ_MODEL

    def test_default_provider_gets_the_groq_backup_when_a_key_is_set(self, monkeypatch):
        monkeypatch.setattr(llm_module, "GROQ_API_KEY", "gsk-test")
        monkeypatch.setattr(llm_module, "DEFAULT_LLM_PROVIDER", "ollama")
        monkeypatch.setattr(llm_module.ollama, "Client", lambda host, headers: object())
        monkeypatch.setattr(llm_module.openai, "OpenAI", lambda **_: object())
        client = get_llm(model="gpt-oss:120b")
        assert isinstance(client, llm_module.FallbackLLM)
        assert isinstance(client.primary, OllamaLLM)
        assert client.backup.model == "openai/gpt-oss-120b"
        assert client.backup.base_url == llm_module.GROQ_BASE_URL

    def test_no_backup_without_a_key_or_for_an_explicit_provider(self, monkeypatch):
        monkeypatch.setattr(llm_module, "DEFAULT_LLM_PROVIDER", "ollama")
        monkeypatch.setattr(llm_module.ollama, "Client", lambda host, headers: object())
        monkeypatch.setattr(llm_module, "GROQ_API_KEY", "")
        assert isinstance(get_llm(), OllamaLLM)
        monkeypatch.setattr(llm_module, "GROQ_API_KEY", "gsk-test")
        assert isinstance(get_llm(provider="ollama"), OllamaLLM)

    def test_groq_provider(self, monkeypatch):
        monkeypatch.setattr(llm_module.openai, "OpenAI", lambda **_: object())
        client = get_llm(provider="groq")
        assert isinstance(client, OpenAILLM)
        assert client.base_url == llm_module.GROQ_BASE_URL
