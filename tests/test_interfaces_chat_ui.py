"""
test_interfaces_chat_ui.py
===========================
Offline tests for the Gradio chat UI glue. ``ResearchSession`` is replaced with a
fake, so these tests check turn-handling and session wiring without a real research
run or launching a Gradio server.
"""

from __future__ import annotations

from research_agent.interfaces import chat_ui


class FakeSession:
    def __init__(
        self,
        thread_id: str,
        history=None,
        ticker: str | None = None,
        ask_error: Exception | None = None,
    ):
        self.thread_id = thread_id
        self.history = history or []
        self.resumed = bool(history)
        self.active_ticker = ticker
        self._ask_error = ask_error
        self.asked_with: str | None = None

    def ask(self, message, show=False):
        self.asked_with = message
        if self._ask_error:
            raise self._ask_error
        return f"Answer to: {message}"


class TestNewSession:
    def test_blank_thread_id_generates_one(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "ResearchSession", FakeSession)
        session = chat_ui._new_session("")
        assert session.thread_id.startswith("gradio-")

    def test_given_thread_id_is_used_as_is(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "ResearchSession", FakeSession)
        session = chat_ui._new_session("my-thread")
        assert session.thread_id == "my-thread"

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "ResearchSession", FakeSession)
        session = chat_ui._new_session("  my-thread  ")
        assert session.thread_id == "my-thread"


class TestActiveTickerLabel:
    def test_none_session(self):
        assert chat_ui._active_ticker_label(None) == "Active ticker: none yet"

    def test_session_without_ticker(self):
        assert chat_ui._active_ticker_label(FakeSession("t1")) == "Active ticker: none yet"

    def test_session_with_ticker(self):
        label = chat_ui._active_ticker_label(FakeSession("t1", ticker="NVDA"))
        assert "NVDA" in label


class TestStartSession:
    def test_new_session_has_empty_history(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "ResearchSession", FakeSession)
        _session, history, label = chat_ui.start_session("")
        assert history == []
        assert label == "Active ticker: none yet"

    def test_resumed_session_replays_history(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "ResearchSession", FakeSession)
        past = [{"question": "Analyse NVDA", "answer": "NVDA looks strong"}]
        monkeypatch.setattr(
            chat_ui, "ResearchSession", lambda thread_id: FakeSession(thread_id, history=past)
        )
        _session, history, _label = chat_ui.start_session("existing-thread")
        assert history == [
            {"role": "user", "content": "Analyse NVDA"},
            {"role": "assistant", "content": "NVDA looks strong"},
        ]


class TestRespond:
    def test_blank_message_asks_for_a_question(self):
        session = FakeSession("t1")
        gen = chat_ui.respond("   ", [], session)
        history, _returned_session, _label, _box = next(gen)
        assert "Please ask a question" in history[-1]["content"]
        assert session.asked_with is None  # never reached session.ask

    def test_creates_session_when_none_given(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "ResearchSession", FakeSession)
        gen = chat_ui.respond("Analyse NVDA", [], None)
        history, session, _label, _box = next(gen)
        assert session is not None
        # Drain the generator so the real .ask() path runs too.
        history, session, _label, _box = next(gen)
        assert "Answer to: Analyse NVDA" in history[-1]["content"]

    def test_successful_turn_shows_thinking_then_answer(self):
        session = FakeSession("t1")
        gen = chat_ui.respond("Analyse NVDA", [], session)

        history, _, _, _ = next(gen)
        assert history[-1]["content"] == chat_ui.THINKING_MESSAGE

        history, _, _, _ = next(gen)
        assert history[-1]["content"] == "Answer to: Analyse NVDA"
        assert session.asked_with == "Analyse NVDA"

    def test_exception_during_ask_is_shown_in_chat(self):
        session = FakeSession("t1", ask_error=RuntimeError("tool exploded"))
        gen = chat_ui.respond("Analyse NVDA", [], session)
        next(gen)  # thinking message
        history, _, _, _ = next(gen)
        assert "Something went wrong" in history[-1]["content"]
        assert "tool exploded" in history[-1]["content"]

    def test_preserves_prior_history(self):
        session = FakeSession("t1")
        prior = [{"role": "user", "content": "earlier question"}]
        gen = chat_ui.respond("follow up", prior, session)
        history, _, _, _ = next(gen)
        assert history[0] == {"role": "user", "content": "earlier question"}


class TestBuildApp:
    def test_returns_blocks_instance(self):
        import gradio as gr

        demo = chat_ui.build_app()
        assert isinstance(demo, gr.Blocks)
