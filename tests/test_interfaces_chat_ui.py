"""
test_interfaces_chat_ui.py
===========================
Offline tests for the Gradio chat UI glue. ``ResearchSession`` is replaced with a
fake, so these tests check turn-handling and session wiring without a real research
run or launching a Gradio server.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from research_agent.interfaces import chat_ui


class FakeSession:
    def __init__(
        self,
        thread_id: str,
        history=None,
        ticker: str | None = None,
        ask_error: Exception | None = None,
        trajectory=None,
        threads=None,
    ):
        self.thread_id = thread_id
        self.history = history or []
        self.resumed = bool(history)
        self.active_ticker = ticker
        self._ask_error = ask_error
        self._trajectory = trajectory
        self._threads = threads or []
        self.asked_with: str | None = None
        self.last_state = None

    def ask(self, message, show=False):
        self.asked_with = message
        if self._ask_error:
            raise self._ask_error
        if self._trajectory is not None:
            self.last_state = SimpleNamespace(trajectory=self._trajectory)
        return f"Answer to: {message}"

    def list_sessions(self):
        return self._threads


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
        assert session.asked_with is None

    def test_creates_session_when_none_given(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "ResearchSession", FakeSession)
        gen = chat_ui.respond("Analyse NVDA", [], None)
        history, session, _label, _box = next(gen)
        assert session is not None
        history, session, _label, _box = next(gen)
        assert "Answer to: Analyse NVDA" in history[-1]["content"]

    def test_successful_turn_shows_thinking_then_answer(self):
        session = FakeSession("t1")
        gen = chat_ui.respond("Analyse NVDA", [], session)

        history, _, _, _ = next(gen)
        assert history[-1]["content"] == chat_ui.THINKING_MESSAGE
        assert history[-1]["metadata"]["status"] == "pending"

        history, _, _, _ = next(gen)
        assert history[-1]["content"] == "Answer to: Analyse NVDA"
        assert all(m["content"] != chat_ui.THINKING_MESSAGE for m in history)
        assert session.asked_with == "Analyse NVDA"

    def test_reasoning_block_precedes_answer(self):
        steps = [{"node": "Planner", "summary": "plan"}, {"node": "Writer", "tools": ["news"]}]
        session = FakeSession("t1", trajectory=steps)
        gen = chat_ui.respond("Analyse NVDA", [], session)
        next(gen)
        history, _, _, _ = next(gen)
        assert history[-2]["metadata"]["title"] == "Reasoning · 2 steps"
        assert "Planner" in history[-2]["content"]
        assert history[-1]["content"] == "Answer to: Analyse NVDA"

    def test_exception_during_ask_is_shown_in_chat(self):
        session = FakeSession("t1", ask_error=RuntimeError("tool exploded"))
        gen = chat_ui.respond("Analyse NVDA", [], session)
        next(gen)
        history, _, _, _ = next(gen)
        assert "Something went wrong" in history[-1]["content"]
        assert "tool exploded" in history[-1]["content"]
        assert "metadata" not in history[-2] or history[-2].get("role") == "user"

    def test_preserves_prior_history(self):
        session = FakeSession("t1")
        prior = [{"role": "user", "content": "earlier question"}]
        gen = chat_ui.respond("follow up", prior, session)
        history, _, _, _ = next(gen)
        assert history[0] == {"role": "user", "content": "earlier question"}


class TestTrajectory:
    def test_markdown_lists_steps_with_fallback_names(self):
        md = chat_ui._trajectory_markdown(
            [{"node": "Planner", "summary": "a  b\nc"}, {"executor": "Fetcher", "tools": "yf"}, {}]
        )
        assert "1. **Planner** — a b c" in md
        assert "_tools: yf_" in md
        assert "3. **Step 3**" in md

    def test_long_summary_is_truncated(self):
        md = chat_ui._trajectory_markdown([{"node": "N", "summary": "x" * 500}])
        assert md.endswith("…")

    def test_no_state_means_no_reasoning(self):
        assert chat_ui._reasoning_message(FakeSession("t1")) is None

    def test_single_step_title(self):
        session = SimpleNamespace(last_state=SimpleNamespace(trajectory=[{"node": "A"}]))
        assert chat_ui._reasoning_message(session)["metadata"]["title"] == "Reasoning · 1 step"


THREADS = [
    {"thread_id": "a", "active_ticker": "NVDA", "turns": 3, "updated_at": "2026-10-11T09:00"},
    {"thread_id": "b", "active_ticker": None, "turns": 1, "updated_at": ""},
    {"thread_id": "c", "active_ticker": "NVDA", "turns": 0, "updated_at": "2026-10-10"},
]


class TestRail:
    def test_thread_choices_skip_empty_threads(self):
        choices = chat_ui._thread_choices(THREADS)
        assert choices == [("NVDA · 3 turns · 2026-10-11", "a"), ("General · 1 turn", "b")]

    def test_tickers_html_dedupes(self):
        out = chat_ui._tickers_html(THREADS)
        assert out.count("NVDA") == 1

    def test_tickers_html_empty(self):
        assert chat_ui._tickers_html([]) == ""

    def test_list_threads_falls_back_to_empty_on_error(self, monkeypatch):
        def boom():
            raise RuntimeError("db locked")

        monkeypatch.setattr(chat_ui, "SessionStore", boom)
        assert chat_ui._list_threads() == []

    def test_list_threads_reads_store(self, monkeypatch):
        monkeypatch.setattr(
            chat_ui, "SessionStore", lambda: SimpleNamespace(list_threads=lambda: THREADS)
        )
        assert chat_ui._list_threads() == THREADS

    def test_threads_for_none_uses_store(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "_list_threads", lambda: ["x"])
        assert chat_ui._threads_for(None) == ["x"]


class TestHeaderAndSuggestions:
    def test_header_without_session(self):
        assert "New research" in chat_ui._header_html(None)

    def test_header_with_ticker_escapes(self):
        out = chat_ui._header_html(FakeSession("<t>", ticker="AAPL"))
        assert "AAPL research" in out
        assert "&lt;t&gt;" in out

    def test_starter_prompts_without_ticker(self):
        assert chat_ui._suggestions(None) == list(chat_ui.STARTER_PROMPTS)

    def test_follow_ups_use_ticker(self):
        assert any("MSFT" in s for s in chat_ui._suggestions(FakeSession("t", ticker="MSFT")))


class TestExport:
    def test_empty_thread_has_no_file(self):
        assert chat_ui.export_thread(FakeSession("t1")) is None
        assert chat_ui.export_thread(None) is None

    def test_writes_markdown_report(self):
        past = [{"question": "Analyse NVDA", "answer": "Strong."}]
        path = chat_ui.export_thread(FakeSession("exp-1", history=past, ticker="NVDA"))
        text = Path(path).read_text(encoding="utf-8")
        assert "# Research report — NVDA" in text
        assert "## 1. Analyse NVDA" in text


class TestRefresh:
    def test_selects_current_thread_and_updates_controls(self):
        session = FakeSession("a", ticker="NVDA", threads=THREADS)
        radio, header, tickers, download, *buttons = chat_ui.refresh(session)
        assert radio.value == "a"
        assert "NVDA research" in header
        assert "NVDA" in tickers
        assert download.interactive is False
        assert len(buttons) == 3

    def test_unlisted_thread_selects_nothing(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "_list_threads", lambda: THREADS)
        radio, *_ = chat_ui.refresh(None)
        assert radio.value is None


class TestClear:
    def test_clear_starts_fresh_thread(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "ResearchSession", FakeSession)
        session, history, _label, box, tid = chat_ui._clear()
        assert history == []
        assert box == ""
        assert tid == session.thread_id


class TestBuildApp:
    def test_returns_blocks_instance(self):
        import gradio as gr

        demo = chat_ui.build_app()
        assert isinstance(demo, gr.Blocks)

    def test_theme_builds(self):
        import gradio as gr

        assert isinstance(chat_ui.make_theme(), gr.themes.Base)
