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

import gradio as gr
import pytest

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

    def reset(self):
        self.history = []
        self.active_ticker = None
        self.was_reset = True


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
        assert history[0] == {"role": "user", "content": "Analyse NVDA"}
        assert history[1]["role"] == "assistant"
        assert "\n\nNVDA looks strong\n\n" in history[1]["content"]
        assert len(history) == 2


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
        assert session is None  # the question is shown before the session is built
        assert "<b>Researching</b>" in history[-1]["content"]
        history, session, _label, _box = next(gen)
        assert session is not None
        assert "Answer to: Analyse NVDA" in history[-1]["content"]

    def test_successful_turn_shows_thinking_then_answer(self):
        session = FakeSession("t1")
        gen = chat_ui.respond("Analyse NVDA", [], session)

        history, _, _, _ = next(gen)
        assert chat_ui.THINKING_MESSAGE in history[-1]["content"]
        assert "<b>Researching</b>" in history[-1]["content"]

        history, _, _, _ = next(gen)
        assert "\n\nAnswer to: Analyse NVDA\n\n" in history[-1]["content"]
        assert all(chat_ui.THINKING_MESSAGE not in m["content"] for m in history)
        assert session.asked_with == "Analyse NVDA"

    def test_reasoning_block_precedes_answer(self):
        steps = [{"node": "Planner", "summary": "plan"}, {"node": "Writer", "tools": ["news"]}]
        session = FakeSession("t1", trajectory=steps)
        gen = chat_ui.respond("Analyse NVDA", [], session)
        next(gen)
        history, _, _, _ = next(gen)
        assert history[-2]["metadata"]["title"] == "Reasoning · 2 steps"
        assert "Planner" in history[-2]["content"]
        assert "\n\nAnswer to: Analyse NVDA\n\n" in history[-1]["content"]

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
        assert choices == [("NVDA\n3 turns · Oct 11", "a"), ("General\n1 turn", "b")]

    def test_short_date(self):
        assert chat_ui._short_date("2026-10-05T09:00") == "Oct 5"
        assert chat_ui._short_date("") == ""
        assert chat_ui._short_date("yesterday") == "yesterday"

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
        assert '<span class="tag tag-ticker">AAPL</span>' in out
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
    def test_marks_current_thread_and_updates_controls(self):
        session = FakeSession("a", ticker="NVDA", threads=THREADS)
        rail, header, download, *buttons = chat_ui.refresh(session)
        assert 'class="thread-row current"' in rail
        assert rail.count('data-rail-action="delete"') == 2
        assert "NVDA research" in header
        assert download.interactive is False
        assert len(buttons) == 3

    def test_without_session_nothing_is_current(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "_list_threads", lambda: THREADS)
        rail, *_ = chat_ui.refresh(None)
        assert "current" not in rail
        assert rail.count('class="thread-row"') == 2


class TestClear:
    def test_clear_starts_fresh_thread(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "ResearchSession", FakeSession)
        session, history, _label, box = chat_ui._clear()
        assert history == []
        assert box == ""
        assert session.thread_id.startswith("gradio-")


class TestBuildApp:
    def test_returns_blocks_instance(self):
        import gradio as gr

        demo = chat_ui.build_app()
        assert isinstance(demo, gr.Blocks)

    def test_theme_builds(self):
        import gradio as gr

        assert isinstance(chat_ui.make_theme(), gr.themes.Base)


class ProgressSession(FakeSession):
    """A session whose ask() reports live progress, like ResearchSession does."""

    def __init__(self, thread_id: str, steps, delay: float = 0.3):
        super().__init__(thread_id)
        self._steps = steps
        self._delay = delay

    def ask(self, message, show=False, on_progress=None):
        import time

        for step in self._steps:
            on_progress(step)
        time.sleep(self._delay)  # still running when the UI polls for progress
        return f"Answer to: {message}"


class TestLiveProgress:
    def test_pending_message_without_steps(self):
        content = chat_ui._pending_message([])["content"]
        assert content.startswith('<div class="ira-progress">')
        assert "<b>Researching</b>" in content
        assert chat_ui.THINKING_MESSAGE in content
        assert "ira-steps" not in content
        assert "Elapsed" not in content

    def test_pending_message_groups_tool_calls_under_their_agent(self):
        steps = [
            "Agent: [ResearchPlanner] Resolving the ticker...",
            "Tool: [ResearchPlanner] lookup_ticker(query='NVDA')",
            "Agent: [ResearchRouter] Dispatching specialists...",
            "Tool: [ResearchRouter] delegate_to_earningsanalyst(task_description='...')",
            "Tool: [EarningsAnalyst] get_earnings_data(ticker='NVDA')",
        ]
        content = chat_ui._pending_message(steps, elapsed=75)["content"]
        assert " · 5 steps" in content
        assert "Elapsed 01:15" in content
        groups = chat_ui._step_groups(steps)
        assert [g[0] for g in groups] == ["ResearchPlanner", "ResearchRouter", "EarningsAnalyst"]
        assert groups[0][2] == ["lookup_ticker(query='NVDA')"]
        assert groups[2][2] == ["get_earnings_data(ticker='NVDA')"]
        # Only the latest agent is marked active.
        assert content.count("ira-badge") == 1
        assert content.index("ira-active") > content.index("ResearchRouter")

    def test_step_text_is_html_escaped(self):
        content = chat_ui._pending_message(["Tool: [A] f(x='<script>')"])["content"]
        assert "<script>" not in content
        assert "&lt;script&gt;" in content

    def test_unrecognised_progress_lines_are_still_listed(self):
        assert chat_ui._step_groups(["warming up"]) == [("", "warming up", [])]

    def test_long_tool_calls_are_shortened(self):
        groups = chat_ui._step_groups(["Tool: [A] f(" + "x" * 300 + ")"])
        assert len(groups[0][2][0]) <= chat_ui.PROGRESS_CALL_CHARS

    def test_accepts_progress(self):
        assert chat_ui._accepts_progress(ProgressSession("t", []).ask)
        assert not chat_ui._accepts_progress(FakeSession("t").ask)
        assert not chat_ui._accepts_progress(42)

    def test_progress_is_streamed_while_the_turn_runs(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "PROGRESS_POLL_SECONDS", 0.05)
        session = ProgressSession("t1", ["Agent: [ResearchPlanner] planning", "Tool: [X] f()"])
        updates = [history[-1] for history, *_ in chat_ui.respond("Analyse NVDA", [], session)]

        assert "<b>Researching</b>" in updates[0]["content"]
        streamed = [u for u in updates[1:-1] if "ira-steps" in u["content"]]
        assert streamed, "expected at least one live progress update"
        assert "ResearchPlanner" in streamed[-1]["content"]
        assert " · 2 steps" in streamed[-1]["content"]
        assert "Elapsed 00:0" in streamed[-1]["content"]
        assert "\n\nAnswer to: Analyse NVDA\n\n" in updates[-1]["content"]


class FakeStore:
    """Records deletes; ``fail_on`` makes deleting that thread raise."""

    def __init__(self, fail_on: str | None = None):
        self.deleted: list[str] = []
        self._fail_on = fail_on

    def delete_thread(self, thread_id):
        if thread_id == self._fail_on:
            raise RuntimeError("database is locked")
        self.deleted.append(thread_id)


class TestDeleteThread:
    def test_deleting_another_thread_keeps_the_open_chat(self):
        session = FakeSession("a", ticker="NVDA")
        session.store = FakeStore()
        returned, history, label = chat_ui.delete_thread("b", session)
        assert session.store.deleted == ["b"]
        assert returned is session
        assert history == gr.skip()  # the chat on screen is left alone
        assert label == "Active ticker: **NVDA**"

    def test_deleting_the_open_thread_starts_a_new_one(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "ResearchSession", FakeSession)
        session = FakeSession("a", ticker="NVDA")
        session.store = FakeStore()
        returned, history, label = chat_ui.delete_thread("a", session)
        assert session.store.deleted == ["a"]
        assert returned is not session
        assert returned.thread_id.startswith("gradio-")
        assert history == []
        assert label == "Active ticker: none yet"

    def test_a_failed_delete_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "ResearchSession", FakeSession)
        session = FakeSession("a")
        session.store = FakeStore(fail_on="a")
        returned, history, _ = chat_ui.delete_thread("a", session)
        assert returned.thread_id.startswith("gradio-")
        assert history == []

    def test_without_session_uses_the_default_store(self, monkeypatch):
        store = FakeStore()
        monkeypatch.setattr(chat_ui, "SessionStore", lambda: store)
        monkeypatch.setattr(chat_ui, "ResearchSession", FakeSession)
        returned, history, _ = chat_ui.delete_thread("x", None)
        assert store.deleted == ["x"]
        assert returned is not None
        assert history == []


class TestDeleteAllThreads:
    def test_deletes_every_thread_and_starts_fresh(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "ResearchSession", FakeSession)
        session = FakeSession("a", threads=THREADS)
        session.store = FakeStore(fail_on="a")  # one failure must not stop the rest
        returned, history, label, clear_btn, confirm_box = chat_ui.delete_all_threads(session)
        assert session.store.deleted == ["b", "c"]
        assert returned.thread_id.startswith("gradio-")
        assert history == []
        assert label == "Active ticker: none yet"
        assert clear_btn.visible is True
        assert confirm_box.visible is False

    def test_confirm_step_toggles(self):
        clear_btn, confirm_box = chat_ui._confirm_clear(True)
        assert clear_btn.visible is False
        assert confirm_box.visible is True


class TestRailHtml:
    def test_one_row_per_conversation_with_the_open_one_marked(self):
        rail = {"choices": [("NVDA\n3 turns", "a"), ("General\n1 turn", "b")], "current": "a"}
        html = chat_ui._rail_html(rail)
        assert html.count('class="thread-row') == 2
        assert (
            '<div class="thread-row current"><button type="button" class="thread-open" '
            'data-rail-action="open" data-tid="a">' in html
        )
        assert (
            '<span class="thread-title">NVDA</span><span class="thread-meta">3 turns</span>' in html
        )

    def test_thread_ids_and_labels_are_escaped(self):
        html = chat_ui._rail_html({"choices": [("<b>X</b>\n1 turn", 'a"b')], "current": None})
        assert 'data-tid="a&quot;b"' in html
        assert "&lt;b&gt;X&lt;/b&gt;" in html

    def test_empty_list_shows_a_hint(self):
        assert "No conversations yet" in chat_ui._rail_html(None)


class TestRailCommand:
    def test_delete_acts_once_per_click(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            chat_ui, "delete_thread", lambda tid, s: calls.append(tid) or (s, [], "")
        )
        session = FakeSession("a")
        chat_ui.rail_command("delete|b|click-1", session)
        chat_ui.rail_command("delete|b|click-1", session)  # the same click delivered again
        assert calls == ["b"]
        chat_ui.rail_command("delete|c|click-2", session)
        assert calls == ["b", "c"]

    def test_open_resumes_the_thread(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "start_session", lambda tid: ("S-" + tid, [], ""))
        returned, _history, _label = chat_ui.rail_command("open|t1|click-3", None)
        assert returned == "S-t1"

    def test_malformed_or_unknown_commands_change_nothing(self, monkeypatch):
        monkeypatch.setattr(chat_ui, "delete_thread", lambda *_: pytest.fail("must not delete"))
        session = FakeSession("a", ticker="NVDA")
        for command in ("", "delete", "delete|b", "rename|b|click-4", None):
            returned, history, _label = chat_ui.rail_command(command, session)
            assert returned is session
            assert history == gr.skip()


class TestAnswerCard:
    def test_full_brief_gets_header_meta_and_footer(self):
        session = FakeSession("a", ticker="NVDA")
        content = chat_ui._answer_message("## Verdict\nStrong", session, steps=3, elapsed=75)[
            "content"
        ]
        assert "Research brief" in content
        assert "3 steps · 01:15" in content
        assert "\n\n## Verdict\nStrong\n\n" in content  # blank lines keep the Markdown rendering
        assert chat_ui.DISCLAIMER in content
        assert "tag-ticker" not in content  # the header above the chat shows the ticker

    def test_short_reply_is_a_plain_answer(self):
        content = chat_ui._answer_message("Hello! Ask me about a stock.", None)["content"]
        assert "Answer" in content
        assert "Research brief" not in content
        assert chat_ui.DISCLAIMER not in content
        assert "ira-report-meta" not in content

    def test_is_brief(self):
        assert chat_ui._is_brief("| a | b |\n|---|---|")
        assert chat_ui._is_brief("x" * (chat_ui.BRIEF_MIN_CHARS + 1))
        assert not chat_ui._is_brief("Short reply.")

    def test_error_message_is_escaped(self):
        content = chat_ui._error_message("<boom>")["content"]
        assert "&lt;boom&gt;" in content
        assert "ira-error" in content
