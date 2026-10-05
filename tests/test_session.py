"""
test_session.py
===============
Tests for SessionStore (SQLite conversation memory) and ResearchSession (multi-turn
follow-ups, ticker tracking, resume, rolling summary, cached-brief freshness).

The agent workflow is replaced by ``FakeWorkflow`` so these tests run offline and fast;
they check what the session sends to the workflow and what it stores, not LLM output.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from research_agent.agent_framework import GraphMemory, GraphState
from research_agent.memory import (
    ResearchSession,
    ResearchSessionError,
    SessionStore,
    TurnKind,
    extract_tickers,
    is_follow_up,
    needs_company_reference,
)
from research_agent.memory.session import EMPTY_INPUT_REPLY, NEEDS_COMPANY_REPLY

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeWorkflow:
    """Stands in for the compiled agent graph and records every payload it receives."""

    def __init__(self, brief: bool = True):
        self.calls: list[dict[str, Any]] = []
        self.brief = brief

    def invoke(self, payload: dict[str, Any]) -> GraphState:
        self.calls.append(payload)
        n = len(self.calls)
        state = GraphState(question=payload["question"], thread_id=payload["thread_id"])
        state.output = f"Answer {n}. Details for: {payload['question'][:40]}"
        if self.brief:
            state.agents["DraftWriter"] = state.output
        state.trajectory = [
            {
                "step": 1,
                "node": "planner",
                "type": "agent",
                "executor": "ResearchPlanner",
                "tools": [],
                "summary": "plan",
                "output": "very long output that is not stored",
            }
        ]
        return state


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="session_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(tmp_dir: Path) -> SessionStore:
    return SessionStore(tmp_dir / "sessions.sqlite")


@pytest.fixture
def memory(tmp_dir: Path) -> GraphMemory:
    return GraphMemory(filepath=str(tmp_dir / "graph.json"))


def make_session(store, memory, thread="t1", workflow=None, **kw) -> ResearchSession:
    return ResearchSession(
        thread_id=thread, store=store, memory=memory, workflow=workflow or FakeWorkflow(), **kw
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestDetection:
    def test_extract_tickers_symbols_and_names(self):
        assert extract_tickers("Analyse NVIDIA (NVDA) margins") == ["NVDA"]
        assert extract_tickers("Compare $AMD with Intel") == ["AMD", "INTC"]
        assert extract_tickers("Compare AMD and Intel") == ["AMD", "INTC"]
        assert extract_tickers("How is Apple doing?") == ["AAPL"]
        assert extract_tickers("Look at BRK.B") == ["BRK-B"]

    def test_extract_tickers_ignores_finance_terms(self):
        assert extract_tickers("What is EPS vs PE and GDP in the US?") == []
        assert extract_tickers("Explain operating margin") == []

    def test_is_follow_up(self):
        assert is_follow_up("What about its margins?")
        assert is_follow_up("How does that compare to last year?")
        assert not is_follow_up("Analyse NVDA")
        assert not is_follow_up(None)

    def test_needs_company_reference(self):
        assert needs_company_reference("What about its debt?")
        assert not needs_company_reference("What about interest rates?")

    def test_extract_tickers_handles_empty_input(self):
        assert extract_tickers("") == []
        assert extract_tickers(None) == []

    def test_longest_company_name_wins(self):
        assert extract_tickers("Advanced Micro Devices guidance") == ["AMD"]


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class TestSessionStore:
    def test_thread_and_turns_roundtrip(self, store: SessionStore):
        store.ensure_thread("a")
        assert store.add_turn("a", "q1", "a1", ticker="NVDA") == 1
        assert store.add_turn("a", "q2", "a2", ticker="NVDA", kind="follow_up") == 2
        turns = store.get_turns("a")
        assert [t["question"] for t in turns] == ["q1", "q2"]
        assert turns[1]["kind"] == "follow_up"
        assert [t["turn_no"] for t in store.get_turns("a", last=1)] == [2]

    def test_threads_are_isolated(self, store: SessionStore):
        store.add_turn("a", "qa", "aa")
        store.add_turn("b", "qb", "ab")
        assert [t["question"] for t in store.get_turns("a")] == ["qa"]
        assert {t["thread_id"] for t in store.list_threads()} == {"a", "b"}

    def test_update_thread_rejects_unknown_fields(self, store: SessionStore):
        store.ensure_thread("a")
        store.update_thread("a", active_ticker="AAPL")
        assert store.get_thread("a")["active_ticker"] == "AAPL"
        with pytest.raises(ValueError, match="Unknown thread fields"):
            store.update_thread("a", bogus=1)

    def test_cache_freshness(self, store: SessionStore):
        store.cache_put("a", "brief:NVDA", {"brief": "x"})
        fresh = store.cache_get("a", "brief:NVDA", max_age_seconds=3600)
        assert fresh["value"] == {"brief": "x"}
        assert fresh["stale"] is False
        stale = store.cache_get("a", "brief:NVDA", max_age_seconds=-1)
        assert stale["stale"] is True
        assert store.cache_get("a", "brief:AAPL") is None

    def test_delete_thread_cascades(self, store: SessionStore):
        store.add_turn("a", "q", "a")
        store.cache_put("a", "k", 1)
        store.delete_thread("a")
        assert store.get_thread("a") is None
        assert store.get_turns("a") == []
        assert store.cache_get("a", "k") is None

    def test_in_memory_store(self):
        mem = SessionStore(":memory:")
        mem.add_turn("x", "q", "a")
        assert mem.count_turns("x") == 1

    def test_persists_on_disk(self, tmp_dir: Path):
        path = tmp_dir / "s.sqlite"
        with SessionStore(path) as first:
            first.add_turn("a", "q", "a")
        with SessionStore(path) as second:
            assert second.count_turns("a") == 1

    def test_blank_thread_id_is_rejected(self, store: SessionStore):
        with pytest.raises(ValueError, match="thread_id"):
            store.add_turn("  ", "q", "a")

    def test_turn_numbers_are_sequential_per_thread(self, store: SessionStore):
        numbers = [store.add_turn("a", f"q{i}", "a") for i in range(5)]
        assert numbers == [1, 2, 3, 4, 5]
        assert store.add_turn("b", "q", "a") == 1

    def test_get_turns_after_turn(self, store: SessionStore):
        for i in range(4):
            store.add_turn("a", f"q{i}", "a")
        assert [t["turn_no"] for t in store.get_turns("a", after_turn=2)] == [3, 4]

    def test_newer_schema_version_is_refused(self, tmp_dir: Path):
        import sqlite3

        path = tmp_dir / "future.sqlite"
        SessionStore(path).close()
        with sqlite3.connect(path) as conn:
            conn.execute("PRAGMA user_version = 999")
        with pytest.raises(RuntimeError):
            SessionStore(path)


# ---------------------------------------------------------------------------
# ResearchSession
# ---------------------------------------------------------------------------


class TestResearchSession:
    def test_first_question_sets_active_ticker(self, store, memory):
        wf = FakeWorkflow()
        s = make_session(store, memory, workflow=wf)
        answer = s.ask("Analyse NVIDIA (NVDA)", show=False)
        assert answer.startswith("Answer 1")
        assert s.active_ticker == "NVDA"
        assert s.last_kind == "new"
        assert wf.calls[0]["active_ticker"] == "NVDA"
        assert "conversation_context" not in wf.calls[0]  # nothing to remember yet

    def test_pronoun_follow_up_is_resolved_to_active_ticker(self, store, memory):
        wf = FakeWorkflow()
        s = make_session(store, memory, workflow=wf)
        s.ask("Analyse NVDA", show=False)
        s.ask("What about its margins?", show=False)
        payload = wf.calls[1]
        assert s.last_kind == "follow_up"
        assert payload["active_ticker"] == "NVDA"
        assert "about NVDA" in payload["question"]
        assert "Q1 [NVDA]: Analyse NVDA" in payload["conversation_context"]
        assert "prior_research_context" in payload  # cached brief is reused

    def test_switching_ticker(self, store, memory):
        wf = FakeWorkflow()
        s = make_session(store, memory, workflow=wf)
        s.ask("Analyse NVDA", show=False)
        s.ask("Now look at TSLA", show=False)
        assert s.last_kind == "switch"
        assert s.active_ticker == "TSLA"
        assert "prior_research_context" not in wf.calls[1]  # no TSLA brief cached yet

    def test_follow_up_without_company_asks_instead_of_running(self, store, memory):
        wf = FakeWorkflow()
        s = make_session(store, memory, workflow=wf)
        reply = s.ask("What about its debt?", show=False)
        assert reply == NEEDS_COMPANY_REPLY
        assert wf.calls == []
        assert s.last_kind == "needs_company"

    def test_empty_input(self, store, memory):
        wf = FakeWorkflow()
        s = make_session(store, memory, workflow=wf)
        assert s.ask("   ", show=False) == EMPTY_INPUT_REPLY
        assert wf.calls == []

    def test_general_question_keeps_no_ticker(self, store, memory):
        wf = FakeWorkflow(brief=False)
        s = make_session(store, memory, workflow=wf)
        s.ask("What is operating margin?", show=False)
        assert s.last_kind == "general"
        assert "active_ticker" not in wf.calls[0]
        assert store.cache_keys("t1") == []

    def test_resume_thread_after_restart(self, store, memory):
        s1 = make_session(store, memory)
        s1.ask("Analyse AAPL", show=False)
        wf2 = FakeWorkflow()
        s2 = make_session(store, memory, workflow=wf2)
        assert s2.resumed is True
        assert s2.active_ticker == "AAPL"
        s2.ask("How did its services revenue grow?", show=False)
        assert wf2.calls[0]["active_ticker"] == "AAPL"
        assert len(s2.history) == 2

    def test_threads_do_not_share_context(self, store, memory):
        a = make_session(store, memory, thread="a")
        b = make_session(store, memory, thread="b")
        a.ask("Analyse NVDA", show=False)
        assert b.active_ticker is None
        assert b.ask("What about its debt?", show=False) == NEEDS_COMPANY_REPLY

    def test_rolling_summary_keeps_context_bounded(self, store, memory):
        wf = FakeWorkflow()
        s = make_session(store, memory, workflow=wf, recent_turns=2, max_summary_chars=400)
        s.ask("Analyse NVDA", show=False)
        for i in range(6):
            s.ask(f"Follow-up number {i} about its outlook?", show=False)
        assert s.summary  # older turns were folded in
        assert len(s.summary) <= 400
        ctx = wf.calls[-1]["conversation_context"]
        assert ctx.count("\nA") <= 2  # only the 2 most recent turns verbatim
        assert "Earlier in this conversation" in ctx

    def test_stale_brief_is_flagged(self, store, memory):
        wf = FakeWorkflow()
        s = make_session(store, memory, workflow=wf, cache_max_age=-1)
        s.ask("Analyse NVDA", show=False)
        s.ask("What about its valuation?", show=False)
        assert "STALE" in wf.calls[1]["prior_research_fetched"]

    def test_reset_clears_thread(self, store, memory):
        s = make_session(store, memory)
        s.ask("Analyse NVDA", show=False)
        s.reset()
        assert s.history == []
        assert s.active_ticker is None

    def test_trajectory_is_stored_compactly(self, store, memory):
        s = make_session(store, memory)
        s.ask("Analyse NVDA", show=False)
        stored = store.get_turns("t1")[0]["trajectory"][0]
        assert stored["node"] == "planner"
        assert "output" not in stored

    def test_turn_kind_compares_as_string(self, store, memory):
        s = make_session(store, memory)
        s.ask("Analyse NVDA", show=False)
        assert s.last_kind is TurnKind.NEW
        assert store.get_turns("t1")[0]["kind"] == "new"

    def test_workflow_failure_is_recorded_and_raised(self, store, memory):
        class BrokenWorkflow:
            def invoke(self, payload):
                raise ConnectionError("LLM endpoint unreachable")

        s = make_session(store, memory, workflow=BrokenWorkflow())
        with pytest.raises(ResearchSessionError) as excinfo:
            s.ask("Analyse NVDA", show=False)
        assert isinstance(excinfo.value.__cause__, ConnectionError)
        turn = store.get_turns("t1")[0]
        assert turn["kind"] == "error"
        assert turn["ticker"] == "NVDA"

    def test_error_turns_are_not_sent_as_context(self, store, memory):
        class FlakyWorkflow(FakeWorkflow):
            def invoke(self, payload):
                if not self.calls:
                    self.calls.append(payload)
                    raise TimeoutError("timed out")
                return super().invoke(payload)

        wf = FlakyWorkflow()
        s = make_session(store, memory, workflow=wf)
        with pytest.raises(ResearchSessionError):
            s.ask("Analyse NVDA", show=False)
        s.ask("Analyse NVDA again", show=False)
        assert "conversation_context" not in wf.calls[1]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"thread_id": " "}, "thread_id"),
            ({"recent_turns": 0}, "recent_turns"),
            ({"max_summary_chars": 0}, "max_summary_chars"),
        ],
    )
    def test_invalid_configuration_is_rejected(self, store, memory, kwargs, message):
        with pytest.raises(ValueError, match=message):
            ResearchSession(store=store, memory=memory, workflow=FakeWorkflow(), **kwargs)

    def test_show_renders_without_ipython(self, store, memory, capsys, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_ipython(name, *args, **kwargs):
            if name.startswith("IPython"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_ipython)
        s = make_session(store, memory)
        assert s.ask("Analyse NVDA", show=True) is None
        assert "Answer 1" in capsys.readouterr().out

    def test_backward_compatible_import(self):
        from research_agent.workflows import ResearchSession as Legacy

        assert Legacy is ResearchSession
