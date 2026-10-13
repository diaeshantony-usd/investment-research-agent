"""
test_long_term_memory.py
========================
Tests for long-term memory (graded requirement F4, ``memory/long_term.py``):

- lessons derived from a Critic report, and the two-run activation threshold
- ticker notes, caps, atomic writes and recovery from corrupt files
- the ResearchSession integration: notes and lessons reach the next run's payload
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from research_agent.agent_framework import GraphMemory, GraphState
from research_agent.memory import LongTermMemory, ResearchSession, SessionStore
from research_agent.memory.long_term import (
    LESSON_MIN_RUNS,
    MAX_LESSONS,
    MAX_NOTES_PER_TICKER,
    derive_lessons,
)

BRIEF = "# Research Brief: NVDA\n## Executive Summary\nGrowth.\n## 4. Investment Stance\nBullish."


def report(
    round_no: int = 0,
    weighted: float = 7.5,
    passed: bool = False,
    unverified: tuple[str, ...] = ("$260", "57\u00d7", "121 %"),
    missing: tuple[str, ...] = (),
    scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """A review-history record shaped like ``GraphState.review["history"][n]``."""
    return {
        "round": round_no,
        "brief": BRIEF,
        "scores": scores or {"coverage": 9, "balance": 6, "clarity": 8, "actionability": 8},
        "weighted": weighted,
        "passed": passed,
        "issues": [
            {"severity": "blocker", "section": "Stance", "problem": "invented target", "fix": "x"}
        ]
        if not passed
        else [],
        "unverified": list(unverified),
        "missing_sections": list(missing),
        "judge_used": True,
        "blockers": 0 if passed else 1,
    }


def review(*records: dict[str, Any]) -> dict[str, Any]:
    return {"history": list(records), "current": None, "change_logs": []}


@pytest.fixture
def memory(tmp_path) -> LongTermMemory:
    return LongTermMemory(tmp_path / "lt")


class TestDeriveLessons:
    def test_unverified_figures_become_lessons_by_kind(self):
        lessons = derive_lessons(report())
        assert {"grounding_money", "grounding_multiple", "grounding_percent"} <= set(lessons)

    def test_missing_sections_and_low_scores_become_lessons(self):
        lessons = derive_lessons(report(unverified=(), missing=("Technicals",)))
        assert set(lessons) == {"missing_technicals", "low_balance"}

    def test_a_clean_draft_teaches_nothing(self):
        clean = report(unverified=(), scores=dict.fromkeys(("coverage", "balance"), 9.0))
        assert derive_lessons(clean) == {}


class TestRecordRun:
    def test_first_run_saves_a_note_but_no_active_lesson(self, memory):
        note = memory.record_run(
            "nvda", "Analyse NVDA", review(report(), report(1, 8.4, True)), BRIEF
        )
        assert note["stance"] == "Bullish"
        assert (note["first_draft_score"], note["final_score"], note["passed"]) == (7.5, 8.4, True)
        assert note["refinements"] == 1
        assert note["first_draft_unverified"] == 3
        assert memory.notes("NVDA") == [note]
        assert memory.lessons() == []  # seen once: not trusted yet
        assert len(memory.lessons(active_only=False)) == 4

    def test_lessons_activate_after_the_minimum_number_of_runs(self, memory):
        for ticker in ("NVDA", "AAPL")[:LESSON_MIN_RUNS]:
            memory.record_run(ticker, "q", review(report()), BRIEF)
        active = memory.lessons()
        assert {lesson["key"] for lesson in active} >= {"grounding_money", "low_balance"}
        assert active[0]["tickers"] == ["AAPL", "NVDA"]

    def test_only_the_first_draft_teaches_lessons(self, memory):
        later_only = report(1, unverified=("$999",))
        first_clean = report(unverified=(), scores=dict.fromkeys(("balance",), 9.0))
        for _ in range(2):
            memory.record_run("NVDA", "q", review(first_clean, later_only), BRIEF)
        assert memory.lessons(active_only=False) == []

    def test_unpassed_run_keeps_its_open_issues(self, memory):
        note = memory.record_run("NVDA", "q", review(report()), BRIEF)
        assert note["open_issues"] == ["Stance: invented target"]

    def test_runs_without_a_review_are_not_recorded(self, memory):
        assert memory.record_run("NVDA", "q", None, BRIEF) is None
        assert memory.record_run("NVDA", "q", {"history": []}, BRIEF) is None
        assert memory.record_run("../etc", "q", review(report()), BRIEF) is None

    def test_notes_and_lessons_are_capped(self, memory):
        for _ in range(MAX_NOTES_PER_TICKER + 3):
            memory.record_run("NVDA", "q", review(report()), BRIEF)
        assert len(memory.notes("NVDA")) == MAX_NOTES_PER_TICKER

        many = {f"missing_s{i}": f"Always include s{i}." for i in range(MAX_LESSONS + 5)}
        memory._count_lessons(many, "NVDA")
        assert len(memory.lessons(active_only=False)) == MAX_LESSONS


class TestFiles:
    def test_corrupt_file_is_treated_as_empty(self, memory):
        (memory.directory / "NVDA.json").write_text("{not json", encoding="utf-8")
        assert memory.notes("NVDA") == []
        memory.record_run("NVDA", "q", review(report()), BRIEF)
        assert len(memory.notes("NVDA")) == 1

    def test_writes_leave_no_temporary_files(self, memory):
        memory.record_run("NVDA", "q", review(report()), BRIEF)
        names = sorted(p.name for p in memory.directory.iterdir())
        assert names == ["NVDA.json", "lessons.json"]
        json.loads((memory.directory / "lessons.json").read_text(encoding="utf-8"))

    def test_reset_forgets_everything(self, memory):
        memory.record_run("NVDA", "q", review(report()), BRIEF)
        memory.reset()
        assert memory.notes("NVDA") == []
        assert memory.lessons(active_only=False) == []


class TestContext:
    def test_first_run_gets_no_context(self, memory):
        assert memory.context_for("NVDA") == {}

    def test_context_holds_recent_notes_and_active_lessons(self, memory):
        for _ in range(4):
            memory.record_run("NVDA", "q", review(report()), BRIEF)
        context = memory.context_for("NVDA")
        assert len(context["ticker_notes"]) == 3
        assert any("price targets" in lesson for lesson in context["critic_lessons"])
        assert "never facts" in context["usage"]

    def test_lessons_are_shared_across_tickers(self, memory):
        for ticker in ("NVDA", "AAPL"):
            memory.record_run(ticker, "q", review(report()), BRIEF)
        context = memory.context_for("TSLA")
        assert "ticker_notes" not in context
        assert context["critic_lessons"]


class ReviewedWorkflow:
    """A fake graph that returns a brief and a Critic history, like the real one."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> GraphState:
        self.calls.append(payload)
        state = GraphState(question=payload["question"], thread_id=payload["thread_id"])
        state.output = BRIEF
        state.agents["DraftWriter"] = BRIEF
        state.review = review(report(), report(1, 8.3, True))
        return state


def test_session_learns_and_feeds_the_next_run(tmp_path):
    long_term = LongTermMemory(tmp_path / "lt")
    workflow = ReviewedWorkflow()

    def session(thread: str) -> ResearchSession:
        return ResearchSession(
            thread_id=thread,
            store=SessionStore(tmp_path / "s.sqlite"),
            memory=GraphMemory(filepath=str(tmp_path / "g.json")),
            long_term=long_term,
            workflow=workflow,
        )

    session("a").ask("Analyse NVDA", show=False)
    assert "long_term_memory" not in workflow.calls[0]  # nothing learned yet
    assert len(long_term.notes("NVDA")) == 1

    # A new conversation the next day still benefits from what the first one learned.
    session("b").ask("Analyse NVDA", show=False)
    session("c").ask("Analyse NVDA", show=False)
    learned = workflow.calls[-1]["long_term_memory"]
    assert len(learned["ticker_notes"]) == 2
    assert learned["critic_lessons"]
