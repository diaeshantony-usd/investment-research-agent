"""
memory/session.py
=================
``ResearchSession``: the single entry point used by the notebook, the CLI and the chat UI.

It wraps the compiled agent workflow (built in ``workflows/graph.py``) and adds
conversation memory on top of it:

* **Thread persistence** - every turn is saved to ``SessionStore`` (SQLite) under a
  ``thread_id``. Creating a session with an existing ``thread_id`` resumes that
  conversation, even after a kernel restart.
* **Active ticker tracking** - the session remembers which company is being discussed.
  Naming a new ticker switches it; questions such as "What about its margins?" are
  resolved to the active ticker before they reach the agents.
* **Clarification without a run** - a follow-up like "What about its debt?" with no
  company in context gets a short clarifying reply instead of a wasted pipeline run.
  Empty input is handled the same way.
* **Bounded conversation context** - the last N turns are passed verbatim; older turns
  are folded into a rolling summary so long conversations stay inside the model's
  context window.
* **Context cache with freshness** - the last research brief per ticker is cached with
  its fetch time. Follow-ups reuse it, and it is flagged as stale after
  ``cache_max_age`` seconds.

Example:
    >>> session = ResearchSession(thread_id="nvda_demo")
    >>> session.ask("Analyse NVDA")              # full research run
    >>> session.ask("What about its debt?")      # resolved to NVDA

Owner: Diaesh Antony. The workflow graph itself (planner, specialists, critic loop) is
built by ``workflows/graph.py`` and injected here, so this module does not change how
the agents run.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Protocol

from research_agent.agent_framework import GraphMemory, GraphState
from research_agent.config import DEFAULT_MEMORY_FILE, logger
from research_agent.memory.session_store import SessionStore
from research_agent.memory.tickers import (
    extract_tickers,
    is_follow_up,
    needs_company_reference,
)

__all__ = [
    "EMPTY_INPUT_REPLY",
    "NEEDS_COMPANY_REPLY",
    "ResearchSession",
    "ResearchSessionError",
    "TurnKind",
    "Workflow",
    "extract_tickers",
    "is_follow_up",
]

# ---------------------------------------------------------------------------
# Public constants and types
# ---------------------------------------------------------------------------

EMPTY_INPUT_REPLY: Final = (
    "Please ask a question about a stock, for example: "
    "'Analyse NVDA' or 'How did Apple's services revenue grow last quarter?'"
)
NEEDS_COMPANY_REPLY: Final = (
    "Which company do you mean? Mention a ticker or company name, for example "
    "'What about NVDA's debt?'"
)

# Agents whose output is a complete research brief worth caching for follow-ups.
_BRIEF_AGENTS: Final = ("BriefRefiner", "Refiner", "DraftWriter")

# Fields of a trajectory step kept in the database; the bulky ``output`` is dropped.
_TRAJECTORY_FIELDS: Final = frozenset(
    {"step", "timestamp", "node", "type", "executor", "agents", "tools", "summary"}
)

_ANSWER_PREVIEW_CHARS: Final = 300  # per recent turn in the conversation context
_SUMMARY_QUESTION_CHARS: Final = 120  # per folded turn in the rolling summary
_SUMMARY_ANSWER_CHARS: Final = 200
_SECONDS_PER_HOUR: Final = 3600
_DEFAULT_CACHE_MAX_AGE: Final = 6 * _SECONDS_PER_HOUR


class TurnKind(str, Enum):
    """How a turn relates to the conversation so far.

    The enum subclasses ``str`` so members compare equal to their plain-string value
    (``TurnKind.NEW == "new"``) and are stored in SQLite as readable text.
    """

    NEW = "new"  # first company discussed in the thread
    SAME = "same"  # the active company named again
    SWITCH = "switch"  # a different company than the active one
    FOLLOW_UP = "follow_up"  # refers to the active company without naming it
    GENERAL = "general"  # no company involved (e.g. "What is EBITDA?")
    NEEDS_COMPANY = "needs_company"  # refers to a company, but none is known yet
    EMPTY = "empty"  # blank input
    ERROR = "error"  # the workflow raised an exception

    @property
    def runs_workflow(self) -> bool:
        """True for kinds that are answered by the agent workflow."""
        return self not in (TurnKind.EMPTY, TurnKind.NEEDS_COMPANY, TurnKind.ERROR)

    def __str__(self) -> str:
        return self.value


class Workflow(Protocol):
    """Anything with ``invoke(payload) -> GraphState``, e.g. ``CompiledStateGraph``."""

    def invoke(self, payload: dict[str, Any]) -> GraphState:
        """Run the agents on ``payload`` and return the final graph state."""
        ...


WorkflowBuilder = Callable[[Any, GraphMemory], Workflow]


class ResearchSessionError(RuntimeError):
    """Raised when the agent workflow fails while answering a question.

    The failed turn is still recorded in the store (kind ``error``) so the
    conversation history shows what was asked and why it was not answered.
    """


@dataclass(frozen=True)
class _Resolution:
    """Result of interpreting a question against the current thread state."""

    question: str  # question as sent to the agents (may carry an added context note)
    ticker: str | None
    kind: TurnKind


# ---------------------------------------------------------------------------
# ResearchSession
# ---------------------------------------------------------------------------


class ResearchSession:
    """Multi-turn research conversation with persistent, thread-scoped memory.

    Args:
        thread_id: Conversation identifier. Reusing an id resumes that conversation.
        llm: LLM client passed to ``workflow_builder``. Defaults to ``get_llm()``.
            Ignored when ``workflow`` is given.
        memory: ``GraphMemory`` checkpoint store for workflow runs.
        store: ``SessionStore`` for conversation history. Defaults to
            ``data/memory/sessions.sqlite``.
        workflow: A compiled workflow with an ``invoke(payload)`` method. When omitted
            it is built with ``workflow_builder``.
        workflow_builder: ``(llm, memory) -> workflow``. Defaults to
            ``research_agent.workflows.graph.build_research_workflow``.
        recent_turns: Number of most recent turns passed to the agents verbatim.
        max_summary_chars: Upper bound on the rolling summary of older turns.
        max_brief_chars: How much of a cached brief is passed back as context.
        cache_max_age: Seconds after which a cached brief is marked stale. A negative
            value marks every cached brief as stale (useful for demos and tests).

    Raises:
        ValueError: If ``thread_id`` is blank or a size limit is not positive.

    Attributes:
        resumed: True when the thread already had turns when the session was created.
        last_state: Final ``GraphState`` of the most recent workflow run, if any.
        last_answer: Answer text of the most recent turn.
        last_kind: ``TurnKind`` of the most recent turn, or None before the first turn.
    """

    def __init__(
        self,
        thread_id: str = "default_session",
        llm: Any = None,
        memory: GraphMemory | None = None,
        store: SessionStore | None = None,
        workflow: Workflow | None = None,
        workflow_builder: WorkflowBuilder | None = None,
        recent_turns: int = 6,
        max_summary_chars: int = 2000,
        max_brief_chars: int = 1500,
        cache_max_age: float = _DEFAULT_CACHE_MAX_AGE,
    ) -> None:
        # Validate configuration up front so mistakes surface at construction time
        # rather than half-way through a conversation.
        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id must be a non-empty string")
        if recent_turns < 1:
            raise ValueError(f"recent_turns must be >= 1, got {recent_turns}")
        if max_summary_chars < 1 or max_brief_chars < 1:
            raise ValueError("max_summary_chars and max_brief_chars must be positive")

        self.thread_id = thread_id
        self.store = store if store is not None else SessionStore()
        self.memory = (
            memory if memory is not None else GraphMemory(filepath=str(DEFAULT_MEMORY_FILE))
        )
        self.recent_turns = recent_turns
        self.max_summary_chars = max_summary_chars
        self.max_brief_chars = max_brief_chars
        self.cache_max_age = cache_max_age

        self.llm = llm
        self._app: Workflow = (
            workflow if workflow is not None else self._build_workflow(workflow_builder)
        )

        # Restore state when the thread already exists (e.g. after a kernel restart).
        thread = self.store.ensure_thread(thread_id)
        turn_count = self.store.count_turns(thread_id)
        self.resumed: bool = turn_count > 0
        self.last_state: GraphState | None = None
        self.last_answer: str = ""
        self.last_kind: TurnKind | None = None
        if self.resumed:
            self.last_answer = self.store.get_turns(thread_id, last=1)[0]["answer"]
            logger.info(
                "Resumed thread '%s' (%d turns, active ticker: %s)",
                thread_id,
                turn_count,
                thread["active_ticker"] or "none",
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(thread_id={self.thread_id!r}, "
            f"active_ticker={self.active_ticker!r})"
        )

    # ------------------------------------------------------------------ properties

    @property
    def active_ticker(self) -> str | None:
        """Ticker of the company currently being discussed in this thread."""
        thread = self.store.get_thread(self.thread_id)
        return thread["active_ticker"] if thread else None

    @property
    def history(self) -> list[dict[str, str]]:
        """All turns in this thread as ``{"question", "answer"}`` dicts, oldest first."""
        return [
            {"question": turn["question"], "answer": turn["answer"]}
            for turn in self.store.get_turns(self.thread_id)
        ]

    @property
    def summary(self) -> str:
        """Rolling summary of turns that have dropped out of the recent-turn window."""
        thread = self.store.get_thread(self.thread_id)
        return (thread["summary"] or "") if thread else ""

    # ------------------------------------------------------------------ public API

    def ask(self, question: str | None, show: bool = True) -> str | None:
        """Answer one question within this conversation.

        Args:
            question: The user's question. Blank input gets a short prompt back.
            show: When True, render the question, answer and agent trajectory as
                Markdown (for notebooks) and return None. When False, return the text.

        Returns:
            The answer text when ``show`` is False, otherwise None.

        Raises:
            ResearchSessionError: If the agent workflow raises. The turn is recorded
                with kind ``error`` before the exception propagates.
        """
        text = (question or "").strip()
        if not text:
            return self._finish(question or "", EMPTY_INPUT_REPLY, kind=TurnKind.EMPTY, show=show)

        resolution = self._resolve(text)
        if resolution.kind is TurnKind.NEEDS_COMPANY:
            # Answer locally: running the agents without a company would waste a call.
            return self._finish(text, NEEDS_COMPANY_REPLY, kind=resolution.kind, show=show)

        payload = self._build_payload(resolution.question, resolution.ticker)
        logger.info(
            "[session %s] turn=%d kind=%s ticker=%s",
            self.thread_id,
            self.store.count_turns(self.thread_id) + 1,
            resolution.kind,
            resolution.ticker,
        )

        try:
            state = self._app.invoke(payload)
        except Exception as exc:
            # Keep the conversation history complete, then surface a single, clear
            # error type to the caller. The original exception is chained for debugging.
            logger.exception("[session %s] workflow failed", self.thread_id)
            self._finish(
                text,
                f"The research workflow failed: {exc}",
                kind=TurnKind.ERROR,
                show=False,
                resolved=resolution.question,
                ticker=resolution.ticker,
            )
            raise ResearchSessionError(
                f"Workflow failed for thread '{self.thread_id}': {exc}"
            ) from exc

        self.last_state = state
        answer = getattr(state, "output", "") or "No response generated."

        # Cache full briefs per ticker so follow-up questions can reuse them.
        if resolution.ticker and _produced_brief(state):
            self.store.cache_put(
                self.thread_id,
                f"brief:{resolution.ticker}",
                {"question": resolution.question, "brief": answer},
            )

        return self._finish(
            text,
            answer,
            kind=resolution.kind,
            show=show,
            resolved=resolution.question,
            ticker=resolution.ticker,
            trajectory=getattr(state, "trajectory", None) or [],
        )

    def reset(self) -> None:
        """Delete this thread's history and cache, and start it again empty."""
        self.store.delete_thread(self.thread_id)
        self.store.ensure_thread(self.thread_id)
        self.last_state = None
        self.last_answer = ""
        self.last_kind = None
        self.resumed = False

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return all stored threads, most recent first, with ticker and turn count."""
        return [dict(thread) for thread in self.store.list_threads()]

    def info(self) -> dict[str, Any]:
        """Return a snapshot of the session state, for notebook demos and debugging."""
        return {
            "thread_id": self.thread_id,
            "turns": self.store.count_turns(self.thread_id),
            "active_ticker": self.active_ticker,
            "resumed": self.resumed,
            "summary_chars": len(self.summary),
            "cached": self.store.cache_keys(self.thread_id),
        }

    # ------------------------------------------------------------------ construction

    def _build_workflow(self, builder: WorkflowBuilder | None) -> Workflow:
        """Build the default agent workflow.

        Imports are local so that callers who inject their own workflow (tests, the
        offline demo) do not need the LLM client or the full agent graph to import.
        """
        if self.llm is None:
            from research_agent.interfaces import get_llm

            self.llm = get_llm()
        if builder is None:
            from research_agent.workflows.graph import build_research_workflow

            builder = build_research_workflow
        return builder(self.llm, self.memory)

    # ------------------------------------------------------------------ turn pipeline

    def _resolve(self, text: str) -> _Resolution:
        """Work out which company the question is about and how it relates to the thread.

        Side effect: when the question names a company, that company becomes the
        thread's active ticker.

        Args:
            text: The stripped user question.

        Returns:
            The question to send to the agents, its ticker and its ``TurnKind``.
        """
        active = self.active_ticker
        mentioned = extract_tickers(text)

        # Case 1: the question names a company. The first one mentioned becomes active.
        if mentioned:
            ticker = mentioned[0]
            if active is None:
                kind = TurnKind.NEW
            elif ticker == active:
                kind = TurnKind.SAME
            else:
                kind = TurnKind.SWITCH
            self.store.update_thread(self.thread_id, active_ticker=ticker)
            return _Resolution(text, ticker, kind)

        # Case 2: no company named, but the question points back at earlier turns.
        if is_follow_up(text):
            if active:
                # Make the reference explicit so every agent sees the same company.
                resolved = f"{text}\n(Context: this question is about {active}.)"
                return _Resolution(resolved, active, TurnKind.FOLLOW_UP)
            if needs_company_reference(text):
                return _Resolution(text, None, TurnKind.NEEDS_COMPANY)

        # Case 3: a general finance question with no company involved.
        return _Resolution(text, None, TurnKind.GENERAL)

    def _build_payload(self, question: str, ticker: str | None) -> dict[str, Any]:
        """Build the workflow input: the question plus compact conversation context.

        Extra keys are surfaced to every agent by the workflow as "Current Context".
        Empty values are omitted so a first question carries no noise.
        """
        payload: dict[str, Any] = {"question": question, "thread_id": self.thread_id}
        if ticker:
            payload["active_ticker"] = ticker

        if conversation := self._conversation_context():
            payload["conversation_context"] = conversation

        if ticker:
            payload.update(self._prior_research(ticker))
        return payload

    def _prior_research(self, ticker: str) -> dict[str, str]:
        """Return the cached brief for ``ticker`` as payload fields, or an empty dict."""
        cached = self.store.cache_get(
            self.thread_id, f"brief:{ticker}", max_age_seconds=self.cache_max_age
        )
        if cached is None:
            return {}

        hours = cached["age_seconds"] / _SECONDS_PER_HOUR
        fetched = f"{cached['fetched_at']} ({hours:.1f} h ago)"
        if cached["stale"]:
            # Prices move quickly; warn the agents not to treat old numbers as current.
            fetched += " - STALE: refresh market data before relying on prices"
        return {
            "prior_research_context": cached["value"]["brief"][: self.max_brief_chars],
            "prior_research_fetched": fetched,
        }

    def _conversation_context(self) -> str:
        """Return the rolling summary plus the most recent turns, as plain text."""
        parts: list[str] = []
        if summary := self.summary:
            parts.append(f"Earlier in this conversation:\n{summary}")

        lines = [
            f"Q{turn['turn_no']} [{turn['ticker'] or '-'}]: {turn['question']}\n"
            f"A{turn['turn_no']}: {_truncate(turn['answer'], _ANSWER_PREVIEW_CHARS)}"
            for turn in self.store.get_turns(self.thread_id, last=self.recent_turns)
            if _is_research_turn(turn["kind"])
        ]
        if lines:
            parts.append("Recent turns:\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _roll_summary(self) -> None:
        """Fold turns that left the recent window into the rolling summary.

        Deterministic (no LLM call): each folded turn becomes one line. When the summary
        exceeds ``max_summary_chars`` the oldest lines are dropped first. The thread's
        ``summarized_upto`` marker makes repeated calls idempotent.
        """
        thread = self.store.get_thread(self.thread_id)
        if thread is None:
            return

        cutoff = self.store.count_turns(self.thread_id) - self.recent_turns
        already_folded = int(thread["summarized_upto"] or 0)
        if cutoff <= already_folded:
            return

        new_lines = [
            f"- Q{turn['turn_no']} [{turn['ticker'] or '-'}] "
            f"{_truncate(turn['question'], _SUMMARY_QUESTION_CHARS)} "
            f"-> {_first_sentence(turn['answer'], _SUMMARY_ANSWER_CHARS)}"
            for turn in self.store.get_turns(self.thread_id, after_turn=already_folded)
            if turn["turn_no"] <= cutoff and _is_research_turn(turn["kind"])
        ]
        lines = [line for line in (thread["summary"] or "").splitlines() if line]
        lines.extend(new_lines)
        while lines and len("\n".join(lines)) > self.max_summary_chars:
            lines.pop(0)

        self.store.update_thread(self.thread_id, summary="\n".join(lines), summarized_upto=cutoff)

    def _finish(
        self,
        question: str,
        answer: str,
        *,
        kind: TurnKind,
        show: bool,
        resolved: str | None = None,
        ticker: str | None = None,
        trajectory: Sequence[Mapping[str, Any]] = (),
    ) -> str | None:
        """Persist the turn, update the summary, then render or return the answer."""
        turn_no = self.store.add_turn(
            self.thread_id,
            question,
            answer,
            resolved_question=resolved,
            ticker=ticker,
            kind=kind.value,
            trajectory=_compact_trajectory(trajectory),
        )
        self._roll_summary()
        self.last_answer = answer
        self.last_kind = kind

        if not show:
            return answer
        self._display(question, answer, turn_no, kind, ticker, trajectory)
        return None

    def _display(
        self,
        question: str,
        answer: str,
        turn_no: int,
        kind: TurnKind,
        ticker: str | None,
        trajectory: Sequence[Mapping[str, Any]],
    ) -> None:
        """Render a turn as Markdown in a notebook, or as plain text elsewhere."""
        # Imported lazily because workflows.session imports this module.
        from research_agent.workflows.session import _format_trajectory_table

        meta = f"`thread: {self.thread_id}` · `turn {turn_no}` · `{kind}`"
        if ticker:
            meta += f" · `ticker: {ticker}`"
        parts = [f"### 💬 Question\n{question}\n\n{meta}", f"### 📋 Output\n\n{answer}"]
        if table := _format_trajectory_table([dict(step) for step in trajectory]):
            parts.append(f"### 🧭 Trajectory\n\n{table}")
        markdown = "\n\n---\n\n".join(parts)

        try:
            from IPython.display import Markdown, display
        except ImportError:  # running outside Jupyter, e.g. a plain script
            print(markdown)
            return
        display(Markdown(markdown))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _is_research_turn(kind: str) -> bool:
    """Return True if a stored turn was answered by the agents.

    Unknown kinds (e.g. written by a newer version) are treated as research turns so
    that no context is silently lost.
    """
    try:
        return TurnKind(kind).runs_workflow
    except ValueError:
        return True


def _produced_brief(state: Any) -> bool:
    """Return True if any brief-writing agent ran in this workflow state."""
    agents = getattr(state, "agents", None) or {}
    return any(name in agents for name in _BRIEF_AGENTS)


def _truncate(text: str | None, limit: int) -> str:
    """Collapse whitespace and cut ``text`` to ``limit`` characters with an ellipsis."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _first_sentence(text: str | None, limit: int) -> str:
    """Return the first sentence of ``text`` (Markdown markers stripped), truncated."""
    flat = " ".join((text or "").lstrip("#* ").split())
    match = re.search(r"(.+?[.!?])(\s|$)", flat)
    return _truncate(match.group(1) if match else flat, limit)


def _compact_trajectory(trajectory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the fields needed to audit a turn, so the database stays small."""
    return [
        {key: value for key, value in step.items() if key in _TRAJECTORY_FIELDS}
        for step in trajectory
    ]
