"""
memory/long_term.py
===================
Long-term memory: what the system learns across runs (graded requirement F4).

Short-term memory (``SessionStore``) lets one conversation answer follow-up questions.
Long-term memory outlives conversations and threads, and makes *later* runs better:

* **Ticker notes** (``{TICKER}.json``) - one dated note per research run: the stance
  taken, the Critic's first-draft and final scores, and the issues left open. The next
  run for that ticker sees them, as hints only: prices and fundamentals are always
  fetched fresh.
* **Critic lessons** (``lessons.json``) - recurring weaknesses in first drafts, derived
  from the Critic's structured report (invented price targets, unsupported multiples,
  a missing section, a low balance score, ...). A lesson becomes *active* once it has
  appeared in two or more runs, so one unlucky run cannot teach a wrong lesson. Active
  lessons are given to the Draft writer before it writes, so the first draft avoids
  mistakes earlier drafts made.

Only the first draft (review round 0) teaches lessons: that is the text the Draft
writer controls, and its score is the metric that should improve over time.

Files are written atomically (temporary file, then rename). A missing file means a
first run; an unreadable one is logged and treated as empty, never fatal.

Example:
    >>> memory = LongTermMemory()
    >>> memory.record_run("NVDA", "Analyse NVDA", state.review, state.output)
    >>> memory.context_for("NVDA")
    {'ticker_notes': [...], 'critic_lessons': [...], 'usage': '...'}

Owner: Diaesh Antony
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from research_agent.config import MEMORY_DIR, logger

__all__ = [
    "DEFAULT_LONG_TERM_DIR",
    "LESSON_MIN_RUNS",
    "LongTermMemory",
    "derive_lessons",
]

DEFAULT_LONG_TERM_DIR: Final = MEMORY_DIR / "long_term"

# A lesson is shown to the agents only after it has appeared in this many runs.
LESSON_MIN_RUNS: Final = 2
# Upper bounds that keep the files, and the prompts built from them, small.
MAX_LESSONS: Final = 20
MAX_NOTES_PER_TICKER: Final = 5
NOTES_IN_CONTEXT: Final = 3
# A judged dimension below this score in a first draft becomes a lesson candidate.
LOW_DIMENSION_SCORE: Final = 7.0

_LESSONS_FILE: Final = "lessons.json"
_TICKER_FILE: Final = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,14}$")

# Lesson texts, keyed so the same weakness is counted once however it was worded.
_GROUNDING_LESSONS: Final = {
    "grounding_money": (
        "Do not write price targets, entry or stop-loss prices, support/resistance levels "
        "or dollar amounts that are not in the retrieved data."
    ),
    "grounding_multiple": (
        "Quote only valuation multiples (P/E, EV/Revenue, ...) that appear in the "
        "retrieved data; never estimate peer or historical multiples."
    ),
    "grounding_percent": (
        "Write a calculated percentage as '\u2248N%' on the same line as one of its inputs, "
        "and check the arithmetic; do not add growth forecasts."
    ),
    "grounding_other": "Copy every figure exactly as it appears in the retrieved data.",
}
_DIMENSION_LESSONS: Final = {
    "coverage": "Address fundamentals, news, price trend and macro context in every brief.",
    "balance": "Argue both the bull and the bear case, each with specific, sourced risks.",
    "clarity": "State the thesis in the first two sentences, in plain language.",
    "actionability": "Give an explicit stance and say what would change it.",
}
_STANCE: Final = re.compile(r"\b(bullish|neutral|bearish)\b", re.IGNORECASE)
_STANCE_HEADING: Final = re.compile(r"^#+.*(stance|outlook|recommendation)", re.I | re.M)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _figure_kind(figure: str) -> str:
    """Classify an unverified figure from its text: money, percent, multiple or other."""
    text = figure.strip().lower()
    if text.endswith("%"):
        return "percent"
    if text.endswith(("x", "\u00d7")):
        return "multiple"
    if text.startswith(("$", "\u20ac", "\u00a3")):
        return "money"
    return "other"


def derive_lessons(report: Mapping[str, Any]) -> dict[str, str]:
    """Turn one Critic report (a first draft's) into lesson candidates.

    Lessons come from the report's structured fields only (unverified figures, missing
    sections, per-dimension scores), not from the judge's free text, so the same
    weakness gets the same key in every run.

    Args:
        report: A review-history record as stored in ``GraphState.review["history"]``.

    Returns:
        ``{lesson_key: lesson_text}``; empty for a clean draft.
    """
    lessons: dict[str, str] = {}
    for figure in report.get("unverified") or []:
        key = f"grounding_{_figure_kind(str(figure))}"
        lessons[key] = _GROUNDING_LESSONS[key]
    for section in report.get("missing_sections") or []:
        lessons[f"missing_{section.lower().replace(' ', '_')}"] = (
            f"Always include the '{section}' section."
        )
    scores = report.get("scores") or {}
    for dimension, text in _DIMENSION_LESSONS.items():
        score = scores.get(dimension)
        if isinstance(score, (int, float)) and score < LOW_DIMENSION_SCORE:
            lessons[f"low_{dimension}"] = text
    return lessons


def _stance(brief: str) -> str | None:
    """The stance the brief takes, read from its stance section if it has one."""
    heading = _STANCE_HEADING.search(brief or "")
    match = _STANCE.search(brief[heading.start() :] if heading else brief or "")
    return match.group(1).capitalize() if match else None


class LongTermMemory:
    """Per-ticker notes and Critic lessons kept on disk across runs and sessions.

    Args:
        directory: Folder for the JSON files; defaults to ``data/memory/long_term``.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else DEFAULT_LONG_TERM_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"LongTermMemory(directory='{self.directory}')"

    # ------------------------------------------------------------------ write

    def record_run(
        self,
        ticker: str,
        question: str,
        review: Mapping[str, Any] | None,
        brief: str,
    ) -> dict[str, Any] | None:
        """Save what one research run taught: a ticker note and lesson counts.

        Args:
            ticker: The researched symbol, e.g. ``"NVDA"``.
            question: The question the run answered.
            review: ``GraphState.review`` from the run (its Critic history).
            brief: The released brief.

        Returns:
            The note that was saved, or None when the run had no review to learn from
            (e.g. a data-unavailable notice or a general question).
        """
        history: Sequence[Mapping[str, Any]] = (review or {}).get("history") or []
        if not history or not self._valid_ticker(ticker):
            return None
        first = history[0]
        best = max(history, key=lambda r: (-r.get("blockers", 0), r.get("weighted", 0.0)))

        note = {
            "date": _now(),
            "question": question[:200],
            "stance": _stance(brief),
            "first_draft_score": first.get("weighted"),
            "final_score": best.get("weighted"),
            "passed": bool(best.get("passed")),
            "refinements": len(history) - 1,
            "first_draft_unverified": len(first.get("unverified") or []),
            "open_issues": [
                f"{i.get('section')}: {i.get('problem')}"[:160]
                for i in (best.get("issues") or [])[:3]
            ]
            if not best.get("passed")
            else [],
        }
        notes = self._read(self._ticker_path(ticker), default={"notes": []})
        notes["notes"] = [*notes.get("notes", []), note][-MAX_NOTES_PER_TICKER:]
        self._write(self._ticker_path(ticker), notes)

        self._count_lessons(derive_lessons(first), ticker.upper())
        return note

    def _count_lessons(self, candidates: Mapping[str, str], ticker: str) -> None:
        if not candidates:
            return
        data = self._read(self.directory / _LESSONS_FILE, default={"lessons": {}})
        lessons: dict[str, dict[str, Any]] = data.get("lessons", {})
        now = _now()
        for key, text in candidates.items():
            entry = lessons.setdefault(key, {"text": text, "count": 0, "first_seen": now})
            entry.update(
                text=text,
                count=entry.get("count", 0) + 1,
                last_seen=now,
                tickers=sorted({*entry.get("tickers", []), ticker}),
            )
        # Keep the most frequent lessons, newest first among equals.
        ranked = sorted(
            lessons.items(), key=lambda kv: (kv[1]["count"], kv[1]["last_seen"]), reverse=True
        )
        data["lessons"] = dict(ranked[:MAX_LESSONS])
        data["updated"] = now
        self._write(self.directory / _LESSONS_FILE, data)

    def reset(self) -> None:
        """Forget all notes and lessons (manual reset after a wrong lesson)."""
        for path in self.directory.glob("*.json"):
            with contextlib.suppress(OSError):
                path.unlink()

    # ------------------------------------------------------------------ read

    def notes(self, ticker: str) -> list[dict[str, Any]]:
        """All saved notes for ``ticker``, oldest first."""
        if not self._valid_ticker(ticker):
            return []
        return list(self._read(self._ticker_path(ticker), default={"notes": []})["notes"])

    def lessons(self, active_only: bool = True) -> list[dict[str, Any]]:
        """Saved lessons, most frequent first; by default only the active ones."""
        data = self._read(self.directory / _LESSONS_FILE, default={"lessons": {}})
        rows = [{"key": key, **entry} for key, entry in data.get("lessons", {}).items()]
        rows.sort(key=lambda r: r.get("count", 0), reverse=True)
        if active_only:
            rows = [r for r in rows if r.get("count", 0) >= LESSON_MIN_RUNS]
        return rows

    def context_for(self, ticker: str | None) -> dict[str, Any]:
        """What the agents should see before researching ``ticker``.

        Returns:
            ``ticker_notes`` (the last few runs, dated), ``critic_lessons`` (active
            lesson texts) and a ``usage`` reminder; an empty dict when there is nothing
            to share yet (a first run).
        """
        context: dict[str, Any] = {}
        recent = self.notes(ticker)[-NOTES_IN_CONTEXT:] if ticker else []
        if recent:
            context["ticker_notes"] = recent
        lessons = [lesson["text"] for lesson in self.lessons()]
        if lessons:
            context["critic_lessons"] = lessons
        if context:
            context["usage"] = (
                "Long-term memory from earlier runs. Ticker notes are dated hints, never "
                "facts: always fetch fresh data and never copy figures from them. Follow "
                "the critic lessons when writing the brief."
            )
        return context

    # ------------------------------------------------------------------ files

    @staticmethod
    def _valid_ticker(ticker: str | None) -> bool:
        return bool(ticker) and _TICKER_FILE.match(str(ticker).upper()) is not None

    def _ticker_path(self, ticker: str) -> Path:
        return self.directory / f"{ticker.upper()}.json"

    @staticmethod
    def _read(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        """Load a JSON file; a missing or unreadable file gives ``default``."""
        if not path.exists():
            return default
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Long-term memory file %s is unreadable (%s); starting fresh", path, exc)
            return default
        return data if isinstance(data, dict) else default

    @staticmethod
    def _write(path: Path, data: Mapping[str, Any]) -> None:
        """Write JSON atomically, so a crash never leaves a half-written file."""
        handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, default=str)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
