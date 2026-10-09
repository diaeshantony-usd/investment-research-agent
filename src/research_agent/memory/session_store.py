"""
memory/session_store.py
=======================
SQLite-backed short-term memory for research conversations.

A single database file holds every conversation thread in three tables:

``threads``
    One row per ``thread_id``: the active ticker, the rolling summary of older turns
    and bookkeeping timestamps.
``turns``
    Every question/answer pair in order, with the ticker it was about, the kind of turn
    and a compact copy of the agent trajectory for auditing.
``context_cache``
    Reusable results (for example the last research brief per ticker) stored with the
    time they were produced, so follow-up questions can reuse them and the session can
    tell when they have gone stale.

This store is deliberately separate from ``GraphMemory``. ``GraphMemory`` checkpoints
what the agents did *inside one workflow run*; ``SessionStore`` records the
*conversation across runs* so a user can ask follow-up questions or resume a thread
after a restart.

Example:
    >>> store = SessionStore(":memory:")
    >>> store.add_turn("demo", "Analyse NVDA", "NVDA looks strong.", ticker="NVDA")
    1
    >>> store.get_turns("demo")[0]["ticker"]
    'NVDA'

Owner: Diaesh Antony
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, TypedDict

from research_agent.config import MEMORY_DIR, logger

__all__ = [
    "DEFAULT_SESSION_DB",
    "SCHEMA_VERSION",
    "CacheEntry",
    "SessionStore",
    "ThreadRecord",
    "ThreadSummary",
    "TurnRecord",
]

#: Default location of the session database. ``*.sqlite`` is ignored by git.
DEFAULT_SESSION_DB: Path = MEMORY_DIR / "sessions.sqlite"

#: Bumped whenever the schema changes, so older databases can be migrated safely.
SCHEMA_VERSION: int = 1

#: Seconds to wait for a write lock held by another process before failing.
_LOCK_TIMEOUT_SECONDS: float = 10.0

#: Thread fields that callers are allowed to update through ``update_thread``.
_UPDATABLE_THREAD_FIELDS: frozenset[str] = frozenset(
    {"active_ticker", "summary", "summarized_upto"}
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id        TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    active_ticker    TEXT,
    summary          TEXT NOT NULL DEFAULT '',
    summarized_upto  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS turns (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id          TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    turn_no            INTEGER NOT NULL,
    question           TEXT NOT NULL,
    resolved_question  TEXT NOT NULL,
    answer             TEXT NOT NULL,
    ticker             TEXT,
    kind               TEXT NOT NULL,
    trajectory         TEXT NOT NULL DEFAULT '[]',
    created_at         TEXT NOT NULL,
    UNIQUE (thread_id, turn_no)
);

CREATE TABLE IF NOT EXISTS context_cache (
    thread_id   TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (thread_id, key)
);

CREATE INDEX IF NOT EXISTS idx_turns_thread ON turns(thread_id, turn_no);
"""


# ---------------------------------------------------------------------------
# Typed records returned by the store
# ---------------------------------------------------------------------------


class ThreadRecord(TypedDict):
    """One row of the ``threads`` table."""

    thread_id: str
    created_at: str
    updated_at: str
    active_ticker: str | None
    summary: str
    summarized_upto: int


class ThreadSummary(TypedDict):
    """Listing entry returned by :meth:`SessionStore.list_threads`."""

    thread_id: str
    active_ticker: str | None
    created_at: str
    updated_at: str
    turns: int


class TurnRecord(TypedDict):
    """One question/answer pair, as returned by :meth:`SessionStore.get_turns`."""

    id: int
    thread_id: str
    turn_no: int
    question: str
    resolved_question: str
    answer: str
    ticker: str | None
    kind: str
    trajectory: list[dict[str, Any]]
    created_at: str


class CacheEntry(TypedDict):
    """A cached value together with its freshness metadata."""

    value: Any
    fetched_at: str
    age_seconds: float
    stale: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Returns the current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _age_seconds(iso_timestamp: str) -> float:
    """Returns the seconds elapsed since an ISO-8601 timestamp written by ``_utc_now``."""
    return (datetime.now(timezone.utc) - datetime.fromisoformat(iso_timestamp)).total_seconds()


def _require_thread_id(thread_id: str) -> None:
    """Rejects empty thread ids, which would silently merge unrelated conversations.

    Raises:
        ValueError: If ``thread_id`` is empty or only whitespace.
    """
    if not thread_id or not thread_id.strip():
        raise ValueError("thread_id must be a non-empty string")


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class SessionStore:
    """Persistent, thread-keyed conversation store backed by one SQLite file.

    A new connection is opened for every operation and closed straight after, so one
    store can be shared by several ``ResearchSession`` objects (for example the notebook
    and the CLI at the same time) without holding locks between calls. The database runs
    in WAL mode, which lets readers continue while another process writes.

    For tests and throwaway sessions pass ``":memory:"``. An in-memory SQLite database
    disappears when its connection closes, so in that mode a single connection is kept
    open for the lifetime of the store; call :meth:`close` (or use the store as a
    context manager) to release it.

    Args:
        db_path: Path to the SQLite file, or ``":memory:"``. Parent folders are created
            if needed. Defaults to :data:`DEFAULT_SESSION_DB`.

    Raises:
        RuntimeError: If the database was created by a newer, incompatible schema.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path: str = str(db_path or DEFAULT_SESSION_DB)
        self._shared_conn: sqlite3.Connection | None = None

        if self.db_path == ":memory:":
            self._shared_conn = self._open()
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._initialise_schema()
        logger.info("SessionStore ready at %s", self.db_path)

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Closes the shared in-memory connection, if any. Safe to call twice."""
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SessionStore(db_path={self.db_path!r})"

    # ------------------------------------------------------------------ connection handling
    def _open(self) -> sqlite3.Connection:
        """Opens a connection configured for this store."""
        conn = sqlite3.connect(self.db_path, timeout=_LOCK_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        # Foreign keys are off by default in SQLite; they are needed for ON DELETE CASCADE.
        conn.execute("PRAGMA foreign_keys = ON")
        if self.db_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Yields a connection inside a transaction: commit on success, rollback on error."""
        conn = self._shared_conn or self._open()
        try:
            with conn:  # sqlite3 commits or rolls back on exit
                yield conn
        finally:
            if conn is not self._shared_conn:
                conn.close()

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Runs a single statement in its own transaction and returns all rows."""
        with self._transaction() as conn:
            return conn.execute(sql, params).fetchall()

    def _initialise_schema(self) -> None:
        """Creates tables on first use and checks the stored schema version."""
        with self._transaction() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self.db_path} uses session schema v{version}, but this code "
                    f"supports up to v{SCHEMA_VERSION}. Update the research_agent package."
                )
            conn.executescript(_SCHEMA)
            # PRAGMA does not accept bound parameters; SCHEMA_VERSION is a trusted int.
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION:d}")

    # ------------------------------------------------------------------ threads
    def ensure_thread(self, thread_id: str) -> ThreadRecord:
        """Returns the thread record, creating an empty thread on first use.

        Args:
            thread_id: Conversation identifier.

        Returns:
            The (possibly new) thread record.

        Raises:
            ValueError: If ``thread_id`` is empty.
        """
        _require_thread_id(thread_id)
        now = _utc_now()
        self._query(
            "INSERT OR IGNORE INTO threads (thread_id, created_at, updated_at) VALUES (?, ?, ?)",
            (thread_id, now, now),
        )
        record = self.get_thread(thread_id)
        if record is None:  # pragma: no cover - only if the row vanished concurrently
            raise RuntimeError(f"Thread '{thread_id}' could not be created")
        return record

    def get_thread(self, thread_id: str) -> ThreadRecord | None:
        """Returns the thread record, or ``None`` if the thread does not exist."""
        rows = self._query("SELECT * FROM threads WHERE thread_id = ?", (thread_id,))
        return ThreadRecord(**dict(rows[0])) if rows else None  # type: ignore[typeddict-item]

    def update_thread(self, thread_id: str, **fields: Any) -> None:
        """Updates selected fields of a thread and refreshes ``updated_at``.

        Args:
            thread_id: Conversation identifier.
            **fields: Any of ``active_ticker``, ``summary`` or ``summarized_upto``.

        Raises:
            ValueError: If an unknown field name is passed.
        """
        unknown = set(fields) - _UPDATABLE_THREAD_FIELDS
        if unknown:
            raise ValueError(f"Unknown thread fields: {sorted(unknown)}")
        if not fields:
            return
        # Column names come from the allow-list above, never from user input.
        assignments = ", ".join(f"{name} = ?" for name in fields)
        self._query(
            f"UPDATE threads SET {assignments}, updated_at = ? WHERE thread_id = ?",  # noqa: S608
            (*fields.values(), _utc_now(), thread_id),
        )

    def list_threads(self) -> list[ThreadSummary]:
        """Returns every thread with its turn count, most recently used first."""
        rows = self._query(
            """
            SELECT t.thread_id, t.active_ticker, t.created_at, t.updated_at,
                   COUNT(u.id) AS turns
            FROM threads AS t
            LEFT JOIN turns AS u ON u.thread_id = t.thread_id
            GROUP BY t.thread_id
            ORDER BY t.updated_at DESC, t.thread_id
            """
        )
        return [ThreadSummary(**dict(row)) for row in rows]  # type: ignore[typeddict-item]

    def delete_thread(self, thread_id: str) -> None:
        """Deletes a thread together with its turns and cached context (cascade)."""
        self._query("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        logger.info("Deleted session thread %s", thread_id)

    # ------------------------------------------------------------------ turns
    def add_turn(
        self,
        thread_id: str,
        question: str,
        answer: str,
        *,
        resolved_question: str | None = None,
        ticker: str | None = None,
        kind: str = "general",
        trajectory: list[dict[str, Any]] | None = None,
    ) -> int:
        """Appends a question/answer pair to a thread.

        The turn number is computed inside the INSERT statement itself, so two processes
        writing to the same thread at once cannot be given the same number.

        Args:
            thread_id: Conversation identifier (created if it does not exist).
            question: The question exactly as the user typed it.
            answer: The answer shown to the user.
            resolved_question: The question as sent to the agents, after follow-up
                resolution. Defaults to ``question``.
            ticker: Ticker the turn was about, if any.
            kind: Turn category, e.g. ``"follow_up"`` (see ``TurnKind``).
            trajectory: Compact agent trajectory for auditing.

        Returns:
            The 1-based number of the new turn within its thread.
        """
        self.ensure_thread(thread_id)
        now = _utc_now()
        with self._transaction() as conn:
            row = conn.execute(
                """
                INSERT INTO turns (thread_id, turn_no, question, resolved_question, answer,
                                   ticker, kind, trajectory, created_at)
                SELECT ?, COALESCE(MAX(turn_no), 0) + 1, ?, ?, ?, ?, ?, ?, ?
                FROM turns WHERE thread_id = ?
                RETURNING turn_no
                """,
                (
                    thread_id,
                    question,
                    resolved_question or question,
                    answer,
                    ticker,
                    kind,
                    json.dumps(trajectory or [], default=str),
                    now,
                    thread_id,
                ),
            ).fetchone()
            conn.execute("UPDATE threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id))
        return int(row["turn_no"])

    def count_turns(self, thread_id: str) -> int:
        """Returns how many turns a thread has (0 for unknown threads)."""
        rows = self._query("SELECT COUNT(*) AS n FROM turns WHERE thread_id = ?", (thread_id,))
        return int(rows[0]["n"])

    def get_turns(
        self,
        thread_id: str,
        *,
        last: int | None = None,
        after_turn: int = 0,
    ) -> list[TurnRecord]:
        """Returns turns of a thread in chronological order.

        Args:
            thread_id: Conversation identifier.
            last: If given, only the most recent ``last`` turns are returned.
            after_turn: Only turns with a number greater than this are returned.

        Returns:
            Turn records, oldest first, with ``trajectory`` decoded from JSON.
        """
        if last is not None:
            # Take the newest N in a subquery, then re-sort them oldest-first.
            rows = self._query(
                """
                SELECT * FROM (
                    SELECT * FROM turns WHERE thread_id = ? AND turn_no > ?
                    ORDER BY turn_no DESC LIMIT ?
                ) ORDER BY turn_no ASC
                """,
                (thread_id, after_turn, max(0, int(last))),
            )
        else:
            rows = self._query(
                "SELECT * FROM turns WHERE thread_id = ? AND turn_no > ? ORDER BY turn_no",
                (thread_id, after_turn),
            )

        turns: list[TurnRecord] = []
        for row in rows:
            record = dict(row)
            record["trajectory"] = json.loads(record["trajectory"] or "[]")
            turns.append(TurnRecord(**record))  # type: ignore[typeddict-item]
        return turns

    # ------------------------------------------------------------------ context cache
    def cache_put(self, thread_id: str, key: str, value: Any) -> None:
        """Stores a JSON-serialisable value under ``key``, stamped with the current time.

        Writing the same key again replaces the value and resets its timestamp.
        """
        self.ensure_thread(thread_id)
        self._query(
            """
            INSERT INTO context_cache (thread_id, key, value, fetched_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (thread_id, key) DO UPDATE
                SET value = excluded.value, fetched_at = excluded.fetched_at
            """,
            (thread_id, key, json.dumps(value, default=str), _utc_now()),
        )

    def cache_get(
        self,
        thread_id: str,
        key: str,
        max_age_seconds: float | None = None,
    ) -> CacheEntry | None:
        """Returns a cached value with its freshness, or ``None`` if nothing is cached.

        Stale entries are still returned (with ``stale=True``) so the caller can decide
        whether to reuse them with a warning or fetch fresh data.

        Args:
            thread_id: Conversation identifier.
            key: Cache key, e.g. ``"brief:NVDA"``.
            max_age_seconds: Entries older than this are flagged as stale. ``None``
                disables the check.
        """
        rows = self._query(
            "SELECT value, fetched_at FROM context_cache WHERE thread_id = ? AND key = ?",
            (thread_id, key),
        )
        if not rows:
            return None
        age = _age_seconds(rows[0]["fetched_at"])
        return CacheEntry(
            value=json.loads(rows[0]["value"]),
            fetched_at=rows[0]["fetched_at"],
            age_seconds=age,
            stale=max_age_seconds is not None and age > max_age_seconds,
        )

    def cache_keys(self, thread_id: str) -> list[str]:
        """Returns the cache keys stored for a thread, sorted alphabetically."""
        rows = self._query(
            "SELECT key FROM context_cache WHERE thread_id = ? ORDER BY key", (thread_id,)
        )
        return [row["key"] for row in rows]

    def cache_delete(self, thread_id: str, key: str) -> None:
        """Removes a single cache entry for a thread if present."""
        self._query(
            "DELETE FROM context_cache WHERE thread_id = ? AND key = ?",
            (thread_id, key),
        )

