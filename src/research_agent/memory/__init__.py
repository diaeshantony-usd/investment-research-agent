"""
memory
======
Conversation memory for the research agent.

* ``SessionStore``    - SQLite store of threads, turns and cached context (short-term memory)
* ``ResearchSession`` - multi-turn session used by the notebook, CLI and chat UI
* ``extract_tickers`` / ``is_follow_up`` - company name/symbol and pronoun detection
"""

from __future__ import annotations

from research_agent.memory.session import (
    ResearchSession,
    ResearchSessionError,
    TurnKind,
)
from research_agent.memory.session_store import DEFAULT_SESSION_DB, SessionStore
from research_agent.memory.tickers import (
    extract_tickers,
    is_follow_up,
    needs_company_reference,
)

__all__ = [
    "DEFAULT_SESSION_DB",
    "ResearchSession",
    "ResearchSessionError",
    "SessionStore",
    "TurnKind",
    "extract_tickers",
    "is_follow_up",
    "needs_company_reference",
]
