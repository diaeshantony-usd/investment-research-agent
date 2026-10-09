"""
workflows/session.py
====================
Trajectory formatting and the command-line entry point. ResearchSession is defined in
research_agent/memory/session.py and re-exported here for backward compatibility.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from research_agent.agent_framework import format_trajectory_markdown

# ResearchSession lives in research_agent.memory.session (persistent SQLite-backed
# conversation memory). It is re-exported here so that existing
# ``from research_agent.workflows import ResearchSession`` imports keep working.
from research_agent.memory.session import ResearchSession

__all__ = ["ResearchSession", "main"]


def _format_trajectory_table(trajectory: list[dict[str, Any]]) -> str:
    """Return the trajectory as a Markdown table (kept for existing notebook imports)."""
    return format_trajectory_markdown(trajectory)


def _configure_console_utf8() -> None:
    """Ensure standard output and error streams handle UTF-8 cleanly."""
    import io

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _run_interactive_loop(session: ResearchSession, thread_id: str) -> None:
    """Run interactive REPL loop in the terminal."""
    sys.stdout.write(f"=== Research Session Started [thread_id: {thread_id}] ===\n")
    sys.stdout.write("Type 'exit' or 'quit' to end session.\n\n")
    while True:
        try:
            sys.stdout.write("Query > ")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                break
            user_input = line.strip()
            if user_input.lower() in {"exit", "quit", "q"}:
                break
            if not user_input:
                continue
            if user_input.lower() in {"memory", "/memory"}:
                import json

                sys.stdout.write(
                    f"\n{json.dumps(session.root_memory, indent=2, default=str)}\n\n"
                )
                continue
            session.ask(user_input, show=False)
            sys.stdout.write(f"\n{session.last_answer}\n\n")
        except (KeyboardInterrupt, EOFError):
            break


def main(argv: list[str] | None = None) -> int:
    """Run one research query, or an interactive multi-turn chat in the terminal.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    _configure_console_utf8()

    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Institutional-grade Multi-Agent Equity Research System",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Equity research prompt (e.g. 'Analyze NVDA') or conceptual financial query",
    )
    parser.add_argument(
        "--thread-id",
        "-t",
        default=None,
        help="Session thread ID (defaults to a unique ID)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start an interactive multi-turn terminal chat session",
    )

    args = parser.parse_args(argv)

    is_interactive = bool(
        args.interactive or (args.query and args.query.lower() == "interactive")
    )

    if args.thread_id:
        thread_id = args.thread_id
    elif is_interactive:
        now_str = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
        thread_id = f"session_{now_str}_{uuid.uuid4().hex[:4]}"
    else:
        now_str = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
        thread_id = f"cli_{now_str}_{uuid.uuid4().hex[:4]}"

    session = ResearchSession(thread_id=thread_id)

    if is_interactive:
        _run_interactive_loop(session, thread_id)
        return 0

    if args.query and args.query.lower() != "interactive":
        session.ask(args.query, show=False)
        sys.stdout.write(f"{session.last_answer}\n")
        return 0

    parser.print_help()
    return 0

