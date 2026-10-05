"""
workflows/session.py
====================
Trajectory formatting and the command-line entry point. ResearchSession is defined in
research_agent/memory/session.py and re-exported here for backward compatibility.
"""

from __future__ import annotations

import argparse
import sys
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


def main(argv: list[str] | None = None) -> int:
    """Run one research query, or an interactive multi-turn chat in the terminal.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
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
        default="cli_session",
        help="Session thread ID for conversational continuity (default: cli_session)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start an interactive multi-turn terminal chat session",
    )

    args = parser.parse_args(argv)

    session = ResearchSession(thread_id=args.thread_id)

    if args.interactive:
        sys.stdout.write(f"=== Research Session Started [thread_id: {args.thread_id}] ===\n")
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
                session.ask(user_input, show=False)
                sys.stdout.write(f"\n{session.last_answer}\n\n")
            except (KeyboardInterrupt, EOFError):
                break
        return 0

    if args.query:
        session.ask(args.query, show=False)
        sys.stdout.write(f"{session.last_answer}\n")
        return 0

    parser.print_help()
    return 0
