"""
workflows/session.py
====================
Trajectory formatting and the command-line entry point. ResearchSession is defined in
research_agent/memory/session.py and re-exported here for backward compatibility.
"""

from __future__ import annotations

from typing import Any

# ResearchSession now lives in research_agent.memory.session (persistent SQLite-backed
# conversation memory). It is re-exported here so existing imports keep working:
#     from research_agent.workflows import ResearchSession
from research_agent.memory.session import ResearchSession


def _format_trajectory_table(trajectory: list[dict[str, Any]]) -> str:
    """Formats the execution trajectory as a GitHub-flavored Markdown table."""
    if not trajectory:
        return ""

    md_lines = [
        "| Step | Node | Type | Executor | Tools Called | Summary / Decision |",
        "| :---: | :--- | :--- | :--- | :--- | :--- |",
    ]
    for step in trajectory:
        s_num = step.get("step", "-")
        s_node = f"`{step.get('node', '-')}`"
        s_type = str(step.get("type", "-")).capitalize()
        s_exec = (
            step.get("executor") or step.get("agent") or ", ".join(step.get("agents", [])) or "-"
        )
        tools = step.get("tools", [])
        s_tools = ", ".join(f"`{t}`" for t in tools) if tools else "-"
        summary = step.get("summary", "")
        if not summary:
            output_val = step.get("output", "")
            if output_val:
                first_line = str(output_val).strip().split("\n")[0]
                summary = (first_line[:80] + "...") if len(first_line) > 80 else first_line
            elif step.get("updates"):
                summary = f"Updated: {list(step['updates'].keys())}"
            else:
                summary = "Executed successfully"
        s_summary = summary.replace("\n", " ").replace("|", "\\|").strip()
        md_lines.append(f"| {s_num} | {s_node} | {s_type} | {s_exec} | {s_tools} | {s_summary} |")

    return "\n".join(md_lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running research queries or interactive sessions."""
    import argparse
    import sys

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
