"""
workflows/session.py
====================
ResearchSession: Manages multi-turn conversation state, context memory, and question-answering.
"""

from __future__ import annotations

from typing import Any

from IPython.display import Markdown, display

from research_agent.agent_framework import (
    END,
    GraphMemory,
    GraphState,
    StateGraph,
)
from research_agent.agents import (
    critic_agent,
    draft_writer_agent,
    earnings_analyst_agent,
    market_analyst_agent,
    news_analyst_agent,
    planner_agent,
    refiner_agent,
)
from research_agent.config import DEFAULT_MEMORY_FILE
from research_agent.interfaces import LLM, get_llm


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


class ResearchSession:
    """Manages multi-turn research conversations and state continuity across follow-up queries."""

    def __init__(
        self,
        thread_id: str = "default_session",
        llm: LLM | None = None,
        memory: GraphMemory | None = None,
    ):
        self.thread_id = thread_id
        self.llm = llm or get_llm()
        self.memory = memory or GraphMemory(filepath=str(DEFAULT_MEMORY_FILE))
        self.history: list[dict[str, str]] = []
        self.last_state: GraphState | None = None
        self.last_answer: str = ""
        self._app = self._build_workflow()

    def _build_workflow(self):
        planner = planner_agent(llm=self.llm, tools=self.memory.as_tools())
        earnings_analyst = earnings_analyst_agent(llm=self.llm)
        market_analyst = market_analyst_agent(llm=self.llm)
        news_analyst = news_analyst_agent(llm=self.llm)
        draft_writer = draft_writer_agent(llm=self.llm)
        critic = critic_agent(llm=self.llm, tools=self.memory.as_tools())
        refiner = refiner_agent(llm=self.llm)

        workflow = StateGraph(state_schema=GraphState, memory=self.memory, verbose=False)
        workflow.add_node("planner", planner)
        workflow.add_node("specialists", [earnings_analyst, market_analyst, news_analyst])
        workflow.add_node("draft_writer", draft_writer)
        workflow.add_node("critic", critic)
        workflow.add_node("refiner", refiner)
        workflow.set_entry_point("planner")

        def route_planner(state: GraphState) -> str:
            ans = state.agents.get("ResearchPlanner", "").upper()
            if "RESEARCH_REQUIRED" in ans:
                return "specialists"
            return END

        workflow.add_conditional_edges(
            "planner",
            decider=route_planner,
            route_map={"specialists": "specialists", END: END},
        )
        workflow.add_edge("specialists", "draft_writer")
        workflow.add_edge("draft_writer", "critic")

        def route_critique(state: GraphState) -> str:
            refine_count = getattr(state, "refine_count", 0)
            critique_text = state.agents.get("ResearchCritic", "").upper()
            needs_work = any(
                k in critique_text
                for k in ["REVISE", "FAIL", "NEEDS WORK", "UNVERIFIED", "DEFICIENCY"]
            )
            if needs_work and refine_count < 2:
                state.refine_count = refine_count + 1
                return "refine"
            return "pass"

        workflow.add_conditional_edges(
            "critic",
            decider=route_critique,
            route_map={"refine": "refiner", "pass": END, "default": END},
        )
        workflow.add_edge("refiner", "critic")
        return workflow.compile()

    def ask(self, question: str, show: bool = True) -> str | None:
        """Executes inquiry within the conversational session context, showing question, output, and trajectory."""
        context_payload: dict[str, Any] = {"question": question, "thread_id": self.thread_id}
        if self.last_state and self.last_state.output:
            context_payload["prior_research_context"] = self.last_state.output[:1500]

        state = self._app.invoke(context_payload)
        self.last_state = state
        answer = state.output or "No response generated."
        self.last_answer = answer

        self.history.append({"question": question, "answer": answer})

        if show:
            parts = [
                f"### 💬 Question\n{question}",
                f"### 📋 Output\n\n{answer}",
            ]
            traj_table = _format_trajectory_table(getattr(state, "trajectory", []))
            if traj_table:
                parts.append(f"### 🧭 Trajectory\n\n{traj_table}")

            display(Markdown("\n\n---\n\n".join(parts)))
            return None

        return answer


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
