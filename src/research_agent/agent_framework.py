"""
agent_framework.py
==================
Modular, state-driven multi-agent orchestration framework.

Key architectural pieces:

- **Tool**: wraps a Python function with metadata and safe execution, so a failing tool
  returns an error string the LLM can react to instead of crashing the run.
- **Agent**: a ReAct loop (reason, call tools, observe) with a bounded iteration count.
- **GraphState**: the shared "blackboard" passed between nodes (agent outputs, tool
  observations, trajectory).
- **GraphMemory**: JSON checkpoint store of finished runs, grouped by ``thread_id``.
- **StateGraph / CompiledStateGraph**: nodes, fixed edges, conditional routing and
  sequential "parallel" groups, compiled into an executable workflow.
- **Observability**: every step is recorded in ``GraphState.trajectory`` and can be
  rendered as a Markdown or ASCII table.

Author: N L N Sai Krishna Akula
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import os
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from research_agent.config import DEFAULT_MODEL, logger
from research_agent.interfaces.llm import LLM

# Sentinel node name that terminates a graph workflow.
END = "__end__"

# Timestamp formats used in traces, trajectory rows and checkpoints.
_CLOCK_FORMAT = "%H:%M:%S"
_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_STATE_ID_FORMAT = "%Y%m%d_%H%M%S"

# Characters of a tool observation shown in verbose logs.
_OBSERVATION_PREVIEW_CHARS = 200
# Characters of the user prompt shown in verbose logs.
_PROMPT_PREVIEW_CHARS = 120

# Agents whose output is the user-facing answer, most authoritative first. The critic
# is deliberately absent: its score must never be returned as the answer.
_OUTPUT_AGENT_PRIORITY = ("BriefRefiner", "Refiner", "DraftWriter", "ResearchPlanner", "Planner")

# GraphState fields that are never repeated to agents as "Current Context".
_STATE_BASE_KEYS = frozenset(
    {
        "question",
        "output",
        "state_unique_id",
        "thread_id",
        "agents",
        "tools",
        "trajectory",
        "refine_count",
        "input",
    }
)


def _local_now(fmt: str) -> str:
    """Return the current local time formatted with ``fmt``."""
    return datetime.now(timezone.utc).astimezone().strftime(fmt)


# ===========================================================================
# Tool Definition and Decorators
# ===========================================================================


@dataclass
class Tool:
    """Encapsulates an executable Python function with schema metadata and runtime isolation.

    Attributes:
        name: Unique identifier for the tool (alphanumeric and underscores).
        description: Natural language description provided to the LLM for tool selection.
        func: The underlying callable executed when the tool is called.
        parameters: Optional JSON Schema or argument specification.
    """

    name: str
    description: str
    func: Callable
    parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Normalise the name so provider SDKs accept it as a function identifier."""
        self.name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name)
        # Some callables (e.g. bound methods, builtins) do not allow these attributes to
        # be set; the Tool still works, it just keeps the callable's original metadata.
        with contextlib.suppress(AttributeError):
            self.func.__name__ = self.name
            if self.description:
                self.func.__doc__ = self.description

    def execute(self, **kwargs: Any) -> str:
        """Run the tool and always return a string.

        Dicts and lists are serialised as JSON. Exceptions are caught and returned as an
        error message so the calling LLM can correct its arguments and retry.

        Args:
            **kwargs: Arguments forwarded to the wrapped function.

        Returns:
            The tool output, or a readable error message.
        """
        try:
            result = self.func(**kwargs)
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2, default=str)
            return str(result)
        except Exception as e:  # noqa: BLE001
            # Intercept tool-level exceptions so the LLM can self-correct
            return f"Error executing tool '{self.name}': {type(e).__name__} - {e!s}"

    def to_callable(self) -> Callable[..., str]:
        """Return a plain function wrapper, as expected by provider SDKs (e.g. Ollama)."""

        @functools.wraps(self.func)
        def wrapper(*args: Any, **kwargs: Any) -> str:
            return self.execute(*args, **kwargs)

        wrapper.__name__ = self.name
        wrapper.__doc__ = self.description or self.func.__doc__ or f"Execute {self.name}"
        return wrapper


def tool(
    name: str | None = None, description: str | None = None
) -> Callable[[Callable[..., Any]], Tool]:
    """Turn a Python function into a ``Tool`` that agents can call.

    Args:
        name: Tool name shown to the LLM. Defaults to the function name.
        description: Tool description shown to the LLM. Defaults to the docstring.

    Returns:
        A decorator that replaces the function with a ``Tool`` instance.

    Example:
        >>> @tool(description="Add two integers.")
        ... def add(a: int, b: int) -> int:
        ...     return a + b
        >>> add.execute(a=1, b=2)
        '3'
    """

    def decorator(fn: Callable) -> Tool:
        tool_name = name or fn.__name__
        tool_desc = description or (inspect.getdoc(fn) or f"Execute {tool_name}")
        return Tool(name=tool_name, description=tool_desc, func=fn)

    return decorator


# ===========================================================================
# Observability and Tracing Data Structures
# ===========================================================================


@dataclass
class TraceStep:
    """Represents a single atomic event within an agent's reasoning loop.

    Step Types:
        - 'thought': Chain-of-thought internal reasoning from the model.
        - 'tool_call': Structured tool invocation action.
        - 'tool_result': Raw observation or return value from tool execution.
        - 'final_answer': Concluded response to the given prompt.
        - 'error': Exception encountered during API communication or runtime.
    """

    timestamp: str
    agent_name: str
    step_type: str
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Standardized response bundle returned by an Agent execution run."""

    agent_name: str
    input_prompt: str
    content: Any
    trace: list[TraceStep]
    messages: list[dict[str, Any]]
    iterations: int
    response_type: str = "text"

    def __str__(self) -> str:
        if isinstance(self.content, (dict, list)):
            return json.dumps(self.content, indent=2)
        return str(self.content)

    @property
    def data(self) -> dict[str, Any] | None:
        """Returns parsed dict payload if available, else None."""
        if isinstance(self.content, dict):
            return self.content
        if isinstance(self.content, str):
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                val = json.loads(self.content)
                if isinstance(val, dict):
                    return val
        return None

    def display_trace(self) -> None:
        """Write each reasoning step of this run to the project log."""
        logger.info("[TRACE] Execution Trace for Agent: %s", self.agent_name)
        for step in self.trace:
            prefix = {
                "thought": "[THOUGHT]",
                "tool_call": "[TOOL CALL]",
                "tool_result": "[OBSERVATION]",
                "final_answer": "[FINAL ANSWER]",
            }.get(step.step_type, f"[{step.step_type.upper()}]")

            body = (
                json.dumps(step.content, indent=2)
                if isinstance(step.content, (dict, list))
                else str(step.content)
            )
            logger.info("  %s (%s) -> %s", prefix, step.timestamp, body)


# ===========================================================================
def _parse_response_content(final_text: str, agent_name: str) -> tuple[str, Any]:
    """Parse agent output text into structured JSON if applicable, returning (type, content)."""
    if not isinstance(final_text, str) or agent_name in {"DraftWriter", "Refiner", "BriefRefiner"}:
        return "text", final_text

    stripped = final_text.strip()
    parsed = None
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
        if match:
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                parsed = json.loads(match.group(1).strip())
    if parsed is None and (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    ):
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            parsed = json.loads(stripped)

    if parsed is not None and isinstance(parsed, (dict, list)):
        return "json", parsed

    return "text", final_text


class Agent:
    """Autonomous ReAct (Reason + Act) agent capable of multi-step tool execution.

    Operates in an iterative loop:
    1. Sends conversation history and available tools to the LLM.
    2. Parses chain-of-thought and tool call proposals.
    3. Executes chosen tools and appends observations back into the context.
    4. Halts when the LLM produces a final answer or reaches max_iterations.
    """

    def __init__(
        self,
        name: str,
        model: str | LLM | None = None,
        tools: Sequence[Tool | Callable[..., Any]] | None = None,
        system_prompt: str | None = None,
        role: str | None = None,
        llm: LLM | None = None,
        max_iterations: int = 10,
        memory: list[dict[str, Any]] | None = None,
        verbose: bool = False,
        description: str | None = None,
    ) -> None:
        """Create an agent.

        Args:
            name: Unique agent name; also the key of its output in ``GraphState.agents``.
            model: Model name, or an ``LLM`` client to use directly.
            tools: Tools (or plain functions) the agent may call.
            system_prompt: Generic instructions appended after the role.
            role: Role description placed at the top of the system prompt.
            llm: LLM client. Ignored when ``model`` is already an ``LLM``.
            max_iterations: Upper bound on LLM calls per ``run`` (prevents tool loops).
            memory: Optional initial message history.
            verbose: Log every thought, tool call and observation.
            description: Concise human-readable statement of purpose or responsibility.
        """
        self.llm: LLM
        self.model: str
        # Resolve the client in priority order: model-as-client, explicit llm, factory.
        if isinstance(model, LLM):
            self.llm = model
            self.model = str(getattr(model, "model", DEFAULT_MODEL))
        elif llm is not None:
            self.llm = llm
            self.model = str(model or getattr(llm, "model", DEFAULT_MODEL))
        else:
            # Local import: interfaces.llm is heavy and only needed for this fallback.
            from research_agent.interfaces.llm import get_llm

            self.model = str(model or DEFAULT_MODEL)
            self.llm = get_llm(model=self.model)

        self.name = name
        self.role = role or "Specialized AI Analyst"
        self.description = description or ""
        self.system_prompt = system_prompt or "You are a helpful and rigorous analytical assistant."
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.messages: list[dict[str, Any]] = list(memory) if memory else []
        self.last_response: AgentResponse | None = None
        self._on_tool_call: Callable[[str, str, dict[str, Any]], None] | None = None

        # Register and normalize tools into internal dictionary
        self.tools: dict[str, Tool] = {}
        if tools:
            for t in tools:
                self.add_tool(t)

    def add_tool(self, tool_item: Tool | Callable[..., Any]) -> None:
        """Register a tool; plain callables are wrapped in a ``Tool`` automatically.

        Raises:
            TypeError: If ``tool_item`` is neither a ``Tool`` nor callable.
        """
        if isinstance(tool_item, Tool):
            self.tools[tool_item.name] = tool_item
        elif callable(tool_item):
            t = Tool(
                name=tool_item.__name__,
                description=inspect.getdoc(tool_item) or f"Execute {tool_item.__name__}",
                func=tool_item,
            )
            self.tools[t.name] = t
        else:
            raise TypeError(f"Expected Tool or Callable, got {type(tool_item)}")

    def _format_system_prompt(self) -> str:
        """Injects contextual timestamps and available tool descriptors into the system prompt."""
        now_str = _local_now(_DATETIME_FORMAT)
        tool_descs = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        header = f"Current Time: {now_str}\nRole: {self.role}\n"
        if tool_descs:
            header += (
                f"\nAvailable Tools:\n{tool_descs}\n\n"
                "MANDATORY TOOL USE & GROUNDING DIRECTIVE:\n"
                "You MUST execute your available tools to retrieve factual data. "
                "Never invent, assume, or hallucinate metrics, financial figures, or dates not "
                "returned by your tools. If data for a ticker or query is not found in the tools, "
                "explicitly report that data is unavailable from the tools rather than fabricating "
                "answers.\n"
            )
        return f"{header}\n{self.system_prompt}".strip()

    def _extract_tool_callables(self) -> Any:
        """Delegate tool formatting to the LLM client (callables or JSON schemas)."""
        return self.llm.get_tool_callables(self.tools)

    def _log_activation(self, prompt: str | None) -> None:
        """Log which agent is starting, with its model, role and prompt preview."""
        logger.info("=" * 80)
        logger.info("[AGENT ACTIVATION: %s]", self.name)
        logger.info("  - Model : %s", self.model)
        logger.info("  - Role  : %s", self.role)
        if prompt:
            preview = (
                prompt[:_PROMPT_PREVIEW_CHARS] + "..."
                if len(prompt) > _PROMPT_PREVIEW_CHARS
                else prompt
            )
            logger.info('  - Prompt: "%s"', preview)
        logger.info("=" * 80)

    def _execute_tool_call(self, call: dict[str, Any], now: str, trace: list[TraceStep]) -> None:
        """Run one tool call requested by the LLM and record call and observation.

        The observation is appended to the conversation as a ``tool`` message so the
        model sees it on the next iteration.
        """
        fn_name = call["name"]
        fn_args = call["args"]
        call_desc = f"{fn_name}({json.dumps(fn_args)})"

        trace.append(
            TraceStep(
                now,
                self.name,
                "tool_call",
                {"tool": fn_name, "arguments": fn_args},
                metadata={"raw_call": call_desc},
            )
        )

        if self.verbose:
            logger.info("[ACTION: %s]: Calling tool -> %s", self.name, call_desc)

        if hasattr(self, "_on_tool_call") and callable(self._on_tool_call):
            self._on_tool_call(self.name, fn_name, fn_args)

        if fn_name in self.tools:
            obs = self.tools[fn_name].execute(**fn_args)
        else:
            # Unknown tool: tell the model which tools exist so it can recover.
            obs = f"Error: Tool '{fn_name}' is not in available tools: {list(self.tools.keys())}"

        if self.verbose:
            preview = (
                obs[:_OBSERVATION_PREVIEW_CHARS] + "..."
                if len(obs) > _OBSERVATION_PREVIEW_CHARS
                else obs
            )
            logger.info("[OBSERVATION: %s]: %s", self.name, preview)

        trace.append(
            TraceStep(
                _local_now(_CLOCK_FORMAT),
                self.name,
                "tool_result",
                obs,
                metadata={"tool": fn_name, "args": fn_args},
            )
        )

        self.messages.append({"role": "tool", "content": str(obs)})

    def _prepare_run(self, prompt: str | None, on_progress: Callable[[str], None] | None) -> None:
        """Initializes system prompt, user prompt, and progress announcement."""
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages.insert(0, {"role": "system", "content": self._format_system_prompt()})

        if prompt:
            self.messages.append({"role": "user", "content": prompt})

        if on_progress:
            desc = self.description or "Analyzing and executing tasks"
            on_progress(f"Agent: [{self.name}] {desc}...")

    def run(
        self,
        prompt: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        """Run the ReAct loop until the model answers without calling a tool.

        Each iteration sends the conversation to the LLM, executes any requested tools
        and appends their observations. LLM errors end the run with an error response
        rather than raising, so one failing agent does not abort the whole graph.

        Args:
            prompt: New user message to append before running. ``None`` continues the
                existing conversation.
            on_progress: Optional callback invoked with live thinking/action messages.

        Returns:
            The final answer, the step-by-step trace and the full message history.
        """
        self._prepare_run(prompt, on_progress)

        cleanup = _attach_tool_hooks(self, on_progress)
        trace: list[TraceStep] = []
        iteration = 0
        final_text = ""

        if self.verbose:
            self._log_activation(prompt)

        try:
            while iteration < self.max_iterations:
                iteration += 1
                now = _local_now(_CLOCK_FORMAT)

                try:
                    content, tool_calls, raw_msg = self.llm.chat(self.messages, self.tools)
                except Exception as e:  # noqa: BLE001 - any provider error ends this agent only
                    err_msg = f"API Error communicating with model '{self.model}': {e!s}"
                    if self.verbose:
                        logger.error("[ERROR: %s] %s", self.name, err_msg)
                    trace.append(TraceStep(now, self.name, "error", err_msg))
                    return AgentResponse(
                        self.name, prompt or "", err_msg, trace, self.messages, iteration
                    )

                self.messages.append(raw_msg)

                if content:
                    trace.append(TraceStep(now, self.name, "thought", content))
                    if self.verbose and tool_calls:
                        logger.info("[THOUGHT: %s]: %s", self.name, content)

                if not tool_calls:
                    final_text = content or "(No text returned)"
                    trace.append(TraceStep(now, self.name, "final_answer", final_text))
                    if self.verbose:
                        logger.info("[FINAL ANSWER: %s]:\n%s", self.name, final_text)
                    break

                for call in tool_calls:
                    self._execute_tool_call(call, now, trace)
        finally:
            cleanup()

        if iteration >= self.max_iterations and not final_text:
            final_text = (
                f"Agent '{self.name}' stopped after reaching maximum iterations "
                f"({self.max_iterations})."
            )
            trace.append(
                TraceStep(
                    _local_now(_CLOCK_FORMAT),
                    self.name,
                    "final_answer",
                    final_text,
                )
            )

        response_type, output_content = _parse_response_content(final_text, self.name)

        resp = AgentResponse(
            agent_name=self.name,
            input_prompt=prompt or "",
            content=output_content,
            trace=trace,
            messages=self.messages,
            iterations=iteration,
            response_type=response_type,
        )
        self.last_response = resp
        return resp

    def as_tool(self, name: str | None = None, description: str | None = None) -> Tool:
        """Convert this agent into a callable Tool so other agents can invoke it.

        This is the foundation of hierarchical multi-agent orchestration.
        """
        tool_name = name or f"delegate_to_{re.sub(r'[^a-zA-Z0-9_]', '_', self.name.lower())}"
        tool_desc = (
            description
            or f"Delegates a sub-task to the specialist agent '{self.name}'. "
            f"Specialty/Role: {self.role}. "
            "Pass a clear, detailed task description for the agent to solve."
        )

        def agent_tool_function(task_description: str) -> str:
            """Execute a sub-task by delegating to this specialist agent.

            Args:
                task_description: Clear, detailed instruction for the specialist agent.
            """
            response = self.run(task_description)
            self.last_response = response
            if isinstance(response.content, (dict, list)):
                return json.dumps(response.content, indent=2)
            return str(response.content)

        agent_tool_function.__name__ = tool_name
        agent_tool_function.__doc__ = tool_desc

        return Tool(name=tool_name, description=tool_desc, func=agent_tool_function)


# ===========================================================================
# Orchestrator / Supervisor Agent
# ===========================================================================


class OrchestratorAgent(Agent):
    """Supervisor agent coordinating multiple sub-agents and tools to solve complex goals."""

    def __init__(
        self,
        name: str = "LeadOrchestrator",
        role: str | None = None,
        subagents: Sequence[Agent] | None = None,
        tools: Sequence[Tool | Callable[..., Any]] | None = None,
        llm: LLM | None = None,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 12,
        verbose: bool = True,
        description: str | None = None,
    ) -> None:
        self.subagents: dict[str, Agent] = {}
        orchestrator_tools: list[Tool | Callable[..., Any]] = []

        if tools:
            orchestrator_tools.extend(tools)

        subagent_descriptions = []
        if subagents:
            for sa in subagents:
                self.subagents[sa.name] = sa
                agent_as_tool = sa.as_tool()
                orchestrator_tools.append(agent_as_tool)
                subagent_descriptions.append(f"- **{sa.name}** ({agent_as_tool.name}): {sa.role}")

        subagent_roster = "\n".join(subagent_descriptions) if subagent_descriptions else "None"

        default_role = (
            "You are the Lead Multi-Agent Orchestrator and Supervisor.\n"
            "Your objective is to solve complex, multi-step queries by decomposing them into "
            "sub-tasks, delegating each sub-task to the most capable specialized sub-agent or "
            "tool, and synthesizing the findings into a cohesive, high-quality final report.\n\n"
            f"Your Team of Specialist Sub-Agents:\n{subagent_roster}\n\n"
            "Orchestration Protocol:\n"
            "1. Deconstruct the user prompt into logical steps.\n"
            "2. Invoke specialist sub-agents for domain-specific work.\n"
            "3. Use direct tools when quick calculations or utilities are required.\n"
            "4. If a sub-agent's output is incomplete or ambiguous, delegate a refined query.\n"
            "5. Synthesize all collected evidence into a structured final response for the user."
        )

        super().__init__(
            name=name,
            role=role or default_role,
            description=description,
            llm=llm,
            system_prompt=None,
            tools=orchestrator_tools,
            model=model,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    def add_subagent(self, subagent: Agent) -> None:
        """Add a new sub-agent to the orchestrator team dynamically."""
        self.subagents[subagent.name] = subagent
        self.add_tool(subagent.as_tool())


# ===========================================================================
# State Schema and Blackboard Architecture
# ===========================================================================


def store_tool_result(store: dict[str, Any], tool_name: str, value: Any) -> str:
    """Save one tool result in an agent's ``GraphState.tools`` entry without overwriting.

    The first call of a tool is stored under its name; repeated calls (an analyst
    searching both the 10-K and the 10-Q) are kept as ``name#2``, ``name#3``, ... so the
    Critic can still verify figures taken from any of them.

    Returns:
        The key the result was stored under.
    """
    key, n = tool_name, 1
    while key in store:
        n += 1
        key = f"{tool_name}#{n}"
    store[key] = value
    return key


@dataclass
class GraphState:
    """Central blackboard state object passed across nodes in the StateGraph."""

    question: str = ""
    output: str = ""
    state_unique_id: str = ""
    thread_id: str = "default_session"
    agents: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    refine_count: int = 0
    # Critic / Refiner review history, filled by ``workflows/review_loop.py``.
    review: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the declared fields as a JSON-serialisable dictionary."""
        return asdict(self)

    @property
    def tools_data(self) -> dict[str, Any]:
        """Alias of ``tools`` kept for notebooks written against the earlier API."""
        return self.tools

    def __getitem__(self, item: str) -> Any:
        """Dict-style access to fields, extra attributes and agent outputs.

        Raises:
            KeyError: If ``item`` is not a field, attribute or agent name.
        """
        if item == "tools_data":
            return self.tools
        if hasattr(self, item):
            return getattr(self, item)
        if item in self.agents:
            return self.agents[item]
        raise KeyError(item)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-style assignment; unknown keys become extra attributes."""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``self[key]``, or ``default`` when the key does not exist."""
        try:
            return self[key]
        except KeyError:
            return default


# ===========================================================================
# Graph Memory and Checkpoint Store
# ===========================================================================


class GraphMemory:
    """Checkpoint store for finished workflow runs, organised as thread_id -> states.

    Records are kept in memory and, when ``filepath`` is set, rewritten to a JSON file
    after every ``write``. It is not safe for concurrent writers across processes.
    """

    def __init__(self, filepath: str | None = None) -> None:
        self.filepath = filepath
        self.records: dict[str, dict[str, Any]] = {}
        if self.filepath and os.path.exists(self.filepath):
            self.load(self.filepath)

    def write(self, state: GraphState | dict[str, Any]) -> str:
        """Store a snapshot of ``state`` under its thread and return the new state id.

        The id is also written back onto ``state`` (``state_unique_id``).
        """
        now_str = _local_now(_STATE_ID_FORMAT)
        unique_id = f"state_{now_str}_{uuid.uuid4().hex[:6]}"

        thread_id = (
            getattr(state, "thread_id", None)
            or (state.get("thread_id") if isinstance(state, dict) else None)
            or "default_session"
        )

        if isinstance(state, GraphState):
            state.state_unique_id = unique_id
            state.thread_id = thread_id
            snapshot = state.to_dict()
        else:
            state["state_unique_id"] = unique_id
            state["thread_id"] = thread_id
            snapshot = dict(state)

        snapshot["state_unique_id"] = unique_id
        snapshot["thread_id"] = thread_id
        snapshot["timestamp"] = _local_now(_DATETIME_FORMAT)
        if "question" in snapshot and "input" not in snapshot:
            snapshot["input"] = snapshot["question"]

        if thread_id not in self.records:
            self.records[thread_id] = {"states": {}}
        elif "states" not in self.records[thread_id]:
            self.records[thread_id]["states"] = {}

        self.records[thread_id]["states"][unique_id] = snapshot
        if self.filepath:
            self.save(self.filepath)

        return unique_id

    def _all_states(self) -> dict[str, dict[str, Any]]:
        """Collects all states flattened across all thread containers."""
        all_s: dict[str, dict[str, Any]] = {}
        for thread_data in self.records.values():
            if isinstance(thread_data, dict) and "states" in thread_data:
                all_s.update(thread_data["states"])
            elif isinstance(thread_data, dict):
                all_s.update(thread_data)
        return all_s

    def get_history_states(
        self, page: int = 1, limit: int = 1, thread_id: str | None = None
    ) -> str:
        """Returns paginated historical records, optionally scoped to thread_id."""
        try:
            page = max(1, int(page))
        except (ValueError, TypeError):
            page = 1
        try:
            limit = max(1, int(limit))
        except (ValueError, TypeError):
            limit = 1

        if thread_id and thread_id in self.records:
            states_dict = self.records[thread_id].get("states", {})
        else:
            states_dict = self._all_states()

        all_keys = list(states_dict.keys())
        total_records = len(all_keys)
        total_pages = max(1, (total_records + limit - 1) // limit)
        page = min(page, total_pages)  # clamp out-of-range requests to the last page

        start = (page - 1) * limit
        end = start + limit
        page_keys = all_keys[start:end]

        result = {
            "page": page,
            "limit": limit,
            "total_records": total_records,
            "total_pages": total_pages,
            "has_next_page": page < total_pages,
            "has_prev_page": page > 1,
            "thread_id": thread_id,
            "records": [states_dict[k] for k in page_keys],
        }
        return json.dumps(result, indent=2, default=str)

    def get_state(self, unique_id: str, thread_id: str | None = None) -> str:
        """Retrieves single snapshot record by unique identifier."""
        if thread_id and thread_id in self.records:
            thread_states = self.records[thread_id].get("states", {})
            if unique_id in thread_states:
                return json.dumps(thread_states[unique_id], indent=2, default=str)

        all_states = self._all_states()
        if unique_id in all_states:
            return json.dumps(all_states[unique_id], indent=2, default=str)
        return json.dumps({"error": f"No state found with id '{unique_id}'"}, indent=2)

    def list_state_ids(self, thread_id: str | None = None) -> str:
        """Returns list of all stored checkpoint IDs, optionally filtered by thread_id."""
        if thread_id and thread_id in self.records:
            return json.dumps(list(self.records[thread_id].get("states", {}).keys()), indent=2)
        return json.dumps(list(self._all_states().keys()), indent=2)

    def list_threads(self) -> str:
        """Returns list of all active thread IDs in graph memory."""
        return json.dumps(list(self.records.keys()), indent=2)

    def as_tools(self) -> list[Tool]:
        """Exposes memory inspection endpoints as callable LLM tools."""
        return [
            Tool(
                name="get_history_states",
                description=(
                    "Retrieve paginated workflow history. Args: page (int), limit (int), "
                    "thread_id (str, optional). "
                    "Check total_pages and has_next_page to navigate."
                ),
                func=self.get_history_states,
            ),
            Tool(
                name="get_state",
                description="Retrieve full state details for a specific state_unique_id.",
                func=self.get_state,
            ),
            Tool(
                name="list_state_ids",
                description=(
                    "List recorded state_unique_id values in graph memory. "
                    "Optional arg: thread_id (str)."
                ),
                func=self.list_state_ids,
            ),
            Tool(
                name="list_threads",
                description="List all active thread IDs in graph memory.",
                func=self.list_threads,
            ),
        ]

    def save(self, filepath: str | None = None) -> None:
        """Write all records to ``filepath`` (defaults to the store's own file)."""
        target = filepath or self.filepath
        if target:
            os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2, default=str)

    def load(self, filepath: str) -> None:
        """Load records from disk, converting the legacy flat layout to threads.

        A missing, empty or corrupt file yields an empty store instead of an error.
        """
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            self.records = {}
            return

        with open(filepath, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                self.records = {}
                return

        is_threaded = (
            bool(data)
            and isinstance(data, dict)
            and all(isinstance(v, dict) and "states" in v for v in data.values())
        )
        if is_threaded:
            self.records = data
        else:
            self.records = {}
            for state_id, snapshot in data.items():
                if isinstance(snapshot, dict):
                    tid = snapshot.get("thread_id", "default_session")
                    if tid not in self.records:
                        self.records[tid] = {"states": {}}
                    self.records[tid]["states"][state_id] = snapshot


# ===========================================================================
# StateGraph Builder and Compiler
# ===========================================================================


class StateGraph:
    """Builder for a workflow graph of agent, function and parallel-group nodes.

    Cycles are allowed (e.g. critic -> refiner -> critic); ``invoke`` bounds them with
    ``max_steps``.
    """

    def __init__(
        self,
        state_schema: type = GraphState,
        memory: GraphMemory | None = None,
        verbose: bool = False,
    ) -> None:
        self.state_schema = state_schema
        self.memory = memory or GraphMemory()
        self.verbose = verbose
        self.nodes: dict[str, Any] = {}
        self.edges: dict[str, str] = {}
        self.conditional_edges: dict[str, tuple[Callable | Agent, dict[str, str] | None]] = {}
        self.entry_point: str | None = None

    def add_node(self, name: str, node_target: Agent | Callable | list[Agent]) -> StateGraph:
        """Registers a node (Agent, Callable, or parallel Agent list)."""
        self.nodes[name] = node_target
        return self

    def add_edge(self, from_node: str, to_node: str) -> StateGraph:
        """Defines a deterministic transition between two graph nodes."""
        self.edges[from_node] = to_node
        return self

    def add_conditional_edges(
        self,
        source: str,
        decider: Callable | Agent,
        route_map: dict[str, str] | None = None,
    ) -> StateGraph:
        """Defines dynamic branching based on function return or router agent classification."""
        self.conditional_edges[source] = (decider, route_map)
        return self

    def set_entry_point(self, node_name: str) -> StateGraph:
        """Sets the root starting node for graph execution."""
        self.entry_point = node_name
        return self

    def compile(self, draw: bool = False) -> CompiledStateGraph:
        """Validates and compiles the graph definition into an executable runtime."""
        if not self.entry_point:
            if self.nodes:
                self.entry_point = next(iter(self.nodes.keys()))
            else:
                raise ValueError("Graph has no nodes or entry point defined.")
        compiled = CompiledStateGraph(self)
        if draw:
            compiled.draw()
        return compiled


# ===========================================================================
# Compiled StateGraph Execution Runtime
# ===========================================================================


def _compact_text(text: Any, max_len: int = 100) -> str:
    """Truncates long strings or serializes structures for compact log rendering."""
    if isinstance(text, (dict, list)):
        try:
            s = json.dumps(text, default=str)
        except (TypeError, ValueError):
            s = str(text)
    else:
        s = str(text)
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:max_len] + "...") if len(s) > max_len else s


def _extract_tool_activity(trace: list[TraceStep]) -> list[dict[str, Any]]:
    """Extracts pairs of tool calls and resulting observations from an execution trace."""
    activities = []
    current_call = None
    for step in trace:
        if step.step_type == "tool_call":
            current_call = {
                "tool": step.content.get("tool", "unknown"),
                "arguments": step.content.get("arguments", {}),
                "observation": None,
            }
            activities.append(current_call)
        elif step.step_type == "tool_result" and current_call is not None:
            current_call["observation"] = step.content
            current_call = None
    return activities


_MERMAID_NODE_DESCRIPTIONS: dict[str, str] = {
    "ResearchPlanner": "<br/>(Intent Classifier & Memory Recall)",
    "ResearchRouter": "<br/>(Lead Orchestrator)",
    "DraftWriter": "<br/>(Synthesis & Report Generator)",
    "critic": "<br/>(Factual Audit & Grounding Gate)",
    "refiner": "<br/>(Targeted Section Optimizer)",
    "release": "<br/>(Publication & Release Gate)",
}

_MERMAID_EDGE_LABELS: dict[tuple[str, str], str] = {
    ("planner", "orchestrator"): "RESEARCH_REQUIRED",
    ("planner", "__end__"): "Direct Concept Answer",
    ("critic", "refiner"): "score < 8.0 / ungrounded",
    ("critic", "release"): "score >= 8.0",
}


def _format_node_mermaid(
    name: str, target: Any, subagent_links: list[tuple[str, str, str]]
) -> list[str]:
    """Format Mermaid node definition for a graph node."""
    lines: list[str] = []
    desc = _MERMAID_NODE_DESCRIPTIONS.get(
        getattr(target, "name", name), _MERMAID_NODE_DESCRIPTIONS.get(name, "")
    )
    if isinstance(target, list):
        names = ", ".join(a.name for a in target)
        lines.append(f'    {name}["Parallel: [{names}]"]')
    elif isinstance(target, Agent):
        subagents = getattr(target, "subagents", None)
        if isinstance(subagents, dict):
            lines.append(f'    {name}["Agent: {target.name}{desc}"]')
            lines.append("")
            lines.append(
                '    subgraph Specialists ["Specialist Sub-Agents & '
                'Tools (Dynamic ReAct Delegation)"]'
            )
            for sa_name, sa in subagents.items():
                sa_id = sa_name.lower()
                sa_tools = list(getattr(sa, "tools", {}))
                sa_tool_info = f"<br/>Tool: {sa_tools[0]}" if sa_tools else ""
                lines.append(f'        {sa_id}["{sa_name}{sa_tool_info}"]')
                subagent_links.append((name, sa_id, f"delegate_to_{sa_name.lower()}"))
            lines.append("    end")
        else:
            agent_tools = [
                t for t in getattr(target, "tools", {}) if not t.startswith("delegate_to_")
            ]
            domain_tools = [
                t for t in agent_tools if not t.startswith(("list_", "get_"))
            ] or agent_tools
            tool_info = f"<br/>Tool: {domain_tools[0]}" if domain_tools else ""
            lines.append(f'    {name}["Agent: {target.name}{desc}{tool_info}"]')
    elif callable(target):
        fn_name = getattr(target, "__name__", str(target))
        lines.append(f'    {name}["{name} (fn: {fn_name}){desc}"]')
    return lines


def _format_node_progress(node_name: str, node_target: Any = None) -> str:
    """Generically formats an agent progress message from the node target metadata."""
    name = ""
    if node_target is not None:
        if isinstance(node_target, Agent):
            name = node_target.name
        elif callable(node_target):
            name = getattr(node_target, "__name__", node_name).replace("_node", "").capitalize()
        elif isinstance(node_target, list):
            name = ", ".join(getattr(a, "name", str(a)) for a in node_target)
    if not name:
        name = node_name.replace("_node", "").capitalize()

    desc = getattr(node_target, "description", None) if node_target is not None else None
    if not desc and callable(node_target):
        doc = (inspect.getdoc(node_target) or "").strip().split("\n")[0]
        if doc:
            desc = doc
    if not desc:
        desc = f"Executing '{node_name}'"

    return f"Agent: [{name}] {desc}..."


def _format_args_preview(fn_args: dict[str, Any], max_len: int = 75) -> str:
    """Formats runtime tool keyword arguments compactly with smart truncation."""
    if not fn_args:
        return ""
    items = []
    for k, v in fn_args.items():
        if isinstance(v, str):
            cleaned = " ".join(v.strip().split())
            val_repr = f"'{cleaned}'" if len(cleaned) <= 35 else f"'{cleaned[:32]}...'"
        else:
            val_repr = repr(v)
            if len(val_repr) > 35:
                val_repr = val_repr[:32] + "..."
        items.append(f"{k}={val_repr}")
    full = ", ".join(items)
    if len(full) > max_len:
        return full[: max_len - 3] + "..."
    return full


def _format_tool_progress(agent: Agent, fn_name: str, fn_args: dict[str, Any]) -> str:
    """Generically formats tool execution progress showing agent, tool name, and call arguments."""
    args_preview = _format_args_preview(fn_args)
    return f"Tool: [{agent.name}] {fn_name}({args_preview})"


def _attach_tool_hooks(
    agent: Agent, on_progress: Callable[[str], None] | None
) -> Callable[[], None]:
    """Attaches progress listeners to agent tools and returns a cleanup callback."""
    if on_progress is None:
        return lambda: None

    targets = [agent]
    if hasattr(agent, "subagents") and isinstance(agent.subagents, dict):
        targets.extend(agent.subagents.values())

    for tgt in targets:

        def _make_hook(target_agent: Agent) -> Callable[[str, str, dict[str, Any]], None]:
            return lambda _aname, tname, targs: on_progress(
                _format_tool_progress(target_agent, tname, targs)
            )

        tgt._on_tool_call = _make_hook(tgt)

    def _cleanup() -> None:
        for tgt in targets:
            if hasattr(tgt, "_on_tool_call"):
                tgt._on_tool_call = None

    return _cleanup


class CompiledStateGraph:
    """Immutable execution engine compiled from StateGraph."""

    def __init__(self, graph: StateGraph) -> None:
        self.graph = graph
        self.memory = graph.memory
        self.verbose = graph.verbose

    def to_mermaid(self) -> str:
        """Constructs a Mermaid flowchart definition representing graph topology."""
        lines = ["flowchart TD"]
        entry = self.graph.entry_point
        lines.append(f"    __start__([Start]) --> {entry}")

        subagent_links: list[tuple[str, str, str]] = []

        for name, target in self.graph.nodes.items():
            lines.extend(_format_node_mermaid(name, target, subagent_links))

        for src, sa_id, call_name in subagent_links:
            lines.append(f"    {src} -.->|{call_name}| {sa_id}")

        for src, dst in self.graph.edges.items():
            dst_lbl = "__end__([End])" if dst == END else dst
            lines.append(f"    {src} --> {dst_lbl}")

        for src, (_, route_map) in self.graph.conditional_edges.items():
            if route_map:
                for cond, dst in route_map.items():
                    dst_lbl = "__end__([End])" if dst == END else dst
                    lbl = _MERMAID_EDGE_LABELS.get((src, dst), cond)
                    lines.append(f"    {src} -.->|{lbl}| {dst_lbl}")
            else:
                lines.append(f"    {src} -.-> __end__([End])")

        return "\n".join(lines)

    def draw(self, as_image: bool = True) -> str:
        """Show the graph as a Mermaid diagram in Jupyter, or log it elsewhere.

        Args:
            as_image: Render through mermaid.ink as an image instead of a code block.

        Returns:
            The Mermaid source.
        """
        chart = self.to_mermaid()
        try:
            import base64
            import urllib.request

            from IPython.display import Image, Markdown, display

            if as_image:
                img_data: bytes | None = None
                try:
                    req = urllib.request.Request(
                        "https://kroki.io/mermaid/png",
                        data=chart.encode("utf-8"),
                        headers={
                            "Content-Type": "text/plain; charset=utf-8",
                            "User-Agent": "Mozilla/5.0",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=12) as resp:  # noqa: S310
                        img_data = resp.read()
                except (urllib.error.URLError, TimeoutError, OSError):
                    b64 = base64.b64encode(chart.encode("utf-8")).decode("ascii")
                    url = f"https://mermaid.ink/img/{b64}"
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with (
                        contextlib.suppress(urllib.error.URLError, TimeoutError, OSError),
                        urllib.request.urlopen(req, timeout=8) as resp,  # noqa: S310
                    ):
                        img_data = resp.read()

                if img_data:
                    display(Image(data=img_data))
                else:
                    b64 = base64.b64encode(chart.encode("utf-8")).decode("ascii")
                    display(Image(url=f"https://mermaid.ink/img/{b64}"))
            else:
                display(Markdown(f"```mermaid\n{chart}\n```"))
        except Exception:  # noqa: BLE001 - no IPython / display: fall back to the log
            logger.info("\n[MERMAID GRAPH]:\n%s", chart)
        return chart

    def _initialize_state(self, state: GraphState | dict[str, Any]) -> GraphState:
        """Instantiates and aligns input state against the defined schema."""
        if isinstance(state, self.graph.state_schema):
            return state  # type: ignore[return-value]  # schema is a GraphState subclass

        if isinstance(state, dict):
            schema_fields = set()
            if hasattr(self.graph.state_schema, "__dataclass_fields__"):
                schema_fields = set(self.graph.state_schema.__dataclass_fields__.keys())

            valid_kwargs = (
                {k: v for k, v in state.items() if k in schema_fields} if schema_fields else state
            )
            try:
                current_state = self.graph.state_schema(**valid_kwargs)
            except TypeError:
                current_state = self.graph.state_schema()

            for k, v in state.items():
                if k not in schema_fields:
                    setattr(current_state, k, v)
            return current_state

        return self.graph.state_schema()

    def _build_agent_prompt(self, agent: Agent, state: GraphState) -> str:
        """Constructs comprehensive prompt injecting blackboard state and prior agent outputs."""
        prompt_parts = []
        user_query = getattr(state, "question", "") or getattr(state, "input", "")
        if user_query:
            prompt_parts.append(f"User Request / Goal:\n{user_query}")

        # Extra payload keys (e.g. active_ticker, conversation_context) are shown to
        # every agent as context; internal and empty values are skipped.
        custom_attrs = {
            k: v
            for k, v in state.__dict__.items()
            if k not in _STATE_BASE_KEYS and not k.startswith("_") and v is not None and v != ""
        }

        if custom_attrs:
            prompt_parts.append(
                f"Current Context / Attributes:\n{json.dumps(custom_attrs, indent=2, default=str)}"
            )

        if state.agents:
            other_agents = {k: v for k, v in state.agents.items() if k != agent.name and v}
            if other_agents:
                prompt_parts.append(
                    f"Prior Specialist Findings:\n{json.dumps(other_agents, indent=2, default=str)}"
                )

        return "\n\n".join(prompt_parts)

    def _harvest_subagent_tools(self, subagent: Agent, sa_name: str, state: GraphState) -> None:
        """Harvest observations and findings from an orchestrator's delegated subagent."""
        last_resp = getattr(subagent, "last_response", None)
        if last_resp is not None:
            if sa_name not in state.agents and last_resp.content:
                state.agents[sa_name] = last_resp.content
            self._harvest_agent_tools(subagent, last_resp, state)
            return

        if subagent.messages and len(subagent.messages) > 1:
            last_asst_msg = next(
                (
                    m.get("content")
                    for m in reversed(subagent.messages)
                    if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content")
                ),
                None,
            )
            if last_asst_msg and sa_name not in state.agents:
                state.agents[sa_name] = last_asst_msg

            if sa_name not in state.tools:
                state.tools[sa_name] = {}
            for m in subagent.messages:
                if isinstance(m, dict) and m.get("role") == "tool":
                    c = m.get("content")
                    if isinstance(c, str):
                        with contextlib.suppress(json.JSONDecodeError, TypeError):
                            c = json.loads(c)
                    store_tool_result(state.tools[sa_name], "data", c)

    def _harvest_agent_tools(self, agent: Agent, resp: AgentResponse, state: GraphState) -> None:
        """Harvests observations from agent traces and updates state.tools."""
        if agent.name not in state.tools:
            state.tools[agent.name] = {}

        for step in resp.trace:
            if step.step_type == "tool_result":
                tool_name = step.metadata.get("tool", "unknown_tool")
                val = step.content
                if isinstance(val, str):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        val = json.loads(val)
                store_tool_result(state.tools[agent.name], tool_name, val)

        # If agent is an Orchestrator with subagents, harvest subagent tools and content
        if hasattr(agent, "subagents") and isinstance(agent.subagents, dict):
            for sa_name, subagent in agent.subagents.items():
                self._harvest_subagent_tools(subagent, sa_name, state)

    def _execute_parallel_node(
        self,
        agents: list[Agent],
        node_name: str,
        state: GraphState,
        step_count: int,
        start_time: str,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Run a group of independent agents (sequentially) as one workflow node."""
        active_agents = agents

        if self.verbose:
            names = ", ".join(a.name for a in active_agents)
            logger.info(">> STEP %s | NODE: %s [PARALLEL: %s]", step_count, node_name, names)

        tools_called = set()
        for agent in active_agents:
            p = self._build_agent_prompt(agent, state)
            resp = agent.run(p, on_progress=on_progress)
            state.agents[agent.name] = resp.content
            self._harvest_agent_tools(agent, resp, state)
            for t in agent.tools:
                if any(
                    step.step_type == "tool_call" and step.content.get("tool") == t
                    for step in resp.trace
                ):
                    tools_called.add(t)

            activities = _extract_tool_activity(resp.trace)
            if self.verbose:
                logger.info("  --> Agent Completed: '%s'", agent.name)
                logger.info("      - Role   : %s", _compact_text(agent.role, 80))
                if activities:
                    logger.info("      - Tools Executed:")
                    for act in activities:
                        args_str = ", ".join(f"{k}={v!r}" for k, v in act["arguments"].items())
                        logger.info("        [TOOL CALL]    %s(%s)", act["tool"], args_str)
                        if act["observation"] is not None:
                            logger.info(
                                "        [OBSERVATION]  %s", _compact_text(act["observation"], 90)
                            )
                logger.info("      - Output : %s", _compact_text(resp.content, 120))

        active_names = [a.name for a in active_agents]
        summary = f"Executed {len(active_agents)} parallel specialists: {active_names}"

        state.trajectory.append(
            {
                "step": step_count,
                "timestamp": start_time,
                "node": node_name,
                "type": "parallel",
                "agents": [a.name for a in active_agents],
                "tools": sorted(tools_called),
                "summary": summary,
            }
        )

    def _extract_subagent_delegations(
        self, agent: Agent, tools_called: list[str]
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Collects called subagents and their executed tools from an orchestrator."""
        subagents_called: list[str] = []
        delegations: dict[str, list[str]] = {}
        if hasattr(agent, "subagents") and isinstance(agent.subagents, dict):
            for sa_name, subagent in agent.subagents.items():
                last_resp = getattr(subagent, "last_response", None)
                if last_resp is not None:
                    subagents_called.append(sa_name)
                    sa_tools = [
                        t
                        for t in subagent.tools
                        if any(
                            step.step_type == "tool_call" and step.content.get("tool") == t
                            for step in last_resp.trace
                        )
                    ]
                    delegations[sa_name] = sa_tools
                    for t in sa_tools:
                        if t not in tools_called:
                            tools_called.append(t)
        return subagents_called, delegations

    def _execute_agent_node(
        self,
        agent: Agent,
        node_name: str,
        state: GraphState,
        step_count: int,
        start_time: str,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Run a single agent node and record its output and tool usage."""
        cleanup = _attach_tool_hooks(agent, on_progress)
        try:
            p = self._build_agent_prompt(agent, state)
            resp = agent.run(p)
            state.agents[agent.name] = resp.content
            self._harvest_agent_tools(agent, resp, state)
        finally:
            cleanup()

        tools_called = [
            t
            for t in agent.tools
            if any(
                step.step_type == "tool_call" and step.content.get("tool") == t
                for step in resp.trace
            )
        ]
        subagents_called, delegations = self._extract_subagent_delegations(agent, tools_called)
        activities = _extract_tool_activity(resp.trace)

        if self.verbose:
            logger.info(">> STEP %s | NODE: %s [AGENT: %s]", step_count, node_name, agent.name)
            logger.info("  - Role    : %s", _compact_text(agent.role, 80))
            if activities:
                logger.info("  - Tools Executed:")
                for act in activities:
                    args_str = ", ".join(f"{k}={v}" for k, v in act["arguments"].items())
                    logger.info("    [TOOL CALL]    %s(%s)", act["tool"], args_str)
                    if act["observation"] is not None:
                        logger.info("    [OBSERVATION]  %s", _compact_text(act["observation"], 90))
            logger.info("  - Output  : %s", _compact_text(resp.content, 120))

        step_entry: dict[str, Any] = {
            "step": step_count,
            "timestamp": start_time,
            "node": node_name,
            "type": "agent",
            "executor": agent.name,
            "tools": tools_called,
            "output": resp.content,
            "summary": _compact_text(resp.content, 80),
        }
        if subagents_called:
            step_entry["subagents"] = subagents_called
            step_entry["delegations"] = delegations
        if activities:
            step_entry["activities"] = activities
        state.trajectory.append(step_entry)

    def _execute_function_node(
        self,
        fn: Callable,
        node_name: str,
        state: GraphState,
        step_count: int,
        start_time: str,
    ) -> GraphState:
        """Executes a Python function node and updates the state blackboard."""
        res = fn(state)
        fn_name = getattr(fn, "__name__", str(fn))
        updates = {}

        if isinstance(res, dict):
            for k, v in res.items():
                setattr(state, k, v)
                updates[k] = _compact_text(v, 40)
        elif isinstance(res, GraphState):
            state = res
            updates["state"] = "replaced with new GraphState"

        update_str = (
            ", ".join(f"{k}={v}" for k, v in updates.items()) if updates else "no state delta"
        )
        if self.verbose:
            logger.info(">> STEP %s | NODE: %s [FUNCTION: %s]", step_count, node_name, fn_name)
            logger.info("  - Updates : %s", update_str)

        state.trajectory.append(
            {
                "step": step_count,
                "timestamp": start_time,
                "node": node_name,
                "type": "function",
                "executor": fn_name,
                "tools": [],
                "updates": updates,
                "summary": f"State updated: {update_str}",
            }
        )
        return state

    def _determine_next_node(self, current_node: str, state: GraphState) -> str:
        """Determines the subsequent node to execute using conditional routers or edges."""
        if current_node in self.graph.conditional_edges:
            decider, route_map = self.graph.conditional_edges[current_node]
            route_key = ""
            if isinstance(decider, Agent):
                decider_prompt = (
                    f"Evaluate current workflow state and determine next routing branch:\n"
                    f"{json.dumps(state.to_dict(), indent=2, default=str)}"
                )
                resp = decider.run(decider_prompt)
                route_key = resp.content.strip().lower()
            elif callable(decider):
                route_key = str(decider(state)).strip().lower()

            next_node = None
            if route_map:
                for k, target in route_map.items():
                    if k.lower() in route_key:
                        next_node = target
                        break
                if next_node is None:
                    next_node = route_map.get("default", END)
            else:
                next_node = route_key

            if self.verbose:
                logger.info(
                    "  --> Transition: '%s' --> '%s' (Condition: '%s')",
                    current_node,
                    next_node,
                    route_key,
                )
            return next_node

        if current_node in self.graph.edges:
            next_node = self.graph.edges[current_node]
            if self.verbose:
                logger.info("  --> Transition: '%s' --> '%s'", current_node, next_node)
            return next_node

        if self.verbose:
            logger.info("  --> Transition: '%s' --> END (Workflow Completed)", current_node)
        return END

    def _finalize_output(self, state: GraphState) -> None:
        """Set ``state.output`` from the best content-producing agent.

        The critic's score is never used as the answer. A leading "Change Summary"
        written by the refiner is stripped so only the polished brief remains.
        """
        if not state.output:
            producer = next((name for name in _OUTPUT_AGENT_PRIORITY if name in state.agents), None)
            if producer is not None:
                state.output = state.agents[producer]
            elif state.trajectory:
                reversed_steps = [
                    s
                    for s in reversed(state.trajectory)
                    if s.get("node", "").lower() not in {"critic", "router"}
                ]
                if reversed_steps:
                    last_step = reversed_steps[0]
                    state.output = str(last_step.get("output", ""))

        if state.output and "### Change Summary" in state.output:
            for header in ["\n# ", "\n## ", "\n---\n"]:
                if header in state.output:
                    idx = state.output.find(header)
                    state.output = state.output[idx:].strip()
                    break

    def invoke(
        self,
        state: GraphState | dict[str, Any],
        max_steps: int = 25,
        on_step: Callable[[dict[str, Any]], None] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> GraphState:
        """Run the graph from the entry point until ``END`` or ``max_steps``.

        Args:
            state: A ``GraphState`` or a payload dict. Unknown dict keys become extra
                state attributes and are shown to agents as context.
            max_steps: Safety limit on node executions (guards against routing loops).
            on_step: Optional callback invoked with the latest step dict after each node runs.
            on_progress: Optional callback invoked with live thinking/action messages.

        Returns:
            The final state; the run is also checkpointed in ``memory``.

        Raises:
            ValueError: If the graph has no entry point or routes to an unknown node.
        """
        current_state = self._initialize_state(state)
        if not self.graph.entry_point:
            raise ValueError("StateGraph has no entry_point defined.")
        current_node: str = self.graph.entry_point
        step_count = 0

        user_query = getattr(current_state, "question", "") or getattr(current_state, "input", "")

        if self.verbose:
            logger.info("=" * 80)
            logger.info("[STATE GRAPH RUN START]")
            logger.info("   - Entry Point : %s", current_node)
            if user_query:
                logger.info('   - User Query  : "%s"', _compact_text(user_query, 80))
            logger.info("=" * 80)

        while current_node != END and step_count < max_steps:
            step_count += 1
            node_target = self.graph.nodes.get(current_node)
            if node_target is None:
                raise ValueError(f"Target node '{current_node}' not found in graph nodes.")

            if on_progress is not None:
                progress_msg = _format_node_progress(current_node, node_target)
                if progress_msg:
                    on_progress(progress_msg)

            node_start_time = _local_now(_CLOCK_FORMAT)

            if isinstance(node_target, list):
                self._execute_parallel_node(
                    node_target,
                    current_node,
                    current_state,
                    step_count,
                    node_start_time,
                    on_progress=on_progress,
                )
            elif isinstance(node_target, Agent):
                self._execute_agent_node(
                    node_target,
                    current_node,
                    current_state,
                    step_count,
                    node_start_time,
                    on_progress=on_progress,
                )
            elif callable(node_target):
                current_state = self._execute_function_node(
                    node_target,
                    current_node,
                    current_state,
                    step_count,
                    node_start_time,
                )

            if on_step is not None and current_state.trajectory:
                on_step(current_state.trajectory[-1])

            current_node = self._determine_next_node(current_node, current_state)

        self._finalize_output(current_state)
        new_state_id = self.memory.write(current_state)

        if self.verbose:
            logger.info("=" * 80)
            logger.info(
                "[SUCCESS] STATE GRAPH RUN COMPLETED | Total Steps: %s | Persisted State ID: %s",
                step_count,
                new_state_id,
            )
            logger.info("=" * 80)

        return current_state


# ===========================================================================
# Trajectory and Telemetry Visualizers
# ===========================================================================


_TRAJECTORY_HEADERS = ("Step", "Node", "Type", "Executor", "Tools Called", "Summary / Decision")


def _format_step_summary(step: dict[str, Any], max_len: int = 80) -> str:
    """Return a one-line summary of a trajectory step for tables."""
    summary = step.get("summary", "")
    if not summary:
        output_val = step.get("output", "")
        if output_val:
            first_line = str(output_val).strip().split("\n")[0]
            summary = (first_line[:max_len] + "...") if len(first_line) > max_len else first_line
        elif step.get("updates"):
            summary = f"Updated: {list(step['updates'].keys())}"
        else:
            summary = "Executed successfully"
    return summary.replace("\n", " ").strip()


def _step_executor(step: dict[str, Any]) -> str:
    """Return who ran a step: the agent/function name, or the parallel agent names."""
    return str(
        step.get("executor") or step.get("agent") or ", ".join(step.get("agents", [])) or "-"
    )


def format_trajectory_markdown(trajectory: list[dict[str, Any]]) -> str:
    """Render a trajectory as a GitHub-flavoured Markdown table.

    Args:
        trajectory: ``GraphState.trajectory`` rows.

    Returns:
        The table, or an empty string when there are no steps.
    """
    if not trajectory:
        return ""

    md_lines = [
        "| " + " | ".join(_TRAJECTORY_HEADERS) + " |",
        "| :---: | :--- | :--- | :--- | :--- | :--- |",
    ]
    for step in trajectory:
        tools = step.get("tools", [])
        cells = (
            step.get("step", "-"),
            f"`{step.get('node', '-')}`",
            str(step.get("type", "-")).capitalize(),
            _step_executor(step),
            ", ".join(f"`{t}`" for t in tools) if tools else "-",
            # Escape pipes so free text cannot break the table layout.
            _format_step_summary(step, max_len=80).replace("|", "\\|"),
        )
        md_lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return "\n".join(md_lines)


def display_trajectory(state: GraphState | dict[str, Any], as_markdown: bool | None = None) -> Any:
    """Show the execution trajectory as a Markdown (Jupyter) or ASCII (log) table.

    Args:
        state: Final ``GraphState`` or a dict with a ``trajectory`` key.
        as_markdown: Force the format; by default Markdown is used inside Jupyter.

    Returns:
        The rendered table text, or None when it was displayed in Jupyter or there is
        nothing to show.
    """
    trajectory = getattr(state, "trajectory", None)
    if trajectory is None and isinstance(state, dict):
        trajectory = state.get("trajectory", [])
    if not trajectory:
        logger.info("No trajectory recorded for this state.")
        return None

    in_ipython = False
    with contextlib.suppress(Exception):
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None and "IPKernelApp" in ip.config:
            in_ipython = True

    use_markdown = as_markdown if as_markdown is not None else in_ipython

    if use_markdown:
        md_content = "### Execution Trajectory Summary\n\n" + format_trajectory_markdown(trajectory)
        try:
            from IPython.display import Markdown, display

            display(Markdown(md_content))
        except Exception:  # noqa: BLE001 - outside Jupyter: return the text instead
            return md_content
        return None

    headers = ["Step", "Node", "Type", "Executor", "Tools Used", "Summary / Decision"]
    rows = []
    for step in trajectory:
        s_num = str(step.get("step", "-"))
        s_node = str(step.get("node", "-"))
        s_type = str(step.get("type", "-")).capitalize()
        s_exec = _step_executor(step)
        tools = step.get("tools", [])
        s_tools = ", ".join(tools) if tools else "-"
        s_summary = _format_step_summary(step, max_len=65)
        rows.append([s_num, s_node, s_type, s_exec, s_tools, s_summary])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = (
        "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=False)) + " |"
    )

    output_lines = ["\n[EXECUTION TRAJECTORY TABLE]", sep, header_line, sep]
    for row in rows:
        row_line = (
            "| " + " | ".join(val.ljust(w) for val, w in zip(row, col_widths, strict=False)) + " |"
        )
        output_lines.append(row_line)
    output_lines.append(sep)
    result = "\n".join(output_lines)
    logger.info(result)
    return result
