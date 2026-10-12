"""
interfaces/chat_ui.py
=====================
Gradio chat UI for the Investment Research Agent ("focused thread" layout).

Layout:
    * **Left rail** - brand, "New research", every stored thread (click to resume),
      resume-by-id and the tickers this conversation store has researched.
    * **Thread** - title + thread tag, export button, the chat, context-aware
      suggestion chips and a rounded composer. Each agent answer is preceded by a
      collapsible "Reasoning · N steps" block built from the workflow trajectory.

Each browser tab gets its own ``ResearchSession`` (held in a ``gr.State``), so
follow-up questions like "What about its risk factors?" resolve to the active ticker
exactly as they do in the CLI or the notebook.

Visual style follows the Organic theme: cream ground, terracotta accent, sage second
accent, Caprasimo headings over Figtree, pill-shaped controls.

Run with:
    python gradio_app.py
"""

from __future__ import annotations

import html
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gradio as gr

from research_agent.config import logger
from research_agent.memory.session import ResearchSession
from research_agent.memory.session_store import SessionStore

APP_TITLE = "Investment Research Agent"
APP_TAGLINE = "Equity research copilot"
THINKING_MESSAGE = "_Researching... this can take a few minutes on a local model._"
DISCLAIMER = "Not investment advice · mock data covers NVDA, AAPL and MSFT."
CHAT_PLACEHOLDER = (
    "## Start a research thread\n"
    'Ask about a ticker (e.g. **"Analyse NVDA"**) or a finance concept '
    '(e.g. **"What is operating margin?"**). Follow-ups reuse the active ticker.'
)
STARTER_PROMPTS: tuple[str, ...] = ("Analyse NVDA", "Analyse AAPL", "What is operating margin?")
FOLLOW_UP_TEMPLATES: tuple[str, ...] = (
    "What about its risk factors?",
    "Summarise {ticker}'s latest earnings",
    "How is {ticker} valued against peers?",
)
MAX_THREADS_IN_RAIL = 20

ChatHistory = list[dict[str, Any]]

# --------------------------------------------------------------------------- theme

_ACCENT = gr.themes.Color(
    c50="#fff2eb",
    c100="#ffe1d0",
    c200="#ffc6a5",
    c300="#f6a06b",
    c400="#d67f48",
    c500="#c67139",
    c600="#b2622d",
    c700="#8c491a",
    c800="#643312",
    c900="#402310",
    c950="#2b170a",
    name="terracotta",
)
_SAGE = gr.themes.Color(
    c50="#f0fae1",
    c100="#e1eecc",
    c200="#ccdbb2",
    c300="#aebf92",
    c400="#8fa073",
    c500="#7a8a5e",
    c600="#728157",
    c700="#56633f",
    c800="#3d472b",
    c900="#272e1b",
    c950="#1a1f12",
    name="sage",
)
_SAND = gr.themes.Color(
    c50="#f9f4ed",
    c100="#eee7db",
    c200="#dcd3c4",
    c300="#c0b6a5",
    c400="#a19786",
    c500="#82796a",
    c600="#645c50",
    c700="#474238",
    c800="#2e2b25",
    c900="#201e1d",
    c950="#141312",
    name="sand",
)


def make_theme() -> gr.themes.Base:
    """Return the Organic-styled Gradio theme."""
    return gr.themes.Base(
        primary_hue=_ACCENT,
        secondary_hue=_SAGE,
        neutral_hue=_SAND,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Figtree"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#f5ead8",
        body_text_color="#201e1d",
        body_text_color_subdued="#645c50",
        background_fill_primary="#f5ead8",
        background_fill_secondary="#ebddc5",
        block_background_fill="transparent",
        block_border_width="0px",
        block_radius="28px",
        block_label_text_color="#645c50",
        border_color_primary="#dcd3c4",
        color_accent_soft="#ffe1d0",
        input_background_fill="#f9f4ed",
        input_radius="999px",
        button_large_radius="999px",
        button_small_radius="999px",
        button_primary_background_fill="#c67139",
        button_primary_background_fill_hover="#b2622d",
        button_primary_text_color="#f5ead8",
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_hover="#eee7db",
        button_secondary_border_color="#dcd3c4",
        button_secondary_text_color="#201e1d",
        shadow_drop="0 3px 10px rgba(46, 43, 37, 0.16)",
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Caprasimo&display=swap');
.gradio-container{max-width:none!important;padding:0!important;background:#f5ead8}
footer{display:none!important}
h1,h2,h3{font-family:Caprasimo,serif;font-weight:400}
:focus-visible{outline:2px solid #c67139;outline-offset:2px}
#shell{gap:0;min-height:100vh;flex-wrap:nowrap;align-items:stretch}
#rail{background:#ebddc5;padding:22px 16px;gap:14px;border-radius:0 28px 28px 0}
.brand{display:flex;align-items:center;gap:10px;padding:0 6px}
.brand-mark{width:32px;height:32px;flex:none;border-radius:50%;background:#7a8a5e;
  color:#f5ead8;display:grid;place-items:center;font-family:Caprasimo,serif;font-size:17px}
.brand-name{font-family:Caprasimo,serif;font-size:18px;line-height:1.1}
.brand-sub{font-size:12px;color:#645c50}
.rail-label{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#645c50;
  padding:4px 8px 0}
.rail-empty{font-size:13px;color:#645c50;padding:0 8px}
#threads .wrap{flex-direction:column;gap:4px}
#threads label{width:100%;background:transparent;border:0;border-radius:18px;
  padding:10px 12px;box-shadow:none;font-size:14px;cursor:pointer}
#threads label:hover{background:rgba(32,30,29,.07)}
#threads label.selected,#threads label:has(input:checked){background:#f5ead8;font-weight:600}
#threads input{display:none}
#resume{background:transparent;border:0}
.tickers{display:flex;flex-direction:column;gap:8px;padding:12px;border-radius:20px;
  background:#f5ead8}
.tickers-row{display:flex;flex-wrap:wrap;gap:6px}
.pill,button.pill{border-radius:999px!important}
#thread{padding:20px 40px 22px;gap:12px}
#thread>*{max-width:820px;width:100%;margin-inline:auto}
#thread-head{align-items:center;flex-wrap:nowrap}
.thread-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.thread-head h2{font-size:22px;margin:0}
.tag{display:inline-flex;align-items:center;font-size:11px;letter-spacing:.02em;
  padding:3px 10px;border-radius:999px}
.tag-sage{background:#f0fae1;color:#3d472b}
.tag-terra{background:#fff2eb;color:#643312}
#chatbot{background:transparent;border:0}
#suggestions{gap:8px;flex-wrap:wrap}
.suggestion{border-radius:999px!important;font-weight:400!important;color:#8c491a!important;
  background:#fff2eb!important;border:0!important;min-width:0!important;flex:none!important}
.suggestion:hover{background:#ffe1d0!important}
#composer{background:#f9f4ed;border-radius:28px!important;border:0;
  box-shadow:0 3px 10px rgba(46,43,37,.16);padding:8px 10px 8px 16px}
#composer textarea{background:transparent!important;border:0!important;box-shadow:none!important;
  font-size:16px}
#composer-meta{justify-content:space-between;flex-wrap:wrap}
#composer-meta p{margin:0;font-size:12px;color:#645c50}
#disclaimer p{text-align:right}
@media (max-width:820px){
  #shell{flex-wrap:wrap}#rail{border-radius:0 0 28px 28px;min-width:100%!important}
  #thread{padding:16px}
}
"""

BRAND_HTML = f"""
<div class="brand">
  <div class="brand-mark">R</div>
  <div><div class="brand-name">{APP_TITLE}</div><div class="brand-sub">{APP_TAGLINE}</div></div>
</div>
"""

# --------------------------------------------------------------------------- helpers


def _new_session(thread_id: str | None = None) -> ResearchSession:
    """Create a ``ResearchSession`` with a fresh (or given) thread id."""
    tid = thread_id.strip() if thread_id and thread_id.strip() else f"gradio-{uuid.uuid4().hex[:8]}"
    return ResearchSession(thread_id=tid)


def _active_ticker_label(session: ResearchSession | None) -> str:
    if session is None:
        return "Active ticker: none yet"
    ticker = session.active_ticker
    return f"Active ticker: **{ticker}**" if ticker else "Active ticker: none yet"


def _one_line(text: Any, limit: int = 160) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _join(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    return str(value or "")


def _trajectory_markdown(trajectory: Sequence[Mapping[str, Any]]) -> str:
    """Render workflow trajectory steps as a numbered Markdown list."""
    lines = []
    for i, step in enumerate(trajectory, 1):
        name = step.get("node") or step.get("executor") or step.get("type") or f"Step {i}"
        line = f"{i}. **{name}**"
        if summary := step.get("summary"):
            line += f" — {_one_line(summary)}"
        if tools := _join(step.get("tools")):
            line += f"  \n   _tools: {tools}_"
        lines.append(line)
    return "\n".join(lines)


def _reasoning_message(session: Any) -> dict[str, Any] | None:
    """Build the collapsible reasoning message for the last workflow run, if any."""
    state = getattr(session, "last_state", None)
    trajectory = list(getattr(state, "trajectory", None) or [])
    if not trajectory:
        return None
    n = len(trajectory)
    return {
        "role": "assistant",
        "content": _trajectory_markdown(trajectory),
        "metadata": {"title": f"Reasoning · {n} step{'s' if n != 1 else ''}", "status": "done"},
    }


def _list_threads() -> list[dict[str, Any]]:
    try:
        return [dict(t) for t in SessionStore().list_threads()]
    except Exception:  # noqa: BLE001 - the rail is optional; never break the chat over it
        logger.exception("Gradio UI: could not list threads")
        return []


def _threads_for(session: Any) -> list[dict[str, Any]]:
    lister = getattr(session, "list_sessions", None)
    return lister() if callable(lister) else _list_threads()


def _thread_choices(threads: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Rail entries as ``(label, thread_id)``; empty threads are hidden."""
    choices = []
    for t in threads:
        turns = int(t.get("turns") or 0)
        if not turns:
            continue
        ticker = t.get("active_ticker") or "General"
        date = str(t.get("updated_at") or "")[:10]
        label = f"{ticker} · {turns} turn{'s' if turns != 1 else ''}"
        choices.append((f"{label} · {date}" if date else label, t["thread_id"]))
    return choices[:MAX_THREADS_IN_RAIL]


def _tickers_html(threads: Sequence[Mapping[str, Any]]) -> str:
    tickers = list(dict.fromkeys(t["active_ticker"] for t in threads if t.get("active_ticker")))
    if not tickers:
        return ""
    tags = "".join(f'<span class="tag tag-sage">{html.escape(t)}</span>' for t in tickers[:12])
    return (
        f'<div class="tickers"><div class="rail-label">Researched</div>'
        f'<div class="tickers-row">{tags}</div></div>'
    )


def _header_html(session: Any) -> str:
    ticker = getattr(session, "active_ticker", None) if session is not None else None
    title = f"{ticker} research" if ticker else "New research"
    tag = ""
    if session is not None:
        tag = f'<span class="tag tag-terra">thread · {html.escape(session.thread_id)}</span>'
    return f'<div class="thread-head"><h2>{html.escape(title)}</h2>{tag}</div>'


def _suggestions(session: Any) -> list[str]:
    ticker = getattr(session, "active_ticker", None) if session is not None else None
    if not ticker:
        return list(STARTER_PROMPTS)
    return [tpl.format(ticker=ticker) for tpl in FOLLOW_UP_TEMPLATES]


def export_thread(session: Any) -> str | None:
    """Write the thread as a Markdown report and return its path (None if empty)."""
    history = list(getattr(session, "history", None) or []) if session is not None else []
    if not history:
        return None
    ticker = getattr(session, "active_ticker", None) or "General"
    parts = [f"# Research report — {ticker}", f"_Thread: {session.thread_id}_", ""]
    for i, turn in enumerate(history, 1):
        parts += [f"## {i}. {turn['question']}", "", turn["answer"], ""]
    parts.append(f"---\n{DISCLAIMER}")
    out = Path(tempfile.gettempdir()) / f"research-{session.thread_id}.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return str(out)


# --------------------------------------------------------------------------- handlers


def start_session(thread_id: str) -> tuple[ResearchSession, ChatHistory, str]:
    """Start a brand-new session, or resume ``thread_id`` if it already has turns.

    Returns the session, the chat history to display and the active-ticker label.
    """
    session = _new_session(thread_id)
    history: ChatHistory = []
    for turn in session.history:
        history.append({"role": "user", "content": turn["question"]})
        history.append({"role": "assistant", "content": turn["answer"]})
    logger.info(
        "Gradio UI: started session thread=%s resumed=%s", session.thread_id, session.resumed
    )
    return session, history, _active_ticker_label(session)


def respond(message: str, history: ChatHistory, session: ResearchSession | None) -> Any:
    """Handle one chat turn: echo the question, show a status line, then the answer."""
    history = list(history or [])
    message = (message or "").strip()

    if session is None:
        session = _new_session()

    if not message:
        reply = "Please ask a question about a stock, e.g. 'Analyse NVDA'."
        history.append({"role": "assistant", "content": reply})
        yield history, session, _active_ticker_label(session), ""
        return

    history.append({"role": "user", "content": message})
    history.append(
        {
            "role": "assistant",
            "content": THINKING_MESSAGE,
            "metadata": {"title": "Researching…", "status": "pending"},
        }
    )
    yield history, session, _active_ticker_label(session), ""

    reasoning = None
    try:
        answer = session.ask(message, show=False) or "No response generated."
        reasoning = _reasoning_message(session)
    except Exception as exc:  # noqa: BLE001 - show the failure in the chat instead of crashing the UI
        logger.exception("Gradio UI: workflow failed for thread %s", session.thread_id)
        answer = f"Something went wrong while researching that: {exc}"

    history.pop()
    if reasoning:
        history.append(reasoning)
    history.append({"role": "assistant", "content": answer})
    yield history, session, _active_ticker_label(session), ""


def refresh(session: Any) -> tuple[Any, ...]:
    """Recompute the rail, header, export file and suggestion chips for ``session``."""
    threads = _threads_for(session)
    choices = _thread_choices(threads)
    current = session.thread_id if session is not None else None
    value = current if current in {tid for _, tid in choices} else None
    path = export_thread(session)
    return (
        gr.Radio(choices=choices, value=value),
        _header_html(session),
        _tickers_html(threads),
        gr.DownloadButton(value=path, interactive=bool(path)),
        *(gr.Button(value=s) for s in _suggestions(session)),
    )


def _clear() -> tuple[ResearchSession, ChatHistory, str, str, str]:
    session, history, label = start_session("")
    return session, history, label, "", session.thread_id


# --------------------------------------------------------------------------- app


def build_app() -> gr.Blocks:
    """Assemble the Gradio Blocks app."""
    with gr.Blocks(title=APP_TITLE) as demo:
        session_state = gr.State(None)

        with gr.Row(equal_height=False, elem_id="shell"):
            with gr.Column(scale=0, min_width=260, elem_id="rail"):
                gr.HTML(BRAND_HTML)
                new_btn = gr.Button("New research", variant="primary", elem_classes="pill")
                gr.HTML('<div class="rail-label">Threads</div>')
                threads = gr.Radio(choices=[], show_label=False, container=False, elem_id="threads")
                with gr.Accordion("Resume by thread id", open=False, elem_id="resume"):
                    thread_box = gr.Textbox(
                        show_label=False, placeholder="e.g. gradio-1a2b3c4d", container=False
                    )
                    resume_btn = gr.Button("Start / Resume", size="sm", elem_classes="pill")
                tickers_html = gr.HTML("")

            with gr.Column(scale=1, elem_id="thread"):
                with gr.Row(elem_id="thread-head"):
                    header = gr.HTML(_header_html(None))
                    export_btn = gr.DownloadButton(
                        "Export report",
                        variant="secondary",
                        size="sm",
                        interactive=False,
                        scale=0,
                        min_width=140,
                        elem_classes="pill",
                    )

                chatbot = gr.Chatbot(
                    show_label=False,
                    height="62vh",
                    placeholder=CHAT_PLACEHOLDER,
                    elem_id="chatbot",
                )

                with gr.Row(elem_id="suggestions"):
                    sugg_btns = [
                        gr.Button(
                            p,
                            size="sm",
                            variant="secondary",
                            scale=0,
                            elem_classes="suggestion",
                        )
                        for p in STARTER_PROMPTS
                    ]

                with gr.Group(elem_id="composer"):
                    msg_box = gr.Textbox(
                        show_label=False,
                        placeholder="Ask about a ticker, or a follow-up…",
                        lines=2,
                        max_lines=8,
                        submit_btn=True,
                        container=False,
                    )
                with gr.Row(elem_id="composer-meta"):
                    ticker_label = gr.Markdown(_active_ticker_label(None))
                    gr.Markdown(DISCLAIMER, elem_id="disclaimer")

        refresh_outputs = [threads, header, tickers_html, export_btn, *sugg_btns]
        turn_outputs = [chatbot, session_state, ticker_label, msg_box]
        session_outputs = [session_state, chatbot, ticker_label]

        demo.load(refresh, inputs=[session_state], outputs=refresh_outputs)

        msg_box.submit(
            respond, inputs=[msg_box, chatbot, session_state], outputs=turn_outputs
        ).then(refresh, inputs=[session_state], outputs=refresh_outputs)

        for btn in sugg_btns:
            btn.click(respond, inputs=[btn, chatbot, session_state], outputs=turn_outputs).then(
                refresh, inputs=[session_state], outputs=refresh_outputs
            )

        resume_btn.click(start_session, inputs=[thread_box], outputs=session_outputs).then(
            refresh, inputs=[session_state], outputs=refresh_outputs
        )
        threads.input(start_session, inputs=[threads], outputs=session_outputs).then(
            refresh, inputs=[session_state], outputs=refresh_outputs
        )
        new_btn.click(
            _clear, outputs=[session_state, chatbot, ticker_label, msg_box, thread_box]
        ).then(refresh, inputs=[session_state], outputs=refresh_outputs)

    return demo


def main() -> None:
    """Launch the Gradio app (blocking)."""
    build_app().launch(theme=make_theme(), css=CSS)


if __name__ == "__main__":
    main()
