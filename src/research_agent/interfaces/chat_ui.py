"""
interfaces/chat_ui.py
=====================
Gradio chat UI for the Investment Research Agent ("focused thread" layout).

Layout:
    * **Left rail** - brand and "New research" at the top; a scrollable list of every stored
      conversation (click to resume, the cross deletes it); "Clear all conversations" (with a
      confirm step) and the light/dark switch fixed at the bottom.
    * **Thread** - title + thread tag, export button, the chat, context-aware
      suggestion chips and a rounded composer. Each agent answer is preceded by a
      collapsible "Reasoning · N steps" block built from the workflow trajectory.
      While a turn runs, the pending message shows the agent and tool steps live.

Each browser tab gets its own ``ResearchSession`` (held in a ``gr.State``), so
follow-up questions like "What about its risk factors?" resolve to the active ticker
exactly as they do in the CLI or the notebook.

The look is a modern chat-app layout: zinc neutrals with an indigo-to-violet accent, card-style
answers ("Research brief" header, styled tables and section headings) and a live step timeline
while a turn runs (Inter and JetBrains Mono). The page opens in dark mode; "Light / dark" in the
rail switches palettes.

Run with:
    python gradio_app.py
"""

from __future__ import annotations

import html
import inspect
import queue
import re
import tempfile
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import gradio as gr

from research_agent.config import logger
from research_agent.memory.session import ResearchSession
from research_agent.memory.session_store import SessionStore

APP_TITLE = "Investment Research Agent"
THINKING_MESSAGE = "A full research brief takes about 2-4 minutes; follow-ups are faster."
DISCLAIMER = "Not investment advice · live market data"
STARTER_PROMPTS: tuple[str, ...] = ("Analyse NVDA", "Analyse AAPL", "What is operating margin?")
FOLLOW_UP_TEMPLATES: tuple[str, ...] = (
    "What about its risk factors?",
    "Summarise {ticker}'s latest earnings",
    "How is {ticker} valued against peers?",
)
MAX_THREADS_IN_RAIL = 20
# Answers longer than this (or with headings or tables) are shown as a full "Research brief".
BRIEF_MIN_CHARS = 600
# Live progress: how often (seconds) the UI checks the running workflow for new steps, and how
# long a tool call may be before it is shortened in the step log.
PROGRESS_POLL_SECONDS = 0.5
PROGRESS_CALL_CHARS = 110
# Progress lines from the workflow look like "Agent: [Name] what it is doing" and
# "Tool: [Name] tool_name(arguments)".
_PROGRESS_LINE = re.compile(r"^(Agent|Tool):\s*\[([^\]]+)\]\s*(.*)$")

ChatHistory = list[dict[str, Any]]

# --------------------------------------------------------------------------- theme


def make_theme() -> gr.themes.Base:
    """Return the Gradio theme: zinc neutrals with an indigo accent.

    Gradio reads the ``*_dark`` values in dark mode (the default here, see ``THEME_JS``) and the
    plain values in light mode. Everything the theme cannot express is in ``CSS``.
    """
    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.zinc,
        radius_size=gr.themes.sizes.radius_md,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#f7f7f8",
        body_background_fill_dark="#0b0b0e",
        body_text_color="#18181b",
        body_text_color_dark="#ececf1",
        body_text_color_subdued="#71717a",
        body_text_color_subdued_dark="#8e8e99",
        background_fill_primary="#f7f7f8",
        background_fill_primary_dark="#0b0b0e",
        background_fill_secondary="#ffffff",
        background_fill_secondary_dark="#16161a",
        block_background_fill="transparent",
        block_background_fill_dark="transparent",
        block_border_width="0px",
        block_radius="12px",
        block_label_text_color="#71717a",
        block_label_text_color_dark="#8e8e99",
        border_color_primary="#e4e4e7",
        border_color_primary_dark="#27272d",
        color_accent_soft="#eef2ff",
        color_accent_soft_dark="#1e1b3a",
        input_background_fill="#ffffff",
        input_background_fill_dark="#16161a",
        input_radius="12px",
        button_large_radius="8px",
        button_small_radius="8px",
        button_primary_background_fill="#4f46e5",
        button_primary_background_fill_dark="#818cf8",
        button_primary_background_fill_hover="#4338ca",
        button_primary_background_fill_hover_dark="#a5b4fc",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#0b0b0e",
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_dark="transparent",
        button_secondary_background_fill_hover="#f4f4f5",
        button_secondary_background_fill_hover_dark="#1d1d22",
        button_secondary_border_color="#e4e4e7",
        button_secondary_border_color_dark="#27272d",
        button_secondary_text_color="#18181b",
        button_secondary_text_color_dark="#ececf1",
        shadow_drop="none",
    )


# Every colour is a variable, defined for light mode and again for dark mode (Gradio puts the
# ``dark`` class on <body>), so the rules below serve both themes.
CSS = """
:root{color-scheme:light;
  --ira-ground:#f7f7f8;--ira-rail:#ffffff;--ira-surface:#ffffff;--ira-surface-2:#f4f4f5;
  --ira-text:#18181b;--ira-muted:#71717a;--ira-border:#e4e4e7;--ira-hover:rgba(24,24,27,.045);
  --ira-accent:#4f46e5;--ira-accent-2:#7c3aed;--ira-accent-soft:#eef2ff;--ira-accent-text:#4338ca;
  --ira-good:#059669;--ira-bad:#dc2626;--ira-shadow:0 1px 2px rgba(16,24,40,.05),
  0 8px 24px -12px rgba(16,24,40,.12);
  --ira-mono:'JetBrains Mono',ui-monospace,monospace}
.dark{color-scheme:dark;
  --ira-ground:#0b0b0e;--ira-rail:#111114;--ira-surface:#16161a;--ira-surface-2:#1d1d22;
  --ira-text:#ececf1;--ira-muted:#8e8e99;--ira-border:#27272d;--ira-hover:rgba(255,255,255,.045);
  --ira-accent:#818cf8;--ira-accent-2:#a78bfa;--ira-accent-soft:rgba(129,140,248,.13);
  --ira-accent-text:#a5b4fc;--ira-good:#34d399;--ira-bad:#f87171;
  --ira-shadow:0 1px 2px rgba(0,0,0,.4),0 12px 32px -16px rgba(0,0,0,.6)}
.gradio-container{max-width:none!important;padding:0!important;background:var(--ira-ground)}
/* Gradio pads and centres the app; the layout below fills the whole window instead. */
.gradio-container .main.app{padding:0!important;max-width:none!important}
footer{display:none!important}
:focus-visible{outline:2px solid var(--ira-accent);outline-offset:2px}

/* ---- layout: fixed-height shell; the conversation list and the thread scroll independently */
#shell{gap:0;height:100vh;overflow:hidden;flex-wrap:nowrap;align-items:stretch}
#rail{background:var(--ira-rail);border-right:1px solid var(--ira-border);padding:16px 12px 12px;
  gap:4px;height:100vh;overflow:hidden;flex-wrap:nowrap!important}
#rail>*{flex-shrink:0}
#thread{padding:0 40px 14px;gap:12px;height:100vh;overflow-y:auto;flex-wrap:nowrap!important}
#thread>*{flex-shrink:0;max-width:880px;width:100%;margin-inline:auto}

/* ---- rail: brand and "New research" on top */
.brand{display:flex;align-items:center;gap:10px;padding:4px 6px 14px}
.brand-mark{width:30px;height:30px;flex:none;border-radius:9px;display:grid;place-items:center;
  background:linear-gradient(135deg,var(--ira-accent),var(--ira-accent-2));color:#fff;
  box-shadow:0 4px 12px -4px var(--ira-accent)}
.brand-mark svg{width:17px;height:17px}
.brand-mark svg,.ira-hero-mark svg,.ira-report-icon svg{stroke:#fff!important;fill:none!important}
.brand-text{min-width:0;display:flex;flex-direction:column;line-height:1.2}
.brand-name{font-size:14px;font-weight:650;color:var(--ira-text);line-height:1.25}
.brand-sub{font-size:11px;color:var(--ira-muted)}
#new-research{justify-content:center;gap:6px;border:0!important;color:#fff!important;
  font-weight:600!important;font-size:13px!important;padding:9px 12px!important;
  background:linear-gradient(135deg,var(--ira-accent),var(--ira-accent-2))!important;
  border-radius:10px!important;box-shadow:0 6px 16px -8px var(--ira-accent)}
#new-research:hover{filter:brightness(1.08)}
.rail-label{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ira-muted);padding:18px 8px 6px}

/* ---- rail: the conversation list (HTML from _rail_html) fills the middle and scrolls */
#rail>#thread-list{flex:1 1 0!important;min-height:0;overflow-y:auto!important;scrollbar-width:thin;
  padding:0!important;border:0!important;background:transparent!important}
#rail>#rail-foot{flex:none!important}
/* The hidden bridge that carries a click on the list to Python (see RAIL_JS). */
#rail-cmd,#rail-go{display:none!important}
.thread-list{display:flex;flex-direction:column;gap:2px}
.thread-row{display:flex;align-items:center;border-radius:10px;position:relative;
  transition:background .15s}
.thread-row:hover{background:var(--ira-hover)}
.thread-row.current{background:var(--ira-accent-soft)}
.thread-row.current::before{content:"";position:absolute;left:0;top:10px;bottom:10px;width:3px;
  border-radius:3px;background:var(--ira-accent)}
.thread-open{flex:1 1 auto;min-width:0;display:flex;flex-direction:column;gap:1px;text-align:left;
  border:0;background:transparent;cursor:pointer;padding:8px 12px;font:inherit;
  color:var(--ira-muted)}
.thread-title{font-size:13.5px;font-weight:600;color:var(--ira-text);white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.thread-row.current .thread-title{color:var(--ira-accent-text)}
.thread-meta{font-size:12px}
.thread-del{flex:none;height:28px;min-width:28px;margin-right:6px;padding:0 8px;border:0;
  border-radius:8px;background:transparent;color:var(--ira-muted);font:inherit;font-size:12px;
  cursor:pointer;opacity:0;transition:opacity .15s}
/* First click arms the delete ("Delete?"); only a second click on the same row deletes. */
.thread-del.armed{opacity:1;background:var(--ira-bad);color:#fff;font-weight:600}
.thread-row:hover .thread-del,.thread-row.current .thread-del,.thread-del:focus-visible{opacity:1}
.thread-del:not(.armed):hover{color:var(--ira-bad);background:var(--ira-hover)}
.rail-empty{font-size:12px;color:var(--ira-muted);padding:6px 10px}

/* ---- rail: "Clear all conversations" and the theme switch pinned to the bottom */
#rail-foot{border-top:1px solid var(--ira-border);padding-top:8px;gap:2px;
  flex-wrap:nowrap!important}
.rail-action{justify-content:flex-start!important;border:0!important;border-radius:8px!important;
  background:transparent!important;box-shadow:none!important;color:var(--ira-muted)!important;
  font-weight:500!important;font-size:12.5px!important}
.rail-action:hover{color:var(--ira-text)!important;background:var(--ira-hover)!important}
#clear-all:hover{color:var(--ira-bad)!important}
#confirm-clear{gap:8px;padding:10px;border:1px solid var(--ira-border);border-radius:10px;
  background:var(--ira-surface-2)}
.confirm-text{font-size:12px;line-height:1.45;color:var(--ira-text)}
#confirm-clear .row{gap:6px;flex-wrap:nowrap}
#confirm-clear-yes{background:var(--ira-bad)!important;color:#fff!important;border:0!important;
  font-weight:600!important}

/* ---- header */
#thread-head{position:sticky;top:0;z-index:5;align-items:center;flex-wrap:nowrap;gap:12px;
  padding:16px 0 12px;background:var(--ira-ground);border-bottom:1px solid var(--ira-border)}
#thread-head>:first-child{flex:1 1 auto;min-width:0}
#export{flex:none!important;white-space:nowrap;border:1px solid var(--ira-border)!important;
  background:var(--ira-surface)!important;color:var(--ira-text)!important;font-weight:500!important;
  border-radius:9px!important}
#export::before{content:"\\2193";margin-right:6px;color:var(--ira-accent)}
#export:disabled{opacity:.45}
.thread-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.thread-head h2{font-size:17px;font-weight:650;margin:0;color:var(--ira-text);letter-spacing:-.01em}
.tag{font-family:var(--ira-mono);font-size:11px;padding:3px 8px;border-radius:999px;
  border:1px solid var(--ira-border);color:var(--ira-muted)}
.tag-ticker{display:inline-flex;align-items:center;gap:6px;font-weight:600;
  color:var(--ira-accent-text);border-color:transparent;background:var(--ira-accent-soft)}
.tag-ticker::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--ira-good)}
.quiet{border:0!important;background:transparent!important;color:var(--ira-muted)!important;
  font-weight:400!important}
.quiet:hover{color:var(--ira-text)!important}

/* ---- chat. 100vh minus the header, suggestions, composer, footer line and paddings, so the
   composer and footer always stay on screen. */
#chatbot,#chatbot .bubble-wrap{background:transparent!important;border:0!important}
#chatbot{height:calc(100vh - 270px)!important;min-height:320px}
/* Gradio's own "Clear" icon only empties the screen, not the stored conversation; deleting is
   done from the rail instead, so the icon is hidden to avoid two meanings of "clear". */
#chatbot .icon-button-wrapper.top-panel{display:none}
#chatbot .placeholder{display:grid;place-items:center;height:100%}
.ira-hero{display:flex;flex-direction:column;align-items:center;text-align:center;gap:10px;
  max-width:560px;margin:auto;padding:24px}
.ira-hero-mark{width:56px;height:56px;border-radius:16px;display:grid;place-items:center;
  color:#fff;background:linear-gradient(135deg,var(--ira-accent),var(--ira-accent-2));
  box-shadow:0 12px 32px -12px var(--ira-accent)}
.ira-hero-mark svg{width:28px;height:28px}
.ira-hero h2{font-size:24px!important;font-weight:700!important;letter-spacing:-.02em;
  margin:8px 0 0!important;color:var(--ira-text)!important;text-transform:none!important;
  font-family:inherit!important;border:0!important;padding:0!important}
.ira-hero p{margin:0;font-size:14px;line-height:1.6;color:var(--ira-muted)}
.ira-hero-tags{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin-top:8px}
.ira-hero-tags span{font-size:12px;padding:5px 11px;border-radius:999px;
  border:1px solid var(--ira-border);background:var(--ira-surface);color:var(--ira-muted)}
#chatbot .message-row{margin-block:6px}
#chatbot .message{border-radius:14px}
#chatbot .user-row .message{background:var(--ira-accent-soft)!important;border:0!important;
  color:var(--ira-text);font-weight:500;padding:10px 16px!important}
/* Gradio nests a second .message inside each bubble; only the outer one is styled. */
#chatbot .user-row .message .message{background:transparent!important;padding:0!important}
#chatbot .bot-row .message:not(.thought){background:transparent!important;border:0!important;
  padding:0!important;box-shadow:none!important;width:100%;max-width:100%}
#chatbot .message.thought{background:var(--ira-surface)!important;
  border:1px solid var(--ira-border)!important;border-radius:12px!important;
  font-size:12.5px;color:var(--ira-muted)}
#chatbot .message.thought .prose{color:var(--ira-muted)}

/* ---- answer card (built by _answer_message) */
.ira-report{background:var(--ira-surface);border:1px solid var(--ira-border);border-radius:16px;
  box-shadow:var(--ira-shadow);padding:0 24px;overflow:hidden}
.ira-report-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 -24px 6px;
  padding:12px 24px;border-bottom:1px solid var(--ira-border);background:var(--ira-surface-2)}
.ira-report-icon{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;
  color:#fff;background:linear-gradient(135deg,var(--ira-accent),var(--ira-accent-2))}
.ira-report-icon svg{width:14px;height:14px}
.ira-report-title{font-size:13px;font-weight:650;color:var(--ira-text)}
.ira-report-meta{margin-left:auto;font-family:var(--ira-mono);font-size:11px;
  color:var(--ira-muted)}
.ira-report-short{padding-bottom:4px}
.ira-report-foot{margin:16px -24px 0;padding:10px 24px;border-top:1px solid var(--ira-border);
  font-size:11.5px;color:var(--ira-muted);background:var(--ira-surface-2)}
.ira-error{border:1px solid var(--ira-bad);border-radius:12px;padding:12px 16px;
  background:var(--ira-surface);color:var(--ira-text)}
#chatbot .prose{font-size:14.5px;line-height:1.7;color:var(--ira-text)}
#chatbot .prose p{margin:10px 0}
#chatbot .prose strong{font-weight:650;color:var(--ira-text)}
#chatbot .prose h1{font-size:21px;font-weight:700;letter-spacing:-.015em;margin:18px 0 8px}
#chatbot .prose h2{font-size:16px;font-weight:650;margin:26px 0 10px;padding-bottom:6px;
  border-bottom:1px solid var(--ira-border)}
#chatbot .prose h3{font-size:12px;font-weight:650;letter-spacing:.07em;text-transform:uppercase;
  color:var(--ira-accent-text);margin:22px 0 8px}
#chatbot .prose ul,#chatbot .prose ol{padding-left:20px;margin:8px 0}
#chatbot .prose li{margin:5px 0}
#chatbot .prose li::marker{color:var(--ira-accent)}
#chatbot .prose a{color:var(--ira-accent-text);text-decoration:underline;
  text-underline-offset:2px}
#chatbot .prose blockquote{margin:14px 0;padding:10px 14px;border-left:3px solid var(--ira-accent);
  border-radius:0 10px 10px 0;background:var(--ira-accent-soft);color:var(--ira-text)}
#chatbot .prose hr{border:0;border-top:1px solid var(--ira-border);margin:20px 0}
#chatbot .prose table{border-collapse:separate;border-spacing:0;width:100%;font-size:13px;
  margin:14px 0;border:1px solid var(--ira-border);border-radius:12px;overflow:hidden;
  display:table}
#chatbot .prose thead th{background:var(--ira-surface-2);font-size:11.5px;font-weight:600;
  text-transform:uppercase;letter-spacing:.05em;color:var(--ira-muted);text-align:left}
#chatbot .prose th,#chatbot .prose td{padding:9px 14px;border:0;
  border-bottom:1px solid var(--ira-border)}
#chatbot .prose tbody tr:last-child td{border-bottom:0}
#chatbot .prose tbody tr:hover td{background:var(--ira-hover)}
#chatbot .prose td{font-variant-numeric:tabular-nums}
#chatbot .prose code{font-family:var(--ira-mono);font-size:12.5px;padding:1px 5px;
  border-radius:5px;background:var(--ira-surface-2)}

/* ---- live progress card (built by _pending_message) */
.ira-progress{border:1px solid var(--ira-border);border-radius:16px;padding:16px 20px;
  background:var(--ira-surface);box-shadow:var(--ira-shadow);position:relative;overflow:hidden}
.ira-progress::before{content:"";position:absolute;inset:0 0 auto;height:2px;
  background:linear-gradient(90deg,transparent,var(--ira-accent),var(--ira-accent-2),transparent);
  background-size:200% 100%;animation:ira-scan 1.6s linear infinite}
@keyframes ira-scan{from{background-position:200% 0}to{background-position:-200% 0}}
.ira-progress-head{display:flex;align-items:center;gap:8px;font-size:14px;color:var(--ira-text)}
.ira-progress-head .ira-count{color:var(--ira-muted);font-size:12px}
.ira-elapsed{margin-left:auto;font-family:var(--ira-mono);font-size:11px;padding:3px 9px;
  border-radius:999px;background:var(--ira-surface-2);color:var(--ira-muted)}
.ira-dot{width:8px;height:8px;border-radius:50%;background:var(--ira-accent);
  box-shadow:0 0 0 0 var(--ira-accent);animation:ira-pulse 1.4s ease-out infinite}
@keyframes ira-pulse{0%{box-shadow:0 0 0 0 var(--ira-accent-soft)}
  100%{box-shadow:0 0 0 8px transparent}}
.ira-note{margin:6px 0 12px;font-size:12.5px;color:var(--ira-muted)}
.ira-steps{list-style:none;margin:0;padding:0 0 0 14px;display:flex;flex-direction:column;gap:8px;
  border-left:2px solid var(--ira-border)}
.ira-steps>li{margin:0;padding:8px 12px;border-radius:10px;background:var(--ira-surface-2);
  position:relative}
.ira-steps>li::before{content:"";position:absolute;left:-20px;top:14px;width:10px;height:10px;
  border-radius:50%;background:var(--ira-good);border:2px solid var(--ira-surface)}
.ira-steps>li.ira-active{background:var(--ira-accent-soft)}
.ira-steps>li.ira-active::before{background:var(--ira-accent)}
.ira-agent{font-size:13px;color:var(--ira-text)}
.ira-agent .ira-detail{color:var(--ira-muted)}
.ira-badge{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;
  padding:2px 7px;margin-left:8px;border-radius:999px;background:var(--ira-accent);color:#fff}
.ira-calls{list-style:none;margin:6px 0 0;padding:0}
.ira-calls li{margin:3px 0;font-family:var(--ira-mono);font-size:11.5px;color:var(--ira-muted);
  overflow-wrap:anywhere}
.ira-calls li::before{content:"\\21B3  ";color:var(--ira-accent)}

/* ---- suggestions and composer */
/* While a turn runs (the "Researching" card is on screen) the chips and the composer are
   locked, so a second question cannot start a parallel turn that overwrites the first. */
body:has(#chatbot .ira-progress) .suggestion,
body:has(#chatbot .ira-progress) #composer,
body:has(#chatbot .ira-progress) #thread-list{pointer-events:none;opacity:.55}
#suggestions{gap:8px;flex-wrap:nowrap!important;align-items:center;overflow-x:auto;
  scrollbar-width:none}
.suggestion{border:1px solid var(--ira-border)!important;background:var(--ira-surface)!important;
  color:var(--ira-text)!important;font-weight:500!important;font-size:12.5px!important;
  border-radius:999px!important;padding:6px 14px!important;min-width:0!important;
  flex:none!important;white-space:nowrap;transition:border-color .15s,color .15s}
.suggestion::before{content:"\\2197";margin-right:6px;color:var(--ira-accent)}
.suggestion:hover{border-color:var(--ira-accent)!important;color:var(--ira-accent-text)!important}
#composer{background:var(--ira-surface);border:1px solid var(--ira-border)!important;
  border-radius:16px!important;padding:6px 8px 6px 14px;box-shadow:var(--ira-shadow);
  transition:border-color .15s,box-shadow .15s}
#composer:focus-within{border-color:var(--ira-accent)!important;
  box-shadow:0 0 0 4px var(--ira-accent-soft)}
#composer .styler{background:transparent!important}
#composer .gr-group,#composer .wrap,#composer label,#composer .input-container,
#composer .container{border:0!important;box-shadow:none!important;
  background:transparent!important}
#composer .block{border:0!important;background:transparent!important;box-shadow:none!important}
#composer textarea{background:transparent!important;border:0!important;box-shadow:none!important;
  font-size:14.5px;color:var(--ira-text)}
#composer button{background:linear-gradient(135deg,var(--ira-accent),var(--ira-accent-2))!important;
  color:#fff!important;border-radius:11px!important}
#composer-meta{justify-content:space-between;flex-wrap:wrap}
#composer-meta p{margin:0;font-size:11.5px;color:var(--ira-muted)}
#disclaimer p{text-align:right}
@media (max-width:820px){
  #shell{flex-wrap:wrap;height:auto;overflow:visible}
  #rail{min-width:100%!important;height:auto;max-height:45vh;border-right:0}
  #thread{padding:0 16px 16px;height:auto;overflow:visible}
  #chatbot{height:60vh!important}
}
"""

# Dark is the default: unless the address already asks for a theme, reload once with
# ?__theme=dark (?__theme=light keeps light mode, so a light link can still be shared).
THEME_JS = """
() => {
  const url = new URL(window.location.href);
  const theme = url.searchParams.get("__theme");
  if (theme !== "dark" && theme !== "light") {
    url.searchParams.set("__theme", "dark");
    window.location.replace(url.href);
  }
}
"""

# The "Light / dark" button: switch Gradio's dark class in place (no reload, so the chat is
# kept) and record the choice in the address, so a page reload keeps it.
TOGGLE_THEME_JS = """
() => {
  const dark = document.body.classList.toggle("dark");
  const url = new URL(window.location.href);
  url.searchParams.set("__theme", dark ? "dark" : "light");
  window.history.replaceState(null, "", url.href);
}
"""

_LOGO_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M3 17l5-5 4 4 8-8"/><path d="M15 8h5v5"/></svg>'
)
_REPORT_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></svg>'
)

# Clicks on the conversation list. The list is plain HTML (see _rail_html), and one delegated
# listener turns each click into exactly one Python call: it writes "<action>|<thread id>|<click
# id>" into a hidden textbox and presses a hidden button. Delete needs two clicks on the same
# row (the first one arms it and shows "Delete?"; it disarms itself after a few seconds).
RAIL_JS = """
() => {
  if (window.__iraRail) return;
  window.__iraRail = true;
  const send = (action, tid) => {
    const box = document.querySelector("#rail-cmd textarea, #rail-cmd input");
    const go = document.querySelector("#rail-go");
    if (!box || !go) return;
    box.value = action + "|" + tid + "|" + Date.now() + Math.random().toString(16).slice(2, 8);
    box.dispatchEvent(new Event("input", { bubbles: true }));
    setTimeout(() => go.click(), 0);
  };
  document.addEventListener("click", (event) => {
    const el = event.target.closest("[data-rail-action]");
    if (!el) return;
    event.preventDefault();
    event.stopPropagation();
    const tid = el.dataset.tid;
    if (el.dataset.railAction === "open") { send("open", tid); return; }
    if (!el.classList.contains("armed")) {
      document.querySelectorAll(".thread-del.armed").forEach((b) => {
        b.classList.remove("armed"); b.textContent = "\u2715";
      });
      el.classList.add("armed");
      el.textContent = "Delete?";
      setTimeout(() => { el.classList.remove("armed"); el.textContent = "\u2715"; }, 4000);
      return;
    }
    el.classList.remove("armed");
    send("delete", tid);
  }, true);
}
"""

# Runs once when the page loads: the dark-mode default, then the conversation-list bridge.
PAGE_JS = f"() => {{ ({THEME_JS.strip()})(); ({RAIL_JS.strip()})(); }}"

BRAND_HTML = f"""
<div class="brand">
  <div class="brand-mark">{_LOGO_SVG}</div>
  <div class="brand-text">
    <div class="brand-name">{APP_TITLE}</div>
    <div class="brand-sub">Multi-agent equity research</div>
  </div>
</div>
"""


# The empty-chat hero: what the agent does and what it draws on.
CHAT_PLACEHOLDER = (
    '<div class="ira-hero">'
    f'<div class="ira-hero-mark">{_LOGO_SVG}</div>'
    "<h2>What should we research today?</h2>"
    '<p>Ask about a ticker, like "Analyse NVDA", or a finance concept, like "What is operating '
    'margin?". A team of agents gathers earnings, market data and news, then a critic reviews '
    "the brief. Follow-ups reuse the active ticker.</p>"
    '<div class="ira-hero-tags"><span>Earnings &amp; filings</span><span>Market data</span>'
    "<span>News sentiment</span><span>Critic review</span></div>"
    "</div>"
)


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


def _step_groups(steps: Sequence[str]) -> list[tuple[str, str, list[str]]]:
    """Group live progress lines into ``(agent, what it is doing, [tool calls])`` turns.

    A tool call made by a different agent than the current group starts that agent's group, so
    each specialist's calls are listed under it.
    """
    groups: list[tuple[str, str, list[str]]] = []
    for raw in steps:
        match = _PROGRESS_LINE.match(" ".join(str(raw).split()))
        if not match:
            groups.append(("", _one_line(raw, PROGRESS_CALL_CHARS), []))
            continue
        kind, agent, detail = match.groups()
        if kind == "Agent" or not groups or groups[-1][0] != agent:
            groups.append((agent, detail if kind == "Agent" else "", []))
        if kind == "Tool":
            groups[-1][2].append(_one_line(detail, PROGRESS_CALL_CHARS))
    return groups


def _step_log(steps: Sequence[str]) -> str:
    """The live step log as HTML: one card per agent turn, its tool calls underneath, and the
    latest agent marked as active.
    """
    items = []
    groups = _step_groups(steps)
    for i, (agent, detail, calls) in enumerate(groups, 1):
        active = i == len(groups)
        name = f"<b>{i}. {html.escape(agent)}</b>" if agent else f"<b>{i}.</b>"
        what = f' <span class="ira-detail">{html.escape(detail)}</span>' if detail else ""
        badge = '<span class="ira-badge">active</span>' if active else ""
        tools = "".join(f"<li>{html.escape(call)}</li>" for call in calls)
        items.append(
            f'<li class="ira-step{" ira-active" if active else ""}">'
            f'<div class="ira-agent">{name}{what}{badge}</div>'
            + (f'<ul class="ira-calls">{tools}</ul>' if tools else "")
            + "</li>"
        )
    return f'<ol class="ira-steps">{"".join(items)}</ol>'


def _pending_message(steps: Sequence[str], elapsed: float | None = None) -> dict[str, Any]:
    """The message shown while a turn runs: a "Researching" card with the live agent steps.

    It is a plain chat message rather than a collapsible "thought": the message is replaced on
    every update, and a collapsible block would snap shut each time, hiding the steps.
    """
    count = len(steps)
    count_text = f" · {count} step{'s' if count != 1 else ''}" if steps else ""
    timer = ""
    if elapsed is not None:
        minutes, seconds = divmod(int(elapsed), 60)
        timer = f'<span class="ira-elapsed">Elapsed {minutes:02d}:{seconds:02d}</span>'
    content = (
        '<div class="ira-progress">'
        '<div class="ira-progress-head"><span class="ira-dot"></span><b>Researching</b>'
        f'<span class="ira-count">{count_text}</span>{timer}</div>'
        f'<p class="ira-note">{html.escape(THINKING_MESSAGE)}</p>'
        + (_step_log(steps) if steps else "")
        + "</div>"
    )
    return {"role": "assistant", "content": content}


def _is_brief(answer: str) -> bool:
    """True for a full research brief (it has Markdown headings or a table, or is long), as
    opposed to a short conversational reply such as a greeting or a refusal.
    """
    text = answer or ""
    has_structure = re.search(r"^\s*(#{1,4}\s|\|)", text, flags=re.MULTILINE) is not None
    return has_structure or len(text) > BRIEF_MIN_CHARS


def _answer_message(
    answer: str, session: Any, steps: int = 0, elapsed: float | None = None
) -> dict[str, Any]:
    """Wrap an agent answer in a card: a header (title, and how many steps the agents took and
    how long it ran), the Markdown answer and, for full briefs, a disclaimer footer.

    The ticker is not repeated on the card: one thread can cover several tickers, and the
    header above the chat already shows the active one. The blank lines around the answer
    matter: they let the Markdown inside the HTML card render.
    """
    del session  # kept in the signature so callers pass the same arguments for every reply
    brief = _is_brief(answer)
    title = "Research brief" if brief else "Answer"
    meta = []
    if steps:
        meta.append(f"{steps} step{'s' if steps != 1 else ''}")
    if elapsed is not None:
        minutes, seconds = divmod(int(elapsed), 60)
        meta.append(f"{minutes:02d}:{seconds:02d}")
    meta_html = f'<span class="ira-report-meta">{" · ".join(meta)}</span>' if meta else ""
    footer = f'<div class="ira-report-foot">{html.escape(DISCLAIMER)}</div>' if brief else ""
    card_class = "ira-report" if brief else "ira-report ira-report-short"
    content = (
        f'<div class="{card_class}"><div class="ira-report-head">'
        f'<span class="ira-report-icon">{_REPORT_SVG}</span>'
        f'<span class="ira-report-title">{title}</span>{meta_html}</div>'
        f"\n\n{answer}\n\n"
        f"{footer}</div>"
    )
    return {"role": "assistant", "content": content}


def _error_message(error: Any) -> dict[str, Any]:
    """The chat message shown when a turn fails (the error text is escaped)."""
    text = f"Something went wrong while researching that: {html.escape(str(error))}"
    return {"role": "assistant", "content": f'<div class="ira-error">{text}</div>'}


def _accepts_progress(ask: Any) -> bool:
    """True if ``ask`` takes an ``on_progress`` callback (the real ``ResearchSession`` does)."""
    try:
        return "on_progress" in inspect.signature(ask).parameters
    except (TypeError, ValueError):
        return False


def _drain(events: queue.Queue[str]) -> list[str]:
    """Return every progress message queued so far, without blocking."""
    items: list[str] = []
    # Only this function reads the queue, so empty() followed by get_nowait() is safe.
    while not events.empty():
        items.append(events.get_nowait())
    return items


def _list_threads() -> list[dict[str, Any]]:
    try:
        return [dict(t) for t in SessionStore().list_threads()]
    except Exception:  # noqa: BLE001 - the rail is optional; never break the chat over it
        logger.exception("Gradio UI: could not list threads")
        return []


def _threads_for(session: Any) -> list[dict[str, Any]]:
    lister = getattr(session, "list_sessions", None)
    return lister() if callable(lister) else _list_threads()


def _short_date(value: Any) -> str:
    """``2026-10-11T09:00`` -> ``Oct 11`` (unparseable or empty values are returned as is)."""
    try:
        return datetime.fromisoformat(str(value)).strftime("%b %d").replace(" 0", " ")
    except ValueError:
        return str(value or "")


def _thread_choices(threads: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Rail entries as ``(label, thread_id)``; empty threads are hidden.

    The label has two lines: the company, then the number of turns and the date.
    """
    choices = []
    for t in threads:
        turns = int(t.get("turns") or 0)
        if not turns:
            continue
        ticker = t.get("active_ticker") or "General"
        meta = f"{turns} turn{'s' if turns != 1 else ''}"
        if date := _short_date(t.get("updated_at")):
            meta += f" · {date}"
        choices.append((f"{ticker}\n{meta}", t["thread_id"]))
    return choices[:MAX_THREADS_IN_RAIL]


def _header_html(session: Any) -> str:
    ticker = getattr(session, "active_ticker", None) if session is not None else None
    title = f"{ticker} research" if ticker else "New research"
    tags = ""
    if ticker:
        tags += f'<span class="tag tag-ticker">{html.escape(ticker)}</span>'
    if session is not None:
        tags += f'<span class="tag">thread · {html.escape(session.thread_id)}</span>'
    return f'<div class="thread-head"><h2>{html.escape(title)}</h2>{tags}</div>'


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
        history.append(_answer_message(turn["answer"], session))
    logger.info(
        "Gradio UI: started session thread=%s resumed=%s", session.thread_id, session.resumed
    )
    return session, history, _active_ticker_label(session)


def respond(message: str, history: ChatHistory, session: ResearchSession | None) -> Any:
    """Handle one chat turn: echo the question, show a status line, then the answer."""
    history = list(history or [])
    message = (message or "").strip()

    if not message:
        reply = "Please ask a question about a stock, e.g. 'Analyse NVDA'."
        history.append({"role": "assistant", "content": reply})
        yield history, session, _active_ticker_label(session), ""
        return

    # Show the question and the "Researching" card first: creating the first session of a tab
    # builds the agent team, which takes a few seconds.
    history.append({"role": "user", "content": message})
    history.append(_pending_message([]))
    yield history, session, _active_ticker_label(session), ""
    if session is None:
        session = _new_session()

    # Run the workflow in a worker thread so the pending message can show the agent and
    # tool steps live; a full research run takes minutes.
    events: queue.Queue[str] = queue.Queue()
    outcome: dict[str, Any] = {}
    ask = session.ask

    def work() -> None:
        try:
            if _accepts_progress(ask):
                outcome["answer"] = ask(message, show=False, on_progress=events.put)
            else:
                outcome["answer"] = ask(message, show=False)
        except Exception as exc:  # noqa: BLE001 - shown in the chat instead of crashing the UI
            logger.exception("Gradio UI: workflow failed for thread %s", session.thread_id)
            outcome["error"] = exc

    worker = threading.Thread(target=work, name=f"research-{session.thread_id}", daemon=True)
    worker.start()
    started = time.monotonic()
    steps: list[str] = []
    shown_second = 0
    while worker.is_alive():
        worker.join(timeout=PROGRESS_POLL_SECONDS)
        fresh = _drain(events)
        steps.extend(fresh)
        elapsed = time.monotonic() - started
        # Update on new steps, and once a second so the timer keeps moving.
        if worker.is_alive() and (fresh or int(elapsed) > shown_second):
            shown_second = int(elapsed)
            history[-1] = _pending_message(steps, elapsed)
            yield history, session, _active_ticker_label(session), ""

    reasoning = None
    if "error" in outcome:
        reply = _error_message(outcome["error"])
    else:
        answer = outcome.get("answer") or "No response generated."
        reasoning = _reasoning_message(session)
        reply = _answer_message(answer, session, len(steps), time.monotonic() - started)

    history.pop()
    if reasoning:
        history.append(reasoning)
    history.append(reply)
    yield history, session, _active_ticker_label(session), ""


def rail_data(session: Any) -> dict[str, Any]:
    """What the conversation list shows: ``(label, thread_id)`` entries and the open thread."""
    return {
        "choices": _thread_choices(_threads_for(session)),
        "current": session.thread_id if session is not None else None,
    }


def refresh(session: Any) -> tuple[Any, ...]:
    """Recompute the rail, header, export file and suggestion chips for ``session``."""
    path = export_thread(session)
    return (
        _rail_html(rail_data(session)),
        _header_html(session),
        gr.DownloadButton(value=path, interactive=bool(path)),
        *(gr.Button(value=s) for s in _suggestions(session)),
    )


def _store_for(session: Any) -> Any:
    """The conversation store behind ``session`` (a default ``SessionStore`` if it has none)."""
    store = getattr(session, "store", None)
    return store if store is not None else SessionStore()


def delete_thread(thread_id: str, session: Any) -> tuple[Any, Any, str]:
    """Delete one stored conversation (its turns and cached brief).

    Deleting the conversation that is open starts a new, empty one; deleting any other
    conversation leaves the chat on screen as it is. Long-term memory (what the system learned
    across runs) is not touched.
    """
    try:
        _store_for(session).delete_thread(thread_id)
        logger.info("Gradio UI: deleted thread %s", thread_id)
    except Exception:  # noqa: BLE001 - a failed delete must not break the chat
        logger.exception("Gradio UI: could not delete thread %s", thread_id)
    if session is None or session.thread_id == thread_id:
        return start_session("")
    return session, gr.skip(), _active_ticker_label(session)


def delete_all_threads(session: Any) -> tuple[Any, ...]:
    """Delete every stored conversation, start a new empty one and close the confirm step."""
    store = _store_for(session)
    for thread in _threads_for(session):
        try:
            store.delete_thread(thread["thread_id"])
        except Exception:  # noqa: BLE001, PERF203 - keep deleting the others
            logger.exception("Gradio UI: could not delete thread %s", thread.get("thread_id"))
    logger.info("Gradio UI: cleared all conversations")
    new_session, history, label = start_session("")
    return new_session, history, label, *_confirm_clear(show=False)


def _confirm_clear(show: bool) -> tuple[Any, Any]:
    """Swap "Clear all conversations" for its confirm box (``show=True``) or back again."""
    return gr.Button(visible=not show), gr.Column(visible=show)


def _clear() -> tuple[ResearchSession, ChatHistory, str, str]:
    """Handle "New research": a fresh, empty session and an empty composer."""
    session, history, label = start_session("")
    return session, history, label, ""


def _rail_html(rail: Mapping[str, Any] | None) -> str:
    """The conversation list as HTML: one row per conversation, with an open button and a
    delete button. Both carry the thread id; RAIL_JS sends the click to ``rail_command``.
    """
    rail = rail or {}
    choices = rail.get("choices") or []
    if not choices:
        return '<div class="rail-empty">No conversations yet</div>'
    rows = []
    for label, tid in choices:
        title, _, meta = str(label).partition("\n")
        current = " current" if tid == rail.get("current") else ""
        safe_tid = html.escape(str(tid), quote=True)
        rows.append(
            f'<div class="thread-row{current}">'
            f'<button type="button" class="thread-open" data-rail-action="open" '
            f'data-tid="{safe_tid}"><span class="thread-title">{html.escape(title)}</span>'
            f'<span class="thread-meta">{html.escape(meta)}</span></button>'
            f'<button type="button" class="thread-del" data-rail-action="delete" '
            f'data-tid="{safe_tid}" title="Delete conversation" '
            f'aria-label="Delete conversation {html.escape(title, quote=True)}">\u2715</button>'
            "</div>"
        )
    return f'<div class="thread-list">{"".join(rows)}</div>'


# Click ids already handled, so a repeated delivery of the same click never acts twice.
_HANDLED_RAIL_CLICKS: deque[str] = deque(maxlen=200)


def rail_command(command: str, session: Any) -> tuple[Any, Any, str]:
    """Run one click from the conversation list: ``"open|<thread id>|<click id>"`` resumes that
    conversation and ``"delete|<thread id>|<click id>"`` deletes it.

    Each click id is handled once, and anything malformed changes nothing.
    """
    action, tid, click_id = [*(command or "").split("|"), "", "", ""][:3]
    unchanged = (session, gr.skip(), _active_ticker_label(session))
    if not tid or not click_id or click_id in _HANDLED_RAIL_CLICKS:
        return unchanged
    _HANDLED_RAIL_CLICKS.append(click_id)
    if action == "open":
        return start_session(tid)
    if action == "delete":
        return delete_thread(tid, session)
    return unchanged


# --------------------------------------------------------------------------- app


def build_app() -> gr.Blocks:
    """Assemble the Gradio Blocks app (layout from the Google Stitch design)."""
    with gr.Blocks(title=APP_TITLE) as demo:
        session_state = gr.State(None)

        with gr.Row(equal_height=False, elem_id="shell"):
            with gr.Column(scale=0, min_width=300, elem_id="rail"):
                gr.HTML(BRAND_HTML)
                new_btn = gr.Button("+  New research", elem_id="new-research")
                gr.HTML('<div class="rail-label">Recent</div>')

                thread_list = gr.HTML(_rail_html(None), elem_id="thread-list")
                # Hidden bridge for clicks on the list (see RAIL_JS and rail_command).
                rail_cmd = gr.Textbox(elem_id="rail-cmd", show_label=False, container=False)
                rail_go = gr.Button("go", elem_id="rail-go")

                with gr.Column(elem_id="rail-foot"):
                    clear_all_btn = gr.Button(
                        "Clear all conversations",
                        size="sm",
                        elem_id="clear-all",
                        elem_classes="rail-action",
                    )
                    with gr.Column(visible=False, elem_id="confirm-clear") as confirm_box:
                        gr.HTML(
                            '<div class="confirm-text">Delete every conversation? '
                            "This cannot be undone.</div>"
                        )
                        with gr.Row():
                            confirm_yes = gr.Button(
                                "Delete all", size="sm", min_width=0, elem_id="confirm-clear-yes"
                            )
                            confirm_no = gr.Button(
                                "Cancel", size="sm", min_width=0, elem_classes="rail-action"
                            )
                    theme_btn = gr.Button("Light / dark", size="sm", elem_classes="rail-action")

            with gr.Column(scale=1, elem_id="thread"):
                with gr.Row(elem_id="thread-head"):
                    header = gr.HTML(_header_html(None))
                    export_btn = gr.DownloadButton(
                        "Export report",
                        size="sm",
                        interactive=False,
                        scale=0,
                        min_width=130,
                        elem_id="export",
                        elem_classes="quiet",
                    )

                chatbot = gr.Chatbot(
                    show_label=False,
                    height="62vh",
                    placeholder=CHAT_PLACEHOLDER,
                    buttons=["copy"],
                    elem_id="chatbot",
                )

                with gr.Row(elem_id="suggestions"):
                    sugg_btns = [
                        gr.Button(p, size="sm", scale=0, elem_classes="suggestion")
                        for p in STARTER_PROMPTS
                    ]

                with gr.Group(elem_id="composer"):
                    msg_box = gr.Textbox(
                        show_label=False,
                        placeholder="Ask about a ticker, or a follow-up (e.g. Analyse MSFT)…",
                        lines=1,
                        max_lines=6,
                        submit_btn=True,
                        container=False,
                    )
                with gr.Row(elem_id="composer-meta"):
                    ticker_label = gr.Markdown(_active_ticker_label(None))
                    gr.Markdown(DISCLAIMER, elem_id="disclaimer")

        refresh_outputs = [thread_list, header, export_btn, *sugg_btns]
        turn_outputs = [chatbot, session_state, ticker_label, msg_box]
        session_outputs = [session_state, chatbot, ticker_label]

        demo.load(refresh, inputs=[session_state], outputs=refresh_outputs)
        rail_go.click(rail_command, inputs=[rail_cmd, session_state], outputs=session_outputs).then(
            refresh, inputs=[session_state], outputs=refresh_outputs
        )
        theme_btn.click(None, js=TOGGLE_THEME_JS)

        # The chat shows its own live "Researching" card, so Gradio's progress overlay is off.
        msg_box.submit(
            respond,
            inputs=[msg_box, chatbot, session_state],
            outputs=turn_outputs,
            show_progress="hidden",
        ).then(refresh, inputs=[session_state], outputs=refresh_outputs)

        for btn in sugg_btns:
            btn.click(
                respond,
                inputs=[btn, chatbot, session_state],
                outputs=turn_outputs,
                show_progress="hidden",
            ).then(refresh, inputs=[session_state], outputs=refresh_outputs)

        # "Clear all conversations" asks first: the button turns into a small confirm box.
        confirm_outputs = [clear_all_btn, confirm_box]
        clear_all_btn.click(partial(_confirm_clear, True), outputs=confirm_outputs)
        confirm_no.click(partial(_confirm_clear, False), outputs=confirm_outputs)
        confirm_yes.click(
            delete_all_threads,
            inputs=[session_state],
            outputs=[*session_outputs, *confirm_outputs],
        ).then(refresh, inputs=[session_state], outputs=refresh_outputs)

        new_btn.click(_clear, outputs=[session_state, chatbot, ticker_label, msg_box]).then(
            refresh, inputs=[session_state], outputs=refresh_outputs
        )

    return demo


def main() -> None:
    """Launch the Gradio app (blocking)."""
    build_app().launch(theme=make_theme(), css=CSS, js=PAGE_JS)


if __name__ == "__main__":
    main()
