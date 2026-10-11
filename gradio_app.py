#!/usr/bin/env python
"""
gradio_app.py
=============
Launches the Gradio chat UI for the Multi-Agent Investment Research Assistant.

Usage:
    python gradio_app.py
"""

from __future__ import annotations

from research_agent.interfaces.chat_ui import main

if __name__ == "__main__":
    main()
