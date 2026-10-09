#!/usr/bin/env python
"""
cli.py
======
Standalone Command-Line Interface for the Multi-Agent Investment Research Assistant.

Usage examples:
    python cli.py "Analyse NVIDIA (NVDA)"
    python cli.py "What is operating margin?"
    python cli.py --interactive
"""

from __future__ import annotations

import sys

from research_agent.workflows.session import main

if __name__ == "__main__":
    import io

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
