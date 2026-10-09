"""
workflows
=========
Multi-agent execution workflows and session management.
"""

from __future__ import annotations

from research_agent.workflows.graph import build_research_workflow
from research_agent.workflows.session import (
    ResearchSession,
    _format_trajectory_table,
)

__all__ = ["ResearchSession", "_format_trajectory_table", "build_research_workflow"]
