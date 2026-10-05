"""
tools/macro.py
==============
Tool for retrieving macroeconomic indicators, treasury yields, and inflation metrics.
"""

from __future__ import annotations

import json
from typing import Any

from research_agent.agent_framework import tool
from research_agent.config import logger

_MACRO_DATA: dict[str, Any] = {
    "fed_funds_rate": "4.75% - 5.00%",
    "us_10y_treasury": "3.85%",
    "us_2y_treasury": "3.60%",
    "cpi_yoy": "2.4%",
    "core_pce_yoy": "2.6%",
    "gdp_growth_annualized": "2.8%",
    "macro_environment": "Easing cycle initiated, soft-landing trajectory intact.",
}


@tool(
    name="get_macro_indicators",
    description=(
        "Retrieves US macroeconomic indicators including 10-year treasury yields, CPI, and Fed "
        "rates."
    ),
)
def get_macro_indicators(category: str = "all") -> str:
    """Returns macroeconomic indicators and monetary policy backdrop.

    Args:
        category: One indicator key (e.g. ``'cpi_yoy'``) or ``'all'``. Unknown values
            return every indicator, so a loosely worded request still gets data.

    Returns:
        JSON with the requested indicator, or all indicators.
    """
    logger.info("Executing get_macro_indicators [category=%s]", category)
    if category in _MACRO_DATA:
        return json.dumps({category: _MACRO_DATA[category]}, indent=2)
    return json.dumps(_MACRO_DATA, indent=2)
