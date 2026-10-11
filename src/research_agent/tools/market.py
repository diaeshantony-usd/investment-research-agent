"""
tools/market.py
===============
Tool for retrieving market technicals, price action, moving averages, and
indicators - computed from real Yahoo Finance data (``tools/yahoo_finance.py``)
rather than a fixed ticker list.

``get_market_technicals`` combines a year of daily closes
(``get_historical_data``) with the live quote/ratios (``get_ticker_details``)
to compute 50/200-day SMAs, 14-day RSI (Wilder's smoothing) and a simple
trend label, alongside beta and the 52-week range. Both underlying calls are
already cached and have their own Alpha Vantage fallback (see
``yahoo_finance.py``), so this tool inherits that resilience for free.
"""

from __future__ import annotations

import json
from typing import Any

from research_agent.agent_framework import tool
from research_agent.config import logger
from research_agent.tools.yahoo_finance import get_historical_data, get_ticker_details

_SMA_WINDOWS = (50, 200)
_RSI_PERIOD = 14
# A year of daily candles comfortably covers a 200-day SMA even accounting
# for market holidays (~252 trading days/year).
_HISTORY_PERIOD = "1y"


def _sma(closes: list[float], window: int) -> float | None:
    """Simple moving average of the last ``window`` closes, or ``None`` if too few."""
    if len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def _rsi(closes: list[float], period: int = _RSI_PERIOD) -> float | None:
    """14-day RSI using Wilder's smoothing, or ``None`` if there isn't enough history."""
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    relative_strength = avg_gain / avg_loss
    return 100 - (100 / (1 + relative_strength))


def _trend_label(current_price: float | None, sma50: float | None, sma200: float | None) -> str:
    """A plain-language trend label from price vs. its moving averages."""
    if current_price is None or sma50 is None or sma200 is None:
        return "Insufficient data"
    if current_price > sma50 > sma200:
        return "Bullish Uptrend"
    if current_price < sma50 < sma200:
        return "Bearish Downtrend"
    if sma50 > sma200:
        return "Moderate Uptrend"
    if sma50 < sma200:
        return "Moderate Downtrend"
    return "Consolidation"


@tool(
    name="get_market_technicals",
    description=(
        "Retrieves technical price action for a stock ticker: 50/200-day simple moving "
        "averages, 14-day RSI, beta, 52-week range and a trend label, computed from live "
        "Yahoo Finance price history."
    ),
)
def get_market_technicals(ticker: str) -> str:
    """Returns price action and technical indicators for a given stock ticker.

    Args:
        ticker: Stock symbol (e.g. 'NVDA', 'AAPL').

    Returns:
        JSON with current price, 52-week range, 50/200-day SMAs, 14-day RSI,
        beta and a trend label, or an ``error`` field when the ticker's
        price history or quote can't be retrieved. SMA/RSI fields are
        ``null`` when there isn't enough price history to compute them
        (e.g. a recent IPO).
    """
    clean_ticker = (ticker or "").strip().upper()
    logger.info("Executing get_market_technicals for ticker: %s", clean_ticker)

    try:
        details = get_ticker_details.func(clean_ticker)
        history = get_historical_data.func(clean_ticker, period=_HISTORY_PERIOD, interval="1d")
    except ValueError as exc:
        logger.warning("get_market_technicals failed for %s: %s", clean_ticker, exc)
        return json.dumps({"ticker": clean_ticker, "error": str(exc)})

    closes = [candle["close"] for candle in history]
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)
    rsi14 = _rsi(closes)
    current_price = details.get("current_price")

    payload: dict[str, Any] = {
        "ticker": clean_ticker,
        "current_price": current_price,
        "52_week_high": details.get("fifty_two_week_high"),
        "52_week_low": details.get("fifty_two_week_low"),
        "50_day_sma": round(sma50, 2) if sma50 is not None else None,
        "200_day_sma": round(sma200, 2) if sma200 is not None else None,
        "rsi_14": round(rsi14, 2) if rsi14 is not None else None,
        "beta": details.get("beta"),
        "trend": _trend_label(current_price, sma50, sma200),
        "source": details.get("source", "yfinance"),
    }
    return json.dumps(payload, indent=2)
