"""
test_tools_earnings.py
======================
Offline tests for ``get_earnings_data``. Provider calls (yfinance / Alpha Vantage) are
replaced with fixed payloads, so these tests check the derived figures and the error
handling without any network access.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from research_agent.tools import cache, earnings
from research_agent.tools.earnings import get_earnings_data
from research_agent.tools.yahoo_finance import get_financial_statements, get_ticker_details


def _quarter(period: str, revenue: float, gross: float, op: float, net: float) -> dict:
    return {
        "period": period,
        "Total Revenue": revenue,
        "Gross Profit": gross,
        "Operating Income": op,
        "Net Income": net,
        "Diluted EPS": round(net / 1e10, 2),
    }


# Five quarters (provider order deliberately scrambled) so YoY growth is computable.
STATEMENTS: dict[str, Any] = {
    "ticker": "NVDA",
    "source": "yfinance",
    "quarterly_income_statement": [
        _quarter("2025-07-31", 30e9, 22e9, 16e9, 14e9),
        _quarter("2026-07-31", 46e9, 33e9, 28e9, 25e9),
        _quarter("2026-04-30", 44e9, 30e9, 26e9, 22e9),
        _quarter("2026-01-31", 39e9, 29e9, 24e9, 21e9),
        _quarter("2025-10-31", 35e9, 26e9, 20e9, 19e9),
    ],
    "annual_cash_flow": [
        {"period": "2025-01-31", "Free Cash Flow": 60e9},
        {"period": "2026-01-31", "Free Cash Flow": 72e9},
    ],
}
DETAILS = {"trailing_pe": 52.1, "forward_pe": 34.2, "market_cap": 4.2e12, "source": "yfinance"}
EPS_HISTORY = [
    {"period": "2026-07-31", "eps_actual": 1.05, "eps_estimate": 1.01, "surprise_pct": 3.96},
    {"period": "2026-04-30", "eps_actual": 0.96, "eps_estimate": 0.93, "surprise_pct": 3.23},
]


@pytest.fixture
def live_data(monkeypatch):
    """Stub every provider call with the fixed payloads above."""
    monkeypatch.setattr(get_financial_statements, "func", lambda ticker: STATEMENTS)
    monkeypatch.setattr(get_ticker_details, "func", lambda ticker: DETAILS)
    monkeypatch.setattr(earnings, "_get_eps_history", lambda ticker: EPS_HISTORY)


def _call(ticker: str = "nvda") -> dict[str, Any]:
    return json.loads(get_earnings_data.execute(ticker=ticker))


class TestEarningsSnapshot:
    def test_uses_latest_quarter_regardless_of_provider_order(self, live_data):
        data = _call()
        assert data["ticker"] == "NVDA"
        assert data["quarter_end"] == "2026-07-31"
        assert data["revenue"] == "$46.00B"

    def test_margins_and_growth_are_derived_from_statements(self, live_data):
        data = _call()
        assert data["gross_margin_pct"] == pytest.approx(71.74)
        assert data["operating_margin_pct"] == pytest.approx(60.87)
        assert data["net_margin_pct"] == pytest.approx(54.35)
        assert data["revenue_growth_yoy_pct"] == pytest.approx(53.33)  # vs 2025-07-31
        assert data["revenue_growth_qoq_pct"] == pytest.approx(4.55)  # vs 2026-04-30

    def test_cash_flow_valuation_and_eps_surprise(self, live_data):
        data = _call()
        assert data["free_cash_flow_annual"] == "$72.00B"
        assert data["fiscal_year_end"] == "2026-01-31"
        assert data["forward_pe"] == 34.2
        assert data["market_cap"] == "$4.20T"
        assert data["eps_actual"] == 1.05
        assert data["eps_quarter"] == "2026-07-31"
        assert data["eps_surprise_pct"] == 3.96
        assert len(data["eps_history"]) == 2
        assert data["unavailable"] == []

    def test_skips_newest_quarter_without_figures(self, monkeypatch):
        # Seen live for AMZN: the newest quarter is listed before its figures exist.
        empty_newest = {"period": "2026-10-31", "Total Revenue": None, "Net Income": None}
        statements = {
            **STATEMENTS,
            "quarterly_income_statement": [
                empty_newest,
                *STATEMENTS["quarterly_income_statement"],
            ],
        }
        monkeypatch.setattr(get_financial_statements, "func", lambda ticker: statements)
        monkeypatch.setattr(get_ticker_details, "func", lambda ticker: DETAILS)
        monkeypatch.setattr(earnings, "_get_eps_history", lambda ticker: EPS_HISTORY)
        data = _call()
        assert data["quarter_end"] == "2026-07-31"
        assert data["revenue"] == "$46.00B"
        assert data["revenue_growth_yoy_pct"] == pytest.approx(53.33)

    def test_yoy_growth_needs_a_quarter_about_one_year_earlier(self, monkeypatch):
        statements = {
            **STATEMENTS,
            "quarterly_income_statement": STATEMENTS["quarterly_income_statement"][1:4],
        }
        monkeypatch.setattr(get_financial_statements, "func", lambda ticker: statements)
        monkeypatch.setattr(get_ticker_details, "func", lambda ticker: DETAILS)
        monkeypatch.setattr(earnings, "_get_eps_history", lambda ticker: EPS_HISTORY)
        data = _call()
        assert data["revenue_growth_yoy_pct"] is None
        assert "revenue_growth_yoy_pct" in data["unavailable"]

    def test_no_mock_numbers_for_unknown_tickers(self, monkeypatch):
        def no_statements(ticker):
            raise ValueError("No financial statements found")

        monkeypatch.setattr(get_financial_statements, "func", no_statements)
        data = _call("ZZZZ")
        assert "error" in data
        assert "revenue" not in data

    def test_optional_sources_degrade_to_unavailable(self, monkeypatch):
        def fail(ticker):
            raise ValueError("provider down")

        monkeypatch.setattr(get_financial_statements, "func", lambda ticker: STATEMENTS)
        monkeypatch.setattr(get_ticker_details, "func", fail)
        monkeypatch.setattr(earnings, "_get_eps_history", fail)
        data = _call()
        assert data["revenue"] == "$46.00B"
        assert {"forward_pe", "eps_surprise_pct", "eps_history"} <= set(data["unavailable"])

    def test_free_cash_flow_falls_back_to_operating_cash_flow_minus_capex(self, monkeypatch):
        statements = {
            **STATEMENTS,
            # Alpha Vantage naming, with capex reported as a positive number.
            "annual_cash_flow": [
                {"period": "2026-01-31", "operatingCashflow": 80e9, "capitalExpenditures": 5e9}
            ],
        }
        monkeypatch.setattr(get_financial_statements, "func", lambda ticker: statements)
        monkeypatch.setattr(get_ticker_details, "func", lambda ticker: DETAILS)
        monkeypatch.setattr(earnings, "_get_eps_history", lambda ticker: [])
        assert _call()["free_cash_flow_annual"] == "$75.00B"


class TestEpsHistory:
    def test_yfinance_surprise_fraction_is_converted_to_percent(self, monkeypatch):
        frame = pd.DataFrame(
            {
                "epsActual": [0.9, 1.05],
                "epsEstimate": [0.88, 1.01],
                "surprisePercent": [0.0227, 0.0396],
            },
            index=pd.to_datetime(["2026-04-30", "2026-07-31"]),
        )

        class FakeTicker:
            earnings_history = frame

        monkeypatch.setattr(earnings, "get_ticker", lambda ticker: FakeTicker())
        history = earnings._eps_history_yfinance("NVDA")
        assert history[0]["period"] == "2026-07-31"  # newest first
        assert history[0]["surprise_pct"] == pytest.approx(3.96)

    def test_falls_back_to_alpha_vantage(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

        def yf_fails(ticker):
            raise ValueError("yfinance rate limited")

        monkeypatch.setattr(earnings, "_eps_history_yfinance", yf_fails)
        monkeypatch.setattr(earnings.alpha_vantage, "is_configured", lambda: True)
        monkeypatch.setattr(
            earnings.alpha_vantage, "get_earnings_history", lambda ticker: EPS_HISTORY
        )
        assert earnings._get_eps_history("NVDA") == EPS_HISTORY
