"""
test_tools_alpha_vantage.py
============================
Offline tests for the Alpha Vantage REST client. ``requests.get`` is replaced with a
fake response, so these tests check request/response handling without network access.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_agent.tools import alpha_vantage


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise alpha_vantage.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture(autouse=True)
def _with_api_key(monkeypatch):
    monkeypatch.setattr(alpha_vantage, "ALPHA_VANTAGE_KEY", "test-key")


class TestIsConfigured:
    def test_true_when_key_set(self, monkeypatch):
        monkeypatch.setattr(alpha_vantage, "ALPHA_VANTAGE_KEY", "abc")
        assert alpha_vantage.is_configured() is True

    def test_false_when_key_missing(self, monkeypatch):
        monkeypatch.setattr(alpha_vantage, "ALPHA_VANTAGE_KEY", "")
        assert alpha_vantage.is_configured() is False


class TestSafeFloat:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("1.5", 1.5), (None, None), ("None", None), ("-", None), ("", None), ("oops", None)],
    )
    def test_coerces_or_falls_back_to_none(self, raw, expected):
        assert alpha_vantage._safe_float(raw) == expected


class TestGet:
    def test_raises_without_key(self, monkeypatch):
        monkeypatch.setattr(alpha_vantage, "ALPHA_VANTAGE_KEY", "")
        with pytest.raises(ValueError, match="not set"):
            alpha_vantage._get({"function": "OVERVIEW"})

    def test_raises_on_request_exception(self, monkeypatch):
        def boom(*args, **kwargs):
            raise alpha_vantage.requests.RequestException("network down")

        monkeypatch.setattr(alpha_vantage.requests, "get", boom)
        with pytest.raises(ValueError, match="request failed"):
            alpha_vantage._get({"function": "OVERVIEW"})

    def test_raises_on_rate_limit_note(self, monkeypatch):
        monkeypatch.setattr(
            alpha_vantage.requests, "get", lambda *a, **k: FakeResponse({"Note": "slow down"})
        )
        with pytest.raises(ValueError, match="slow down"):
            alpha_vantage._get({"function": "OVERVIEW"})

    def test_raises_on_information_notice(self, monkeypatch):
        monkeypatch.setattr(
            alpha_vantage.requests,
            "get",
            lambda *a, **k: FakeResponse({"Information": "quota exhausted"}),
        )
        with pytest.raises(ValueError, match="quota exhausted"):
            alpha_vantage._get({"function": "OVERVIEW"})

    def test_raises_on_error_message(self, monkeypatch):
        monkeypatch.setattr(
            alpha_vantage.requests,
            "get",
            lambda *a, **k: FakeResponse({"Error Message": "bad symbol"}),
        )
        with pytest.raises(ValueError, match="bad symbol"):
            alpha_vantage._get({"function": "OVERVIEW"})

    def test_returns_payload_on_success(self, monkeypatch):
        monkeypatch.setattr(
            alpha_vantage.requests, "get", lambda *a, **k: FakeResponse({"Symbol": "NVDA"})
        )
        assert alpha_vantage._get({"function": "OVERVIEW"}) == {"Symbol": "NVDA"}


class TestGetTickerDetails:
    def test_raises_without_overview(self, monkeypatch):
        monkeypatch.setattr(alpha_vantage, "_get", lambda params: {})
        with pytest.raises(ValueError, match="no overview"):
            alpha_vantage.get_ticker_details("ZZZZ")

    def test_maps_overview_and_quote_fields(self, monkeypatch):
        def fake_get(params):
            if params["function"] == "OVERVIEW":
                return {
                    "Symbol": "NVDA",
                    "Name": "NVIDIA Corp",
                    "Exchange": "NASDAQ",
                    "Currency": "USD",
                    "Sector": "Technology",
                    "Industry": "Semiconductors",
                    "MarketCapitalization": "4200000000000",
                    "52WeekHigh": "190.0",
                    "52WeekLow": "80.0",
                    "Beta": "1.6",
                    "TrailingPE": "52.1",
                    "ForwardPE": "34.2",
                    "DividendYield": "0.03",
                }
            return {"Global Quote": {"05. price": "180.5", "08. previous close": "178.0"}}

        monkeypatch.setattr(alpha_vantage, "_get", fake_get)
        details = alpha_vantage.get_ticker_details("NVDA")
        assert details["ticker"] == "NVDA"
        assert details["name"] == "NVIDIA Corp"
        assert details["current_price"] == 180.5
        assert details["previous_close"] == 178.0
        assert details["forward_pe"] == 34.2
        assert details["market_cap"] == 4.2e12

    def test_quote_failure_still_returns_overview_fields(self, monkeypatch):
        def fake_get(params):
            if params["function"] == "OVERVIEW":
                return {"Symbol": "NVDA", "Name": "NVIDIA Corp"}
            raise ValueError("quote endpoint down")

        monkeypatch.setattr(alpha_vantage, "_get", fake_get)
        details = alpha_vantage.get_ticker_details("NVDA")
        assert details["name"] == "NVIDIA Corp"
        assert details["current_price"] is None


class TestGetEarningsHistory:
    def test_raises_without_quarterly_earnings(self, monkeypatch):
        monkeypatch.setattr(alpha_vantage, "_get", lambda params: {})
        with pytest.raises(ValueError, match="no earnings"):
            alpha_vantage.get_earnings_history("ZZZZ")

    def test_maps_quarterly_earnings_newest_first(self, monkeypatch):
        payload = {
            "quarterlyEarnings": [
                {
                    "fiscalDateEnding": "2026-07-31",
                    "reportedEPS": "1.05",
                    "estimatedEPS": "1.01",
                    "surprisePercentage": "3.96",
                },
                {
                    "fiscalDateEnding": "2026-04-30",
                    "reportedEPS": "0.96",
                    "estimatedEPS": "0.93",
                    "surprisePercentage": "3.23",
                },
            ]
        }
        monkeypatch.setattr(alpha_vantage, "_get", lambda params: payload)
        history = alpha_vantage.get_earnings_history("NVDA")
        assert len(history) == 2
        assert history[0] == {
            "period": "2026-07-31",
            "eps_actual": 1.05,
            "eps_estimate": 1.01,
            "surprise_pct": 3.96,
        }


class TestGetFinancialStatements:
    def test_raises_when_all_statements_empty(self, monkeypatch):
        monkeypatch.setattr(alpha_vantage, "_get", lambda params: {})
        with pytest.raises(ValueError, match="no financial statements"):
            alpha_vantage.get_financial_statements("ZZZZ")

    def test_maps_income_balance_and_cash_flow(self, monkeypatch):
        def fake_get(params):
            if params["function"] == "INCOME_STATEMENT":
                return {
                    "annualReports": [
                        {
                            "fiscalDateEnding": "2026-01-31",
                            "totalRevenue": "100",
                            "reportedCurrency": "USD",
                        }
                    ],
                    "quarterlyReports": [{"fiscalDateEnding": "2026-07-31", "totalRevenue": "30"}],
                }
            if params["function"] == "BALANCE_SHEET":
                return {"annualReports": [{"fiscalDateEnding": "2026-01-31", "totalAssets": "500"}]}
            return {
                "annualReports": [
                    {
                        "fiscalDateEnding": "2026-01-31",
                        "operatingCashflow": "80",
                        "capitalExpenditures": "5",
                    }
                ]
            }

        monkeypatch.setattr(alpha_vantage, "_get", fake_get)
        statements = alpha_vantage.get_financial_statements("NVDA")
        assert statements["ticker"] == "NVDA"
        assert statements["annual_income_statement"][0]["totalRevenue"] == 100.0
        assert statements["annual_income_statement"][0]["reportedCurrency"] == "USD"
        assert statements["quarterly_income_statement"][0]["totalRevenue"] == 30.0
        assert statements["annual_balance_sheet"][0]["totalAssets"] == 500.0
        assert statements["annual_cash_flow"][0]["operatingCashflow"] == 80.0
