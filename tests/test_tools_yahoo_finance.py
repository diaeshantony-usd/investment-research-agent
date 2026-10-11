"""
test_tools_yahoo_finance.py
============================
Offline tests for the Yahoo Finance tools. ``yfinance.Ticker``/``yfinance.download`` are
replaced with fakes, so these tests check shaping, caching and the Alpha Vantage fallback
without network access.
"""

from __future__ import annotations

import pandas as pd
import pytest

from research_agent.tools import cache
from research_agent.tools import yahoo_finance as yf_tools


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)


class FakeTicker:
    def __init__(
        self,
        info: dict | None = None,
        history: pd.DataFrame | None = None,
        financials: pd.DataFrame | None = None,
        quarterly_financials: pd.DataFrame | None = None,
        balance_sheet: pd.DataFrame | None = None,
        cashflow: pd.DataFrame | None = None,
        info_error: Exception | None = None,
    ) -> None:
        self._info = info or {}
        self._history = history if history is not None else pd.DataFrame()
        self.financials = financials if financials is not None else pd.DataFrame()
        self.quarterly_financials = (
            quarterly_financials if quarterly_financials is not None else pd.DataFrame()
        )
        self.balance_sheet = balance_sheet if balance_sheet is not None else pd.DataFrame()
        self.cashflow = cashflow if cashflow is not None else pd.DataFrame()
        self._info_error = info_error

    def get_info(self):
        if self._info_error:
            raise self._info_error
        return self._info

    def history(self, period, interval):
        return self._history


class TestGetTicker:
    def test_raises_on_blank_ticker(self):
        with pytest.raises(ValueError, match="must not be empty"):
            yf_tools.get_ticker("   ")

    def test_cleans_and_upper_cases(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            yf_tools.yf, "Ticker", lambda symbol: captured.setdefault("symbol", symbol)
        )
        yf_tools.get_ticker(" nvda ")
        assert captured["symbol"] == "NVDA"


_FULL_INFO = {
    "symbol": "NVDA",
    "longName": "NVIDIA Corporation",
    "exchange": "NASDAQ",
    "currency": "USD",
    "sector": "Technology",
    "industry": "Semiconductors",
    "website": "https://nvidia.com",
    "fullTimeEmployees": 30000,
    "marketCap": 4.2e12,
    "currentPrice": 180.5,
    "previousClose": 178.0,
    "fiftyTwoWeekHigh": 190.0,
    "fiftyTwoWeekLow": 80.0,
    "beta": 1.6,
    "trailingPE": 52.1,
    "forwardPE": 34.2,
    "dividendYield": 0.0,
}


class TestGetTickerDetails:
    def test_yfinance_success(self, monkeypatch):
        monkeypatch.setattr(yf_tools.yf, "Ticker", lambda symbol: FakeTicker(info=_FULL_INFO))
        details = yf_tools.get_ticker_details.func("NVDA")
        assert details["name"] == "NVIDIA Corporation"
        assert details["source"] == "yfinance"

    def test_yfinance_raises_falls_back_to_alpha_vantage(self, monkeypatch):
        monkeypatch.setattr(
            yf_tools.yf, "Ticker", lambda symbol: FakeTicker(info_error=RuntimeError("blocked"))
        )
        monkeypatch.setattr(yf_tools.alpha_vantage, "is_configured", lambda: True)
        monkeypatch.setattr(
            yf_tools.alpha_vantage,
            "get_ticker_details",
            lambda ticker: {"ticker": ticker, "name": "AV Name"},
        )
        details = yf_tools.get_ticker_details.func("NVDA")
        assert details["name"] == "AV Name"
        assert details["source"] == "alpha_vantage"

    def test_no_price_in_info_raises_then_falls_back(self, monkeypatch):
        monkeypatch.setattr(
            yf_tools.yf, "Ticker", lambda symbol: FakeTicker(info={"symbol": "NVDA"})
        )
        monkeypatch.setattr(yf_tools.alpha_vantage, "is_configured", lambda: False)
        with pytest.raises(ValueError, match="No ticker details found"):
            yf_tools.get_ticker_details.func("NVDA")

    def test_both_sources_fail_raises_combined_error(self, monkeypatch):
        monkeypatch.setattr(
            yf_tools.yf, "Ticker", lambda symbol: FakeTicker(info_error=RuntimeError("blocked"))
        )
        monkeypatch.setattr(yf_tools.alpha_vantage, "is_configured", lambda: True)

        def av_fail(ticker):
            raise ValueError("AV down")

        monkeypatch.setattr(yf_tools.alpha_vantage, "get_ticker_details", av_fail)
        with pytest.raises(ValueError, match="Alpha Vantage fallback error"):
            yf_tools.get_ticker_details.func("NVDA")


class TestGetHistoricalData:
    def test_returns_candles(self, monkeypatch):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-02", "2026-01-03"]),
                "Open": [1.0, 2.0],
                "High": [1.5, 2.5],
                "Low": [0.5, 1.5],
                "Close": [1.2, 2.2],
                "Volume": [100, 200],
            }
        ).set_index("Date")
        monkeypatch.setattr(yf_tools.yf, "Ticker", lambda symbol: FakeTicker(history=frame))
        candles = yf_tools.get_historical_data.func("NVDA", period="1mo", interval="1d")
        assert len(candles) == 2
        assert candles[0]["close"] == 1.2

    def test_empty_history_raises(self, monkeypatch):
        monkeypatch.setattr(
            yf_tools.yf, "Ticker", lambda symbol: FakeTicker(history=pd.DataFrame())
        )
        with pytest.raises(ValueError, match="No historical data found"):
            yf_tools.get_historical_data.func("NVDA")

    def test_skips_rows_with_missing_close(self, monkeypatch):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-02", "2026-01-03"]),
                "Open": [1.0, 2.0],
                "High": [1.5, 2.5],
                "Low": [0.5, 1.5],
                "Close": [1.2, float("nan")],
                "Volume": [100, 200],
            }
        ).set_index("Date")
        monkeypatch.setattr(yf_tools.yf, "Ticker", lambda symbol: FakeTicker(history=frame))
        candles = yf_tools.get_historical_data.func("NVDA")
        assert len(candles) == 1


class TestDownloadPriceHistory:
    def test_single_ticker(self, monkeypatch):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-02"]),
                "Open": [1.0],
                "High": [1.5],
                "Low": [0.5],
                "Close": [1.2],
                "Volume": [100],
            }
        ).set_index("Date")
        monkeypatch.setattr(yf_tools.yf, "download", lambda *a, **k: frame)
        result = yf_tools.download_price_history.func("nvda")
        assert "NVDA" in result
        assert result["NVDA"][0]["close"] == 1.2

    def test_multi_ticker_comma_separated(self, monkeypatch):
        columns = pd.MultiIndex.from_product(
            [["NVDA", "AAPL"], ["Open", "High", "Low", "Close", "Volume"]]
        )
        data = [[1.0, 1.5, 0.5, 1.2, 100, 2.0, 2.5, 1.5, 2.2, 200]]
        frame = pd.DataFrame(data, index=pd.to_datetime(["2026-01-02"]), columns=columns)
        frame.index.name = "Date"
        monkeypatch.setattr(yf_tools.yf, "download", lambda *a, **k: frame)
        result = yf_tools.download_price_history.func("nvda, aapl")
        assert set(result) == {"NVDA", "AAPL"}

    def test_blank_tickers_raises(self):
        with pytest.raises(ValueError, match="At least one ticker"):
            yf_tools.download_price_history.func("   ")

    def test_empty_download_raises(self, monkeypatch):
        monkeypatch.setattr(yf_tools.yf, "download", lambda *a, **k: pd.DataFrame())
        with pytest.raises(ValueError, match="returned no data"):
            yf_tools.download_price_history.func("NVDA")


def _statement_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {pd.Timestamp("2026-01-31"): [100.0, 40.0]},
        index=["Total Revenue", "Net Income"],
    )


class TestGetFinancialStatements:
    def test_yfinance_success(self, monkeypatch):
        monkeypatch.setattr(
            yf_tools.yf,
            "Ticker",
            lambda symbol: FakeTicker(
                financials=_statement_frame(), quarterly_financials=_statement_frame()
            ),
        )
        statements = yf_tools.get_financial_statements.func("NVDA")
        assert statements["source"] == "yfinance"
        assert statements["annual_income_statement"][0]["Total Revenue"] == 100.0

    def test_all_empty_falls_back_to_alpha_vantage(self, monkeypatch):
        monkeypatch.setattr(yf_tools.yf, "Ticker", lambda symbol: FakeTicker())
        monkeypatch.setattr(yf_tools.alpha_vantage, "is_configured", lambda: True)
        monkeypatch.setattr(
            yf_tools.alpha_vantage,
            "get_financial_statements",
            lambda ticker: {
                "ticker": ticker,
                "annual_income_statement": [{"period": "2026-01-31"}],
            },
        )
        statements = yf_tools.get_financial_statements.func("NVDA")
        assert statements["source"] == "alpha_vantage"

    def test_both_sources_fail_raises(self, monkeypatch):
        monkeypatch.setattr(yf_tools.yf, "Ticker", lambda symbol: FakeTicker())
        monkeypatch.setattr(yf_tools.alpha_vantage, "is_configured", lambda: False)
        with pytest.raises(ValueError, match="No financial statements found"):
            yf_tools.get_financial_statements.func("NVDA")


class TestFrameToCandles:
    def test_handles_datetime_index_column(self):
        frame = pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2026-01-02T09:30"]),
                "Open": [1.0],
                "High": [1.5],
                "Low": [0.5],
                "Close": [1.2],
                "Volume": [100],
            }
        )
        candles = yf_tools._frame_to_candles(frame)
        assert candles[0]["volume"] == 100

    def test_missing_volume_is_none(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-02"]),
                "Open": [1.0],
                "High": [1.5],
                "Low": [0.5],
                "Close": [1.2],
                "Volume": [float("nan")],
            }
        )
        candles = yf_tools._frame_to_candles(frame)
        assert candles[0]["volume"] is None


class TestStatementToRecords:
    def test_empty_frame_returns_empty_list(self):
        assert yf_tools._statement_to_records(pd.DataFrame()) == []

    def test_none_returns_empty_list(self):
        assert yf_tools._statement_to_records(None) == []

    def test_converts_periods_and_line_items(self):
        frame = _statement_frame()
        records = yf_tools._statement_to_records(frame)
        assert records[0]["period"] == "2026-01-31"
        assert records[0]["Total Revenue"] == 100.0
