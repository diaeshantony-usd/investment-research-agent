"""
test_tools_exchange.py
=======================
Offline tests for the active-tickers tool. ``requests.get`` is replaced with a fixed
pipe-delimited directory, so these tests check parsing/filtering without network access.
"""

from __future__ import annotations

import json

import pytest

from research_agent.tools import cache, exchange


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        pass


_NASDAQ_TEXT = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|"
    "NextShares\n"
    "AAPL|Apple Inc. Common Stock|Q|N|N|100|N|N\n"
    "QQQQ|Invesco QQQ Trust|Q|N|N|100|Y|N\n"
    "ZZZT|Test Issue Common Stock|Q|Y|N|100|N|N\n"
    "File Creation Time: 1010202026|||||||\n"
)
_OTHER_TEXT = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    "NVDA|NVIDIA Corp Common Stock|N|NVDA|N|100|N|NVDA\n"
    "SPY|SPDR S&P 500 ETF|P|SPY|Y|100|N|SPY\n"
    "TSTZ|Test Security|N|TSTZ|N|100|Y|TSTZ\n"
    "File Creation Time: 1010202026|||||||\n"
)


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)


@pytest.fixture
def fake_requests(monkeypatch):
    def fake_get(url, timeout):
        if url == exchange._NASDAQ_LISTED_URL:
            return FakeResponse(_NASDAQ_TEXT)
        return FakeResponse(_OTHER_TEXT)

    monkeypatch.setattr(exchange.requests, "get", fake_get)


class TestDownloadSymbolDirectory:
    def test_strips_trailer_and_parses_rows(self, fake_requests):
        rows = exchange._download_symbol_directory(exchange._NASDAQ_LISTED_URL)
        assert len(rows) == 3
        assert rows[0]["Symbol"] == "AAPL"


class TestLoadNasdaqListed:
    def test_excludes_test_issues(self, fake_requests):
        df = exchange._load_nasdaq_listed()
        assert set(df["ticker"]) == {"AAPL", "QQQQ"}
        assert bool(df.loc[df["ticker"] == "QQQQ", "is_etf"].iloc[0]) is True
        assert (df["exchange"] == "NASDAQ").all()


class TestLoadOtherListed:
    def test_maps_exchange_codes_and_excludes_test_issues(self, fake_requests):
        df = exchange._load_other_listed(None)
        assert set(df["ticker"]) == {"NVDA", "SPY"}
        assert df.loc[df["ticker"] == "NVDA", "exchange"].iloc[0] == "NYSE"
        assert df.loc[df["ticker"] == "SPY", "exchange"].iloc[0] == "NYSE Arca"

    def test_filters_to_one_exchange(self, fake_requests):
        df = exchange._load_other_listed("NYSE")
        assert set(df["ticker"]) == {"NVDA"}


class TestLoadListings:
    def test_all_combines_both_directories(self, fake_requests):
        df = exchange._load_listings("ALL")
        assert set(df["ticker"]) == {"AAPL", "QQQQ", "NVDA", "SPY"}

    def test_nasdaq_only(self, fake_requests):
        df = exchange._load_listings("NASDAQ")
        assert set(df["ticker"]) == {"AAPL", "QQQQ"}


class TestGetActiveTickers:
    def test_unknown_exchange_returns_error(self, fake_requests):
        data = json.loads(exchange.get_active_tickers.func(exchange="MOONSTOCK"))
        assert "error" in data
        assert data["tickers"] == []

    def test_lists_tickers_for_an_exchange(self, fake_requests):
        data = json.loads(exchange.get_active_tickers.func(exchange="ALL"))
        assert data["count"] == 4
        tickers = {row["ticker"] for row in data["tickers"]}
        assert tickers == {"AAPL", "QQQQ", "NVDA", "SPY"}

    def test_query_filters_by_ticker_or_name(self, fake_requests):
        data = json.loads(exchange.get_active_tickers.func(exchange="ALL", query="nvidia"))
        assert data["count"] == 1
        assert data["tickers"][0]["ticker"] == "NVDA"

    def test_limit_caps_results(self, fake_requests):
        data = json.loads(exchange.get_active_tickers.func(exchange="ALL", limit=1))
        assert len(data["tickers"]) == 1
        assert data["count"] == 4  # count reflects the full match, before the limit

    def test_non_numeric_limit_falls_back_to_default(self, fake_requests):
        data = json.loads(exchange.get_active_tickers.func(exchange="ALL", limit="not a number"))
        assert len(data["tickers"]) == 4  # fewer rows than the default limit

    def test_download_failure_returns_error(self, monkeypatch):
        def boom(url, timeout):
            raise exchange.requests.RequestException("network down")

        monkeypatch.setattr(exchange.requests, "get", boom)
        data = json.loads(exchange.get_active_tickers.func(exchange="NASDAQ"))
        assert "error" in data
        assert data["tickers"] == []
