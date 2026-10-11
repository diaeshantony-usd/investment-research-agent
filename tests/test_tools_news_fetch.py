"""
test_tools_news_fetch.py
=========================
Offline tests for the rest of ``tools/news.py``: company-name resolution, date
validation, the yfinance fallback, and the two public tools. ``test_tools_news.py``
covers ``_fetch_from_newsapi``/``_title_query`` already; this file covers everything
built on top of them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from research_agent.tools import cache, news


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)


class TestSimplifyCompanyName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("NVIDIA Corporation", "NVIDIA"),
            ("Apple Inc.", "Apple"),
            ("Ford Motor Company", "Ford Motor"),
            ("Alphabet Inc.", "Alphabet"),
            ("Plain Name", "Plain Name"),
        ],
    )
    def test_strips_trailing_legal_suffixes(self, raw, expected):
        assert news._simplify_company_name(raw) == expected


class TestSearchTerm:
    def test_uses_resolved_name(self, monkeypatch):
        monkeypatch.setattr(
            news.get_ticker_details, "func", lambda ticker: {"name": "NVIDIA Corporation"}
        )
        assert news._search_term("nvda") == "NVIDIA"

    def test_falls_back_to_ticker_on_lookup_failure(self, monkeypatch):
        def fail(ticker):
            raise ValueError("no details")

        monkeypatch.setattr(news.get_ticker_details, "func", fail)
        assert news._search_term("nvda") == "NVDA"

    def test_falls_back_to_ticker_when_name_missing(self, monkeypatch):
        monkeypatch.setattr(news.get_ticker_details, "func", lambda ticker: {"name": None})
        assert news._search_term("nvda") == "NVDA"


class TestClient:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.setattr(news, "NEWS_API_KEY", "")
        with pytest.raises(ValueError, match="not set"):
            news._client()


class TestValidateDateRange:
    def test_accepts_valid_range(self):
        news._validate_date_range("2026-01-01", "2026-01-31")  # does not raise

    def test_rejects_malformed_date(self):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            news._validate_date_range("01-01-2026", "2026-01-31")

    def test_rejects_start_after_end(self):
        with pytest.raises(ValueError, match="must not be after"):
            news._validate_date_range("2026-02-01", "2026-01-01")


class TestParseArticleDate:
    def test_parses_iso8601_with_z_suffix(self):
        parsed = news._parse_article_date("2026-10-10T12:00:00Z")
        assert parsed == datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)

    def test_returns_none_for_blank(self):
        assert news._parse_article_date(None) is None
        assert news._parse_article_date("") is None

    def test_returns_none_for_unparsable(self):
        assert news._parse_article_date("not a date") is None


class FakeYfTicker:
    def __init__(self, news_items) -> None:
        self._news = news_items

    @property
    def news(self):
        return self._news


def _news_item(title: str, pub_date: str) -> dict:
    return {
        "content": {
            "title": title,
            "pubDate": pub_date,
            "provider": {"displayName": "Yahoo"},
            "canonicalUrl": {"url": f"https://example.com/{title}"},
            "summary": "summary",
        }
    }


class TestFetchFromYfinance:
    def test_filters_to_date_window(self, monkeypatch):
        items = [
            _news_item("in window", "2026-10-05T00:00:00Z"),
            _news_item("too old", "2026-01-01T00:00:00Z"),
        ]
        monkeypatch.setattr(news.yf, "Ticker", lambda ticker: FakeYfTicker(items))
        articles = news._fetch_from_yfinance("NVDA", "2026-10-01", "2026-10-10")
        assert [a["headline"] for a in articles] == ["in window"]

    def test_raises_when_no_news_at_all(self, monkeypatch):
        monkeypatch.setattr(news.yf, "Ticker", lambda ticker: FakeYfTicker([]))
        with pytest.raises(ValueError, match="no news"):
            news._fetch_from_yfinance("NVDA", "2026-10-01", "2026-10-10")

    def test_raises_when_nothing_in_window(self, monkeypatch):
        items = [_news_item("too old", "2020-01-01T00:00:00Z")]
        monkeypatch.setattr(news.yf, "Ticker", lambda ticker: FakeYfTicker(items))
        with pytest.raises(ValueError, match="had no news"):
            news._fetch_from_yfinance("NVDA", "2026-10-01", "2026-10-10")

    def test_raises_on_sdk_error(self, monkeypatch):
        class Boom:
            @property
            def news(self):
                raise RuntimeError("yfinance down")

        monkeypatch.setattr(news.yf, "Ticker", lambda ticker: Boom())
        with pytest.raises(ValueError, match="request failed"):
            news._fetch_from_yfinance("NVDA", "2026-10-01", "2026-10-10")

    def test_undated_items_are_kept(self, monkeypatch):
        items = [{"content": {"title": "no date", "provider": {}, "summary": "s"}}]
        monkeypatch.setattr(news.yf, "Ticker", lambda ticker: FakeYfTicker(items))
        articles = news._fetch_from_yfinance("NVDA", "2026-10-01", "2026-10-10")
        assert articles[0]["headline"] == "no date"
        assert articles[0]["source"] == "Yahoo Finance"


class TestFetchArticles:
    def test_newsapi_primary(self, monkeypatch):
        monkeypatch.setattr(
            news, "_fetch_from_newsapi", lambda name, f, t, ticker: [{"headline": "from newsapi"}]
        )
        articles = news._fetch_articles("NVDA", "NVIDIA", "2026-10-01", "2026-10-10")
        assert articles == [{"headline": "from newsapi"}]

    def test_falls_back_to_yfinance(self, monkeypatch):
        def newsapi_fail(name, f, t, ticker):
            raise ValueError("no key")

        monkeypatch.setattr(news, "_fetch_from_newsapi", newsapi_fail)
        monkeypatch.setattr(
            news, "_fetch_from_yfinance", lambda ticker, f, t: [{"headline": "from yfinance"}]
        )
        articles = news._fetch_articles("NVDA", "NVIDIA", "2026-10-01", "2026-10-10")
        assert articles == [{"headline": "from yfinance"}]

    def test_raises_when_both_fail(self, monkeypatch):
        def fail(*a, **k):
            raise ValueError("down")

        monkeypatch.setattr(news, "_fetch_from_newsapi", fail)
        monkeypatch.setattr(news, "_fetch_from_yfinance", fail)
        with pytest.raises(ValueError, match="All news sources failed"):
            news._fetch_articles("NVDA", "NVIDIA", "2026-10-01", "2026-10-10")


class TestGetNewsData:
    def test_returns_articles(self, monkeypatch):
        monkeypatch.setattr(news, "_search_term", lambda ticker: "NVIDIA")
        monkeypatch.setattr(news, "_fetch_articles", lambda ticker, name, f, t: [{"headline": "h"}])
        data = json.loads(news.get_news_data.func("nvda"))
        assert data["ticker"] == "NVDA"
        assert data["articles"] == [{"headline": "h"}]

    def test_zero_articles_returns_message(self, monkeypatch):
        monkeypatch.setattr(news, "_search_term", lambda ticker: "NVIDIA")
        monkeypatch.setattr(news, "_fetch_articles", lambda ticker, name, f, t: [])
        data = json.loads(news.get_news_data.func("nvda", days=5))
        assert "message" in data

    def test_fetch_failure_returns_error(self, monkeypatch):
        monkeypatch.setattr(news, "_search_term", lambda ticker: "NVIDIA")

        def fail(ticker, name, f, t):
            raise ValueError("all sources down")

        monkeypatch.setattr(news, "_fetch_articles", fail)
        data = json.loads(news.get_news_data.func("nvda"))
        assert "error" in data

    def test_non_numeric_days_falls_back_to_default(self, monkeypatch):
        captured = {}

        monkeypatch.setattr(news, "_search_term", lambda ticker: "NVIDIA")

        def fake_fetch(ticker, name, f, t):
            captured["from"] = f
            return [{"headline": "h"}]

        monkeypatch.setattr(news, "_fetch_articles", fake_fetch)
        news.get_news_data.func("nvda", days="not a number")
        from_date = datetime.strptime(captured["from"], "%Y-%m-%d").date()
        today = datetime.now(timezone.utc).date()
        assert (today - from_date).days == 7


class TestGetNewsByPeriod:
    def test_returns_articles_for_explicit_range(self, monkeypatch):
        monkeypatch.setattr(news, "_search_term", lambda ticker: "NVIDIA")
        monkeypatch.setattr(news, "_fetch_articles", lambda ticker, name, f, t: [{"headline": "h"}])
        data = json.loads(news.get_news_by_period.func("nvda", "2026-01-01", "2026-01-31"))
        assert data["from"] == "2026-01-01"
        assert data["articles"] == [{"headline": "h"}]

    def test_invalid_range_returns_error_without_fetching(self, monkeypatch):
        monkeypatch.setattr(news, "_search_term", lambda ticker: "NVIDIA")

        def should_not_run(*a, **k):
            raise AssertionError("should not be called")

        monkeypatch.setattr(news, "_fetch_articles", should_not_run)
        data = json.loads(news.get_news_by_period.func("nvda", "2026-02-01", "2026-01-01"))
        assert "error" in data
