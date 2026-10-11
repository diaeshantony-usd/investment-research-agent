"""
test_tools_news.py
==================
Offline tests for the NewsAPI query built by ``tools/news.py``. A fake NewsAPI client
records the request, so the tests check what is searched without network access.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_agent.tools import news


class FakeNewsApi:
    """Records ``get_everything`` arguments and returns a fixed article list."""

    def __init__(self, articles: list[dict[str, Any]]) -> None:
        self.articles = articles
        self.kwargs: dict[str, Any] = {}

    def get_everything(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        return {"status": "ok", "articles": self.articles}


def _article(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "source": {"name": "Reuters"},
        "publishedAt": "2026-10-10T12:00:00Z",
        "url": f"https://example.com/{len(title)}",
        "description": "summary",
    }


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeNewsApi([_article("Nvidia beats estimates"), _article("[Removed]")])
    monkeypatch.setattr(news, "_client", lambda: client)
    return client


def test_searches_headlines_for_name_or_ticker(fake_client):
    news._fetch_from_newsapi("NVIDIA", "2026-09-11", "2026-10-11", ticker="NVDA")
    assert fake_client.kwargs["qintitle"] == '"NVIDIA" OR NVDA'
    assert "q" not in fake_client.kwargs  # no full-text search
    assert "pypi.org" in fake_client.kwargs["exclude_domains"]
    assert fake_client.kwargs["sort_by"] == "relevancy"


def test_removed_articles_are_skipped(fake_client):
    articles = news._fetch_from_newsapi("NVIDIA", "2026-09-11", "2026-10-11", ticker="NVDA")
    assert [a["headline"] for a in articles] == ["Nvidia beats estimates"]


@pytest.mark.parametrize(
    ("name", "ticker", "expected"),
    [
        ("Apple", "AAPL", '"Apple" OR AAPL'),
        ("Advanced Micro Devices", "AMD", '"Advanced Micro Devices" OR AMD'),
        ("Ford Motor", "F", '"Ford Motor"'),  # one-letter ticker would match noise
        ("NVDA", "NVDA", '"NVDA"'),  # name lookup failed, ticker used as the name
    ],
)
def test_title_query(name, ticker, expected):
    assert news._title_query(name, ticker) == expected
