"""
test_ticker_lookup.py
=====================
Offline tests for Yahoo Finance-backed ticker resolution (``memory/tickers.py``) and
how ``ResearchSession`` uses it. Yahoo search is replaced by a small fake directory.
"""

from __future__ import annotations

import pytest

from research_agent.agent_framework import GraphMemory
from research_agent.memory import SessionStore, tickers
from research_agent.memory.session import TurnKind
from research_agent.memory.tickers import resolve_tickers

# What the fake "Yahoo search box" returns for each query.
FAKE_YAHOO = {
    "COST": [{"symbol": "COST", "name": "Costco Wholesale", "type": "EQUITY"}],
    "Costco": [{"symbol": "COST", "name": "Costco Wholesale", "type": "EQUITY"}],
    "costco": [{"symbol": "COST", "name": "Costco Wholesale", "type": "EQUITY"}],
    "Home Depot": [{"symbol": "HD", "name": "Home Depot", "type": "EQUITY"}],
    "SPY": [{"symbol": "SPY", "name": "SPDR S&P 500 ETF", "type": "ETF"}],
    "Bitcoin": [{"symbol": "BTC-USD", "name": "Bitcoin USD", "type": "CRYPTOCURRENCY"}],
    "nvidia": [{"symbol": "NVDA", "name": "NVIDIA Corporation", "type": "EQUITY"}],
    "home depot": [{"symbol": "HD", "name": "The Home Depot, Inc.", "type": "EQUITY"}],
    # Yahoo matches loosely: an unrelated company comes back for a finance phrase.
    "dividend yield": [{"symbol": "DVY", "name": "iShares Select Dividend", "type": "ETF"}],
    "buyback": [{"symbol": "XYZ", "name": "Unrelated Holdings", "type": "EQUITY"}],
    "NVDAA": [],
    "NVDAA~fuzzy": [{"symbol": "NVDA", "name": "NVIDIA", "type": "EQUITY"}],
    "XQZT": [],
    "XQZT~fuzzy": [],
}


@pytest.fixture
def yahoo(monkeypatch):
    """Serve FAKE_YAHOO and record every query sent to Yahoo."""
    queries: list[str] = []

    def fake_search(query, fuzzy=False):
        key = f"{query}~fuzzy" if fuzzy else query
        queries.append(key)
        return tuple(FAKE_YAHOO.get(key, []))

    monkeypatch.setattr(tickers, "_yahoo_search", fake_search)
    return queries


class TestResolveTickers:
    def test_company_name_is_found_via_yahoo(self, yahoo):
        assert resolve_tickers("How is Costco doing?").tickers == ("COST",)

    def test_multi_word_company_name(self, yahoo):
        assert resolve_tickers("Analyse Home Depot margins").tickers == ("HD",)

    def test_typed_symbol_is_confirmed(self, yahoo):
        assert resolve_tickers("Analyse COST").tickers == ("COST",)
        assert resolve_tickers("Is SPY expensive?").tickers == ("SPY",)  # ETFs are valid symbols

    def test_lowercase_names_after_cue_words(self, yahoo):
        assert resolve_tickers("tell me about nvidia").tickers == ("NVDA",)
        assert resolve_tickers("what are the margins of home depot").tickers == ("HD",)

    @pytest.mark.parametrize(
        "question",
        [
            "What about Costco's debt?",
            "what about costco\u2019s margins",
            "and Costco?",
            "and costco?",
        ],
    )
    def test_follow_up_phrasings_find_the_company(self, yahoo, question):
        assert resolve_tickers(question).tickers == ("COST",)

    def test_result_must_match_the_company_name(self, yahoo):
        # "buyback" returns an unrelated company from the fake search: rejected.
        assert resolve_tickers("tell me about buyback").tickers == ()

    def test_no_company_names_are_hard_coded(self):
        assert not hasattr(tickers, "COMPANY_TICKERS")

    def test_typo_gets_a_suggestion(self, yahoo):
        found = resolve_tickers("Analyse NVDAA")
        assert found.tickers == ()
        assert (found.unknown, found.suggestion) == ("NVDAA", "NVDA")

    def test_extra_letter_typo_is_corrected_when_fuzzy_search_fails(self, monkeypatch):
        # Seen live: Yahoo's fuzzy search returned nothing useful for NVDAA.
        directory = {"NVDA": [{"symbol": "NVDA", "name": "NVIDIA", "type": "EQUITY"}]}

        def fake_search(query, fuzzy=False):
            return () if fuzzy else tuple(directory.get(query, []))

        monkeypatch.setattr(tickers, "_yahoo_search", fake_search)
        found = resolve_tickers("Analyse NVDAA")
        assert (found.unknown, found.suggestion) == ("NVDAA", "NVDA")

    def test_non_stock_results_are_ignored(self, yahoo):
        assert resolve_tickers("What about Bitcoin?").tickers == ()

    def test_finance_terms_are_not_searched_as_names(self, yahoo):
        assert resolve_tickers("Explain Return on Equity and Price to Earnings").tickers == ()
        assert yahoo == []

    def test_acronyms_are_not_tickers(self, yahoo):
        assert resolve_tickers("Is GPU demand rising?").tickers == ()
        assert yahoo == []

    def test_yahoo_unreachable_accepts_typed_symbols(self):
        # The autouse conftest fixture makes Yahoo unreachable.
        assert resolve_tickers("Analyse XQZT").tickers == ("XQZT",)
        assert resolve_tickers("How is Costco doing?").tickers == ()


@pytest.fixture
def store():
    return SessionStore(":memory:")


@pytest.fixture
def memory(tmp_path):
    return GraphMemory(filepath=str(tmp_path / "graph.json"))


class TestSessionUsesLookup:
    def test_company_name_question_runs_research(self, yahoo, store, memory):
        from tests.test_session import FakeWorkflow, make_session

        workflow = FakeWorkflow()
        session = make_session(store, memory, workflow=workflow)
        session.ask("How is Costco doing?", show=False)
        assert session.active_ticker == "COST"
        assert workflow.calls[0]["active_ticker"] == "COST"

    def test_typo_is_answered_without_running_agents(self, yahoo, store, memory):
        from tests.test_session import FakeWorkflow, make_session

        workflow = FakeWorkflow()
        session = make_session(store, memory, workflow=workflow)
        reply = session.ask("Analyse NVDAA", show=False)
        assert reply == "I couldn't find a stock with the ticker NVDAA. Did you mean NVDA?"
        assert session.last_kind is TurnKind.UNKNOWN_TICKER
        assert workflow.calls == []
