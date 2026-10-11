"""
test_news_chain.py
==================
Offline tests for the News analyst's five-stage prompt chain
(``research_agent.workflows.news_chain``). The news tool and the LLM are replaced with
fixed responses, so each stage's behaviour and code checks can be asserted exactly.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from research_agent.agent_framework import GraphMemory, Tool
from research_agent.interfaces.llm import LLM
from research_agent.workflows.news_chain import (
    NewsChainAgent,
    _normalise_citations,
    _parse_json_array,
    _resolve_ticker,
)

TODAY = date(2026, 10, 11)

RAW_ARTICLES = [
    {
        "headline": "NVIDIA wins <b>$5 billion</b> data-center order",
        "source": "Reuters",
        "date": "2026-10-09",
        "url": "https://example.com/a",
        "summary": "Order worth $5 billion lifts guidance by 3%.",
    },
    {  # duplicate headline (different case/punctuation) -> dropped
        "headline": "Nvidia wins $5 billion data center order!",
        "source": "Yahoo",
        "date": "2026-10-09",
        "url": "https://example.com/b",
        "summary": "Same story.",
    },
    {
        "headline": "US reviews chip export rules",
        "source": "Bloomberg",
        "date": "2026-10-05",
        "url": "https://example.com/c",
        "summary": "Officials weigh new limits on AI chips.",
    },
    {  # older than the 30-day window -> dropped
        "headline": "Old story from last quarter",
        "source": "FT",
        "date": "2026-08-01",
        "url": "https://example.com/d",
        "summary": "Stale.",
    },
    {"headline": "", "source": "x", "date": "2026-10-08", "url": "", "summary": "no headline"},
]

CLASSIFY_REPLY = """```json
[{"id": "A1", "topic": "earnings", "sentiment": "bullish", "relevance": 5},
 {"id": "A2", "topic": "not-a-topic", "sentiment": "Bearish", "relevance": 9}]
```"""

EXTRACT_REPLY = json.dumps(
    [
        {
            "id": "A1",
            "entities": ["NVIDIA"],
            "figures": [
                {"value": "$5 billion", "context": "order size"},  # in the article -> kept
                {"value": "$7 billion", "context": "invented"},  # not in the article -> rejected
            ],
        },
        {"id": "A2", "entities": ["US Commerce Department"], "figures": []},
    ]
)

SUMMARY_REPLY = """- NVIDIA won a $5 billion data-center order and raised guidance [A1]
- This bullet cites nothing and must be dropped
- Export rules may tighten [A9]"""


class ScriptedLLM(LLM):
    """Returns queued replies in order and records every prompt it receives."""

    model = "scripted"

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def chat(self, messages, tools=None):
        self.prompts.append(messages[-1]["content"])
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return reply, [], {"role": "assistant", "content": reply}


def make_tool(articles: list[dict[str, Any]]) -> tuple[Tool, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def fake_news(ticker: str, days: int = 7) -> str:
        calls.append({"ticker": ticker, "days": days})
        return json.dumps({"ticker": ticker, "articles": articles})

    return Tool(name="get_news_data", description="news", func=fake_news), calls


def make_agent(llm: LLM, articles=RAW_ARTICLES) -> tuple[NewsChainAgent, list[dict[str, Any]]]:
    tool, calls = make_tool(articles)
    agent = NewsChainAgent(
        name="NewsAnalyst", role="news", news_tool=tool, llm=llm, today=lambda: TODAY
    )
    return agent, calls


PROMPT = 'User Request / Goal:\nAnalyse NVIDIA\n\nCurrent Context:\n{"active_ticker": "NVDA"}'


class TestStages:
    def test_full_chain_produces_cited_digest(self):
        llm = ScriptedLLM([CLASSIFY_REPLY, EXTRACT_REPLY, SUMMARY_REPLY])
        agent, calls = make_agent(llm)
        response = agent.run(PROMPT)

        # Stage 1: ingest uses a 30-day window for the resolved ticker.
        assert calls == [{"ticker": "NVDA", "days": 30}]
        # Stages 3-5 each make exactly one LLM call.
        assert len(llm.prompts) == 3
        assert response.iterations == 3

        digest = response.content
        assert "## News digest: NVDA (last 30 days, 2 articles)" in digest
        assert "[A1] NVIDIA wins $5 billion data-center order - Reuters" in digest  # HTML stripped
        assert "Old story" not in digest  # out of window
        assert "cites nothing" not in digest  # uncited bullet dropped
        assert "[A9]" not in digest  # citation to a non-existent article dropped
        # The shortfall is filled from the remaining article, still cited.
        assert "US reviews chip export rules (Bloomberg, 2026-10-05) [A2]" in digest

    def test_preprocess_cleans_dedupes_and_filters(self):
        agent, _ = make_agent(ScriptedLLM([]))
        articles = agent._preprocess({"articles": RAW_ARTICLES}, [])
        assert [a.id for a in articles] == ["A1", "A2"]
        assert articles[0].headline == "NVIDIA wins $5 billion data-center order"

    def test_classify_validates_labels(self):
        llm = ScriptedLLM([CLASSIFY_REPLY, EXTRACT_REPLY, SUMMARY_REPLY])
        agent, _ = make_agent(llm)
        articles = agent._preprocess({"articles": RAW_ARTICLES}, [])
        agent._classify("NVDA", articles, [])
        assert (articles[0].topic, articles[0].sentiment, articles[0].relevance) == (
            "earnings",
            "Bullish",
            5,
        )
        # Unknown topic falls back to "other"; out-of-range relevance is clamped.
        assert (articles[1].topic, articles[1].sentiment, articles[1].relevance) == (
            "other",
            "Bearish",
            5,
        )

    def test_extract_rejects_figures_not_in_article(self):
        llm = ScriptedLLM([EXTRACT_REPLY])
        agent, _ = make_agent(llm)
        articles = agent._preprocess({"articles": RAW_ARTICLES}, [])
        trace: list = []
        agent._extract("NVDA", articles, trace)
        assert articles[0].figures == [{"value": "$5 billion", "context": "order size"}]
        record = [
            s for s in trace if s.metadata.get("stage") == "extract" and isinstance(s.content, dict)
        ]
        assert record[-1].content["result"]["figures_rejected"] == ["A1:$7 billion"]

    def test_off_topic_articles_are_dropped_after_classify(self):
        classify = json.dumps(
            [
                {"id": "A1", "topic": "earnings", "sentiment": "Bullish", "relevance": 5},
                {"id": "A2", "topic": "other", "sentiment": "Neutral", "relevance": 1},
            ]
        )
        llm = ScriptedLLM([classify, EXTRACT_REPLY, "- NVIDIA won an order [A1]"])
        agent, _ = make_agent(llm)
        digest = agent.run(PROMPT).content
        assert "(last 30 days, 1 articles)" in digest
        assert "US reviews chip export rules" not in digest  # A2 was rated off-topic
        assert "(1 bullish, 0 neutral, 0 bearish)" in digest

    def test_all_off_topic_keeps_articles(self):
        classify = json.dumps([{"id": f"A{i}", "relevance": 1} for i in (1, 2)])
        llm = ScriptedLLM([classify, EXTRACT_REPLY, SUMMARY_REPLY])
        agent, _ = make_agent(llm)
        assert "2 articles" in agent.run(PROMPT).content

    def test_overall_sentiment_is_counted_in_code(self):
        llm = ScriptedLLM([CLASSIFY_REPLY, EXTRACT_REPLY, SUMMARY_REPLY])
        agent, _ = make_agent(llm)
        digest = agent.run(PROMPT).content
        # One bullish and one bearish article: the tie is reported as Mixed.
        assert "**Overall news sentiment:** Mixed (1 bullish, 0 neutral, 1 bearish)" in digest


class TestResilience:
    def test_no_articles_means_no_llm_calls(self):
        llm = ScriptedLLM([])
        agent, _ = make_agent(llm, articles=[])
        response = agent.run(PROMPT)
        assert llm.prompts == []
        assert response.content.startswith("No news data available from tools for NVDA")

    def test_llm_failures_fall_back_to_cited_headlines(self):
        llm = ScriptedLLM([RuntimeError("timeout")] * 3)
        agent, _ = make_agent(llm)
        response = agent.run(PROMPT)
        assert "[A1]" in response.content
        assert "[A2]" in response.content
        assert sum(1 for s in response.trace if s.step_type == "error") >= 3

    def test_no_ticker_is_reported_not_guessed(self):
        agent, calls = make_agent(ScriptedLLM([]))
        response = agent.run("What is operating margin?")
        assert calls == []
        assert "no company ticker" in response.content

    def test_trace_records_tool_call_for_the_graph(self):
        llm = ScriptedLLM([CLASSIFY_REPLY, EXTRACT_REPLY, SUMMARY_REPLY])
        agent, _ = make_agent(llm)
        trace = agent.run(PROMPT).trace
        tool_calls = [s for s in trace if s.step_type == "tool_call"]
        assert tool_calls[0].content["tool"] == "get_news_data"
        assert any(s.step_type == "tool_result" for s in trace)


class TestHelpers:
    @pytest.mark.parametrize(
        ("prompt", "expected"),
        [
            ('{"active_ticker": "AAPL"} RESEARCH_REQUIRED: NVDA', "AAPL"),
            ("Prior findings: RESEARCH_REQUIRED: **TSLA** - margins", "TSLA"),
            ("Tell me about Microsoft", "MSFT"),
        ],
    )
    def test_resolve_ticker_priority(self, prompt, expected):
        assert _resolve_ticker(prompt) == expected

    @pytest.mark.parametrize(
        ("text", "expected_text", "expected_ids"),
        [
            ("Order won [A1]", "Order won [A1]", {"A1"}),
            ("Order won [A1, A2]", "Order won [A1][A2]", {"A1", "A2"}),
            ("Order won (A1)", "Order won [A1]", {"A1"}),
            ("Order won 【A1】", "Order won [A1]", {"A1"}),
            ("Order won [A1 and A2]", "Order won [A1][A2]", {"A1", "A2"}),
            ("Order won, no source", "Order won, no source", set()),
        ],
    )
    def test_citation_styles_are_normalised(self, text, expected_text, expected_ids):
        assert _normalise_citations(text) == (expected_text, expected_ids)

    def test_leading_numbers_in_a_sentence_are_kept(self):
        summary = "- 2026 revenue guidance was raised [A1]"
        llm = ScriptedLLM([CLASSIFY_REPLY, EXTRACT_REPLY, summary])
        agent, _ = make_agent(llm)
        assert "- 2026 revenue guidance was raised [A1]" in agent.run(PROMPT).content

    def test_summary_with_grouped_citations_is_kept(self):
        summary = "1. NVIDIA won an order and exports face review [A1, A2]"
        llm = ScriptedLLM([CLASSIFY_REPLY, EXTRACT_REPLY, summary])
        agent, _ = make_agent(llm)
        digest = agent.run(PROMPT).content
        assert "- NVIDIA won an order and exports face review [A1][A2]" in digest

    def test_parse_json_array_tolerates_prose_and_fences(self):
        assert _parse_json_array('Sure!\n```json\n[{"id": "A1"}]\n```') == [{"id": "A1"}]
        assert _parse_json_array("not json") is None


def test_news_analyst_runs_inside_the_graph(monkeypatch):
    """The chain plugs into the existing graph and its tool data reaches GraphState."""
    from research_agent.tools.news import get_news_data
    from research_agent.workflows.graph import build_research_workflow

    monkeypatch.setattr(
        get_news_data,
        "func",
        lambda ticker, days=7: json.dumps({"ticker": ticker, "articles": RAW_ARTICLES}),
    )
    monkeypatch.setattr("research_agent.workflows.news_chain._utc_today", lambda: TODAY)
    llm = ScriptedLLM(
        [
            "RESEARCH_REQUIRED: NVDA - news",  # Planner
            "Earnings summary",  # EarningsAnalyst (no tool call)
            "Market summary",  # MarketAnalyst (no tool call)
            CLASSIFY_REPLY,  # NewsAnalyst stage 3
            EXTRACT_REPLY,  # NewsAnalyst stage 4
            SUMMARY_REPLY,  # NewsAnalyst stage 5
            "# Research Brief: NVDA",  # DraftWriter
            "SCORE: 9/10 STATUS: PASS",  # Critic
        ]
    )
    state = build_research_workflow(llm=llm, memory=GraphMemory()).invoke(
        {"question": "Analyse NVDA"}
    )
    assert "## News digest: NVDA" in state.agents["NewsAnalyst"]
    assert "get_news_data" in state.tools["NewsAnalyst"]
    specialists = next(s for s in state.trajectory if s["node"] == "specialists")
    assert "get_news_data" in specialists["tools"]
