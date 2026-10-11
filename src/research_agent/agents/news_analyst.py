"""
agents/news_analyst.py
======================
News Analyst: a real 5-stage prompt chain (W1), not a single ReAct call.

    ingest -> preprocess -> classify -> extract -> summarize

Each stage's output is appended to the response trace as its own
``TraceStep`` (``step_type="chain_stage"``), so a notebook can print every
stage individually for one batch of headlines - the W1 demonstration
requirement - instead of only seeing the final digest.

``NewsAnalystChain`` is not an ``Agent`` subclass (ingest/preprocess are
deterministic, not LLM reasoning), but it duck-types the interface
``CompiledStateGraph._execute_parallel_node`` needs - ``.name``, ``.tools``,
``.run(prompt) -> AgentResponse`` - so it drops into the existing
``[earnings_analyst, market_analyst, news_analyst]`` parallel group in
``workflows/graph.py`` with no changes to the orchestrator.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from research_agent.agent_framework import AgentResponse, Tool, TraceStep
from research_agent.config import DEFAULT_MODEL, logger
from research_agent.interfaces.llm import LLM
from research_agent.memory.tickers import extract_tickers
from research_agent.tools.news import get_news_data

NEWS_ANALYST_ROLE = (
    "You are the Financial News and Catalyst Analyst. You run a fixed 5-stage pipeline - "
    "ingest, preprocess, classify, extract, summarize - over get_news_data's results. "
    "Report ONLY news items the tool actually returned; never invent headlines, articles, "
    "or dates."
)

_CLASSIFY_PROMPT = (
    "You are a financial news classifier. Given a numbered list of headlines, return the "
    "topic and sentiment of each. Topics: earnings, product, legal, macro, leadership, "
    "other. Sentiment: positive, negative, neutral. Reply with JSON only: "
    '{"items": [{"index": 0, "topic": "...", "sentiment": "..."}]}'
)
_EXTRACT_PROMPT = (
    "You are a financial news entity extractor. Given a numbered list of headlines, pull "
    "out any company names, people, and numeric figures (dollar amounts, percentages) "
    "mentioned. Reply with JSON only: "
    '{"items": [{"index": 0, "entities": ["..."], "figures": ["..."]}]}'
)
_SUMMARIZE_PROMPT = (
    "You write concise, cited investment research summaries. Given a numbered list of "
    "classified headlines, write exactly 5 bullet points summarizing what matters for an "
    "investor. Each bullet must end with a citation to the headline index it is based on, "
    "like [2]. Reply with JSON only: "
    '{"bullets": ["...", "...", "...", "...", "..."]}'
)

_POSITIVE_WORDS = ("beat", "surge", "record", "growth", "upgrade", "gain", "strong", "rally")
_NEGATIVE_WORDS = ("miss", "fall", "lawsuit", "downgrade", "loss", "decline", "cut", "probe")

_PREPROCESS_WINDOW_DAYS = 30


def _resolve_llm(model: str | LLM | None, llm: LLM | None) -> tuple[LLM, str]:
    """Mirror ``Agent.__init__``'s model/llm resolution so this chain is a drop-in."""
    if isinstance(model, LLM):
        return model, str(getattr(model, "model", DEFAULT_MODEL))
    if llm is not None:
        return llm, str(model or getattr(llm, "model", DEFAULT_MODEL))

    from research_agent.interfaces.llm import get_llm

    resolved_model = str(model or DEFAULT_MODEL)
    return get_llm(model=resolved_model), resolved_model


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse an LLM's JSON reply, tolerating a ```json fenced block."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    return json.loads(text)


def _keyword_sentiment(headline: str) -> str:
    lowered = headline.lower()
    if any(word in lowered for word in _NEGATIVE_WORDS):
        return "negative"
    if any(word in lowered for word in _POSITIVE_WORDS):
        return "positive"
    return "neutral"


def _numbered_headlines(articles: list[dict[str, Any]]) -> str:
    return "\n".join(f"{i}. {a['headline']}" for i, a in enumerate(articles))


class NewsAnalystChain:
    """The News analyst's 5-stage prompt chain: ingest, preprocess, classify, extract, summarize."""

    name = "NewsAnalyst"
    role = NEWS_ANALYST_ROLE

    def __init__(
        self,
        model: str | LLM | None = None,
        llm: LLM | None = None,
        verbose: bool = False,
    ) -> None:
        self.llm, self.model = _resolve_llm(model, llm)
        self.verbose = verbose
        # A dict keyed by tool name, like Agent.tools, so the orchestrator's
        # `for t in agent.tools` / tool-call bookkeeping works unchanged.
        self.tools: dict[str, Tool] = {"get_news_data": get_news_data}

    def run(self, prompt: str | None = None) -> AgentResponse:
        """Run all 5 stages for the ticker named in ``prompt`` and return the digest."""
        trace: list[TraceStep] = []
        now = _now()

        tickers = extract_tickers(prompt or "")
        if not tickers:
            final_text = "No ticker found in the request; the News analyst needs an active company."
            trace.append(TraceStep(now, self.name, "final_answer", final_text))
            return AgentResponse(self.name, prompt or "", final_text, trace, [], 0)
        ticker = tickers[0]

        # --- Stage 1: ingest ----------------------------------------------------------
        raw_json = self.tools["get_news_data"].execute(ticker=ticker)
        tool_call = {"tool": "get_news_data", "arguments": {"ticker": ticker}}
        trace.append(TraceStep(now, self.name, "tool_call", tool_call))
        trace.append(
            TraceStep(now, self.name, "tool_result", raw_json, metadata={"tool": "get_news_data"})
        )
        articles = json.loads(raw_json).get("articles", [])
        self._log_stage(trace, "ingest", {"ticker": ticker, "count": len(articles)})

        # --- Stage 2: preprocess (deterministic: dedupe, clean, 30-day filter) --------
        preprocessed = self._preprocess(articles)
        preprocess_summary = {
            "count": len(preprocessed),
            "headlines": [a["headline"] for a in preprocessed],
        }
        self._log_stage(trace, "preprocess", preprocess_summary)

        if not preprocessed:
            final_text = f"No news data available from tools for {ticker}."
            trace.append(TraceStep(now, self.name, "final_answer", final_text))
            return AgentResponse(self.name, prompt or "", final_text, trace, [], 1)

        # --- Stage 3: classify (topic + sentiment per headline) -----------------------
        classified = self._classify(preprocessed)
        self._log_stage(trace, "classify", classified)

        # --- Stage 4: extract (entities + figures) -------------------------------------
        extracted = self._extract(preprocessed)
        self._log_stage(trace, "extract", extracted)

        # --- Stage 5: summarize (5 cited bullets) ---------------------------------------
        bullets = self._summarize(classified)
        self._log_stage(trace, "summarize", bullets)

        final_text = "\n".join(f"- {b}" for b in bullets)
        trace.append(TraceStep(_now(), self.name, "final_answer", final_text))

        if self.verbose:
            logger.info(
                "[NewsAnalyst] %s: %d articles -> %d bullets", ticker, len(articles), len(bullets)
            )

        return AgentResponse(
            agent_name=self.name,
            input_prompt=prompt or "",
            content=final_text,
            trace=trace,
            messages=[],
            iterations=5,
        )

    # ------------------------------------------------------------------ stages

    def _preprocess(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Dedupe by headline, drop empties, and keep only the last 30 days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=_PREPROCESS_WINDOW_DAYS)).date()
        seen_headlines: set[str] = set()
        cleaned: list[dict[str, Any]] = []
        for article in articles:
            headline = (article.get("headline") or "").strip()
            if not headline or headline.lower() in seen_headlines:
                continue
            date_str = article.get("date")
            if date_str:
                try:
                    if datetime.strptime(date_str[:10], "%Y-%m-%d").date() < cutoff:
                        continue
                except ValueError:
                    pass  # unparsable date: keep the article rather than drop it
            seen_headlines.add(headline.lower())
            cleaned.append({**article, "headline": headline})
        return cleaned

    def _classify(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            content, _, _ = self.llm.chat(
                [
                    {"role": "system", "content": _CLASSIFY_PROMPT},
                    {"role": "user", "content": _numbered_headlines(articles)},
                ]
            )
            items = _parse_json(content)["items"]
            return [
                {
                    "index": item["index"],
                    "headline": articles[item["index"]]["headline"],
                    "topic": item["topic"],
                    "sentiment": item["sentiment"],
                }
                for item in items
                if 0 <= item["index"] < len(articles)
            ]
        except Exception as exc:  # noqa: BLE001 - any LLM/parse failure falls back to a rule-based pass
            logger.warning("News classify stage fell back to keyword sentiment: %s", exc)
            return [
                {
                    "index": i,
                    "headline": a["headline"],
                    "topic": "other",
                    "sentiment": _keyword_sentiment(a["headline"]),
                }
                for i, a in enumerate(articles)
            ]

    def _extract(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            content, _, _ = self.llm.chat(
                [
                    {"role": "system", "content": _EXTRACT_PROMPT},
                    {"role": "user", "content": _numbered_headlines(articles)},
                ]
            )
            return _parse_json(content)["items"]
        except Exception as exc:  # noqa: BLE001 - fall back to a regex figure scan
            logger.warning("News extract stage fell back to regex figures: %s", exc)
            return [
                {
                    "index": i,
                    "entities": [],
                    "figures": re.findall(r"[$]?\d[\d.,]*%?", a["headline"]),
                }
                for i, a in enumerate(articles)
            ]

    def _summarize(self, classified: list[dict[str, Any]]) -> list[str]:
        numbered = "\n".join(
            f"{item['index']}. [{item['topic']}/{item['sentiment']}] {item['headline']}"
            for item in classified
        )
        try:
            content, _, _ = self.llm.chat(
                [
                    {"role": "system", "content": _SUMMARIZE_PROMPT},
                    {"role": "user", "content": numbered},
                ]
            )
            bullets = _parse_json(content).get("bullets") or []
        except Exception as exc:  # noqa: BLE001 - fall back to the raw headlines, cited
            bullets = []
            logger.warning("News summarize stage fell back to raw headlines: %s", exc)

        if not bullets:
            return [f"{item['headline']} [{item['index']}]" for item in classified[:5]]
        return bullets[:5]

    def _log_stage(self, trace: list[TraceStep], stage: str, output: Any) -> None:
        stage_payload = {"stage": stage, "output": output}
        trace.append(TraceStep(_now(), self.name, "chain_stage", stage_payload))
        if self.verbose:
            logger.info("[NewsAnalyst] stage=%s output=%s", stage, output)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")


def news_analyst_agent(
    model: str | LLM | None = None,
    llm: LLM | None = None,
    verbose: bool = False,
) -> NewsAnalystChain:
    """Create the News analyst's 5-stage prompt chain.

    Args:
        model: Model name, or an ``LLM`` client to use directly.
        llm: LLM client shared with the rest of the graph.
        verbose: Log each chain stage as it runs.

    Returns:
        A ``NewsAnalystChain`` - drop-in compatible with the ``Agent`` slot
        in the orchestrator's parallel specialists group.
    """
    return NewsAnalystChain(model=model, llm=llm, verbose=verbose)
