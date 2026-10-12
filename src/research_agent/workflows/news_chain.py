"""
workflows/news_chain.py
=======================
The News analyst's five-stage **prompt chain** (workflow pattern: prompt chaining).

Instead of one open-ended prompt, the News analyst runs a fixed pipeline in which each
stage has one narrow job and hands a structured result to the next::

    1. ingest      (code) fetch the last 30 days of articles with ``get_news_data``
    2. preprocess  (code) clean text, drop duplicates and out-of-window items, number A1..An
    3. classify    (LLM)  topic, sentiment and relevance for every article -> JSON; articles
                          rated off-topic (relevance 1) are then dropped
    4. extract     (LLM)  entities and figures per article -> JSON, then a code check keeps
                          only figures that literally appear in that article
    5. summarize   (LLM)  five bullets, each citing article ids like [A3]; a code check drops
                          uncited bullets and fills gaps from the top-ranked articles

Narrow stages are easier for the model to get right, every intermediate result is
recorded in the trace, and the code checks between stages stop invented headlines or
numbers from reaching the brief. If an LLM stage fails or returns malformed JSON, the
chain falls back to a deterministic result for that stage rather than aborting, so the
graph always receives a usable, cited digest.

``NewsChainAgent`` subclasses ``Agent`` and returns a normal ``AgentResponse``, so it
drops into the existing graph (``workflows/graph.py``) unchanged: its tool call is
recorded in the trace and harvested into ``GraphState.tools`` like any other agent's.

Owner of the News analyst: Ashok Bhairwal. Prompt chain implementation: Diaesh Antony.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Final

from research_agent.agent_framework import Agent, AgentResponse, Tool, TraceStep
from research_agent.config import logger
from research_agent.memory.tickers import extract_tickers

__all__ = ["NEWS_LOOKBACK_DAYS", "SUMMARY_BULLETS", "NewsArticle", "NewsChainAgent"]

NEWS_LOOKBACK_DAYS: Final = 30
SUMMARY_BULLETS: Final = 5
# Upper bound on articles sent through the LLM stages (keeps prompts small).
_MAX_ARTICLES: Final = 12
# Articles the classifier rates below this relevance (1-5) are treated as off-topic.
_MIN_RELEVANCE: Final = 2

TOPICS: Final = (
    "earnings",
    "product",
    "regulation_legal",
    "macro_market",
    "management",
    "deal_partnership",
    "analyst_rating",
    "other",
)
SENTIMENTS: Final = ("Bullish", "Neutral", "Bearish")

_NEWS_TOOL: Final = "get_news_data"
# Citation groups as models actually write them: [A3], [A3, A5], [A3][A5], (A3), 【A3】.
_CITATION_GROUP = re.compile(r"[\[(【]\s*(A\d+(?:\s*(?:,|;|/|&|and)\s*A\d+)*)\s*[\])】]")
_ARTICLE_ID = re.compile(r"A\d+")
# A leading bullet or numbered-list marker: "- ", "* ", "• ", "1. ", "2) ".
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_TAG = re.compile(r"<[^>]+>")
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_ACTIVE_TICKER = re.compile(r'"active_ticker"\s*:\s*"([A-Z][A-Z.\-]{0,9})"')
_PLANNED_TICKER = re.compile(r"RESEARCH_REQUIRED:\s*\**\s*\$?([A-Z][A-Z.\-]{0,9})")

_SYSTEM = (
    "You are one stage of a financial news analysis pipeline. Follow the instructions "
    "exactly, use only the articles provided, and reply with the requested format only."
)

_CLASSIFY_PROMPT = """Classify each news article about {ticker}.

For every article return:
- "id": the article id (e.g. "A1")
- "topic": one of {topics}
- "sentiment": the article's likely effect on {ticker} stock: one of {sentiments}
- "relevance": 1 (barely about {ticker}) to 5 (directly about {ticker})

Articles:
{articles}

Reply with a JSON array only, one object per article."""

_EXTRACT_PROMPT = """Extract the key facts from each news article about {ticker}.

For every article return:
- "id": the article id
- "entities": companies, people, products or regulators named in the article
- "figures": numbers exactly as written in the article text (amounts, percentages,
  dates), each as {{"value": "<exact text>", "context": "<what it measures>"}}

Copy figures character-for-character from the article. Do not compute, convert or
add figures that are not in the text.

Articles:
{articles}

Reply with a JSON array only, one object per article."""

_SUMMARIZE_PROMPT = """Write a news digest for {ticker} from the analysed articles below.

Rules:
- Exactly {count} bullet points, most important first, each starting with "- ".
- Each bullet states one development and its likely effect on {ticker}.
- End every bullet with the id(s) of the supporting article(s) in brackets, e.g. [A2].
- Use only facts and figures that appear in the articles. No outside knowledge.

Analysed articles:
{articles}

Reply with the {count} bullets only."""


@dataclass
class NewsArticle:
    """One article as it moves through the chain; later stages fill in more fields."""

    id: str
    headline: str
    source: str
    date: str
    url: str
    summary: str
    topic: str = "other"
    sentiment: str = "Neutral"
    relevance: int = 3
    entities: list[str] = field(default_factory=list)
    figures: list[dict[str, str]] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Headline and summary together: the only text figures may come from."""
        return f"{self.headline} {self.summary}"

    def brief(self) -> dict[str, Any]:
        """Compact form passed to the classify and extract prompts."""
        return {
            "id": self.id,
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "date": self.date,
        }

    def analysed(self) -> dict[str, Any]:
        """Full analysed form passed to the summarize prompt."""
        return {
            **self.brief(),
            "topic": self.topic,
            "sentiment": self.sentiment,
            "relevance": self.relevance,
            "entities": self.entities,
            "figures": self.figures,
        }


class NewsChainAgent(Agent):
    """News analyst that runs the five-stage prompt chain instead of a free-form ReAct loop.

    Args:
        name: Agent name (its output key in ``GraphState.agents``).
        role: Role description, kept for display and traces.
        news_tool: The ``get_news_data`` tool used by the ingest stage.
        lookback_days: News window for ingest and preprocess.
        today: Override for the current date (tests); defaults to today in UTC.
        **agent_kwargs: Passed to ``Agent`` (``llm``, ``model``, ``verbose``).
    """

    def __init__(
        self,
        name: str,
        role: str,
        news_tool: Tool,
        lookback_days: int = NEWS_LOOKBACK_DAYS,
        today: Callable[[], date] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        super().__init__(name=name, role=role, tools=[news_tool], **agent_kwargs)
        self.news_tool = news_tool
        self.lookback_days = lookback_days
        self._today = today or _utc_today

    # ------------------------------------------------------------------ entry point

    def run(
        self,
        prompt: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        """Run all five stages for the ticker named in ``prompt``.

        Args:
            prompt: The graph's agent prompt (user request, context and prior findings).
            on_progress: Optional callback invoked with live thinking/action messages.

        Returns:
            An ``AgentResponse`` whose content is a cited Markdown news digest and whose
            trace records the tool call and every stage's intermediate result.
        """
        if on_progress:
            desc = self.description or self.name
            on_progress(f"Thinking: {desc}...")

        trace: list[TraceStep] = []
        llm_calls = 0
        ticker = _resolve_ticker(prompt or "")
        if ticker is None:
            return self._finish(
                prompt, "No news data available: no company ticker was identified.", trace, 0
            )

        raw = self._ingest(ticker, trace)
        articles = self._preprocess(raw, trace)
        if not articles:
            message = raw.get("error") or raw.get("message") or "no articles returned"
            content = f"No news data available from tools for {ticker} ({message})."
            return self._finish(prompt, content, trace, 0)

        llm_calls += self._classify(ticker, articles, trace)
        articles = self._drop_off_topic(articles, trace)
        llm_calls += self._extract(ticker, articles, trace)
        bullets, used_llm = self._summarize(ticker, articles, trace)
        llm_calls += used_llm

        return self._finish(
            prompt, _render_digest(ticker, articles, bullets, self.lookback_days), trace, llm_calls
        )

    # ------------------------------------------------------------------ stages

    def _ingest(self, ticker: str, trace: list[TraceStep]) -> dict[str, Any]:
        """Stage 1: fetch articles through the news tool (recorded like a normal tool call)."""
        args = {"ticker": ticker, "days": self.lookback_days}
        trace.append(
            TraceStep(_clock(), self.name, "tool_call", {"tool": _NEWS_TOOL, "arguments": args})
        )
        if getattr(self, "_on_tool_call", None) and callable(self._on_tool_call):
            self._on_tool_call(self.name, _NEWS_TOOL, args)
        observation = self.news_tool.execute(**args)
        trace.append(
            TraceStep(
                _clock(),
                self.name,
                "tool_result",
                observation,
                metadata={"tool": _NEWS_TOOL, "args": args},
            )
        )
        try:
            payload = json.loads(observation)
        except json.JSONDecodeError:
            return {"articles": [], "error": observation}
        return payload if isinstance(payload, dict) else {"articles": []}

    def _preprocess(self, payload: dict[str, Any], trace: list[TraceStep]) -> list[NewsArticle]:
        """Stage 2 (code): clean, de-duplicate, keep the lookback window, number A1..An."""
        cutoff = self._today() - timedelta(days=self.lookback_days)
        seen: set[str] = set()
        kept: list[NewsArticle] = []
        dropped = Counter[str]()

        raw_articles = payload.get("articles") or []
        # Newest first, so the cap keeps the most recent coverage.
        for raw in sorted(raw_articles, key=lambda a: str(a.get("date") or ""), reverse=True):
            headline = _clean(raw.get("headline"))
            if not headline:
                dropped["no_headline"] += 1
                continue
            key = re.sub(r"[^a-z0-9]", "", headline.lower())
            if key in seen or (raw.get("url") and raw["url"] in seen):
                dropped["duplicate"] += 1
                continue
            published = _parse_date(raw.get("date"))
            if published is not None and published < cutoff:
                dropped["older_than_window"] += 1
                continue
            seen.update({key, raw.get("url") or key})
            kept.append(
                NewsArticle(
                    id=f"A{len(kept) + 1}",
                    headline=headline,
                    source=_clean(raw.get("source")) or "unknown source",
                    date=str(raw.get("date") or "undated")[:10],
                    url=str(raw.get("url") or ""),
                    summary=_clean(raw.get("summary")),
                )
            )
            if len(kept) == _MAX_ARTICLES:
                break

        self._record(
            trace,
            "preprocess",
            {"received": len(raw_articles), "kept": len(kept), "dropped": dict(dropped)},
        )
        return kept

    def _classify(self, ticker: str, articles: list[NewsArticle], trace: list[TraceStep]) -> int:
        """Stage 3 (LLM): topic, sentiment and relevance per article. Returns LLM calls made."""
        prompt = _CLASSIFY_PROMPT.format(
            ticker=ticker,
            topics=", ".join(TOPICS),
            sentiments=", ".join(SENTIMENTS),
            articles=json.dumps([a.brief() for a in articles], indent=1),
        )
        rows = self._ask_json(prompt, "classify", trace)
        by_id = {a.id: a for a in articles}
        for row in rows:
            article = by_id.get(str(row.get("id")))
            if article is None:
                continue
            # Validate against the fixed label sets; unknown labels keep the defaults.
            topic = str(row.get("topic", "")).strip().lower()
            article.topic = topic if topic in TOPICS else "other"
            sentiment = str(row.get("sentiment", "")).strip().capitalize()
            article.sentiment = sentiment if sentiment in SENTIMENTS else "Neutral"
            article.relevance = _clamp_int(row.get("relevance"), 1, 5, default=3)
        self._record(
            trace,
            "classify",
            {a.id: f"{a.topic}/{a.sentiment}/r{a.relevance}" for a in articles},
        )
        return 1

    def _drop_off_topic(
        self, articles: list[NewsArticle], trace: list[TraceStep]
    ) -> list[NewsArticle]:
        """Remove articles the classifier rated as not about the company.

        Keyword news search also returns loosely related items (package release notes,
        articles that only mention the company in passing). They are excluded from
        extraction, sentiment counts and sources. If every article would be removed,
        all are kept so the digest is never empty. Article ids are not renumbered, so
        citations stay stable.
        """
        relevant = [a for a in articles if a.relevance >= _MIN_RELEVANCE]
        dropped = [a.id for a in articles if a.relevance < _MIN_RELEVANCE]
        self._record(trace, "filter_relevance", {"kept": len(relevant), "dropped": dropped})
        return relevant or articles

    def _extract(self, ticker: str, articles: list[NewsArticle], trace: list[TraceStep]) -> int:
        """Stage 4 (LLM + code check): entities and figures; ungrounded figures are dropped."""
        prompt = _EXTRACT_PROMPT.format(
            ticker=ticker, articles=json.dumps([a.brief() for a in articles], indent=1)
        )
        rows = self._ask_json(prompt, "extract", trace)
        by_id = {a.id: a for a in articles}
        rejected: list[str] = []
        for row in rows:
            article = by_id.get(str(row.get("id")))
            if article is None:
                continue
            article.entities = [str(e) for e in row.get("entities") or [] if str(e).strip()][:8]
            for figure in row.get("figures") or []:
                value = str((figure or {}).get("value", "")).strip()
                # Grounding check: the figure must appear verbatim in this article's text.
                if value and _normalise(value) in _normalise(article.text):
                    article.figures.append(
                        {"value": value, "context": str(figure.get("context", ""))}
                    )
                elif value:
                    rejected.append(f"{article.id}:{value}")
        self._record(
            trace,
            "extract",
            {"figures_kept": sum(len(a.figures) for a in articles), "figures_rejected": rejected},
        )
        return 1

    def _summarize(
        self, ticker: str, articles: list[NewsArticle], trace: list[TraceStep]
    ) -> tuple[list[str], int]:
        """Stage 5 (LLM + code check): cited bullets; uncited ones are replaced."""
        count = min(SUMMARY_BULLETS, len(articles))
        ranked = sorted(articles, key=lambda a: (-a.relevance, a.id))
        prompt = _SUMMARIZE_PROMPT.format(
            ticker=ticker,
            count=count,
            articles=json.dumps([a.analysed() for a in ranked], indent=1),
        )
        reply = self._ask(prompt, "summarize", trace)
        valid_ids = {a.id for a in articles}

        bullets: list[str] = []
        cited: set[str] = set()
        for line in (reply or "").splitlines():
            text, ids = _normalise_citations(_LIST_MARKER.sub("", line).strip())
            # Keep only bullets whose every citation points at a real article.
            if text and ids and ids <= valid_ids:
                bullets.append(text)
                cited |= ids
            if len(bullets) == count:
                break

        # Fill any shortfall deterministically from the most relevant uncited articles.
        filled = 0
        for article in ranked:
            if len(bullets) >= count:
                break
            if article.id not in cited:
                bullets.append(
                    f"{article.headline} ({article.source}, {article.date}) [{article.id}]"
                )
                cited.add(article.id)
                filled += 1
        self._record(
            trace, "summarize_check", {"bullets": len(bullets), "filled_from_headlines": filled}
        )
        return bullets, 1

    # ------------------------------------------------------------------ LLM helpers

    def _ask(self, prompt: str, stage: str, trace: list[TraceStep]) -> str:
        """One stateless LLM call; returns "" (and records the error) if it fails."""
        messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
        try:
            content, _, _ = self.llm.chat(messages, None)
        except Exception as exc:  # noqa: BLE001 - a failed stage falls back, it never aborts
            logger.warning("[%s] %s stage failed: %s", self.name, stage, exc)
            trace.append(TraceStep(_clock(), self.name, "error", f"{stage} stage failed: {exc}"))
            return ""
        trace.append(TraceStep(_clock(), self.name, "thought", content, metadata={"stage": stage}))
        return content or ""

    def _ask_json(self, prompt: str, stage: str, trace: list[TraceStep]) -> list[dict[str, Any]]:
        """LLM call that must return a JSON array; malformed output yields an empty list."""
        rows = _parse_json_array(self._ask(prompt, stage, trace))
        if rows is None:
            trace.append(
                TraceStep(_clock(), self.name, "error", f"{stage}: reply was not a JSON array")
            )
            return []
        return rows

    def _record(self, trace: list[TraceStep], stage: str, detail: Any) -> None:
        """Record a code stage's result in the trace (and the log when verbose)."""
        trace.append(
            TraceStep(
                _clock(),
                self.name,
                "thought",
                {"stage": stage, "result": detail},
                metadata={"stage": stage},
            )
        )
        if self.verbose:
            logger.info("[%s] %s: %s", self.name, stage, detail)

    def _finish(
        self, prompt: str | None, content: str, trace: list[TraceStep], llm_calls: int
    ) -> AgentResponse:
        trace.append(TraceStep(_clock(), self.name, "final_answer", content))
        return AgentResponse(
            agent_name=self.name,
            input_prompt=prompt or "",
            content=content,
            trace=trace,
            messages=[],
            iterations=llm_calls,
        )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _resolve_ticker(prompt: str) -> str | None:
    """Find the ticker to research: session context, then the Planner's dispatch, then text."""
    for pattern in (_ACTIVE_TICKER, _PLANNED_TICKER):
        match = pattern.search(prompt)
        if match:
            return match.group(1).replace(".", "-")
    found = extract_tickers(prompt)
    return found[0] if found else None


def _render_digest(ticker: str, articles: list[NewsArticle], bullets: list[str], days: int) -> str:
    """Assemble the final Markdown digest with sentiment counts and a numbered source list."""
    counts = Counter(a.sentiment for a in articles)
    overall = _overall_sentiment(counts)
    topics = Counter(a.topic for a in articles).most_common()
    lines = [
        f"## News digest: {ticker} (last {days} days, {len(articles)} articles)",
        "",
        f"**Overall news sentiment:** {overall} "
        f"({counts['Bullish']} bullish, {counts['Neutral']} neutral, {counts['Bearish']} bearish)",
        f"**Topics:** {', '.join(f'{t} ({n})' for t, n in topics)}",
        "",
        "**Key developments**",
        *[f"- {b}" for b in bullets],
        "",
        "**Sources**",
        *[
            f"- [{a.id}] {a.headline} - {a.source}, {a.date}" + (f" ({a.url})" if a.url else "")
            for a in articles
        ],
    ]
    return "\n".join(lines)


def _normalise_citations(text: str) -> tuple[str, set[str]]:
    """Rewrite every citation group in ``text`` as ``[A1][A2]`` and return the ids cited."""
    ids: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        group = _ARTICLE_ID.findall(match.group(1))
        ids.update(group)
        return "".join(f"[{article_id}]" for article_id in group)

    return _CITATION_GROUP.sub(replace, text), ids


def _overall_sentiment(counts: Counter[str]) -> str:
    """Majority sentiment; a bullish/bearish tie is "Mixed", any other tie is "Neutral"."""
    top = max(counts[s] for s in SENTIMENTS)
    leaders = {s for s in SENTIMENTS if counts[s] == top}
    if len(leaders) == 1:
        return leaders.pop()
    return "Mixed" if {"Bullish", "Bearish"} <= leaders else "Neutral"


def _parse_json_array(text: str) -> list[dict[str, Any]] | None:
    """Parse a JSON array from an LLM reply, tolerating ```json fences and extra prose."""
    if not text:
        return None
    candidates = [*_JSON_BLOCK.findall(text), text]
    for candidate in candidates:
        start, end = candidate.find("["), candidate.rfind("]")
        if start == -1 or end <= start:
            continue
        try:
            data = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
    return None


def _clean(value: Any) -> str:
    """Strip HTML tags/entities and collapse whitespace."""
    return " ".join(html.unescape(_TAG.sub(" ", str(value or ""))).split())


def _normalise(text: str) -> str:
    """Lower-case and drop spaces/commas so '$1,200 M' matches '$1200m'."""
    return re.sub(r"[\s,]", "", text.lower())


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _utc_today() -> date:
    """Today's date in UTC (module-level so tests can pin it)."""
    return datetime.now(timezone.utc).date()


def _clock() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
