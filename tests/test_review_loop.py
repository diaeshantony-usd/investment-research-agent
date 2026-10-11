"""
test_review_loop.py
===================
Offline tests for the Critic -> Refiner review loop (``workflows/review_loop.py``):

- figure parsing and number grounding against the stored tool results
- required-section check
- the Critic's decision (code checks + scripted LLM judge)
- Refiner node, stopping rules and release
- the loop wired into the full research graph

Every LLM is scripted, so the tests need no network access.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import pytest

from research_agent.agent_framework import Agent, GraphState, store_tool_result, tool
from research_agent.interfaces import LLM
from research_agent.workflows import review_loop
from research_agent.workflows.review_loop import (
    MAX_REFINEMENT_ROUNDS,
    Critic,
    check_grounding,
    check_sections,
    evidence_numbers,
    make_review_nodes,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Tool results as the graph stores them: nested dicts, JSON strings and plain numbers.
TOOLS: dict[str, Any] = {
    "EarningsAnalyst": {
        "get_earnings_data": {
            "ticker": "NVDA",
            "revenue": "$38.2 Billion",
            "operating_income": "$22.8 Billion",
            "gross_margin": "75.4%",
            "forward_pe": "34.2",
        }
    },
    "MarketAnalyst": {
        "get_market_technicals": json.dumps(
            {"current_price": "$138.50", "50_day_sma": "$126.80", "beta": 1.68}
        )
    },
}

GOOD_BRIEF = """# Research Brief: NVDA

## Executive Summary
- Bullish: revenue of $38.2B with a 75.4% gross margin [A1].

## 1. Fundamental & Earnings Performance
- Operating income $22.8B (\u224860% of revenue); forward P/E 34.2x.

## 2. Technical Profile & Market Momentum
- Price $138.50, \u22489.2% above the 50\u2011day SMA of $126.80.
- Beta 1.68 (~68% more volatile than the market).

## 3. Catalysts, Macro Context & Risk Analysis
- Export-control review reported on 2026-09-28 [A2]; see the 10-Q for Q2 2026.

## 4. Investment Stance & Outlook
- Bullish over the next 12 months, with 3 key risks to monitor.
"""

INVENTED_BRIEF = GOOD_BRIEF.replace(
    "- Bullish over the next 12 months",
    "- Price target $155 on ~2\u00d7 performance and +30% FY2027 growth. Bullish over the "
    "next 12 months",
)

UNAVAILABLE_NOTICE = "# Research Brief: ZZZZ - Data Unavailable\n\nI do not have data."


def judge_reply(score: float, issues: Iterable[dict[str, str]] = ()) -> str:
    """A judge reply giving every judged dimension the same score."""
    scores = dict.fromkeys(("coverage", "balance", "clarity", "actionability"), score)
    return json.dumps({"scores": scores, "issues": list(issues)})


class ScriptedLLM(LLM):
    """Returns queued replies in order and records every prompt it receives."""

    def __init__(self, replies: Iterable[tuple[str, list[dict[str, Any]]] | str] = ()):
        self.model = "scripted"
        self.replies = [r if isinstance(r, tuple) else (r, []) for r in replies]
        self.prompts: list[str] = []

    def chat(
        self, messages: list[dict[str, Any]], tools: dict[str, Any] | None = None
    ) -> tuple[str, list[dict[str, Any]], Any]:
        self.prompts.append(messages[-1].get("content", "") if messages else "")
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        text, calls = self.replies.pop(0)
        return text, calls, None


class FailingLLM(LLM):
    """An LLM whose every call fails, standing in for an unreachable endpoint."""

    model = "failing"

    def chat(self, messages: list[dict[str, Any]], tools: dict[str, Any] | None = None) -> Any:
        raise ConnectionError("endpoint unreachable")


def refiner_with(llm: LLM, tools: list[Any] | None = None) -> Agent:
    return Agent(name="BriefRefiner", role="Reviser", tools=tools or [], llm=llm)


def new_state(brief: str = INVENTED_BRIEF) -> GraphState:
    state = GraphState(question="Analyse NVDA", tools=json.loads(json.dumps(TOOLS)))
    state.agents["DraftWriter"] = brief
    return state


# ---------------------------------------------------------------------------
# Figure parsing and grounding
# ---------------------------------------------------------------------------


class TestGrounding:
    def test_figures_take_their_unit_and_rounding(self):
        (figure,) = review_loop._parse_figures("Revenue was $96.2B.")
        assert figure.kind == "money"
        assert figure.value == pytest.approx(96.2e9)
        assert figure.tolerance == pytest.approx(0.05e9)

    def test_labels_dates_and_counts_are_not_figures(self):
        text = (
            "## 2. Technicals\n1. The 50-day and 200 d SMA, 52\u2011wk high, 14-day RSI "
            "below 70, on 2026-09-28 [A3, A5] per the 10-K for Q2 2026 and FY2025; score "
            "8/10; https://example.com/a/123; 3 risks."
        )
        assert review_loop._parse_figures(text) == []

    def test_typographic_spaces_and_dashes_are_normalised(self):
        (figure,) = review_loop._parse_figures("Gross margin 75.4\u00a0%")
        assert (figure.kind, figure.value) == ("percent", 75.4)

    def test_approximation_markers_are_detected(self):
        figures = review_loop._parse_figures("\u224860% and about 12% but 30%")
        assert [f.approximate for f in figures] == [True, True, False]

    def test_evidence_reads_nested_json_strings_and_numbers(self):
        evidence = evidence_numbers(TOOLS)
        values = {value for value, _ in evidence.figures}
        assert {38.2e9, 138.5, 126.8, 1.68, 75.4, 34.2} <= values

    def test_grounded_brief_has_no_unverified_figures(self):
        report = check_grounding(GOOD_BRIEF, evidence_numbers(TOOLS))
        assert report.unverified == []
        assert report.checked >= 8
        assert report.score == 10.0

    def test_invented_figures_are_flagged(self):
        report = check_grounding(INVENTED_BRIEF, evidence_numbers(TOOLS))
        assert report.unverified == ["$155", "2\u00d7", "30%"]
        assert report.score < 10.0

    def test_rounding_cannot_stretch_a_small_number(self):
        # "2x" would be within half a unit of the 1.68 beta, but it is not a rounding of it.
        report = check_grounding("Speed-up of 2x.", evidence_numbers(TOOLS))
        assert report.unverified == ["2x"]

    def test_calculated_percentage_needs_an_input_on_the_same_line(self):
        evidence = evidence_numbers(TOOLS)
        assert check_grounding("Operating margin \u224860%.", evidence).unverified == ["60%"]
        assert check_grounding("Operating income $22.8B, \u224860%.", evidence).unverified == []

    def test_filing_tables_in_millions_support_scaled_figures(self):
        evidence = evidence_numbers({"SEC": {"passage": "(In millions) Revenue $ 88,299"}})
        assert check_grounding("Compute revenue $88,299 M ($88.3B).", evidence).unverified == []
        assert check_grounding("Compute revenue $88,299.", evidence).unverified == []

    def test_gap_between_retrieved_prices_is_verified(self):
        evidence = evidence_numbers(TOOLS)
        assert check_grounding("50-day SMA $126.80 (price $11.70 above)", evidence).score == 10
        assert check_grounding("50-day SMA $126.80 (price $15 above)", evidence).unverified == [
            "$15"
        ]

    def test_calculated_percentages_from_filing_tables(self):
        # Found live: "Graphics $7.922B, +46% YoY" against a 10-Q table in millions, and
        # "5.8% below the 52-week high".
        evidence = evidence_numbers(
            {
                "SEC": "(In millions) Revenue 2026 $ 7,922 Revenue 2025 $ 5,412",
                "price": 229.28,
                "high": 243.37,
            }
        )
        assert check_grounding("Graphics $7.922B (YoY \u2248+46%)", evidence).unverified == []
        assert check_grounding("Graphics $7.922B (YoY +60%)", evidence).unverified == ["60%"]
        assert check_grounding("Price $229.28, \u22485.8% below $243.37", evidence).score == 10

    def test_unverified_figures_are_located_by_section(self):
        report = check_grounding(INVENTED_BRIEF, evidence_numbers(TOOLS))
        assert set(report.sections.values()) == {"4. Investment Stance & Outlook"}

    def test_a_brief_without_figures_scores_full_grounding(self):
        assert check_grounding("No figures here.", evidence_numbers({})).score == 10.0

    def test_missing_sections_are_reported(self):
        assert check_sections(GOOD_BRIEF) == []
        brief = GOOD_BRIEF.replace("## 2. Technical Profile & Market Momentum", "Prices:")
        assert check_sections(brief) == ["Technicals"]


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


class TestCritic:
    def test_data_unavailable_notice_passes_without_an_llm_call(self):
        report = Critic(FailingLLM()).review(UNAVAILABLE_NOTICE, {}, "q", 0)
        assert report.passed
        assert report.weighted == 10.0
        assert not report.judge_used

    def test_unavailable_section_in_a_full_brief_is_still_reviewed(self):
        # Found live: "### Macro Environment (data unavailable)" must not skip the review.
        brief = INVENTED_BRIEF + "\n### Macro Environment (data unavailable)\n- n/a\n"
        report = Critic(ScriptedLLM([judge_reply(10)])).review(brief, TOOLS, "q", 0)
        assert report.judge_used
        assert not report.passed

    def test_grounded_brief_with_good_scores_passes(self):
        report = Critic(ScriptedLLM([judge_reply(9)])).review(GOOD_BRIEF, TOOLS, "q", 0)
        assert report.passed
        assert report.scores["grounding"] == 10.0
        assert report.weighted == pytest.approx(9.3)
        assert report.judge_used

    def test_unverified_figures_block_a_pass_whatever_the_judge_says(self):
        report = Critic(ScriptedLLM([judge_reply(10)])).review(INVENTED_BRIEF, TOOLS, "q", 0)
        assert not report.passed
        assert report.blockers == 1  # one grouped grounding issue, not one per figure
        assert report.issues[0].section == "4. Investment Stance & Outlook"
        assert "'$155'" in report.issues[0].problem

    def test_code_check_results_are_given_to_the_judge(self):
        llm = ScriptedLLM([judge_reply(8)])
        Critic(llm).review(INVENTED_BRIEF, TOOLS, "q", 0)
        assert "$155" in llm.prompts[0]

    def test_judge_disputing_retrieved_data_is_ignored(self):
        # Found live: the judge called a real $38.2B-style figure "far too large" from its
        # own outdated knowledge, and the Refiner deleted correct data.
        disputes = {
            "section": "Fundamentals",
            "problem": "Revenue of $38.2 B is far above the company's actual value",
            "fix": "correct it",
            "severity": "major",
        }
        style = {
            "section": "Fundamentals",
            "problem": "Revenue $38.2 B has no citation",
            "fix": "cite it",
            "severity": "minor",
        }
        llm = ScriptedLLM([judge_reply(9, [disputes, style])])
        report = Critic(llm).review(GOOD_BRIEF, TOOLS, "q", 0)
        assert [i.problem for i in report.issues] == [style["problem"]]
        assert "38.2 Billion" in llm.prompts[0]  # the judge sees the retrieved data
        assert "Today is" in llm.prompts[0]

    def test_judge_issues_are_sorted_after_blockers(self):
        issue = {"section": "Stance", "problem": "vague", "fix": "commit", "severity": "major"}
        report = Critic(ScriptedLLM([judge_reply(9, [issue])])).review(
            INVENTED_BRIEF, TOOLS, "q", 0
        )
        assert [i.severity for i in report.issues] == ["blocker", "major"]

    @pytest.mark.parametrize("llm", [ScriptedLLM(["not json at all"]), FailingLLM()])
    def test_unusable_judge_falls_back_to_neutral_scores(self, llm):
        report = Critic(llm).review(GOOD_BRIEF, TOOLS, "q", 0)
        assert not report.judge_used
        assert report.scores["coverage"] == review_loop._FALLBACK_SCORE
        assert not report.passed  # 0.3 * 10 + 0.7 * 6 = 7.2 < 8

    def test_report_markdown_lists_scores_and_issues(self):
        report = Critic(ScriptedLLM([judge_reply(7)])).review(INVENTED_BRIEF, TOOLS, "q", 2)
        text = report.markdown()
        assert "STATUS: REVISE" in text
        assert "review round 2" in text
        assert "| Grounding | 30% |" in text
        assert "[blocker] 4. Investment Stance & Outlook: figures not found" in text


# ---------------------------------------------------------------------------
# Refiner, stopping rules and release
# ---------------------------------------------------------------------------


def run_loop(state: GraphState, judge: LLM, refiner: Agent, max_steps: int = 20) -> list[str]:
    """Drive the review nodes the way the graph does; return the nodes visited."""
    critic_node, refiner_node, release_node, route = make_review_nodes(judge, refiner)
    visited = []
    node = "critic"
    for _ in range(max_steps):
        visited.append(node)
        if node == "critic":
            critic_node(state)
            node = route(state)
        elif node == "refine":
            refiner_node(state)
            node = "critic"
        else:
            release_node(state)
            return visited
    raise AssertionError("review loop did not terminate")


class TestReviewLoop:
    def test_split_change_log(self):
        brief, log = review_loop._split_change_log("# Brief\ntext\n### Change Log\n- a\n* b\n")
        assert brief == "# Brief\ntext"
        assert log == ["a", "b"]
        assert review_loop._split_change_log("only a brief") == ("only a brief", [])

    def test_passing_draft_is_released_unchanged(self):
        state = new_state(GOOD_BRIEF)
        visited = run_loop(state, ScriptedLLM([judge_reply(9)]), refiner_with(ScriptedLLM()))
        assert visited == ["critic", "release"]
        assert state.output == GOOD_BRIEF
        assert "Reviewer notes" not in state.output

    def test_invented_figures_are_refined_until_the_brief_passes(self):
        judge = ScriptedLLM([judge_reply(9), judge_reply(9)])
        refiner_llm = ScriptedLLM([GOOD_BRIEF + "\n### Change Log\n- removed invented figures"])
        state = new_state()

        visited = run_loop(state, judge, refiner_with(refiner_llm))

        assert visited == ["critic", "refine", "critic", "release"]
        assert state.output == GOOD_BRIEF.strip()
        assert state.refine_count == 1
        assert state.review["change_logs"] == [["removed invented figures"]]
        assert "$155" in refiner_llm.prompts[0]  # the refiner was told what to fix
        assert [r["passed"] for r in state.review["history"]] == [False, True]
        assert "STATUS: PASS" in state.agents["ResearchCritic"]

    def test_loop_stops_after_the_maximum_number_of_rounds(self):
        # Scores improve by 1 each round (>= MIN_IMPROVEMENT) but the brief never passes.
        judge = ScriptedLLM([judge_reply(5 + n) for n in range(MAX_REFINEMENT_ROUNDS + 1)])
        refiner_llm = ScriptedLLM([INVENTED_BRIEF] * MAX_REFINEMENT_ROUNDS)
        state = new_state()

        visited = run_loop(state, judge, refiner_with(refiner_llm))

        assert visited.count("refine") == MAX_REFINEMENT_ROUNDS
        assert state.refine_count == MAX_REFINEMENT_ROUNDS
        assert visited[-1] == "release"

    def test_loop_stops_when_a_round_does_not_improve_enough(self):
        judge = ScriptedLLM([judge_reply(7), judge_reply(7)])
        state = new_state()

        visited = run_loop(state, judge, refiner_with(ScriptedLLM([INVENTED_BRIEF])))

        assert visited == ["critic", "refine", "critic", "release"]

    def test_unpassed_brief_is_released_with_reviewer_notes(self):
        judge = ScriptedLLM([judge_reply(7), judge_reply(7)])
        state = new_state()

        run_loop(state, judge, refiner_with(ScriptedLLM([INVENTED_BRIEF])))

        assert "### Reviewer notes" in state.output
        assert "did not pass review" in state.output
        assert "figures not found in the retrieved data" in state.output

    def test_best_version_is_released(self):
        # Round 1 scores higher than round 0, so its brief is the one released.
        judge = ScriptedLLM([judge_reply(5), judge_reply(7), judge_reply(7)])
        better = INVENTED_BRIEF.replace("## Executive Summary", "## Executive Summary\nv2")
        refiner_llm = ScriptedLLM([better, better])
        state = new_state()

        run_loop(state, judge, refiner_with(refiner_llm))

        assert state.output.startswith(better.strip())

    def test_empty_refiner_reply_keeps_the_previous_brief(self):
        judge = ScriptedLLM([judge_reply(7), judge_reply(7)])
        state = new_state()

        run_loop(state, judge, refiner_with(ScriptedLLM([""])))

        assert state.review["history"][1]["brief"] == INVENTED_BRIEF
        assert state.review["change_logs"] == [["Refiner returned no usable revision"]]

    def test_data_fetched_by_the_refiner_counts_as_evidence(self):
        @tool(name="get_cash_flow", description="Free cash flow for a ticker")
        def get_cash_flow(ticker: str) -> str:
            return json.dumps({"ticker": ticker, "free_cash_flow": "$14.5 Billion"})

        with_fcf = GOOD_BRIEF.replace(
            "forward P/E 34.2x.", "forward P/E 34.2x; free cash flow $14.5B."
        )
        refiner_llm = ScriptedLLM(
            [
                ("Fetching cash flow", [{"name": "get_cash_flow", "args": {"ticker": "NVDA"}}]),
                with_fcf,
            ]
        )
        judge = ScriptedLLM([judge_reply(9), judge_reply(9)])
        state = new_state()

        run_loop(state, judge, refiner_with(refiner_llm, tools=[get_cash_flow]))

        assert state.tools["BriefRefiner"]["get_cash_flow"]["free_cash_flow"] == "$14.5 Billion"
        assert state.review["history"][-1]["passed"]
        assert "$14.5B" in state.output


def test_repeated_tool_calls_are_all_kept_as_evidence():
    # Found live: a 10-Q search overwrote the earlier 10-K search, losing the segment table.
    store: dict[str, Any] = {}
    assert store_tool_result(store, "search_sec_filings", {"form": "10-K"}) == (
        "search_sec_filings"
    )
    assert store_tool_result(store, "search_sec_filings", {"text": "$88,299 million"}) == (
        "search_sec_filings#2"
    )
    report = check_grounding("Compute & Networking $88.299B.", evidence_numbers({"A": store}))
    assert report.unverified == []


# ---------------------------------------------------------------------------
# Graph integration
# ---------------------------------------------------------------------------


def test_research_graph_runs_the_review_loop(monkeypatch):
    """Planner -> specialists -> writer -> critic -> refiner -> critic -> release."""
    from research_agent.agent_framework import GraphMemory
    from research_agent.tools.earnings import get_earnings_data
    from research_agent.tools.news import get_news_data
    from research_agent.workflows.graph import build_research_workflow

    earnings = TOOLS["EarningsAnalyst"]["get_earnings_data"]
    monkeypatch.setattr(get_earnings_data, "func", lambda ticker: json.dumps(earnings))
    monkeypatch.setattr(
        get_news_data,
        "func",
        lambda ticker, days=7: json.dumps({"ticker": ticker, "articles": []}),
    )
    grounded = GOOD_BRIEF.split("## 2.", maxsplit=1)[0] + (
        "## 2. Technical Profile & Market Momentum\n- No technical data.\n\n"
        "## 3. Catalysts & Risks\n- None reported.\n\n## 4. Investment Stance\n- Bullish.\n"
    )
    invented = grounded.replace("- Bullish.\n", "- Bullish, target $155.\n")
    llm = ScriptedLLM(
        [
            "RESEARCH_REQUIRED: NVDA - evaluate",  # Planner
            ("Fetching", [{"name": "get_earnings_data", "args": {"ticker": "NVDA"}}]),
            "Revenue $38.2B.",  # EarningsAnalyst
            "No technicals.",  # MarketAnalyst (NewsAnalyst makes no call without articles)
            invented,  # DraftWriter
            judge_reply(9),  # Critic, round 0: blocked by "$155"
            grounded + "\n### Change Log\n- removed the price target",  # Refiner
            judge_reply(9),  # Critic, round 1: passes
        ]
    )

    state = build_research_workflow(llm=llm, memory=GraphMemory()).invoke(
        {"question": "Analyse NVDA"}
    )

    nodes = [step["node"] for step in state.trajectory]
    assert nodes[-4:] == ["critic", "refiner", "critic", "release"]
    assert state.output == grounded.strip()
    assert state.to_dict()["review"]["history"][-1]["passed"]
    assert not llm.replies  # every scripted reply was used
