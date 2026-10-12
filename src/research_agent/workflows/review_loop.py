"""
workflows/review_loop.py
========================
The Critic -> Refiner review loop (workflow pattern: **evaluator-optimizer**).

The Draft writer's brief is reviewed before it is released. Unlike an "LLM grades LLM"
loop, the Critic checks the brief against the raw tool data the specialists actually
retrieved (``GraphState.tools``), so invented figures cannot pass review.

**Critic** (three stages)

1. Code checks, no LLM: every number in the brief must match a number in the stored
   tool results (within the rounding the brief uses: "75%" matches 74.98), or be a
   percentage calculated from a retrieved figure on the same line ("$22.8B, ~60% of
   revenue"); and the required sections must be present.
2. LLM judge: scores five dimensions 0-10, given the code-check results as facts. The
   grounding score is computed by code, not judged.
3. Decision: any unverified number sends the brief back. Otherwise it passes when the
   weighted score is at least 8.

**Refiner** fixes at most five issues per round, blockers first, rewrites only the
flagged sections, never writes a number that is not in the tool results, and returns a
change log. It keeps its data tools, so it can fetch a missing figure instead of guessing.

**Stopping rules**: the loop ends when the brief passes, after three refinement rounds,
or when a round improves the weighted score by less than 0.5. If it ends without a pass,
the best-scoring version is released with the remaining issues attached as reviewer
notes.

The loop runs as three function nodes (``critic``, ``refiner``, ``release``) in the
graph built by ``workflows/graph.py``; its state lives in ``GraphState.review``.

Owner: Diaesh Antony
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Final

from research_agent.agent_framework import Agent, GraphState, store_tool_result
from research_agent.config import logger
from research_agent.interfaces.llm import LLM

__all__ = [
    "DIMENSION_WEIGHTS",
    "MAX_REFINEMENT_ROUNDS",
    "MIN_IMPROVEMENT",
    "PASS_SCORE",
    "Evidence",
    "Issue",
    "ReviewReport",
    "check_grounding",
    "check_sections",
    "evidence_numbers",
    "make_review_nodes",
]

# ---------------------------------------------------------------------------
# Policy (from the design doc)
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: Final = {
    "grounding": 0.30,
    "coverage": 0.20,
    "balance": 0.20,
    "clarity": 0.15,
    "actionability": 0.15,
}
PASS_SCORE: Final = 8.0
MAX_REFINEMENT_ROUNDS: Final = 3
MIN_IMPROVEMENT: Final = 0.5
MAX_ISSUES_PER_ROUND: Final = 5
# Score assumed for a dimension when the LLM judge is unavailable or omits it.
_FALLBACK_SCORE: Final = 6.0

# Required sections, each matched by any of its keywords in a Markdown heading.
REQUIRED_SECTIONS: Final = {
    "Executive Summary": ("summary",),
    "Fundamentals": ("fundamental", "earnings"),
    "Technicals": ("technical", "momentum", "price"),
    "Catalysts & Risks": ("catalyst", "risk"),
    "Stance": ("stance", "outlook", "recommendation"),
}

# Markers of the Draft writer's "no data for this ticker" notice, which needs no review.
_UNAVAILABLE_MARKERS: Final = ("DATA UNAVAILABLE", "COVERAGE UNAVAILABLE")

# Evidence passed to the LLM prompts is capped to keep them small.
_EVIDENCE_PROMPT_CHARS: Final = 9000

# ---------------------------------------------------------------------------
# Number grounding (code check)
# ---------------------------------------------------------------------------

_MULTIPLIERS: Final = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "tn": 1e12,
    "trillion": 1e12,
}
# A number with optional currency sign, thousands separators, decimals and unit.
_NUMBER: Final = re.compile(
    r"(?<![\w.])(?P<cur>[$€£])?\s?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
    r"\s?(?P<unit>%|x|\u00d7|trillion|billion|million|thousand|tn|bn|mn|[kmbt])?(?![\w])",
    re.IGNORECASE,
)
# Typographic dashes and spaces that models emit ("52\u2011week", "75.4\u00a0%"), mapped to
# their ASCII forms so the patterns below stay simple.
_TYPOGRAPHY: Final = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u202f": " ",
    }
)
# Text that contains digits but is not a reported figure.
_NOT_FIGURES: Final = (
    re.compile(r"https?://\S+"),  # URLs
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # ISO dates
    re.compile(r"\[[A-Z]?\d+(?:\s*,\s*[A-Z]?\d+)*\]"),  # citations like [A3] or [A3, A5]
    re.compile(r"^\s*#+\s*\d+(?:\.\d+)*[.)]?", re.MULTILINE),  # numbered headings "## 3.1"
    re.compile(r"^\s*\d+[.)]\s", re.MULTILINE),  # numbered list items
    # Look-back windows and indicator settings: "50-day", "52-wk", "50 d SMA", "14-day RSI".
    re.compile(r"\b\d+\s?-?\s?(?:d|day|wk|week|mo|month|yr|year|quarter)s?\b", re.IGNORECASE),
    re.compile(r"\b\d+-?[QK]\b"),  # filing forms 10-Q, 10-K
    re.compile(r"\b(?:Q[1-4]|FY|H[12])\s?\d{2,4}\b", re.IGNORECASE),  # periods Q2 2026
    re.compile(r"\b\d+(?:\.\d+)?\s?/\s?10\b"),  # scores "8/10"
)
# Words in front of a figure that mark it as an approximation or a calculation.
_APPROXIMATE: Final = re.compile(
    r"(?:\u2248|~|\babout|\baround|\broughly|\bapprox(?:\.|imately)?)\s*[+-]?\s*$",
    re.IGNORECASE,
)
# A derived percentage is only computed from two figures of similar size, so that
# "revenue / share price" style coincidences cannot verify an invented number.
_DERIVED_MAX_RATIO: Final = 10.0
# Absolute values below this are treated as betas or ratios: "beta 1.68" supports
# "68% more volatile than the market".
_MAX_RATIO_VALUE: Final = 5.0


@dataclass(frozen=True)
class _Figure:
    """A number parsed from text, with its rounding tolerance and kind."""

    text: str
    value: float
    tolerance: float
    kind: str  # "money", "percent", "multiple" or "plain"
    approximate: bool = False
    multiplier: float = 1.0  # 1e9 for "$96.2B"


def _kind(currency: str | None, unit: str) -> str:
    if unit == "%":
        return "percent"
    if unit in ("x", "\u00d7"):
        return "multiple"
    if currency or unit in _MULTIPLIERS:
        return "money"
    return "plain"


def _parse_figures(text: str) -> list[_Figure]:
    """Every reported-looking number in ``text`` with its value and rounding tolerance.

    Plain integers without a currency sign, unit or decimals are skipped: in a brief they
    are labels, counts and thresholds ("RSI below 70", "3 risks"), not reported data.
    """
    cleaned = text.translate(_TYPOGRAPHY)
    for pattern in _NOT_FIGURES:
        cleaned = pattern.sub(" ", cleaned)

    figures = []
    for match in _NUMBER.finditer(cleaned):
        digits = match.group("int").replace(",", "")
        decimals = match.group("dec") or ""
        unit = (match.group("unit") or "").lower()
        currency = match.group("cur")
        if not decimals and not unit and not currency:
            continue
        value = float(digits + decimals)
        multiplier = _MULTIPLIERS.get(unit, 1.0)
        places = len(decimals) - 1 if decimals else 0
        # Half a unit in the last shown digit: "$96.2B" covers 96.15B-96.25B.
        tolerance = 0.5 * 10 ** (-places) * multiplier
        approximate = _APPROXIMATE.search(cleaned[max(0, match.start() - 16) : match.start()])
        figures.append(
            _Figure(
                match.group(0).strip(),
                value * multiplier,
                tolerance,
                _kind(currency, unit),
                approximate is not None,
                multiplier,
            )
        )
    return figures


def _walk(value: Any) -> Iterable[Any]:
    """Yield every scalar inside nested dicts/lists (parsing JSON strings on the way)."""
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    elif isinstance(value, str):
        parsed = _maybe_json(value)
        if parsed is None:
            yield value
        else:
            yield from _walk(parsed)
    else:
        yield value


def _maybe_json(text: str) -> Any:
    """Parse ``text`` as a JSON object/array if it looks like one, else return None."""
    stripped = text.strip()
    if not stripped.startswith(("[", "{")):
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


# Largest share of a figure that its rounding may account for.
_MAX_ROUNDING: Final = 0.05
# Units that filings state their tables in ("In thousands", "In millions").
_FILING_UNITS: Final = (1e3, 1e6)
# Figure kinds that can be compared with each other. Tool results often hold amounts as
# bare numbers (38200000000.0) that the brief writes as "$38.2B".
_COMPATIBLE_KINDS: Final = {
    "money": ("money", "plain"),
    "plain": ("money", "plain"),
    "percent": ("percent", "plain"),
    "multiple": ("multiple", "plain"),
}


@dataclass(frozen=True)
class Evidence:
    """The numbers in the stored tool results, each with its kind."""

    figures: tuple[tuple[float, str], ...]

    def matches(self, figure: _Figure) -> float | None:
        """The retrieved value ``figure`` reports, or None when there is none.

        A reported figure may differ from the data by its own rounding, or 0.5%, but
        never by more than 5%: "~2x" is not a rounding of a 1.68 beta.
        """
        target = abs(figure.value)
        rounding = min(figure.tolerance, _MAX_ROUNDING * target)
        kinds = _COMPATIBLE_KINDS[figure.kind]
        candidates = [
            scaled
            for value, kind in self.figures
            if kind in kinds
            for scaled in _scalings(value, figure)
            if abs(target - abs(scaled)) <= max(rounding, 0.005 * abs(scaled))
        ]
        return min(candidates, key=lambda v: abs(target - abs(v)), default=None)

    def amounts(self) -> set[float]:
        """Absolute values of the money and plain figures (operands for percentages)."""
        return {abs(value) for value, kind in self.figures if value and kind != "percent"}


def _scalings(value: float, figure: _Figure) -> tuple[float, ...]:
    """``value`` as stated, plus its thousands/millions reading for a scaled figure.

    SEC filings state tables "(In millions)", so "$ 88,299" in a filing supports
    "$88,299M" or "$88.3B" in the brief. Only applied when the brief gives the figure
    a scale (M, B, ...) and the filing number has at least three digits.
    """
    if figure.kind != "money" or figure.multiplier < _FILING_UNITS[0] or abs(value) < 100:
        return (value,)
    return (value, *(value * unit for unit in _FILING_UNITS if unit <= figure.multiplier))


def evidence_numbers(tools: Mapping[str, Any]) -> Evidence:
    """Collect the numbers in the stored tool results (``GraphState.tools``).

    Numeric fields are taken as-is; numbers inside text fields ("$96.22B", headlines,
    filing passages) are parsed the same way as the brief.
    """
    found: list[tuple[float, str]] = []
    for value in _walk(tools):
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            found.append((float(value), "plain"))
        elif isinstance(value, str):
            found.extend((f.value, f.kind) for f in _parse_figures(value))
    return Evidence(tuple(dict.fromkeys(found)))


def _derived_percentages(anchor: float, amounts: Iterable[float]) -> set[float]:
    """Percentages calculable from ``anchor`` and one other retrieved amount.

    Covers "X is N% of Y" and "X is N% above/below Y" in both directions, and a beta or
    ratio written as a percentage ("beta 1.68" -> "68% more volatile"). Only amounts of
    similar size are paired, so an unrelated pair cannot verify an invented number.
    """
    derived = set()
    a = abs(anchor)
    if 0 < a < _MAX_RATIO_VALUE:
        derived.add(100 * (a - 1))
    for b in amounts:
        if a and b != a and 1 / _DERIVED_MAX_RATIO <= a / b <= _DERIVED_MAX_RATIO:
            # Changes are compared as magnitudes: "5.8% below" is a -5.8% change.
            derived.update(
                (100 * a / b, 100 * b / a, abs(100 * (a / b - 1)), abs(100 * (b / a - 1)))
            )
    return derived


def _gaps(anchors: Iterable[float], amounts: Iterable[float]) -> set[float]:
    """Differences between each anchor and every retrieved amount of similar size."""
    gaps = set()
    for anchor in anchors:
        a = abs(anchor)
        for b in amounts:
            if a and b != a and 1 / _DERIVED_MAX_RATIO <= a / b <= _DERIVED_MAX_RATIO:
                gaps.add(abs(a - b))
    return gaps


@dataclass
class GroundingReport:
    """Result of the number-grounding code check."""

    checked: int = 0
    unverified: list[str] = field(default_factory=list)
    # Section heading each unverified figure appears under, so the Refiner knows
    # which sections to rewrite.
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Grounding score 0-10: the share of figures found in the tool data."""
        if not self.checked:
            return 10.0
        return round(10 * (self.checked - len(self.unverified)) / self.checked, 1)


def _near(figure: _Figure, candidates: Iterable[float]) -> bool:
    """True when a calculated percentage matches ``figure``.

    Allows one unit in the figure's last digit (inputs are rounded before dividing),
    widened to 1% when the brief marks the figure as approximate.
    """
    slack = 2 * figure.tolerance
    if figure.approximate:
        slack = max(slack, 0.01 * abs(figure.value))
    return any(abs(figure.value - c) <= slack for c in candidates)


def check_grounding(brief: str, evidence: Evidence) -> GroundingReport:
    """Check that every figure in ``brief`` can be traced to the tool evidence.

    A figure is verified when it matches a retrieved number. A percentage is also
    verified when it can be calculated from a retrieved figure written on the same line
    ("Operating income $22.8B, about 60% of revenue") and one other retrieved amount.
    A calculated percentage repeated elsewhere must again sit next to its input, so a
    figure verified in one context cannot vouch for an invented claim in another.

    Args:
        brief: The Markdown research brief.
        evidence: Numbers from the stored tool results (see ``evidence_numbers``).

    Returns:
        How many figures were checked and which could not be traced to the data.
    """
    report = GroundingReport()
    amounts = evidence.amounts()
    section = "Introduction"
    for line in brief.splitlines():
        if line.lstrip().startswith("#"):
            section = line.lstrip("# ").strip() or section
        figures = _parse_figures(line)
        anchors = [
            value
            for figure in figures
            if figure.kind in ("money", "plain") and (value := evidence.matches(figure)) is not None
        ]
        # "$7.922B" can be compared with filing figures stated in millions ("$ 5,412").
        anchors += [a / unit for a in anchors for unit in _FILING_UNITS if abs(a) / unit >= 100]
        for figure in figures:
            report.checked += 1
            if evidence.matches(figure) is not None:
                continue
            if figure.kind == "percent":
                derived = set().union(*(_derived_percentages(a, amounts) for a in anchors))
                if _near(figure, derived):
                    continue
            # A gap between a figure on the line and another retrieved amount of similar
            # size: "50-day SMA $221.94 (price $7.34 above)".
            if figure.kind == "money" and _near(figure, _gaps(anchors, amounts)):
                continue
            if figure.text not in report.unverified:
                report.unverified.append(figure.text)
                report.sections[figure.text] = section
    return report


def check_sections(brief: str) -> list[str]:
    """Return the names of required sections missing from the brief's headings."""
    headings = " ".join(
        line.lower() for line in brief.splitlines() if line.lstrip().startswith("#")
    )
    return [
        name
        for name, keywords in REQUIRED_SECTIONS.items()
        if not any(keyword in headings for keyword in keywords)
    ]


def _is_unavailable_notice(brief: str) -> bool:
    """True when the brief is the writer's "no data" notice.

    Only the title is checked: a full brief may well say "macro data unavailable" in a
    section heading, and must still be reviewed.
    """
    title = next((line for line in brief.splitlines() if line.strip()), "").upper()
    return title.lstrip().startswith("#") and any(m in title for m in _UNAVAILABLE_MARKERS)


# ---------------------------------------------------------------------------
# Review report
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    """One problem the Refiner should fix."""

    severity: str  # "blocker" (must fix), "major" or "minor"
    section: str
    problem: str
    fix: str

    def line(self) -> str:
        """One-line form used in prompts and reviewer notes."""
        return f"[{self.severity}] {self.section}: {self.problem} -> {self.fix}"


_SEVERITY_ORDER: Final = {"blocker": 0, "major": 1, "minor": 2}


@dataclass
class ReviewReport:
    """The Critic's verdict on one version of the brief."""

    round: int
    brief: str
    scores: dict[str, float]
    weighted: float
    passed: bool
    issues: list[Issue]
    unverified: list[str]
    missing_sections: list[str]
    judge_used: bool

    @property
    def blockers(self) -> int:
        """Number of issues that must be fixed before the brief can pass."""
        return sum(1 for issue in self.issues if issue.severity == "blocker")

    def markdown(self) -> str:
        """Human-readable critique, stored as the Critic's output."""
        status = "PASS" if self.passed else "REVISE"
        lines = [
            f"SCORE: {self.weighted:.1f}/10  STATUS: {status}  (review round {self.round})",
            "",
            "| Dimension | Weight | Score |",
            "| :--- | ---: | ---: |",
            *[
                f"| {name.capitalize()} | {int(weight * 100)}% | {self.scores[name]:.1f} |"
                for name, weight in DIMENSION_WEIGHTS.items()
            ],
        ]
        if self.issues:
            lines += [
                "",
                "Issues:",
                *[f"{i}. {issue.line()}" for i, issue in enumerate(self.issues, 1)],
            ]
        return "\n".join(lines)


def _weighted(scores: Mapping[str, float]) -> float:
    return round(sum(DIMENSION_WEIGHTS[d] * scores[d] for d in DIMENSION_WEIGHTS), 2)


# ---------------------------------------------------------------------------
# LLM judge (Critic stage 2)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM: Final = (
    "You are the Senior Quality Assurance Auditor for institutional equity research briefs. "
    "You score briefs strictly and reply with JSON only."
)

_JUDGE_PROMPT: Final = """Review this research brief for: {question}

Facts established by automated checks (treat as true):
- Figures in the brief that do NOT match the retrieved data: {unverified}
- Required sections missing: {missing}
- Grounding score (computed): {grounding}/10

Today is {today}. The retrieved data below is current and authoritative; your own
knowledge of the company may be out of date. Every figure not listed above matches the
retrieved data: never call it wrong, too large or implausible because it differs from
what you remember. Judge only how figures are presented and argued. Do not ask for
price targets, stop-loss levels, peer or competitor comparisons, or any other figure the
retrieved data does not contain.

Retrieved data (JSON):
{evidence}

Score each dimension from 0 to 10:
- coverage: fundamentals, news, price/technicals and macro are all addressed
- balance: both bull and bear cases are argued with evidence
- clarity: concise, professional, well structured
- actionability: a clear, defensible stance

List the specific problems a reviser should fix (most important first), each with the
section, the problem and a concrete fix. Do not list the unverified figures above; they
are already reported.

Brief:
{brief}

Reply with JSON only:
{{"scores": {{"coverage": 0, "balance": 0, "clarity": 0, "actionability": 0}},
 "issues": [{{"section": "...", "problem": "...", "fix": "...", "severity": "major|minor"}}]}}"""

_JSON_OBJECT: Final = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_reply(text: str) -> dict[str, Any] | None:
    match = _JSON_OBJECT.search(text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _clamp_score(value: Any) -> float | None:
    try:
        return max(0.0, min(10.0, float(value)))
    except (TypeError, ValueError):
        return None


# Words a judge uses when it claims a figure is wrong rather than badly presented.
_DISPUTE_WORDS: Final = re.compile(
    r"\b(?:incorrect|inaccurate|wrong|implausible|unrealistic\w*|too (?:large|high|low|small)|"
    r"far (?:above|below)|error|erroneous|mistake|outdated|does not match|inconsistent with)\b",
    re.IGNORECASE,
)


def _disputes_data(issue: Issue, evidence: Evidence) -> bool:
    """True when a judge issue claims that a figure from the retrieved data is wrong.

    The judge's own knowledge can be out of date (it called a live $5.54T market cap
    "far above NVIDIA's actual value"), and acting on that would make the Refiner delete
    correct data. Code-verified figures are trusted over the judge's memory.
    """
    if not _DISPUTE_WORDS.search(issue.problem):
        return False
    return any(evidence.matches(f) is not None for f in _parse_figures(issue.problem))


class Critic:
    """Reviews a brief: code checks, then the LLM judge, then the decision.

    Args:
        llm: Chat model used for the judge stage.
    """

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def review(
        self, brief: str, tools: Mapping[str, Any], question: str, round_no: int
    ) -> ReviewReport:
        """Review one version of the brief against the stored tool results."""
        if _is_unavailable_notice(brief):
            # A "no data" notice has nothing to verify; reviewing it would only invite
            # the Refiner to invent content.
            scores = dict.fromkeys(DIMENSION_WEIGHTS, 10.0)
            return ReviewReport(round_no, brief, scores, 10.0, True, [], [], [], False)

        # Stage 1: code checks.
        evidence = evidence_numbers(tools)
        grounding = check_grounding(brief, evidence)
        missing = check_sections(brief)
        issues = []
        if grounding.unverified:
            # One issue for all of them, so that a brief with many invented figures does
            # not crowd every other problem out of the Refiner's five-issue budget.
            by_section: dict[str, list[str]] = {}
            for figure in grounding.unverified:
                by_section.setdefault(grounding.sections.get(figure, "General"), []).append(
                    f"'{figure}'"
                )
            issues.append(
                Issue(
                    "blocker",
                    "; ".join(by_section),
                    "figures not found in the retrieved data: "
                    + "; ".join(f"{name}: {', '.join(f)}" for name, f in by_section.items()),
                    "replace each with the exact figure from the tool data, write a "
                    "calculated percentage as '\u2248N%' next to one of its inputs, or "
                    "remove it",
                )
            )
        issues += [
            Issue("major", section, "required section is missing", "add this section")
            for section in missing
        ]

        # Stage 2: LLM judge, with the code-check results given as facts.
        scores, judge_issues, judge_used = self._judge(brief, question, grounding, missing, tools)
        judge_issues = [i for i in judge_issues if not _disputes_data(i, evidence)]
        scores["grounding"] = grounding.score
        issues += judge_issues
        issues.sort(key=lambda issue: _SEVERITY_ORDER.get(issue.severity, 3))

        # Stage 3: decision.
        weighted = _weighted(scores)
        passed = not grounding.unverified and not missing and weighted >= PASS_SCORE
        return ReviewReport(
            round_no,
            brief,
            scores,
            weighted,
            passed,
            issues,
            grounding.unverified,
            missing,
            judge_used,
        )

    def _judge(
        self,
        brief: str,
        question: str,
        grounding: GroundingReport,
        missing: list[str],
        tools: Mapping[str, Any],
    ) -> tuple[dict[str, float], list[Issue], bool]:
        prompt = _JUDGE_PROMPT.format(
            today=date.today().isoformat(),
            evidence=_compact_evidence(tools),
            question=question or "the requested company",
            unverified=", ".join(grounding.unverified) or "none",
            missing=", ".join(missing) or "none",
            grounding=grounding.score,
            brief=brief,
        )
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        try:
            reply, _, _ = self.llm.chat(messages, None)
        except Exception as exc:  # noqa: BLE001 - the decision falls back to code checks
            logger.warning("Critic judge call failed: %s", exc)
            reply = ""

        data = _parse_judge_reply(reply)
        raw_scores = (data or {}).get("scores") or {}
        scores = {}
        for dimension in DIMENSION_WEIGHTS:
            if dimension == "grounding":
                continue
            score = _clamp_score(raw_scores.get(dimension))
            scores[dimension] = _FALLBACK_SCORE if score is None else score

        issues = []
        for item in (data or {}).get("issues") or []:
            if not isinstance(item, Mapping) or not item.get("problem"):
                continue
            severity = str(item.get("severity", "minor")).lower()
            issues.append(
                Issue(
                    severity if severity in ("major", "minor") else "minor",
                    str(item.get("section") or "General"),
                    str(item["problem"]),
                    str(item.get("fix") or "revise accordingly"),
                )
            )
        return scores, issues, data is not None


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------

_REFINE_PROMPT: Final = """Revise this research brief for: {question}

Fix ONLY these issues (most important first):
{issues}

Rules:
- Rewrite only the sections named in the issues; keep every other section word-for-word.
- Fix every listed figure wherever it appears in those sections; the brief is checked
  again by code, so a figure left in place fails again.
- The change log must describe only changes you actually made.
- The retrieved data is current and authoritative. Never remove or change a figure that
  matches it, even if it differs from what you remember about the company.
- Use only figures that appear in the retrieved data below. If a figure you need is not
  there, call a data tool to fetch it, or write that the figure is unavailable.
- Never invent numbers, estimates, price targets or catalysts.
- A percentage you calculate must be marked with "\u2248" and written on the same line as
  one of its inputs, e.g. "Operating income $22.8B (\u224860% of revenue)".

Retrieved data (JSON):
{evidence}

Current brief:
{brief}

Reply with the complete revised brief in Markdown, then a line "### Change Log"
followed by one bullet per issue saying what you changed."""

_CHANGE_LOG: Final = re.compile(r"^#{2,4}\s*change\s*log\s*$", re.IGNORECASE | re.MULTILINE)


def _split_change_log(text: str) -> tuple[str, list[str]]:
    """Separate the revised brief from the Refiner's change log."""
    match = _CHANGE_LOG.search(text)
    if not match:
        return text.strip(), []
    brief = text[: match.start()].strip()
    log = [
        line.strip().lstrip("-*• ").strip()
        for line in text[match.end() :].splitlines()
        if line.strip().lstrip("-*• ").strip()
    ]
    return brief, log


def _compact_evidence(tools: Mapping[str, Any]) -> str:
    text = json.dumps(tools, default=str, separators=(",", ":"))
    return text[:_EVIDENCE_PROMPT_CHARS]


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def _review_state(state: GraphState) -> dict[str, Any]:
    """The loop's bookkeeping in ``state.review``, created on first use."""
    review = state.review
    if not review:
        review.update({"history": [], "current": None, "change_logs": []})
    return review


def _harvest_tool_results(agent: Agent, trace: Iterable[Any], state: GraphState) -> None:
    """Add data the Refiner fetched to the evidence, as the graph does for agent nodes."""
    store = state.tools.setdefault(agent.name, {})
    for step in trace:
        if step.step_type == "tool_result":
            value = step.content
            if isinstance(value, str):
                parsed = _maybe_json(value)
                value = value if parsed is None else parsed
            store_tool_result(store, step.metadata.get("tool", "unknown_tool"), value)


def _best(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Best version so far: fewest blockers, then highest score, then latest."""
    return max(history, key=lambda r: (-r["blockers"], r["weighted"], r["round"]))


def make_review_nodes(
    llm: LLM, refiner: Agent
) -> tuple[
    Callable[[GraphState], dict],
    Callable[[GraphState], dict],
    Callable[[GraphState], dict],
    Callable[[GraphState], str],
]:
    """Build the ``critic``, ``refiner`` and ``release`` nodes and the router between them.

    Args:
        llm: Chat model for the Critic's judge stage.
        refiner: The Refiner agent (keeps its data tools for fetching missing figures).

    Returns:
        ``(critic_node, refiner_node, release_node, route_review)``.
    """
    critic = Critic(llm)

    def critic_node(state: GraphState) -> dict[str, Any]:
        review = _review_state(state)
        brief = review["current"] or state.agents.get("DraftWriter", "")
        review["current"] = brief
        report = critic.review(brief, state.tools, state.question, len(review["history"]))
        record = {**asdict(report), "blockers": report.blockers}
        review["history"].append(record)
        state.agents["ResearchCritic"] = report.markdown()
        status = "PASS" if report.passed else "REVISE"
        return {"review_status": f"{status} {report.weighted:.1f}/10, {report.blockers} blockers"}

    def refiner_node(state: GraphState) -> dict[str, Any]:
        review = _review_state(state)
        latest = review["history"][-1]
        issues = [Issue(**i) for i in latest["issues"]][:MAX_ISSUES_PER_ROUND]
        prompt = _REFINE_PROMPT.format(
            question=state.question,
            issues="\n".join(f"{n}. {issue.line()}" for n, issue in enumerate(issues, 1)),
            evidence=_compact_evidence(state.tools),
            brief=latest["brief"],
        )
        refiner.messages = []  # each round starts fresh; the prompt carries the context
        response = refiner.run(prompt)
        _harvest_tool_results(refiner, response.trace, state)

        revised, change_log = _split_change_log(response.content or "")
        if not revised or revised.startswith(("(No text returned)", "API Error")):
            revised, change_log = latest["brief"], ["Refiner returned no usable revision"]
        review["current"] = revised
        review["change_logs"].append(change_log)
        state.agents["BriefRefiner"] = revised
        state.refine_count += 1
        return {"refine_summary": f"round {state.refine_count}: {len(change_log)} changes"}

    def route_review(state: GraphState) -> str:
        review = _review_state(state)
        history = review["history"]
        latest = history[-1]
        if latest["passed"]:
            return "release"
        if state.refine_count >= MAX_REFINEMENT_ROUNDS:
            return "release"
        if len(history) > 1 and latest["weighted"] - history[-2]["weighted"] < MIN_IMPROVEMENT:
            return "release"
        return "refine"

    def release_node(state: GraphState) -> dict[str, Any]:
        review = _review_state(state)
        best = _best(review["history"])
        output = best["brief"]
        if not best["passed"]:
            notes = [Issue(**i).line() for i in best["issues"][:MAX_ISSUES_PER_ROUND]]
            output += (
                "\n\n---\n### Reviewer notes\n"
                f"This brief did not pass review (score {best['weighted']:.1f}/10). "
                "Open issues:\n" + "\n".join(f"- {note}" for note in notes)
            )
        state.output = output
        return {
            "released": f"round {best['round']} ({'passed' if best['passed'] else 'with notes'})"
        }

    critic_node.description = (  # type: ignore[attr-defined]
        "Auditing draft report for numeric accuracy and factual grounding"
    )
    refiner_node.description = (  # type: ignore[attr-defined]
        "Refining draft sections and grounding unverified assertions"
    )
    release_node.description = (  # type: ignore[attr-defined]
        "Finalizing published institutional research briefing"
    )

    return critic_node, refiner_node, release_node, route_review
