"""The critic node (W2-05): contract checks, units against tool results, metric vs the question.

    make_critic()           -> the node: state -> {verdict}
    critique(state)         Verdict from check_contract + unit and metric checks + the W3-03 checks of what the draft
                            answers against what the question asks (agent/critic_semantic.py; no LLM)
    question_scope(state)   planner entities as QuestionScope (spec §8 invariant 4)

The graph owns the send-back (MAX_CRITIC_RUNS=2) and with_verification (finish). This node writes only
`verdict` and does not rewrite the draft (ADR-024: the composer owns the draft); "owns units" means it
*flags* scale/unit mismatches so the send-back can fix them. Calculator recomputation is already in
check_calculations (W1-07 / ADR-022 → tools.calculator.compute).

Unit and metric checks read fact-tool rows the *draft uses* (same ticker, year, period, duration, and
metric or line-item label), not the whole append-only tool_results log, so a send-back that fetches the
right metric can clear an earlier wrong fetch (ADR-026). Phrase tables come from config/concept_map.yaml
(loaded on first use, cached) plus glossary aliases from the development set / tool schema — never gold-v1.
"""
from __future__ import annotations

import functools
import re
from typing import Any

from agent.contract import AnswerContract, Value
from agent.critic_checks import (
    DERIVED_UNITS,
    UNIT_SCALE,
    QuestionScope,
    Violation,
    check_contract,
    expected_operations,
    same_number,
)
from agent.critic_semantic import semantic_violations
from agent.injection import answer_problems
from agent.state import AgentState, ToolResult, Verdict
from ingest.xbrl import load_concept_map

FACT_TOOLS = frozenset({"get_financial_fact", "compare_periods"})
# Line items that are not among the 13 metrics: any XBRL fact used in the draft is a forbidden substitute.
SEARCH_ONLY = (
    re.compile(r"\bstock[ -]?based compensation\b", re.IGNORECASE),
    re.compile(r"\bshare[ -]?based compensation\b", re.IGNORECASE),
    re.compile(r"\bequity[ -]?based compensation\b", re.IGNORECASE),
)
# Spans that contain a metric word but are not that metric (masked before phrase matching).
PHRASE_MASKS = (
    re.compile(r"\bcost of revenue\b", re.IGNORECASE),
    re.compile(r"\bnoninterest revenue\b", re.IGNORECASE),
    re.compile(r"\bnet income attributable to\b(?:\s+\w+){0,6}", re.IGNORECASE),
)
# Bare "cash" must not fire on cash-flow / dividend wording.
CASH_BARE = re.compile(
    r"\bcash\b(?!\s+(?:and\s+cash\s+equivalents|flow|flows|generated|used|paid|dividend|dividends|"
    r"from\s+operat|provided\s+by|provided\s+from))",
    re.IGNORECASE,
)
# Non-contiguous cash-flow wording ("cash did X generate from operating activities").
# Stop at a clause boundary so "cash used in investing … and operating income" is not stolen.
OPERATING_CASH_LOOSE = re.compile(
    r"\bcash\b(?:(?![,;?]|\band\b).){0,60}?"
    r"\b(?:flow|flows|generate|generated|used|provided)\b"
    r"(?:\s+(?:from|by|in))?\s+operat\w*",
    re.IGNORECASE,
)
OPERATING_CASH_PHRASES = (
    "operating cash flow", "cash flow from operations", "cash flows from operations",
    "cash flow from operating activities", "cash flows from operating activities",
    "cash provided by operating activities", "cash provided by operations",
    "cash from operations", "cash from operating", "net cash from operations",
    "net cash provided by operating activities", "net cash from operating activities",
)


def question_scope(state: AgentState) -> QuestionScope | None:
    """The planner's entities as the critic's scope (spec §8 invariant 4); None when nothing was read."""
    entities = state.get("entities")
    if entities is None:
        return None
    return QuestionScope(frozenset(entities.tickers), frozenset(entities.years), frozenset(entities.periods))


def _scaled(value: float, unit: str) -> float | None:
    scale = UNIT_SCALE.get(unit)
    return None if scale is None else value * scale


def _fact_rows(results: list[ToolResult]) -> list[dict[str, Any]]:
    """Successful get_financial_fact / compare_periods outputs as flat fact dicts (one row per period)."""
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.call.name not in FACT_TOOLS or result.error_type is not None:
            continue
        out = result.output
        if isinstance(out, dict) and "value" in out and "metric" in out:
            rows.append(out)
        elif isinstance(out, list):
            # compare_periods always sets missing: None on success, an error type when the year has no value
            rows.extend(row for row in out if isinstance(row, dict) and row.get("value") is not None
                        and row.get("metric") is not None and not row.get("missing"))
    return rows


def _same_period(value: Value, row: dict[str, Any]) -> bool:
    if (row.get("ticker") != value.ticker or row.get("fiscal_year") != value.fiscal_year
            or row.get("fiscal_period") != value.fiscal_period):
        return False
    row_duration, value_duration = row.get("duration"), value.duration
    if row_duration is not None and value_duration is not None and row_duration != value_duration:
        return False
    return True


def _name_matches_row(value: Value, row: dict[str, Any]) -> bool:
    """Match the mapped metric key or the company's line-item label (spec §7.5)."""
    if row.get("metric") == value.name:
        return True
    label = row.get("label")
    return isinstance(label, str) and label.casefold() == value.name.casefold()


def _named_period_facts(value: Value, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fact rows for the value's company and period whose metric or label matches the value's name."""
    return [row for row in rows
            if (row.get("ticker") == value.ticker and row.get("fiscal_year") == value.fiscal_year
                and row.get("fiscal_period") == value.fiscal_period and _name_matches_row(value, row))]


def _matching_fact(value: Value, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A tool fact for the same company, period, duration and metric/label as `value`, if any."""
    for row in _named_period_facts(value, rows):
        if _same_period(value, row):
            return row
    return None


def draft_fact_rows(contract: AnswerContract, results: list[ToolResult]) -> list[dict[str, Any]]:
    """Fact rows the draft actually uses (values or matching xbrl citations), ignoring stale tool log rows."""
    rows = _fact_rows(results)
    if not rows:
        return []
    used: list[dict[str, Any]] = []
    seen: set[int] = set()
    for value in contract.values:
        if value.unit in DERIVED_UNITS:
            continue
        for row in rows:
            if id(row) in seen:
                continue
            if _same_period(value, row) and _name_matches_row(value, row):
                used.append(row)
                seen.add(id(row))
    for citation in contract.citations:
        if citation.source_type != "xbrl":
            continue
        for row in rows:
            if id(row) in seen:
                continue
            if (row.get("ticker") == citation.ticker and row.get("fiscal_year") == citation.fiscal_year
                    and row.get("fiscal_period") == citation.fiscal_period
                    and row.get("concept") == citation.concept):
                used.append(row)
                seen.add(id(row))
    return used


def check_units_against_tools(contract: AnswerContract, results: list[ToolResult]) -> list[Violation]:
    """Draft values must use the filing units the tools returned (common-scale compare via UNIT_SCALE)."""
    rows = _fact_rows(results)
    if not rows:
        return []
    out: list[Violation] = []
    for value in contract.values:
        if value.unit in DERIVED_UNITS:
            continue
        named = _named_period_facts(value, rows)
        timed = [row for row in named if _same_period(value, row)]
        if timed:
            fact = timed[0]
        elif named:
            fact = named[0]
            out.append(Violation(
                "units",
                f"value {value.name} has duration {value.duration} but the tool returned "
                f"{fact.get('duration')}",
            ))
            continue
        else:
            # No metric/label match: a filing-text figure next to an unrelated fact must not be compared
            # (mixed answers). Company line-item names are covered by the label match above.
            continue
        fact_unit = fact.get("unit")
        if not isinstance(fact_unit, str) or fact_unit not in UNIT_SCALE or value.unit not in UNIT_SCALE:
            continue
        draft_abs, fact_abs = _scaled(value.value, value.unit), _scaled(float(fact["value"]), fact_unit)
        if draft_abs is None or fact_abs is None:
            continue
        if same_number(draft_abs, fact_abs) and value.unit != fact_unit:
            out.append(Violation(
                "units",
                f"value {value.name} is {value.unit} but the tool returned {fact_unit} "
                f"({fact['value']}); values must use filing units",
            ))
        elif not same_number(draft_abs, fact_abs):
            out.append(Violation(
                "units",
                f"value {value.name} = {value.value} {value.unit} is not the tool's "
                f"{fact['value']} {fact_unit} in filing units",
            ))
    return out


@functools.cache
def _metric_phrases() -> tuple[tuple[str, str], ...]:
    """(phrase, metric) pairs from the concept map, longest phrase first.

    Loads config/concept_map.yaml on first call (cached). A missing or invalid map raises on that first call.
    """
    pairs: list[tuple[str, str]] = []
    for name, metric in load_concept_map().items():
        phrases = {name.replace("_", " "), name.replace("_", "-")}
        for concept in metric.concepts:
            phrases.add(concept.label.lower())
        # Full company labels only (do not split on commas: "total revenue, net of interest expense" is one label).
        for _, label in metric.ticker_labels:
            text = " ".join(label.lower().split())
            if text:
                phrases.add(text)
        if name == "revenue":
            phrases.update({"total revenue", "net sales", "net revenues", "total net sales", "total revenues",
                            "revenues", "total sales"})
        elif name == "net_income":
            phrases.update({"net earnings", "net income"})
        elif name == "capex":
            phrases.update({"capital expenditures", "capital expenditure", "capex"})
        elif name == "cash":
            phrases.add("cash and cash equivalents")
            # bare "cash" is matched separately (CASH_BARE) so cash-flow wording is not stolen
        elif name == "rd_expense":
            phrases.update({"research and development", "r&d", "r & d", "rd expense", "r&d expense"})
        elif name == "operating_income":
            phrases.update({"operating income", "operating earnings"})
        elif name == "gross_profit":
            phrases.add("gross profit")
        elif name == "stockholders_equity":
            phrases.update({"stockholders' equity", "stockholders equity", "shareholders' equity",
                            "shareholders equity"})
        elif name == "total_assets":
            phrases.add("total assets")
        elif name == "total_liabilities":
            phrases.add("total liabilities")
        elif name == "long_term_debt":
            phrases.update({"long-term debt", "long term debt"})
        elif name == "operating_cash_flow":
            phrases.update(OPERATING_CASH_PHRASES)
        elif name == "eps_diluted":
            phrases.update({"diluted eps", "diluted earnings per share", "eps"})
        for phrase in phrases:
            cleaned = " ".join(phrase.lower().split())
            if cleaned and cleaned not in {"sales", "cash"}:
                pairs.append((cleaned, name))
    pairs.sort(key=lambda item: (-len(item[0]), item[0], item[1]))
    return tuple(pairs)


def _mask_false_metric_spans(text: str) -> str:
    """Blank out spans that contain a metric word but are not that metric."""
    out = text
    for pattern in PHRASE_MASKS:
        out = pattern.sub(lambda m: " " * len(m.group(0)), out)
    return out


def named_metrics(question: str) -> frozenset[str]:
    """Mapped metrics the question's wording clearly names (longest phrase wins per span)."""
    text = _mask_false_metric_spans(" ".join(question.lower().split()))
    found: set[str] = set()
    covered = [False] * len(text)

    def _cover(start: int, end: int, metric: str) -> None:
        if any(covered[i] for i in range(start, end)):
            return
        for i in range(start, end):
            covered[i] = True
        found.add(metric)

    for match in OPERATING_CASH_LOOSE.finditer(text):
        _cover(match.start(), match.end(), "operating_cash_flow")
    for phrase, metric in _metric_phrases():
        pattern = re.compile(rf"(?<![\w]){re.escape(phrase)}(?![\w])", re.IGNORECASE)
        for match in pattern.finditer(text):
            _cover(match.start(), match.end(), metric)
    for match in CASH_BARE.finditer(text):
        _cover(match.start(), match.end(), "cash")
    return frozenset(found)


def search_only_line_item(question: str) -> str | None:
    """A line item the question names that must come from search_filings, never another metric."""
    for pattern in SEARCH_ONLY:
        match = pattern.search(question)
        if match:
            return match.group(0)
    return None


def metric_question_text(question: str, history: list[str] | None = None) -> str:
    """The turn that names the line item: the question, or the latest history turn (follow-ups)."""
    if named_metrics(question) or search_only_line_item(question):
        return question
    for prior in reversed(history or []):
        if named_metrics(prior) or search_only_line_item(prior):
            return prior
    return question


def check_metric_matches_question(question: str, results: list[ToolResult],
                                  contract: AnswerContract | None = None,
                                  history: list[str] | None = None) -> list[Violation]:
    """Draft-used fact metrics must fit the question; search-only line items may not use XBRL facts.

    Judges the draft, not the append-only tool log: only rows the draft uses (values / xbrl citations).
    Without a contract (unit tests of the phrase table), falls back to all successful fact rows.
    """
    text = metric_question_text(question, history)
    rows = draft_fact_rows(contract, results) if contract is not None else _fact_rows(results)
    if not rows:
        return []
    forbidden = search_only_line_item(text)
    wanted = named_metrics(text)
    if forbidden:
        # W3-03: only the substituted fact is wrong; a fact the question also names (the revenue a share of revenue
        # divides by) is a right input to a searched numerator
        metrics = sorted({row["metric"] for row in rows} - wanted)
        return [Violation(
            "metric",
            f"the question asks for {forbidden!r}, which is not a mapped metric; "
            f"search_filings must be used for it, not get_financial_fact ({', '.join(metrics)})",
        )] if metrics else []
    if not wanted:
        return []
    fetched = {row["metric"] for row in rows}
    if expected_operations(text):
        if wanted & fetched:
            return []
        return [Violation(
            "metric",
            f"the question names {', '.join(sorted(wanted))}, but the tools returned "
            f"{', '.join(sorted(fetched))}",
        )]
    return [Violation(
        "metric",
        f"the question names {', '.join(sorted(wanted))}, but a tool returned {metric}",
    ) for metric in sorted(fetched - wanted)]


def check_injected_instructions(contract: AnswerContract) -> list[Violation]:
    """W2-13 (docs/spec.md §6.3): the answer must not show what an injected instruction asks for: a web address or
    e-mail, the prompt or configuration, a request for credentials, a changed role (agent/injection.answer_problems).
    Independent of the composer leaving tampered passages out, so it also catches an injection the passage detector
    missed. A refusal's fixed sentence passes it like any other clean answer."""
    return [Violation("injection", problem) for problem in answer_problems(contract.answer)]


def critique(state: AgentState, *, injection_guard: bool = True) -> Verdict:
    """Every critic check on the latest draft: contract (W1-07) + units + metric (no invariant 7) + injected
    instructions (W2-13; off only for the measurement of the defences)."""
    draft = state.get("draft")
    if draft is None:
        return Verdict(violations=[("critic", "no draft to check")])
    question = state.get("question") or ""
    history = list(state.get("history") or [])
    results = list(state.get("tool_results") or [])
    violations = (check_contract(draft, question_scope(state), question)
                  + check_units_against_tools(draft, results)
                  + check_metric_matches_question(question, results, contract=draft, history=history)
                  + semantic_violations(question, history, draft, results)
                  + (check_injected_instructions(draft) if injection_guard else []))
    return Verdict.of(violations)


def make_critic(*, injection_guard: bool = True) -> Any:
    """The critic node for agent.graph.Nodes(critic=...). `injection_guard=False` only for the W2-13 measurement of
    the defences (eval/run_injection.py, the "before" run)."""

    async def critic(state: AgentState) -> dict[str, Any]:
        return {"verdict": critique(state, injection_guard=injection_guard)}

    return critic
