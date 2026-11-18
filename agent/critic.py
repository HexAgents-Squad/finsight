"""The critic node (W2-05): contract checks, units against tool results, metric vs the question.

    make_critic()           -> the node: state -> {verdict}
    critique(state)         Verdict from check_contract + unit and metric checks (no LLM, no I/O)
    question_scope(state)   planner entities as QuestionScope (spec §8 invariant 4)

The graph owns the send-back (MAX_CRITIC_RUNS=2) and with_verification (finish). This node writes only
`verdict`. Units: draft values are compared to get_financial_fact / compare_periods outputs on a common scale
(UNIT_SCALE); a plain-dollar integer for a millions figure, or a matching number with the wrong unit label, is a
`units` violation so the send-back can fix it (Week 1: Qwen wrote "$47.52 billion" as a wrong plain-dollar
integer). Metric: when the question clearly names a mapped line item (or a search-only phrase such as
stock-based compensation), a get_financial_fact / compare_periods result for a different metric is a `metric`
violation (ADR-026).
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
from agent.state import AgentState, ToolResult, Verdict
from ingest.xbrl import load_concept_map
from tools.financial_facts import METRICS

FACT_TOOLS = frozenset({"get_financial_fact", "compare_periods"})
# Line items that are not among the 13 metrics: any XBRL fact fetch for them is a forbidden substitute (ADR-026).
SEARCH_ONLY = (
    re.compile(r"\bstock[ -]?based compensation\b", re.IGNORECASE),
    re.compile(r"\bshare[ -]?based compensation\b", re.IGNORECASE),
    re.compile(r"\bequity[ -]?based compensation\b", re.IGNORECASE),
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


def _matching_fact(value: Value, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A tool fact for the same company, period and metric/name as `value`, if any."""
    for row in rows:
        if (row.get("ticker") == value.ticker and row.get("fiscal_year") == value.fiscal_year
                and row.get("fiscal_period") == value.fiscal_period
                and row.get("metric") == value.name):
            return row
    return None


def check_units_against_tools(contract: AnswerContract, results: list[ToolResult]) -> list[Violation]:
    """Draft values must use the filing units the tools returned (common-scale compare via UNIT_SCALE)."""
    rows = _fact_rows(results)
    if not rows:
        return []
    out: list[Violation] = []
    for value in contract.values:
        if value.unit in DERIVED_UNITS:
            continue
        fact = _matching_fact(value, rows)
        if fact is None:
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
                f"value {value.name} = {value.value} {value.unit} does not match the tool's "
                f"{fact['value']} {fact_unit} (compare on a common scale)",
            ))
    return out


@functools.cache
def _metric_phrases() -> tuple[tuple[str, str], ...]:
    """(phrase, metric) pairs from the concept map, longest phrase first (word-boundary match)."""
    pairs: list[tuple[str, str]] = []
    for name, metric in load_concept_map().items():
        phrases = {name.replace("_", " "), name.replace("_", "-")}
        for concept in metric.concepts:
            phrases.add(concept.label.lower())
        for _, label in metric.ticker_labels:
            for part in label.split(","):
                text = part.strip().lower()
                if text:
                    phrases.add(text)
        # Glossary aliases used in the tool schema (ADR-026), not every synonym.
        if name == "revenue":
            phrases.update({"total revenue", "net sales", "net revenues", "total net sales", "total revenues",
                            "revenues", "total sales"})
        elif name == "net_income":
            phrases.update({"net earnings", "net income"})
        elif name == "capex":
            phrases.update({"capital expenditures", "capital expenditure", "capex"})
        elif name == "cash":
            phrases.update({"cash and cash equivalents", "cash"})
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
            phrases.update({"operating cash flow", "cash from operations", "cash from operating"})
        elif name == "eps_diluted":
            phrases.update({"diluted eps", "diluted earnings per share", "eps"})
        for phrase in phrases:
            cleaned = " ".join(phrase.lower().split())
            if cleaned and cleaned not in {"sales"}:          # too short / ambiguous alone
                pairs.append((cleaned, name))
    pairs.sort(key=lambda item: (-len(item[0]), item[0], item[1]))
    return tuple(pairs)


def named_metrics(question: str) -> frozenset[str]:
    """Mapped metrics the question's wording clearly names (longest phrase wins per span)."""
    text = " ".join(question.lower().split())
    found: set[str] = set()
    covered = [False] * len(text)
    for phrase, metric in _metric_phrases():
        if metric not in METRICS:
            continue
        pattern = re.compile(rf"(?<![\w]){re.escape(phrase)}(?![\w])", re.IGNORECASE)
        for match in pattern.finditer(text):
            if any(covered[i] for i in range(match.start(), match.end())):
                continue
            for i in range(match.start(), match.end()):
                covered[i] = True
            found.add(metric)
    return frozenset(found)


def search_only_line_item(question: str) -> str | None:
    """A line item the question names that must come from search_filings, never another metric."""
    for pattern in SEARCH_ONLY:
        match = pattern.search(question)
        if match:
            return match.group(0)
    return None


def check_metric_matches_question(question: str, results: list[ToolResult]) -> list[Violation]:
    """Fetched metrics must fit the question; search-only line items may not use XBRL facts.

    Fact lookups: every returned metric must be one the question named. Ratio / percent / growth (and other
    calculator) questions may fetch companion inputs the wording does not name (e.g. revenue with R&D); those
    extras are allowed when at least one named metric was fetched. A wholly wrong set (none of the named
    metrics) is still a violation.
    """
    rows = _fact_rows(results)
    if not rows:
        return []
    forbidden = search_only_line_item(question)
    if forbidden:
        metrics = sorted({row["metric"] for row in rows})
        return [Violation(
            "metric",
            f"the question asks for {forbidden!r}, which is not a mapped metric; "
            f"search_filings must be used, not get_financial_fact ({', '.join(metrics)})",
        )]
    wanted = named_metrics(question)
    if not wanted:
        return []
    fetched = {row["metric"] for row in rows}
    if expected_operations(question):
        if wanted & fetched:
            return []
        return [Violation(
            "metric",
            f"the question names {', '.join(sorted(wanted))}, but the tools returned "
            f"{', '.join(sorted(fetched))}",
        )]
    out: list[Violation] = []
    for metric in sorted(fetched - wanted):
        out.append(Violation(
            "metric",
            f"the question names {', '.join(sorted(wanted))}, but a tool returned {metric}",
        ))
    return out


def critique(state: AgentState) -> Verdict:
    """Every critic check on the latest draft: contract (W1-07) + units + metric (no invariant 7)."""
    draft = state.get("draft")
    if draft is None:
        return Verdict(violations=[("critic", "the critic returned no verdict")])
    question = state.get("question") or ""
    results = list(state.get("tool_results") or [])
    violations = (check_contract(draft, question_scope(state), question)
                  + check_units_against_tools(draft, results)
                  + check_metric_matches_question(question, results))
    return Verdict.of(violations)


def make_critic() -> Any:
    """The critic node for agent.graph.Nodes(critic=...)."""

    async def critic(state: AgentState) -> dict[str, Any]:
        return {"verdict": critique(state)}

    return critic
