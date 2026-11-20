"""The executor node (W2-04): run the plan's tool steps, then the plan's calculation, within the run's limits.

    make_executor(call_tool=None, count_tokens=None)   -> the node for agent.graph.Nodes(executor=...)
    execute(state, call_tool, count_tokens=None)       the work (tests pass a fake call_tool and counter)
    calculation_call(op, results)                      the calculate call ADR-025 asks for (fetched figures)
    checked_section_call(result)                       the statement a "not reported" fact was checked on (W2-07)

Each step of the plan (search_filings, get_financial_fact, compare_periods; ADR-025) runs in order through
tools.registry.call_tool_async, the agent's single entry point, which returns a typed error instead of raising
(ADR-014). Every call and its output is recorded as a ToolResult. The node returns only the round's NEW results:
`tool_results` is an append-only reducer (agent/state.py), so returning the whole list would duplicate it.

Guards, from state["limits"] (W2-01: MAX_AGENT_STEPS, MAX_TOKENS_PER_QUERY; the QUERY_TIMEOUT_S wall clock wraps
the whole run in agent.graph.run):
- step cap: no call beyond max_steps tool calls for the question (all rounds together); a plan step that would
  pass it stops the round with a `stop_reason` (the graph then ends the run with "could not verify"); a
  calculation that would pass it is skipped with a note, so the composer still answers from the fetched figures;
- token budget: no call once the question's LLM tokens (`state["tokens"]`) plus the tokens of the tool output
  fetched so far reach max_tokens, handled like the step cap. The executor uses no LLM, but everything it fetches
  goes into the composer's prompt; it counts that output with core.tokens.count_tokens (Qwen3's tokenizer; a
  conservative 2 characters per token if the tokenizer cannot load) and does not add it to `state["tokens"]`,
  which the composer's own prompt tokens will report (adding it would count it twice; for the same reason, once
  the composer has run, earlier rounds' output is not counted again on a send-back, which relies on the W2-06
  composer reporting its prompt tokens);
- loop guard: a call identical to an earlier one of the question (same tool and arguments) is never sent again.
  If the earlier one succeeded, its result is reused, so a send-back or a replan that keeps a step costs no step;
  if it failed, it is skipped with a note and the plan's other steps run (a replan's fallback search still runs),
  and no calculation is made when any of the plan's fact steps failed before (conservative: its figure may be an
  input). Only a new error sets `tool_error`, so the graph's own bound applies: one replan, then the composer
  answers "not found" (W2-01). The one exception is `service_unavailable` (tools/base.py: "retry later"): a
  repeat of such a call is sent once more (never a third time, and not past a guard).

A typed tool error stops the round and sets `tool_error`, so the graph replans once (W2-01). `not_reported` and
the calculator's `undefined` are answers, not errors (spec §4.2, §7.2, §7.5: the answer says "not reported" or that
the measure is undefined), so they do not trigger a replan; a fourth-quarter question asked as Q4 is answered this
way. An exception from the tool caller (a bug: call_tool_async does not raise) becomes an internal_error result.

The checked statement (W2-07): a "not reported" answer cites the section that was checked (spec §5, category G),
so for each not_reported fact (a get_financial_fact error, or a compare_periods year missing as not_reported) the
executor also searches that filing's financial statements for the statement the metric would be on
(checked_section_call: the statement from config/concept_map.yaml, five passages, of which the composer cites the
statement's own page; the 10-K for a fourth quarter). It runs after the calculation, so it never takes the
calculation's step or tokens, and is otherwise a step like any other (counted, guarded, never sent twice); its
failure is not a tool error, since the fact's answer stands.

The calculation (ADR-025): a plan has no numbers, so when `plan.calculation` is set the executor builds the
calculate call from the figures the plan's fact steps fetched (each figure keeps its company, metric, period and
duration; the same figure fetched twice counts once; a not_reported fact is a planned figure with no value):
- all operations: one unit and one duration (a balance-sheet instant has none, so it pairs with any; full-year
  flows are 12M and quarterly ones 3M/6M/9M, so the two are never mixed), and no missing planned figure, except
  a gap inside a series (a compare_periods trend keeps its first and last period);
- growth and cagr: one series (one company and metric), oldest to newest by fiscal year then quarter; a longer
  series uses its first and last period; cagr needs the same period each year, so `years` = last minus first
  fiscal year is right;
- percent and ratio: two figures for the same fiscal year and period; percent is [part, whole] with the
  company's revenue as the whole whichever comes first (the planner fetches the line item, then revenue); ratio
  keeps step order;
- difference: one series as for growth, or two companies (or metrics) for the same period in step order (second
  minus first: the composer states which is which); sum: every planned figure, balance-sheet values all at one
  date; difference and sum pass the shared unit.
Anything else (too few figures, several series for growth, a count or mix the operation cannot take) means no
calculation and a note in `errors`, never a guessed pair: the composer then answers from the fetched figures.
"""
from __future__ import annotations

import functools
import json
import logging
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agent.state import AgentState, Plan, ToolCall, ToolResult
from core import tracing
from ingest.xbrl import load_concept_map
from retrieval.statement import statement_item
from tools.base import INTERNAL_ERROR, NOT_REPORTED, SERVICE_UNAVAILABLE, UNDEFINED, error_result

log = logging.getLogger(__name__)

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[Any]]
TokenCounter = Callable[[str], int]
CHARS_PER_TOKEN_FLOOR = 2       # core/tokens.py: filing tables run at 2.2 characters per token, prose at 4.6
ANSWERS = frozenset({NOT_REPORTED, UNDEFINED})       # typed results that answer the question: no replan
TRANSIENT = frozenset({SERVICE_UNAVAILABLE})         # tools/base.py: "retry later": a repeat is sent once more
FACT_TOOLS = frozenset({"get_financial_fact", "compare_periods"})
PAIR_OPERATIONS = frozenset({"ratio", "percent", "growth", "difference", "cagr"})   # tools.calculator: two inputs
OVER_TIME = frozenset({"growth", "cagr"})            # one series, always chronological
PERIOD_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4}   # within a fiscal year (the year ends with Q4)
WHOLE_METRIC = "revenue"                             # the whole of a percent (a margin, a share of revenue)
DEFAULT_MAX_STEPS, DEFAULT_MAX_TOKENS = 10, 12_000   # agent.graph.Limits() defaults, for a state without limits
STEP_LIMIT, TOKEN_LIMIT = "step limit reached", "token budget exceeded"
# The search for each statement, worded like the statement's own page, and its title as filings print it (the
# composer cites the statement's page, agent.composer.is_statement_page). Measured offline on every 10-K of the
# corpus (20 companies x 3 years x 3 statements, Item 8, 5 hits): the page is among the hits for 162 of 180 with
# these, 158 with "consolidated balance sheet: total assets, ..."-style ones, 113 with the bare statement names
# (which find the auditor's report, the index or notes first)
STATEMENT_QUERIES = {
    "cash_flows": "CONSOLIDATED STATEMENTS OF CASH FLOWS (in millions) operating activities net income adjustments "
                  "investing activities financing activities",
    "balance_sheet": "CONSOLIDATED BALANCE SHEETS (in millions) ASSETS current assets cash and cash equivalents total "
                     "assets LIABILITIES AND STOCKHOLDERS' EQUITY total liabilities",
    "income": "CONSOLIDATED STATEMENTS OF INCOME (in millions, except per share data) revenues costs and expenses "
              "income before income taxes net income earnings per share basic diluted",
}
STATEMENT_TITLES = {
    "cash_flows": re.compile(r"statements?\s+of\s+cash\s+flows|cash\s+flows?\s+statements?",
                             re.IGNORECASE),                              # MSFT: "CASH FLOWS STATEMENTS"
    "balance_sheet": re.compile(r"balance\s+sheets?|statements?\s+of\s+(?:financial\s+)?condition", re.IGNORECASE),
    "income": re.compile(r"statements?\s+of\s+(?:income|operations|earnings)|income\s+statements?", re.IGNORECASE),
}
STATEMENT_OF_QUERY = {query: name for name, query in STATEMENT_QUERIES.items()}
CHECKED_PASSAGES = 5                 # 162 of 180 pages among the top 5 (top 3: fewer); the search runs last


def call_key(call: ToolCall) -> str:
    """The identity of a call for the loop guard: the tool and its arguments, independent of key order."""
    return json.dumps([call.name, call.arguments], sort_keys=True, default=str)


def is_error(result: ToolResult) -> bool:
    """A typed error the planner should know about (not an answer such as not_reported)."""
    return result.error_type is not None and result.error_type not in ANSWERS


async def _default_call_tool(name: str, arguments: dict[str, Any]) -> Any:
    from tools.registry import call_tool_async  # loads the tools on first use, not at import
    return await call_tool_async(name, arguments)


def _default_count_tokens(text: str) -> int:
    from core.exceptions import FinSightError
    from core.tokens import count_tokens
    try:
        return count_tokens(text)
    except FinSightError as err:                       # no tokenizer (offline, first run): over- not under-count
        log.warning("executor: %s; counting %d characters per token", err, CHARS_PER_TOKEN_FLOOR)
        return -(-len(text) // CHARS_PER_TOKEN_FLOOR)


def output_tokens(results: list[ToolResult], count: TokenCounter) -> int:
    """Tokens of the tool output in `results`, as JSON (what a prompt carries)."""
    return sum(count(json.dumps(r.output, default=str, ensure_ascii=False)) for r in results)


async def _run(call_tool: ToolCaller, call: ToolCall) -> ToolResult:
    with tracing.span(f"tool:{call.name}", input=call.arguments) as sp:     # no-op unless the run is traced (W2-08)
        try:
            output = await call_tool(call.name, call.arguments)
        except Exception as err:                       # noqa: BLE001 - the graph must never see a tool exception
            log.exception("tool %s raised (call_tool_async should return a typed error)", call.name)
            output = error_result(INTERNAL_ERROR, f"{type(err).__name__}: {err}")
        result = ToolResult(call=call, output=output)
        # not_reported and undefined are answers (ADR-028), not failures: only a typed error is a WARNING
        sp.update(output=output, metadata={"error_type": result.error_type},
                  level="WARNING" if is_error(result) else None)
    return result


# ----------------------------------------------------------------------------------------------- calculation
@dataclass(frozen=True)
class Figure:
    """One fetched figure; `value` is None for a period compare_periods asked for but has no value for."""
    value: float | None
    unit: str | None
    ticker: str | None
    metric: str | None
    fiscal_year: int | None
    fiscal_period: str
    duration: str | None

    @property
    def series(self) -> tuple[str | None, str | None]:
        return self.ticker, self.metric

    @property
    def when(self) -> tuple[int, int]:
        return self.fiscal_year or 0, PERIOD_ORDER.get(self.fiscal_period, 4)

    def label(self) -> str:
        return f"{self.ticker} {self.metric} {self.fiscal_period} {self.fiscal_year}"


def _figures(results: list[ToolResult]) -> list[Figure]:
    """Every figure the fact steps fetched, in step order, each once (a compare_periods call gives one per year)."""
    out: list[Figure] = []
    seen: set[tuple] = set()
    for result in results:
        args = result.call.arguments
        if result.call.name == "get_financial_fact" and result.error_type == NOT_REPORTED:
            rows: list[Any] = [{"value": None}]        # planned, not reported: a figure with no value
        elif result.call.name not in FACT_TOOLS or result.error_type is not None:
            continue
        else:
            rows = result.output if isinstance(result.output, list) else [result.output]
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = row.get("value")
            number = isinstance(value, int | float) and not isinstance(value, bool)
            if value is not None and not number:
                continue
            figure = Figure(value=float(value) if number else None,
                            unit=row.get("unit") if isinstance(row.get("unit"), str) else None,
                            ticker=row.get("ticker") or args.get("ticker"),
                            metric=row.get("metric") or args.get("metric"),
                            fiscal_year=row.get("fiscal_year") or args.get("fiscal_year"),
                            fiscal_period=row.get("fiscal_period") or args.get("fiscal_period") or "FY",
                            duration=row.get("duration"))
            key = (figure.series, figure.when, figure.fiscal_period, figure.duration)
            if key not in seen:
                seen.add(key)
                out.append(figure)
    return out


def calculation_call(operation: str, results: list[ToolResult]) -> tuple[ToolCall | None, str | None]:
    """The calculate call for `operation` over the fetched figures, or (None, why not)."""
    fetched = _figures(results)
    figures = [f for f in fetched if f.value is not None and f.unit is not None]
    missing = [f for f in fetched if f.value is None]        # planned, but not reported or null
    if len(figures) < 2:
        return None, f"no value for {missing[0].label()}" if missing else f"needs two figures, {len(figures)} fetched"
    units = {f.unit for f in figures}
    if len(units) > 1:
        return None, f"the figures are in different units ({', '.join(sorted(units))})"
    unit = units.pop()
    # One duration (a balance-sheet instant has none, so it pairs with any). Real full-year flows are 12M and
    # quarterly ones 3M/6M/9M (ingest/xbrl.py), so this also keeps full-year and quarterly flows apart.
    durations = {f.duration for f in figures if f.duration is not None}
    if len(durations) > 1:
        return None, f"the figures cover different durations ({', '.join(sorted(durations))})"
    series = list(dict.fromkeys(f.series for f in figures))
    over_time = operation in OVER_TIME or (operation == "difference" and len(series) == 1)
    periods = {f.fiscal_period for f in figures}
    moments = {(f.fiscal_year, f.fiscal_period) for f in figures}
    if operation == "cagr" and len(periods) > 1:
        return None, f"cagr needs the same period each year ({', '.join(sorted(periods))} fetched)"
    if not over_time and operation in PAIR_OPERATIONS and len(moments) > 1:
        return None, "the figures are for different periods (" + ", ".join(f.label() for f in figures) + ")"
    if operation == "sum" and not durations and len(moments) > 1:
        return None, "balance-sheet values at different dates (" + ", ".join(f.label() for f in figures) + ")"
    if over_time:
        if len(series) > 1:
            names = ", ".join(f"{ticker} {metric}" for ticker, metric in series)
            return None, f"{operation} takes one series, figures for {len(series)} were fetched ({names}; ambiguous)"
        figures = sorted(figures, key=lambda f: f.when)
        if len({f.when for f in figures}) != len(figures):
            return None, f"two figures for the same period ({figures[0].label()}; ambiguous)"
        # a gap inside a trend leaves its first and last period; any other missing figure may be an input
        gaps = [f for f in missing if f.series != series[0] or not figures[0].when <= f.when <= figures[-1].when]
        if gaps:
            return None, f"no value for {gaps[0].label()}"
        if operation in OVER_TIME and len(figures) > 2:
            figures = [figures[0], figures[-1]]       # a trend over several periods: first and last
    elif missing:                                     # ratio, percent, sum, difference between series
        return None, f"no value for {missing[0].label()}"
    elif (operation == "percent" and len(figures) == 2 and figures[0].ticker == figures[1].ticker
          and figures[0].metric == WHOLE_METRIC != figures[1].metric):
        figures = [figures[1], figures[0]]            # [part, whole]: the company's revenue is the whole
    if operation in PAIR_OPERATIONS and len(figures) != 2:
        return None, f"{operation} takes two figures, {len(figures)} fetched (ambiguous)"
    arguments: dict[str, Any] = {"operation": operation, "inputs": [f.value for f in figures]}
    if operation == "cagr":
        first, last = figures[0].fiscal_year, figures[-1].fiscal_year
        if first is None or last is None or last <= first:
            return None, "cagr needs two different fiscal years"
        arguments["years"] = float(last - first)
    if operation in ("difference", "sum"):
        arguments["unit"] = unit
    return ToolCall(name="calculate", arguments=arguments), None


@functools.cache
def _metrics() -> dict[str, Any]:
    return load_concept_map()


def statement_of(metric: str) -> str:
    """The statement a metric is on, from config/concept_map.yaml: an instant is on the balance sheet, a flow that
    10-Qs report only year to date on the cash-flow statement, any other flow on the income statement."""
    m = _metrics().get(metric)
    if m is not None and m.period == "instant":
        return "balance_sheet"
    return "cash_flows" if m is not None and m.quarterly_ytd_only else "income"


def _statement_search(ticker: Any, metric: Any, year: Any, period: Any) -> ToolCall | None:
    if not (isinstance(ticker, str) and isinstance(metric, str) and isinstance(year, int)):
        return None
    period = "FY" if period in (None, "Q4") else period          # Q4 is in the 10-K (never filed alone, spec §4.2)
    return ToolCall(name="search_filings", arguments={
        "query": STATEMENT_QUERIES[statement_of(metric)], "ticker": ticker, "fiscal_year": year,
        "fiscal_period": period, "form": "10-K" if period == "FY" else "10-Q", "item": statement_item(period),
        "k": CHECKED_PASSAGES})


def checked_section_call(result: ToolResult) -> ToolCall | None:
    """For a fact that came back not reported: the search of that filing's financial statements for the statement
    the metric would be on (spec §5: a "not reported" answer cites the section checked). A get_financial_fact with
    a not_reported error, or a compare_periods trend whose years are missing as not_reported (the newest such year:
    its statement prints the earlier years too). None for anything else."""
    args = result.call.arguments
    if result.call.name == "get_financial_fact" and result.error_type == NOT_REPORTED:
        return _statement_search(args.get("ticker"), args.get("metric"), args.get("fiscal_year"),
                                 args.get("fiscal_period"))
    if result.call.name == "compare_periods" and isinstance(result.output, list):
        gaps = [r for r in result.output if isinstance(r, dict) and r.get("missing") == NOT_REPORTED]
        if gaps:
            newest = max(gaps, key=lambda r: r.get("fiscal_year") or 0)
            return _statement_search(newest.get("ticker"), newest.get("metric"), newest.get("fiscal_year"),
                                     newest.get("fiscal_period"))
    return None


# ----------------------------------------------------------------------------------------------- node
async def execute(state: AgentState, call_tool: ToolCaller, count_tokens: TokenCounter | None = None) -> dict[str, Any]:
    """One executor round (see the module docstring). Returns the new results, the round's tool error (None when
    it had none), and a stop_reason / notes when a guard or the calculation needs saying."""
    plan: Plan | None = state.get("plan")
    if plan is None or plan.refuse:
        return {}
    limits = state.get("limits") or {}
    max_steps = int(limits.get("max_steps", DEFAULT_MAX_STEPS))
    max_tokens = int(limits.get("max_tokens", DEFAULT_MAX_TOKENS))
    earlier = list(state.get("tool_results") or [])
    count = count_tokens or _default_count_tokens
    llm_tokens = int(state.get("tokens") or 0)
    # Earlier output is already inside the LLM tokens once the composer has run (its prompt carries it).
    composed = bool((state.get("visits") or {}).get("composer"))
    fetched = 0 if composed else output_tokens(earlier, count)    # grows with each new result
    known = {call_key(r.call): r for r in earlier}                 # the latest result of each call
    sends = Counter(call_key(r.call) for r in earlier)            # how often each call was sent
    new: list[ToolResult] = []
    used: list[ToolResult] = []                        # this plan's results, in step order (reused ones included)
    failed_before: list[ToolCall] = []                 # this plan's calls that failed in an earlier round
    notes: list[str] = []

    async def attempt(call: ToolCall) -> tuple[ToolResult | None, str | None]:
        """The call's result (run, or reused from an earlier successful identical call); (None, None) for a call
        that failed in an earlier round and is not sent again (a service_unavailable one is retried once, unless a
        guard blocks the retry); (None, why) when a guard stopped a new call."""
        nonlocal fetched
        key = call_key(call)
        prior = known.get(key)

        def skip() -> tuple[None, None]:
            args = json.dumps(call.arguments, sort_keys=True)
            notes.append(f"executor: not sent again, failed before ({prior.error_type}): {call.name} {args}")
            failed_before.append(call)
            return None, None

        retry = prior is not None and prior.error_type in TRANSIENT and sends[key] < 2   # one more try, then not
        if prior is not None and is_error(prior) and not retry:
            return skip()
        if prior is not None and not retry:
            return prior, None
        stop = None
        if len(earlier) + len(new) >= max_steps:
            stop = f"{STEP_LIMIT} ({max_steps} tool calls)"
        elif llm_tokens + fetched >= max_tokens:
            log.info("executor: %d LLM tokens + %d tokens of tool output, at most %d", llm_tokens, fetched, max_tokens)
            stop = f"{TOKEN_LIMIT} ({llm_tokens} LLM tokens + {fetched} tokens of tool output, at most {max_tokens})"
        if stop:
            return skip() if retry else (None, stop)  # a guard blocking a retry: the call simply failed before
        sends[key] += 1
        result = await _run(call_tool, call)
        fetched += output_tokens([result], count)
        new.append(result)
        known[call_key(call)] = result
        return result, None

    for step in plan.steps:
        result, stop = await attempt(ToolCall(name=step.tool, arguments=step.arguments))
        if stop:
            return _update(state, new, None, notes, stop)
        if result is None:                             # failed in an earlier round: the plan's other steps run
            continue
        used.append(result)
        if is_error(result):                           # a new error: the graph replans once, then "not found"
            return _update(state, new, result.error_type, notes, None)
    calc_error: str | None = None
    if plan.calculation:
        call, why = calculation_call(plan.calculation, used)
        lost = [c for c in failed_before if c.name in FACT_TOOLS]
        if lost:
            call, why = None, f"no value for {lost[0].name} {json.dumps(lost[0].arguments, sort_keys=True)}"
        if call is not None:
            result, why = await attempt(call)         # a guard skips only the calculation: the figures stand
            if result is not None and is_error(result):
                calc_error, why = result.error_type, None    # a new error: the graph replans, after the check below
            if why and why.startswith(TOKEN_LIMIT):
                why = TOKEN_LIMIT                      # a stable note: the counts change between rounds
        if why:
            notes.append(f"executor: {plan.calculation} not calculated: {why}")
    # Last, so it never takes the calculation's step or tokens (review): the citation for a "not reported" fact
    checks = {call_key(c): c for r in used if (c := checked_section_call(r)) is not None}
    for call in checks.values():
        _, stop = await attempt(call)                  # its own failure is no tool error: the fact's answer stands
        if stop:
            notes.append("executor: the checked statement was not fetched: "
                         + (TOKEN_LIMIT if stop.startswith(TOKEN_LIMIT) else stop))
            break
    return _update(state, new, calc_error, notes, None)


def _update(state: AgentState, new: list[ToolResult], tool_error: str | None, notes: list[str],
            stop: str | None) -> dict[str, Any]:
    update: dict[str, Any] = {"tool_results": new, "tool_error": tool_error}
    logged = set(state.get("errors") or [])            # errors is append-only: a send-back repeats no note
    fresh = [note for note in dict.fromkeys(notes) if note not in logged]
    if fresh:
        update["errors"] = fresh
    if stop:
        update["stop_reason"] = stop
    return update


def make_executor(call_tool: ToolCaller | None = None, count_tokens: TokenCounter | None = None) -> Any:
    """The executor node for agent.graph.Nodes(executor=...); `call_tool` defaults to the tool registry and
    `count_tokens` to core.tokens.count_tokens (offline tests pass their own)."""
    caller = call_tool or _default_call_tool

    async def executor(state: AgentState) -> dict[str, Any]:
        return await execute(state, caller, count_tokens)

    return executor
