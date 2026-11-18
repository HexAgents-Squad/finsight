"""The executor node (W2-04): run the plan's tool steps, then the plan's calculation, within the run's limits.

    make_executor(call_tool=None)   -> the node for agent.graph.Nodes(executor=...)
    execute(state, call_tool)       the work (tests pass a fake call_tool)
    calculation_call(op, results)   the calculate call ADR-025 asks for, built from the fetched figures

Each step of the plan (search_filings, get_financial_fact, compare_periods; ADR-025) runs in order through
tools.registry.call_tool_async, the agent's single entry point, which returns a typed error instead of raising
(ADR-014). Every call and its output is recorded as a ToolResult. The node returns only the round's NEW results:
`tool_results` is an append-only reducer (agent/state.py), so returning the whole list would duplicate it.

Guards, from state["limits"] (W2-01: MAX_AGENT_STEPS, MAX_TOKENS_PER_QUERY; the QUERY_TIMEOUT_S wall clock wraps
the whole run in agent.graph.run):
- step cap: no call beyond max_steps tool calls for the question (all rounds together); the round stops with a
  `stop_reason`, and the graph ends the run with "could not verify";
- token budget: no call once the question's LLM tokens (`state["tokens"]`) plus the tokens of the tool output
  fetched so far reach max_tokens. The executor uses no LLM, but everything it fetches goes into the composer's
  prompt; it counts that output with core.tokens.count_tokens (Qwen3's tokenizer; a conservative 2 characters per
  token if the tokenizer cannot load) and does not add it to `state["tokens"]`, which the composer's own prompt
  tokens will report (adding it would count it twice);
- loop guard: a call identical to an earlier one (same tool and arguments) is not sent again. If the earlier one
  failed, the run stops (the plan would repeat a failing call); if it succeeded, its result is reused, so a
  send-back or a replan that keeps a step costs no extra step.

A typed tool error stops the round and sets `tool_error`, so the graph replans once (W2-01). `not_reported` and
the calculator's `undefined` are answers, not errors (spec §4.2, §7.2, §7.5: the answer says "not reported" or that
the measure is undefined), so they do not trigger a replan; a fourth-quarter question asked as Q4 is answered this
way. An exception from the tool caller (a bug: call_tool_async does not raise) becomes an internal_error result.

The calculation (ADR-025): a plan has no numbers, so when `plan.calculation` is set the executor builds the
calculate call from the figures the plan's steps fetched, in step order: [part, whole] for ratio and percent (the
planner fetches the line item, then revenue), chronological [old, new] for growth, difference over years and cagr
(with `years` from the fiscal years), and the inputs' shared unit for difference and sum. Figures in different
units, too few figures, or more than two figures for a two-input operation (ambiguous) mean no calculation and a
note in `errors`: the composer then answers from the fetched figures and the critic checks it.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agent.state import AgentState, Plan, ToolCall, ToolResult
from tools.base import INTERNAL_ERROR, NOT_REPORTED, UNDEFINED, error_result

log = logging.getLogger(__name__)

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[Any]]
TokenCounter = Callable[[str], int]
CHARS_PER_TOKEN_FLOOR = 2       # core/tokens.py: filing tables run at 2.2 characters per token, prose at 4.6
ANSWERS = frozenset({NOT_REPORTED, UNDEFINED})       # typed results that answer the question: no replan
FACT_TOOLS = frozenset({"get_financial_fact", "compare_periods"})
PAIR_OPERATIONS = frozenset({"ratio", "percent", "growth", "difference", "cagr"})   # tools.calculator: two inputs
OVER_TIME = frozenset({"growth", "cagr"})            # always chronological
DEFAULT_MAX_STEPS, DEFAULT_MAX_TOKENS = 10, 12_000   # agent.graph.Limits() defaults, for a state without limits


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
    try:
        output = await call_tool(call.name, call.arguments)
    except Exception as err:                           # noqa: BLE001 - the graph must never see a tool exception
        log.exception("tool %s raised (call_tool_async should return a typed error)", call.name)
        output = error_result(INTERNAL_ERROR, f"{type(err).__name__}: {err}")
    return ToolResult(call=call, output=output)


# ----------------------------------------------------------------------------------------------- calculation
def _figures(results: list[ToolResult]) -> list[tuple[float, str, int | None]]:
    """(value, unit, fiscal year) of every figure the fact steps fetched, in step order (a compare_periods call
    gives one per year that has a value)."""
    out: list[tuple[float, str, int | None]] = []
    for result in results:
        if result.call.name not in FACT_TOOLS or result.error_type is not None:
            continue
        rows = result.output if isinstance(result.output, list) else [result.output]
        for row in rows:
            value = row.get("value") if isinstance(row, dict) else None
            if isinstance(value, int | float) and not isinstance(value, bool) and isinstance(row.get("unit"), str):
                out.append((float(value), row["unit"], row.get("fiscal_year")))
    return out


def calculation_call(operation: str, results: list[ToolResult]) -> tuple[ToolCall | None, str | None]:
    """The calculate call for `operation` over the fetched figures, or (None, why not)."""
    figures = _figures(results)
    if len(figures) < 2:
        return None, f"needs two figures, {len(figures)} fetched"
    units = {unit for _, unit, _ in figures}
    if len(units) > 1:
        return None, f"the figures are in different units ({', '.join(sorted(units))})"
    unit = units.pop()
    years = [year for _, _, year in figures]
    by_year = None not in years and len(set(years)) == len(years)
    if operation in OVER_TIME or (operation == "difference" and by_year):
        figures = sorted(figures, key=lambda f: f[2] if f[2] is not None else 0)
        if operation in OVER_TIME and len(figures) > 2:
            figures = [figures[0], figures[-1]]       # a trend over several years: first and last
    if operation in PAIR_OPERATIONS and len(figures) != 2:
        return None, f"{operation} takes two figures, {len(figures)} fetched (ambiguous)"
    arguments: dict[str, Any] = {"operation": operation, "inputs": [value for value, _, _ in figures]}
    if operation == "cagr":
        first, last = figures[0][2], figures[-1][2]
        if first is None or last is None or last <= first:
            return None, "cagr needs two different fiscal years"
        arguments["years"] = float(last - first)
    if operation in ("difference", "sum"):
        arguments["unit"] = unit
    return ToolCall(name="calculate", arguments=arguments), None


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
    fetched = output_tokens(earlier, count)            # grows with each new result
    known = {call_key(r.call): r for r in earlier}
    new: list[ToolResult] = []
    used: list[ToolResult] = []                        # this plan's results, in step order (reused ones included)
    notes: list[str] = []

    async def attempt(call: ToolCall) -> tuple[ToolResult | None, str | None]:
        """The call's result (run or reused), or a stop reason."""
        nonlocal fetched
        prior = known.get(call_key(call))
        if prior is not None:
            if is_error(prior):
                args = json.dumps(call.arguments, sort_keys=True)
                return None, f"repeated identical tool call: {call.name} {args} failed before ({prior.error_type})"
            return prior, None
        if len(earlier) + len(new) >= max_steps:
            return None, f"step limit reached ({max_steps} tool calls)"
        if llm_tokens + fetched >= max_tokens:
            return None, (f"token budget exceeded ({llm_tokens} LLM tokens + {fetched} tokens of tool output, "
                          f"at most {max_tokens})")
        result = await _run(call_tool, call)
        fetched += output_tokens([result], count)
        new.append(result)
        known[call_key(call)] = result
        return result, None

    for step in plan.steps:
        result, stop = await attempt(ToolCall(name=step.tool, arguments=step.arguments))
        if stop:
            return _update(new, None, notes, stop)
        used.append(result)
        if is_error(result):
            return _update(new, result.error_type, notes, None)
    if plan.calculation:
        call, why = calculation_call(plan.calculation, used)
        if call is None:
            notes.append(f"executor: {plan.calculation} not calculated: {why}")
        else:
            result, stop = await attempt(call)
            if stop:
                return _update(new, None, notes, stop)
            if is_error(result):
                return _update(new, result.error_type, notes, None)
    return _update(new, None, notes, None)


def _update(new: list[ToolResult], tool_error: str | None, notes: list[str], stop: str | None) -> dict[str, Any]:
    update: dict[str, Any] = {"tool_results": new, "tool_error": tool_error}
    if notes:
        update["errors"] = notes
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
