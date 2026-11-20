"""W2-04: the executor node: steps in order, guards, typed errors, the ADR-025 calculation, and plans from the real
planner. Offline: a fake tool caller (no data/, no index, no LLM)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent import executor as ex
from agent import graph as g
from agent.contract import AnswerContract
from agent.state import Plan, PlanStep, ToolCall, ToolResult, Verdict

P8 = Path(__file__).resolve().parents[1] / "eval" / "results" / "planner_runs" / "P8.jsonl"
NULLS = {"ticker": None, "fiscal_year": None, "fiscal_period": None, "form": None, "item": None, "k": 5}
VALUES = {("PG", "operating_income", 2023): 18134.0, ("PG", "revenue", 2023): 82006.0,
          ("GS", "net_income", 2023): 8516.0, ("GS", "net_income", 2024): 14276.0,
          ("PFE", "net_income", 2023): 2119.0, ("JNJ", "net_income", 2023): 35153.0,
          ("WFC", "total_assets", 2024): 1929845.0, ("AAPL", "revenue", 2024): 391035.0,
          ("AAPL", "revenue", 2023): 383285.0, ("AAPL", "revenue", 2025): 416161.0}


INSTANT_METRICS = {"total_assets", "total_liabilities", "stockholders_equity", "cash", "long_term_debt"}


def fact_row(ticker, metric, year, period, value) -> dict:
    """The real tools' fact shape (tools.financial_facts.FACT_FIELDS): duration 12M for FY, 3M for a quarter, null
    for a balance-sheet value (ingest/xbrl.py VALID_DURATIONS)."""
    duration = None if metric in INSTANT_METRICS else ("12M" if period == "FY" else "3M")
    return {"value": value, "unit": "USD millions", "ticker": ticker, "fiscal_year": year, "fiscal_period": period,
            "duration": duration, "period_start": None, "period_end": f"{year}-12-31", "concept": "us-gaap:X",
            "accession_no": "0000000000-00-000000", "metric": metric, "label": metric, "form": "10-K",
            "url": "https://www.sec.gov/x", "notes": []}


class FakeTools:
    """call_tool_async's stand-in: facts from VALUES, Q4 as not_reported, a calculator, scripted overrides."""

    def __init__(self, overrides=None):
        self.calls: list[tuple[str, dict]] = []
        self.overrides = overrides or {}

    async def __call__(self, name, arguments):
        self.calls.append((name, arguments))
        key = json.dumps([name, arguments], sort_keys=True)
        if key in self.overrides:
            out = self.overrides[key]
            if isinstance(out, Exception):
                raise out
            return out
        if name == "get_financial_fact":
            period = arguments.get("fiscal_period") or "FY"
            if period == "Q4":
                return {"error": {"type": "not_reported", "message": "Q4 is not reported separately"}}
            value = VALUES.get((arguments["ticker"], arguments["metric"], arguments["fiscal_year"]))
            if value is None:
                return {"error": {"type": "data_missing", "message": "no fact"}}
            return fact_row(arguments["ticker"], arguments["metric"], arguments["fiscal_year"], period, value)
        if name == "compare_periods":
            return [fact_row(arguments["ticker"], arguments["metric"], y, "FY",
                             VALUES.get((arguments["ticker"], arguments["metric"], y))) for y in arguments["years"]]
        if name == "calculate":
            from tools.calculator import calculate
            return calculate(**arguments)
        return [{"chunk_id": "x", "text": "passage"}]


def fact(ticker="AAPL", metric="revenue", year=2024, period="FY") -> PlanStep:
    return PlanStep(tool="get_financial_fact", purpose="x", arguments={
        "ticker": ticker, "metric": metric, "fiscal_year": year, "fiscal_period": period, "duration": None})


def search(query="risks") -> PlanStep:
    return PlanStep(tool="search_filings", purpose="x", arguments={**NULLS, "query": query, "ticker": "AAPL"})


def plan(*steps, calculation=None, rule=None) -> Plan:
    return Plan(reasoning="x", refuse=False, refusal_rule=rule, steps=list(steps), calculation=calculation)


def words(text: str) -> int:
    """An offline token counter (core.tokens needs the Qwen tokenizer)."""
    return len(text.split())


def run(p: Plan, tools: FakeTools | None = None, *, results=(), max_steps=10, tokens=0, max_tokens=12_000,
        count=words):
    tools = tools or FakeTools()
    state = {"plan": p, "tool_results": list(results), "tokens": tokens,
             "limits": {"max_steps": max_steps, "max_tokens": max_tokens, "timeout_s": 60.0}}
    return asyncio.run(ex.execute(state, tools, count)), tools


def done(name, arguments, output) -> ToolResult:
    return ToolResult(call=ToolCall(name=name, arguments=arguments), output=output)


# ----------------------------------------------------------------------------------------------- steps
def test_the_steps_run_in_order_and_only_new_results_are_returned():
    out, tools = run(plan(search(), fact()))
    assert [name for name, _ in tools.calls] == ["search_filings", "get_financial_fact"]
    assert [r.call.name for r in out["tool_results"]] == ["search_filings", "get_financial_fact"]
    assert all(isinstance(r, ToolResult) for r in out["tool_results"]) and out["tool_error"] is None
    earlier = [done("search_filings", {"query": "older"}, [])]
    again, _ = run(plan(fact()), results=earlier)
    assert len(again["tool_results"]) == 1                     # the reducer appends: never the earlier ones


def test_a_refusal_or_no_plan_runs_nothing():
    refusal = Plan(reasoning="x", refuse=True, refusal_rule="R1", steps=[])
    assert run(refusal)[0] == {} and asyncio.run(ex.execute({"plan": None}, FakeTools())) == {}


def test_a_partial_refusal_runs_its_steps():
    out, tools = run(plan(fact("WFC", "total_assets", 2024), rule="R1"))
    assert len(tools.calls) == 1 and out["tool_error"] is None


# ----------------------------------------------------------------------------------------------- guards
def test_the_step_cap_stops_the_round_at_max_steps_for_the_whole_question():
    out, tools = run(plan(fact(year=2023), fact(year=2024), fact(year=2025)), max_steps=2)
    assert len(tools.calls) == 2 and out["stop_reason"] == "step limit reached (2 tool calls)"
    earlier = [done("search_filings", {"query": "a"}, [])]     # earlier rounds count too
    out, tools = run(plan(fact(year=2023), fact(year=2024)), results=earlier, max_steps=2)
    assert len(tools.calls) == 1 and "step limit" in out["stop_reason"]


def test_the_token_budget_stops_before_any_call():
    out, tools = run(plan(fact()), tokens=12_500, max_tokens=12_000)
    assert tools.calls == []
    assert out["stop_reason"] == "token budget exceeded (12500 LLM tokens + 0 tokens of tool output, at most 12000)"


def test_fetched_tool_output_counts_toward_the_token_budget():
    """Everything fetched goes into the composer's prompt: a large search result stops further calls."""
    big = [{"chunk_id": "x", "text": "word " * 900}]
    tools = FakeTools({json.dumps(["search_filings", search().arguments], sort_keys=True): big})
    out, _ = run(plan(search(), fact()), tools, tokens=1_000, max_tokens=1_800)
    assert [n for n, _ in tools.calls] == ["search_filings"]
    assert out["stop_reason"].startswith("token budget exceeded (1000 LLM tokens + 9")
    assert "tokens" not in out                          # the composer's prompt reports them: no double count
    earlier = [done("search_filings", {"query": "a"}, big)]        # earlier rounds' output counts too
    out, tools = run(plan(fact()), results=earlier, tokens=1_000, max_tokens=1_800)
    assert tools.calls == [] and "token budget exceeded" in out["stop_reason"]


def test_the_default_counter_is_core_tokens_with_a_conservative_fallback(monkeypatch):
    import core.tokens
    from core.exceptions import FinSightError
    monkeypatch.setattr(core.tokens, "count_tokens", lambda text: 7)
    assert ex._default_count_tokens("anything") == 7

    def unavailable(text):
        raise FinSightError("no tokenizer", error_type="TokenizerUnavailable")

    monkeypatch.setattr(core.tokens, "count_tokens", unavailable)
    assert ex._default_count_tokens("12345") == 3       # 2 characters per token, rounded up


def test_a_call_that_failed_before_is_not_sent_again_and_the_other_steps_run():
    """A replan that keeps the failing step and adds a fallback: the fallback runs; only a new error replans."""
    failed = done("get_financial_fact", fact(year=2019).arguments, {"error": {"type": "out_of_corpus", "message": "x"}})
    out, tools = run(plan(fact(year=2019), search()), results=[failed])
    assert [n for n, _ in tools.calls] == ["search_filings"] and out["tool_error"] is None
    assert "stop_reason" not in out and out["errors"][0].startswith(
        "executor: not sent again, failed before (out_of_corpus): get_financial_fact")


def test_no_calculation_without_a_figure_that_failed_before():
    failed = done("get_financial_fact", fact(year=2025).arguments, {"error": {"type": "data_missing", "message": "x"}})
    out, tools = run(plan(fact(year=2023), fact(year=2024), fact(year=2025), calculation="growth"), results=[failed])
    assert all(n != "calculate" for n, _ in tools.calls)                # never 2023 -> 2024 in place of 2025
    assert out["errors"][-1].startswith("executor: growth not calculated: no value for get_financial_fact")


def test_a_repeated_call_that_succeeded_is_reused_not_sent_again():
    ok = done("get_financial_fact", fact().arguments, {"value": 391035.0, "unit": "USD millions", "fiscal_year": 2024})
    out, tools = run(plan(fact(), fact(year=2023)), results=[ok])
    assert [a["fiscal_year"] for _, a in tools.calls] == [2023] and len(out["tool_results"]) == 1


def test_the_loop_guard_ignores_argument_order():
    reordered = done("get_financial_fact", dict(reversed(list(fact().arguments.items()))),
                     {"error": {"type": "no_results", "message": "x"}})
    out, tools = run(plan(fact()), results=[reordered])
    assert tools.calls == [] and out["errors"][0].startswith("executor: not sent again, failed before (no_results)")


def test_a_service_unavailable_call_is_retried_once_never_twice():
    """tools/base.py: service_unavailable means 'retry later'; any other failed call is never sent again."""
    down = {"error": {"type": "service_unavailable", "message": "endpoint down"}}
    once = done("search_filings", search().arguments, down)
    out, tools = run(plan(search()), results=[once])
    assert [n for n, _ in tools.calls] == ["search_filings"] and out["tool_error"] is None
    out, tools = run(plan(search()), results=[once, once])                 # sent twice already
    assert tools.calls == [] and "failed before (service_unavailable)" in out["errors"][0]
    out, tools = run(plan(search()), results=[once], max_steps=1)          # a guard blocks the retry
    assert tools.calls == [] and "stop_reason" not in out and "failed before" in out["errors"][0]


# ----------------------------------------------------------------------------------------------- errors
@pytest.mark.parametrize("kind", ["invalid_input", "out_of_corpus", "data_missing", "no_results",
                                  "service_unavailable", "internal_error"])
def test_a_typed_tool_error_stops_the_round_for_a_replan(kind):
    error = {"error": {"type": kind, "message": "x"}}
    tools = FakeTools({json.dumps(["search_filings", search().arguments], sort_keys=True): error})
    out, _ = run(plan(search(), fact()), tools)
    assert out["tool_error"] == kind and len(tools.calls) == 1 and len(out["tool_results"]) == 1


def test_not_reported_is_an_answer_not_a_replan():
    out, tools = run(plan(fact("PG", "revenue", 2024, "Q4"), search()))      # a Q4 question asked as Q4
    assert out["tool_error"] is None and len(tools.calls) == 3
    assert out["tool_results"][0].error_type == "not_reported"
    # W2-07: the statement it was checked on, in the 10-K (Q4 is never filed on its own)
    assert tools.calls[2] == ("search_filings", CHECKED_PG)


CHECKED_PG = {"query": ex.STATEMENT_QUERIES["income"], "ticker": "PG", "fiscal_year": 2024, "fiscal_period": "FY",
              "form": "10-K", "item": "Item 8", "k": ex.CHECKED_PASSAGES}


@pytest.mark.parametrize(("metric", "period", "statement", "form", "item"), [
    ("capex", "FY", "cash_flows", "10-K", "Item 8"),
    ("total_liabilities", "FY", "balance_sheet", "10-K", "Item 8"),
    ("gross_profit", "Q2", "income", "10-Q", "Part I Item 1"),
])
def test_a_not_reported_fact_is_checked_on_its_statement(metric, period, statement, form, item):
    call = ToolCall(name="get_financial_fact", arguments={"ticker": "WFC", "metric": metric, "fiscal_year": 2025,
                                                         "fiscal_period": period, "duration": None})
    checked = ex.checked_section_call(ToolResult(call=call, output={"error": {"type": "not_reported", "message": "x"}}))
    assert checked.name == "search_filings" and checked.arguments["query"] == ex.STATEMENT_QUERIES[statement]
    assert (checked.arguments["form"], checked.arguments["item"], checked.arguments["fiscal_period"]) == (
        form, item, period)
    missing = ToolResult(call=call, output={"error": {"type": "data_missing", "message": "x"}})
    assert ex.checked_section_call(missing) is None
    assert ex.checked_section_call(ToolResult(call=call, output={"value": 1.0})) is None


def test_every_metric_has_its_statement_from_the_concept_map():
    from ingest.xbrl import load_concept_map
    metrics = load_concept_map()
    assert {ex.statement_of(m) for m in metrics} == {"income", "balance_sheet", "cash_flows"}
    assert (ex.statement_of("capex"), ex.statement_of("total_assets"), ex.statement_of("revenue")) == (
        "cash_flows", "balance_sheet", "income")
    assert all(ex.statement_of(m) == "balance_sheet" for m, d in metrics.items() if d.period == "instant")


def test_a_trend_over_unreported_years_is_checked_on_the_newest_statement():
    """Review: compare_periods reports a declared gap as rows with missing = not_reported, not as an error."""
    call = ToolCall(name="compare_periods", arguments={"ticker": "WFC", "metric": "capex", "years": [2023, 2024]})
    rows = [{"ticker": "WFC", "metric": "capex", "fiscal_year": y, "fiscal_period": "FY", "value": None,
             "missing": "not_reported", "notes": ["no capital-expenditure line"]} for y in (2023, 2024)]
    checked = ex.checked_section_call(ToolResult(call=call, output=rows))
    assert checked.arguments["fiscal_year"] == 2024 and checked.arguments["query"] == ex.STATEMENT_QUERIES["cash_flows"]
    reported = [{**rows[0], "missing": None, "value": 1.0}]
    assert ex.checked_section_call(ToolResult(call=call, output=reported)) is None


def test_the_checked_statement_failing_is_no_tool_error_and_is_not_sent_twice():
    tools = FakeTools({json.dumps(["search_filings", CHECKED_PG], sort_keys=True):
                       {"error": {"type": "no_results", "message": "nothing"}}})
    q4 = fact("PG", "revenue", 2024, "Q4")
    out, _ = run(plan(q4, fact("PG", "revenue", 2024, "Q4")), tools)       # the same fact twice: one check
    assert out["tool_error"] is None and [n for n, _ in tools.calls].count("search_filings") == 1


def test_the_checked_statement_never_takes_the_calculations_step():
    """Review: a gap inside a trend still gets its growth; the check runs after the calculation, so at the step cap
    it is the check that is skipped, not the calculation."""
    steps = [fact("AAPL", "revenue", 2023), fact("AAPL", "revenue", 2024, "Q4"), fact("AAPL", "revenue", 2024)]
    out, tools = run(plan(*steps, calculation="growth"), max_steps=4)
    assert [n for n, _ in tools.calls] == ["get_financial_fact"] * 3 + ["calculate"]
    assert any("checked statement was not fetched: step limit" in n for n in out["errors"])


def test_a_calculation_error_still_gets_the_checked_statement():
    """Verification review: the replan after a calculation error must not lose the "not reported" citation."""
    steps = [fact("AAPL", "revenue", 2023), fact("AAPL", "revenue", 2024, "Q4"), fact("AAPL", "revenue", 2024)]
    calc_key = json.dumps(["calculate", {"operation": "growth", "inputs": [383285.0, 391035.0]}], sort_keys=True)
    tools = FakeTools({calc_key: {"error": {"type": "invalid_input", "message": "x"}}})
    out, _ = run(plan(*steps, calculation="growth"), tools)
    assert out["tool_error"] == "invalid_input"
    assert [n for n, _ in tools.calls] == ["get_financial_fact"] * 3 + ["calculate", "search_filings"]


def test_the_checked_statement_respects_the_step_cap():
    out, tools = run(plan(fact("PG", "revenue", 2024, "Q4")), max_steps=1)
    assert [n for n, _ in tools.calls] == ["get_financial_fact"]
    assert any("checked statement was not fetched: step limit" in n for n in out["errors"])


def test_an_exception_from_the_tool_caller_becomes_an_internal_error_result():
    tools = FakeTools({json.dumps(["get_financial_fact", fact().arguments], sort_keys=True): RuntimeError("boom")})
    out, _ = run(plan(fact()), tools)
    assert out["tool_error"] == "internal_error"
    assert out["tool_results"][0].output == {"error": {"type": "internal_error", "message": "RuntimeError: boom"}}


# ----------------------------------------------------------------------------------------------- calculation
def calc_args(out) -> dict:
    calc = [r for r in out["tool_results"] if r.call.name == "calculate"]
    assert len(calc) == 1, out
    return calc[0].call.arguments


def test_a_percent_takes_the_part_then_the_whole_in_step_order():
    out, _ = run(plan(fact("PG", "operating_income", 2023), fact("PG", "revenue", 2023), calculation="percent"))
    assert calc_args(out) == {"operation": "percent", "inputs": [18134.0, 82006.0]}
    assert out["tool_results"][-1].output["result"] == 22.1


def test_growth_is_chronological_whatever_the_step_order():
    out, _ = run(plan(fact("GS", "net_income", 2024), fact("GS", "net_income", 2023), calculation="growth"))
    assert calc_args(out)["inputs"] == [8516.0, 14276.0]


def test_cagr_gets_its_years_and_a_compare_periods_trend_uses_first_and_last():
    trend = PlanStep(tool="compare_periods", purpose="x",
                     arguments={"ticker": "AAPL", "metric": "revenue", "years": [2023, 2024, 2025]})
    out, _ = run(plan(trend, calculation="cagr"))
    assert calc_args(out) == {"operation": "cagr", "inputs": [383285.0, 416161.0], "years": 2.0}
    out, _ = run(plan(trend, calculation="growth"))
    assert calc_args(out)["inputs"] == [383285.0, 416161.0]


def test_a_difference_between_companies_keeps_step_order_and_the_unit():
    out, _ = run(plan(fact("PFE", "net_income", 2023), fact("JNJ", "net_income", 2023), calculation="difference"))
    assert calc_args(out) == {"operation": "difference", "inputs": [2119.0, 35153.0], "unit": "USD millions"}


@pytest.mark.parametrize(("steps", "why"), [
    ((fact(),), "needs two figures, 1 fetched"),
    ((fact(year=2023), fact(year=2024)),
     "the figures are for different periods (AAPL revenue FY 2023, AAPL revenue FY 2024)"),
])
def test_no_calculation_when_the_figures_do_not_fit(steps, why):
    out, tools = run(plan(*steps, calculation="percent"))
    assert all(name != "calculate" for name, _ in tools.calls)
    assert out["errors"] == [f"executor: percent not calculated: {why}"] and out["tool_error"] is None


def test_no_calculation_across_units():
    eps = ToolCall(name="get_financial_fact", arguments=fact(metric="eps_diluted").arguments)
    results = [ToolResult(call=eps, output={"value": 6.08, "unit": "USD per share", "fiscal_year": 2024}),
               ToolResult(call=ToolCall(name="get_financial_fact", arguments=fact().arguments),
                          output={"value": 391035.0, "unit": "USD millions", "fiscal_year": 2024})]
    call, why = ex.calculation_call("ratio", results)
    assert call is None and why == "the figures are in different units (USD millions, USD per share)"


def test_an_undefined_calculation_is_an_answer():
    zero = json.dumps(["get_financial_fact", fact(year=2023).arguments], sort_keys=True)
    tools = FakeTools({zero: {"value": 0.0, "unit": "USD millions", "fiscal_year": 2023}})
    out, _ = run(plan(fact(year=2023), fact(year=2024), calculation="growth"), tools)
    assert out["tool_results"][-1].error_type == "undefined" and out["tool_error"] is None


def test_a_guard_skips_only_the_calculation_and_the_figures_stand():
    out, tools = run(plan(fact(year=2023), fact(year=2024), calculation="growth"), max_steps=2)
    assert all(name != "calculate" for name, _ in tools.calls) and "stop_reason" not in out
    assert out["errors"] == ["executor: growth not calculated: step limit reached (2 tool calls)"]
    assert len(out["tool_results"]) == 2 and out["tool_error"] is None


def test_a_send_back_reuses_everything_and_repeats_no_note():
    first, _ = run(plan(fact(year=2023), fact(year=2024), calculation="growth"))
    again, tools = run(plan(fact(year=2023), fact(year=2024), calculation="growth"), results=first["tool_results"])
    assert tools.calls == [] and again["tool_results"] == []            # a send-back costs no step
    p = plan(fact(), calculation="percent")
    first, _ = run(p)
    state = {"plan": p, "tool_results": first["tool_results"], "errors": first["errors"], "limits": {}}
    assert "errors" not in asyncio.run(ex.execute(state, FakeTools(), words))


def test_a_calculation_skipped_by_the_token_budget_has_a_stable_note():
    p = plan(fact(year=2023), fact(year=2024), calculation="growth")
    first_output = asyncio.run(FakeTools()("get_financial_fact", fact(year=2023).arguments))
    budget = 900 + words(json.dumps(first_output)) + 1                   # room for both facts, not the calculation
    out, tools = run(p, tokens=900, max_tokens=budget)
    assert len(tools.calls) == 2 and out["errors"] == ["executor: growth not calculated: token budget exceeded"]


def test_after_the_composer_earlier_output_is_already_in_the_llm_tokens():
    big = [{"chunk_id": "x", "text": "word " * 900}]
    state = {"plan": plan(fact()), "tool_results": [done("search_filings", {"query": "a"}, big)], "tokens": 1_000,
             "limits": {"max_tokens": 1_800}, "visits": {"composer": 1}}
    tools = FakeTools()
    out = asyncio.run(ex.execute(state, tools, words))
    assert len(tools.calls) == 1 and "stop_reason" not in out                 # not counted twice


def row(value, *, ticker="AAPL", metric="revenue", year=2024, period="FY", unit="USD millions") -> dict:
    return {"value": value, "unit": unit, "ticker": ticker, "metric": metric, "fiscal_year": year,
            "fiscal_period": period, "duration": "12M" if period == "FY" else "3M"}


def fetched(*rows) -> ToolResult:
    first = rows[0]
    return done("get_financial_fact", {"ticker": first["ticker"], "metric": first["metric"],
                                       "fiscal_year": first["fiscal_year"], "fiscal_period": first["fiscal_period"]},
                first)


@pytest.mark.parametrize("operation", ["growth", "cagr"])
def test_growth_never_mixes_companies_or_metrics(operation):
    two_companies = [fetched(row(100.0, ticker="AAA", year=2023)), fetched(row(110.0, ticker="AAA")),
                     fetched(row(500.0, ticker="BBB", year=2023)), fetched(row(900.0, ticker="BBB"))]
    call, why = ex.calculation_call(operation, two_companies)
    assert call is None and why.startswith(f"{operation} takes one series, figures for 2 were fetched (AAA revenue")
    context_metric = [fetched(row(100.0, year=2024)), fetched(row(110.0, year=2025)),
                      fetched(row(20.0, metric="net_income", year=2025))]
    call, why = ex.calculation_call(operation, context_metric)
    assert call is None and "ambiguous" in why


def test_quarters_of_one_year_are_ordered_by_quarter():
    call, _ = ex.calculation_call("growth", [fetched(row(300.0, year=2025, period="Q3")),
                                             fetched(row(100.0, year=2025, period="Q1"))])
    assert call.arguments["inputs"] == [100.0, 300.0]


def test_a_trend_missing_its_last_year_is_not_calculated():
    trend = done("compare_periods", {"ticker": "AAPL", "metric": "revenue", "years": [2023, 2024, 2025]},
                 [row(90.0, year=2023), row(100.0, year=2024), row(None, year=2025)])
    call, why = ex.calculation_call("growth", [trend])
    assert call is None and why == "no value for AAPL revenue FY 2025"
    inner_gap = done("compare_periods", {"ticker": "AAPL", "metric": "revenue", "years": [2023, 2024, 2025]},
                     [row(90.0, year=2023), row(None, year=2024), row(110.0, year=2025)])
    assert ex.calculation_call("growth", [inner_gap])[0].arguments["inputs"] == [90.0, 110.0]


def test_a_percent_takes_revenue_as_the_whole_whichever_comes_first():
    call, _ = ex.calculation_call("percent", [fetched(row(1000.0)), fetched(row(200.0, metric="operating_income"))])
    assert call.arguments["inputs"] == [200.0, 1000.0]
    call, _ = ex.calculation_call("ratio", [fetched(row(1000.0)), fetched(row(200.0, metric="operating_income"))])
    assert call.arguments["inputs"] == [1000.0, 200.0]                    # a ratio keeps the plan's order
    call, _ = ex.calculation_call("percent", [fetched(row(1000.0, ticker="AAA")),
                                              fetched(row(200.0, ticker="BBB", metric="operating_income"))])
    assert call.arguments["inputs"] == [1000.0, 200.0]                    # another company's revenue: plan order


def not_reported(ticker="AAPL", metric="revenue", year=2024, period="FY") -> ToolResult:
    return done("get_financial_fact", {"ticker": ticker, "metric": metric, "fiscal_year": year,
                                       "fiscal_period": period}, {"error": {"type": "not_reported", "message": "x"}})


def test_a_missing_planned_figure_never_leaves_a_guessed_pair():
    """Three planned figures with one not reported must not become a two-figure calculation of the other two."""
    three = [fetched(row(100.0, ticker="AAPL")), not_reported("MSFT"), fetched(row(300.0, ticker="GOOGL"))]
    assert ex.calculation_call("difference", three) == (None, "no value for MSFT revenue FY 2024")
    debt_to_equity = [not_reported("JPM", "long_term_debt"), fetched(row(10.0, ticker="JPM", metric="total_liabilities")),
                      fetched(row(5.0, ticker="JPM", metric="stockholders_equity"))]
    assert ex.calculation_call("ratio", debt_to_equity) == (None, "no value for JPM long_term_debt FY 2024")
    margin = [not_reported("MRK", "operating_income", 2025), fetched(row(65011.0, ticker="MRK", year=2025))]
    assert ex.calculation_call("percent", margin) == (None, "no value for MRK operating_income FY 2025")
    other_series = [fetched(row(100.0, year=2023)), fetched(row(110.0)), not_reported(metric="operating_income")]
    assert ex.calculation_call("growth", other_series) == (None, "no value for AAPL operating_income FY 2024")


def test_a_sum_of_balance_sheet_values_needs_one_date():
    year_end = {**row(1932468.0, ticker="WFC", metric="total_assets", year=2023), "duration": None}
    quarter_end = {**row(1940122.0, ticker="WFC", metric="total_assets", period="Q2"), "duration": None}
    call, why = ex.calculation_call("sum", [fetched(year_end), fetched(quarter_end)])
    assert call is None and why.startswith("balance-sheet values at different dates")
    two_banks = [fetched(year_end), fetched({**year_end, "ticker": "JPM", "value": 3875393.0})]
    assert ex.calculation_call("sum", two_banks)[0].arguments["inputs"] == [1932468.0, 3875393.0]


def test_a_percent_of_three_figures_is_ambiguous():
    call, why = ex.calculation_call("percent", [fetched(row(1.0, metric="gross_profit")),
                                                fetched(row(2.0, metric="operating_income")), fetched(row(9.0))])
    assert call is None and why == "percent takes two figures, 3 fetched (ambiguous)"


def test_a_sum_needs_every_planned_figure():
    quarters = [fetched(row(v, year=2024, period=q)) for v, q in ((119575.0, "Q1"), (90753.0, "Q2"), (85777.0, "Q3"))]
    q4 = done("get_financial_fact", {"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024, "fiscal_period": "Q4"},
              {"error": {"type": "not_reported", "message": "Q4 is not reported separately"}})
    call, why = ex.calculation_call("sum", [*quarters, q4])
    assert call is None and why == "no value for AAPL revenue Q4 2024"
    assert ex.calculation_call("sum", quarters)[0].arguments["inputs"] == [119575.0, 90753.0, 85777.0]
    trend = done("compare_periods", {"ticker": "AAPL", "metric": "revenue", "years": [2022, 2023, 2024]},
                 [row(None, year=2022), row(383285.0, year=2023), row(391035.0, year=2024)])
    assert ex.calculation_call("sum", [trend])[1] == "no value for AAPL revenue FY 2022"


def with_duration(figure_row: dict, duration) -> dict:
    return {**figure_row, "duration": duration}


@pytest.mark.parametrize(("operation", "rows", "why"), [
    ("growth", [with_duration(row(5.0, year=2023, period="Q2"), "6M"), row(3.0, year=2024, period="Q2")],
     "the figures cover different durations (3M, 6M)"),
    ("percent", [row(4.0, metric="gross_profit", period="Q2"), with_duration(row(9.0, period="Q2"), "6M")],
     "the figures cover different durations (3M, 6M)"),
    ("percent", [row(4.0, metric="gross_profit", year=2023), row(9.0, year=2024)],
     "the figures are for different periods (AAPL gross_profit FY 2023, AAPL revenue FY 2024)"),
    ("difference", [row(4.0, ticker="AAA", year=2023), row(9.0, ticker="BBB", year=2024)],
     "the figures are for different periods (AAA revenue FY 2023, BBB revenue FY 2024)"),
    ("growth", [row(1.0, period="Q1"), row(4.0)], "the figures cover different durations (12M, 3M)"),   # Q1 to FY
    ("cagr", [row(1.0, year=2023, period="Q1"), row(2.0, year=2024, period="Q3")],
     "cagr needs the same period each year (Q1, Q3 fetched)"),
])
def test_figures_from_different_periods_or_durations_are_not_combined(operation, rows, why):
    assert ex.calculation_call(operation, [fetched(r) for r in rows]) == (None, why)


def test_a_balance_sheet_instant_pairs_with_a_duration():
    assets = {**row(364980.0, metric="total_assets"), "duration": None}
    call, _ = ex.calculation_call("ratio", [fetched(row(391035.0)), fetched(assets)])
    assert call.arguments["inputs"] == [391035.0, 364980.0]
    year_end = {**row(1932468.0, ticker="WFC", metric="total_assets", year=2023), "duration": None}
    quarter_end = {**row(1940122.0, ticker="WFC", metric="total_assets", period="Q2"), "duration": None}
    call, _ = ex.calculation_call("difference", [fetched(year_end), fetched(quarter_end)])
    assert call.arguments["inputs"] == [1932468.0, 1940122.0]           # instants: year end to a quarter end
    call, _ = ex.calculation_call("cagr", [fetched(row(1.0, year=2022, period="Q2")),
                                           fetched(row(2.0, year=2024, period="Q2"))])
    assert call.arguments["years"] == 2.0


def test_the_same_figure_fetched_twice_counts_once():
    trend = done("compare_periods", {"ticker": "AAPL", "metric": "revenue", "years": [2023, 2024]},
                 [row(383285.0, year=2023), row(391035.0, year=2024)])
    call, _ = ex.calculation_call("growth", [fetched(row(391035.0)), trend])
    assert call.arguments["inputs"] == [383285.0, 391035.0]
    call, why = ex.calculation_call("percent", [fetched(row(391035.0)), fetched(row(391035.0))])
    assert call is None and why == "needs two figures, 1 fetched"


# ----------------------------------------------------------------------------------------------- real plans
def p8_plan(question_id: str) -> Plan:
    for line in P8.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["id"] == question_id:
            return Plan.model_validate(record["plan"])
    raise KeyError(question_id)


@pytest.mark.parametrize(("question_id", "operation", "inputs"), [
    ("dev-B02", "percent", [18134.0, 82006.0]),            # P&G operating margin: operating income / revenue
    ("dev-C01", "growth", [8516.0, 14276.0]),              # GS net earnings fiscal 2023 -> 2024
    ("dev-D01", "difference", [2119.0, 35153.0]),          # Pfizer vs J&J net income
])
def test_plans_from_the_real_planner_run_with_their_calculation(question_id, operation, inputs):
    out, _ = run(p8_plan(question_id))
    assert out["tool_error"] is None and calc_args(out)["operation"] == operation
    assert calc_args(out)["inputs"] == inputs


def test_a_partial_refusal_plan_from_the_real_planner_runs():
    p = p8_plan("pdev-18")                                  # Wells Fargo total assets + "worth owning?" (R1)
    assert p.partial
    out, tools = run(p)
    assert out["tool_error"] is None and tools.calls[0][1]["metric"] == "total_assets"


# ----------------------------------------------------------------------------------------------- graph
def test_in_the_graph_a_tool_error_replans_once_and_reuses_the_good_step():
    bad = fact(year=2019)
    plans = [plan(fact(year=2023), bad), plan(fact(year=2023), fact(year=2024), calculation="growth")]

    async def planner(state):
        return {"plan": plans[(state.get("visits") or {}).get("planner", 0)]}

    async def composer(state):
        return {"draft": AnswerContract(answer="A draft.", verified=False, refused=False)}

    async def critic(state):
        return {"verdict": Verdict()}

    tools = FakeTools()
    graph = g.build_graph(g.Nodes(planner, ex.make_executor(tools, words), composer, critic))
    result = asyncio.run(g.run("How did Apple's revenue change from fiscal 2023 to 2024?", graph=graph,
                               limits=g.Limits()))
    assert result.state["visits"] == {"planner": 2, "executor": 2, "composer": 1, "critic": 1}
    assert [a.get("fiscal_year") for n, a in tools.calls if n == "get_financial_fact"] == [2023, 2019, 2024]
    assert [r.call.name for r in result.state["tool_results"]][-1] == "calculate"


def scripted_graph(plans, tools, verdicts=()):
    async def planner(state):
        return {"plan": plans[min((state.get("visits") or {}).get("planner", 0), len(plans) - 1)]}

    async def composer(state):
        return {"draft": AnswerContract(answer="A draft.", verified=False, refused=False)}

    async def critic(state):
        n = (state.get("visits") or {}).get("critic", 0)
        return {"verdict": verdicts[n] if n < len(verdicts) else Verdict()}

    return g.build_graph(g.Nodes(planner, ex.make_executor(tools, words), composer, critic))


def test_in_the_graph_a_replan_that_repeats_the_failing_step_is_answered_not_found():
    tools = FakeTools()
    same = plan(fact(year=2023), fact(year=2019))                  # 2019: data_missing, planned twice
    result = asyncio.run(g.run("q", graph=scripted_graph([same, same], tools), limits=g.Limits()))
    assert result.state["visits"] == {"planner": 2, "executor": 2, "composer": 1, "critic": 1}
    assert [a["fiscal_year"] for _, a in tools.calls] == [2023, 2019]          # neither call sent twice
    assert not result.state.get("stop_reason")


def test_in_the_graph_a_send_back_after_a_second_error_reaches_the_critic_again():
    tools = FakeTools()
    plans = [plan(fact(year=2019)), plan(fact(year=2023), fact(year=2018))]   # both fail, on different calls
    failed = Verdict(violations=[("1", "a number is not in any tool result")])
    result = asyncio.run(g.run("q", graph=scripted_graph(plans, tools, [failed]), limits=g.Limits()))
    assert result.state["visits"] == {"planner": 2, "executor": 3, "composer": 2, "critic": 2}
    assert len(tools.calls) == 3 and result.answer.verified


def test_the_default_caller_is_the_tool_registry(monkeypatch):
    seen = []

    async def fake_registry(name, arguments):
        seen.append(name)
        return {"result": 1.0}

    import core.tokens
    import tools.registry
    monkeypatch.setattr(tools.registry, "call_tool_async", fake_registry)
    monkeypatch.setattr(core.tokens, "count_tokens", words)
    out = asyncio.run(ex.make_executor()({"plan": plan(search()), "tool_results": [], "limits": {}}))
    assert seen == ["search_filings"] and out["tool_error"] is None
