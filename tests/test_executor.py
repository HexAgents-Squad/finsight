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
            if arguments.get("fiscal_period") == "Q4":
                return {"error": {"type": "not_reported", "message": "Q4 is not reported separately"}}
            value = VALUES.get((arguments["ticker"], arguments["metric"], arguments["fiscal_year"]))
            if value is None:
                return {"error": {"type": "data_missing", "message": "no fact"}}
            return {"value": value, "unit": "USD millions", "ticker": arguments["ticker"],
                    "metric": arguments["metric"], "fiscal_year": arguments["fiscal_year"], "fiscal_period": "FY"}
        if name == "compare_periods":
            return [{"value": VALUES.get((arguments["ticker"], arguments["metric"], y)), "unit": "USD millions",
                     "fiscal_year": y, "ticker": arguments["ticker"], "metric": arguments["metric"]}
                    for y in arguments["years"]]
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


def test_a_repeated_call_that_failed_before_stops_the_run():
    failed = done("get_financial_fact", fact(year=2019).arguments, {"error": {"type": "out_of_corpus", "message": "x"}})
    out, tools = run(plan(fact(year=2019)), results=[failed])
    assert tools.calls == [] and out["stop_reason"].startswith("repeated identical tool call: get_financial_fact")
    assert "failed before (out_of_corpus)" in out["stop_reason"]


def test_a_repeated_call_that_succeeded_is_reused_not_sent_again():
    ok = done("get_financial_fact", fact().arguments, {"value": 391035.0, "unit": "USD millions", "fiscal_year": 2024})
    out, tools = run(plan(fact(), fact(year=2023)), results=[ok])
    assert [a["fiscal_year"] for _, a in tools.calls] == [2023] and len(out["tool_results"]) == 1


def test_the_loop_guard_ignores_argument_order():
    reordered = done("get_financial_fact", dict(reversed(list(fact().arguments.items()))),
                     {"error": {"type": "no_results", "message": "x"}})
    out, tools = run(plan(fact()), results=[reordered])
    assert tools.calls == [] and "repeated identical tool call" in out["stop_reason"]


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
    assert out["tool_error"] is None and len(tools.calls) == 2
    assert out["tool_results"][0].error_type == "not_reported"


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
    ((fact(year=2023), fact(year=2024), fact(year=2025)), "percent takes two figures, 3 fetched (ambiguous)"),
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


def test_the_calculation_is_capped_and_reused_like_any_call():
    out, tools = run(plan(fact(year=2023), fact(year=2024), calculation="growth"), max_steps=2)
    assert all(name != "calculate" for name, _ in tools.calls) and "step limit" in out["stop_reason"]
    first, _ = run(plan(fact(year=2023), fact(year=2024), calculation="growth"))
    again, tools = run(plan(fact(year=2023), fact(year=2024), calculation="growth"), results=first["tool_results"])
    assert tools.calls == [] and again["tool_results"] == []            # a send-back costs no step


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
