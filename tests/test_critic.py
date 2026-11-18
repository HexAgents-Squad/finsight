"""agent/critic.py (W2-05): unit ownership, metric vs question, the node, one send-back. Offline."""
from __future__ import annotations

import asyncio

from agent import critic as cr
from agent import graph as g
from agent.contract import COULD_NOT_VERIFY, AnswerContract
from agent.critic_checks import Violation
from agent.state import Plan, PlanStep, ToolCall, ToolResult, Verdict
from agent.stubs import stub_nodes
from retrieval.entities import Entities
from tests.test_contract import VALUE, spec_example, with_

QUESTION = "What was Microsoft's operating margin in fiscal 2024?"
REVENUE_QUESTION = "What was Apple's total revenue in fiscal 2024?"
MSFT = Entities(tickers=("MSFT",), years=(2024,), periods=("FY",))
LIMITS = g.Limits(max_steps=10, max_tokens=12_000, timeout_s=5)


def draft(**changes) -> AnswerContract:
    return AnswerContract(**with_(spec_example(), verified=False, **changes))


def fact_result(metric="operating_income", value=109433, unit="USD millions", ticker="MSFT",
                year=2024, period="FY") -> ToolResult:
    return ToolResult(
        call=ToolCall(name="get_financial_fact",
                      arguments={"ticker": ticker, "metric": metric, "fiscal_year": year,
                                 "fiscal_period": period}),
        output={"value": value, "unit": unit, "ticker": ticker, "fiscal_year": year,
                "fiscal_period": period, "metric": metric, "duration": "12M",
                "period_end": "2024-06-30"},
    )


def revenue_fact(**kwargs) -> ToolResult:
    return fact_result(metric="revenue", value=245122, **kwargs)


def state(draft_contract=None, results=None, question=QUESTION, entities=MSFT) -> dict:
    return {"question": question, "history": [], "draft": draft_contract if draft_contract is not None
            else draft(), "tool_results": list(results or []), "entities": entities}


def run_nodes(nodes: g.Nodes, question: str = QUESTION) -> g.AgentRun:
    return asyncio.run(g.run(question, (), graph=g.build_graph(nodes), limits=LIMITS))


def plan_ok() -> Plan:
    return Plan(reasoning="test", refuse=False, refusal_rule=None,
                steps=[PlanStep(tool="get_financial_fact",
                                arguments={"ticker": "MSFT", "metric": "revenue", "fiscal_year": 2024},
                                purpose="fetch")])


# ----------------------------------------------------------------------------------------------- units
def test_units_pass_when_draft_matches_tool_filing_units():
    results = [fact_result(), revenue_fact()]
    assert cr.check_units_against_tools(draft(), results) == []


def test_units_flag_wrong_unit_label_same_scale():
    # 109433 USD millions written as plain dollars (Week 1 failure mode shape).
    bad = draft(values=[
        with_(VALUE, name="operating_income", value=109_433_000_000, unit="USD"),
        dict(VALUE),
        with_(VALUE, name="operating_margin", value=44.6, unit="%"),
    ])
    violations = cr.check_units_against_tools(bad, [fact_result()])
    assert rules(violations) == ["units"]
    assert "USD millions" in violations[0].message and "filing units" in violations[0].message


def test_units_flag_different_scaled_number():
    bad = draft(values=[
        with_(VALUE, name="operating_income", value=47_520_000_000, unit="USD"),
        dict(VALUE),
        with_(VALUE, name="operating_margin", value=44.6, unit="%"),
    ])
    violations = cr.check_units_against_tools(bad, [fact_result(value=109433)])
    assert rules(violations) == ["units"]
    assert "common scale" in violations[0].message


def test_units_ignore_derived_values_and_missing_tools():
    assert cr.check_units_against_tools(draft(), []) == []
    only_margin = draft(values=[with_(VALUE, name="operating_margin", value=44.6, unit="%")])
    assert cr.check_units_against_tools(only_margin, [fact_result()]) == []


def test_units_read_compare_periods_rows():
    # Production rows always include missing: None on success (tools/compare_periods.py).
    row = {"value": 245122, "unit": "USD millions", "ticker": "MSFT", "fiscal_year": 2024,
           "fiscal_period": "FY", "metric": "revenue", "missing": None}
    results = [ToolResult(call=ToolCall(name="compare_periods",
                                        arguments={"ticker": "MSFT", "metric": "revenue", "years": [2024]}),
                          output=[row])]
    bad = draft(values=[with_(VALUE, value=245_122_000_000, unit="USD")])
    assert rules(cr.check_units_against_tools(bad, results)) == ["units"]


def test_compare_periods_missing_year_is_skipped():
    rows = [
        {"value": None, "unit": "USD millions", "ticker": "MSFT", "fiscal_year": 2023,
         "fiscal_period": "FY", "metric": "revenue", "missing": "not_reported"},
        {"value": 245122, "unit": "USD millions", "ticker": "MSFT", "fiscal_year": 2024,
         "fiscal_period": "FY", "metric": "revenue", "missing": None},
    ]
    results = [ToolResult(call=ToolCall(name="compare_periods",
                                        arguments={"ticker": "MSFT", "metric": "revenue",
                                                    "years": [2023, 2024]}),
                          output=rows)]
    assert len(cr._fact_rows(results)) == 1
    assert cr._fact_rows(results)[0]["fiscal_year"] == 2024


# ----------------------------------------------------------------------------------------------- metric
def test_metric_passes_when_fetched_metric_is_named():
    results = [ToolResult(call=ToolCall(name="get_financial_fact",
                                        arguments={"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024}),
                          output={"value": 391035, "unit": "USD millions", "ticker": "AAPL",
                                  "fiscal_year": 2024, "fiscal_period": "FY", "metric": "revenue"})]
    assert cr.check_metric_matches_question(REVENUE_QUESTION, results) == []


def test_metric_flags_a_different_mapped_metric():
    results = [fact_result(metric="rd_expense", value=1841, ticker="AAPL")]
    violations = cr.check_metric_matches_question(REVENUE_QUESTION, results)
    assert rules(violations) == ["metric"]
    assert "revenue" in violations[0].message and "rd_expense" in violations[0].message


def test_metric_flags_stock_based_compensation_substitute():
    question = "What was Apple's stock-based compensation in fiscal 2024?"
    results = [fact_result(metric="rd_expense", value=1841, ticker="AAPL")]
    violations = cr.check_metric_matches_question(question, results)
    assert rules(violations) == ["metric"]
    assert "stock-based compensation" in violations[0].message
    assert "search_filings" in violations[0].message


def test_metric_skips_when_question_names_no_mapped_line_item():
    # Operating margin needs operating_income + revenue; the question names neither line item alone.
    assert cr.check_metric_matches_question(QUESTION, [fact_result(), revenue_fact()]) == []


def test_metric_allows_companion_revenue_on_ratio_worded_with_revenues():
    """'revenues' maps to revenue; extras on a % question must not false-reject."""
    question = "What was Apple's R&D as a percentage of revenues?"
    assert "revenue" in cr.named_metrics(question) and "rd_expense" in cr.named_metrics(question)
    results = [fact_result(metric="rd_expense", value=1841, ticker="AAPL"),
               fact_result(metric="revenue", value=391035, ticker="AAPL")]
    assert cr.check_metric_matches_question(question, results) == []


def test_metric_ratio_rejects_when_no_named_metric_was_fetched():
    question = "What was Apple's R&D as a percentage of revenues?"
    results = [fact_result(metric="operating_income", value=1, ticker="AAPL")]
    violations = cr.check_metric_matches_question(question, results)
    assert rules(violations) == ["metric"] and "operating_income" in violations[0].message


def test_named_metrics_reads_concept_map_labels():
    assert "revenue" in cr.named_metrics("What were Apple's net sales in fiscal 2024?")
    assert "revenue" in cr.named_metrics("as a percentage of revenues")
    assert "capex" in cr.named_metrics("How much were capital expenditures in fiscal 2024?")


# ----------------------------------------------------------------------------------------------- critique / node
def rules(violations: list[Violation]) -> list[str]:
    return [v.rule for v in violations]


def test_critique_passes_the_spec_example_with_matching_tools():
    verdict = cr.critique(state(results=[fact_result(), revenue_fact()]))
    assert verdict.passed and verdict.violations == []


def test_critique_includes_unit_and_contract_violations():
    bad = draft(values=[
        with_(VALUE, name="operating_income", value=109_433_000_000, unit="USD"),
        dict(VALUE),
        with_(VALUE, name="operating_margin", value=44.6, unit="%"),
    ])
    verdict = cr.critique(state(draft_contract=bad, results=[fact_result(), revenue_fact()]))
    assert not verdict.passed
    assert "units" in {rule for rule, _ in verdict.violations}


def test_critique_without_draft_fails_closed():
    verdict = cr.critique({"question": QUESTION, "draft": None})
    assert not verdict.passed and verdict.violations[0][0] == "critic"


def test_make_critic_writes_only_verdict():
    node = cr.make_critic()
    update = asyncio.run(node(state(results=[fact_result(), revenue_fact()])))
    assert set(update) == {"verdict"} and isinstance(update["verdict"], Verdict)
    assert update["verdict"].passed


def test_question_scope_from_entities():
    scope = cr.question_scope(state())
    assert scope is not None and scope.tickers == frozenset({"MSFT"}) and scope.fiscal_years == frozenset({2024})
    assert cr.question_scope({"question": QUESTION}) is None


# ----------------------------------------------------------------------------------------------- graph send-back
def test_real_critic_send_back_at_most_once_then_unverified():
    """A failing draft is sent back once; the second failure ends with could not verify (graph bound)."""
    bad = draft(answer="Operating income was $999.0 billion; the margin was 44.6% of revenue of $245.1 billion.")
    good_results = [fact_result(), revenue_fact()]
    calls: list[str] = []

    async def planner(s):
        calls.append("planner")
        return {"plan": plan_ok(), "entities": MSFT}

    async def executor(s):
        calls.append("executor")
        return {"tool_results": good_results}

    async def composer(s):
        calls.append("composer")
        return {"draft": bad}

    result = run_nodes(g.Nodes(planner, executor, composer, cr.make_critic()))
    assert result.state["visits"]["critic"] == 2
    assert result.state["visits"]["executor"] == 2
    assert calls == ["planner", "executor", "composer", "executor", "composer"]
    assert not result.answer.verified
    assert any(n.startswith(COULD_NOT_VERIFY) for n in result.answer.notes)


def test_real_critic_passes_and_finishes_verified():
    async def planner(s):
        return {"plan": plan_ok(), "entities": MSFT}

    async def executor(s):
        return {"tool_results": [fact_result(), revenue_fact()]}

    async def composer(s):
        return {"draft": draft()}

    result = run_nodes(g.Nodes(planner, executor, composer, cr.make_critic()))
    assert result.answer.verified
    assert result.state["visits"] == {"planner": 1, "executor": 1, "composer": 1, "critic": 1}


def test_stub_nodes_use_the_real_critic():
    """The skeleton's critic stub is critique(); a clean refusal still verifies."""
    nodes = stub_nodes()

    async def refuse(s):
        return {"plan": Plan(reasoning="advice", refuse=True, refusal_rule="R1", steps=[]),
                "entities": Entities(tickers=())}

    result = run_nodes(g.Nodes(refuse, nodes.executor, nodes.composer, nodes.critic),
                       question="Should I buy Apple?")
    assert result.answer.verified and result.answer.refused
