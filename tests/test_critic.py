"""agent/critic.py (W2-05): unit ownership, metric vs question, the node, send-back. Offline."""
from __future__ import annotations

import asyncio

from agent import critic as cr
from agent import graph as g
from agent.contract import COULD_NOT_VERIFY, AnswerContract, Citation, Value
from agent.critic_checks import Violation
from agent.state import Plan, PlanStep, ToolCall, ToolResult
from agent.stubs import stub_nodes
from retrieval.entities import Entities
from tests.test_contract import FILING, VALUE, XBRL, spec_example, with_

QUESTION = "What was Microsoft's operating margin in fiscal 2024?"
REVENUE_QUESTION = "What was Apple's total revenue in fiscal 2024?"
MSFT = Entities(tickers=("MSFT",), years=(2024,), periods=("FY",))
AAPL = Entities(tickers=("AAPL",), years=(2024,), periods=("FY",))
LIMITS = g.Limits(max_steps=10, max_tokens=12_000, timeout_s=5)


def draft(**changes) -> AnswerContract:
    return AnswerContract(**with_(spec_example(), verified=False, **changes))


def fact_result(metric="operating_income", value=109433, unit="USD millions", ticker="MSFT",
                year=2024, period="FY", duration="12M", label=None, concept=None) -> ToolResult:
    out = {"value": value, "unit": unit, "ticker": ticker, "fiscal_year": year, "fiscal_period": period,
           "metric": metric, "duration": duration, "period_end": "2024-06-30",
           "label": label or metric.replace("_", " "), "concept": concept or f"us-gaap:{metric}"}
    return ToolResult(
        call=ToolCall(name="get_financial_fact",
                      arguments={"ticker": ticker, "metric": metric, "fiscal_year": year,
                                 "fiscal_period": period}),
        output=out,
    )


def revenue_fact(**kwargs) -> ToolResult:
    return fact_result(metric="revenue", value=245122, label="total revenues", **kwargs)


def apple_revenue_draft(value=391035, unit="USD millions", name="revenue") -> AnswerContract:
    period = {"ticker": "AAPL", "fiscal_year": 2024, "fiscal_period": "FY"}
    return AnswerContract(
        answer=f"Apple's total revenue for fiscal 2024 was {value} {unit}.",
        values=[Value(name=name, value=value, unit=unit, **period, duration="12M",
                      period_end="2024-09-28")],
        citations=[Citation(source_type="xbrl", form="10-K", **period,
                            concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                            accession_no="0000320193-24-000123",
                            url="https://www.sec.gov/Archives/edgar/data/320193/")],
        calculations=[], verified=False, refused=False, refusal_rule=None, notes=[],
    )


def state(draft_contract=None, results=None, question=QUESTION, entities=MSFT, history=None) -> dict:
    return {"question": question, "history": list(history or []),
            "draft": draft_contract if draft_contract is not None else draft(),
            "tool_results": list(results or []), "entities": entities}


def run_nodes(nodes: g.Nodes, question: str = QUESTION, history=()) -> g.AgentRun:
    return asyncio.run(g.run(question, history, graph=g.build_graph(nodes), limits=LIMITS))


def plan_ok(ticker="MSFT", metric="revenue") -> Plan:
    return Plan(reasoning="test", refuse=False, refusal_rule=None,
                steps=[PlanStep(tool="get_financial_fact",
                                arguments={"ticker": ticker, "metric": metric, "fiscal_year": 2024},
                                purpose="fetch")])


def rules(violations: list[Violation]) -> list[str]:
    return [v.rule for v in violations]


# ----------------------------------------------------------------------------------------------- units
def test_units_pass_when_draft_matches_tool_filing_units():
    assert cr.check_units_against_tools(draft(), [fact_result(), revenue_fact()]) == []


def test_units_flag_wrong_unit_label_same_scale():
    bad = draft(values=[
        with_(VALUE, name="operating_income", value=109_433_000_000, unit="USD"),
        dict(VALUE),
        with_(VALUE, name="operating_margin", value=44.6, unit="%"),
    ])
    violations = cr.check_units_against_tools(bad, [fact_result()])
    assert rules(violations) == ["units"]
    assert "filing units" in violations[0].message


def test_units_flag_value_not_in_filing_units():
    bad = draft(values=[
        with_(VALUE, name="operating_income", value=47_520_000_000, unit="USD"),
        dict(VALUE),
        with_(VALUE, name="operating_margin", value=44.6, unit="%"),
    ])
    violations = cr.check_units_against_tools(bad, [fact_result(value=109433)])
    assert rules(violations) == ["units"]
    assert "filing units" in violations[0].message


def test_units_label_match_alone_catches_wrong_scale():
    """Spec §7.5: company line-item names match the tool label (no peer-row fallback)."""
    bad = apple_revenue_draft(value=391_035_000_000, unit="USD", name="Total net sales")
    tool = fact_result(metric="revenue", value=391035, unit="USD millions", ticker="AAPL",
                       label="Total net sales",
                       concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax")
    assert cr._name_matches_row(bad.values[0], tool.output)
    assert cr._matching_fact(bad.values[0], cr._fact_rows([tool])) is tool.output
    violations = cr.check_units_against_tools(bad, [tool])
    assert rules(violations) == ["units"] and "filing units" in violations[0].message


def test_units_filing_text_value_next_to_unrelated_fact_is_not_flagged():
    """A passage figure next to a same-period revenue fact must not be compared to that fact."""
    period = {"ticker": "AAPL", "fiscal_year": 2024, "fiscal_period": "FY"}
    contract = AnswerContract(
        answer="Revenue was 391035 USD millions; stock-based compensation was 11688 USD millions.",
        values=[
            Value(name="revenue", value=391035, unit="USD millions", **period, duration="12M",
                  period_end="2024-09-28"),
            Value(name="stock-based compensation", value=11688, unit="USD millions", **period,
                  duration="12M", period_end="2024-09-28"),
        ],
        citations=[
            Citation(source_type="xbrl", form="10-K", **period,
                     concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                     accession_no="0000320193-24-000123",
                     url="https://www.sec.gov/Archives/edgar/data/320193/"),
            Citation(**with_(FILING, **period, chunk_id="AAPL_10-K_2024_Item8_C3_0001",
                             url="https://www.sec.gov/Archives/edgar/data/320193/a.htm")),
        ],
        calculations=[], verified=False, refused=False, refusal_rule=None, notes=[],
    )
    tool = fact_result(metric="revenue", value=391035, ticker="AAPL",
                       concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax")
    assert cr.check_units_against_tools(contract, [tool]) == []


def test_units_distinguish_quarter_duration():
    q2 = {"ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "Q2"}
    draft_ok = AnswerContract(
        answer="Revenue was 95359 for the three months and 219659 year to date.",
        values=[
            Value(name="revenue", value=95359, unit="USD millions", **q2, duration="3M",
                  period_end="2025-03-29"),
            Value(name="revenue", value=219659, unit="USD millions", **q2, duration="6M",
                  period_end="2025-03-29"),
        ],
        citations=[Citation(source_type="xbrl", form="10-Q", **q2, concept="us-gaap:Revenues",
                            accession_no="0000320193-25-000001",
                            url="https://www.sec.gov/Archives/edgar/data/320193/")],
        calculations=[], verified=False, refused=False, refusal_rule=None, notes=[],
    )
    results = [
        fact_result(metric="revenue", value=95359, ticker="AAPL", year=2025, period="Q2", duration="3M"),
        fact_result(metric="revenue", value=219659, ticker="AAPL", year=2025, period="Q2", duration="6M"),
    ]
    assert cr.check_units_against_tools(draft_ok, results) == []
    # Wrong duration pairing must not silently reuse the other row's figure.
    swapped = draft_ok.model_copy(update={"values": [
        Value(name="revenue", value=219659, unit="USD millions", **q2, duration="3M", period_end="2025-03-29"),
        Value(name="revenue", value=95359, unit="USD millions", **q2, duration="6M", period_end="2025-03-29"),
    ]})
    assert rules(cr.check_units_against_tools(swapped, results)) == ["units", "units"]


def test_units_matching_requires_ticker_year_and_period():
    row = fact_result()
    assert cr._matching_fact(Value(**with_(VALUE, name="operating_income", value=109433)),
                             cr._fact_rows([row])) is not None
    assert cr._matching_fact(Value(**with_(VALUE, name="operating_income", value=109433, ticker="AAPL")),
                             cr._fact_rows([row])) is None
    assert cr._matching_fact(Value(**with_(VALUE, name="operating_income", value=109433, fiscal_year=2023)),
                             cr._fact_rows([row])) is None
    assert cr._matching_fact(Value(**with_(VALUE, name="operating_income", value=109433,
                                           fiscal_period="Q1", duration="3M")),
                             cr._fact_rows([row])) is None


def test_units_read_compare_periods_rows_with_missing_none():
    row = {"value": 245122, "unit": "USD millions", "ticker": "MSFT", "fiscal_year": 2024,
           "fiscal_period": "FY", "metric": "revenue", "missing": None, "duration": "12M",
           "label": "total revenues"}
    results = [ToolResult(call=ToolCall(name="compare_periods",
                                        arguments={"ticker": "MSFT", "metric": "revenue", "years": [2024]}),
                          output=[row])]
    bad = draft(values=[with_(VALUE, value=245_122_000_000, unit="USD")])
    assert rules(cr.check_units_against_tools(bad, results)) == ["units"]


def test_units_ignore_derived_values_and_missing_tools():
    assert cr.check_units_against_tools(draft(), []) == []
    only_margin = draft(values=[with_(VALUE, name="operating_margin", value=44.6, unit="%")])
    assert cr.check_units_against_tools(only_margin, [fact_result()]) == []


# ----------------------------------------------------------------------------------------------- metric phrases
def test_named_metrics_reads_concept_map_labels_and_plurals():
    assert "revenue" in cr.named_metrics("What were Apple's net sales in fiscal 2024?")
    assert "revenue" in cr.named_metrics("as a percentage of revenues")
    assert "capex" in cr.named_metrics("How much were capital expenditures in fiscal 2024?")


def test_cash_flow_wording_maps_to_operating_cash_flow_not_cash():
    for question in (
        "How much cash flow from operations did Apple generate in fiscal 2024?",
        "What were cash flows from operating activities?",
        "How much cash did Apple generate from operating activities?",
        "How much cash provided by operating activities did Apple report?",
    ):
        named = cr.named_metrics(question)
        assert "operating_cash_flow" in named, question
        assert "cash" not in named, question
    assert "cash" not in cr.named_metrics("What was the cash dividend in fiscal 2024?")
    assert "cash" not in cr.named_metrics("What was free cash flow in fiscal 2024?")
    assert "cash" in cr.named_metrics("How much cash did Apple hold at fiscal year end 2024?")
    assert "cash" in cr.named_metrics("What was Apple's cash and cash equivalents in fiscal 2024?")
    # Must not cross a clause into "operating income".
    crossed = "How much cash was used in investing activities and what was operating income?"
    assert "operating_cash_flow" not in cr.named_metrics(crossed)


def test_named_metrics_does_not_take_cost_of_revenue_or_noncontrolling_income():
    assert "revenue" not in cr.named_metrics("What was cost of revenue in fiscal 2024?")
    assert "revenue" not in cr.named_metrics("What was noninterest revenue?")
    assert cr.named_metrics("net income attributable to noncontrolling interests") == frozenset()
    assert "net_income" in cr.named_metrics("What was net income in fiscal 2024?")


def test_named_metrics_keeps_full_comma_labels_not_net_of_interest_expense():
    # BAC label is one phrase; splitting on commas would falsely map "net of interest expense" → revenue.
    assert "revenue" not in cr.named_metrics("What is net of interest expense?")


def test_longest_phrase_wins_over_sales_alone():
    assert "revenue" in cr.named_metrics("What were total net sales?")
    assert cr.named_metrics("What about sales?") == frozenset()


# ----------------------------------------------------------------------------------------------- metric vs draft
def test_metric_passes_when_draft_uses_the_named_metric():
    contract = apple_revenue_draft()
    tool = fact_result(metric="revenue", value=391035, ticker="AAPL",
                       concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax")
    assert cr.check_metric_matches_question(REVENUE_QUESTION, [tool], contract=contract) == []


def test_metric_flags_wrong_metric_used_in_the_draft():
    # Draft uses rd_expense while the question names revenue.
    bad = apple_revenue_draft(name="rd_expense", value=1841)
    tool = fact_result(metric="rd_expense", value=1841, ticker="AAPL")
    violations = cr.check_metric_matches_question(REVENUE_QUESTION, [tool], contract=bad)
    assert rules(violations) == ["metric"]
    assert "revenue" in violations[0].message and "rd_expense" in violations[0].message


def test_metric_ignores_stale_wrong_fetch_when_draft_uses_the_right_one():
    """tool_results is append-only: an earlier rd_expense must not block a corrected revenue draft."""
    contract = apple_revenue_draft()
    stale = fact_result(metric="rd_expense", value=1841, ticker="AAPL")
    good = fact_result(metric="revenue", value=391035, ticker="AAPL",
                       concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax")
    assert cr.check_metric_matches_question(REVENUE_QUESTION, [stale, good], contract=contract) == []


def test_metric_flags_stock_based_compensation_substitute_in_draft():
    question = "What was Apple's stock-based compensation in fiscal 2024?"
    bad = apple_revenue_draft(name="rd_expense", value=1841)
    tool = fact_result(metric="rd_expense", value=1841, ticker="AAPL")
    violations = cr.check_metric_matches_question(question, [tool], contract=bad)
    assert rules(violations) == ["metric"] and "search_filings" in violations[0].message


def test_metric_search_only_passes_when_draft_uses_only_search_filings():
    question = "What was Apple's stock-based compensation in fiscal 2024?"
    period = {"ticker": "AAPL", "fiscal_year": 2024, "fiscal_period": "FY"}
    contract = AnswerContract(
        answer="Stock-based compensation was 11,688 USD millions.",
        values=[Value(name="stock_based_compensation", value=11688, unit="USD millions", **period,
                      duration="12M", period_end="2024-09-28")],
        citations=[Citation(**with_(FILING, **period, chunk_id="AAPL_10-K_2024_Item8_C3_0001",
                                    url="https://www.sec.gov/Archives/edgar/data/320193/a.htm"))],
        calculations=[], verified=False, refused=False, refusal_rule=None, notes=[],
    )
    stale = fact_result(metric="rd_expense", value=1841, ticker="AAPL")
    search = ToolResult(call=ToolCall(name="search_filings", arguments={"query": "stock-based"}),
                        output={"results": [{"chunk_id": "AAPL_10-K_2024_Item8_C3_0001"}]})
    assert cr.check_metric_matches_question(question, [stale, search], contract=contract) == []


def test_metric_allows_companion_on_ratio_and_rejects_wholly_wrong_set():
    question = "What was Apple's R&D as a percentage of revenues?"
    assert "revenue" in cr.named_metrics(question) and "rd_expense" in cr.named_metrics(question)
    period = {"ticker": "AAPL", "fiscal_year": 2024, "fiscal_period": "FY"}
    contract = AnswerContract(
        answer="R&D was 4.7% of revenues.",
        values=[
            Value(name="rd_expense", value=1841, unit="USD millions", **period, duration="12M",
                  period_end="2024-09-28"),
            Value(name="revenue", value=391035, unit="USD millions", **period, duration="12M",
                  period_end="2024-09-28"),
            Value(name="rd_margin", value=0.5, unit="%", **period, duration="12M", period_end="2024-09-28"),
        ],
        citations=[Citation(**with_(XBRL, **period, concept="us-gaap:ResearchAndDevelopmentExpense",
                                   accession_no="0000320193-24-000123",
                                   url="https://www.sec.gov/Archives/edgar/data/320193/"))],
        calculations=[], verified=False, refused=False, refusal_rule=None, notes=[],
    )
    good = [fact_result(metric="rd_expense", value=1841, ticker="AAPL"),
            fact_result(metric="revenue", value=391035, ticker="AAPL")]
    assert cr.check_metric_matches_question(question, good, contract=contract) == []
    wrong = [fact_result(metric="operating_income", value=1, ticker="AAPL")]
    bad = contract.model_copy(update={"values": [
        Value(name="operating_income", value=1, unit="USD millions", **period, duration="12M",
              period_end="2024-09-28")]})
    assert rules(cr.check_metric_matches_question(question, wrong, contract=bad)) == ["metric"]


def test_metric_follow_up_uses_history_line_item():
    follow = "And in fiscal 2023?"
    assert cr.named_metrics(follow) == frozenset()
    assert cr.metric_question_text(follow, [REVENUE_QUESTION]) == REVENUE_QUESTION
    # Wrong metric in the draft for a follow-up that inherits "revenue" from history.
    bad = apple_revenue_draft(name="rd_expense", value=1841)
    tool = fact_result(metric="rd_expense", value=1841, ticker="AAPL")
    violations = cr.check_metric_matches_question(follow, [tool], contract=bad, history=[REVENUE_QUESTION])
    assert rules(violations) == ["metric"]


def test_metric_operating_cash_flow_not_flagged_as_cash():
    question = "How much cash flow from operations did Apple generate in fiscal 2024?"
    period = {"ticker": "AAPL", "fiscal_year": 2024, "fiscal_period": "FY"}
    contract = AnswerContract(
        answer="Operating cash flow was 118_254 USD millions.",
        values=[Value(name="operating_cash_flow", value=118254, unit="USD millions", **period,
                      duration="12M", period_end="2024-09-28")],
        citations=[Citation(**with_(XBRL, **period, concept="us-gaap:NetCashProvidedByUsedInOperatingActivities",
                                   accession_no="0000320193-24-000123",
                                   url="https://www.sec.gov/Archives/edgar/data/320193/"))],
        calculations=[], verified=False, refused=False, refusal_rule=None, notes=[],
    )
    tool = fact_result(metric="operating_cash_flow", value=118254, ticker="AAPL",
                       concept="us-gaap:NetCashProvidedByUsedInOperatingActivities")
    assert cr.check_metric_matches_question(question, [tool], contract=contract) == []
    # Failing: bare cash question with operating_cash_flow in the draft.
    cash_q = "How much cash did Apple hold at year end fiscal 2024?"
    assert rules(cr.check_metric_matches_question(cash_q, [tool], contract=contract)) == ["metric"]


# ----------------------------------------------------------------------------------------------- critique / wiring
def test_critique_passes_the_spec_example_with_matching_tools():
    assert cr.critique(state(results=[fact_result(), revenue_fact()])).passed


def test_critique_includes_unit_and_metric_checks():
    bad = draft(values=[
        with_(VALUE, name="operating_income", value=109_433_000_000, unit="USD"),
        dict(VALUE),
        with_(VALUE, name="operating_margin", value=44.6, unit="%"),
    ])
    verdict = cr.critique(state(draft_contract=bad, results=[fact_result(), revenue_fact()]))
    assert not verdict.passed and "units" in {rule for rule, _ in verdict.violations}


def test_critique_without_draft_says_no_draft_to_check():
    verdict = cr.critique({"question": QUESTION, "draft": None})
    assert verdict.violations == [("critic", "no draft to check")]


def test_critique_uses_question_scope_from_entities():
    # Without entities, scope checks are skipped; with MSFT scope, an AAPL citation fails invariant 4.
    foreign = draft(citations=[with_(XBRL, ticker="AAPL"), with_(FILING, ticker="AAPL",
                                                                  chunk_id="AAPL_10-K_2024_Item8_C3_0012")])
    verdict = cr.critique(state(draft_contract=foreign, results=[fact_result(), revenue_fact()],
                                entities=MSFT))
    assert not verdict.passed
    assert any(rule == "4" for rule, _ in verdict.violations)


def test_make_critic_writes_only_verdict():
    update = asyncio.run(cr.make_critic()(state(results=[fact_result(), revenue_fact()])))
    assert set(update) == {"verdict"} and update["verdict"].passed


def test_stub_nodes_wire_make_critic():
    nodes = stub_nodes()
    assert nodes.critic.__name__ == "critic"  # make_critic's inner function


# ----------------------------------------------------------------------------------------------- graph send-back
def test_send_back_clears_metric_violation_when_round_two_fixes_the_fetch():
    """Round 1 used rd_expense; round 2 fetches revenue and the draft uses it → verified."""
    wrong = apple_revenue_draft(name="rd_expense", value=1841)
    right = apple_revenue_draft()
    stale = fact_result(metric="rd_expense", value=1841, ticker="AAPL")
    good = fact_result(metric="revenue", value=391035, ticker="AAPL",
                       concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax")
    rounds = {"n": 0}

    async def planner(s):
        return {"plan": plan_ok("AAPL", "revenue"), "entities": AAPL}

    async def executor(s):
        rounds["n"] += 1
        return {"tool_results": [stale] if rounds["n"] == 1 else [good]}

    async def composer(s):
        # After the first critic failure, the second draft uses the corrected revenue figure.
        used = wrong if rounds["n"] == 1 else right
        return {"draft": used}

    result = run_nodes(g.Nodes(planner, executor, composer, cr.make_critic()), question=REVENUE_QUESTION)
    assert result.state["visits"]["critic"] == 2
    assert result.answer.verified
    # Stale rd_expense remains in the log; draft-scoped metric check still passes.
    assert any(r.call.arguments.get("metric") == "rd_expense" for r in result.state["tool_results"])
    assert any(r.call.arguments.get("metric") == "revenue" for r in result.state["tool_results"])


def test_real_critic_send_back_at_most_once_then_unverified():
    bad = draft(answer="Operating income was $999.0 billion; the margin was 44.6% of revenue of $245.1 billion.")

    async def planner(s):
        return {"plan": plan_ok(), "entities": MSFT}

    async def executor(s):
        return {"tool_results": [fact_result(), revenue_fact()]}

    async def composer(s):
        return {"draft": bad}

    result = run_nodes(g.Nodes(planner, executor, composer, cr.make_critic()))
    assert result.state["visits"]["critic"] == 2
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
    nodes = stub_nodes()

    async def refuse(s):
        return {"plan": Plan(reasoning="advice", refuse=True, refusal_rule="R1", steps=[]),
                "entities": Entities(tickers=())}

    result = run_nodes(g.Nodes(refuse, nodes.executor, nodes.composer, nodes.critic),
                       question="Should I buy Apple?")
    assert result.answer.verified and result.answer.refused
