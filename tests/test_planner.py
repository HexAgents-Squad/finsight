"""W2-02: the planner node, its prompt, the guided call, the entities in search steps, and the tuning set and runner.

Offline: a fake LLM client returns scripted replies (no endpoint, no key)."""
from __future__ import annotations

import asyncio
import json
import re

import pytest

from agent import graph as g
from agent import planner as pl
from agent.contract import COULD_NOT_VERIFY
from agent.guided import RETRY_INSTRUCTION, ask_guided
from agent.state import Plan, PlanStep, ToolCall, ToolResult, plan_json_schema
from agent.stubs import stub_nodes
from core.exceptions import FinSightError
from core.llm import Reply
from eval import run_planner as rp
from eval.validate_dev_set import NEAR_DUPLICATE, similarity
from ingest.edgar import load_scope
from retrieval.entities import Entities, from_turns

REFUSAL = {"reasoning": "Asks whether to buy a stock: investment advice.", "refuse": True, "refusal_rule": "R1",
           "steps": [], "calculation": None}
ANSWER = {"reasoning": "Revenue is a mapped metric.", "refuse": False, "refusal_rule": None, "calculation": None,
          "steps": [{"tool": "get_financial_fact", "purpose": "the revenue",
                     "arguments": {"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024, "fiscal_period": "FY",
                                   "duration": None}}]}
SEARCH_ONLY = {"reasoning": "A qualitative question.", "refuse": False, "refusal_rule": None, "calculation": None,
               "steps": [{"tool": "search_filings", "purpose": "what drove it",
                          "arguments": {"query": "revenue drivers", "ticker": None, "fiscal_year": None,
                                        "fiscal_period": None, "form": None, "item": None, "k": 5}}]}


class FakeLLM:
    """complete() returns the scripted replies in order (the last repeats); an exception in the script is raised."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def complete(self, messages, *, temperature=0.0, model=None, sample=0, seed=None, response_format=None):
        self.calls.append({"messages": messages, "response_format": response_format, "temperature": temperature})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return Reply(text, text, 300, 40, False, 0, 0.8)


def plan_node(llm, question, history=(), **state):
    return asyncio.run(pl.make_planner(llm)({"question": question, "history": list(history), **state}))


def protected_questions() -> list[tuple[str, str]]:
    out = []
    for name in ("gold_set", "refusal_tests", "injection_tests", "dev_set"):
        for line in (rp.EVAL_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out.append((d["id"], d.get("question") or d.get("user_question") or ""))
    return out


# ----------------------------------------------------------------------------------------------- prompt
def test_the_prompt_is_filled_from_the_config_and_the_tools():
    prompt = pl.system_prompt()
    assert "$" not in prompt                                            # every placeholder filled
    for company in load_scope().companies:
        assert f"({company.ticker})" in prompt
    assert "10-K annual reports for fiscal years 2023, 2024, 2025" in prompt and "fiscal 2025 Q1, Q2, Q3" in prompt
    assert "- get_financial_fact(ticker, metric, fiscal_year; optional: fiscal_period, duration):" in prompt
    assert "revenue" in prompt and "net_income" in prompt and "calculate(" not in prompt
    assert prompt.index("Refusal rules") < prompt.index("Tools a plan can list")       # refusal first


def test_the_prompt_is_pinned():
    """Re-tune on the development questions (eval/run_planner.py), then update PROMPT_SHA256 (and the version)."""
    assert pl.prompt_sha256() == pl.PROMPT_SHA256


def examples() -> list[str]:
    return re.findall(r'^- "(.+?)" ->', pl.PROMPT_FILE.read_text(encoding="utf-8"), re.M)


def test_the_prompt_examples_carry_no_figures():
    """ADR-018: S1's examples helped refusals but S1 copied a real figure from one, so these have none."""
    assert len(examples()) == 10
    for example in examples():
        numbers = [n for n in re.findall(r"\d+(?:[.,]\d+)*", example) if not re.fullmatch(r"20\d\d", n)]
        assert not numbers and "$" not in example, example


def test_the_prompt_examples_are_far_from_every_test_and_tuning_question():
    """Not gold-v1, the refusal or injection tests, the development set or the tuning set, reworded."""
    others = protected_questions() + [(q["id"], q["question"]) for q in rp.load_questions("pdev")]
    for example in examples():
        score, ref = max((similarity(example, q), i) for i, q in others)
        assert score < 0.5, (example, ref, score)


# ----------------------------------------------------------------------------------------------- tuning set
def test_the_tuning_set_is_well_formed_and_covers_every_rule():
    rows = [json.loads(line) for line in rp.PLANNER_DEV_FILE.read_text(encoding="utf-8").splitlines()]
    assert len({r["id"] for r in rows}) == len(rows) and all(r["id"].startswith("pdev-") for r in rows)
    assert all(r["written_by"] and r["rationale"] for r in rows)
    for r in rows:
        assert set(r) == {"id", "kind", "question", "history", "expected_refused", "expected_rule", "rationale",
                          "written_by"}
        if r["kind"] == "refusal":
            assert r["expected_refused"] is True and r["expected_rule"]
        elif r["kind"] == "partial":
            assert r["expected_refused"] is False and r["expected_rule"]
        else:
            assert r["kind"] == "control" and r["expected_refused"] is False and r["expected_rule"] is None
    refusals = {f"R{n}": sum(r["kind"] == "refusal" and r["expected_rule"] == f"R{n}" for r in rows) for n in range(1, 8)}
    assert min(refusals.values()) >= 2, refusals
    assert sum(r["kind"] == "partial" for r in rows) >= 3 and sum(r["kind"] == "control" for r in rows) >= 12
    assert any(r["history"] for r in rows)                              # a follow-up


def test_the_tuning_set_is_separate_from_gold_and_the_test_sets():
    """The development set's rule (eval/validate_dev_set.py): no reworded gold or test question."""
    protected = protected_questions()
    for q in rp.load_questions("pdev"):
        score, ref = max((similarity(q["question"], p), i) for i, p in protected)
        assert score < NEAR_DUPLICATE, (q["id"], ref, score)


# ----------------------------------------------------------------------------------------------- node
def test_a_refusal_plan():
    llm = FakeLLM(REFUSAL)
    out = plan_node(llm, "Should I buy Apple stock?")
    assert out["plan"].refuse and out["plan"].refusal_rule == "R1" and out["plan"].steps == []
    assert out["entities"].tickers == ("AAPL",) and out["tokens"] == 340 and "errors" not in out


def test_the_call_uses_strict_guided_decoding_of_the_plan_schema():
    llm = FakeLLM(ANSWER)
    plan_node(llm, "What was Apple's revenue in fiscal 2024?")
    call = llm.calls[0]
    assert call["temperature"] == 0.0
    assert call["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "Plan", "schema": plan_json_schema(), "strict": True}}
    system, user = call["messages"]
    assert system == {"role": "system", "content": pl.system_prompt()}
    assert "Question: What was Apple's revenue in fiscal 2024?" in user["content"]
    assert "- companies (indexed tickers): AAPL" in user["content"] and "- fiscal years: 2024" in user["content"]


def test_unindexed_periods_are_pointed_out():
    text = pl.entities_text(from_turns(["What was Microsoft's revenue in fiscal 2026?"]))
    assert "- not indexed: fiscal 2026 FY" in text
    assert "not indexed" not in pl.entities_text(from_turns(["What was Meta's revenue in Q2 2025?"]))
    assert "- not indexed: fiscal 2024 Q2" in pl.entities_text(from_turns(["Meta's revenue in Q2 2024?"]))


def test_the_earlier_turns_are_in_the_message_and_the_entities():
    llm = FakeLLM(ANSWER)
    out = plan_node(llm, "And in fiscal 2023?", history=["What was Apple's revenue in fiscal 2024?"])
    user = llm.calls[0]["messages"][1]["content"]
    assert "Earlier questions in this conversation (oldest first):\n- What was Apple's revenue in fiscal 2024?" in user
    assert out["entities"].tickers == ("AAPL",) and out["entities"].years == (2023,)


def test_an_invalid_reply_is_asked_again_once_with_the_error():
    bad = {**REFUSAL, "steps": ANSWER["steps"]}                         # a refusal with steps breaks the Plan rules
    llm = FakeLLM(bad, REFUSAL)
    out = plan_node(llm, "Should I buy Apple stock?")
    assert out["plan"].refuse and len(llm.calls) == 2 and out["tokens"] == 680
    retry = llm.calls[1]["messages"][-1]["content"]
    assert retry.startswith(RETRY_INSTRUCTION.split("{")[0]) and "no tool steps" in retry
    assert out["errors"] == ["planner: valid plan after 1 JSON retry"]


def test_a_reply_still_invalid_after_the_retry_gives_no_plan():
    llm = FakeLLM("not json at all")
    out = plan_node(llm, "Should I buy Apple stock?")
    assert out["plan"] is None and len(llm.calls) == 2
    assert out["errors"][0].startswith("planner: JSONValidationError")


def test_a_failed_call_gives_no_plan():
    llm = FakeLLM(FinSightError("cannot reach the endpoint", error_type="ConnectionError"))
    out = plan_node(llm, "Should I buy Apple stock?")
    assert out["plan"] is None and out["errors"] == ["planner: ConnectionError: cannot reach the endpoint"]


def test_a_replan_shows_the_previous_steps_and_the_errors():
    previous = Plan.model_validate(ANSWER)
    error = ToolResult(call=ToolCall(name="get_financial_fact", arguments={"ticker": "AAPL", "metric": "revenue",
                                                                            "fiscal_year": 2024}),
                       output={"error": {"type": "not_reported", "message": "no value for the period"}})
    llm = FakeLLM(SEARCH_ONLY)
    plan_node(llm, "What was Apple's revenue in fiscal 2024?", plan=previous, tool_results=[error])
    user = llm.calls[0]["messages"][1]["content"]
    assert "Your previous plan failed." in user and '"tool": "get_financial_fact"' in user
    assert "-> not_reported: no value for the period" in user
    first = FakeLLM(ANSWER)
    plan_node(first, "What was Apple's revenue in fiscal 2024?")
    assert "previous plan" not in first.calls[0]["messages"][1]["content"]


def test_ask_guided_reports_cost_and_cache():
    reply = asyncio.run(ask_guided(FakeLLM(REFUSAL), [{"role": "user", "content": "x"}], Plan, plan_json_schema()))
    assert reply.parsed.refuse and reply.tokens == 340 and reply.seconds == 0.8 and not reply.cached


# ----------------------------------------------------------------------------------------------- entities
def search(**args) -> PlanStep:
    base = {"query": "revenue drivers", "ticker": None, "fiscal_year": None, "fiscal_period": None, "form": None,
            "item": None, "k": 5}
    return PlanStep(tool="search_filings", arguments={**base, **args}, purpose="find it")


def with_steps(*steps) -> Plan:
    return Plan(reasoning="x", refuse=False, refusal_rule=None, steps=list(steps))


def test_search_steps_get_the_questions_company_year_and_period():
    out = pl.apply_entities(with_steps(search()), Entities(("AAPL",), (2024,), ("Q2",)))[0]
    assert out.steps[0].arguments["ticker"] == "AAPL" and out.steps[0].arguments["fiscal_year"] == 2024
    assert out.steps[0].arguments["fiscal_period"] is None          # fiscal 2024 Q2 is not indexed: not set
    out = pl.apply_entities(with_steps(search()), Entities(("META",), (2025,), ("Q2",)))[0]
    assert out.steps[0].arguments["fiscal_period"] == "Q2"


def test_several_companies_give_one_search_each_and_set_values_stay():
    out, cut = pl.apply_entities(with_steps(search(), search(ticker="MSFT", query="other")),
                                 Entities(("ABBV", "MRK"), (2024,), ()))
    assert cut == 0
    assert [(s.arguments["ticker"], s.arguments["query"]) for s in out.steps] == [
        ("ABBV", "revenue drivers"), ("MRK", "revenue drivers"), ("MSFT", "other")]


def test_entities_leave_refusals_and_other_tools_alone_and_keep_the_cap():
    refusal = Plan.model_validate(REFUSAL)
    assert pl.apply_entities(refusal, Entities(("AAPL",), (2024,), ()))[0] is refusal
    fact = Plan.model_validate(ANSWER)
    assert pl.apply_entities(fact, Entities(("MSFT",), (2023,), ()))[0].steps == fact.steps
    many, cut = pl.apply_entities(with_steps(*[search(query=f"q{n}") for n in range(5)]),
                                  Entities(("AAPL", "MSFT", "KO"), (2024,), ()))
    assert len(many.steps) == 10 and cut == 5                           # 15 searches, 5 over the cap (noted)
    _, notes = pl.finalize(with_steps(*[search(query=f"q{n}") for n in range(5)]),
                           Entities(("AAPL", "MSFT", "KO"), (2024,), ()), "Compare their fiscal 2024 risks.")
    assert notes == ["planner: 5 search step(s) beyond the 10-step cap were dropped"]


def test_a_plan_decoded_with_nulls_validates_for_every_tool():
    """What strict guided decoding writes (every argument present, null where optional) is a valid plan."""
    plan = Plan.model_validate({**SEARCH_ONLY, "steps": SEARCH_ONLY["steps"] + ANSWER["steps"] + [
        {"tool": "compare_periods", "purpose": "trend",
         "arguments": {"ticker": "AAPL", "metric": "revenue", "years": [2023, 2024]}}], "calculation": "growth"})
    assert [s.tool for s in plan.steps] == ["search_filings", "get_financial_fact", "compare_periods"]


# ----------------------------------------------------------------------------------------------- graph
def test_the_planner_in_the_graph_refuses_through_composer_and_critic():
    nodes = stub_nodes()
    graph = g.build_graph(g.Nodes(pl.make_planner(FakeLLM(REFUSAL)), nodes.executor, nodes.composer, nodes.critic))
    result = asyncio.run(g.run("Should I buy Apple stock?", graph=graph, limits=g.Limits()))
    assert result.answer.refused and result.answer.refusal_rule == "R1" and result.answer.verified
    assert result.state["visits"] == {"planner": 1, "composer": 1, "critic": 1} and result.state["tokens"] == 340


def test_a_planner_failure_in_the_graph_gives_up_with_the_reason():
    nodes = stub_nodes()
    llm = FakeLLM(FinSightError("down", error_type="ConnectionError"))
    graph = g.build_graph(g.Nodes(pl.make_planner(llm), nodes.executor, nodes.composer, nodes.critic))
    result = asyncio.run(g.run("What was Apple's revenue in fiscal 2024?", graph=graph, limits=g.Limits()))
    assert result.answer.notes == [COULD_NOT_VERIFY + "the planner returned no plan"]
    assert result.state["errors"] == ["planner: ConnectionError: down"]


# ----------------------------------------------------------------------------------------------- runner
def test_the_runner_reads_only_development_questions():
    questions = rp.load_questions("all")
    assert len(questions) == 15 + len(rp.load_questions("pdev"))
    by_id = {q["id"]: q for q in questions}
    assert by_id["dev-G01"]["expected_refused"] and by_id["dev-G01"]["expected_rule"] == "R2"
    assert not by_id["dev-G02"]["expected_refused"] and by_id["dev-G02"]["expected_rule"] is None
    assert by_id["dev-F02"]["history"] == [by_id["dev-A03"]["question"]]


def test_the_runner_refuses_any_other_question_file(tmp_path, monkeypatch):
    gold_like = tmp_path / "planner_dev.jsonl"
    gold_like.write_text(json.dumps({"id": "A01", "kind": "control", "question": "q", "history": [],
                                     "expected_refused": False, "expected_rule": None}) + "\n", encoding="utf-8")
    monkeypatch.setitem(rp.SOURCES, "pdev", (gold_like,))
    with pytest.raises(SystemExit, match="not development questions"):
        rp.load_questions("pdev")


def test_the_runner_scores_and_summarises(tmp_path):
    questions = [{"id": "pdev-x1", "kind": "refusal", "question": "Should I buy Apple stock?", "history": [],
                  "expected_refused": True, "expected_rule": "R1"},
                 {"id": "pdev-x2", "kind": "control", "question": "What drove Apple's revenue in fiscal 2024?",
                  "history": [], "expected_refused": False, "expected_rule": None}]
    records = asyncio.run(rp.run(FakeLLM(REFUSAL, SEARCH_ONLY), questions))
    assert [r["decision_correct"] for r in records] == [True, True]
    assert records[1]["entity_filters"] is True and records[1]["tools"] == ["search_filings"]
    runs, summary = rp.write(records, "T1", "fake-model", runs_dir=tmp_path, summary_file=tmp_path / "s.md")
    rows = [json.loads(line) for line in runs.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["prompt_sha256"] == pl.prompt_sha256() and rows[0]["model"] == "fake-model"
    text = summary.read_text(encoding="utf-8")
    assert "| refusal decision correct (refused or not, and the rule) | 2/2 = 100% |" in text
    assert "Wrong decisions" not in text


def test_a_wrong_decision_is_listed():
    q = {"id": "pdev-x3", "kind": "control", "question": "What was Apple's revenue in fiscal 2024?", "history": [],
         "expected_refused": False, "expected_rule": None}
    records = asyncio.run(rp.run(FakeLLM(REFUSAL), [q]))
    assert not records[0]["decision_correct"]
    assert "- pdev-x3 (control): expected refused=False rule=None, got refused=True rule=R1" in rp.summary(
        records, "T2", "m")


# ----------------------------------------------------------------------------------------------- scope check
@pytest.mark.parametrize(("question", "rule"), [
    ("What were Procter & Gamble's net sales in fiscal 2021?", "R3"),          # reported, not indexed
    ("What was Alphabet's net income in the first quarter of fiscal 2024?", "R3"),
    ("Will Coca-Cola's net operating revenues grow in fiscal 2026?", "R2"),     # R2 before R3
])
def test_the_scope_check_refuses_a_plan_for_unindexed_periods(question, rule):
    plan, note = pl.scope_check(Plan.model_validate(ANSWER), from_turns([question]), question)
    assert plan.refuse and plan.refusal_rule == rule and plan.steps == [] and rule in note


@pytest.mark.parametrize("question", [
    "What was Apple's revenue in fiscal 2024?",                               # indexed
    "What were Amazon's net sales in fiscal 2024, and what will they be in fiscal 2026?",   # one indexed year
    "What drove Apple's results?",                                             # no year named
])
def test_the_scope_check_leaves_answerable_plans_alone(question):
    answer = Plan.model_validate(ANSWER)
    assert pl.scope_check(answer, from_turns([question]), question) == (answer, None)


def test_the_scope_check_keeps_the_models_own_refusal():
    refusal = Plan.model_validate({**REFUSAL, "refusal_rule": "R1"})
    question = "Should I have bought Pfizer in fiscal 2019?"
    assert pl.scope_check(refusal, from_turns([question]), question) == (refusal, None)


def test_the_node_notes_a_scope_check_refusal():
    out = plan_node(FakeLLM(ANSWER), "What were Procter & Gamble's net sales in fiscal 2021?")
    assert out["plan"].refusal_rule == "R3"
    assert out["errors"] == ["planner: scope check refused the plan (R3: fiscal 2021 FY not indexed)"]


def test_no_indexed_company_points_at_r3():
    assert "not in the indexed list is not indexed: R3" in pl.entities_text(from_turns(["What was Tesla's revenue?"]))
    assert "R3" not in pl.entities_text(from_turns(["What was Apple's revenue in fiscal 2024?"]))


# ----------------------------------------------------------------------------------------------- arguments
def fact(period="FY", duration="12M", metric="revenue") -> PlanStep:
    return PlanStep(tool="get_financial_fact", purpose="x", arguments={
        "ticker": "META", "metric": metric, "fiscal_year": 2025, "fiscal_period": period, "duration": duration})


@pytest.mark.parametrize(("question", "period", "duration"), [
    ("What was Meta's revenue in Q2 fiscal 2025?", "Q2", None),                     # the tool picks 3M
    ("What was Meta's revenue for the six months ended June 30, 2025?", "Q2", "6M"),
    ("What was Meta's year-to-date revenue in Q3 fiscal 2025?", "Q3", "9M"),
    ("How much cash did Meta hold at the end of fiscal 2025?", "FY", None),          # point in time: the tool's
])
def test_the_tool_chooses_the_duration_unless_year_to_date_is_asked(question, period, duration):
    plan = pl.normalize_arguments(with_steps(fact(period)), question, from_turns([question]))
    assert plan.steps[0].arguments["duration"] == duration


def test_a_guessed_item_is_dropped_unless_the_question_names_one():
    guessed = with_steps(search(item="Item 1"))
    assert pl.normalize_arguments(guessed, "What drove Microsoft's revenue?", Entities()).steps[0].arguments["item"] is None
    named = pl.normalize_arguments(with_steps(search(item="Item 7")), "What does Item 7 of Apple's 10-K say?",
                                    Entities())
    assert named.steps[0].arguments["item"] == "Item 7"


@pytest.mark.parametrize(("question", "planned", "expected"), [
    ("What was Merck's operating margin in fiscal 2025?", "ratio", "percent"),      # the critic expects percent
    ("What was NVIDIA's gross margin in fiscal 2023?", None, "percent"),
    ("By what percentage did Citigroup's net income change from fiscal 2023 to 2024?", None, "growth"),
    ("What was Apple's current ratio in fiscal 2024?", "ratio", "ratio"),           # ratio or percent: kept
    ("What was Apple's revenue in fiscal 2024?", None, None),                       # nothing expected
])
def test_the_calculation_follows_what_the_critic_expects(question, planned, expected):
    plan = with_steps(fact()).model_copy(update={"calculation": planned})
    assert pl.align_calculation(plan, question).calculation == expected


def test_finalize_keeps_a_refusal_untouched():
    refusal = Plan.model_validate(REFUSAL)
    assert pl.finalize(refusal, from_turns(["Should I buy Apple?"]), "Should I buy Apple?") == (refusal, [])


# ----------------------------------------------------------------------------------------------- R3 recheck
FOLLOW_UP = ("What about fiscal 2025?", ["What was Pfizer's revenue in fiscal 2023?"])
R3 = {**REFUSAL, "reasoning": "Fiscal 2025 is not indexed.", "refusal_rule": "R3"}
PFE_2025 = {**ANSWER, "steps": [{"tool": "get_financial_fact", "purpose": "the revenue", "arguments": {
    "ticker": "PFE", "metric": "revenue", "fiscal_year": 2025, "fiscal_period": "FY", "duration": None}}]}


def test_an_r3_refusal_that_contradicts_the_scope_is_asked_again():
    llm = FakeLLM(R3, PFE_2025)
    out = plan_node(llm, FOLLOW_UP[0], history=FOLLOW_UP[1])
    assert not out["plan"].refuse and out["plan"].steps[0].arguments["ticker"] == "PFE" and len(llm.calls) == 2
    correction = llm.calls[1]["messages"][-1]["content"]
    assert "PFE, fiscal 2025 FY ARE in the corpus" in correction and "R3 does not apply" in correction
    assert llm.calls[1]["messages"][-2]["role"] == "assistant"                # the refusal it is asked to revise
    assert out["errors"] == ["planner: an R3 refusal contradicted the indexed scope; asked again"]
    assert out["tokens"] == 680


def test_a_refusal_it_keeps_after_the_recheck_stands():
    out = plan_node(FakeLLM(R3), FOLLOW_UP[0], history=FOLLOW_UP[1])
    assert out["plan"].refuse and out["plan"].refusal_rule == "R3"


@pytest.mark.parametrize("question", [
    "What was Tesla's revenue in fiscal 2024?",                 # no indexed company: R3 is right
    "What were Procter & Gamble's net sales in fiscal 2021?",   # an unindexed year
    "What is JPMorgan's share price today in fiscal 2025?",     # live price
    "What was Apple's revenue?",                                # no year named
])
def test_a_consistent_r3_refusal_is_not_rechecked(question):
    assert pl.scope_contradiction(Plan.model_validate(R3), from_turns([question]), question) is None


def test_other_rules_are_never_rechecked():
    question = "Should I buy Apple stock after its fiscal 2024 results?"
    assert pl.scope_contradiction(Plan.model_validate(REFUSAL), from_turns([question]), question) is None


def test_free_text_arguments_are_narrowed_for_decoding():
    """Measured: an unconstrained `item` let the model write 1,024 tokens of 'or cyber attack or ...'."""
    from core.spec import ITEMS_OF_FORM
    from ingest.xbrl import load_concept_map
    variants = {v["properties"]["tool"]["const"]: v["properties"]["arguments"]["properties"]
                for v in plan_json_schema()["$defs"]["PlanStep"]["anyOf"]}
    item = variants["search_filings"]["item"]["anyOf"]
    assert set(item[0]["enum"]) == ITEMS_OF_FORM["10-K"] | ITEMS_OF_FORM["10-Q"] and item[1] == {"type": "null"}
    assert variants["search_filings"]["query"]["maxLength"] == 300
    for tool in ("get_financial_fact", "compare_periods"):
        assert set(variants[tool]["metric"]["enum"]) == set(load_concept_map())
        assert variants[tool]["ticker"]["pattern"] == "^[A-Z]{1,5}$"


# ----------------------------------------------------------------------------------------------- second review
def test_every_search_returns_the_tools_top_five():
    """Strict decoding hides k's default: the model wrote k=1 on every search (P2-P5)."""
    plan = pl.normalize_arguments(with_steps(search(k=1)), "What drove Apple's revenue?", Entities())
    assert plan.steps[0].arguments["k"] == pl.SEARCH_K == 5


@pytest.mark.parametrize(("question", "planned", "expected"), [
    ("What was AbbVie's revenue in fiscal 2025?", "Q4", "FY"),          # P5 pdev-19 planned Q4
    ("What was Merck's operating income in fiscal 2025?", "Q1", "FY"),  # P5 pdev-25 planned Q1
    ("What was Meta's revenue in Q2 fiscal 2025?", "FY", "Q2"),
    ("What were P&G's net sales in the fourth quarter of fiscal 2024?", "FY", "Q4"),   # the tool answers Q4
])
def test_a_fact_is_fetched_for_the_questions_period(question, planned, expected):
    plan = pl.normalize_arguments(with_steps(fact(planned)), question, from_turns([question]))
    assert plan.steps[0].arguments["fiscal_period"] == expected


def test_a_search_form_follows_the_period():
    question = "What did Meta say about AI in the second quarter of fiscal 2025?"
    step = search(ticker="META", fiscal_year=2025, form="10-K")
    plan = pl.normalize_arguments(with_steps(step), question, from_turns([question]))
    assert plan.steps[0].arguments["fiscal_period"] == "Q2" and plan.steps[0].arguments["form"] == "10-Q"
    annual = pl.normalize_arguments(with_steps(search(ticker="KO", fiscal_year=2024, form="10-Q")),
                                    "What did Coca-Cola say about sugar in fiscal 2024?", Entities(("KO",), (2024,), ()))
    assert annual.steps[0].arguments["fiscal_period"] == "FY" and annual.steps[0].arguments["form"] == "10-K"


def test_redundant_compare_periods_steps_are_dropped():
    one_year = PlanStep(tool="compare_periods", purpose="x",
                        arguments={"ticker": "PG", "metric": "operating_income", "years": [2023]})
    repeat = PlanStep(tool="compare_periods", purpose="x",
                      arguments={"ticker": "META", "metric": "revenue", "years": [2025]})
    trend = PlanStep(tool="compare_periods", purpose="x",
                     arguments={"ticker": "META", "metric": "revenue", "years": [2023, 2024, 2025]})
    plan = pl.normalize_arguments(with_steps(fact(), one_year, repeat, trend), "Meta's revenue trend?", Entities())
    assert [s.tool for s in plan.steps] == ["get_financial_fact", "compare_periods"]
    assert plan.steps[1].arguments["years"] == [2023, 2024, 2025]


@pytest.mark.parametrize(("question", "years"), [
    ("What does Microsoft say about its 2030 carbon-negative commitment?", ()),
    ("What does Microsoft say about its 2030 goal in its fiscal 2024 10-K?", (2024,)),
    ("How did Walmart describe the 2008 financial crisis in fiscal 2024?", (2024,)),
    ("What was Pfizer's revenue in fiscal 2019?", (2019,)),             # written as fiscal: kept (R3)
    ("What will Apple's revenue be in FY2030?", (2030,)),
    ("What was Apple's revenue in 2026?", (2026,)),                     # bare, near the indexed years
])
def test_only_fiscal_years_are_planned_with(question, years):
    assert pl.planning_entities(from_turns([question]), question).years == years


def test_a_non_fiscal_year_neither_refuses_nor_searches():
    question = "What does Microsoft say about its 2030 carbon-negative commitment?"
    out = plan_node(FakeLLM(SEARCH_ONLY), question)
    assert not out["plan"].refuse and out["entities"].years == ()
    assert [s.arguments["fiscal_year"] for s in out["plan"].steps] == [None]


@pytest.mark.parametrize("question", [
    "What expected credit losses did JPMorgan report in fiscal 2021?",
    "Which R&D projects did Merck describe in fiscal 2021?",
    "What was Pfizer's projected benefit obligation in fiscal 2021?",
])
def test_accounting_terms_are_not_forecasts(question):
    plan, _ = pl.scope_check(Plan.model_validate(ANSWER), from_turns([question]), question)
    assert plan.refusal_rule == "R3"


@pytest.mark.parametrize("question", [
    "Will Apple's fiscal 2025 revenue be restated?",
    "Do you expect Apple's fiscal 2025 results to be revised?",
])
def test_the_r3_recheck_skips_questions_about_the_future(question):
    assert pl.scope_contradiction(Plan.model_validate(R3), from_turns([question]), question) is None


def test_the_r3_correction_keeps_an_unindexed_second_company_refused():
    question = "Compare Apple's and Tesla's revenue in fiscal 2024."
    correction = pl.scope_contradiction(Plan.model_validate(R3), from_turns([question]), question)
    assert "not in the indexed list is still R3" in correction and "not out of scope" not in correction
    # "not a forecast" is stated only because the recheck never runs on a question with forecast words (above)


def test_the_pin_covers_the_templates_and_the_schema(monkeypatch):
    pinned = pl.prompt_sha256()
    monkeypatch.setattr(pl, "REPLAN_TEXT", pl.REPLAN_TEXT + " ")
    assert pl.prompt_sha256() != pinned
    monkeypatch.undo()
    monkeypatch.setattr(pl, "SEARCH_K", 4)
    assert pl.prompt_sha256() != pinned
