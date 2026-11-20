"""W2-07: the graph end to end. The real nodes are the default, scripts/ask.py answers one question, and the demo
runner (eval/run_demo.py) writes transcripts. Offline: stub or fake nodes, no endpoint, no data/, no .env."""
from __future__ import annotations

import asyncio
import json

import pytest

from agent import graph as g
from agent.contract import AnswerContract, Value
from agent.state import Plan
from agent.stubs import stub_nodes
from agent.transcript import markdown, run_record, verdict
from eval import run_demo as rd
from eval.validate_dev_set import NEAR_DUPLICATE, other_test_questions, similarity

LIMITS = g.Limits(max_steps=10, max_tokens=12_000, timeout_s=10)
CATEGORIES = {"fact", "ratio", "period_comparison", "multi_company", "qualitative", "multi_turn", "refuse"}


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ----------------------------------------------------------------------------------------------- real nodes
def test_real_nodes_are_the_four_real_factories():
    from agent.nodes import real_nodes
    nodes = real_nodes(llm=object(), call_tool=object())
    assert nodes.planner.__qualname__.startswith("make_planner")
    assert nodes.executor.__qualname__.startswith("make_executor")
    assert nodes.composer.__qualname__.startswith("make_composer")
    assert nodes.critic.__qualname__.startswith("make_critic")


def test_build_graph_defaults_to_the_real_nodes(monkeypatch):
    import agent.nodes
    made = []
    monkeypatch.setattr(agent.nodes, "real_nodes", lambda: made.append(1) or stub_nodes())
    g.build_graph()
    assert made == [1]


def test_trace_demo_uses_the_same_factory():
    from scripts import trace_demo
    nodes = trace_demo.demo_nodes(llm=object())
    assert nodes.composer.__qualname__.startswith("make_composer")


# ----------------------------------------------------------------------------------------------- transcript
def agent_run(answer: AnswerContract, **state) -> g.AgentRun:
    return g.AgentRun(answer=answer, state={"history": [], **state}, seconds=1.234)


@pytest.mark.parametrize(("answer", "expected"), [
    (AnswerContract(answer="x", verified=True, refused=False), "verified"),
    (AnswerContract(answer="x", verified=False, refused=False, notes=["could not verify: y"]), "not verified"),
    (AnswerContract(answer="I can't help with that because it is investment advice. I can share facts from the "
                           "indexed SEC filings instead, for example Apple's total revenue for fiscal 2024.",
                    verified=True, refused=True, refusal_rule="R1"), "refused"),
    (AnswerContract(answer="x", verified=True, refused=False, refusal_rule="R1"), "partial refusal, verified"),
    (AnswerContract(answer="x", verified=False, refused=False, refusal_rule="R1", notes=["could not verify: y"]),
     "partial refusal, not verified"),
])
def test_verdict_names_the_outcome(answer, expected):
    assert verdict(agent_run(answer)) == expected


def test_a_record_carries_the_run_and_reads_as_markdown():
    result = asyncio.run(g.run("What was Apple's total revenue in fiscal 2024?", graph=g.build_graph(stub_nodes()),
                               limits=LIMITS))
    record = run_record(result, "What was Apple's total revenue in fiscal 2024?")
    assert record["verdict"] == "not verified" and record["steps"] == 0 and record["seconds"] >= 0
    assert record["visits"]["planner"] == 1 and record["plan"]["steps"] == []
    text = markdown(record, title="demo-x")
    assert text.startswith("### demo-x") and "**Answer (not verified):**" in text and "**Run:** 0 tool calls" in text
    json.dumps(record)                                                    # serialisable as is


# ----------------------------------------------------------------------------------------------- ask.py
def test_ask_prints_the_answer_and_the_run(capsys):
    from scripts import ask
    assert ask.main(["What was Apple's total revenue in fiscal 2024?"], nodes=stub_nodes(), limits=LIMITS) == 0
    out = capsys.readouterr().out
    assert "**Answer (not verified):**" in out and "tool calls" in out


def test_ask_json_and_history(capsys):
    from scripts import ask
    assert ask.main(["And in fiscal 2023?", "--history", "What was Apple's revenue in fiscal 2024?", "--json"],
                    nodes=stub_nodes(), limits=LIMITS) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["history"] == ["What was Apple's revenue in fiscal 2024?"] and record["question"] == "And in fiscal 2023?"


def test_ask_reports_an_incomplete_env(monkeypatch, capsys):
    import core.settings
    from scripts import ask

    def broken():
        raise ValueError("2 validation errors for Settings\nLLM_BASE_URL\n  Field required")

    monkeypatch.setattr(core.settings, "get_settings", broken)
    assert ask.main(["q"]) == 2
    assert "setup incomplete" in capsys.readouterr().err


# ----------------------------------------------------------------------------------------------- the demo set
def test_the_demo_set_covers_every_category_and_is_well_formed():
    items = rd.load_items()
    ids = [d["id"] for d in items]
    assert len(items) == 10 and len(set(ids)) == 10
    assert {d["category"] for d in items} == CATEGORIES
    assert all(d["expect"] in rd.EXPECTATIONS for d in items)
    assert all((d["refusal_rule"] is not None) == (d["expect"] in ("refuse", "partial")) for d in items)
    for d in items:
        if d["previous_turn_id"]:
            assert ids.index(d["previous_turn_id"]) < ids.index(d["id"])


def test_dev_items_are_the_development_questions_word_for_word():
    dev = rd.dev_values()
    for d in rd.load_items():
        if d["source"] != "new":
            assert d["question"] == dev[d["source"]]["question"], d["id"]


def test_new_items_stay_apart_from_gold():
    """Never the gold set (CLAUDE.md): no company-year gold asks about or cites, no near-duplicate wording of a gold
    or refusal/injection test question (eval/validate_dev_set.py's rules)."""
    gold = jsonl(rd.ROOT / "eval" / "gold_set.jsonl")
    asked = {(t, y) for q in gold for t in q["companies"] for y in q["fiscal_years"]}
    asked |= {(s["ticker"], s["fiscal_year"]) for q in gold for s in q["gold_sources"]}
    references = [q["question"] for q in gold] + other_test_questions()
    for d in rd.load_items():
        if d["source"] != "new":
            continue
        assert not {(t, y) for t in d["companies"] for y in d["fiscal_years"]} & asked, d["id"]
        if d["category"] != "multi_turn":
            assert max(similarity(d["question"], q) for q in references) < NEAR_DUPLICATE, d["id"]


def test_conversations_put_a_follow_up_after_its_first_turn():
    items = [{"id": "a", "previous_turn_id": None}, {"id": "b", "previous_turn_id": None},
             {"id": "c", "previous_turn_id": "a"}, {"id": "d", "previous_turn_id": "c"}]
    assert [[d["id"] for d in chain] for chain in rd.conversations(items)] == [["a", "c", "d"], ["b"]]


# ----------------------------------------------------------------------------------------------- expectations
def record(**kw):
    base = {"verified": True, "refused": False, "refusal_rule": None, "answer": "x", "verdict": "verified",
            "values": []}
    return {**base, **kw}


@pytest.mark.parametrize(("expect", "rule", "rec", "met"), [
    ("answer", None, record(), True),
    ("answer", None, record(verified=False, verdict="not verified"), False),
    ("refuse", "R2", record(refused=True, refusal_rule="R2"), True),
    ("refuse", "R2", record(refused=True, refusal_rule="R1"), False),
    ("partial", "R1", record(refusal_rule="R1"), True),
    ("partial", "R1", record(refusal_rule="R1", verified=False), False),
    ("partial", "R1", record(refused=True, refusal_rule="R1"), False),
    ("not_reported", None, record(answer="Wells Fargo does not report capital expenditures."), True),
    # review: the failure W2-07 fixed must not count as met again
    ("not_reported", None, record(answer="Wells Fargo does not report capital expenditures.", verified=False), False),
    ("not_reported", None, record(answer="It spent $1 million."), False),
])
def test_expectation_met(expect, rule, rec, met):
    assert rd.expectation_met({"expect": expect, "refusal_rule": rule}, rec)[0] is met


def test_a_wrong_development_value_is_not_met():
    item = {"expect": "answer", "refusal_rule": None}
    assert rd.expectation_met(item, record(), dev_value_ok=False)[0] is False
    assert rd.expectation_met(item, record(), dev_value_ok=True)[0] is True
    assert rd.expectation_met(item, record(), dev_value_ok=None)[0] is True


def test_a_zero_tolerance_stays_zero():
    dev = {"dev-X": {"gold_value": 1.0, "tolerance_pct": 0.0}}
    assert rd.dev_value_check({"source": "dev-X"}, record(values=[{"value": 1.04}]), dev) is False
    assert rd.dev_value_check({"source": "dev-X"}, record(values=[{"value": 1.0}]), dev) is True


def test_the_development_value_keeps_its_sign():
    dev = {"dev-X": {"gold_value": 33034.0, "tolerance_pct": 1.0}}
    assert rd.dev_value_check({"source": "dev-X"}, record(values=[{"value": -33034.0}]), dev) is False


def test_a_development_value_is_checked_within_its_tolerance():
    dev = {"dev-X": {"gold_value": 22.1, "tolerance_pct": 1.0}}
    assert rd.dev_value_check({"source": "dev-X"}, record(values=[{"value": 22.1}]), dev) is True
    assert rd.dev_value_check({"source": "dev-X"}, record(values=[{"value": 25.0}]), dev) is False
    assert rd.dev_value_check({"source": "new"}, record(), dev) is None


# ----------------------------------------------------------------------------------------------- the runner
def test_the_runner_writes_transcripts_and_fails_on_unmet_expectations(tmp_path):
    code = rd.main(["--out-dir", str(tmp_path), "--stem", "t"], nodes=stub_nodes(), limits=LIMITS)
    assert code == 1                                                      # the stubs answer nothing
    records = jsonl(tmp_path / "t.jsonl")
    assert [r["id"] for r in records] == [d["id"] for d in rd.load_items()]
    follow_up = next(r for r in records if r["id"] == "demo-07")
    assert follow_up["history"] == ["What was Amazon's net income in fiscal 2023?"]     # memory on its thread
    refusal = next(r for r in records if r["id"] == "demo-08")
    assert refusal["met"] is False and refusal["under_time_limit"] is True
    md = (tmp_path / "t.md").read_text(encoding="utf-8")
    assert md.startswith("# W2-07 demo: ") and "### demo-10" in md and "| dev value |" in md
    assert b"\r\n" not in (tmp_path / "t.md").read_bytes() + (tmp_path / "t.jsonl").read_bytes()


def test_the_runner_takes_default_limits_with_given_nodes(tmp_path):
    assert rd.main(["--out-dir", str(tmp_path), "--stem", "t"], nodes=stub_nodes()) == 1


def test_the_runner_meets_expectations_when_the_nodes_answer(tmp_path):
    """Scripted nodes that refuse, answer partly or answer in full as each demo item expects."""
    items = {d["question"]: d for d in rd.load_items()}

    async def planner(state):
        from agent.state import PlanStep
        d = items[state["question"]]
        refuse = d["expect"] == "refuse"
        steps = [] if refuse else [PlanStep(tool="search_filings", purpose="x", arguments={"query": "x"})]
        return {"plan": Plan(reasoning="x", refuse=refuse, refusal_rule=d["refusal_rule"], steps=steps)}

    async def executor(state):
        return {}

    async def composer(state):
        d = items[state["question"]]
        if d["expect"] == "refuse":
            from agent.contract import refusal_answer
            return {"draft": AnswerContract(answer=refusal_answer(d["refusal_rule"], "Apple's total revenue for "
                                                                  "fiscal 2024"), verified=False, refused=True,
                                            refusal_rule=d["refusal_rule"])}
        text = "It does not report that figure." if d["expect"] == "not_reported" else "An answer."
        dev = rd.dev_values().get(d["source"], {})
        values = [] if dev.get("gold_value") is None else [Value(
            name="x", value=dev["gold_value"], unit="%", ticker=d["companies"][0], fiscal_year=d["fiscal_years"][0],
            fiscal_period="FY", duration=None, period_end="2023-06-30")]
        return {"draft": AnswerContract(answer=text, values=values, verified=False, refused=False,
                                        refusal_rule=d["refusal_rule"])}

    async def critic(state):
        from agent.state import Verdict
        return {"verdict": Verdict(violations=[])}

    code = rd.main(["--out-dir", str(tmp_path), "--stem", "t"], nodes=g.Nodes(planner, executor, composer, critic),
                   limits=LIMITS)
    records = jsonl(tmp_path / "t.jsonl")
    assert code == 0, [(r["id"], r["why"]) for r in records if not r["met"]]


def test_the_budget_stops_live_requests_but_not_cached_ones():
    class Usage:
        calls = 0

    class Inner:
        usage = Usage()

        async def complete(self, messages, **kw):
            self.usage.calls += 1
            return "reply"

    llm = rd.BudgetedLLM(Inner(), max_requests=2)
    assert asyncio.run(llm.complete([])) == "reply" and asyncio.run(llm.complete([])) == "reply"
    with pytest.raises(rd.BudgetExceeded):
        asyncio.run(llm.complete([]))
