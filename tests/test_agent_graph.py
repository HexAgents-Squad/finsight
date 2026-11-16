"""The W2-01 agent skeleton: state models, every conditional edge of agent/graph.py, the loop bounds and limits.

Offline: scripted nodes and the stubs only (no LLM, no tools)."""
from __future__ import annotations

import asyncio
import typing

import pytest
from pydantic import ValidationError

from agent import graph as g
from agent.contract import COULD_NOT_VERIFY, AnswerContract, refusal_answer
from agent.critic_checks import Violation, check_contract
from agent.state import Plan, PlanStep, ToolCall, ToolName, ToolResult, Verdict, plan_json_schema
from agent.stubs import REFUSAL_SUGGESTION, SKELETON_ANSWER, stub_nodes
from tools.registry import TOOLS

QUESTION = "What was Apple's total revenue in fiscal 2024?"
LIMITS = g.Limits(max_steps=10, max_tokens=12_000, timeout_s=5)
SEARCH = ToolCall(name="search_filings", arguments={"query": "total revenue"})


def run(nodes: g.Nodes, limits: g.Limits = LIMITS, question: str = QUESTION, history=()) -> g.AgentRun:
    return asyncio.run(g.run(question, history, graph=g.build_graph(nodes, limits), limits=limits))


def plan(refuse: bool = False, rule: str | None = None, steps: int = 1) -> Plan:
    return Plan(reasoning="test", refuse=refuse, refusal_rule=rule,
                steps=[PlanStep(tool="search_filings", arguments={"query": "revenue"}, purpose="find it")] * steps)


def ok_result() -> ToolResult:
    return ToolResult(call=SEARCH, output={"results": []})


def error_result(kind: str = "no_results") -> ToolResult:
    return ToolResult(call=SEARCH, output={"error": {"type": kind, "message": "nothing found"}})


def verified_draft() -> AnswerContract:
    return AnswerContract(answer="A draft.", verified=False, refused=False)


class Script:
    """Scripted nodes: each node pops its next update from a list (the last one repeats) and records its calls."""

    def __init__(self, planner=None, executor=None, composer=None, critic=None):
        self.updates = {"planner": planner or [{"plan": plan()}], "executor": executor or [{}],
                        "composer": composer or [{"draft": verified_draft()}],
                        "critic": critic or [{"verdict": Verdict()}]}
        self.calls: list[str] = []
        self.seen: dict[str, list[dict]] = {name: [] for name in self.updates}

    def node(self, name: str):
        async def step(state):
            self.calls.append(name)
            self.seen[name].append(dict(state))
            queue = self.updates[name]
            return dict(queue.pop(0) if len(queue) > 1 else queue[0])
        return step

    def nodes(self) -> g.Nodes:
        return g.Nodes(**{name: self.node(name) for name in self.updates})


# ----------------------------------------------------------------------------------------------- state models
def test_plan_refusal_rules():
    assert plan(refuse=True, rule="R1", steps=0).refusal_rule == "R1"
    with pytest.raises(ValidationError, match="names its rule"):
        plan(refuse=True, rule=None, steps=0)
    with pytest.raises(ValidationError, match="only when refuse"):
        plan(refuse=False, rule="R1")
    with pytest.raises(ValidationError, match="no tool steps"):
        plan(refuse=True, rule="R2", steps=1)


def test_plan_is_strict_and_capped():
    with pytest.raises(ValidationError):
        Plan.model_validate({"reasoning": "x", "refuse": "false", "refusal_rule": None, "steps": []})
    with pytest.raises(ValidationError):
        Plan.model_validate({"reasoning": "x", "refuse": False, "refusal_rule": None, "steps": [], "extra": 1})
    with pytest.raises(ValidationError):
        Plan.model_validate({"reasoning": " ", "refuse": False, "refusal_rule": None, "steps": []})
    with pytest.raises(ValidationError):
        plan(steps=11)
    with pytest.raises(ValidationError):
        PlanStep(tool="delete_files", arguments={}, purpose="x")


def test_plan_schema_for_guided_decoding():
    schema = plan_json_schema()
    assert list(schema["properties"]) == ["reasoning", "refuse", "refusal_rule", "steps"]     # reasoning first
    assert schema["required"] == list(schema["properties"])
    assert schema["additionalProperties"] is False
    step = schema["$defs"]["PlanStep"]
    assert step["required"] == ["tool", "arguments", "purpose"]
    assert set(step["properties"]["tool"]["enum"]) == set(TOOLS)
    assert "title" not in schema and "default" not in str(schema)


def test_tool_names_match_the_registry():
    assert set(typing.get_args(ToolName)) == set(TOOLS)


def test_tool_result_error_type():
    assert ok_result().error_type is None
    assert error_result("out_of_corpus").error_type == "out_of_corpus"
    assert ToolResult(call=SEARCH, output={"error": {"message": "?"}}).error_type == "internal_error"
    assert ToolResult(call=SEARCH, output=[1, 2]).error_type is None


def test_verdict_leaves_out_invariant_seven():
    verdict = Verdict.of([Violation("7", "verified is false"), Violation("citations", "none")])
    assert verdict.violations == [("citations", "none")]
    assert not verdict.passed and Verdict.of([Violation("7", "x")]).passed
    assert verdict.as_violations() == [Violation("citations", "none")]


# ----------------------------------------------------------------------------------------------- skeleton
def test_skeleton_runs_every_node_and_returns_a_valid_answer():
    result = run(stub_nodes())
    answer = result.answer
    assert AnswerContract.model_validate(answer.model_dump()) == answer
    assert answer.answer == SKELETON_ANSWER and not answer.verified and not answer.refused
    assert any(n.startswith(COULD_NOT_VERIFY + "citations") for n in answer.notes)
    assert result.state["visits"] == {"planner": 1, "executor": 2, "composer": 2, "critic": 2}
    assert result.state["entities"].tickers == ("AAPL",) and result.state["entities"].years == (2024,)
    assert result.state["stop_reason"] is None


def test_skeleton_reads_entities_across_turns():
    result = run(stub_nodes(), question="And in fiscal 2023?", history=[QUESTION])
    assert result.state["entities"].tickers == ("AAPL",) and result.state["entities"].years == (2023,)


def test_skeleton_refusal_passes_the_critic():
    async def refuse(state):
        return {"plan": plan(refuse=True, rule="R1", steps=0)}
    nodes = stub_nodes()
    result = run(g.Nodes(refuse, nodes.executor, nodes.composer, nodes.critic), question="Should I buy Apple?")
    assert result.answer.refused and result.answer.refusal_rule == "R1" and result.answer.verified
    assert result.answer.answer == refusal_answer("R1", REFUSAL_SUGGESTION)
    assert check_contract(result.answer) == []
    assert "executor" not in result.state["visits"]


# ----------------------------------------------------------------------------------------------- edges
def test_happy_path():
    script = Script(executor=[{"tool_results": [ok_result()]}])
    result = run(script.nodes())
    assert script.calls == ["planner", "executor", "composer", "critic"]
    assert result.answer.verified and result.answer.notes == []
    assert len(result.state["tool_results"]) == 1


def test_refuse_skips_the_executor():
    script = Script(planner=[{"plan": plan(refuse=True, rule="R2", steps=0)}])
    run(script.nodes())
    assert script.calls == ["planner", "composer", "critic"]


def test_tool_error_replans_once():
    script = Script(executor=[{"tool_results": [error_result()], "tool_error": "no_results"},
                              {"tool_results": [ok_result()]}])
    result = run(script.nodes())
    assert script.calls == ["planner", "executor", "planner", "executor", "composer", "critic"]
    assert script.seen["planner"][1]["tool_error"] == "no_results"           # the replan sees the error
    assert result.state["tool_error"] is None                               # cleared by the clean round
    assert len(result.state["tool_results"]) == 2


def test_second_tool_error_goes_to_the_composer():
    script = Script(executor=[{"tool_results": [error_result()], "tool_error": "no_results"}])
    run(script.nodes())
    assert script.calls == ["planner", "executor", "planner", "executor", "composer", "critic"]


def test_failed_check_is_sent_back_once():
    bad = Verdict(violations=[("1", "a figure without a value")])
    script = Script(critic=[{"verdict": bad}, {"verdict": Verdict()}])
    result = run(script.nodes())
    assert script.calls == ["planner", "executor", "composer", "critic", "executor", "composer", "critic"]
    assert script.seen["executor"][1]["verdict"] == bad                    # the send-back carries the verdict
    assert script.seen["critic"][1]["verdict"] is None                     # a new draft's old verdict is cleared
    assert result.answer.verified


def test_second_failed_check_gives_up_unverified():
    bad = Verdict(violations=[("1", "a figure without a value")])
    script = Script(critic=[{"verdict": bad}])
    result = run(script.nodes())
    assert script.calls.count("critic") == 2 and script.calls[-1] == "critic"
    assert not result.answer.verified
    assert result.answer.notes == [f"{COULD_NOT_VERIFY}invariant 1: a figure without a value"]
    assert result.answer.answer == "A draft."                             # the best draft, marked unverified


def test_longest_legal_run_fits_the_recursion_backstop():
    bad = Verdict(violations=[("1", "x")])
    error = {"tool_results": [error_result()], "tool_error": "no_results"}
    script = Script(executor=[error, {}, error, {}], critic=[{"verdict": bad}])
    run(script.nodes())
    assert len(script.calls) + 1 == g.LONGEST_RUN == 10                    # + finish
    assert g.RECURSION_LIMIT > g.LONGEST_RUN


def test_no_plan_gives_up():
    script = Script(planner=[{"plan": None, "errors": ["planner: invalid JSON twice"]}])
    result = run(script.nodes())
    assert script.calls == ["planner"]
    assert result.answer.answer == g.GIVE_UP_ANSWER and not result.answer.verified
    assert result.answer.notes == [COULD_NOT_VERIFY + "no answer was drafted"]
    assert result.state["errors"] == ["planner: invalid JSON twice"]


def test_no_draft_gives_up():
    script = Script(composer=[{"draft": None}])
    result = run(script.nodes())
    assert script.calls == ["planner", "executor", "composer"]
    assert result.answer.answer == g.GIVE_UP_ANSWER


def test_node_stop_reason_gives_up():
    script = Script(executor=[{"stop_reason": "repeated identical tool call"}])
    result = run(script.nodes())
    assert script.calls == ["planner", "executor"]
    assert result.answer.notes == [COULD_NOT_VERIFY + "repeated identical tool call"]


# ----------------------------------------------------------------------------------------------- limits
def test_step_limit_gives_up():
    script = Script(executor=[{"tool_results": [ok_result()] * 3}])
    result = run(script.nodes(), limits=g.Limits(max_steps=2, max_tokens=100, timeout_s=5))
    assert script.calls == ["planner", "executor"]
    assert "step limit reached (3 tool calls, at most 2)" in result.answer.notes[0]


def test_token_budget_gives_up_and_keeps_the_draft_unverified():
    script = Script(composer=[{"draft": verified_draft(), "tokens": 150}])
    result = run(script.nodes(), limits=g.Limits(max_steps=10, max_tokens=100, timeout_s=5))
    assert script.calls == ["planner", "executor", "composer"]
    assert result.answer.answer == "A draft." and not result.answer.verified
    assert result.answer.notes == [f"{COULD_NOT_VERIFY}limit: token budget exceeded (150 tokens, at most 100)"]


def test_tokens_add_up_across_nodes():
    script = Script(planner=[{"plan": plan(), "tokens": 40}], composer=[{"draft": verified_draft(), "tokens": 70}])
    result = run(script.nodes(), limits=g.Limits(max_steps=10, max_tokens=100, timeout_s=5))
    assert result.state["tokens"] == 110 and not result.answer.verified


def test_time_limit_gives_up():
    async def slow(state):
        await asyncio.sleep(5)
        return {}
    script = Script()
    nodes = g.Nodes(script.node("planner"), slow, script.node("composer"), script.node("critic"))
    result = run(nodes, limits=g.Limits(max_steps=10, max_tokens=100, timeout_s=0.05))
    assert result.answer.notes == [COULD_NOT_VERIFY + "time limit reached (0.05 s)"] and result.state == {}


def test_recursion_backstop_gives_up(monkeypatch):
    monkeypatch.setattr(g, "RECURSION_LIMIT", 3)
    result = run(Script().nodes())
    assert result.answer.answer == g.GIVE_UP_ANSWER


def test_nodes_cannot_write_graph_keys():
    script = Script(planner=[{"plan": plan(), "visits": {"planner": -5}}])
    with pytest.raises(ValueError, match="only the graph writes"):
        run(script.nodes())


def test_finish_without_a_verdict_is_unverified():
    out = g.finish({"draft": verified_draft(), "verdict": None, "tool_results": [], "tokens": 0}, LIMITS)
    assert not out["answer"].verified
    assert out["answer"].notes == [f"{COULD_NOT_VERIFY}critic: the answer was not checked"]


def test_limits_from_settings():
    class S:
        MAX_AGENT_STEPS, MAX_TOKENS_PER_QUERY, QUERY_TIMEOUT_S = 7, 900, 12.5
    assert g.Limits.from_settings(S) == g.Limits(7, 900, 12.5)
