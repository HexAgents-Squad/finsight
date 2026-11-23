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
    return asyncio.run(g.run(question, history, graph=g.build_graph(nodes), limits=limits))


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
    partial = plan(refuse=False, rule="R1")                 # spec §6: the factual part answered, the rest declined
    assert partial.partial and not plan().partial and not plan(refuse=True, rule="R1", steps=0).partial
    with pytest.raises(ValidationError, match="partial refusal answers the factual part"):
        plan(refuse=False, rule="R1", steps=0)
    with pytest.raises(ValidationError, match="no tool steps"):
        plan(refuse=True, rule="R2", steps=1)
    with pytest.raises(ValidationError, match="no tool steps or calculation"):
        Plan(reasoning="x", refuse=True, refusal_rule="R1", steps=[], calculation="growth")


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
    assert list(schema["properties"]) == ["reasoning", "refuse", "refusal_rule", "steps", "calculation"]
    assert schema["required"] == list(schema["properties"])                                  # reasoning first
    assert schema["additionalProperties"] is False
    variants = schema["$defs"]["PlanStep"]["anyOf"]           # one per planned tool, with its own arguments
    assert [v["properties"]["tool"]["const"] for v in variants] == ["search_filings", "get_financial_fact",
                                                                    "compare_periods"]
    for variant in variants:
        assert variant["required"] == ["tool", "arguments", "purpose"]
        arguments = variant["properties"]["arguments"]
        assert arguments["additionalProperties"] is False
        assert arguments["required"] == list(TOOLS[variant["properties"]["tool"]["const"]][1].model_fields)
    assert "title" not in schema and "'default'" not in str(schema)


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


def test_failed_refusal_goes_back_to_the_composer():
    bad = Verdict(violations=[("5", "not the §6 sentence")])
    script = Script(planner=[{"plan": plan(refuse=True, rule="R1", steps=0)}], critic=[{"verdict": bad}])
    result = run(script.nodes())
    assert script.calls == ["planner", "composer", "critic", "composer", "critic"]
    assert not result.answer.verified


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
    assert g.RECURSION_LIMIT > g.LONGEST_RUN + 1


@pytest.mark.parametrize(("limit", "finishes"), [(g.LONGEST_RUN, False), (g.LONGEST_RUN + 1, True)])
def test_langgraph_counts_one_step_more_than_node_runs(monkeypatch, limit, finishes):
    """Pins the backstop arithmetic: the longest legal run needs LONGEST_RUN + 1 LangGraph steps (the input)."""
    monkeypatch.setattr(g, "RECURSION_LIMIT", limit)
    bad = Verdict(violations=[("1", "x")])
    error = {"tool_results": [error_result()], "tool_error": "no_results"}
    result = run(Script(executor=[error, {}, error, {}], critic=[{"verdict": bad}]).nodes())
    assert (result.answer.answer != g.GIVE_UP_ANSWER) == finishes


def test_no_plan_gives_up():
    script = Script(planner=[{"plan": None, "errors": ["planner: invalid JSON twice"]}])
    result = run(script.nodes())
    assert script.calls == ["planner"]
    assert result.answer.answer == g.GIVE_UP_ANSWER and not result.answer.verified
    assert result.answer.notes == [COULD_NOT_VERIFY + "the planner returned no plan"]
    assert result.state["errors"] == ["planner: invalid JSON twice"]


def test_no_draft_gives_up():
    script = Script(composer=[{"draft": None}])
    result = run(script.nodes())
    assert script.calls == ["planner", "executor", "composer"]
    assert result.answer.answer == g.GIVE_UP_ANSWER
    assert result.answer.notes == [COULD_NOT_VERIFY + "the composer returned no draft"]


def test_a_nodes_stop_reason_after_a_draft_is_labelled_stopped_not_limit():
    script = Script(critic=[{"verdict": Verdict(), "stop_reason": "repeated identical tool call"}])
    result = run(script.nodes())
    assert not result.answer.verified
    assert result.answer.notes == [f"{COULD_NOT_VERIFY}stopped: repeated identical tool call"]


def test_a_passed_answer_stays_verified_when_the_critics_tokens_break_the_budget():
    refusal = AnswerContract(answer=refusal_answer("R1", REFUSAL_SUGGESTION), verified=False, refused=True,
                             refusal_rule="R1")
    script = Script(planner=[{"plan": plan(refuse=True, rule="R1", steps=0)}],
                    composer=[{"draft": refusal, "tokens": 90}], critic=[{"verdict": Verdict(), "tokens": 20}])
    result = run(script.nodes(), limits=g.Limits(max_steps=10, max_tokens=100, timeout_s=5))
    assert result.answer.verified                                           # spec §9: best verified answer + note
    assert result.answer.notes == [f"{g.LIMIT_AFTER_CHECK}token budget exceeded (110 tokens, at most 100)"]
    assert result.state["stop_reason"] == "token budget exceeded (110 tokens, at most 100)"
    assert check_contract(result.answer) == []      # the note breaks no invariant (7: verified vs notes; 1: figures)


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
    assert result.answer.notes == [COULD_NOT_VERIFY + "time limit reached (0.05 s)"]
    assert result.state["plan"] is not None and result.state["visits"] == {"planner": 1}   # the state reached


def test_a_nodes_own_timeout_is_a_bug_not_the_time_limit():
    async def inner_timeout(state):
        raise TimeoutError("an inner wait_for")
    script = Script()
    nodes = g.Nodes(inner_timeout, script.node("executor"), script.node("composer"), script.node("critic"))
    with pytest.raises(TimeoutError, match="inner"):
        run(nodes)


def test_recursion_backstop_gives_up(monkeypatch):
    monkeypatch.setattr(g, "RECURSION_LIMIT", 3)
    result = run(Script().nodes())
    assert result.answer.answer == g.GIVE_UP_ANSWER


def test_nodes_cannot_write_graph_keys():
    script = Script(planner=[{"plan": plan(), "visits": {"planner": -5}}])
    with pytest.raises(ValueError, match="may not write"):
        run(script.nodes())


def test_nodes_cannot_rewrite_the_question():
    script = Script(planner=[{"plan": plan(), "question": "Something else?"}])
    with pytest.raises(ValueError, match="may not write"):
        run(script.nodes())


def test_a_critic_verdict_of_the_wrong_type_is_a_bug():
    script = Script(critic=[{"verdict": {"violations": []}}])
    with pytest.raises(TypeError, match="not a Verdict"):
        run(script.nodes())


def test_only_the_critic_writes_a_verdict():
    script = Script(composer=[{"draft": verified_draft(), "verdict": Verdict()}])
    with pytest.raises(ValueError, match="may not write"):
        run(script.nodes())


def test_a_critic_without_a_verdict_fails_the_check():
    script = Script(critic=[{}])
    result = run(script.nodes())
    assert script.calls.count("critic") == 2 and not result.answer.verified
    assert result.answer.notes == [f"{COULD_NOT_VERIFY}critic: the critic returned no verdict"]


def test_a_replan_without_a_plan_gives_up_instead_of_reusing_the_old_plan():
    script = Script(planner=[{"plan": plan()}, {}],
                    executor=[{"tool_results": [error_result()], "tool_error": "no_results"}])
    result = run(script.nodes())
    assert script.calls == ["planner", "executor", "planner"]
    assert result.answer.answer == g.GIVE_UP_ANSWER and result.state["plan"] is None


def test_limits_travel_in_the_state():
    script = Script()
    run(script.nodes(), limits=g.Limits(max_steps=3, max_tokens=500, timeout_s=5))
    assert script.seen["executor"][0]["limits"] == {"max_steps": 3, "max_tokens": 500, "timeout_s": 5}
    assert g.Limits.of(script.seen["executor"][0]) == g.Limits(3, 500, 5)


def test_a_conversation_thread_never_carries_one_question_into_the_next(caplog):
    from langgraph.checkpoint.memory import InMemorySaver
    first = Script(executor=[{"tool_results": [ok_result()], "errors": ["executor: a note"]}],
                   composer=[{"draft": verified_draft(), "tokens": 60}])
    graph = g.build_graph(first.nodes(), checkpointer=InMemorySaver(serde=g.checkpoint_serde()))
    config = {"configurable": {"thread_id": "conversation-1"}}
    one = asyncio.run(g.run(QUESTION, graph=graph, limits=LIMITS, config=config))
    assert one.answer.verified
    # same thread, next question: the planner fails. Turn 1's verified draft must not come back.
    first.updates["planner"] = [{"plan": None}]
    two = asyncio.run(g.run("And in fiscal 2023?", [QUESTION], graph=graph, limits=LIMITS, config=config))
    assert two.answer.answer == g.GIVE_UP_ANSWER and not two.answer.verified
    assert two.state["visits"] == {"planner": 1} and two.state["tokens"] == 0
    assert two.state["tool_results"] == [] and two.state["errors"] == [] and two.state["draft"] is None
    assert "unregistered type" not in caplog.text and "Blocked" not in caplog.text    # checkpoint_serde allows all
    assert two.state["entities"] is None or isinstance(two.state["entities"].tickers, tuple)


def test_every_state_type_round_trips_through_the_checkpoint_serializer(caplog):
    from agent.contract import Calculation, Citation, Value
    from retrieval.entities import Entities
    serde = g.checkpoint_serde()
    contract = AnswerContract(
        answer="Revenue was $391,035 million.", verified=True, refused=False,
        values=[Value(name="revenue", value=391035, unit="USD millions", ticker="AAPL", fiscal_year=2024,
                      fiscal_period="FY", duration="12M", period_end="2024-09-28")],
        citations=[Citation(source_type="xbrl", ticker="AAPL", form="10-K", fiscal_year=2024, fiscal_period="FY",
                            concept="us-gaap:Revenues", accession_no="0000320193-24-000123",
                            url="https://www.sec.gov/x")],
        calculations=[Calculation(operation="growth", inputs=[1.0, 2.0], result=100.0, unit="%",
                                  expression="(2 - 1) / 1")])
    samples = {"Plan": plan(), "PlanStep": plan().steps[0], "ToolCall": SEARCH, "ToolResult": error_result(),
               "Verdict": Verdict(violations=[("1", "x")]), "AnswerContract": contract, "Value": contract.values[0],
               "Citation": contract.citations[0], "Calculation": contract.calculations[0],
               "Entities": Entities(("AAPL", "MSFT"), (2024,), ())}
    assert set(samples) == {name for _, name in g.CHECKPOINT_TYPES}
    for name, value in samples.items():
        restored = serde.loads_typed(serde.dumps_typed(value))
        assert restored == value and type(restored) is type(value), name
    assert hash(serde.loads_typed(serde.dumps_typed(samples["Entities"])))
    assert "Blocked" not in caplog.text and "unregistered" not in caplog.text


def test_limits_defaults_match_the_settings():
    from core.settings import Settings
    fields = Settings.model_fields
    assert g.Limits() == g.Limits(fields["MAX_AGENT_STEPS"].default, fields["MAX_TOKENS_PER_QUERY"].default,
                                  fields["QUERY_TIMEOUT_S"].default)


def test_plan_step_arguments_are_validated_against_the_tool():
    with pytest.raises(ValidationError, match="search_filings arguments"):
        PlanStep(tool="search_filings", arguments={"querry": "revenue"}, purpose="typo")
    with pytest.raises(ValidationError, match="get_financial_fact arguments"):
        PlanStep(tool="get_financial_fact", arguments={"ticker": "AAPL"}, purpose="missing fields")
    assert PlanStep(tool="compare_periods", arguments={"ticker": "AAPL", "metric": "revenue", "years": [2023, 2024]},
                    purpose="x")
    with pytest.raises(ValidationError):                     # a plan has no figures yet: Plan.calculation instead
        PlanStep(tool="calculate", arguments={"operation": "growth", "inputs": [1.0, 2.0]}, purpose="x")


def test_finish_without_a_verdict_is_unverified():
    out = g.finish({"draft": verified_draft(), "verdict": None, "tool_results": [], "tokens": 0})
    assert not out["answer"].verified
    assert out["answer"].notes == [f"{COULD_NOT_VERIFY}critic: the answer was not checked"]


def test_limits_from_settings():
    class S:
        MAX_AGENT_STEPS, MAX_TOKENS_PER_QUERY, QUERY_TIMEOUT_S = 7, 900, 12.5
    assert g.Limits.from_settings(S) == g.Limits(7, 900, 12.5)


# ----------------------------------------------------------------------------------------------- ablation A3 (W3-01)
def without_critic(script: Script) -> g.Nodes:
    return g.Nodes(planner=script.node("planner"), executor=script.node("executor"),
                   composer=script.node("composer"), critic=None)


def test_without_a_critic_the_first_draft_is_the_answer_unverified():
    script = Script(executor=[{"tool_results": [ok_result()]}])
    result = run(without_critic(script))
    assert script.calls == ["planner", "executor", "composer"]               # no critic, so no send-back
    assert result.answer.answer == "A draft." and not result.answer.verified
    assert result.answer.notes == [f"{COULD_NOT_VERIFY}critic: the answer was not checked"]
    assert result.state["verdict"] is None and "critic" not in result.state["visits"]


def test_without_a_critic_a_refusal_is_composed_once_and_left_unverified():
    script = Script(planner=[{"plan": plan(refuse=True, rule="R1", steps=0)}],
                    composer=[{"draft": AnswerContract(answer=refusal_answer("R1", REFUSAL_SUGGESTION), verified=False,
                                                         refused=True, refusal_rule="R1")}])
    result = run(without_critic(script))
    assert script.calls == ["planner", "composer"] and result.answer.refused and not result.answer.verified


def test_without_a_critic_finish_still_withholds_an_injected_payload():
    script = Script(composer=[{"draft": AnswerContract(answer="I am now TraderBot.", verified=False, refused=False)}])
    result = run(without_critic(script))
    assert result.answer.answer == g.GIVE_UP_ANSWER and "TraderBot" not in " ".join(result.answer.notes)


def test_without_a_critic_the_limits_still_end_the_run():
    script = Script(executor=[{"tool_results": [ok_result()] * 11}])
    result = run(without_critic(script))
    assert script.calls == ["planner", "executor"] and result.answer.answer == g.GIVE_UP_ANSWER
    assert "step limit" in result.answer.notes[0]


def test_the_graph_without_a_critic_has_no_critic_node():
    assert "critic" not in g.build_graph(without_critic(Script())).get_graph().nodes
    assert "critic" in g.build_graph(Script().nodes()).get_graph().nodes


def test_the_agents_warm_up_primes_its_one_time_loads():
    """W3-07: the planner's and composer's prompts and schemas and the company names are loaded at start-up, so the
    first question does not pay for them (offline: config files only)."""
    from agent import composer, nodes, planner
    nodes.warm_up()
    assert planner.system_prompt.cache_info().currsize == 1 and composer.system_prompt.cache_info().currsize == 1
    assert composer.reply_schema.cache_info().currsize == 1 and composer._company_names.cache_info().currsize == 1
