"""eval/critic_eval.py (W3-03): the critic measured on recorded first drafts. Offline: scripted nodes, no client."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

from agent import graph as g
from agent.contract import AnswerContract
from agent.state import Plan, Verdict
from agent.stubs import stub_nodes
from eval import critic_eval as ce
from eval.run_agent import Item

LIMITS = g.Limits(max_steps=10, max_tokens=12_000, timeout_s=10)


def scripted(composed: list[str]):
    """Nodes whose composer counts its drafts; the critic is replaced by critic_eval's recording one."""
    base = stub_nodes()

    async def composer(state):
        composed.append(state["question"])
        return {"draft": AnswerContract(answer=f"draft for {state['question']}", verified=False, refused=False)}

    async def planner(state):
        return {"plan": Plan(reasoning="x", refuse=False, refusal_rule=None, steps=[])}

    def flagging(state):
        return Verdict(violations=[("line_item", "flagged")]) if "wrong" in state["question"] else Verdict()

    return lambda record, current: replace(base, planner=planner, composer=composer,
                                           critic=ce.recording_critic(record, current, flagging))


def test_the_replay_stops_at_the_first_draft_so_nothing_is_recomposed():
    composed: list[str] = []
    items = [Item("q1", "a wrong answer", ()), Item("q2", "a right answer", ())]
    judged = asyncio.run(ce.replay(items, scripted(composed), LIMITS))
    assert composed == ["a wrong answer", "a right answer"]          # one draft each: a flag sends nothing back
    assert judged["q1"].flagged and not judged["q2"].flagged
    assert judged["q1"].answer == "draft for a wrong answer"


def scores(**correct):
    return {q: {"id": q, "correct": c, "answer_type": "control" if q.startswith("RF") else "short_numeric"}
            for q, c in correct.items()}


def test_the_summary_counts_catches_misses_and_false_flags_and_names_decision_only_tests():
    judged = {"A": ce.Judged("A", "x", violations=[("line_item", "m")]), "B": ce.Judged("B", "x"),
              "C": ce.Judged("C", "x", violations=[("summary", "m")]), "RF1": ce.Judged("RF1", "x", violations=[
                  ("net_sales", "m")]), "D": ce.Judged("D", "x")}
    sc = scores(A=False, B=False, C=True, RF1=True, D=None)
    s = ce.summary(judged, sc)
    assert (s["caught"], s["missed"], s["false_flags"], s["undecided"]) == (["A"], ["B"], ["C", "RF1"], ["D"])
    text = ce.report("dev2", judged, sc, {"run_id": "A1", "git": "abc", "live": 0})
    assert "- Catch rate (wrong answers flagged): 1/2 (50%)" in text
    assert "- False flags (right answers flagged): 2/2 (100%): C, RF1" in text
    assert "refusal decision only (their figures are not scored, so the flag may be right): RF1" in text
    assert "| A | wrong | line_item | m |" in text and "- B: x" in text


def test_a_replayed_draft_that_differs_from_a1s_recorded_first_draft_is_reported():
    runs = {"q1": {"raw_reply": json.dumps({"answer": "recorded"}), "agent": {"visits": {"critic": 1},
                                                                             "verified": True}},
            "q2": {"raw_reply": json.dumps({"answer": "final after a send-back"}),
                   "agent": {"visits": {"critic": 2}, "verified": True}}}
    judged = {"q1": ce.Judged("q1", "replayed"), "q2": ce.Judged("q2", "the first draft")}
    assert ce.same_as_recorded(judged, runs) == ["q1"]             # q2's recorded answer is its second draft
    assert ce.same_as_recorded({"q1": ce.Judged("q1", "recorded")}, runs) == []


def test_main_refuses_a_replay_that_is_not_a1s_and_writes_nothing(tmp_path, capsys):
    composed: list[str] = []
    settings = SimpleNamespace(MAX_AGENT_STEPS=10, MAX_TOKENS_PER_QUERY=12_000, QUERY_TIMEOUT_S=10)
    code = ce.main(["--questions", "dev2", "--out-dir", str(tmp_path)], nodes=scripted(composed), settings=settings,
                   limits=LIMITS)
    assert code == 2 and not list(tmp_path.iterdir())               # the scripted drafts are not A1's answers
    assert "differ from A1's recorded answers" in capsys.readouterr().err and len(composed) == 30
