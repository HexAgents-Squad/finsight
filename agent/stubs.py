"""Stub nodes for offline graph tests (the W2-01 skeleton). The real nodes are agent.nodes.real_nodes (W2-07),
which agent.graph.build_graph() uses by default; these stubs need no endpoint, index or .env, so the graph,
memory and tracing tests pass them explicitly. Only the critic is real (W2-05: agent.critic.make_critic); the
other stubs make no LLM or tool call:
- planner: reads the companies, years and periods (retrieval.entities.from_turns, ADR-017) and plans no steps;
- executor: runs nothing;
- composer: a refusal sentence for a refusal plan (docs/spec.md §6), otherwise a placeholder without citations;
- critic: agent.critic.make_critic() (contract checks, units against tools, metric vs the question).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.contract import AnswerContract, refusal_answer
from agent.critic import make_critic
from agent.state import AgentState, Plan
from retrieval.entities import from_turns

if TYPE_CHECKING:
    from agent.graph import Nodes

SKELETON_ANSWER = ("FinSight's agent skeleton ran every step but cannot answer yet: the planner, tools and writer "
                   "are still stubs.")
REFUSAL_SUGGESTION = "Apple's total revenue for fiscal 2024"


async def plan_stub(state: AgentState) -> dict[str, Any]:
    entities = from_turns([*state.get("history", []), state["question"]])
    plan = Plan(reasoning="W2-01 skeleton: no planner yet, so no tool steps.", refuse=False, refusal_rule=None,
                steps=[])
    return {"entities": entities, "plan": plan}


async def execute_stub(state: AgentState) -> dict[str, Any]:
    return {}


async def compose_stub(state: AgentState) -> dict[str, Any]:
    plan = state.get("plan")
    if plan is not None and plan.refuse:
        draft = AnswerContract(answer=refusal_answer(plan.refusal_rule, REFUSAL_SUGGESTION), verified=False,
                               refused=True, refusal_rule=plan.refusal_rule)
    else:
        draft = AnswerContract(answer=SKELETON_ANSWER, verified=False, refused=False)
    return {"draft": draft}


def stub_nodes() -> Nodes:
    """The four stubs (agent.graph imports this module lazily, so importing agent.graph here is no cycle)."""
    from agent.graph import Nodes

    return Nodes(planner=plan_stub, executor=execute_stub, composer=compose_stub, critic=make_critic())
