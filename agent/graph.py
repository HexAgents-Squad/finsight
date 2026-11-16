"""The LangGraph agent (W2-01): planner -> executor -> composer -> critic -> finish, and its loop bounds.

    question (+ earlier turns)
        -> planner     refuse ------------------------------------------> composer
        -> executor    tool error, first time -> planner (replan once); a second error -> composer
        -> composer    draft AnswerContract
        -> critic      fails, first time -> executor (send back once); fails again -> finish (verified false)
        -> finish      the final AnswerContract (with_verification), the only thing that leaves the graph

Any step that finds a limit broken (more tool calls than MAX_AGENT_STEPS, more tokens than MAX_TOKENS_PER_QUERY)
goes straight to finish, which gives up with a "could not verify: ..." note (docs/spec.md §8 invariant 7, §9).
The whole run is wrapped in QUERY_TIMEOUT_S. The four nodes are injected (Nodes), so each Week 2 ticket replaces
one stub (agent/stubs.py) and tests drive every edge with scripted nodes; the graph itself owns the bounds: it
counts each node's runs (state.visits), so a node cannot loop the graph.

    run(question, history)  -> AgentRun(answer, state)      the entry point (async)
    build_graph(nodes, limits)                               the compiled graph
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from agent.contract import COULD_NOT_VERIFY, AnswerContract
from agent.critic_checks import Violation, with_verification
from agent.state import AgentState

log = logging.getLogger(__name__)

Node = Callable[[AgentState], Awaitable[dict[str, Any]]]
PLANNER, EXECUTOR, COMPOSER, CRITIC, FINISH = "planner", "executor", "composer", "critic", "finish"
MAX_PLANS = 2               # the first plan and one replan after a tool error
MAX_CRITIC_RUNS = 2         # the first check and one after the send-back
# The longest legal run: planner x2, executor x3 (after each plan and after the send-back), composer x2, critic x2,
# finish. LangGraph's recursion limit is only a backstop above it: hitting it means a routing bug.
LONGEST_RUN = MAX_PLANS + (MAX_PLANS + MAX_CRITIC_RUNS - 1) + MAX_CRITIC_RUNS * 2 + 1
RECURSION_LIMIT = LONGEST_RUN + 2
GRAPH_KEYS = frozenset({"answer", "visits"})      # written by the graph only
GIVE_UP_ANSWER = "I could not find a verified answer to this question in the indexed SEC filings."


@dataclass(frozen=True)
class Limits:
    """One question's limits (core/settings.py: MAX_AGENT_STEPS, MAX_TOKENS_PER_QUERY, QUERY_TIMEOUT_S)."""
    max_steps: int = 10
    max_tokens: int = 12_000
    timeout_s: float = 60.0

    @classmethod
    def from_settings(cls, settings: Any) -> Limits:
        return cls(settings.MAX_AGENT_STEPS, settings.MAX_TOKENS_PER_QUERY, settings.QUERY_TIMEOUT_S)


@dataclass(frozen=True)
class Nodes:
    """The four nodes. Each takes the state and returns a partial update (never `answer` or `visits`)."""
    planner: Node
    executor: Node
    composer: Node
    critic: Node


@dataclass
class AgentRun:
    """A run's final answer and its state (tool calls, plan, verdict, tokens, visits: for traces and the
    evaluation's runs file). `state` is empty when the run was stopped by the time limit."""
    answer: AnswerContract
    state: dict[str, Any] = field(default_factory=dict)


def give_up_answer(reason: str) -> AnswerContract:
    """The answer when no draft can be verified (spec §9: '"could not verify"')."""
    return AnswerContract(answer=GIVE_UP_ANSWER, verified=False, refused=False, notes=[COULD_NOT_VERIFY + reason])


def limit_reason(state: AgentState, limits: Limits) -> str | None:
    """Which limit the run has broken, or None."""
    steps = len(state.get("tool_results") or [])
    if steps > limits.max_steps:
        return f"step limit reached ({steps} tool calls, at most {limits.max_steps})"
    tokens = state.get("tokens") or 0
    if tokens > limits.max_tokens:
        return f"token budget exceeded ({tokens} tokens, at most {limits.max_tokens})"
    return None


def _visits(state: AgentState, node: str) -> int:
    return (state.get("visits") or {}).get(node, 0)


# ----------------------------------------------------------------------------------------------- routing
def route_after_planner(state: AgentState, limits: Limits) -> str:
    plan = state.get("plan")
    if limit_reason(state, limits) or state.get("stop_reason") or plan is None:
        return FINISH
    return COMPOSER if plan.refuse else EXECUTOR


def route_after_executor(state: AgentState, limits: Limits) -> str:
    if limit_reason(state, limits) or state.get("stop_reason"):
        return FINISH
    if state.get("tool_error") and _visits(state, PLANNER) < MAX_PLANS:
        return PLANNER                                  # replan once; a second error is answered as not found
    return COMPOSER


def route_after_composer(state: AgentState, limits: Limits) -> str:
    if limit_reason(state, limits) or state.get("stop_reason") or state.get("draft") is None:
        return FINISH
    return CRITIC


def route_after_critic(state: AgentState, limits: Limits) -> str:
    verdict = state.get("verdict")
    if verdict is not None and verdict.passed:
        return FINISH
    if limit_reason(state, limits) or state.get("stop_reason") or _visits(state, CRITIC) >= MAX_CRITIC_RUNS:
        return FINISH                                   # failed twice: verified false
    return EXECUTOR                                     # send back once, with the verdict in the state


# ----------------------------------------------------------------------------------------------- finish
def finish(state: AgentState, limits: Limits) -> dict[str, Any]:
    """The final answer: the draft as the critic judged it (with_verification), or a give-up answer. A draft that
    the critic did not see, or a run that broke a limit, is never verified."""
    reason = state.get("stop_reason") or limit_reason(state, limits)
    draft, verdict = state.get("draft"), state.get("verdict")
    if draft is None:
        return {"answer": give_up_answer(reason or "no answer was drafted"), "stop_reason": reason}
    violations = verdict.as_violations() if verdict is not None else []
    if reason:
        violations.append(Violation("limit", reason))
    elif verdict is None:
        violations.append(Violation("critic", "the answer was not checked"))
    return {"answer": with_verification(draft, violations), "stop_reason": reason}


def _counted(name: str, node: Node) -> Node:
    """The node, with the graph's bookkeeping: its run counted, a new draft's old verdict cleared, the executor's
    round error reset when a round has none."""
    async def run(state: AgentState) -> dict[str, Any]:
        update = dict(await node(state) or {})
        if GRAPH_KEYS & update.keys():
            raise ValueError(f"node {name!r} wrote {sorted(GRAPH_KEYS & update.keys())}, which only the graph writes")
        if name == COMPOSER:
            update.setdefault("verdict", None)
        if name == EXECUTOR:
            update.setdefault("tool_error", None)
        update["visits"] = {name: 1}
        return update
    run.__name__ = name
    return run


def build_graph(nodes: Nodes | None = None, limits: Limits | None = None, checkpointer: Any = None) -> Any:
    """The compiled graph (nodes default to the stubs; limits to Limits())."""
    if nodes is None:
        from agent.stubs import stub_nodes
        nodes = stub_nodes()
    limits = limits or Limits()
    graph = StateGraph(AgentState)
    for name in (PLANNER, EXECUTOR, COMPOSER, CRITIC):
        graph.add_node(name, _counted(name, getattr(nodes, name)))
    graph.add_node(FINISH, lambda state: finish(state, limits))
    graph.add_edge(START, PLANNER)
    graph.add_conditional_edges(PLANNER, lambda s: route_after_planner(s, limits), [EXECUTOR, COMPOSER, FINISH])
    graph.add_conditional_edges(EXECUTOR, lambda s: route_after_executor(s, limits), [PLANNER, COMPOSER, FINISH])
    graph.add_conditional_edges(COMPOSER, lambda s: route_after_composer(s, limits), [CRITIC, FINISH])
    graph.add_conditional_edges(CRITIC, lambda s: route_after_critic(s, limits), [EXECUTOR, FINISH])
    graph.add_edge(FINISH, END)
    return graph.compile(checkpointer=checkpointer)


def initial_state(question: str, history: Sequence[str] = ()) -> AgentState:
    return {"question": question, "history": list(history), "tool_results": [], "tokens": 0, "visits": {},
            "errors": [], "stop_reason": None}


async def run(question: str, history: Sequence[str] = (), *, graph: Any = None, limits: Limits | None = None,
              config: dict[str, Any] | None = None) -> AgentRun:
    """Answer one question (with the conversation's earlier questions) within the limits. Always an
    AnswerContract: a run stopped by the time limit or the recursion backstop gives up with a note. Other
    exceptions are bugs and propagate."""
    if limits is None:
        from core.settings import get_settings
        limits = Limits.from_settings(get_settings())
    graph = graph or build_graph(limits=limits)
    try:
        async with asyncio.timeout(limits.timeout_s):
            state = await graph.ainvoke(initial_state(question, history),
                                        config={"recursion_limit": RECURSION_LIMIT, **(config or {})})
    except TimeoutError:
        return AgentRun(give_up_answer(f"time limit reached ({limits.timeout_s:g} s)"))
    except GraphRecursionError:
        log.error("agent graph hit its recursion backstop (%d steps): a routing bug", RECURSION_LIMIT)
        return AgentRun(give_up_answer("the agent did not finish within its step limit"))
    answer = state.get("answer")
    if not isinstance(answer, AnswerContract):          # finish always writes one; anything else is a bug
        raise TypeError(f"the graph returned {type(answer).__name__}, not an AnswerContract")
    return AgentRun(answer, dict(state))
