"""The LangGraph agent (W2-01): planner -> executor -> composer -> critic -> finish, and its loop bounds.

    question (+ earlier turns)
        -> planner     refuse ------------------------------------------> composer
        -> executor    tool error, first time -> planner (replan once); a second error -> composer
        -> composer    draft AnswerContract
        -> critic      fails, first time -> executor (send back once; a refusal goes back to the composer);
                       fails again -> finish (verified false; the give-up answer if the injection check failed)
        -> finish      the final AnswerContract (with_verification), the only thing that leaves the graph

Any step that finds a limit broken (more tool calls than MAX_AGENT_STEPS, more tokens than MAX_TOKENS_PER_QUERY)
goes straight to finish, which gives up with a "could not verify: ..." note (docs/spec.md §8 invariant 7, §9).
The whole run is wrapped in QUERY_TIMEOUT_S. The limits travel in the state (state["limits"]), so routing and the
nodes (the executor stops itself at the step cap, W2-04) read the same numbers.

The four nodes are injected (Nodes): the real ones by default (agent/nodes.py, W2-07), the stubs (agent/stubs.py)
or scripted nodes in tests, which drive every edge. The graph owns the bounds and the verdict: it counts each node's runs (state.visits),
rejects a node that writes `answer`, `visits`, `limits`, `question` or `history`, lets only the critic write
`verdict`, and treats a critic run without a verdict as a failed check, so no draft leaves verified without the
critic's judgement (ADR-022). A draft the critic passed stays verified when a limit is broken afterwards (by the
critic's own tokens): spec §9 returns "its best verified partial answer with a note".

    run(question, history)  -> AgentRun(answer, state, seconds)     the entry point (async)
    build_graph(nodes)                                               the compiled graph
    checkpoint_serde()                                               a checkpointer serializer for the state's types
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from agent.contract import COULD_NOT_VERIFY, AnswerContract
from agent.critic_checks import Violation, with_verification
from agent.injection import IGNORED_NOTE
from agent.state import AgentState, Verdict
from core import tracing
from core.tracing import TraceInfo, Tracer, trace_run
from retrieval.entities import extract

log = logging.getLogger(__name__)

Node = Callable[[AgentState], Awaitable[dict[str, Any]]]
PLANNER, EXECUTOR, COMPOSER, CRITIC, FINISH = "planner", "executor", "composer", "critic", "finish"
MAX_PLANS = 2               # the first plan and one replan after a tool error
MAX_CRITIC_RUNS = 2         # the first check and one after the send-back
# The longest legal run: planner x2, executor x3 (after each plan and after the send-back), composer x2, critic x2,
# finish. LangGraph's recursion limit is only a backstop above it: hitting it means a routing bug. LangGraph counts
# one more step than node runs (the input), so the longest run needs LONGEST_RUN + 1; the backstop leaves 2 spare.
LONGEST_RUN = MAX_PLANS + (MAX_PLANS + MAX_CRITIC_RUNS - 1) + MAX_CRITIC_RUNS * 2 + 1
RECURSION_LIMIT = LONGEST_RUN + 1 + 2
GRAPH_KEYS = frozenset({"answer", "visits", "limits", "question", "history"})   # the graph's or the caller's only
GIVE_UP_ANSWER = "I could not find a verified answer to this question in the indexed SEC filings."
NO_VERDICT = Verdict(violations=[("critic", "the critic returned no verdict")])
LIMIT_AFTER_CHECK = "limit reached after the answer was checked: "
# Per-question fields, reset at the start of every question (None resets a reducer field, see agent/state.py)
PER_QUESTION = ("messages", "entities", "plan", "tool_results", "tool_error", "draft", "verdict", "answer", "tokens",
                "visits", "errors", "stop_reason")
# Earlier questions a turn sees (W2-06): the gold set's follow-ups go back one turn, entities are filled from the
# latest turn that names them; six keeps a long conversation's prompts bounded (~30 tokens per question)
MAX_HISTORY_TURNS = 6


@dataclass(frozen=True)
class Limits:
    """One question's limits (core/settings.py: MAX_AGENT_STEPS, MAX_TOKENS_PER_QUERY, QUERY_TIMEOUT_S)."""
    max_steps: int = 10
    max_tokens: int = 12_000
    timeout_s: float = 60.0

    @classmethod
    def from_settings(cls, settings: Any) -> Limits:
        return cls(settings.MAX_AGENT_STEPS, settings.MAX_TOKENS_PER_QUERY, settings.QUERY_TIMEOUT_S)

    @classmethod
    def of(cls, state: AgentState) -> Limits:
        """The limits a run carries (the defaults when a state has none, e.g. in a unit test)."""
        return cls(**state["limits"]) if state.get("limits") else cls()


@dataclass(frozen=True)
class Nodes:
    """The four nodes. Each takes the state and returns a partial update (never `answer`, `visits` or `limits`;
    only the critic writes `verdict`)."""
    planner: Node
    executor: Node
    composer: Node
    critic: Node


@dataclass
class AgentRun:
    """A run's final answer, its last state (tool calls, plan, verdict, tokens, visits: for traces and the
    evaluation's runs file; the state reached so far when the time limit stopped the run) and its seconds."""
    answer: AnswerContract
    state: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


def give_up_answer(reason: str) -> AnswerContract:
    """The answer when no draft can be verified (spec §9: '"could not verify"')."""
    return AnswerContract(answer=GIVE_UP_ANSWER, verified=False, refused=False, notes=[COULD_NOT_VERIFY + reason])


def limit_reason(state: AgentState) -> str | None:
    """Which limit the run has broken, or None."""
    limits = Limits.of(state)
    steps = len(state.get("tool_results") or [])
    if steps > limits.max_steps:
        return f"step limit reached ({steps} tool calls, at most {limits.max_steps})"
    tokens = state.get("tokens") or 0
    if tokens > limits.max_tokens:
        return f"token budget exceeded ({tokens} tokens, at most {limits.max_tokens})"
    return None


def _stopped(state: AgentState) -> bool:
    return bool(state.get("stop_reason") or limit_reason(state))


def _visits(state: AgentState, node: str) -> int:
    return (state.get("visits") or {}).get(node, 0)


# ----------------------------------------------------------------------------------------------- routing
def route_after_planner(state: AgentState) -> str:
    plan = state.get("plan")
    if _stopped(state) or plan is None:
        return FINISH
    return COMPOSER if plan.refuse else EXECUTOR


def route_after_executor(state: AgentState) -> str:
    if _stopped(state):
        return FINISH
    if state.get("tool_error") and _visits(state, PLANNER) < MAX_PLANS:
        return PLANNER                                  # replan once; a second error is answered as not found
    return COMPOSER


def route_after_composer(state: AgentState) -> str:
    if _stopped(state) or state.get("draft") is None:
        return FINISH
    return CRITIC


def route_after_critic(state: AgentState) -> str:
    verdict = state.get("verdict")
    if verdict is not None and verdict.passed:
        return FINISH
    if _stopped(state) or _visits(state, CRITIC) >= MAX_CRITIC_RUNS:
        return FINISH                                   # failed twice: verified false
    draft, plan = state.get("draft"), state.get("plan")
    if (draft is not None and draft.refused) or (plan is not None and plan.refuse):
        return COMPOSER                                 # a refusal has no tool steps: rewrite the sentence
    return EXECUTOR                                     # send back once, with the verdict in the state


# ----------------------------------------------------------------------------------------------- finish
def finish(state: AgentState) -> dict[str, Any]:
    """The final answer: the draft as the critic judged it (with_verification), or a give-up answer. A draft the
    critic did not see, a failed verdict, a node's stop_reason or a limit broken before the critic passed the draft
    is never verified. A draft the critic passed keeps `verified` when a limit is broken afterwards (only the
    critic's own tokens can do that), with a note saying so (spec §9: the best verified partial answer). A draft
    still failing the critic's injection check gives the give-up answer with the spec's note instead (W2-13)."""
    stopped, limit = state.get("stop_reason"), limit_reason(state)
    draft, verdict = state.get("draft"), state.get("verdict")
    if draft is None:
        why = stopped or limit or ("the composer returned no draft" if _visits(state, COMPOSER)
                                   else "the planner returned no plan")
        return {"answer": give_up_answer(why), "stop_reason": stopped or limit}
    if verdict is not None and verdict.passed and not stopped:
        answer = with_verification(draft, [])
        if limit:
            answer = answer.model_copy(update={"notes": [*answer.notes, f"{LIMIT_AFTER_CHECK}{limit}"]})
        return {"answer": answer, "stop_reason": limit}
    violations = verdict.as_violations() if verdict is not None else []
    if stopped:
        violations.append(Violation("stopped", stopped))
    elif limit:
        violations.append(Violation("limit", limit))
    elif verdict is None:
        violations.append(Violation("critic", "the answer was not checked"))
    injected = [v for v in violations if v.rule == "injection"]
    if injected:                # W2-13: a draft showing an injection's payload never reaches the user, even unverified
        answer = give_up_answer(str(injected[0]))
        return {"answer": answer.model_copy(update={"notes": [IGNORED_NOTE, *answer.notes]}),
                "stop_reason": stopped or limit}
    return {"answer": with_verification(draft, violations), "stop_reason": stopped or limit}


def _node_input(state: AgentState) -> dict[str, Any]:
    """What a node span records as its input: the per-question counters, not the filing text it works on."""
    return {"question": state.get("question"), "visits": dict(state.get("visits") or {}),
            "tokens": state.get("tokens") or 0, "tool_results": len(state.get("tool_results") or [])}


def _node_output(update: dict[str, Any]) -> dict[str, Any]:
    """A node's update as a trace records it: tool outputs are summarised (each tool call has its own span with the
    full output), the rest is what the node wrote."""
    out = {key: value for key, value in update.items() if key not in {"tool_results", "messages"}}
    if update.get("tool_results"):
        out["tool_results"] = [{"tool": r.call.name, "error": r.error_type} for r in update["tool_results"]]
    return out


async def _traced(name: str, node: Node, state: AgentState) -> dict[str, Any]:
    """The node run inside its span (a no-op unless the run is traced, W2-08)."""
    with tracing.span(name, input=_node_input(state)) as sp:
        update = await node(state)
        sp.update(output=_node_output(update or {}))
    return update


def _traced_finish(state: AgentState) -> dict[str, Any]:
    with tracing.span(FINISH, input=_node_input(state)) as sp:
        update = finish(state)
        sp.update(output={"verified": update["answer"].verified, "stop_reason": update.get("stop_reason")})
    return update


def _counted(name: str, node: Node) -> Node:
    """The node with the graph's bookkeeping: its run counted; a planner run without a plan clears the old one (a
    replan never silently re-executes it); a new draft clears the old verdict; the executor's round error resets
    when a round has none; only the critic writes a verdict (a Verdict), and a critic run without one fails the
    check."""
    async def run(state: AgentState) -> dict[str, Any]:
        update = dict(await _traced(name, node, state) or {})
        forbidden = GRAPH_KEYS & update.keys() | ({"verdict"} & update.keys() if name != CRITIC else set())
        if forbidden:
            raise ValueError(f"node {name!r} wrote {sorted(forbidden)}, which it may not write")
        if name == PLANNER:
            update.setdefault("plan", None)
        elif name == EXECUTOR:
            update.setdefault("tool_error", None)
        elif name == COMPOSER:
            update["verdict"] = None
        elif update.get("verdict") is None:
            update["verdict"] = NO_VERDICT
        elif not isinstance(update["verdict"], Verdict):
            raise TypeError(f"the critic wrote a {type(update['verdict']).__name__} verdict, not a Verdict")
        update["visits"] = {name: 1}
        return update
    run.__name__ = name
    return run


# The state's own types: a checkpointer (W2-06 memory) must be allowed to restore them. LangGraph 1.2 warns about
# unregistered types and will block them in a future version.
CHECKPOINT_TYPES = (("agent.state", "Plan"), ("agent.state", "PlanStep"), ("agent.state", "ToolCall"),
                    ("agent.state", "ToolResult"), ("agent.state", "Verdict"), ("agent.contract", "AnswerContract"),
                    ("agent.contract", "Value"), ("agent.contract", "Citation"), ("agent.contract", "Calculation"),
                    ("retrieval.entities", "Entities"))


def checkpoint_serde() -> Any:
    """The serializer to give a checkpointer, e.g. InMemorySaver(serde=checkpoint_serde())."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    return JsonPlusSerializer(allowed_msgpack_modules=list(CHECKPOINT_TYPES))


def in_memory_checkpointer() -> Any:
    """Conversation memory for one process (W2-06): build_graph(nodes, checkpointer=in_memory_checkpointer()), then
    run(question, graph=graph, config=thread(session_id)) per turn. Lost on restart, by design for now (ADR-030)."""
    from langgraph.checkpoint.memory import InMemorySaver
    return InMemorySaver(serde=checkpoint_serde())


def thread(session_id: str) -> dict[str, Any]:
    """The run config that keeps one conversation on its own checkpointer thread."""
    if not session_id or not session_id.strip():
        raise ValueError("a conversation needs a non-empty session id")
    return {"configurable": {"thread_id": session_id}}


async def thread_history(graph: Any, config: dict[str, Any] | None) -> list[str]:
    """The conversation's earlier questions, oldest first, as the checkpointer thread named in `config` keeps them
    (the last turn's history plus its question); [] without a checkpointer or a thread. Capped (cap_history) at the last
    MAX_HISTORY_TURNS questions, so a long conversation does not grow the prompts without bound."""
    thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
    if getattr(graph, "checkpointer", None) is None or not thread_id:
        return []
    snapshot = await graph.aget_state(config)
    values = snapshot.values or {}
    question = values.get("question")
    return cap_history([*(values.get("history") or []), *([question] if question else [])])


def cap_history(questions: Sequence[str]) -> list[str]:
    """The last MAX_HISTORY_TURNS questions, oldest first, plus (before them) the latest older question that names
    a company, a fiscal year or a period when none of the kept ones does: a long run of follow-ups ("And Q3?",
    "And the margin?") keeps the turn that named the company, so retrieval.entities.from_turns reads the same
    entities from the capped history as from the whole conversation."""
    questions = list(questions)
    kept, older = questions[-MAX_HISTORY_TURNS:], questions[:-MAX_HISTORY_TURNS]
    if not older:
        return kept
    named = [extract(q) for q in kept]
    anchors: set[int] = set()
    for field_name in ("tickers", "years", "periods"):
        if any(getattr(e, field_name) for e in named):
            continue
        latest = next((i for i in range(len(older) - 1, -1, -1) if getattr(extract(older[i]), field_name)), None)
        if latest is not None:
            anchors.add(latest)
    return [older[i] for i in sorted(anchors)] + kept


def build_graph(nodes: Nodes | None = None, checkpointer: Any = None) -> Any:
    """The compiled graph. Nodes default to the real ones (agent.nodes.real_nodes, W2-07: the shared endpoint from
    .env and the tools on data/); tests pass agent.stubs.stub_nodes() or fakes. The limits come with each run's
    state (run())."""
    if nodes is None:
        from agent.nodes import real_nodes
        nodes = real_nodes()
    graph = StateGraph(AgentState)
    for name in (PLANNER, EXECUTOR, COMPOSER, CRITIC):
        graph.add_node(name, _counted(name, getattr(nodes, name)))
    graph.add_node(FINISH, _traced_finish)
    graph.add_edge(START, PLANNER)
    graph.add_conditional_edges(PLANNER, route_after_planner, [EXECUTOR, COMPOSER, FINISH])
    graph.add_conditional_edges(EXECUTOR, route_after_executor, [PLANNER, COMPOSER, FINISH])
    graph.add_conditional_edges(COMPOSER, route_after_composer, [CRITIC, FINISH])
    graph.add_conditional_edges(CRITIC, route_after_critic, [EXECUTOR, COMPOSER, FINISH])
    graph.add_edge(FINISH, END)
    return graph.compile(checkpointer=checkpointer)


def initial_state(question: str, history: Sequence[str] = (), limits: Limits | None = None) -> dict[str, Any]:
    """A question's input. Every per-question field is reset (None), so a checkpointer thread that keeps the
    conversation never carries a counter, a tool result or a verified draft into the next question."""
    return {**dict.fromkeys(PER_QUESTION), "question": question, "history": list(history),
            "limits": asdict(limits or Limits())}


async def run(question: str, history: Sequence[str] | None = None, *, graph: Any = None, limits: Limits | None = None,
              config: dict[str, Any] | None = None, trace: TraceInfo | None = None,
              tracer: Tracer | None = None) -> AgentRun:
    """Answer one question (with the conversation's earlier questions) within the limits. Always an
    AnswerContract: a run stopped by the time limit or the recursion backstop gives up with a note and keeps the
    state reached so far. Other exceptions, a node's own timeouts included, are bugs and propagate.

    Memory (W2-06): `history=None` takes the earlier questions from the checkpointer thread in `config` (a graph
    built with in_memory_checkpointer(), config=thread(session_id)), so a follow-up ("And in the third quarter?")
    keeps the company and year of the turn before; without a thread there are none. A given `history` is used as
    is (the evaluation replays gold chains that way). Either way capped by cap_history (the last MAX_HISTORY_TURNS
    questions and the older turns that named the company, year or period). Runs on one thread must not overlap:
    the API (W2-09) serialises a session's questions.

    `trace` (W2-08) sends the run to Langfuse: one trace tagged q:<question_id>, run:<run_id> and the verdict, with a
    span per node, tool call and LLM call. Without it nothing is traced and no setting is read; `tracer` replaces
    the one from .env (tests pass a recording fake)."""
    with trace_run(trace, question, tracer=tracer) as root:
        result = await _run(question, history, graph=graph, limits=limits, config=config)
        root.update(output=result.answer.answer, metadata=_trace_summary(result))
        # the outcome as a tag (verdict:verified / unverified / refused), so the dashboard can filter on it
        root.tag("verdict:" + ("refused" if result.answer.refused else "verified" if result.answer.verified
                               else "unverified"))
    return result


def _trace_summary(result: AgentRun) -> dict[str, Any]:
    """PII-free numbers for the trace: what the dashboard (latency, steps, tokens, failures) is built from."""
    state = result.state
    return {"seconds": round(result.seconds, 3), "tokens": state.get("tokens") or 0,
            "tool_calls": len(state.get("tool_results") or []), "visits": dict(state.get("visits") or {}),
            "verified": result.answer.verified, "refused": result.answer.refused,
            "stop_reason": state.get("stop_reason"), "notes": list(result.answer.notes)}


async def _run(question: str, history: Sequence[str] | None, *, graph: Any, limits: Limits | None,
               config: dict[str, Any] | None) -> AgentRun:
    """run() without tracing."""
    if limits is None:                                  # needs a complete .env (core/settings.py); tests pass limits
        from core.settings import get_settings
        limits = Limits.from_settings(get_settings())
    graph = graph or build_graph()
    history = await thread_history(graph, config) if history is None else cap_history(history)
    state: dict[str, Any] = {}
    started = time.monotonic()
    try:
        async with asyncio.timeout(limits.timeout_s) as deadline:
            async for snapshot in graph.astream(initial_state(question, history, limits), stream_mode="values",
                                                config={"recursion_limit": RECURSION_LIMIT, **(config or {})}):
                state = snapshot                        # the latest full state, kept if the time limit stops the run
    except TimeoutError:
        if not deadline.expired():
            raise                                       # a node's own timeout, not the question's limit
        return AgentRun(give_up_answer(f"time limit reached ({limits.timeout_s:g} s)"), dict(state),
                        time.monotonic() - started)
    except GraphRecursionError:
        log.error("agent graph hit its recursion backstop (%d steps): a routing bug", RECURSION_LIMIT)
        return AgentRun(give_up_answer("the agent did not finish within its step limit"), dict(state),
                        time.monotonic() - started)
    answer = state.get("answer")
    if not isinstance(answer, AnswerContract):          # finish always writes one; anything else is a bug
        raise TypeError(f"the graph returned {type(answer).__name__}, not an AnswerContract")
    return AgentRun(answer, dict(state), time.monotonic() - started)
