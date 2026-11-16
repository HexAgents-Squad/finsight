"""The agent's state and the JSON messages its nodes exchange (W2-01, docs/week2_plan.txt §2).

    AgentState     what every node of agent/graph.py reads and returns (a TypedDict, LangGraph's state type)
    Plan           the planner's output: reasoning, the refusal decision, ordered tool steps
    PlanStep       one tool step of a plan
    ToolCall       a tool name and its arguments, as the executor sends them to tools.registry.call_tool
    ToolResult     a call and what the tool returned (a typed error is a result, never an exception)
    Verdict        the critic's violations (agent/critic_checks.py), empty when the draft passes

Every model that an LLM writes (Plan, PlanStep) is strict pydantic with extra fields forbidden, as the answer
contract is (agent/contract.py); plan_json_schema() is its schema for response_format json_schema (ADR-016). The
final answer is agent/contract.py's AnswerContract: nothing else leaves the graph (ADR-022).

State fields are written by nodes as partial updates. The ones with a reducer (Annotated[..., reducer]) add up
across nodes: messages, tool_results and errors grow, tokens is a running sum, visits counts each node's runs (kept
by agent/graph.py, not by the nodes, so a node cannot get the loop bounds wrong). Writing None to one of them resets
it: agent.graph.initial_state does, so a question never inherits the counters, results or draft of the question
before it, even on a checkpointer thread that keeps the conversation (W2-06).
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent.contract import AnswerContract, NonBlank, decoding_schema
from agent.critic_checks import Violation
from core.spec import RefusalRule
from retrieval.entities import Entities
from tools.base import validation_message
from tools.registry import TOOLS

STRICT = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
ToolName = Literal["search_filings", "get_financial_fact", "calculate", "compare_periods"]   # tools.registry.TOOLS
MAX_PLAN_STEPS = 10         # = the default MAX_AGENT_STEPS; the executor also stops at the configured cap


class PlanStep(BaseModel):
    """One tool call the planner wants, with what it is for (the executor fills it in or replans)."""
    model_config = STRICT

    tool: ToolName
    arguments: dict[str, Any]
    purpose: NonBlank

    @model_validator(mode="after")
    def _arguments_fit_the_tool(self) -> PlanStep:
        """The arguments validated against the tool's own model (tools.registry), so a misspelt or missing
        argument is a validation error chat_json can re-prompt with, not a tool error after the plan is made.
        (Guided decoding sees `arguments` as a free object; per-tool variants are W2-02/W2-03's.)"""
        try:
            TOOLS[self.tool][1].model_validate(self.arguments)
        except ValidationError as err:
            raise ValueError(f"{self.tool} arguments: {validation_message(err)}") from None
        return self


class Plan(BaseModel):
    """The planner's decision. `reasoning` comes first, so guided decoding writes it before the decision (ADR-018:
    a reasoning field before a figure cut ungrounded numbers 20/74 -> 4/85). A refusal (R1-R7, docs/spec.md §6) is
    decided before any step and has no steps (ADR-018: strategies that started extracting under-refused)."""
    model_config = STRICT

    reasoning: NonBlank
    refuse: bool
    refusal_rule: RefusalRule | None
    steps: list[PlanStep] = Field(max_length=MAX_PLAN_STEPS)

    @model_validator(mode="after")
    def _refusal_is_consistent(self) -> Plan:
        if self.refuse and self.refusal_rule is None:
            raise ValueError("a refusal names its rule (R1-R7)")
        if not self.refuse and self.refusal_rule is not None:
            raise ValueError("refusal_rule is set only when refuse is true")
        if self.refuse and self.steps:
            raise ValueError("a refusal has no tool steps (refusal is decided before any extraction)")
        return self


def plan_json_schema() -> dict[str, Any]:
    """The Plan's JSON schema for response_format json_schema with guided decoding (every field required,
    nullable where optional; the cross-field rules are checked after parsing)."""
    return decoding_schema(Plan)


class ToolCall(BaseModel):
    """A call as the executor makes it (tools.registry.call_tool_async(name, arguments))."""
    model_config = ConfigDict(extra="forbid")

    name: ToolName
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """A tool call and its output: the tool's result, or {"error": {"type", "message"}} (tools/base.py)."""
    model_config = ConfigDict(extra="forbid")

    call: ToolCall
    output: Any

    @property
    def error_type(self) -> str | None:
        """The typed error (invalid_input, out_of_corpus, data_missing, no_results, ...) or None."""
        if isinstance(self.output, dict) and isinstance(self.output.get("error"), dict):
            return self.output["error"].get("type") or "internal_error"
        return None


class Verdict(BaseModel):
    """The critic's result: the violated rules (agent/critic_checks.Violation), empty when the draft passes."""
    model_config = ConfigDict(extra="forbid")

    violations: list[tuple[str, str]] = []          # (rule, message)

    @classmethod
    def of(cls, violations: list[Violation]) -> Verdict:
        """check_contract's result as a verdict. Invariant 7 (verified vs notes) is left out: a draft is never
        verified yet; the graph's finish node sets `verified` and the notes (critic_checks.with_verification)."""
        return cls(violations=[(v.rule, v.message) for v in violations if v.rule != "7"])

    @property
    def passed(self) -> bool:
        return not self.violations

    def as_violations(self) -> list[Violation]:
        return [Violation(rule, message) for rule, message in self.violations]


def _extend(old: list[Any] | None, new: list[Any] | None) -> list[Any]:
    """Append; None resets (a new question)."""
    return [] if new is None else [*(old or []), *new]


def _add(old: int | None, new: int | None) -> int:
    """Running sum; None resets."""
    return 0 if new is None else (old or 0) + new


def _count_visits(old: dict[str, int] | None, new: dict[str, int] | None) -> dict[str, int]:
    """Per-node run counts; None resets."""
    if new is None:
        return {}
    old = old or {}
    return {**old, **{name: old.get(name, 0) + n for name, n in new.items()}}


class AgentState(TypedDict, total=False):
    """One question's run through the graph.

    question       the user's question (input)
    history        the conversation's earlier questions, oldest first (input; memory across turns is W2-06)
    limits         {"max_steps", "max_tokens", "timeout_s"} (input, agent.graph.Limits): the executor stops
                   itself at max_steps - len(tool_results); routing gives up when a limit is broken
    messages       chat messages ({"role", "content", ...}, core.llm's format) for native tool calling (W2-03/04)
    entities       companies, fiscal years and periods (retrieval.entities.from_turns over history + question)
    plan           the planner's latest Plan (replaced on a replan)
    tool_results   every tool call and its result, in order (len = steps taken; more than max_steps gives up)
    tool_error     the typed error of the executor's latest round, None when it had none (drives the replan)
    draft          the composer's AnswerContract, before the critic
    verdict        the critic's Verdict on the latest draft
    answer         the final AnswerContract (written once, by the graph's finish node)
    tokens         LLM tokens used so far (each node returns the tokens it used; the sum is checked)
    visits         how often each node has run (agent/graph.py counts; the replan and send-back bounds)
    errors         messages for the trace (never shown as an answer)
    stop_reason    why the run stopped early (a limit or a give-up), None when it ran to the critic's verdict
    """
    question: str
    history: list[str]
    limits: dict[str, float]
    messages: Annotated[list[dict[str, Any]], _extend]
    entities: Entities | None
    plan: Plan | None
    tool_results: Annotated[list[ToolResult], _extend]
    tool_error: str | None
    draft: AnswerContract | None
    verdict: Verdict | None
    answer: AnswerContract | None
    tokens: Annotated[int, _add]
    visits: Annotated[dict[str, int], _count_visits]
    errors: Annotated[list[str], _extend]
    stop_reason: str | None
