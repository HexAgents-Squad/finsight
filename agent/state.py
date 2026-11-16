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
across nodes: tool_results and errors grow, tokens is a running sum, visits counts each node's runs (kept by
agent/graph.py, not by the nodes, so a node cannot get the loop bounds wrong).
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.contract import AnswerContract, NonBlank, decoding_schema
from agent.critic_checks import Violation
from core.spec import RefusalRule
from retrieval.entities import Entities

STRICT = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
ToolName = Literal["search_filings", "get_financial_fact", "calculate", "compare_periods"]   # tools.registry.TOOLS
MAX_PLAN_STEPS = 10         # the default MAX_AGENT_STEPS: a plan longer than the step cap could never finish


class PlanStep(BaseModel):
    """One tool call the planner wants, with what it is for (the executor fills it in or replans)."""
    model_config = STRICT

    tool: ToolName
    arguments: dict[str, Any]
    purpose: NonBlank


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


def _count_visits(old: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    return {**old, **{name: old.get(name, 0) + n for name, n in new.items()}}


class AgentState(TypedDict, total=False):
    """One question's run through the graph.

    question       the user's question (input)
    history        the conversation's earlier questions, oldest first (input; memory across turns is W2-06)
    entities       companies, fiscal years and periods (retrieval.entities.from_turns over history + question)
    plan           the planner's latest Plan (replaced on a replan)
    tool_results   every tool call and its result, in order (len = steps taken, capped by MAX_AGENT_STEPS)
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
    entities: Entities | None
    plan: Plan | None
    tool_results: Annotated[list[ToolResult], operator.add]
    tool_error: str | None
    draft: AnswerContract | None
    verdict: Verdict | None
    answer: AnswerContract | None
    tokens: Annotated[int, operator.add]
    visits: Annotated[dict[str, int], _count_visits]
    errors: Annotated[list[str], operator.add]
    stop_reason: str | None
