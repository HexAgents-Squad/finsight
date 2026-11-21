"""The model-comparison runs files (docs/week1_plan.txt §2.4-2.5): written by the runner (W1-13), read by the scorer
(W1-08, eval/score_answers.py). Both sides use these models, so the format cannot drift between them.

    eval/results/model_runs/<run_id>.jsonl       one RunRecord per question (80 gold + 22 refusal tests)
    eval/results/model_runs/<run_id>.meta.json   one RunMeta

Run ids start with M<n> (a model run with the zero-shot prompt B0, W1-13), S<n> (a prompting strategy on the
same pack, W1-09: S1 few-shot, S2 chain-of-thought with self-consistency, S3 decomposition) or A<n> (the agent
graph, W2-11: eval/run_agent.py; it retrieves for itself, so its evidence is what its tools returned, recorded in
`tool_calls`, and `agent` keeps its whole answer). A strategy that makes several calls per question records each in
`steps`; S2 records how many valid samples agreed in `agreement`. An agent run's `parsed` is its answer mapped to
the B0 fields by eval/run_agent.b0_answer, so the same scorer judges it.
`parsed` is the B0 answer when a reply was valid JSON for B0Answer (after at most one retry), with json_valid true.
One documented repair (eval/run_models.py, REPAIR_VERSION) can parse a reply that stayed invalid only because a
number in "value" or "fiscal_year" had thousands separators: then `parsed` is set and json_valid is false.
Otherwise an invalid reply has `parsed` null.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from eval.schema import RefusalRule

# M1_qwen3-8b_rag_think-off_b0-v1, A1_qwen3-8b_agent_think-off_a-v1
RUN_ID_PATTERN = r"^[MSA]\d+_[a-z0-9.\-]+_(rag|closed|agent)_think-(on|off)_[a-z0-9.\-]+$"
PID_PATTERN = r"^P\d+$"


class B0Answer(BaseModel):
    """The zero-shot answer format (week1_plan §2.4). Strict, as the model must produce exactly this."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    answer: str
    value: float | int | None = None
    unit: str | None = None
    company: str | None = None
    fiscal_year: int | None = None
    citations: list[str] = []
    refused: bool
    refusal_rule: RefusalRule | None = None


class Step(BaseModel):
    """One call of a multi-call strategy (an S2 sample, an S3 stage), in the order made."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    name: str = Field(min_length=1)                  # "sample 0" ... / "extract", "answer"
    raw_reply: str | None
    json_valid: bool
    retries: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_s: float = Field(ge=0)
    cached: bool
    error: str | None = None


class Evidence(BaseModel):
    """One thing a tool returned, as the scorer reads it: a passage (source = its chunk_id, text = the passage) or
    figures (source = "<concept> <accession_no>" for an XBRL fact, "calculate" for a calculation's result; text =
    the numbers). `label` says it in words for a grader (whose figure, which year, which unit); the scorer never
    reads it."""
    model_config = ConfigDict(extra="forbid", strict=True)

    source: str = Field(min_length=1)
    text: str
    label: str | None = None


class ToolCallRecord(BaseModel):
    """One of an agent run's tool calls, in order: its arguments, its typed error (or None) and what it returned."""
    model_config = ConfigDict(extra="forbid", strict=True)

    tool: str = Field(min_length=1)
    arguments: dict[str, Any]
    error: str | None = None
    evidence: list[Evidence] = []


class AgentInfo(BaseModel):
    """An agent run's own record of a question (agent/transcript.run_record): the whole answer contract, the plan,
    the critic's verdict and the run's cost. `llm_calls` counts chat replies (cache hits included);
    `live_requests` the HTTP requests sent (retries included); `cache_hits` the replies served by the cache."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    verdict: str
    verified: bool
    values: list[dict[str, Any]]
    calculations: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    notes: list[str]
    history: list[str]
    plan: dict[str, Any] | None
    visits: dict[str, int]
    stop_reason: str | None
    node_notes: list[str]
    tool_steps: int = Field(ge=0)
    tokens: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    live_requests: int = Field(ge=0)
    cache_hits: int = Field(ge=0)


class RunRecord(BaseModel):
    """One question's result. `error` is set when no reply was obtained (endpoint failure after retries)."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    id: str
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    raw_reply: str | None
    parsed: B0Answer | None
    json_valid: bool
    retries: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_s: float = Field(ge=0)
    cached: bool
    error: str | None = None
    # W1-09 strategies only (absent from B0 runs files): every call made, and S2's vote ("4/5" valid samples)
    steps: list[Step] | None = None
    agreement: str | None = Field(None, pattern=r"^\d+/\d+$")
    # W2-11 agent runs only (absent from M and S runs files): its tool calls with their evidence, and its answer
    tool_calls: list[ToolCallRecord] | None = None
    agent: AgentInfo | None = None

    def to_line(self) -> str:
        """The record as a runs-file line, without the fields of other run types it does not have: a B0 record's
        line keeps its exact bytes whatever fields later run types add."""
        return self.model_dump_json(exclude={k for k in RUN_TYPE_FIELDS if getattr(self, k) is None}) + "\n"


RUN_TYPE_FIELDS = ("steps", "agreement", "tool_calls", "agent")       # each only in the runs files of one run type


class RunMeta(BaseModel):
    """The run's settings. `context` decides what counts as evidence when scoring: the pack's passages for
    "rag", nothing but the gold answer for "closed" (a closed-book model saw no passages), what the run's own tool
    calls returned for "agent" (it read the packs only for the questions and their earlier turns)."""
    model_config = ConfigDict(extra="allow", strict=True)

    run_id: str = Field(pattern=RUN_ID_PATTERN)
    model: str = Field(min_length=1)
    endpoint: Literal["hosted", "local"]
    context: Literal["rag", "closed", "agent"]
    thinking: bool
    temperature: float
    max_tokens: int = Field(gt=0)
    prompt_version: str = Field(min_length=1)
    context_pack: str = Field(min_length=1)          # e.g. "eval/results/context_pack_gold.jsonl (cp-v1)"
    # sha256 of each pack file the run read ({"context_pack_gold.jsonl": ..., "context_pack_refusal.jsonl": ...}):
    # the scorer refuses a run made from other packs, whose passages it would misjudge
    context_pack_sha256: dict[str, str]
