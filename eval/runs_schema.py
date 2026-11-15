"""The model-comparison runs files (docs/week1_plan.txt §2.4-2.5): written by the runner (W1-13), read by the scorer
(W1-08, eval/score_answers.py). Both sides use these models, so the format cannot drift between them.

    eval/results/model_runs/<run_id>.jsonl       one RunRecord per question (80 gold + 22 refusal tests)
    eval/results/model_runs/<run_id>.meta.json   one RunMeta

Run ids start with M<n> (a model run with the zero-shot prompt B0, W1-13) or S<n> (a prompting strategy on the
same pack, W1-09: S1 few-shot, S2 chain-of-thought with self-consistency, S3 decomposition). A strategy that makes
several calls per question records each in `steps`; S2 records how many valid samples agreed in `agreement`.
`parsed` is the B0 answer when a reply was valid JSON for B0Answer (after at most one retry), with json_valid true.
One documented repair (eval/run_models.py, REPAIR_VERSION) can parse a reply that stayed invalid only because a
number in "value" or "fiscal_year" had thousands separators: then `parsed` is set and json_valid is false.
Otherwise an invalid reply has `parsed` null.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from eval.schema import RefusalRule

RUN_ID_PATTERN = r"^[MS]\d+_[a-z0-9.\-]+_(rag|closed)_think-(on|off)_[a-z0-9.\-]+$"  # M1_qwen3-8b_rag_think-off_b0-v1
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


class RunMeta(BaseModel):
    """The run's settings. `context` decides what counts as evidence when scoring: the pack's passages for
    "rag", nothing but the gold answer for "closed" (a closed-book model saw no passages)."""
    model_config = ConfigDict(extra="allow", strict=True)

    run_id: str = Field(pattern=RUN_ID_PATTERN)
    model: str = Field(min_length=1)
    endpoint: Literal["hosted", "local"]
    context: Literal["rag", "closed"]
    thinking: bool
    temperature: float
    max_tokens: int = Field(gt=0)
    prompt_version: str = Field(min_length=1)
    context_pack: str = Field(min_length=1)          # e.g. "eval/results/context_pack_gold.jsonl (cp-v1)"
    # sha256 of each pack file the run read ({"context_pack_gold.jsonl": ..., "context_pack_refusal.jsonl": ...}):
    # the scorer refuses a run made from other packs, whose passages it would misjudge
    context_pack_sha256: dict[str, str]
