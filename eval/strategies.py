"""W1-09: the prompting strategies S1-S3, run by eval/run_models.py on the same context pack as B0.

Each strategy keeps B0's frozen rules text (agent/prompts/b0_zero_shot_rag.txt) word for word and adds one thing,
so a difference from B0 is the strategy's:
    S1 few-shot    four worked examples from the development set (agent/prompts/s1_examples.json) as earlier
                   user/assistant turns: a quarter's three-month column, a computed change, a line item the
                   statement does not show ("not reported"), a forecast refused under R2. The examples' questions
                   (dev-A03, dev-C01, dev-G02, dev-G01) are left out when S1 is tuned on the development set.
    S2 chain of    a "reasoning" field before the answer fields (agent/prompts/s2_cot_rag.txt), SAMPLES samples at
       thought +   SAMPLE_TEMPERATURE with seed = sample index (ADR-016: the endpoint honours the seed), and a
       self-     majority vote over the valid samples on what the answer decides (vote_key); the first sample with
       consistency the winning decision is the answer. agreement = winning votes / valid samples.
    S3 decomposition  two calls: a research step (agent/prompts/s3_extract_rag.txt) breaks the question into
                   sub-questions and copies each needed figure with its passage id and line, or lists it as
                   missing; the answer step (B0 rules + agent/prompts/s3_answer_rag.txt) sees the passages, the
                   question and the research step's JSON, and checks it against the passages.
Every call asks for its JSON shape with response_format json_schema (strict guided decoding, ADR-016), parses it
strictly, and is re-asked once with the error if it is still invalid, as in B0. A run records every call of a
multi-call strategy in RunRecord.steps (tokens and seconds add up over them).
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from core.exceptions import FinSightError
from core.llm import LLMClient, extract_json
from eval.context_pack import PASSAGE_SEPARATOR, render_passage
from eval.run_models import (
    FATAL_ERRORS,
    JSON_RETRIES,
    MAX_ERROR_CHARS,
    PROMPTS_DIR,
    RETRY_INSTRUCTION,
    FatalRunError,
    build_messages,
    prompt_file,
)
from eval.runs_schema import B0Answer, RunRecord, Step
from eval.schema import RefusalRule

STRATEGIES = {"S1": "s1", "S2": "s2", "S3": "s3"}          # run name -> strategy
PROMPT_VERSIONS = {"s1": "s1-v1", "s2": "s2-v1", "s3": "s3-v1"}
EXAMPLES_FILE = PROMPTS_DIR / "s1_examples.json"
COT_FILE = PROMPTS_DIR / "s2_cot_rag.txt"
EXTRACT_FILE = PROMPTS_DIR / "s3_extract_rag.txt"
ANSWER_FILE = PROMPTS_DIR / "s3_answer_rag.txt"
SAMPLES = 5                                   # the plan's 3-5; 5 lets a 3/5 majority decide
SAMPLE_TEMPERATURE = 0.7
VOTE_DIGITS = 3                               # values agree when equal to 3 significant digits in the same scale
RESEARCH_HEADING = "Research step (check it against the passages):"


def strategy_for(run: str) -> str:
    """'b0' for a model run (M1, M2, ...), else the strategy an S run names."""
    if run.startswith("M"):
        return "b0"
    if run not in STRATEGIES:
        raise ValueError(f"unknown run {run!r}: use M<n> for a model run or one of {', '.join(STRATEGIES)}")
    return STRATEGIES[run]


# --------------------------------------------------------------------------------------------- reply shapes
class CoTAnswer(BaseModel):
    """S2's reply: the reasoning first (guided decoding writes fields in schema order), then the B0 fields."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    reasoning: str
    answer: str
    value: float | int | None = None
    unit: str | None = None
    company: str | None = None
    fiscal_year: int | None = None
    citations: list[str] = []
    refused: bool
    refusal_rule: RefusalRule | None = None

    def b0(self) -> B0Answer:
        return B0Answer.model_validate(self.model_dump(exclude={"reasoning"}))


class ExtractedFigure(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    label: str
    value: float | int | None = None
    unit: str | None = None
    company: str | None = None
    fiscal_year: int | None = None
    period: str | None = None
    pid: str | None = None
    quote: str


class Extraction(BaseModel):
    """S3's research step."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    sub_questions: list[str]
    figures: list[ExtractedFigure]
    missing: list[str]
    out_of_scope: bool


def guided(schema: type[BaseModel]) -> dict[str, Any]:
    """response_format for strict guided decoding of `schema` (ADR-016)."""
    return {"type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": True}}


# --------------------------------------------------------------------------------------------- prompts
def b0_rules() -> str:
    return prompt_file("rag").read_text(encoding="utf-8")


def system_prompt(strategy: str) -> str:
    """The system prompt of S1 (B0's), S2 (B0's + the reasoning step) or S3's answer step (B0's + the research
    step's note)."""
    extra = {"s1": "", "s2": COT_FILE.read_text(encoding="utf-8"), "s3": ANSWER_FILE.read_text(encoding="utf-8")}
    return b0_rules() + extra[strategy]


def prompt_texts(strategy: str) -> list[str]:
    """Every text that defines the strategy's prompts, for the run's prompt_sha256 and the version pin."""
    texts = [system_prompt(strategy)]
    if strategy == "s1":
        texts.append(EXAMPLES_FILE.read_text(encoding="utf-8"))
    if strategy == "s3":
        texts.append(EXTRACT_FILE.read_text(encoding="utf-8"))
    return texts


def prompt_sha256(strategy: str) -> str:
    return hashlib.sha256("\n\0\n".join(prompt_texts(strategy)).encode("utf-8")).hexdigest()


def load_examples() -> list[dict]:
    return json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))


def example_ids() -> list[str]:
    return [e["dev_id"] for e in load_examples()]


def example_messages() -> list[dict[str, str]]:
    """S1's examples as earlier turns, framed exactly like a real question (passages, then the question)."""
    out = []
    for e in load_examples():
        block = PASSAGE_SEPARATOR.join(render_passage(p["pid"], p["citation"], "\n".join(p["lines"]))
                                       for p in e["passages"])
        answer = B0Answer.model_validate(e["answer"])        # an example must itself be a valid B0 answer
        out += [{"role": "user", "content": f"Passages:\n{block}\n\nQuestion: {e['question']}"},
                {"role": "assistant", "content": answer.model_dump_json()}]
    return out


def s1_messages(record: dict) -> list[dict[str, str]]:
    messages = build_messages(record, "rag", system_prompt("s1"))
    return [messages[0], *example_messages(), *messages[1:]]


def s2_messages(record: dict) -> list[dict[str, str]]:
    return build_messages(record, "rag", system_prompt("s2"))


def extract_messages(record: dict) -> list[dict[str, str]]:
    return build_messages(record, "rag", EXTRACT_FILE.read_text(encoding="utf-8"))


def answer_messages(record: dict, research: str) -> list[dict[str, str]]:
    """S3's answer step: B0's messages with the research step's JSON (or why it failed) after the question."""
    messages = build_messages(record, "rag", system_prompt("s3"))
    last = messages[-1]
    return [*messages[:-1], {"role": "user", "content": f"{last['content']}\n\n{RESEARCH_HEADING}\n{research}"}]


# --------------------------------------------------------------------------------------------- calls
@dataclass
class Attempt:
    """One call (with its JSON retry): the parsed reply or the error, and what it cost."""
    parsed: BaseModel | None
    raw: str | None
    json_valid: bool
    retries: int
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    cached: bool
    error: str | None

    def step(self, name: str) -> Step:
        return Step(name=name, raw_reply=self.raw, json_valid=self.json_valid, retries=self.retries,
                    prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
                    latency_s=round(self.seconds, 3), cached=self.cached and self.error is None, error=self.error)


async def ask(llm: LLMClient, messages: list[dict[str, str]], schema: type[BaseModel], *, temperature: float = 0.0,
              seed: int | None = None, sample: int = 0) -> Attempt:
    """One call for `schema` (guided decoding), re-asked once with the error if the reply still does not
    validate. An error in FATAL_ERRORS raises FatalRunError (the run stops); any other is recorded."""
    tokens_in = tokens_out = retries = 0
    seconds, cached, raw, parsed, error = 0.0, True, None, None, None
    working = list(messages)
    for attempt in range(JSON_RETRIES + 1):
        try:
            reply = await llm.complete(working, temperature=temperature, sample=sample, seed=seed,
                                       response_format=guided(schema))
        except FinSightError as err:
            if err.error_type in FATAL_ERRORS:
                raise FatalRunError(f"{err.error_type}: {err.message}") from err
            error, cached = f"{err.error_type}: {err.message}", False
            break
        seconds += reply.seconds
        tokens_in, tokens_out = tokens_in + reply.prompt_tokens, tokens_out + reply.completion_tokens
        cached, raw = cached and reply.cached, reply.raw
        try:
            parsed = schema.model_validate_json(extract_json(reply.text))
            break
        except ValidationError as err:
            if attempt == JSON_RETRIES:
                break
            retries += 1
            working = [*working, {"role": "assistant", "content": reply.text},
                       {"role": "user", "content": RETRY_INSTRUCTION.format(error=str(err)[:MAX_ERROR_CHARS])}]
    return Attempt(parsed, raw, parsed is not None, retries, tokens_in, tokens_out, seconds,
                   cached and error is None, error)


def to_record(record_id: str, run_id: str, attempts: list[tuple[str, Attempt]], chosen: Attempt | None,
              answer: B0Answer | None, agreement: str | None = None, steps: bool = True) -> RunRecord:
    """A RunRecord whose tokens, seconds and retries add up over every call; the chosen call's reply is the
    answer (None: no call gave a valid one)."""
    calls = [a for _, a in attempts]
    errors = [a.error for a in calls if a.error]
    last = chosen or calls[-1]
    return RunRecord(id=record_id, run_id=run_id, raw_reply=last.raw, parsed=answer,
                     json_valid=chosen is not None and chosen.json_valid, retries=sum(a.retries for a in calls),
                     prompt_tokens=sum(a.prompt_tokens for a in calls),
                     completion_tokens=sum(a.completion_tokens for a in calls),
                     latency_s=round(sum(a.seconds for a in calls), 3), cached=all(a.cached for a in calls),
                     error=None if answer is not None else (errors[0] if errors else None),
                     steps=[a.step(name) for name, a in attempts] if steps else None, agreement=agreement)


# --------------------------------------------------------------------------------------------- S2 vote
@functools.cache
def _matchers() -> dict:
    from eval.score_answers import company_matchers
    return company_matchers()


def vote_key(a: B0Answer) -> tuple:
    """What an answer decides: refused and the rule, the company it names (as tickers), and the value in one
    scale to VOTE_DIGITS significant digits (None when it gives none: not reported, a summary, a refusal)."""
    from eval.score_answers import companies_named, unit_scale
    value = None
    if a.value is not None and not a.refused:
        scaled = float(a.value) * (unit_scale(a.unit) or 1.0)
        value = 0.0 if scaled == 0 else round(scaled, VOTE_DIGITS - 1 - math.floor(math.log10(abs(scaled))))
    return (a.refused, a.refusal_rule, frozenset(companies_named(a.company, _matchers())), value)


def vote(answers: list[B0Answer]) -> tuple[int, int]:
    """(index of the answer to keep, its votes): the most common decision, ties going to the decision seen
    first; the kept answer is the first sample with that decision."""
    keys = [vote_key(a) for a in answers]
    counts = Counter(keys)
    best = max(counts, key=lambda k: (counts[k], -keys.index(k)))
    return keys.index(best), counts[best]


# --------------------------------------------------------------------------------------------- strategies
async def run_s1(llm: LLMClient, run_id: str, record: dict) -> RunRecord:
    a = await ask(llm, s1_messages(record), B0Answer)
    return to_record(record["id"], run_id, [("answer", a)], a if a.parsed else None, a.parsed, steps=False)


async def run_s2(llm: LLMClient, run_id: str, record: dict, samples: int = SAMPLES) -> RunRecord:
    messages = s2_messages(record)
    attempts = [(f"sample {i}", await ask(llm, messages, CoTAnswer, temperature=SAMPLE_TEMPERATURE, seed=i,
                                          sample=i)) for i in range(samples)]
    valid = [a for _, a in attempts if a.parsed is not None]
    if not valid:
        return to_record(record["id"], run_id, attempts, None, None, agreement=f"0/{samples}")
    index, votes = vote([a.parsed.b0() for a in valid])
    chosen = valid[index]
    return to_record(record["id"], run_id, attempts, chosen, chosen.parsed.b0(), agreement=f"{votes}/{len(valid)}")


async def run_s3(llm: LLMClient, run_id: str, record: dict) -> RunRecord:
    research = await ask(llm, extract_messages(record), Extraction)
    note = research.parsed.model_dump_json() if research.parsed else \
        json.dumps({"failed": research.error or "the research step gave no valid JSON"})
    answer = await ask(llm, answer_messages(record, note), B0Answer)
    return to_record(record["id"], run_id, [("extract", research), ("answer", answer)],
                     answer if answer.parsed else None, answer.parsed)


RUNNERS: dict[str, Callable[[LLMClient, str, dict], Awaitable[RunRecord]]] = {"s1": run_s1, "s2": run_s2,
                                                                              "s3": run_s3}
