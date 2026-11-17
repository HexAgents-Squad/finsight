"""One LLM call for a JSON object with strict guided decoding (ADR-016), as the agent's nodes make it.

    ask_guided(llm, messages, Model, schema)  -> GuidedReply(parsed, raw, retries, tokens, seconds, cached, error)

The reply is constrained by `schema` (response_format json_schema, strict) and then validated with `Model`, which
also checks the cross-field rules a JSON schema cannot express. A reply that still fails validation is re-asked once
with the error (the tickets' "one retry on invalid JSON"); after that, or when the call itself fails (the endpoint
is unreachable, the reply was cut off, ...), `parsed` is None and `error` says why. The node decides what that means
(the planner returns no plan, so the graph gives up with a note); nothing here raises into the graph.

`llm` is anything with core.llm.LLMClient's `complete` (tests pass a fake). This is the agent's version of
eval/strategies.ask: production code never imports eval/ (CLAUDE.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from core.exceptions import FinSightError
from core.llm import Message, Reply, extract_json

JSON_RETRIES = 1
MAX_ERROR_CHARS = 800               # of a validation error sent back to the model
RETRY_INSTRUCTION = ("Your previous reply did not validate: {error}\n"
                     "Reply again with ONE JSON object that fixes this, and nothing else.")


class CompletionClient(Protocol):
    """What ask_guided needs from core.llm.LLMClient."""

    async def complete(self, messages: list[Message], *, temperature: float = 0.0, model: str | None = None,
                       sample: int = 0, seed: int | None = None,
                       response_format: dict[str, Any] | None = None) -> Reply:
        ...


@dataclass
class GuidedReply:
    """One guided call (with its retry): the parsed object or the error, and what it cost."""
    parsed: BaseModel | None
    raw: str | None                 # the last reply as the model wrote it
    retries: int                    # validation re-asks (0 or 1)
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    cached: bool                    # every reply came from the disk cache
    error: str | None

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """response_format for strict guided decoding of `schema` (ADR-016)."""
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}


async def ask_guided(llm: CompletionClient, messages: list[Message], model: type[BaseModel], schema: dict[str, Any],
                     *, retries: int = JSON_RETRIES, llm_model: str | None = None) -> GuidedReply:
    """Ask for one `model` object, decoded against `schema`; re-asked `retries` times with the validation error."""
    fmt = response_format(model.__name__, schema)
    working = list(messages)
    tokens_in = tokens_out = asked_again = 0
    seconds, cached, raw = 0.0, True, None
    for attempt in range(retries + 1):
        try:
            reply = await llm.complete(working, temperature=0.0, model=llm_model, response_format=fmt)
        except FinSightError as err:
            return GuidedReply(None, raw, asked_again, tokens_in, tokens_out, seconds, False,
                               f"{err.error_type}: {err.message}")
        seconds += reply.seconds
        tokens_in, tokens_out = tokens_in + reply.prompt_tokens, tokens_out + reply.completion_tokens
        cached, raw = cached and reply.cached, reply.raw
        try:
            parsed = model.model_validate_json(extract_json(reply.text))
        except ValidationError as err:
            if attempt == retries:
                return GuidedReply(None, raw, asked_again, tokens_in, tokens_out, seconds, cached,
                                   f"JSONValidationError: {str(err)[:MAX_ERROR_CHARS]}")
            asked_again += 1
            working = [*working, {"role": "assistant", "content": reply.text},
                       {"role": "user", "content": RETRY_INSTRUCTION.format(error=str(err)[:MAX_ERROR_CHARS])}]
            continue
        return GuidedReply(parsed, raw, asked_again, tokens_in, tokens_out, seconds, cached, None)
    raise AssertionError("unreachable")
