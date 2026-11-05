"""The single access point to the chat model and embeddings (CLAUDE.md "core/llm.py").

Talks to any OpenAI-compatible endpoint through the `openai` client (Ollama locally, the official endpoint
from 8 Nov), so switching provider means editing `.env` only.

    chat(messages) -> str
    chat_json(messages, Schema) -> Schema instance (extract JSON, validate, re-prompt with the error)
    embed(texts) -> one vector per text (OpenAI-compatible endpoint or local sentence-transformers)

All three are async. Use the client as `async with LLMClient(settings) as llm:` (or call `await llm.aclose()`)
so its HTTP connections close before the event loop ends. Every call adds to `client.usage` (chat calls,
tokens, JSON retries, seconds; embedding requests and seconds), so the agent can enforce MAX_TOKENS_PER_QUERY
and the evaluation can report cost and latency.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from core.exceptions import FinSightError
from core.settings import Settings

logger = logging.getLogger("finsight.llm")
T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]

JSON_SYSTEM_PROMPT = (
    "You are a precise data extraction engine. You must ALWAYS and ONLY return valid JSON "
    "matching the requested schema. Do not include conversational filler, greetings, or explanations."
)
CODE_FENCE = re.compile(r"```[A-Za-z]*\s*(.*?)```", re.DOTALL)     # any language tag, e.g. ```json
OBJECT_START = re.compile(r"\{")
FINISH_LENGTH = "length"          # finish_reason when a reply stopped at max_tokens
MAX_ERROR_CHARS = 2_000          # validation errors sent back to the model are cut to this length


@dataclass
class Usage:
    """Running totals for one client (create one client per query, or call reset()). Failed requests count
    too: they cost time, and the agent's budget has to see them."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    json_retries: int = 0
    seconds: float = 0.0
    embed_calls: int = 0
    embed_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def reset(self) -> None:
        for f in fields(self):
            setattr(self, f.name, f.default)


def _is_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


def extract_json(reply: str) -> str:
    """The JSON object in a model reply, tried in this order: the whole reply, each ``` fence, then the first
    position where a complete JSON object decodes (so braces in prose and backticks inside JSON strings do not
    break it). If nothing decodes, the stripped reply is returned so validation reports the real error."""
    text = reply.strip()
    for candidate in [text, *(m.group(1).strip() for m in CODE_FENCE.finditer(reply))]:
        if _is_object(candidate):
            return candidate
    decoder = json.JSONDecoder()
    for start in OBJECT_START.finditer(reply):
        try:
            obj, end = decoder.raw_decode(reply, start.start())
        except ValueError:
            continue
        if isinstance(obj, dict):
            return reply[start.start():end]
    return text


def _api_error(err: openai.APIError, model: str) -> FinSightError:
    """The openai client has already retried what is retryable (network, 429, 5xx); report what is left."""
    if isinstance(err, openai.APIConnectionError):          # includes timeouts
        kind = "NetworkError"
    else:
        kind = type(err).__name__                           # e.g. AuthenticationError, RateLimitError
    logger.error("LLM request to %s failed: %s", model, err)
    return FinSightError(f"LLM request to {model} failed: {err}", error_type=kind)


class AbstractLLMClient(ABC):
    """What the agents depend on; tests can substitute a fake implementation."""

    @abstractmethod
    async def chat(self, messages: list[Message], *, temperature: float = 0.0, model: str | None = None) -> str:
        ...

    @abstractmethod
    async def chat_json(self, messages: list[Message], schema: type[T], *, model: str | None = None) -> T:
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class LLMClient(AbstractLLMClient):
    """Async LLM client for an OpenAI-compatible endpoint, with local or remote embeddings."""

    def __init__(self, settings: Settings, *, chat_client: Any = None, embed_client: Any = None) -> None:
        self.settings = settings
        self.usage = Usage()
        self.client = chat_client or AsyncOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_HTTP_RETRIES,
        )
        self.embed_client = embed_client or AsyncOpenAI(
            base_url=settings.EMBED_BASE_URL or settings.LLM_BASE_URL,
            api_key=settings.EMBED_API_KEY or settings.LLM_API_KEY,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_HTTP_RETRIES,
        )
        self._st_model: Any = None
        self._st_lock = asyncio.Lock()

    async def aclose(self) -> None:
        """Close the HTTP connections. Call it (or use `async with LLMClient(...)`) before the event loop
        ends, otherwise pooled connections fail later with 'Event loop is closed' (seen on Windows)."""
        for client in {id(c): c for c in (self.client, self.embed_client)}.values():
            close = getattr(client, "close", None)
            if close is not None:
                await close()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def chat(self, messages: list[Message], *, temperature: float = 0.0, model: str | None = None) -> str:
        """One completion for a message list (system, previous turns, tool results, user)."""
        target_model = model or self.settings.LLM_MODEL
        start = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=target_model, messages=messages, temperature=temperature,
                max_tokens=self.settings.LLM_MAX_COMPLETION_TOKENS,
            )
        except openai.APIError as err:
            raise _api_error(err, target_model) from err
        finally:
            self.usage.calls += 1
            self.usage.seconds += time.perf_counter() - start
        if response.usage:
            self.usage.prompt_tokens += response.usage.prompt_tokens or 0
            self.usage.completion_tokens += response.usage.completion_tokens or 0

        if response.choices and getattr(response.choices[0], "finish_reason", None) == FINISH_LENGTH:
            # Cut off at LLM_MAX_COMPLETION_TOKENS: half an answer (or half a JSON object) must not pass as whole
            raise FinSightError(f"Reply from {target_model} hit the {self.settings.LLM_MAX_COMPLETION_TOKENS}-"
                                f"token cap (LLM_MAX_COMPLETION_TOKENS)", error_type="TruncatedResponseError")
        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise FinSightError(f"Empty reply from {target_model}", error_type="EmptyResponseError")
        return content.strip()

    async def chat_json(self, messages: list[Message], schema: type[T], *, model: str | None = None) -> T:
        """A reply validated against `schema`. Invalid output is sent back with the validation error,
        at most LLM_JSON_RETRIES times; then FinSightError("JSONValidationError")."""
        working = self._json_messages(messages, schema)
        retries = self.settings.LLM_JSON_RETRIES
        for attempt in range(retries + 1):
            reply = await self.chat(working, temperature=0.0, model=model)
            try:
                return schema.model_validate_json(extract_json(reply))    # malformed JSON is a ValidationError too
            except ValidationError as err:
                logger.warning("chat_json attempt %d/%d: invalid output: %s", attempt + 1, retries + 1, err)
                if attempt == retries:
                    raise FinSightError(
                        f"Model output failed {schema.__name__} validation after {retries} retries: {err}",
                        error_type="JSONValidationError",
                    ) from err
                self.usage.json_retries += 1
                working += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": f"Your previous output failed validation with error: "
                                                f"{str(err)[:MAX_ERROR_CHARS]}\nReturn ONLY valid JSON "
                                                f"matching the schema, with no extra text."},
                ]
        raise AssertionError("unreachable")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in input order. Requests are batched by EMBED_BATCH_SIZE."""
        if isinstance(texts, str):
            raise TypeError("embed() takes a list of texts; wrap a single text as [text]")
        if not texts:
            return []
        start = time.perf_counter()
        try:
            if self.settings.EMBED_BACKEND == "sentence-transformers":
                return await self._embed_local(texts)
            return await self._embed_remote(texts)
        finally:
            self.usage.embed_seconds += time.perf_counter() - start

    async def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        size = self.settings.EMBED_BATCH_SIZE
        for i in range(0, len(texts), size):
            self.usage.embed_calls += 1
            try:
                response = await self.embed_client.embeddings.create(
                    model=self.settings.EMBED_MODEL, input=texts[i:i + size]
                )
            except openai.APIError as err:
                raise _api_error(err, self.settings.EMBED_MODEL) from err
            batch = texts[i:i + size]
            if len(response.data) != len(batch):        # a short reply would misalign every later vector
                raise FinSightError(f"{self.settings.EMBED_MODEL} returned {len(response.data)} embeddings for "
                                    f"{len(batch)} texts", error_type="EmbeddingCountError")
            vectors += [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        return vectors

    async def _embed_local(self, texts: list[str]) -> list[list[float]]:
        # One model, loaded once off the event loop; encodes run one at a time (CPU-bound, and parallel
        # encodes on the same model would only multiply memory). usage.embed_calls counts one per embed() call
        # here, one per batch request for the remote backend.
        async with self._st_lock:
            if self._st_model is None:
                name = self.settings.EMBED_LOCAL_MODEL_NAME
                try:
                    from sentence_transformers import SentenceTransformer
                    self._st_model = await asyncio.to_thread(SentenceTransformer, name)
                except (ImportError, OSError) as err:  # package missing, or the first download failed
                    raise FinSightError(f"local embedding model {name} unavailable: {err}",
                                        error_type="EmbeddingModelError") from err
            self.usage.embed_calls += 1
            vectors = await asyncio.to_thread(
                self._st_model.encode, texts, batch_size=self.settings.EMBED_BATCH_SIZE
            )
        return vectors.tolist()

    @staticmethod
    def _json_messages(messages: list[Message], schema: type[BaseModel]) -> list[Message]:
        """Copy of `messages` with a JSON-only system prompt (unless the caller set one) and the schema."""
        schema_prompt = (
            f"\n\nReturn output adhering strictly to this JSON Schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
            f"Output ONLY the JSON object."
        )
        working = [dict(m) for m in messages]
        if not any(m.get("role") == "system" for m in working):
            working.insert(0, {"role": "system", "content": JSON_SYSTEM_PROMPT})
        if working[-1]["role"] == "user":
            working[-1]["content"] += schema_prompt
        else:
            working.append({"role": "user", "content": schema_prompt.strip()})
        return working
