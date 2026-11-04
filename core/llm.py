"""The single access point to the chat model and embeddings (CLAUDE.md "core/llm.py").

Talks to any OpenAI-compatible endpoint through the `openai` client (Ollama locally, the official endpoint
from 8 Nov), so switching provider means editing `.env` only.

    chat(messages) -> str
    chat_json(messages, Schema) -> Schema instance (extract JSON, validate, re-prompt with the error)
    embed(texts) -> one vector per text (OpenAI-compatible endpoint or local sentence-transformers)

All three are async; from synchronous code (e.g. an indexing script) use `asyncio.run(client.embed(texts))`.
Every chat call adds to `client.usage` (calls, tokens, JSON retries, seconds), so the agent can enforce
MAX_TOKENS_PER_QUERY and the evaluation can report cost and latency.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
MAX_ERROR_CHARS = 2_000          # validation errors sent back to the model are cut to this length


@dataclass
class Usage:
    """Running totals for one client (create one client per query, or call reset())."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    json_retries: int = 0
    seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def reset(self) -> None:
        self.calls = self.prompt_tokens = self.completion_tokens = self.json_retries = 0
        self.seconds = 0.0


def extract_json(reply: str) -> str:
    """The JSON object in a model reply: the inside of a ``` fence if there is one, else the outermost {...}."""
    fence = CODE_FENCE.search(reply)
    text = (fence.group(1) if fence else reply).strip()
    if not (text.startswith("{") and text.endswith("}")):
        found = JSON_OBJECT.search(text)
        if found:
            text = found.group(0)
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

    async def chat(self, messages: list[Message], *, temperature: float = 0.0, model: str | None = None) -> str:
        """One completion for a message list (system, previous turns, tool results, user)."""
        target_model = model or self.settings.LLM_MODEL
        start = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=target_model, messages=messages, temperature=temperature
            )
        except openai.APIError as err:
            raise _api_error(err, target_model) from err
        finally:
            self.usage.seconds += time.perf_counter() - start
        self.usage.calls += 1
        if response.usage:
            self.usage.prompt_tokens += response.usage.prompt_tokens or 0
            self.usage.completion_tokens += response.usage.completion_tokens or 0

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
        if self.settings.EMBED_BACKEND == "sentence-transformers":
            return await self._embed_local(texts)

        vectors: list[list[float]] = []
        size = self.settings.EMBED_BATCH_SIZE
        for i in range(0, len(texts), size):
            try:
                response = await self.embed_client.embeddings.create(
                    model=self.settings.EMBED_MODEL, input=texts[i:i + size]
                )
            except openai.APIError as err:
                raise _api_error(err, self.settings.EMBED_MODEL) from err
            vectors += [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        return vectors

    async def _embed_local(self, texts: list[str]) -> list[list[float]]:
        async with self._st_lock:                        # load the model once, off the event loop
            if self._st_model is None:
                from sentence_transformers import SentenceTransformer
                self._st_model = await asyncio.to_thread(SentenceTransformer, self.settings.EMBED_LOCAL_MODEL_NAME)
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
