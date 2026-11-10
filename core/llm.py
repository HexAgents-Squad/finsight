"""The single access point to the chat model and embeddings (CLAUDE.md "core/llm.py").

Talks to any OpenAI-compatible endpoint through the `openai` client (Ollama locally; the hosted Qwen3-8B
endpoint over HTTPS verified against the group's CA certificate, LLM_CA_CERT), so switching provider means
editing `.env` only. Qwen3's reasoning mode is off unless LLM_ENABLE_THINKING says otherwise, and any
<think>...</think> block is removed from replies before they are returned or parsed as JSON.

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
import ssl
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlsplit

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
# Qwen3 reasoning: complete blocks, a closing tag whose opening tag was part of the prompt template, and an
# opening tag never closed. IGNORECASE: the tags are model output, not a fixed protocol.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
FINISH_LENGTH = "length"          # finish_reason when a reply stopped at max_tokens
MAX_CAUSE_DEPTH = 10             # exception-chain links searched for the TLS error behind a connection error
DEFAULT_PORTS = {"http": 80, "https": 443}
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


def strip_thinking(reply: str) -> str:
    """The reply without Qwen3 reasoning. With thinking off, Qwen3 still emits an empty <think></think> pair;
    with it on, the reasoning comes first. Reasoning is never part of the answer: it may contain draft numbers
    or JSON-looking text that must not reach the critic or the JSON parser. Text after a lone closing tag is
    the answer (some templates put the opening tag in the prompt); text after a lone opening tag is unfinished
    reasoning and is dropped."""
    text = THINK_BLOCK.sub("", reply)
    closes = list(THINK_CLOSE.finditer(text))
    if closes:
        text = text[closes[-1].end():]
    opening = THINK_OPEN.search(text)
    if opening:
        text = text[:opening.start()]
    return text.strip()


def _is_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


def extract_json(reply: str) -> str:
    """The JSON object in a model reply, tried in this order: the whole reply, each ``` fence, then the first
    position where a complete JSON object decodes (so braces in prose and backticks inside JSON strings do not
    break it). Reasoning blocks are removed first (strip_thinking), so a draft object inside <think> is never
    picked. If nothing decodes, the stripped reply is returned so validation reports the real error."""
    reply = strip_thinking(reply)
    for candidate in [reply, *(m.group(1).strip() for m in CODE_FENCE.finditer(reply))]:
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
    return reply


def build_ssl_context(ca_file: Path) -> ssl.SSLContext:
    """A client TLS context that trusts exactly `ca_file` (the group's CA), with certificate and hostname
    checks on. `create_default_context(cafile=...)` loads only that file, not the system store: the endpoint
    is the only host these clients talk to, and its issuer is in no system store anyway."""
    try:
        return ssl.create_default_context(cafile=str(ca_file))
    except (ssl.SSLError, OSError) as err:      # not PEM, no certificate inside, unreadable
        raise FinSightError(f"LLM_CA_CERT {ca_file} is not a readable PEM CA certificate: {err}",
                            error_type="TLSConfigError") from err


def _http_client(ssl_context: ssl.SSLContext | None) -> Any:
    """None keeps the openai client's own HTTP client (the HTTP library's default trust store: the OS store or
    certifi, depending on the openai/httpx version); otherwise the openai default client (same limits and
    redirects; timeout and retries still come from AsyncOpenAI) verifying against the given context."""
    return None if ssl_context is None else openai.DefaultAsyncHttpxClient(verify=ssl_context)


def _origin(url: str) -> tuple[str, str, int | None]:
    """(scheme, host, port) with case and default ports normalised: the unit a CA certificate vouches for,
    so `https://H.test/v1` and `https://h.test:443/v1/` are the same server."""
    parts = urlsplit(url.strip())                    # urlsplit lower-cases the scheme; hostname is lower-cased too
    return parts.scheme, parts.hostname or "", parts.port or DEFAULT_PORTS.get(parts.scheme)


def _tls_failure(err: BaseException) -> ssl.SSLError | None:
    """The ssl error behind a connection error, if any (openai -> httpx -> httpcore -> ssl, chained)."""
    current: BaseException | None = err
    for _ in range(MAX_CAUSE_DEPTH):
        if current is None:
            return None
        if isinstance(current, ssl.SSLError):
            return current
        current = current.__cause__ or current.__context__
    return None


def _tls_advice(tls: ssl.SSLError, ca_file: Path | None) -> str:
    """What to fix for a failed handshake. Only a certificate that did not verify is a CA question; any other
    TLS error (e.g. WRONG_VERSION_NUMBER) means the server did not answer with TLS at all."""
    if not isinstance(tls, ssl.SSLCertVerificationError):
        return "the server did not complete a TLS handshake: check http:// vs https:// and the port in the URL"
    if ca_file is None:
        return "the certificate is not trusted: set LLM_CA_CERT to your group's CA file (LOCAL_SETUP.md)"
    return (f"the certificate does not verify against {ca_file.name}: check LLM_CA_CERT is your group's CA file "
            f"and the URL is your group's (LOCAL_SETUP.md)")


def _api_error(err: openai.APIError, model: str, ca_file: Path | None = None) -> FinSightError:
    """The openai client has already retried what is retryable (network, 429, 5xx); report what is left.
    `ca_file` is the CA the failing client verified against (None = default trust store)."""
    if isinstance(err, openai.APIConnectionError):          # includes timeouts
        tls = _tls_failure(err)
        if tls is not None:
            # A TLS failure looks like a network error to openai; name the actual fix instead
            logger.error("TLS handshake for %s failed: %s", model, tls)
            return FinSightError(f"TLS handshake for {model} failed ({tls}); {_tls_advice(tls, ca_file)}.",
                                 error_type="TLSError")
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
        needs_client = chat_client is None or embed_client is None
        ssl_context = build_ssl_context(settings.LLM_CA_CERT) if settings.LLM_CA_CERT and needs_client else None
        self.client = chat_client or AsyncOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY.get_secret_value(),
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_HTTP_RETRIES,
            http_client=_http_client(ssl_context),
        )
        # The group CA vouches for the LLM endpoint's server only; an embeddings URL on another server keeps
        # the default trust store
        embed_url = settings.EMBED_BASE_URL or settings.LLM_BASE_URL
        self._embed_ca = settings.LLM_CA_CERT if _origin(embed_url) == _origin(settings.LLM_BASE_URL) else None
        self.embed_client = embed_client or AsyncOpenAI(
            base_url=embed_url,
            api_key=(settings.EMBED_API_KEY or settings.LLM_API_KEY).get_secret_value(),
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_HTTP_RETRIES,
            http_client=_http_client(ssl_context if self._embed_ca else None),
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
        """One completion for a message list (system, previous turns, tool results, user), without any
        reasoning block (strip_thinking)."""
        target_model = model or self.settings.LLM_MODEL
        request: dict[str, Any] = {"model": target_model, "messages": messages, "temperature": temperature,
                                   "max_tokens": self.settings.LLM_MAX_COMPLETION_TOKENS}
        if self.settings.LLM_ENABLE_THINKING is not None:
            # Not a standard OpenAI parameter: vLLM passes it to the chat template (Qwen3), Ollama ignores it
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.settings.LLM_ENABLE_THINKING}}
        start = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(**request)
        except openai.APIError as err:
            raise _api_error(err, target_model, self.settings.LLM_CA_CERT) from err
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
        answer = strip_thinking(content or "")
        if not answer:
            # Also a reply that was nothing but reasoning: there is no answer to return
            raise FinSightError(f"Empty reply from {target_model}", error_type="EmptyResponseError")
        return answer

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
                raise _api_error(err, self.settings.EMBED_MODEL, self._embed_ca) from err
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
