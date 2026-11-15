"""The single access point to the chat model and embeddings (CLAUDE.md "core/llm.py").

Talks to any OpenAI-compatible endpoint through the `openai` client (Ollama locally; the hosted Qwen3-8B
endpoint over HTTPS verified against the group's CA certificate, LLM_CA_CERT), so switching provider means
editing `.env` only. Qwen3's reasoning mode is off unless LLM_ENABLE_THINKING says otherwise, and any
<think>...</think> block is removed from replies before they are returned or parsed as JSON.

    complete(messages) -> Reply (answer, raw reply, tokens, cached, retries, seconds): what a runner records
    chat(messages) -> str (complete(...).text)
    chat_json(messages, Schema) -> Schema instance (extract JSON, validate, re-prompt with the error)
    embed(texts) -> one vector per text (OpenAI-compatible endpoint or local sentence-transformers)

Shared-key rules (docs/week1_plan.txt §2.2; the team key allows one active request, 60 per minute, 8,192 tokens
per call): one request at a time per endpoint URL in this process, starts at least LLM_MIN_INTERVAL_S apart;
429, 5xx, timeouts and network errors retried after 2, 4, 8 s (LLM_RETRY_BASE_S doubling, or the server's
Retry-After if longer), never TLS, authentication or request errors; a request whose prompt + max_tokens cannot
fit LLM_CONTEXT_TOKENS is refused before sending; replies are cached on disk (LLM_CACHE_DIR), so an identical
request is answered without a call. Two processes on one key are not coordinated: run evaluations one at a time.

All three are async. Use the client as `async with LLMClient(settings) as llm:` (or call `await llm.aclose()`)
so its HTTP connections close before the event loop ends. Every call adds to `client.usage` (chat calls,
tokens, JSON retries, seconds; embedding requests and seconds), so the agent can enforce MAX_TOKENS_PER_QUERY
and the evaluation can report cost and latency.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import ssl
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
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
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})   # busy key, rate limit, server or gateway trouble
MAX_RETRY_DELAY_S = 30.0         # no wait between attempts is longer, whatever Retry-After or the back-off say
LOCK_POLL_S = 0.05               # how often a waiting request checks whether its turn has come
# Qwen3's chat template around the content, measured against the endpoint's prompt_tokens (10 Nov 2026: 1, 2
# and 4 messages gave 12, 17 and 27 tokens over the content, with or without the thinking switch)
TEMPLATE_BASE_TOKENS, TEMPLATE_TOKENS_PER_MESSAGE = 7, 5
CACHE_VERSION = 1                # part of every cache key: bump it to invalidate all cached replies
TokenCounter = Callable[[str], int]


EmbedCutter = Callable[[str, int], tuple[str, bool]]   # (text, max tokens) -> (text that fits, was cut)


@dataclass
class Usage:
    """Running totals for one client (create one client per query, or call reset()). Failed requests count
    too: they cost time, and the agent's budget has to see them."""

    calls: int = 0                  # HTTP requests sent, retries included (cache hits send none)
    cache_hits: int = 0
    http_retries: int = 0
    prompt_tokens: int = 0          # of every reply returned, cached or not: what the model's answers consumed
    completion_tokens: int = 0
    json_retries: int = 0
    seconds: float = 0.0
    embed_calls: int = 0            # embedding requests sent (retries included), or local encode calls
    embed_seconds: float = 0.0
    embed_retries: int = 0
    embed_truncated: int = 0        # texts cut to EMBED_MAX_INPUT_TOKENS before sending

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


def build_ssl_context(ca_file: Path, setting: str = "LLM_CA_CERT") -> ssl.SSLContext:
    """A client TLS context that trusts exactly `ca_file` (the group's CA), with certificate and hostname
    checks on. `create_default_context(cafile=...)` loads only that file, not the system store: the endpoint
    is the only host these clients talk to, and its issuer is in no system store anyway."""
    try:
        return ssl.create_default_context(cafile=str(ca_file))
    except (ssl.SSLError, OSError) as err:      # not PEM, no certificate inside, unreadable
        raise FinSightError(f"{setting} {ca_file} is not a readable PEM CA certificate: {err}",
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


def embed_ca(settings: Settings) -> tuple[Path | None, str]:
    """The CA file the embeddings server is verified against and the setting to fix when it fails. The group CA
    vouches for the LLM endpoint's server only: EMBED_CA_CERT (the course's embedding endpoint has its own CA,
    W1-16), else LLM_CA_CERT when it is the same server (scheme, host, port), else the default trust store
    (None), for which the setting to add is EMBED_CA_CERT. scripts/smoke_llm.py reports the same choice."""
    if settings.EMBED_CA_CERT is not None:
        return settings.EMBED_CA_CERT, "EMBED_CA_CERT"
    embed_url = settings.EMBED_BASE_URL or settings.LLM_BASE_URL
    if settings.LLM_CA_CERT is not None and _origin(embed_url) == _origin(settings.LLM_BASE_URL):
        return settings.LLM_CA_CERT, "LLM_CA_CERT"
    return None, "EMBED_CA_CERT"


def _tls_advice(tls: ssl.SSLError, ca_file: Path | None, setting: str = "LLM_CA_CERT") -> str:
    """What to fix for a failed handshake. Only a certificate that did not verify is a CA question; any other
    TLS error (e.g. WRONG_VERSION_NUMBER) means the server did not answer with TLS at all. `setting` is the
    one that chooses the CA for the failing server (LLM_CA_CERT for chat, embed_ca() for embeddings)."""
    if not isinstance(tls, ssl.SSLCertVerificationError):
        return "the server did not complete a TLS handshake: check http:// vs https:// and the port in the URL"
    if ca_file is None:
        return f"the certificate is not trusted: set {setting} to the server's CA file (LOCAL_SETUP.md)"
    return (f"the certificate does not verify against {ca_file.name}: check {setting} is that server's CA file "
            f"and the URL is your group's (LOCAL_SETUP.md)")


def _api_error(err: openai.APIError, model: str, ca_file: Path | None = None, setting: str = "LLM_CA_CERT",
               kind_of_request: str = "LLM") -> FinSightError:
    """The error left after the retries (LLMClient._send for chat, LLMClient._embed_batch for embeddings).
    `ca_file` is the CA the failing client verified against (None = default trust store), `setting` the one
    that chooses it."""
    if isinstance(err, openai.APIConnectionError):          # includes timeouts
        tls = _tls_failure(err)
        if tls is not None:
            # A TLS failure looks like a network error to openai; name the actual fix instead
            logger.error("TLS handshake for %s failed: %s", model, tls)
            return FinSightError(f"TLS handshake for {model} failed ({tls}); {_tls_advice(tls, ca_file, setting)}.",
                                 error_type="TLSError")
        kind = "NetworkError"
    else:
        kind = type(err).__name__                           # e.g. AuthenticationError, RateLimitError
    logger.error("%s request to %s failed: %s", kind_of_request, model, err)
    return FinSightError(f"{kind_of_request} request to {model} failed: {err}", error_type=kind)


@dataclass(frozen=True)
class Reply:
    """One chat completion, as a runner records it."""
    text: str                   # the answer: the reply without reasoning (strip_thinking)
    raw: str                    # the reply as the model wrote it
    prompt_tokens: int
    completion_tokens: int
    cached: bool                # answered from the disk cache, no request sent
    retries: int                # 429/5xx/network retries this call needed
    seconds: float              # the request's own time (spacing and back-off included, not the wait for the
                                # endpoint lock), the same live and replayed from the cache; an old entry without
                                # it reports the time this call took


class _Endpoint:
    """One request at a time per endpoint, first come first served: each caller takes a ticket and waits until
    it is served (polled, so the event loop never blocks; the state sits behind a threading lock, so the order
    holds across threads and event loops in one process). A plain lock would let a caller making back-to-back
    calls take it again before the others' next poll, starving them into timeouts. A caller cancelled while
    waiting gives up its ticket, so the queue never stalls."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._next = self._serving = 0
        self._abandoned: set[int] = set()
        self.last_start = float("-inf")

    def take_ticket(self) -> int:
        with self._guard:
            self._next += 1
            return self._next - 1

    def is_turn(self, ticket: int) -> bool:
        with self._guard:
            return self._serving == ticket

    def done(self, ticket: int) -> None:
        """Finish a turn (served) or give up a ticket still waiting (cancelled)."""
        with self._guard:
            if ticket != self._serving:
                self._abandoned.add(ticket)
                return
            self._serving += 1
            while self._serving in self._abandoned:
                self._abandoned.remove(self._serving)
                self._serving += 1

    async def turn(self) -> int:
        ticket = self.take_ticket()
        try:
            while not self.is_turn(ticket):
                await asyncio.sleep(LOCK_POLL_S)
        except BaseException:                       # cancelled (or interrupted) while waiting
            self.done(ticket)
            raise
        return ticket


_ENDPOINTS: dict[tuple[str, str, int | None], _Endpoint] = {}
_ENDPOINTS_GUARD = threading.Lock()
# The Qwen tokenizer for the prompt-size check, loaded once per process: None = not tried yet, False = failed
_TOKEN_COUNTER: TokenCounter | bool | None = None
_TOKEN_COUNTER_GUARD = threading.Lock()


def _endpoint(url: str) -> _Endpoint:
    """The endpoint state for the server behind `url` (scheme, host, port; case and default port normalised)."""
    with _ENDPOINTS_GUARD:
        return _ENDPOINTS.setdefault(_origin(url), _Endpoint())


def _load_token_counter() -> TokenCounter | None:
    """core.tokens.count_tokens once it has loaded, or None (remembered: an offline process does not retry a
    download on every client)."""
    global _TOKEN_COUNTER
    with _TOKEN_COUNTER_GUARD:
        if _TOKEN_COUNTER is None:
            try:
                from core.tokens import count_tokens
                count_tokens("warm-up")
                _TOKEN_COUNTER = count_tokens
            except (ImportError, FinSightError) as err:      # first run offline, or tokenizers not installed
                logger.warning("prompt-size check off (Qwen tokenizer unavailable): %s", err)
                _TOKEN_COUNTER = False
        return _TOKEN_COUNTER or None


def count_prompt_tokens(messages: list[Message], count: TokenCounter) -> int:
    """Prompt tokens of a chat request: the content plus Qwen3's chat template (measured, exact on the endpoint)."""
    return TEMPLATE_BASE_TOKENS + sum(TEMPLATE_TOKENS_PER_MESSAGE + count(m["content"]) for m in messages)


def _retryable(err: openai.APIError) -> bool:
    """Busy key, rate limit, server trouble or a dropped connection: worth another try. A TLS failure, a bad key
    or a bad request fails the same way again."""
    if isinstance(err, openai.APIConnectionError):                     # timeouts included
        return _tls_failure(err) is None
    return isinstance(err, openai.APIStatusError) and err.status_code in RETRYABLE_STATUS


def _retry_after(err: openai.APIError) -> float | None:
    """The server's Retry-After in seconds, when it sends a number."""
    response = getattr(err, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None                                                    # an HTTP date: use our own back-off


class AbstractLLMClient(ABC):
    """What the agents depend on; tests can substitute a fake implementation."""

    @abstractmethod
    async def chat(self, messages: list[Message], *, temperature: float = 0.0, model: str | None = None,
                   sample: int = 0) -> str:
        ...

    @abstractmethod
    async def chat_json(self, messages: list[Message], schema: type[T], *, model: str | None = None) -> T:
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class LLMClient(AbstractLLMClient):
    """Async LLM client for an OpenAI-compatible endpoint, with local or remote embeddings."""

    def __init__(self, settings: Settings, *, chat_client: Any = None, embed_client: Any = None,
                 token_counter: TokenCounter | None = None, embed_cutter: EmbedCutter | None = None) -> None:
        self.settings = settings
        self.usage = Usage()
        self._endpoint = _endpoint(settings.LLM_BASE_URL)
        self._count: TokenCounter | None = token_counter        # None: the process-wide Qwen tokenizer
        self._cut: EmbedCutter | None = embed_cutter            # None: the pinned embedding tokenizer
        needs_client = chat_client is None or embed_client is None
        ssl_context = build_ssl_context(settings.LLM_CA_CERT) if settings.LLM_CA_CERT and needs_client else None
        self.client = chat_client or AsyncOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY.get_secret_value(),
            timeout=settings.LLM_TIMEOUT,
            max_retries=0,                      # retries are ours (_send), so they never multiply with the SDK's
            http_client=_http_client(ssl_context),
        )
        embed_url = settings.EMBED_BASE_URL or settings.LLM_BASE_URL
        self._embed_ca, self._embed_ca_setting = embed_ca(settings)
        if self._embed_ca is None or embed_client is not None:
            embed_ssl = None
        elif self._embed_ca == settings.LLM_CA_CERT and ssl_context is not None:
            embed_ssl = ssl_context                             # one context for one CA file
        else:
            embed_ssl = build_ssl_context(self._embed_ca, self._embed_ca_setting)
        self.embed_client = embed_client or AsyncOpenAI(
            base_url=embed_url,
            api_key=(settings.EMBED_API_KEY or settings.LLM_API_KEY).get_secret_value(),
            timeout=settings.LLM_TIMEOUT,
            max_retries=0,                      # ours (_embed_batch): the same 2/4/8 s back-off as chat requests
            http_client=_http_client(embed_ssl),
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

    async def chat(self, messages: list[Message], *, temperature: float = 0.0, model: str | None = None,
                   sample: int = 0) -> str:
        """One completion for a message list (system, previous turns, tool results, user), without any
        reasoning block (strip_thinking)."""
        return (await self.complete(messages, temperature=temperature, model=model, sample=sample)).text

    async def complete(self, messages: list[Message], *, temperature: float = 0.0, model: str | None = None,
                       sample: int = 0, seed: int | None = None,
                       response_format: dict[str, Any] | None = None) -> Reply:
        """One completion with what a runner records. `sample` numbers repeated samples of the same request
        (self-consistency at temperature > 0): each has its own cache entry, so a re-run replays every sample.
        `seed` is sent to the server (the hosted endpoint honours it at temperature > 0, ADR-016) and
        `response_format` too, e.g. {"type": "json_schema", ...} for guided decoding (ADR-016); both are part of
        the cache key when set, and requests without them keep their old cache entries."""
        target_model = model or self.settings.LLM_MODEL
        request: dict[str, Any] = {"model": target_model, "messages": messages, "temperature": temperature,
                                   "max_tokens": self.settings.LLM_MAX_COMPLETION_TOKENS}
        if seed is not None:
            request["seed"] = seed
        if response_format is not None:
            request["response_format"] = response_format
        if self.settings.LLM_ENABLE_THINKING is not None:
            # Not a standard OpenAI parameter: vLLM passes it to the chat template (Qwen3), Ollama ignores it
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.settings.LLM_ENABLE_THINKING}}
        start = time.perf_counter()
        path = self._cache_path(request, sample)
        entry = self._read_cache(path) if path else None
        cached, retries = entry is not None, 0
        if entry is None:
            await self._check_budget(messages)
            entry, cached, retries = await self._fetch(request, target_model, path)
        if cached:
            self.usage.cache_hits += 1
        self.usage.prompt_tokens += entry["prompt_tokens"]
        self.usage.completion_tokens += entry["completion_tokens"]
        if entry["finish_reason"] == FINISH_LENGTH:
            # Cut off at LLM_MAX_COMPLETION_TOKENS: half an answer (or half a JSON object) must not pass as whole
            raise FinSightError(f"Reply from {target_model} hit the {self.settings.LLM_MAX_COMPLETION_TOKENS}-"
                                f"token cap (LLM_MAX_COMPLETION_TOKENS)", error_type="TruncatedResponseError")
        raw = entry["content"] or ""
        answer = strip_thinking(raw)
        if not answer:
            # Also a reply that was nothing but reasoning: there is no answer to return
            raise FinSightError(f"Empty reply from {target_model}", error_type="EmptyResponseError")
        # One figure live and replayed, so a re-run of an evaluation reproduces its records exactly. It leaves out
        # the wait for the endpoint lock (other callers' requests) but keeps request spacing and retry back-off
        seconds = entry.get("seconds", time.perf_counter() - start)
        return Reply(answer, raw, entry["prompt_tokens"], entry["completion_tokens"], cached, retries, seconds)

    async def _fetch(self, request: dict[str, Any], target_model: str,
                     path: Path | None) -> tuple[dict[str, Any], bool, int]:
        """The reply entry in this endpoint's turn: from the cache if an identical request finished while this
        one waited, else from the endpoint (then cached). Returns (entry, cached, retries)."""
        endpoint, start = self._endpoint, time.perf_counter()
        ticket = await endpoint.turn()
        try:
            entry = self._read_cache(path) if path else None
            if entry is not None:
                return entry, True, 0
            sent = time.perf_counter()
            response, retries = await self._send(request, target_model)
            choice = response.choices[0] if response.choices else None
            # "seconds": the request's own time (spacing and back-off included), replayed on a cache hit so a
            # re-run of an evaluation keeps its latency figures
            entry = {"content": choice.message.content if choice else None,
                     "finish_reason": getattr(choice, "finish_reason", None),
                     "prompt_tokens": (response.usage.prompt_tokens or 0) if response.usage else 0,
                     "completion_tokens": (response.usage.completion_tokens or 0) if response.usage else 0,
                     "seconds": round(time.perf_counter() - sent, 3)}
            # Cached before the checks in complete(), so a re-run replays them exactly; an empty reply is not
            # cached (likely a one-off glitch, and a cached one would be replayed until the folder is deleted)
            if path and strip_thinking(entry["content"] or ""):
                self._write_cache(path, entry)
            return entry, False, retries
        finally:
            endpoint.done(ticket)
            self.usage.seconds += time.perf_counter() - start

    async def _send(self, request: dict[str, Any], target_model: str) -> tuple[Any, int]:
        """The request, spaced from the previous start and retried; call it only during the endpoint's turn.
        Returns (response, retries)."""
        endpoint, retries = self._endpoint, 0
        while True:
            wait = endpoint.last_start + self.settings.LLM_MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            endpoint.last_start = time.monotonic()
            self.usage.calls += 1
            try:
                return await self.client.chat.completions.create(**request), retries
            except openai.APIError as err:
                if retries >= self.settings.LLM_HTTP_RETRIES or not _retryable(err):
                    raise _api_error(err, target_model, self.settings.LLM_CA_CERT) from err
                delay = self.settings.LLM_RETRY_BASE_S * 2 ** retries
                after = _retry_after(err)
                delay = min(max(delay, after or 0.0), MAX_RETRY_DELAY_S)
                retries += 1
                self.usage.http_retries += 1
                logger.warning("LLM request to %s failed (%s); retry %d/%d in %.0f s", target_model,
                               type(err).__name__, retries, self.settings.LLM_HTTP_RETRIES, delay)
                await asyncio.sleep(delay)

    async def _check_budget(self, messages: list[Message]) -> None:
        """Refuse a request whose prompt + max_tokens cannot fit the context (the endpoint would answer 400). The
        tokenizer loads in a worker thread, so a first-run download never freezes the event loop."""
        limit = self.settings.LLM_CONTEXT_TOKENS
        if limit is None:
            return
        count = self._count or await asyncio.to_thread(_load_token_counter)
        if count is None:
            return
        cap = self.settings.LLM_MAX_COMPLETION_TOKENS
        tokens = count_prompt_tokens(messages, count)
        if tokens + cap > limit:
            raise FinSightError(f"prompt is {tokens} tokens; with max_tokens {cap} it exceeds the {limit}-token "
                                f"context (LLM_CONTEXT_TOKENS)", error_type="PromptTooLongError")

    def _cache_path(self, request: dict[str, Any], sample: int) -> Path | None:
        if self.settings.LLM_CACHE_DIR is None:
            return None
        # The server is part of the key: another provider serving a model of the same name is another model
        key = {"v": CACHE_VERSION, "server": list(_origin(self.settings.LLM_BASE_URL)), "model": request["model"],
               "messages": request["messages"], "temperature": request["temperature"],
               "max_tokens": request["max_tokens"], "thinking": self.settings.LLM_ENABLE_THINKING, "sample": sample}
        key |= {k: request[k] for k in ("seed", "response_format") if k in request}   # absent: the old keys
        # ensure_ascii: any text, even a lone surrogate, gives a key
        digest = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
        return self.settings.LLM_CACHE_DIR / digest[:2] / f"{digest}.json"

    @staticmethod
    def _read_cache(path: Path) -> dict[str, Any] | None:
        """A cached reply, or None (absent, or unreadable: then it is fetched again and rewritten)."""
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as err:
            logger.warning("ignoring unreadable cache entry %s: %s", path.name, err)
            return None
        types = {"content": (str, type(None)), "finish_reason": (str, type(None)), "prompt_tokens": int,
                 "completion_tokens": int}
        ok = isinstance(entry, dict) and all(isinstance(entry.get(k, ...), t) for k, t in types.items())
        # "seconds" is optional (entries written before it was recorded) but must be a number when present
        ok = ok and isinstance(entry.get("seconds", 0.0), int | float) and not isinstance(entry.get("seconds"), bool)
        return entry if ok else None

    @staticmethod
    def _write_cache(path: Path, entry: dict[str, Any]) -> None:
        """Atomic: a crash never leaves half an entry; a write failure only costs the cache, never the reply."""
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")   # unique per writer
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as err:
            logger.warning("could not write cache entry %s: %s", path.name, err)
        finally:
            # missing_ok covers FileNotFoundError only: where a path component is a file, Linux raises
            # NotADirectoryError (Windows FileNotFoundError), and cleanup must never fail the reply either
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

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
        texts = await self._cut_long_texts(texts)
        vectors: list[list[float]] = []
        size = self.settings.EMBED_BATCH_SIZE
        for i in range(0, len(texts), size):
            batch = texts[i:i + size]
            response = await self._embed_batch(batch)
            if len(response.data) != len(batch):        # a short reply would misalign every later vector
                raise FinSightError(f"{self.settings.EMBED_MODEL} returned {len(response.data)} embeddings for "
                                    f"{len(batch)} texts", error_type="EmbeddingCountError")
            vectors += [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        return vectors

    async def _cut_long_texts(self, texts: list[str]) -> list[str]:
        """Texts over EMBED_MAX_INPUT_TOKENS cut to fit (the hosted endpoint refuses them, failing the whole
        batch); one warning per call says how many. The tokenizer loads off the event loop."""
        limit = self.settings.EMBED_MAX_INPUT_TOKENS
        if limit is None:
            return texts
        cut = self._cut
        if cut is None:
            from core.tokens import cut_to_embedding_tokens
            cut = cut_to_embedding_tokens
        results = await asyncio.to_thread(lambda: [cut(t, limit) for t in texts])
        shortened = sum(was for _, was in results)
        if shortened:
            self.usage.embed_truncated += shortened
            logger.warning("%d of %d texts cut to %d embedding tokens (EMBED_MAX_INPUT_TOKENS)", shortened,
                           len(texts), limit)
        return [text for text, _ in results]

    async def _embed_batch(self, batch: list[str]) -> Any:
        """One embeddings request, retried like a chat request: 429, 5xx, timeouts and network errors after
        2, 4, 8 s (LLM_HTTP_RETRIES, LLM_RETRY_BASE_S; a longer Retry-After honoured, at most 30 s); a refused
        request (422: too many texts, a text too long) fails at once."""
        retries = 0
        while True:
            self.usage.embed_calls += 1
            try:
                return await self.embed_client.embeddings.create(model=self.settings.EMBED_MODEL, input=batch)
            except openai.APIError as err:
                if retries >= self.settings.LLM_HTTP_RETRIES or not _retryable(err):
                    raise _api_error(err, self.settings.EMBED_MODEL, self._embed_ca, self._embed_ca_setting,
                                     "Embeddings") from err
                delay = min(max(self.settings.LLM_RETRY_BASE_S * 2 ** retries, _retry_after(err) or 0.0),
                            MAX_RETRY_DELAY_S)
                retries += 1
                self.usage.embed_retries += 1
                logger.warning("embeddings request to %s failed (%s); retry %d/%d in %.0f s", self.settings.EMBED_MODEL,
                               type(err).__name__, retries, self.settings.LLM_HTTP_RETRIES, delay)
                await asyncio.sleep(delay)

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
