"""The FinSight API (W2-09, ADR-032): the agent behind HTTP, with a reply cache and rate limits.

    POST /ask      {"question": "...", "history": [...]?}, header X-Session-Id (optional)  -> AskResponse
    GET  /health   -> HealthResponse: ready or warming up, the model, the cache and session counts

    uvicorn api.main:app --port 8000      (needs .env and data/, like scripts/ask.py; one worker: state is in memory)

- One compiled graph on the real nodes (agent.nodes.real_nodes) serves every request, on one LLM client opened and
  closed with the app (core/llm.py). The graph is async; its blocking tool calls run on worker threads through
  tools.registry.call_tool_async, and the index warm-up (tools.registry.warm_up, tens of seconds) runs on a thread
  at start-up, so the event loop keeps serving /health meanwhile.
- Sessions: the server issues a signed X-Session-Id with a conversation's first answer; the client sends it with
  the next questions. An id this server did not issue starts a new session (a client cannot mint ids). The API keeps
  the session's earlier questions and passes them to the agent (capped by agent.graph.cap_history); a body "history"
  replaces them (the evaluation replays chains that way). A session's questions run one at a time.
- Reply cache: keyed on the normalised question and the earlier questions it was asked with; only a verified answer
  or a refusal is kept (an unverified one may be a time-out). A cache hit still counts as the session's turn; the
  session's last question sent again (a client's retry) is the same turn, not a follow-up of itself. `run` describes
  the run that produced the answer; `seconds` is this request's.
- Rate limits (slowapi): API_RATE_LIMIT per session (an issued id, else the client address) and API_ADDRESS_RATE_LIMIT
  per client address whatever the id; over one, 429 with Retry-After. At most API_MAX_CONCURRENT questions run at
  once (the shared endpoint key); one that waits longer than API_QUEUE_TIMEOUT_S for its turn gets 503.
- Errors: {"detail": ...} throughout. An invalid request is 422; an error inside the agent is a 500 with a generic
  message (logged here, never the internals in the response). Nothing logs a question or a key.

create_app(graph=..., config=...) serves a given graph (the tests' fake one: no endpoint, no .env); create_app()
builds the real one when the app starts, so importing this module needs no .env.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from limits import parse_many
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent.contract import AnswerContract
from agent.graph import AgentRun, Limits, cap_history, run
from agent.transcript import verdict
from api.cache import ReplyCache, cache_key, normalise
from api.sessions import SessionIds, Sessions, session_digest

log = logging.getLogger(__name__)

SESSION_HEADER = "X-Session-Id"
SESSION_HEADER_SHAPE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")      # anything else is a malformed header: 422
MAX_QUESTION_CHARS = 1_000              # spec §2's questions are a sentence or two
MAX_HISTORY = 20                        # earlier questions in a body; the agent sees at most cap_history of them
MAX_BODY_BYTES = 64_000                 # 21 questions of 1,000 characters are ~22 kB of JSON
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)]


# ----------------------------------------------------------------------------------------------- messages
class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    history: list[Question] | None = Field(default=None, max_length=MAX_HISTORY, description=(
        "earlier questions of the conversation, oldest first; omit to use the session's own"))


class ToolCallSummary(BaseModel):
    tool: str
    arguments: dict[str, Any]
    error: str | None = None


class RunSummary(BaseModel):
    """What the run that produced the answer cost and did (the UI's trace expander), without the tool outputs."""
    verdict: str
    steps: int
    tokens: int
    seconds: float
    visits: dict[str, int]
    stop_reason: str | None = None
    tool_calls: list[ToolCallSummary]


class AskResponse(BaseModel):
    session_id: str
    cached: bool = Field(description="answered from the reply cache: `run` is the run that produced it")
    seconds: float = Field(description="this request's time")
    history: list[str] = Field(description="the earlier questions the answer was given with")
    answer: AnswerContract
    run: RunSummary


class HealthResponse(BaseModel):
    status: Literal["ok", "starting", "degraded"]
    warm_up: str = Field(description='"ok", "pending", or the error type that stopped it')
    model: str | None
    cache_entries: int
    cache_hits: int = Field(description="questions answered from the reply cache since start-up")
    cache_misses: int = Field(description="questions the agent answered since start-up")
    cache_hit_rate: float | None = Field(description="cache_hits / (cache_hits + cache_misses); null before the "
                                                     "first question")
    sessions: int


# ----------------------------------------------------------------------------------------------- config
@dataclass(frozen=True)
class ApiConfig:
    rate_limit: str = "10/minute"
    address_rate_limit: str = "60/minute"
    cache_ttl_s: float = 3600.0
    cache_max_entries: int = 512
    max_concurrent: int = 2
    queue_timeout_s: float = 30.0
    session_ttl_s: float = 3600.0
    max_sessions: int = 1000
    session_secret: bytes | None = None             # None: a random one per process
    limits: Limits = field(default_factory=Limits)
    model: str | None = None
    trace: bool = False

    def __post_init__(self) -> None:
        for limit in (self.rate_limit, self.address_rate_limit):
            parse_many(limit)                       # a malformed limit fails here, not silently per request
        if self.max_concurrent < 1 or self.queue_timeout_s <= 0:
            raise ValueError("max_concurrent must be at least 1 and queue_timeout_s positive")

    @classmethod
    def from_settings(cls, settings: Any) -> ApiConfig:
        secret = settings.API_SESSION_SECRET
        return cls(rate_limit=settings.API_RATE_LIMIT, address_rate_limit=settings.API_ADDRESS_RATE_LIMIT,
                   cache_ttl_s=settings.API_CACHE_TTL_S, cache_max_entries=settings.API_CACHE_MAX_ENTRIES,
                   max_concurrent=settings.API_MAX_CONCURRENT, queue_timeout_s=settings.API_QUEUE_TIMEOUT_S,
                   session_ttl_s=settings.API_SESSION_TTL_S, max_sessions=settings.API_MAX_SESSIONS,
                   session_secret=secret.get_secret_value().encode() if secret else None,
                   limits=Limits.from_settings(settings), model=settings.LLM_MODEL, trace=settings.LANGFUSE_ENABLED)


class _State:
    """The app's moving parts, set when it starts (create_app's arguments, or the real ones in the lifespan)."""

    def __init__(self, config: ApiConfig, graph: Any):
        self.graph = graph
        self.configure(config)
        self.warm_up = "pending"

    def configure(self, config: ApiConfig) -> None:
        self.config = config
        self.cache = ReplyCache(config.cache_max_entries, config.cache_ttl_s)
        self.sessions = Sessions(config.max_sessions, config.session_ttl_s)
        self.ids = SessionIds(config.session_secret)
        self.slots = asyncio.Semaphore(config.max_concurrent)


def session_header(session_id: Annotated[str | None, Header(alias=SESSION_HEADER)] = None) -> str | None:
    """The X-Session-Id sent, if well formed (ASCII letters, digits, ".", "-", "_"); 422 otherwise, before any rate
    limit is counted."""
    if session_id is not None and not (session_id.isascii() and SESSION_HEADER_SHAPE.match(session_id)):
        raise HTTPException(status_code=422, detail=f"{SESSION_HEADER} must be up to 128 letters, digits, ., - or _")
    return session_id


def run_summary(result: AgentRun) -> RunSummary:
    state = result.state
    return RunSummary(verdict=verdict(result), steps=len(state.get("tool_results") or []),
                      tokens=state.get("tokens") or 0, seconds=round(result.seconds, 2),
                      visits=dict(state.get("visits") or {}), stop_reason=state.get("stop_reason"),
                      tool_calls=[ToolCallSummary(tool=r.call.name, arguments=r.call.arguments, error=r.error_type)
                                  for r in state.get("tool_results") or []])


def cacheable(answer: AnswerContract) -> bool:
    """A verified answer or a refusal: an unverified one may be a time-out or a passing failure worth retrying."""
    return answer.verified or answer.refused


def rate_limited(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """429 in the API's error shape, with Retry-After: the limit's window."""
    window = exc.limit.limit.get_expiry() if getattr(exc, "limit", None) is not None else 60
    return JSONResponse(status_code=429, content={"detail": f"rate limit exceeded: {exc.detail}"},
                        headers={"Retry-After": str(int(window))})


# ----------------------------------------------------------------------------------------------- app
def create_app(graph: Any = None, config: ApiConfig | None = None,
               warm_up: Callable[[], dict[str, Any]] | None = None) -> FastAPI:
    """The app. With `graph` it serves that graph with `config` (default ApiConfig()); without, it builds the real
    graph from .env when it starts. `warm_up` runs on a thread at start-up (default real_warm_up when the graph is
    the real one, else nothing)."""
    from core.pii import install_log_masking
    install_log_masking()                       # W2-13: no e-mail or phone number from a question reaches a log
    state = _State(config or ApiConfig(), graph)

    def session_key(request: Request) -> str:
        """The per-session limit's key: the hash of an id this server issued (slowapi logs keys); a request with no
        issued id starts a conversation and gets a bucket of its own, the per-address limit being its ceiling
        (review: behind the UI's server, every new conversation would otherwise share one address bucket)."""
        sent = request.headers.get(SESSION_HEADER)
        return f"session:{session_digest(sent)}" if sent and state.ids.valid(sent) else f"new:{uuid.uuid4().hex}"

    app = FastAPI(title="FinSight", version="0.1.0", lifespan=_lifespan(state, warm_up),
                  description="Questions about US companies' SEC 10-K and 10-Q filings, answered with citations.")
    limiter = Limiter(key_func=session_key)
    app.state.limiter = limiter
    app.state.finsight = state
    app.add_exception_handler(RateLimitExceeded, rate_limited)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        # one line here: Starlette re-raises after this handler, so the server logs the traceback itself
        log.error("unhandled %s on %s", type(exc).__name__, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    @app.middleware("http")
    async def limit_body(request: Request, call_next: Callable[[Request], Any]) -> Any:
        """A body larger than MAX_BODY_BYTES is refused before it is read (a valid one is a few kilobytes)."""
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            return JSONResponse(status_code=413, content={"detail": "request body too large"})
        return await call_next(request)

    @app.get("/health", response_model=HealthResponse)
    async def health(response: Response) -> HealthResponse:
        """200 when ready, 503 while starting or degraded (for a container healthcheck), the body either way."""
        return _health(state, response)

    # The session limit is checked first (slowapi checks the decorator nearest the function first), so a request it
    # refuses costs no hit of the per-address limit, which all of a proxy's users share; the header is validated in a
    # dependency, before either limit is counted.
    @app.post("/ask", response_model=AskResponse)
    @limiter.limit(lambda: state.config.address_rate_limit, key_func=get_remote_address)
    @limiter.limit(lambda: state.config.rate_limit)
    async def ask(request: Request, response: Response, body: AskRequest,
                  session_id: Annotated[str | None, Depends(session_header)] = None) -> AskResponse:
        return await _ask(state, response, body, session_id)

    return app


def _lifespan(state: _State, warm_up: Callable[[], dict[str, Any]] | None) -> Callable[[FastAPI], Any]:
    """Start-up and shut-down: a given graph (tests) needs no settings and no client; else the real graph on one
    LLM client for the app's life, and the warm-up on a worker thread (its result goes to /health)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if state.graph is not None:
            task = asyncio.create_task(_warm(state, warm_up)) if warm_up else None
            if task is None:
                state.warm_up = "ok"
            yield
            return
        from agent.graph import build_graph
        from agent.nodes import real_nodes
        from core.llm import LLMClient
        from core.settings import get_settings
        settings = get_settings()                   # an incomplete .env stops the start-up here, loudly
        state.configure(ApiConfig.from_settings(settings))
        async with LLMClient(settings) as llm:
            state.graph = build_graph(real_nodes(llm))
            task = asyncio.create_task(_warm(state, warm_up or real_warm_up))
            try:
                yield
            finally:
                task.cancel()                       # a warm-up still running on its thread is left to finish

    return lifespan


def _health(state: _State, response: Response) -> HealthResponse:
    status = "ok" if state.warm_up == "ok" else "starting" if state.warm_up == "pending" else "degraded"
    if state.graph is None:
        status = "starting"
    if status != "ok":
        response.status_code = 503
    return HealthResponse(status=status, warm_up=state.warm_up, model=state.config.model,
                          cache_entries=len(state.cache), cache_hits=state.cache.hits,
                          cache_misses=state.cache.misses, cache_hit_rate=state.cache.hit_rate,
                          sessions=len(state.sessions))


async def _ask(state: _State, response: Response, body: AskRequest, session_id: str | None) -> AskResponse:
    """/ask after the rate limits: the session (a new signed id unless a valid one was sent), its history, the reply
    cache, else the agent; a session's questions one at a time (W2-06)."""
    started = time.monotonic()
    if state.graph is None:                         # only without the lifespan (an app driven without starting it)
        raise HTTPException(status_code=503, detail="starting up")
    sid = session_id if state.ids.valid(session_id) else state.ids.issue()
    response.headers[SESSION_HEADER] = sid
    async with state.sessions.hold(sid) as session:
        history = cap_history(body.history if body.history is not None else session.history)
        if history and normalise(history[-1]) == normalise(body.question):
            history = history[:-1]                  # the last question again (a client's retry): the same turn
        key = cache_key(body.question, history)
        hit = state.cache.get(key)
        if hit is not None:
            answer, summary = hit
        else:
            result = await _answer(state, body.question, history, sid)
            answer, summary = result.answer, run_summary(result)
            if cacheable(answer):
                state.cache.put(key, (answer, summary))
        state.sessions.remember(sid, [*history, body.question])
    return AskResponse(session_id=sid, cached=hit is not None, seconds=round(time.monotonic() - started, 2),
                       history=history, answer=answer, run=summary)


async def _answer(state: _State, question: str, history: list[str], session_id: str) -> AgentRun:
    """The agent's run, in one of API_MAX_CONCURRENT slots; 503 when no slot frees within API_QUEUE_TIMEOUT_S."""
    try:
        await asyncio.wait_for(state.slots.acquire(), timeout=state.config.queue_timeout_s)
    except TimeoutError:
        raise HTTPException(status_code=503, detail="busy: too many questions at once, try again shortly",
                            headers={"Retry-After": str(int(state.config.queue_timeout_s)),
                                     SESSION_HEADER: session_id}) from None
    try:
        return await run(question, history, graph=state.graph, limits=state.config.limits,
                         trace=_trace(state.config, session_id))
    finally:
        state.slots.release()


async def _warm(state: _State, warm_up: Callable[[], dict[str, Any]]) -> None:
    """The warm-up on a worker thread; its result goes to /health."""
    try:
        result = await asyncio.to_thread(warm_up)
    except Exception:                               # noqa: BLE001 - a failed warm-up is reported, not fatal
        log.exception("warm-up failed")
        state.warm_up = "internal_error"
        return
    error = result.get("error") if isinstance(result, dict) else None
    state.warm_up = "ok" if error is None else str(error.get("type", "error")) if isinstance(error, dict) else "error"


def real_warm_up() -> dict[str, Any]:
    """The real app's warm-up: the fact store, the index and one search (tools.registry.warm_up), the agent's
    one-time loads (agent.nodes.warm_up: prompts, schemas, config files, company names), then the Qwen tokenizer
    the executor counts with, so none of them is loaded during the first question."""
    from agent.nodes import warm_up as warm_agent
    from core.exceptions import FinSightError
    from core.tokens import count_tokens
    from tools.registry import warm_up
    result = warm_up()
    warm_agent()
    try:
        count_tokens("warm up")
    except FinSightError as err:                    # the executor then estimates (agent/executor.py)
        log.warning("tokenizer not loaded: %s", err)
    return result


def _trace(config: ApiConfig, session_id: str) -> Any:
    """Langfuse tracing when .env turns it on (W2-08): one trace per question (q:<request id>, run:api), a
    conversation's turns grouped by its session."""
    if not config.trace:
        return None
    from core.tracing import TraceInfo
    return TraceInfo(question_id=uuid.uuid4().hex, run_id="api", session_id=session_digest(session_id))


app = create_app()
