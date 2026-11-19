"""Langfuse tracing behind a small interface (W2-08, ADR-029).

Design:
  * One tracer per process, found through a context variable, so the graph, the executor and the LLM client call
    `span(...)` / `generation(...)` without receiving a tracer argument. With no tracer set (the default, and every
    test) the calls are no-ops: no network, no import of `langfuse`.
  * A run is traced only when the caller passes `TraceInfo` to `agent.graph.run`; the tracer is then
    `default_tracer()` (from LANGFUSE_* in .env) or the one the caller gives (tests pass a recording fake).
  * Everything sent is first passed through `scrub`: e-mail addresses and phone numbers are masked (plan §10) and long
    texts are cut. A failure inside Langfuse is logged and ignored: tracing never changes an answer.
  * This module imports nothing from agent/, tools/ or the rest of core/ at module level (core/llm.py and agent/ import it).

All Langfuse SDK calls (Python SDK v3, `langfuse>=3,<4`) are in `LangfuseTracer`; swapping the SDK means editing it only.
"""
from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from typing import Any, Protocol

log = logging.getLogger(__name__)

MAX_TEXT_CHARS = 4000          # per string sent to Langfuse; the full text stays in the run's own records
MAX_ITEMS = 50                 # per list
MAX_DEPTH = 6
MASK_EMAIL = "[email]"
MASK_PHONE = "[phone]"
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Three shapes only, so a run of years ("2023 2024 2025"), a bare figure (383285000000), a date (2024-09-28) or an
# accession number (0000320193-24-000123) is never taken for a phone number: +CC groups, (area) groups, 3-3/4-4 groups
# joined by hyphens or dots.
PHONE = re.compile(r"(?<![\w.+-])(?:\+\d{1,3}[\s.-]?\d{2,5}(?:[\s.-]\d{2,5}){1,3}"
                   r"|\(\d{2,4}\)[\s.-]?\d{3,4}[\s.-]\d{3,4}"
                   r"|\d{3}[.-]\d{3,4}[.-]\d{4})(?![\w-])")


# ----------------------------------------------------------------------------------------------- scrubbing
def scrub(value: Any, _depth: int = 0) -> Any:
    """`value` as plain JSON-able data with e-mails and phone numbers masked and long texts cut."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        text = PHONE.sub(MASK_PHONE, EMAIL.sub(MASK_EMAIL, value))
        if len(text) > MAX_TEXT_CHARS:
            return f"{text[:MAX_TEXT_CHARS]}…[{len(text) - MAX_TEXT_CHARS} more characters]"
        return text
    if _depth >= MAX_DEPTH:
        return scrub(str(value), _depth)
    dump = getattr(value, "model_dump", None)           # a pydantic model
    if callable(dump):
        return scrub(dump(mode="json"), _depth + 1)
    if isinstance(value, dict):
        return {str(k): scrub(v, _depth + 1) for k, v in list(value.items())[:MAX_ITEMS]}
    if isinstance(value, list | tuple | set | frozenset):
        items = [scrub(v, _depth + 1) for v in list(value)[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"…[{len(value) - MAX_ITEMS} more items]")
        return items
    return scrub(str(value), _depth)


# ----------------------------------------------------------------------------------------------- the interface
class Span(Protocol):
    def tag(self, *tags: str) -> None: ...    # the trace's tags (only the root span of a run has them; no-op elsewhere)

    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None, level: str | None = None,
               status_message: str | None = None, usage: dict[str, int] | None = None) -> None: ...


class Tracer(Protocol):
    enabled: bool

    def span(self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None,
             ) -> contextlib.AbstractContextManager[Span]: ...

    def generation(self, name: str, *, model: str, input: Any = None, metadata: dict[str, Any] | None = None,
                   ) -> contextlib.AbstractContextManager[Span]: ...

    def trace(self, info: TraceInfo, *, input: Any = None) -> contextlib.AbstractContextManager[Span]: ...

    def flush(self) -> None: ...


@dataclass(frozen=True)
class TraceInfo:
    """Who a traced run belongs to. `question_id` (a gold-set id such as 'B07', or an API request id) and `run_id`
    become the tags `q:<question_id>` and `run:<run_id>`; `session_id` groups the turns of one conversation."""
    question_id: str
    run_id: str
    session_id: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.question_id.strip() or not self.run_id.strip():
            raise ValueError("TraceInfo needs a question_id and a run_id")

    def all_tags(self, *extra: str) -> list[str]:
        return list(dict.fromkeys([f"q:{self.question_id}", f"run:{self.run_id}", *self.tags, *extra]))


class _NoopSpan:
    def tag(self, *tags: str) -> None:
        return None

    def update(self, **_: Any) -> None:
        return None


class NoopTracer:
    enabled = False

    @contextlib.contextmanager
    def span(self, name: str, **_: Any) -> Iterator[Span]:
        yield _NoopSpan()

    generation = span

    @contextlib.contextmanager
    def trace(self, info: TraceInfo, **_: Any) -> Iterator[Span]:
        yield _NoopSpan()

    def flush(self) -> None:
        return None


NOOP_TRACER: Tracer = NoopTracer()
_current: ContextVar[Tracer] = ContextVar("finsight_tracer", default=NOOP_TRACER)


# ----------------------------------------------------------------------------------------------- Langfuse
class _LangfuseSpan:
    def __init__(self, raw: Any, client: Any = None, info: TraceInfo | None = None) -> None:
        self.raw, self.client, self.info = raw, client, info

    def tag(self, *tags: str) -> None:
        if self.info is None:
            return
        try:
            self.client.update_current_trace(tags=self.info.all_tags(*tags))
        except Exception:                                   # noqa: BLE001
            log.warning("langfuse: could not tag the trace", exc_info=True)

    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None, level: str | None = None,
               status_message: str | None = None, usage: dict[str, int] | None = None) -> None:
        fields = {"output": scrub(output) if output is not None else None,
                  "metadata": scrub(metadata) if metadata else None, "level": level,
                  "status_message": status_message, "usage_details": usage}
        try:
            self.raw.update(**{k: v for k, v in fields.items() if v is not None})
        except Exception:                                   # noqa: BLE001 - tracing never breaks a run
            log.warning("langfuse: span update failed", exc_info=True)


class LangfuseTracer:
    """The tracer on a `langfuse.Langfuse` client (or any object with the same four methods; tests pass a fake)."""
    enabled = True

    def __init__(self, client: Any) -> None:
        self.client = client

    @classmethod
    def from_settings(cls, settings: Any) -> LangfuseTracer:
        try:
            from langfuse import Langfuse
        except ImportError as err:
            raise ImportError("LANGFUSE_ENABLED=true needs the langfuse package: pip install -r requirements.txt "
                              "(langfuse>=3,<4)") from err
        return cls(Langfuse(public_key=settings.LANGFUSE_PUBLIC_KEY.get_secret_value(),
                            secret_key=settings.LANGFUSE_SECRET_KEY.get_secret_value(),
                            host=settings.LANGFUSE_HOST))

    @contextlib.contextmanager
    def _open(self, factory: Any, after_enter: Any = None, info: TraceInfo | None = None) -> Iterator[Span]:
        """Enter a Langfuse context manager without letting its failures reach the caller's code, and close it
        with the caller's exception (so an error inside the block marks the span as failed)."""
        cm = raw = None
        try:
            cm = factory()
            raw = cm.__enter__()
            if after_enter:
                after_enter()
        except Exception:                                   # noqa: BLE001
            log.warning("langfuse: could not open a span", exc_info=True)
        handle: Span = _LangfuseSpan(raw, self.client, info) if raw is not None else _NoopSpan()
        try:
            yield handle
        except BaseException as err:                       # incl. CancelledError: the span still closes
            self._close(cm, type(err), err, err.__traceback__)
            raise
        else:
            self._close(cm, None, None, None)

    @staticmethod
    def _close(cm: Any, *exc_info: Any) -> None:
        if cm is None:
            return
        try:
            cm.__exit__(*exc_info)
        except Exception:                                   # noqa: BLE001
            log.warning("langfuse: could not close a span", exc_info=True)

    def span(self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None,
             ) -> contextlib.AbstractContextManager[Span]:
        return self._open(lambda: self.client.start_as_current_span(
            name=name, input=scrub(input), metadata=scrub(metadata) if metadata else None))

    def generation(self, name: str, *, model: str, input: Any = None, metadata: dict[str, Any] | None = None,
                   ) -> contextlib.AbstractContextManager[Span]:
        return self._open(lambda: self.client.start_as_current_generation(
            name=name, model=model, input=scrub(input), metadata=scrub(metadata) if metadata else None))

    def trace(self, info: TraceInfo, *, input: Any = None) -> contextlib.AbstractContextManager[Span]:
        def tag_trace() -> None:
            self.client.update_current_trace(session_id=info.session_id, tags=info.all_tags(),
                                             metadata={"question_id": info.question_id, "run_id": info.run_id},
                                             input=scrub(input))
        return self._open(lambda: self.client.start_as_current_span(name="finsight.ask", input=scrub(input)),
                          after_enter=tag_trace, info=info)

    def flush(self) -> None:
        try:
            self.client.flush()
        except Exception:                                   # noqa: BLE001
            log.warning("langfuse: flush failed", exc_info=True)


# ----------------------------------------------------------------------------------------------- entry points
def make_tracer(settings: Any) -> Tracer:
    """The tracer the settings ask for: Langfuse when LANGFUSE_ENABLED, else the no-op one (no import, no network)."""
    if not getattr(settings, "LANGFUSE_ENABLED", False):
        return NOOP_TRACER
    return LangfuseTracer.from_settings(settings)


@cache
def default_tracer() -> Tracer:
    """The tracer for this process, from `.env` (read once). Callers that trace a run get it unless they pass one."""
    from core.settings import get_settings
    return make_tracer(get_settings())


def span(name: str, *, input: Any = None, metadata: dict[str, Any] | None = None,
         ) -> contextlib.AbstractContextManager[Span]:
    """A span under the current one; a no-op when no run is being traced."""
    return _current.get().span(name, input=input, metadata=metadata)


def generation(name: str, *, model: str, input: Any = None, metadata: dict[str, Any] | None = None,
               ) -> contextlib.AbstractContextManager[Span]:
    """An LLM call (model, tokens, latency); a no-op when no run is being traced."""
    return _current.get().generation(name, model=model, input=input, metadata=metadata)


@contextlib.contextmanager
def trace_run(info: TraceInfo | None, question: str, *, tracer: Tracer | None = None) -> Iterator[Span]:
    """The root of one question's trace: makes `tracer` (default: `default_tracer()`) the current one for everything
    inside, and tags the trace. `info=None` traces nothing, and does not even read the settings."""
    if info is None:
        yield _NoopSpan()
        return
    chosen = tracer or default_tracer()
    token = _current.set(chosen)
    try:
        with chosen.trace(info, input=question) as root:
            yield root
    finally:
        _current.reset(token)


def flush() -> None:
    """Send what is buffered (call at the end of a script; the API calls it at shutdown)."""
    _current.get().flush()
    if default_tracer.cache_info().currsize:
        default_tracer().flush()
