"""The UI's client of the FinSight API (W2-10, ADR-033): the only way the UI reaches the agent (no model access).

    ApiClient(base_url, timeout)   ask(question, session_id) -> Reply; health() -> dict or None
    api_url()                      FINSIGHT_API_URL, else http://127.0.0.1:8000

Every outcome is a Reply, never an exception: an answer (ok), or a message the UI can show as it is: a refused
request (422), the rate limit with its wait (429), the API busy or starting with its wait (503), an error inside the
agent (500), a response that is not an answer, or the API unreachable. The session id of the latest response is the
one to send next (W2-09: after an API restart an old id starts a new session), when it has the API's own shape (an
id that does not would make every later question fail, so it is ignored). The read timeout leaves room for a
session's queue: 30 s waiting for a slot plus a 60 s run; connecting gets 5 s, so a stopped API is reported at once.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"
SESSION_HEADER = "X-Session-Id"
TIMEOUT_S = 120.0
CONNECT_TIMEOUT_S = 5.0
HEALTH_TIMEOUT_S = 3.0
SESSION_ID_SHAPE = re.compile(r"[A-Za-z0-9._-]{1,128}")     # api/main.py SESSION_HEADER_SHAPE


def api_url() -> str:
    return os.environ.get("FINSIGHT_API_URL", DEFAULT_API_URL).rstrip("/")


@dataclass(frozen=True)
class Reply:
    """One /ask outcome: `body` is the API's AskResponse when ok, else `error` says what happened."""
    ok: bool
    status: int | None = None
    body: dict[str, Any] | None = None
    error: str | None = None
    session_id: str | None = None


@dataclass
class Conversation:
    """One chat: its messages, the session id to send next, and the question being asked (None when idle).

    The UI keeps one in the Streamlit session and changes it only through plain attribute writes: Streamlit can stop
    a run at any st.* call and at any read or write of st.session_state (a user's click while a question is asked),
    so a reply recorded here, on an object fetched before the request, cannot be lost between the API's answer and
    the session."""
    messages: list[dict[str, Any]] = field(default_factory=list)
    session_id: str | None = None
    pending: str | None = None

    def ask(self, question: str) -> None:
        self.messages.append({"role": "user", "text": question})
        self.pending = question

    def record(self, reply: Reply) -> None:
        if reply.session_id:
            self.session_id = reply.session_id            # always the latest (W2-09)
        self.messages.append({"role": "assistant", "body": reply.body, "session_id": self.session_id} if reply.ok
                             else {"role": "assistant", "error": reply.error})
        self.pending = None

    def clear(self) -> None:
        self.messages, self.session_id, self.pending = [], None, None


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:                               # not JSON, bad encoding, or nested past the recursion limit
        return None


def _detail(response: httpx.Response) -> str:
    payload = _json(response)
    if not isinstance(payload, dict):
        return response.text[:200].strip() or "no detail"
    detail = payload.get("detail")
    if isinstance(detail, list):                    # FastAPI's validation errors: [{"msg": ...}, ...]
        return "; ".join(str(d.get("msg", d)) for d in detail if isinstance(d, dict)) or "invalid request"
    return str(detail) if detail is not None else "no detail"


def _seconds(value: str | None, default: int) -> int:
    """A Retry-After in seconds (the API sends whole seconds; anything else falls back to the default)."""
    return int(value) if value and value.isascii() and value.isdigit() else default


def _session_id(value: Any) -> str | None:
    """A session id with the API's shape, else None (sending anything else would fail every later question)."""
    return value if isinstance(value, str) and value.isascii() and SESSION_ID_SHAPE.fullmatch(value) else None


def is_answer(body: Any) -> bool:
    """The AskResponse shape the card reads: a dict with an `answer` dict holding its text, and a `run` dict."""
    return (isinstance(body, dict) and isinstance(body.get("answer"), dict)
            and isinstance(body["answer"].get("answer"), str) and isinstance(body.get("run"), dict))


class ApiClient:
    """The API over HTTP (`transport` lets tests answer without a server)."""

    def __init__(self, base_url: str | None = None, timeout: float = TIMEOUT_S,
                 transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or api_url()).rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, transport=transport,
                                  timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT_S))

    def close(self) -> None:
        self._http.close()

    def ask(self, question: str, session_id: str | None = None) -> Reply:
        session_id = _session_id(session_id)
        headers = {SESSION_HEADER: session_id} if session_id else {}
        try:
            response = self._http.post("/ask", json={"question": question}, headers=headers)
        except httpx.TimeoutException:
            return Reply(ok=False, error="The answer took too long. Try again in a moment.")
        except httpx.HTTPError:
            return Reply(ok=False, error=f"The FinSight API is not reachable at {self.base_url}.")
        except Exception:                           # e.g. a question httpx cannot encode
            return Reply(ok=False, error="The question could not be sent. Try rephrasing it.")
        sid = _session_id(response.headers.get(SESSION_HEADER))
        if response.status_code == 200:
            body = _json(response)
            if not is_answer(body):
                return Reply(ok=False, status=200, session_id=sid,
                             error="The API's response was not an answer FinSight can show. Try again.")
            return Reply(ok=True, status=200, body=body, session_id=_session_id(body.get("session_id")) or sid)
        wait = response.headers.get("Retry-After")
        status = response.status_code
        if status == 503:
            starting = "starting" in _detail(response).lower()
            message = (f"FinSight is still starting. Try again in {_seconds(wait, 30)} seconds." if starting else
                       f"FinSight is busy with other questions. Try again in {_seconds(wait, 30)} seconds.")
        else:
            message = {
                422: f"The question could not be sent: {_detail(response)}.",
                429: f"Too many questions in a short time. Try again in {_seconds(wait, 60)} seconds.",
                500: "FinSight hit an internal error on this question. Try rephrasing it.",
                413: "The question is too long.",
            }.get(status, f"The API answered {status}: {_detail(response)}")
        return Reply(ok=False, status=status, error=message, session_id=sid)

    def health(self) -> dict[str, Any] | None:
        """The API's /health body (also when it answers 503 while starting), or None when unreachable or not JSON."""
        try:
            body = self._http.get("/health", timeout=HEALTH_TIMEOUT_S).json()
        except Exception:                           # unreachable, or a body that is not JSON
            return None
        return body if isinstance(body, dict) else None
