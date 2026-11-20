"""The UI's client of the FinSight API (W2-10, ADR-033): the only way the UI reaches the agent (no model access).

    ApiClient(base_url, timeout)   ask(question, session_id) -> Reply; health() -> dict or None
    api_url()                      FINSIGHT_API_URL, else http://127.0.0.1:8000

Every outcome is a Reply, never an exception: an answer (ok), or a message the UI can show as it is: a refused
request (422), the rate limit with its wait (429), the API busy or starting (503), an error inside the agent (500),
or the API unreachable. The session id of the latest response is the one to send next (W2-09: after an API restart an
old id starts a new session). The timeout leaves room for a session's queue: 30 s waiting for a slot plus a 60 s run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"
SESSION_HEADER = "X-Session-Id"
TIMEOUT_S = 120.0


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


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:200]
    if isinstance(detail, list):                    # FastAPI's validation errors: [{"msg": ...}, ...]
        return "; ".join(str(d.get("msg", d)) for d in detail if isinstance(d, dict)) or "invalid request"
    return str(detail)


class ApiClient:
    """The API over HTTP (`transport` lets tests answer without a server)."""

    def __init__(self, base_url: str | None = None, timeout: float = TIMEOUT_S,
                 transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or api_url()).rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._http.close()

    def ask(self, question: str, session_id: str | None = None) -> Reply:
        headers = {SESSION_HEADER: session_id} if session_id else {}
        try:
            response = self._http.post("/ask", json={"question": question}, headers=headers)
        except httpx.TimeoutException:
            return Reply(ok=False, error="The answer took too long. Try again in a moment.")
        except httpx.HTTPError:
            return Reply(ok=False, error=f"The FinSight API is not reachable at {self.base_url}.")
        sid = response.headers.get(SESSION_HEADER)
        if response.status_code == 200:
            body = response.json()
            return Reply(ok=True, status=200, body=body, session_id=body.get("session_id") or sid)
        wait = response.headers.get("Retry-After")
        message = {
            422: f"The question could not be sent: {_detail(response)}.",
            429: f"Too many questions in a short time. Try again in {wait or 60} seconds.",
            503: "FinSight is busy or still starting. Try again shortly.",
            500: "FinSight hit an internal error on this question. Try rephrasing it.",
            413: "The question is too long.",
        }.get(response.status_code, f"The API answered {response.status_code}: {_detail(response)}")
        return Reply(ok=False, status=response.status_code, error=message, session_id=sid)

    def health(self) -> dict[str, Any] | None:
        """The API's /health body (also when it answers 503 while starting), or None when unreachable."""
        try:
            return self._http.get("/health", timeout=5.0).json()
        except (httpx.HTTPError, ValueError):
            return None
