"""Conversations (W2-09, ADR-032): each session's earlier questions, a lock that answers its questions one at a
time, and the session ids the server issues.

    SessionIds(secret)              issue() -> a signed id; valid(session_id) -> whether this server issued it
    Sessions(max_sessions, ttl_s)   hold(session_id): the session, its lock held, for one question;
                                    remember(session_id, questions)

The API keeps the earlier questions itself and gives them to agent.graph.run as an explicit history (capped by
agent.graph.cap_history), instead of a checkpointer thread: a reply-cache hit never runs the graph, so a thread would
miss that turn and the follow-up after it would lose its company and year (W2-06). A session's questions run one at a
time (W2-06: two overlapping runs would read the same history).

Ids are issued by the server and signed (HMAC-SHA256 with the server's secret), so a client cannot mint ids to get a
fresh rate-limit bucket per request or to push real users' sessions out of the store (review): an id this server did
not issue starts a new session. In memory, per process: lost on restart (an old id then starts a new session).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from agent.graph import cap_history

TOKEN_BYTES = 16
SIGNATURE_HEX = 16


class SessionIds:
    """Signed session ids: "<32 hex>.<16 hex>", the random part and its truncated HMAC."""

    def __init__(self, secret: bytes | None = None):
        self._secret = secret or secrets.token_bytes(32)

    def _sign(self, token: str) -> str:
        return hmac.new(self._secret, token.encode(), hashlib.sha256).hexdigest()[:SIGNATURE_HEX]

    def issue(self) -> str:
        token = secrets.token_hex(TOKEN_BYTES)
        return f"{token}.{self._sign(token)}"

    def valid(self, session_id: str | None) -> bool:
        if not session_id or session_id.count(".") != 1:
            return False
        token, signature = session_id.split(".")
        return len(token) == 2 * TOKEN_BYTES and hmac.compare_digest(signature, self._sign(token))


@dataclass
class Session:
    history: list[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    seen: float = 0.0
    users: int = 0                                  # requests holding or waiting for this session


class Sessions:
    """At most `max_sessions` sessions, each forgotten `ttl_s` seconds after its last question (least recently used
    dropped first). A session some request holds or waits for is never dropped or replaced: its lock must stay the
    one in use (review: a `locked()` check misses a waiter between the holder's release and its own resumption)."""

    def __init__(self, max_sessions: int, ttl_s: float, clock: Callable[[], float] = time.monotonic):
        self.max_sessions, self.ttl_s, self._clock = max_sessions, ttl_s, clock
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def __len__(self) -> int:
        return len(self._sessions)

    @asynccontextmanager
    async def hold(self, session_id: str) -> AsyncIterator[Session]:
        """The session for one question, its lock held (the questions of a session one at a time)."""
        session = self._get(session_id)
        session.users += 1
        try:
            async with session.lock:
                yield session
        finally:
            session.users -= 1

    def remember(self, session_id: str, questions: Sequence[str]) -> None:
        """The session's earlier questions after a turn (the history it was answered with, then its question)."""
        session = self._sessions.get(session_id)
        if session is not None:
            session.history = cap_history(questions)
            session.seen = self._clock()
            self._sessions.move_to_end(session_id)

    def _get(self, session_id: str) -> Session:
        now = self._clock()
        session = self._sessions.get(session_id)
        if session is None or (now - session.seen >= self.ttl_s and session.users == 0):
            session = Session()
            self._sessions[session_id] = session
        session.seen = now
        self._sessions.move_to_end(session_id)
        self._drop_old(now, keep=session_id)
        return session

    def _drop_old(self, now: float, keep: str) -> None:
        for sid in list(self._sessions):
            if len(self._sessions) <= self.max_sessions and now - self._sessions[sid].seen < self.ttl_s:
                break                                 # oldest first: the rest are newer
            if sid != keep and self._sessions[sid].users == 0:
                del self._sessions[sid]
