"""Conversations (W2-09, ADR-032): each session's earlier questions, and a lock that answers its questions one at a
time.

    Sessions(max_sessions, ttl_s)   get(session_id) -> Session (made on first use); remember(session_id, questions)
    new_session_id()                a random id for a client that sent none

The API keeps the earlier questions itself and gives them to agent.graph.run as an explicit history (capped by
agent.graph.cap_history), instead of a checkpointer thread: a reply-cache hit never runs the graph, so a thread would
miss that turn and the follow-up after it would lose its company and year (W2-06). A session's questions run one at
a time (W2-06: two overlapping runs would read the same history). In memory, per process: lost on restart.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from agent.graph import cap_history


def new_session_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Session:
    history: list[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    seen: float = 0.0


class Sessions:
    """At most `max_sessions` sessions, each forgotten `ttl_s` seconds after its last question (least recently used
    dropped first); a session whose question is running is never dropped, so its lock stays the one in use."""

    def __init__(self, max_sessions: int, ttl_s: float, clock: Callable[[], float] = time.monotonic):
        self.max_sessions, self.ttl_s, self._clock = max_sessions, ttl_s, clock
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def __len__(self) -> int:
        return len(self._sessions)

    def get(self, session_id: str) -> Session:
        now = self._clock()
        session = self._sessions.get(session_id)
        if session is None or (now - session.seen >= self.ttl_s and not session.lock.locked()):
            session = Session()
            self._sessions[session_id] = session
        session.seen = now
        self._sessions.move_to_end(session_id)
        self._drop_old(now, keep=session_id)
        return session

    def remember(self, session_id: str, questions: Sequence[str]) -> None:
        """The session's earlier questions after a turn (the history it was answered with, then its question)."""
        session = self._sessions.get(session_id)
        if session is not None:
            session.history = cap_history(questions)
            session.seen = self._clock()

    def _drop_old(self, now: float, keep: str) -> None:
        for sid in list(self._sessions):
            if len(self._sessions) <= self.max_sessions and now - self._sessions[sid].seen < self.ttl_s:
                break                                 # oldest first: the rest are newer
            if sid != keep and not self._sessions[sid].lock.locked():
                del self._sessions[sid]
