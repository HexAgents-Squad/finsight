"""The reply cache (W2-09, ADR-032): an answer kept under its normalised question and the earlier questions it was
asked with, so asking again costs no request to the shared endpoint.

    normalise(text)                 the form a question is compared in
    cache_key(question, history)    the key: the normalised question and each normalised earlier question
    ReplyCache(max_entries, ttl_s)  bounded (least recently used dropped first) and expiring; get / put; its hits,
                                    misses and hit rate since start-up (W3-07, shown by /health)

In memory, per process: the API runs one process with one event loop, so the cache needs no lock. The history is
part of the key because a follow-up ("And in fiscal 2023?") means a different question after a different turn.
"""
from __future__ import annotations

import re
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from typing import Any

WHITESPACE = re.compile(r"\s+")
TRAILING_PUNCTUATION = re.compile(r"[\s?.!]+$")
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'"})


def normalise(text: str) -> str:
    """NFKC, case-folded, straight apostrophes, one space between words, no trailing "?", "." or "!":
    "What was Apple’s  revenue in fiscal 2024?" and "what was apple's revenue in fiscal 2024" are one question."""
    folded = unicodedata.normalize("NFKC", text).translate(APOSTROPHES).casefold()
    return TRAILING_PUNCTUATION.sub("", WHITESPACE.sub(" ", folded).strip())


def cache_key(question: str, history: Sequence[str]) -> tuple[str, ...]:
    return (normalise(question), *(normalise(q) for q in history))


class ReplyCache:
    """At most `max_entries` values, each valid for `ttl_s` seconds after it was put; reading one makes it the most
    recently used. A cache with no room or no lifetime keeps nothing."""

    def __init__(self, max_entries: int, ttl_s: float, clock: Callable[[], float] = time.monotonic):
        self.max_entries, self.ttl_s, self._clock = max_entries, ttl_s, clock
        self._entries: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self.hits = self.misses = 0                 # lookups since start-up (an expired entry is a miss)

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.ttl_s > 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float | None:
        """The share of lookups answered from the cache, None before the first lookup (and for a disabled cache,
        which counts none: its 0% would read as a cold cache)."""
        lookups = self.hits + self.misses
        return round(self.hits / lookups, 4) if lookups else None

    def get(self, key: Hashable) -> Any | None:
        if not self.enabled:
            return None
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        stored, value = entry
        if self._clock() - stored >= self.ttl_s:
            del self._entries[key]
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: Hashable, value: Any) -> None:
        if not self.enabled:
            return
        self._entries[key] = (self._clock(), value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
