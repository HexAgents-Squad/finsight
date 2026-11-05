"""Hybrid search: Reciprocal Rank Fusion of the BM25 and dense rankings (week0_plan WP5 step 4).

score(chunk) = sum over the two rankings of 1 / (RRF_K + rank), ranks starting at 1, over the top
max(k, CANDIDATES) of each. RRF uses ranks only, so the two retrievers' scores need no calibration.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from retrieval.common import NO_FILTERS, Filters, SearchResult

RRF_K = 60
CANDIDATES = 50


class Retriever(Protocol):
    def search(self, query: str, k: int = 5, filters: Filters = NO_FILTERS) -> list[SearchResult]: ...


def rrf(rankings: list[list[SearchResult]], k: int, rrf_k: int = RRF_K) -> list[SearchResult]:
    """Fuse ranked lists into the top k. Equal scores are ordered by (rank, list number) where the chunk first
    appears, scanning the lists in order; no two chunks share that pair, so the order is fully deterministic."""
    scores: dict[str, float] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    results: dict[str, SearchResult] = {}
    for list_no, ranking in enumerate(rankings):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            first_seen.setdefault(hit.chunk_id, (rank, list_no))
            results.setdefault(hit.chunk_id, hit)
    order = sorted(scores, key=lambda cid: (-scores[cid], first_seen[cid]))
    return [replace(results[cid], score=scores[cid]) for cid in order[:k]]


class HybridIndex:
    def __init__(self, bm25: Retriever, dense: Retriever, candidates: int = CANDIDATES):
        if not isinstance(candidates, int) or isinstance(candidates, bool) or candidates <= 0:
            raise ValueError(f"candidates must be a positive int, not {candidates!r}")
        self.bm25, self.dense, self.candidates = bm25, dense, candidates

    def search(self, query: str, k: int = 5, filters: Filters = NO_FILTERS) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        depth = max(k, self.candidates)
        return rrf([self.bm25.search(query, depth, filters), self.dense.search(query, depth, filters)], k)
