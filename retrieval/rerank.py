"""Cross-encoder reranking of the hybrid top 30 (W1-05).

A cross-encoder reads the query and one passage together and scores how well the passage answers the query, which
a bi-encoder (dense) or term matching (BM25) cannot: they score query and passage separately. It is too slow to
run over the corpus, so it reorders the first CANDIDATES results of another retriever. The passage it reads starts
with the chunk's filing (company, form, period, Item), as the chunk header does in the index, because the
question names the company and period and the chunk text often does not.

The model is cross-encoder/ms-marco-MiniLM-L-6-v2 (MS MARCO passage ranking, 22M parameters, 512 tokens), run
locally with sentence-transformers and pinned to a revision; a longer passage is cut to 512 tokens.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from retrieval.common import NO_FILTERS, Filters, SearchResult
from retrieval.hybrid import Retriever

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"         # refs/main when pinned (W1-05)
MAX_TOKENS = 512
CANDIDATES = 30

Score = Callable[[str, list[str]], list[float]]


def passage(hit: SearchResult) -> str:
    """What the cross-encoder reads: the filing, then the chunk text."""
    period = "" if hit.fiscal_period == "FY" else f" {hit.fiscal_period}"
    return f"[{hit.ticker} {hit.form} FY{hit.fiscal_year}{period} · {hit.item}]\n{hit.text}"


def cross_encoder(model: str = MODEL, revision: str = MODEL_REVISION) -> Score:
    """The pinned cross-encoder as a scoring function, loaded once (heavy import, only where reranking runs)."""
    from sentence_transformers import CrossEncoder
    encoder = CrossEncoder(model, revision=revision, max_length=MAX_TOKENS)

    def score(query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in encoder.predict([(query, p) for p in passages], show_progress_bar=False)]

    return score


class RerankedIndex:
    """`base`'s top `candidates`, reordered by the cross-encoder; ties keep `base`'s order. The score of a result
    is the cross-encoder's."""

    def __init__(self, base: Retriever, score: Score, candidates: int = CANDIDATES):
        if not isinstance(candidates, int) or isinstance(candidates, bool) or candidates <= 0:
            raise ValueError(f"candidates must be a positive int, not {candidates!r}")
        self.base, self.score, self.candidates = base, score, candidates

    def search(self, query: str, k: int = 5, filters: Filters = NO_FILTERS) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        hits = self.base.search(query, max(k, self.candidates), filters)
        if not hits:
            return []
        scores = self.score(query, [passage(h) for h in hits])
        if len(scores) != len(hits):
            raise RuntimeError(f"the reranker returned {len(scores)} scores for {len(hits)} passages")
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
        return [replace(hits[i], score=scores[i]) for i in order[:k]]
