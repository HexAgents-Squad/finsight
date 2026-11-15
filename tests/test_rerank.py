"""retrieval/rerank.py and retrieval/dense.cached (W1-05), offline: a fake scorer stands in for the cross-encoder."""
import pytest

from retrieval.common import NO_FILTERS, Filters, SearchResult
from retrieval.dense import Embedder, cached
from retrieval.rerank import CANDIDATES, RerankedIndex, passage


def hit(cid, text="t", period="FY", item="Item 8"):
    return SearchResult(cid, text, "AAPL", "10-K" if period == "FY" else "10-Q", 2024, period, item, "u", 0.5)


class Base:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, query, k=5, filters=NO_FILTERS):
        self.calls.append((query, k, filters))
        return self.hits[:k]


def by_length(query, passages):
    return [float(len(p)) for p in passages]                     # longer passage, higher score


def test_reranking_reorders_the_candidates_and_keeps_k():
    base = Base([hit("a", "x"), hit("b", "xxx"), hit("c", "xx")])
    top = RerankedIndex(base, by_length).search("q", k=2, filters=Filters.of(ticker="AAPL"))
    assert [h.chunk_id for h in top] == ["b", "c"]
    assert top[0].score == len(passage(hit("b", "xxx")))          # the result carries the cross-encoder's score
    assert base.calls == [("q", CANDIDATES, Filters.of(ticker="AAPL"))]   # asks the base for its top 30


def test_ties_keep_the_base_order_and_k_above_the_candidates_asks_for_k():
    base = Base([hit(c) for c in "abcd"])
    assert [h.chunk_id for h in RerankedIndex(base, lambda q, ps: [1.0] * len(ps)).search("q", 4)] == list("abcd")
    RerankedIndex(base, by_length, candidates=2).search("q", k=3)
    assert base.calls[-1][1] == 3


def test_no_candidates_means_no_scoring():
    def never(query, passages):
        raise AssertionError("the scorer must not be called")
    assert RerankedIndex(Base([]), never).search("q") == []


def test_bad_arguments_and_a_scorer_that_loses_passages():
    with pytest.raises(ValueError):
        RerankedIndex(Base([]), by_length, candidates=0)
    with pytest.raises(ValueError):
        RerankedIndex(Base([hit("a")]), by_length).search("q", k=0)
    with pytest.raises(RuntimeError, match="1 scores for 2 passages"):
        RerankedIndex(Base([hit("a"), hit("b")]), lambda q, ps: [0.0]).search("q")


def test_the_passage_names_the_filing_first():
    assert passage(hit("a", "Total net sales | 391,035")) == "[AAPL 10-K FY2024 · Item 8]\nTotal net sales | 391,035"
    assert passage(hit("b", "x", "Q2", "Part I Item 1")).startswith("[AAPL 10-Q FY2024 Q2 · Part I Item 1]")


def test_the_query_cache_embeds_each_text_once_and_keeps_the_order():
    seen = []

    def embed(texts):
        seen.append(list(texts))
        return [[float(len(t))] for t in texts]

    e = cached(Embedder(embed, "bge-small-en-v1.5"), maxsize=2)
    assert e.embed(["ab", "c", "ab"]) == [[2.0], [1.0], [2.0]]
    assert e.embed(["c"]) == [[1.0]] and seen == [["ab", "c"]]       # the second call is served from the cache
    e.embed(["dddd"])                                                 # the oldest ("ab") is dropped at maxsize 2
    e.embed(["ab"])
    assert seen[-1] == ["ab"] and e.model_id == "bge-small-en-v1.5"


def test_the_query_cache_refuses_a_short_answer():
    e = cached(Embedder(lambda texts: [[0.0]], "m"))
    with pytest.raises(RuntimeError, match="1 vectors for 2 texts"):
        e.embed(["a", "b"])
