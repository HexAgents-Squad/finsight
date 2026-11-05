"""retrieval/: filters, BM25, dense (fake and real Chroma) and RRF hybrid search (offline)."""
import pytest

from retrieval.bm25 import INDEX_FORMAT, BM25Index, tokenize
from retrieval.common import NO_FILTERS, Chunk, Filters, SearchResult
from retrieval.dense import DenseIndex, Embedder, build_collection, where_clause
from retrieval.hybrid import RRF_K, HybridIndex, rrf


def chunk(n, text, ticker="AAPL", year=2024, item="Item 7", period="FY", form="10-K", kind="text"):
    return Chunk(f"{ticker}_{form}_{year}_{item.replace(' ', '')}_C2_{n:04d}", f"{ticker}_{year}_{item}", "C2",
                 kind, f"[{ticker} {form} FY{year} · {item}]", text, ticker, form, year, period, item,
                 "2024-12-31", "https://www.sec.gov/x")


CHUNKS = [chunk(0, "Total net sales were 391,035 million, driven by iPhone."),
          chunk(1, "Research and development expense increased.", item="Item 8"),
          chunk(2, "Net interest income rose on higher rates.", ticker="JPM"),
          chunk(3, "Total net sales grew again.", year=2025),
          chunk(4, "Quarterly net sales of the second quarter.", year=2025, period="Q2", form="10-Q",
                item="Part I Item 2")]


# --- filters ---------------------------------------------------------------------------------------------------

def test_filters_are_normalised():
    f = Filters.of(ticker=" aapl ", fiscal_period="q2", form="10-q", item="part i  item 2")
    assert f.active() == {"ticker": "AAPL", "fiscal_period": "Q2", "form": "10-Q", "item": "Part I Item 2"}
    assert Filters.of(item="item 1a").item == "Item 1A"


@pytest.mark.parametrize("kwargs, error", [({"fiscal_year": "2024"}, "must be an int"),
                                           ({"fiscal_year": True}, "must be an int"),
                                           ({"fiscal_period": "Q4"}, "fiscal_period"),
                                           ({"form": "8-K"}, "form")])
def test_bad_filters_are_errors_not_empty_results(kwargs, error):
    with pytest.raises(ValueError, match=error):
        Filters.of(**kwargs)


# --- BM25 ------------------------------------------------------------------------------------------------------

def test_tokenizer_keeps_numbers_whatever_their_format():
    assert tokenize("Net sales were $391,035 million (up 2.0%) in FY2024") == \
        ["net", "sales", "were", "391035", "million", "up", "2.0", "in", "fy", "2024"]


def test_bm25_ranks_the_matching_chunk_first_and_uses_the_header():
    index = BM25Index(CHUNKS)
    assert index.search("391,035 total net sales", k=2)[0].chunk_id == CHUNKS[0].chunk_id
    assert index.search("JPM interest", k=1)[0].ticker == "JPM"             # "JPM" is only in the header


@pytest.mark.parametrize("filters, expected", [
    (Filters.of(ticker="AAPL", fiscal_year=2025), {3, 4}),
    (Filters.of(ticker="AAPL", fiscal_year=2025, fiscal_period="FY"), {3}),
    (Filters.of(form="10-Q"), {4}),
    (Filters.of(item="Item 8"), set()),                    # no "sales" chunk in Item 8
    (Filters.of(ticker="MSFT"), set()),
])
def test_bm25_filters(filters, expected):
    hits = BM25Index(CHUNKS).search("net sales", k=10, filters=filters)
    assert {CHUNKS.index(next(c for c in CHUNKS if c.chunk_id == h.chunk_id)) for h in hits} == expected


def test_bm25_save_and_load(tmp_path):
    path = tmp_path / "bm25_C2.pkl"
    BM25Index(CHUNKS).save(path)
    assert BM25Index.load(path).search("iPhone", k=1)[0].chunk_id == CHUNKS[0].chunk_id
    import pickle
    path.write_bytes(pickle.dumps({"format": INDEX_FORMAT - 1, "index": None}))
    with pytest.raises(ValueError, match="older version"):
        BM25Index.load(path)


def test_bm25_input_checks():
    index = BM25Index(CHUNKS)
    assert index.search("!!!", k=3) == []
    with pytest.raises(ValueError, match="k must be positive"):
        index.search("sales", k=0)
    with pytest.raises(ValueError, match="without chunks"):
        BM25Index([])


# --- dense -----------------------------------------------------------------------------------------------------

def test_where_clause():
    assert where_clause(Filters()) is None
    assert where_clause(Filters.of(ticker="AAPL")) == {"ticker": "AAPL"}
    assert where_clause(Filters.of(ticker="AAPL", fiscal_year=2024)) == \
        {"$and": [{"ticker": "AAPL"}, {"fiscal_year": 2024}]}


def fake_embed(texts):
    """2-d vectors: [mentions sales, mentions interest]."""
    return [[float("sales" in t.lower()), float("interest" in t.lower())] for t in texts]


EMBEDDER = Embedder(fake_embed, "fake:model")


class FakeCollection:
    def __init__(self, name="chunks_C2", metadata=None):
        self.name, self.metadata, self.rows = name, metadata or {}, {}

    def add(self, ids, embeddings, metadatas):
        self.rows.update(zip(ids, zip(embeddings, metadatas, strict=True), strict=True))

    def count(self):
        return len(self.rows)

    def query(self, query_embeddings, n_results, where, include):
        q = query_embeddings[0]
        ok = [(cid, v) for cid, (v, meta) in self.rows.items() if _matches(meta, where)]
        scored = sorted(ok, key=lambda x: -sum(a * b for a, b in zip(q, x[1], strict=True)))[:n_results]
        return {"ids": [[c for c, _ in scored]],
                "distances": [[1.0 - sum(a * b for a, b in zip(q, v, strict=True)) for _, v in scored]]}


def _matches(meta, where):
    if where is None:
        return True
    conditions = where.get("$and", [where])
    return all(meta[k] == v for cond in conditions for k, v in cond.items())


class FakeClient:
    def __init__(self):
        self.collections, self.deleted = {}, []

    def list_collections(self):
        return list(self.collections)

    def delete_collection(self, name):
        self.deleted.append(name)
        del self.collections[name]

    def create_collection(self, name, metadata):
        self.collections[name] = FakeCollection(name, metadata)
        return self.collections[name]


def test_build_replaces_the_collection_and_records_the_model():
    client = FakeClient()
    build_collection(client, CHUNKS, EMBEDDER)
    col = build_collection(client, CHUNKS, EMBEDDER)              # rebuild: no duplicate vectors
    assert client.deleted == ["chunks_C2"] and col.count() == len(CHUNKS)
    assert col.metadata == {"hnsw:space": "cosine", "embed_model": "fake:model"}


def test_build_checks_the_embedder_output():
    with pytest.raises(RuntimeError, match="vectors for"):
        build_collection(FakeClient(), CHUNKS, Embedder(lambda texts: [[1.0, 0.0]], "fake:model"))


def test_dense_search_filters_and_scores_by_cosine_similarity():
    col = build_collection(FakeClient(), CHUNKS, EMBEDDER)
    dense = DenseIndex(col, CHUNKS, EMBEDDER)
    hits = dense.search("interest", k=1)
    assert (hits[0].ticker, hits[0].score) == ("JPM", 1.0)
    assert {h.fiscal_year for h in dense.search("sales", k=5, filters=Filters.of(fiscal_year=2025))} == {2025}


def test_dense_refuses_a_different_model_or_a_stale_collection():
    col = build_collection(FakeClient(), CHUNKS, EMBEDDER)
    with pytest.raises(ValueError, match="embedded with 'fake:model'"):
        DenseIndex(col, CHUNKS, Embedder(fake_embed, "fake:other"))
    with pytest.raises(ValueError, match="run ingest/04_index.py again"):
        DenseIndex(col, CHUNKS[:-1], EMBEDDER)


def test_real_chroma_roundtrip(tmp_path):
    pytest.importorskip("chromadb")
    from retrieval.dense import open_client
    client = open_client(tmp_path)
    build_collection(client, CHUNKS, EMBEDDER)
    col = build_collection(client, CHUNKS, EMBEDDER)
    assert col.count() == len(CHUNKS)
    hits = DenseIndex(col, CHUNKS, EMBEDDER).search("net sales", k=2, filters=Filters.of(ticker="AAPL",
                                                                                           fiscal_year=2025))
    assert {h.chunk_id for h in hits} == {CHUNKS[3].chunk_id, CHUNKS[4].chunk_id}


# --- hybrid ----------------------------------------------------------------------------------------------------

def hit(cid, score=0.0):
    return SearchResult(cid, "t", "AAPL", "10-K", 2024, "FY", "Item 7", "u", score)


def test_rrf_scores_and_deterministic_ties():
    fused = rrf([[hit("a"), hit("b")], [hit("b"), hit("c")]], k=3)
    assert [h.chunk_id for h in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert [h.chunk_id for h in rrf([[hit("y")], [hit("x")]], k=2)] == ["y", "x"]   # tie: earlier list first


class Recorder:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, query, k=5, filters=NO_FILTERS):
        self.calls.append((query, k, filters))
        return self.hits[:k]


def test_hybrid_asks_both_retrievers_for_candidates_with_the_same_filters():
    bm25, dense = Recorder([hit("a"), hit("b")]), Recorder([hit("b")])
    f = Filters.of(ticker="AAPL")
    assert [h.chunk_id for h in HybridIndex(bm25, dense).search("q", k=1, filters=f)] == ["b"]
    assert bm25.calls == dense.calls == [("q", 50, f)]
