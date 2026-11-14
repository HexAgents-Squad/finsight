"""retrieval/: filters, BM25, dense (fake and real Chroma) and RRF hybrid search (offline)."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from retrieval.bm25 import INDEX_FORMAT, BM25Index, tokenize
from retrieval.common import NO_FILTERS, Chunk, Filters, SearchResult
from retrieval.dense import DenseIndex, Embedder, build_collection, where_clause
from retrieval.hybrid import CANDIDATES, RRF_K, HybridIndex, rrf


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


def test_direct_construction_is_normalised_too():
    assert Filters(ticker="aapl", item="item 8") == Filters.of(ticker="AAPL", item="Item 8")
    assert Filters(ticker="aapl").active() == {"ticker": "AAPL"}


@pytest.mark.parametrize("kwargs, error", [({"fiscal_year": "2024"}, "must be an int"),
                                           ({"fiscal_year": True}, "must be an int"),
                                           ({"fiscal_year": 24}, "must be an int from 1993"),
                                           ({"fiscal_period": "Q4"}, "invalid fiscal_period"),
                                           ({"form": "8-K"}, "invalid form"),
                                           ({"ticker": ""}, "invalid ticker"),
                                           ({"ticker": "  "}, "invalid ticker"),
                                           ({"ticker": "AA PL"}, "invalid ticker"),
                                           ({"ticker": 123}, "ticker must be a string"),
                                           ({"item": " "}, "invalid item"),
                                           ({"item": "MD&A"}, "invalid item"),
                                           ({"form": ["10-K"]}, "form must be a string")])
def test_bad_filters_are_errors_not_empty_or_unfiltered_results(kwargs, error):
    with pytest.raises(ValueError, match=error):
        Filters.of(**kwargs)
    with pytest.raises(ValueError, match=error):
        Filters(**kwargs)


# --- BM25 ------------------------------------------------------------------------------------------------------

def test_tokenizer_keeps_numbers_whatever_their_format():
    assert tokenize("Net sales were $391,035 million (up 2.0%) in FY2024") == \
        ["net", "sales", "were", "391035", "million", "up", "2.0", "in", "fy", "2024"]


@pytest.mark.parametrize("text, tokens", [("FY2023,2024", ["fy", "2023", "2024"]),   # a list, not one number
                                          ("1,234.5", ["1234.5"]),
                                          ("12,3456", ["12", "3456"]),
                                          ("1,234,567 and 2023,2024,2025", ["1234567", "and", "2023", "2024",
                                                                            "2025"])])
def test_tokenizer_joins_only_real_thousands_separators(text, tokens):
    assert tokenize(text) == tokens


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

    def get(self, include):
        return {"ids": list(self.rows)}

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
    col.metadata = {"hnsw:space": "cosine"}                          # no recorded model: refused too
    with pytest.raises(ValueError, match="embedded with None"):
        DenseIndex(col, CHUNKS, EMBEDDER)


def test_model_names_ignore_the_backend_and_the_publisher():
    from retrieval.dense import model_name
    assert model_name("bge-small-en-v1.5") == model_name("BAAI/bge-small-en-v1.5") == "bge-small-en-v1.5"
    assert model_name("sentence-transformers:all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"     # recorded before W1-04
    assert model_name("openai:bge-small-en-v1.5") == "bge-small-en-v1.5"
    # Ollama tags stay whole: two models tagged "latest" must not look alike (review)
    assert model_name("nomic-embed-text:latest") != model_name("mxbai-embed-large:latest")
    assert model_name("nomic-embed-text:latest") == "nomic-embed-text:latest"


def test_an_index_built_through_the_endpoint_is_searched_with_the_local_model():
    # ADR-021: the hosted bge-small and BAAI/bge-small-en-v1.5 run locally give the same vectors
    col = build_collection(FakeClient(), CHUNKS, Embedder(fake_embed, "bge-small-en-v1.5"))
    assert DenseIndex(col, CHUNKS, Embedder(fake_embed, "BAAI/bge-small-en-v1.5")).search("interest", k=1)
    legacy = build_collection(FakeClient(), CHUNKS, Embedder(fake_embed, "sentence-transformers:all-MiniLM-L6-v2"))
    assert DenseIndex(legacy, CHUNKS, Embedder(fake_embed, "all-MiniLM-L6-v2")).search("interest", k=1)
    with pytest.raises(ValueError, match="embedded with 'bge-small-en-v1.5'"):
        DenseIndex(col, CHUNKS, Embedder(fake_embed, "all-MiniLM-L6-v2"))


def test_dense_refuses_chunks_with_other_ids_even_at_the_same_count():
    col = build_collection(FakeClient(), CHUNKS, EMBEDDER)
    renamed = [replace(c, chunk_id=c.chunk_id + "x") if i == 0 else c for i, c in enumerate(CHUNKS)]
    with pytest.raises(ValueError, match="1 unknown, 1 missing"):
        DenseIndex(col, renamed, EMBEDDER)


def test_embedder_close_is_safe_twice_and_closes_the_client(monkeypatch):
    import core.llm
    import core.settings
    from retrieval.dense import embedder_from_settings

    class FakeLLM:
        closed = 0

        def __init__(self, settings):
            pass

        async def embed(self, texts):
            return [[1.0] for _ in texts]

        async def aclose(self):
            FakeLLM.closed += 1

    settings = SimpleNamespace(EMBED_BACKEND="sentence-transformers", EMBED_LOCAL_MODEL_NAME="m", EMBED_MODEL="x")
    monkeypatch.setattr(core.llm, "LLMClient", FakeLLM)
    monkeypatch.setattr(core.settings, "get_settings", lambda: settings)
    emb = embedder_from_settings()
    assert emb.model_id == "m" and emb.embed(["a", "b"]) == [[1.0], [1.0]]
    emb.close()
    emb.close()
    assert FakeLLM.closed == 1


def embed_with_bias(texts):
    """[sales, interest, 1]: the constant keeps every vector non-zero, so cosine is defined for all chunks."""
    return [[*v, 1.0] for v in fake_embed(texts)]


def test_real_chroma_roundtrip(tmp_path):
    pytest.importorskip("chromadb")
    from retrieval.dense import open_client
    embedder = Embedder(embed_with_bias, "fake:bias")
    client = open_client(tmp_path)
    build_collection(client, CHUNKS, embedder)
    col = build_collection(client, CHUNKS, embedder)                 # rebuild: still one vector per chunk
    assert col.count() == len(CHUNKS) and col.metadata["hnsw:space"] == "cosine"
    dense = DenseIndex(col, CHUNKS, embedder)
    hits = dense.search("interest", k=2)
    # cosine: [0,1,1]·[0,1,1] -> 1.0 (JPM); [0,1,1] vs [0,0,1] -> 1/√2 (the R&D chunk); an l2 space would give 0
    assert [h.chunk_id for h in hits] == [CHUNKS[2].chunk_id, CHUNKS[1].chunk_id]
    assert [h.score for h in hits] == pytest.approx([1.0, 2 ** -0.5], abs=1e-4)
    filtered = dense.search("net sales", k=5, filters=Filters.of(ticker="AAPL", fiscal_year=2025))
    assert {h.chunk_id for h in filtered} == {CHUNKS[3].chunk_id, CHUNKS[4].chunk_id}


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
    assert bm25.calls == dense.calls == [("q", CANDIDATES, f)]
    HybridIndex(bm25, dense).search("q", k=CANDIDATES + 5)
    assert bm25.calls[-1][1] == CANDIDATES + 5                  # never fewer candidates than results asked for


@pytest.mark.parametrize("candidates", [0, -1, True, 2.5])
def test_hybrid_rejects_bad_candidate_counts(candidates):
    with pytest.raises(ValueError, match="candidates must be a positive int"):
        HybridIndex(Recorder([]), Recorder([]), candidates)
