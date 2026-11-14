"""Dense search: a persistent Chroma collection (cosine) per chunk variant, embedded with core.llm.embed().

The collection records which embedding backend and model built it; searching with a different one is an
error, because vectors from two models are not comparable. Rebuilding deletes and recreates the collection,
so re-running the indexer never duplicates vectors.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from retrieval import CHROMA_DIR
from retrieval.common import NO_FILTERS, Chunk, Filters, SearchResult

Embed = Callable[[list[str]], list[list[float]]]
EMBED_SLICE = 1_024                  # chunks per embed + add round (progress output, bounded memory)
DISTANCE = "cosine"


LEGACY_BACKENDS = ("sentence-transformers", "openai")   # recorded as "<backend>:<model>" before W1-04


def model_name(model_id: str) -> str:
    """The model an embedder id names, without the backend or the publisher: "bge-small-en-v1.5" for the hosted
    endpoint's "bge-small-en-v1.5" and for "BAAI/bge-small-en-v1.5" run locally (the same vectors, cosine 1.0,
    W1-16), so an index built through one is searched with the other (ADR-021). Ids recorded before W1-04
    carried the backend ("sentence-transformers:all-MiniLM-L6-v2") and read the same."""
    for backend in LEGACY_BACKENDS:
        model_id = model_id.removeprefix(f"{backend}:")
    return model_id.rpartition("/")[2]               # an Ollama tag ("nomic-embed-text:latest") stays whole


@dataclass
class Embedder:
    """A synchronous embed function plus the model behind it (model_name: "bge-small-en-v1.5")."""
    embed: Embed
    model_id: str
    close: Callable[[], None] = lambda: None


def embedder_from_settings() -> Embedder:
    """core.llm.embed() (WP7) configured by .env, run on one event loop that lives as long as the embedder,
    so pooled HTTP connections stay valid between calls."""
    from core.llm import LLMClient
    from core.settings import get_settings
    settings = get_settings()
    model = (settings.EMBED_LOCAL_MODEL_NAME if settings.EMBED_BACKEND == "sentence-transformers"
             else settings.EMBED_MODEL)
    client = LLMClient(settings)                 # first, so a settings error cannot leak an event loop
    loop = asyncio.new_event_loop()

    def close() -> None:
        """Safe to call more than once; the loop is closed even if closing the client fails."""
        if loop.is_closed():
            return
        try:
            loop.run_until_complete(client.aclose())
        finally:
            loop.close()

    return Embedder(lambda texts: loop.run_until_complete(client.embed(texts)), model_name(model), close)


def collection_name(variant: str) -> str:
    return f"chunks_{variant}"


def where_clause(filters: Filters) -> dict[str, Any] | None:
    """Chroma metadata filter: one condition as is, several under "$and" (Chroma rejects a one-item $and)."""
    conditions = [{k: v} for k, v in filters.active().items()]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def collection_names(client: Any) -> list[str]:
    """Chroma 0.6+ lists names; older versions list Collection objects."""
    return [getattr(c, "name", c) for c in client.list_collections()]


def open_client(path=CHROMA_DIR) -> Any:
    import chromadb  # heavy import, only where dense search is used (not needed by CI tests)
    from chromadb.config import Settings as ChromaSettings
    return chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))


def build_collection(client: Any, chunks: list[Chunk], embedder: Embedder,
                     progress: Callable[[int, int, float], None] | None = None) -> Any:
    """(Re)create the variant's collection from scratch and add every chunk with its embedding."""
    if not chunks:
        raise ValueError("cannot build a dense index without chunks")
    name = collection_name(chunks[0].variant)
    if name in collection_names(client):
        client.delete_collection(name)
    collection = client.create_collection(name, metadata={"hnsw:space": DISTANCE,
                                                            "embed_model": embedder.model_id})
    start = time.perf_counter()
    for i in range(0, len(chunks), EMBED_SLICE):
        part = chunks[i:i + EMBED_SLICE]
        vectors = embedder.embed([c.index_text for c in part])
        if len(vectors) != len(part):
            raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(part)} texts")
        collection.add(ids=[c.chunk_id for c in part], embeddings=vectors, metadatas=[c.metadata() for c in part])
        if progress:
            progress(i + len(part), len(chunks), time.perf_counter() - start)
    if collection.count() != len(chunks):
        raise RuntimeError(f"collection {name} holds {collection.count()} vectors for {len(chunks)} chunks")
    return collection


class DenseIndex:
    def __init__(self, collection: Any, chunks: list[Chunk], embedder: Embedder):
        built_with = (collection.metadata or {}).get("embed_model")
        if built_with is None or model_name(built_with) != model_name(embedder.model_id):
            raise ValueError(f"collection {collection.name} was embedded with {built_with!r} but the current "
                             f"embedder is {embedder.model_id!r}; set the same EMBED_* settings or rebuild "
                             f"the index")
        self.by_id = {c.chunk_id: c for c in chunks}
        # Why IDs, not just the count: chunks re-made with other IDs but the same count would otherwise pass here
        # and fail with a KeyError mid-search. Listing the IDs takes ~1 s for 72k vectors, once per load.
        stored = set(collection.get(include=[])["ids"])
        if stored != self.by_id.keys():
            raise ValueError(f"collection {collection.name} holds {len(stored)} vectors whose IDs do not match "
                             f"the {len(chunks)} chunks ({len(stored - self.by_id.keys())} unknown, "
                             f"{len(self.by_id.keys() - stored)} missing); run ingest/04_index.py again")
        self.collection, self.embedder = collection, embedder

    def search(self, query: str, k: int = 5, filters: Filters = NO_FILTERS) -> list[SearchResult]:
        """The k chunks nearest to `query` among those matching `filters`; score = cosine similarity."""
        if k <= 0:
            raise ValueError("k must be positive")
        if not query.strip():
            return []
        vector = self.embedder.embed([query])[0]
        found = self.collection.query(query_embeddings=[vector], n_results=k, where=where_clause(filters),
                                      include=["distances"])
        return [SearchResult.of(self.by_id[cid], 1.0 - dist)
                for cid, dist in zip(found["ids"][0], found["distances"][0], strict=True)]
