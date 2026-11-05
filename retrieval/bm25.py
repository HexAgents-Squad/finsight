"""BM25 keyword search (rank_bm25.BM25Okapi) over the chunks of one variant, with metadata filters.

Tokens are lower-cased runs of letters and digits; numbers keep their decimal point ("107.6") and lose their
thousands separators ("391,035" -> "391035"), so a figure matches however the query writes it.
"""
from __future__ import annotations

import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from retrieval import INDEX_DIR
from retrieval.common import NO_FILTERS, Chunk, Filters, SearchResult

# Words; numbers with thousands separators only when every group has exactly three digits ("391,035"), so a
# list such as "2023,2024" stays two numbers; other numbers with an optional decimal part ("107.6").
TOKEN = re.compile(r"[a-z]+|\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?")
INDEX_FORMAT = 3                      # bump when the pickled layout or the tokenizer changes, so old files are rebuilt
FILTER_FIELDS = ("ticker", "fiscal_year", "fiscal_period", "form", "item")


def tokenize(text: str) -> list[str]:
    return [t.replace(",", "") for t in TOKEN.findall(text.lower())]


def index_path(variant: str, directory: Path = INDEX_DIR) -> Path:
    return directory / f"bm25_{variant}.pkl"


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("cannot build a BM25 index without chunks")
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.index_text) for c in chunks])
        self.variant = chunks[0].variant
        # field -> value -> positions of the chunks with that value, so a filter selects without a full scan
        self.positions: dict[str, dict] = {f: defaultdict(set) for f in FILTER_FIELDS}
        for i, c in enumerate(chunks):
            for f in FILTER_FIELDS:
                self.positions[f][getattr(c, f)].add(i)

    def matching(self, filters: Filters) -> list[int]:
        sets = [self.positions[f].get(v, set()) for f, v in filters.active().items()]
        return sorted(set.intersection(*sets)) if sets else list(range(len(self.chunks)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pkl.part")
        with tmp.open("wb") as f:
            pickle.dump({"format": INDEX_FORMAT, "index": self}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> BM25Index:
        """Our own index file (built by ingest/04_index.py); never load a pickle from elsewhere."""
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run ingest/04_index.py first")
        with path.open("rb") as f:
            data = pickle.load(f)
        if data.get("format") != INDEX_FORMAT:
            raise ValueError(f"{path} was built by an older version; run ingest/04_index.py again")
        return data["index"]

    def search(self, query: str, k: int = 5, filters: Filters = NO_FILTERS) -> list[SearchResult]:
        """The k best chunks for `query` among those matching `filters` (ties: chunk order)."""
        if k <= 0:
            raise ValueError("k must be positive")
        tokens = tokenize(query)
        if not tokens:
            return []
        if filters.active():
            ids = self.matching(filters)
            if not ids:
                return []
            scores = np.asarray(self.bm25.get_batch_scores(tokens, ids))
        else:
            ids = list(range(len(self.chunks)))
            scores = self.bm25.get_scores(tokens)
        top = np.argsort(-scores, kind="stable")[:k]
        # Why > 0: a chunk sharing no query term scores 0 and is not a result. (In a tiny corpus a term in
        # exactly half the chunks gets IDF 0 under BM25Okapi and also scores 0; irrelevant at 50k+ chunks.)
        return [SearchResult.of(self.chunks[ids[i]], float(scores[i])) for i in top if scores[i] > 0]
