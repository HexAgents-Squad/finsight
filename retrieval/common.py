"""Types shared by the chunker and the retrievers: chunks, metadata filters and search results."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from retrieval import CHUNKS_DIR

FORMS = frozenset({"10-K", "10-Q"})
FISCAL_PERIODS = frozenset({"FY", "Q1", "Q2", "Q3"})


@dataclass(frozen=True)
class Chunk:
    chunk_id: str               # "<doc_id>_<variant>_<number>", stable across runs (deterministic chunking)
    doc_id: str
    variant: str                # "C1" | "C2" | "C3"
    kind: str                   # "text" or "table" (C3 keeps tables whole)
    header: str                 # "[AAPL 10-K FY2024 · Item 7 · MD&A]", indexed with the text
    text: str                   # the filing text of the chunk, without the header
    ticker: str
    form: str
    fiscal_year: int
    fiscal_period: str
    item: str
    period_end: str
    source_url: str

    @property
    def index_text(self) -> str:
        """What BM25 and the embedder see: the header lets a chunk that never names the company still match."""
        return f"{self.header}\n{self.text}"

    def metadata(self) -> dict[str, str | int]:
        """Filterable fields, as Chroma stores them (str/int only)."""
        return {"doc_id": self.doc_id, "kind": self.kind, "ticker": self.ticker, "form": self.form,
                "fiscal_year": self.fiscal_year, "fiscal_period": self.fiscal_period, "item": self.item}


CHUNK_FIELDS = tuple(f.name for f in fields(Chunk))


def chunks_path(variant: str, directory: Path = CHUNKS_DIR) -> Path:
    return directory / f"chunks_{variant}.jsonl"


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.part")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    tmp.replace(path)                          # never leave a half-written chunk file behind


def read_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run ingest/03_chunk.py first")
    return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class Filters:
    """Metadata filters for search_filings (week0 §2.3). Unset fields do not filter."""
    ticker: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    form: str | None = None
    item: str | None = None

    @classmethod
    def of(cls, ticker: str | None = None, fiscal_year: int | None = None, fiscal_period: str | None = None,
           form: str | None = None, item: str | None = None) -> Filters:
        """Validated, normalised filters: tickers and periods are upper-cased, Item names whitespace-normalised
        ("item  7" -> "Item 7"); a wrong type or value is a ValueError, never a silent empty result."""
        if fiscal_year is not None and (not isinstance(fiscal_year, int) or isinstance(fiscal_year, bool)):
            raise ValueError(f"fiscal_year must be an int, not {fiscal_year!r}")
        ticker = ticker.strip().upper() if ticker else None
        fiscal_period = fiscal_period.strip().upper() if fiscal_period else None
        form = form.strip().upper() if form else None
        if fiscal_period is not None and fiscal_period not in FISCAL_PERIODS:
            raise ValueError(f"fiscal_period must be one of {sorted(FISCAL_PERIODS)}, not {fiscal_period!r}")
        if form is not None and form not in FORMS:
            raise ValueError(f"form must be one of {sorted(FORMS)}, not {form!r}")
        item = _item_name(item) if item else None
        return cls(ticker, fiscal_year, fiscal_period, form, item)

    def active(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def matches(self, chunk: Chunk) -> bool:
        return all(getattr(chunk, k) == v for k, v in self.active().items())


def _item_name(item: str) -> str:
    words = item.split()
    return " ".join(w.capitalize() if w.lower() in ("item", "part") else w.upper() for w in words)


NO_FILTERS = Filters()


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk, with the fields search_filings returns (week0 §2.3) plus the retriever's score."""
    chunk_id: str
    text: str
    ticker: str
    form: str
    fiscal_year: int
    fiscal_period: str
    item: str
    source_url: str
    score: float

    @classmethod
    def of(cls, chunk: Chunk, score: float) -> SearchResult:
        return cls(chunk.chunk_id, chunk.text, chunk.ticker, chunk.form, chunk.fiscal_year, chunk.fiscal_period,
                   chunk.item, chunk.source_url, score)
