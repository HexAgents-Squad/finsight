"""Types shared by the chunker and the retrievers: chunks, metadata filters and search results."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from retrieval import CHUNKS_DIR

FORMS = frozenset({"10-K", "10-Q"})
FISCAL_PERIODS = frozenset({"FY", "Q1", "Q2", "Q3"})
# EDGAR electronic filings start in 1993; the range catches two-digit years ("24") that would match nothing
MIN_FISCAL_YEAR, MAX_FISCAL_YEAR = 1993, 2100
TICKER = re.compile(r"[A-Z][A-Z0-9.\-]{0,9}")                    # e.g. "C", "GOOGL", "BRK.B"
ITEM_NAME = re.compile(r"(?:Part I{1,2} )?Item \d{1,2}[A-Z]?")      # "Item 1A", "Part II Item 1A"


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
    """Metadata filters for search_filings (week0 §2.3). Unset fields (None) do not filter.

    Every construction path is validated and normalised (`__post_init__`), because the retrievers compare
    values exactly: tickers, forms and periods are upper-cased, Item names whitespace- and case-normalised
    ("part i  item 2" -> "Part I Item 2"). A wrong type or value, or an empty string, is a ValueError, never
    a silently empty (or silently unfiltered) search.
    """
    ticker: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    form: str | None = None
    item: str | None = None

    def __post_init__(self) -> None:
        year = self.fiscal_year
        if year is not None and (not isinstance(year, int) or isinstance(year, bool)
                                 or not MIN_FISCAL_YEAR <= year <= MAX_FISCAL_YEAR):
            raise ValueError(f"fiscal_year must be an int from {MIN_FISCAL_YEAR} to {MAX_FISCAL_YEAR}, "
                             f"not {year!r}")
        rules = {"ticker": (_upper, TICKER.fullmatch), "fiscal_period": (_upper, FISCAL_PERIODS.__contains__),
                 "form": (_upper, FORMS.__contains__), "item": (_item_name, ITEM_NAME.fullmatch)}
        for name, (normalise, valid) in rules.items():
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string, not {value!r}")
            clean = normalise(value)
            if not valid(clean):
                raise ValueError(f"invalid {name} {value!r}")
            object.__setattr__(self, name, clean)     # frozen dataclass: set the normalised value once

    @classmethod
    def of(cls, ticker: str | None = None, fiscal_year: int | None = None, fiscal_period: str | None = None,
           form: str | None = None, item: str | None = None) -> Filters:
        """Same as Filters(...); kept so call sites read as `Filters.of(ticker=..., fiscal_year=...)`."""
        return cls(ticker, fiscal_year, fiscal_period, form, item)

    def active(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def matches(self, chunk: Chunk) -> bool:
        return all(getattr(chunk, k) == v for k, v in self.active().items())


def _upper(value: str) -> str:
    return value.strip().upper()


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
    # Last and defaulted so positional construction keeps working; a figure quoted from the text needs it for
    # the answer contract's values[].period_end (spec §8)
    period_end: str = ""

    @classmethod
    def of(cls, chunk: Chunk, score: float) -> SearchResult:
        return cls(chunk.chunk_id, chunk.text, chunk.ticker, chunk.form, chunk.fiscal_year, chunk.fiscal_period,
                   chunk.item, chunk.source_url, score, chunk.period_end)
