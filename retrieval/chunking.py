"""Split filing sections (data/sections, one Item of one filing each) into chunks, in three variants
(week0_plan WP5 step 1), all kept for comparison:

C1  fixed windows of 800 characters with 120 overlap (the baseline);
C2  paragraphs packed up to ~1,200 characters with ~150 characters of overlap, never crossing a section;
C3  like C2, but every table is its own chunk, kept whole with the short caption paragraph before it (the
    statement title and "(In millions)"), so a figure stays next to its row and column labels.

Every chunk is indexed with a context header such as "[AAPL 10-K FY2024 · Item 7 · MD&A]" and gets a
deterministic ID (doc_id + variant + number), so re-chunking the same sections gives the same IDs.
"""
from __future__ import annotations

import re
from collections.abc import Iterator

from retrieval.common import Chunk

VARIANTS = ("C1", "C2", "C3")
C1_SIZE, C1_OVERLAP = 800, 120
PACK_SIZE, PACK_OVERLAP = 1_200, 150          # C2 and C3 text chunks
# C3: tables up to this size stay whole (99.7% of the corpus's table blocks); larger ones are split by rows,
# each part repeating the table's caption and column headers
TABLE_MAX_CHARS = 4_000
CAPTION_MAX_CHARS = 300                       # a short paragraph right before a table is its caption
TABLE_HEADER_LINES = 2                        # column-header lines repeated in each part of a split table
TABLE_ROW_SHARE = 0.5                         # a paragraph is a table when at least half its lines are rows
MIN_TABLE_ROWS = 2

PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
CELL_SEPARATOR = " | "                        # tables are flattened to "label | value | value" (WP2)
# Short names for the context header; other Items use the section's own title
SHORT_TITLES = {"Item 1": "Business", "Item 1A": "Risk Factors", "Item 7": "MD&A", "Item 7A": "Market Risk",
                "Item 8": "Financial Statements", "Part I Item 1": "Financial Statements",
                "Part I Item 2": "MD&A", "Part I Item 3": "Market Risk", "Part II Item 1A": "Risk Factors"}
CHUNK_ID_DIGITS = 4


def header(doc: dict) -> str:
    period = "" if doc["fiscal_period"] == "FY" else f" {doc['fiscal_period']}"
    title = SHORT_TITLES.get(doc["item"]) or doc.get("item_title") or ""
    return f"[{doc['ticker']} {doc['form']} FY{doc['fiscal_year']}{period} · {doc['item']}" \
           + (f" · {title}]" if title else "]")


def chunk_document(doc: dict, variant: str) -> list[Chunk]:
    """The chunks of one section document (a record of data/sections/sections.jsonl)."""
    if variant == "C1":
        pieces = [("text", p) for p in fixed_windows(doc["text"], C1_SIZE, C1_OVERLAP)]
    elif variant == "C2":
        pieces = [("text", p) for p in pack(units(doc["text"]), PACK_SIZE, PACK_OVERLAP)]
    elif variant == "C3":
        pieces = list(table_aware(doc["text"]))
    else:
        raise ValueError(f"unknown chunking variant {variant!r}; use one of {VARIANTS}")
    head = header(doc)
    return [Chunk(f"{doc['doc_id']}_{variant}_{n:0{CHUNK_ID_DIGITS}d}", doc["doc_id"], variant, kind, head, text,
                  doc["ticker"], doc["form"], doc["fiscal_year"], doc["fiscal_period"], doc["item"],
                  doc["period_end"], doc["source_url"])
            for n, (kind, text) in enumerate((k, t) for k, t in pieces if t.strip())]


def fixed_windows(text: str, size: int, overlap: int) -> list[str]:
    """C1: windows of `size` characters, each starting `size - overlap` after the previous one."""
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def units(text: str) -> list[str]:
    """Paragraphs; a paragraph longer than PACK_SIZE is broken into lines, then sentences, then windows, so
    every unit fits in one chunk."""
    out = []
    for para in PARAGRAPH_BREAK.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= PACK_SIZE:
            out.append(para)
            continue
        for line in para.split("\n"):
            pieces = [line] if len(line) <= PACK_SIZE else SENTENCE_END.split(line)
            for piece in pieces:
                out += [piece] if len(piece) <= PACK_SIZE else fixed_windows(piece, PACK_SIZE, 0)
    return [u for u in out if u.strip()]


def pack(parts: list[str], size: int, overlap: int) -> list[str]:
    """Join consecutive units into chunks of at most `size` characters (plus the overlap). Each new chunk starts
    with the last ~`overlap` characters of the previous one, cut at a word boundary, so a sentence split across
    two chunks is still found whole in one of them."""
    chunks, current = [], ""
    for part in parts:
        if current and len(current) + 1 + len(part) > size:
            chunks.append(current)
            current = _tail(current, overlap)
        current = f"{current}\n{part}" if current else part
    if current:
        chunks.append(current)
    return chunks


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if 0 <= space < len(tail) - 1 else tail


def is_table(paragraph: str) -> bool:
    lines = [ln for ln in paragraph.split("\n") if ln.strip()]
    rows = sum(CELL_SEPARATOR in ln for ln in lines)
    return rows >= MIN_TABLE_ROWS and rows >= TABLE_ROW_SHARE * len(lines)


def table_aware(text: str) -> Iterator[tuple[str, str]]:
    """C3: text packed as in C2; each table (with its caption) as its own chunk."""
    paragraphs = [p.strip() for p in PARAGRAPH_BREAK.split(text) if p.strip()]
    pending: list[str] = []                      # text paragraphs waiting to be packed
    for para in paragraphs:
        if not is_table(para):
            pending.append(para)
            continue
        caption = ""
        if pending and len(pending[-1]) <= CAPTION_MAX_CHARS and not is_table(pending[-1]):
            caption = pending.pop()              # the statement title stays with its table
        yield from (("text", c) for c in pack(units("\n\n".join(pending)), PACK_SIZE, PACK_OVERLAP))
        pending = []
        yield from (("table", t) for t in split_table(para, caption))
    yield from (("text", c) for c in pack(units("\n\n".join(pending)), PACK_SIZE, PACK_OVERLAP))


def split_table(table: str, caption: str = "") -> list[str]:
    whole = f"{caption}\n{table}" if caption else table
    if len(whole) <= TABLE_MAX_CHARS:
        return [whole]
    lines = table.split("\n")
    head = "\n".join(([caption] if caption else []) + lines[:TABLE_HEADER_LINES])
    parts, current = [], []
    for line in lines[TABLE_HEADER_LINES:]:
        if current and len(head) + sum(len(x) + 1 for x in current) + len(line) > TABLE_MAX_CHARS:
            parts.append(head + "\n" + "\n".join(current))
            current = []
        current.append(line)
    if current:
        parts.append(head + "\n" + "\n".join(current))
    return parts


def chunk_corpus(docs: list[dict], variant: str) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, variant)]
