"""WP5 step 1: split the filing sections into chunks, in all three variants (C1 fixed, C2 section-aware,
C3 table-aware; retrieval/chunking.py).

Usage:
    python ingest/03_chunk.py                       # all variants
    python ingest/03_chunk.py --variants C2 C3
    python ingest/03_chunk.py --sections data/sample/sections_sample.jsonl --out-dir data/chunks_sample

Input : data/sections/sections.jsonl (ingest/02_parse_sections.py)
Output: data/chunks/chunks_<variant>.jsonl, one chunk per line (retrieval/common.py Chunk)
Checks (exit 1 if one fails): every section has at least one chunk; chunk IDs are unique; no chunk is
empty; text chunks stay within their size limit (tables may be larger only in C3, up to TABLE_MAX_CHARS
per part); chunking twice gives identical chunks (deterministic IDs).
Exit codes: 0 = ok, 1 = a check failed, 2 = input missing.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest import SECTIONS_DIR  # noqa: E402
from retrieval import CHUNKS_DIR  # noqa: E402
from retrieval.chunking import (  # noqa: E402
    C1_SIZE,
    PACK_OVERLAP,
    PACK_SIZE,
    TABLE_MAX_CHARS,
    VARIANTS,
    chunk_corpus,
    chunk_document,
)
from retrieval.common import Chunk, chunks_path, write_chunks  # noqa: E402

# Largest chunk a variant may produce: its window, or a packed chunk plus its overlap and a line break
MAX_TEXT_CHARS = {"C1": C1_SIZE, "C2": PACK_SIZE + PACK_OVERLAP + 1, "C3": PACK_SIZE + PACK_OVERLAP + 1}


def problems(docs: list[dict], chunks: list[Chunk], variant: str) -> list[str]:
    out = []
    covered = {c.doc_id for c in chunks}
    if missing := [d["doc_id"] for d in docs if d["doc_id"] not in covered]:
        out.append(f"{variant}: {len(missing)} sections without chunks, e.g. {missing[:3]}")
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        out.append(f"{variant}: duplicate chunk IDs")
    if empty := [c.chunk_id for c in chunks if not c.text.strip()]:
        out.append(f"{variant}: {len(empty)} empty chunks, e.g. {empty[:3]}")
    too_long = [c.chunk_id for c in chunks if len(c.text) > MAX_TEXT_CHARS[variant] and c.kind == "text"]
    too_long += [c.chunk_id for c in chunks if c.kind == "table" and len(c.text) > TABLE_MAX_CHARS
                 and "\n" in c.text.split("\n", 3)[-1]]          # a single row longer than the cap is allowed
    if too_long:
        out.append(f"{variant}: {len(too_long)} chunks over the size limit, e.g. {too_long[:3]}")
    sample = docs[: max(1, len(docs) // 50)]
    if [c for d in sample for c in chunk_document(d, variant)] != [c for c in chunks if c.doc_id in
                                                                   {d["doc_id"] for d in sample}]:
        out.append(f"{variant}: chunking is not deterministic")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--sections", type=Path, default=SECTIONS_DIR / "sections.jsonl")
    ap.add_argument("--out-dir", type=Path, default=CHUNKS_DIR)
    args = ap.parse_args(argv)

    if not args.sections.exists():
        print(f"ERROR: {args.sections} not found; run ingest/02_parse_sections.py first", file=sys.stderr)
        return 2
    docs = [json.loads(line) for line in args.sections.read_text(encoding="utf-8").splitlines() if line.strip()]
    found = []
    print(f"{len(docs)} sections from {args.sections}")
    print(f"{'variant':8}{'chunks':>9}{'tables':>8}{'mean chars':>12}{'max chars':>11}{'seconds':>9}")
    for variant in args.variants:
        start = time.perf_counter()
        chunks = chunk_corpus(docs, variant)
        seconds = time.perf_counter() - start
        write_chunks(chunks, chunks_path(variant, args.out_dir))
        sizes = [len(c.text) for c in chunks]
        print(f"{variant:8}{len(chunks):>9,}{sum(c.kind == 'table' for c in chunks):>8,}"
              f"{statistics.mean(sizes):>12,.0f}{max(sizes):>11,}{seconds:>9.1f}")
        found += problems(docs, chunks, variant)
    for p in found:
        print(f"CHECK FAILED: {p}")
    if not found:
        print(f"OK: chunks written to {args.out_dir}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
