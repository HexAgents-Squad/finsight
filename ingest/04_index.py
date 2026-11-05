"""WP5 steps 2-3: build the BM25 and dense (Chroma) indexes for each chunk variant.

Usage:
    python ingest/04_index.py                          # BM25 + dense for C1, C2 and C3
    python ingest/04_index.py --variants C3 --no-dense # BM25 only (seconds)

The embedder comes from .env through core.llm.embed() (EMBED_BACKEND, EMBED_MODEL / EMBED_LOCAL_MODEL_NAME);
an environment variable overrides .env for one run, e.g. EMBED_BACKEND=sentence-transformers. The collection
records the model, and searching with a different one is an error (retrieval/dense.py).

Input : data/chunks/chunks_<variant>.jsonl (ingest/03_chunk.py)
Output: data/index/bm25_<variant>.pkl, data/index/chroma/ (collection chunks_<variant>)
Re-running rebuilds from scratch: the BM25 file is replaced and the Chroma collection is deleted and
recreated, so vectors are never duplicated (checked: vectors = chunks after every build).
Exit codes: 0 = built, 1 = a build check failed, 2 = input or settings missing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError  # noqa: E402

from retrieval import CHROMA_DIR, CHUNKS_DIR, INDEX_DIR  # noqa: E402
from retrieval.bm25 import BM25Index, index_path  # noqa: E402
from retrieval.chunking import VARIANTS  # noqa: E402
from retrieval.common import chunks_path, read_chunks  # noqa: E402
from retrieval.dense import build_collection, embedder_from_settings, open_client  # noqa: E402

PROGRESS_EVERY = 10                   # print dense progress every N embed slices


def progress(done: int, total: int, seconds: float) -> None:
    progress.calls = getattr(progress, "calls", 0) + 1
    if progress.calls % PROGRESS_EVERY == 0 or done == total:
        rate = done / seconds if seconds else 0.0
        left = (total - done) / rate / 60 if rate else 0.0
        print(f"    {done:,}/{total:,} chunks, {rate:.1f}/s, ~{left:.0f} min left", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--no-dense", action="store_true", help="build only the BM25 indexes")
    ap.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    ap.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    args = ap.parse_args(argv)

    try:
        chunk_sets = {v: read_chunks(chunks_path(v, args.chunks_dir)) for v in args.variants}
    except FileNotFoundError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    for variant, chunks in chunk_sets.items():
        start = time.perf_counter()
        BM25Index(chunks).save(index_path(variant, args.index_dir))
        print(f"{variant}: BM25 over {len(chunks):,} chunks in {time.perf_counter() - start:.0f} s")
    if args.no_dense:
        return 0

    try:
        embedder = embedder_from_settings()
    except ValidationError as err:
        print(f"ERROR: settings are incomplete; copy .env.example to .env and fill it in.\n{err}", file=sys.stderr)
        return 2
    client = open_client(args.index_dir / CHROMA_DIR.name)
    try:
        for variant, chunks in chunk_sets.items():
            print(f"{variant}: embedding {len(chunks):,} chunks with {embedder.model_id}", flush=True)
            start = time.perf_counter()
            progress.calls = 0
            build_collection(client, chunks, embedder, progress)
            seconds = time.perf_counter() - start
            print(f"{variant}: dense index built in {seconds / 60:.1f} min ({len(chunks) / seconds:.1f} chunks/s)")
    except RuntimeError as err:
        print(f"CHECK FAILED: {err}", file=sys.stderr)
        return 1
    finally:
        embedder.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
