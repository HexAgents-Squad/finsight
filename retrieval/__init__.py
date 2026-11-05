"""Retrieval over the filing sections (WP5): chunking, BM25, dense (Chroma) and hybrid (RRF) search."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = ROOT / "data" / "chunks"
INDEX_DIR = ROOT / "data" / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
