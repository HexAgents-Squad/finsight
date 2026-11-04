"""Corpus ingestion (WP2): download SEC filings, split them into Item sections."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
XBRL_DIR = DATA_DIR / "xbrl"
SECTIONS_DIR = DATA_DIR / "sections"
SAMPLE_DIR = DATA_DIR / "sample"
COMPANIES_FILE = ROOT / "config" / "companies.yaml"
MANIFEST_FILE = RAW_DIR / "manifest.jsonl"
