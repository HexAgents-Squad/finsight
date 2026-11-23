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


def file_version(path: Path | str) -> tuple[str, int, int]:
    """(resolved path, modification time in ns, size): the key under which a parsed config file is cached, so an
    edited file (or a test's own copy) is read again and an unchanged one is parsed once (W3-07: the planner read
    the two YAML files five times a question, ~0.15 s)."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return str(resolved), stat.st_mtime_ns, stat.st_size
