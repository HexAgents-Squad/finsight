"""Check that a machine has what `make up` mounts into the containers (W3-11, ADR-046): .env and the data the API
loads on start-up. It names each missing file and how to get it, before Docker builds or starts anything.

    python3 scripts/check_data.py            (`make check-data`; `make up` runs it first)

Standard library only, so it runs on a machine with Docker and no virtual environment (python3 on WSL2/Linux).
Exit code 0 when everything is there, 1 otherwise.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest import COMPANIES_FILE, MANIFEST_FILE, XBRL_DIR  # noqa: E402  (both packages' __init__ use pathlib only)
from retrieval import CHROMA_DIR, CHUNKS_DIR, INDEX_DIR  # noqa: E402

VARIANT = "C3"              # tools/search_filings.DEFAULT_VARIANT (tests/test_deploy.py checks they agree)
BUILD = "make ingest index (over an hour on a CPU), or copy it from a teammate who built it (README.md)"


def tickers(companies_file: Path = COMPANIES_FILE) -> list[str]:
    """The tickers in config/companies.yaml, read without PyYAML (not installed outside the venv)."""
    return re.findall(r"^\s*(?:-\s*)?ticker:\s*\"?([A-Z.]+)\"?\s*$", companies_file.read_text(encoding="utf-8"), re.M)


def problems(root: Path = ROOT) -> list[str]:
    """One line per missing file: what it is and the command that makes it."""
    def rel(path: Path) -> Path:
        return root / path.relative_to(ROOT)

    out = []
    if not (root / ".env").is_file():
        out.append(".env is missing: cp .env.example .env, then set the LLM_* lines (README.md, LOCAL_SETUP.md)")
    needed = [(rel(INDEX_DIR) / f"bm25_{VARIANT}.pkl", "the BM25 index", BUILD),
              (rel(CHROMA_DIR) / "chroma.sqlite3", "the dense (Chroma) index", BUILD),
              (rel(CHUNKS_DIR) / f"chunks_{VARIANT}.jsonl", f"the {VARIANT} chunks", BUILD),
              (rel(MANIFEST_FILE), "the filing manifest", BUILD)]
    out += [f"{path.relative_to(root).as_posix()} is missing ({what}): {fix}" for path, what, fix in needed
            if not path.is_file()]
    companies = tickers(rel(COMPANIES_FILE))
    facts = [t for t in companies if not (rel(XBRL_DIR) / f"{t}.json").is_file()]
    if facts:
        out.append(f"{rel(XBRL_DIR).relative_to(root).as_posix()}/ is missing the XBRL facts of {len(facts)} of "
                   f"{len(companies)} companies ({', '.join(facts)}): {BUILD}")
    return out


def main() -> int:
    found = problems()
    if found:
        print("Not ready for `make up`:", *(f"  - {line}" for line in found), sep="\n")
        return 1
    print(f"OK: .env and the data the API loads are present ({VARIANT} index, chunks, XBRL facts, manifest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
