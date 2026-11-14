"""W1-04: compare a re-parsed corpus with the current one before promoting it (offline).

Run:  python ingest/02_parse_sections.py --out-dir data/cache/sections_new      # re-parse beside the corpus
      python scripts/compare_sections.py data/sections data/cache/sections_new  # -> eval/results/furniture_removal.md

Checks, each a reason to stop (exit code 1):
  1. the same documents: no doc_id lost or added (every Item of every filing is still there);
  2. only page furniture removed: each document's new text equals its old text without the furniture lines
     (ingest.sections.strip_furniture, with the filing's footers and headers found in the old text), word for
     word; a sentence the page break had cut is joined, which changes no word;
  3. the gold verbatim-figure check (ADR-012 check 2): every dollar figure of a gold or development answer that
     appears verbatim in its gold source sections in the old corpus still does in the new one.
The report counts what was removed by kind, lists every footer and header shape, and shows a fixed random sample
of removed lines with their context, for review. Exit codes: 0 = all checks pass, 1 = a check failed, 2 = input
missing.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.score_answers import GROUPED_FIGURE  # noqa: E402
from eval.validate_dev_set import DEV_FILE  # noqa: E402
from eval.validate_gold import DEFAULT_PATH as GOLD_FILE  # noqa: E402
from ingest import DATA_DIR, MANIFEST_FILE  # noqa: E402
from ingest.edgar import company_names  # noqa: E402
from ingest.sections import (  # noqa: E402
    DIGIT,
    PAGE_NUMBER_LINE,
    TOC_LINK,
    Furniture,
    find_furniture,
    html_to_text,
    strip_furniture,
)

REPORT = ROOT / "eval" / "results" / "furniture_removal.md"
SAMPLE_SIZE = 40
SAMPLE_SEED = 4                          # W1-04: the same sample on every run
CONTEXT_CHARS = 70


def load(directory: Path) -> dict[str, dict]:
    path = directory / "sections.jsonl"
    return {d["doc_id"]: d for d in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                                     if line.strip())}


def filing_key(doc: dict) -> tuple:
    return doc["ticker"], doc["form"], doc["fiscal_year"], doc["fiscal_period"]


def kind(line: str) -> str:
    s = line.strip()
    if TOC_LINK.fullmatch(s):
        return "table of contents"
    if PAGE_NUMBER_LINE.fullmatch(s):
        return "page number"
    return "running footer" if DIGIT.search(s) else "running header"


def words(text: str) -> list[str]:
    return text.split()


def gold_figures(items: list[dict], docs: dict[str, dict]) -> dict[tuple[str, str], bool]:
    """(item id, figure) -> whether the figure appears verbatim in the item's gold source sections."""
    by_source = defaultdict(list)
    for d in docs.values():
        by_source[(d["ticker"], d["fiscal_year"], d["fiscal_period"], d["item"])].append(d["text"])
    out = {}
    for item in items:
        texts = [t for s in item.get("gold_sources") or []
                 for t in by_source.get((s["ticker"], s["fiscal_year"], s["fiscal_period"], s["item"]), [])]
        for figure in GROUPED_FIGURE.findall(item.get("gold_answer") or ""):
            out[(item["id"], figure)] = any(re.search(rf"(?<![\d,]){re.escape(figure)}(?![\d,])", t) for t in texts)
    return out


def filing_furniture(names: dict[str, tuple[str, ...]]) -> dict[tuple, Furniture]:
    """Each filing's running footers and headers, found as ingest/02_parse_sections.py finds them: in the whole
    downloaded filing (main document and exhibit), so a shape counts its repeats in text no section kept too."""
    out = {}
    for m in (json.loads(line) for line in MANIFEST_FILE.read_text(encoding="utf-8").splitlines() if line.strip()):
        texts = [html_to_text((DATA_DIR / m["path"]).read_bytes())]
        if m.get("exhibit_path"):
            texts.append(html_to_text((DATA_DIR / m["exhibit_path"]).read_bytes()))
        out[(m["ticker"], m["form"], m["fiscal_year"], m["fiscal_period"])] = find_furniture(texts, names[m["ticker"]])
    return out


def compare(old: dict[str, dict], new: dict[str, dict], furniture_of: dict[tuple, Furniture],
            items: list[dict]) -> tuple[list[str], list[str]]:
    """(problems, report lines). `furniture_of`: each filing's furniture (filing_furniture)."""
    problems: list[str] = []
    lost, added = sorted(old.keys() - new.keys()), sorted(new.keys() - old.keys())
    if lost:
        problems.append(f"{len(lost)} documents lost: {lost[:10]}")
    if added:
        problems.append(f"{len(added)} documents added: {added[:10]}")
    filings = defaultdict(list)
    for d in old.values():
        filings[filing_key(d)].append(d)
    removed_by_kind: Counter = Counter()
    shapes: Counter = Counter()
    removed_lines: list[tuple[str, str, str, str]] = []
    changed = []
    for key, docs in sorted(filings.items()):
        furniture = furniture_of.get(key, Furniture())
        for d in docs:
            if d["doc_id"] not in new:
                continue
            expected, _ = strip_furniture(d["text"], furniture)
            if words(expected) != words(new[d["doc_id"]]["text"]):
                changed.append(d["doc_id"])
            lines = d["text"].split("\n")
            for i, line in enumerate(lines):
                if line in furniture:
                    k = kind(line)
                    removed_by_kind[k] += 1
                    if k.startswith("running"):
                        shapes[(key[0], DIGIT.sub("#", line.strip()))] += 1
                    before = " ".join(lines[max(0, i - 2):i])[-CONTEXT_CHARS:]
                    after = " ".join(lines[i + 1:i + 3])[:CONTEXT_CHARS]
                    removed_lines.append((d["doc_id"], before, line.strip(), after))
    if changed:
        problems.append(f"{len(changed)} documents changed beyond page furniture: {changed[:10]}")
    still = sorted(d["doc_id"] for d in new.values() if any(line in Furniture() for line in d["text"].split("\n")))
    if still:
        problems.append(f"{len(still)} new documents still hold 'Table of Contents' lines or page numbers")
    before, after = gold_figures(items, old), gold_figures(items, new)
    regressed = sorted(k for k, found in before.items() if found and not after.get(k))
    if regressed:
        problems.append(f"gold figures no longer found verbatim in their sources: {regressed}")

    old_chars, new_chars = sum(d["n_chars"] for d in old.values()), sum(d["n_chars"] for d in new.values())
    out = ["# Page furniture removed from the section corpus (W1-04)", "",
           "Generated by `scripts/compare_sections.py` (old corpus vs the re-parse); decision in ADR-023.", "",
           f"- Documents: {len(old):,} before, {len(new):,} after; lost {len(lost)}, added {len(added)}.",
           f"- Characters: {old_chars:,} before, {new_chars:,} after ({100 * (old_chars - new_chars) / old_chars:.2f}% "
           f"removed).",
           f"- Lines removed: {sum(removed_by_kind.values()):,} — "
           + ", ".join(f"{k} {n:,}" for k, n in removed_by_kind.most_common()) + ".",
           f"- Documents whose text changed beyond the removed lines (word for word): {len(changed)}.",
           f"- New documents still holding 'Table of Contents' lines or page numbers: {len(still)}.",
           f"- Gold verbatim-figure check (ADR-012): {sum(before.values())} of {len(before)} gold and development "
           f"answer figures found in their source sections before, {sum(after.values())} after; "
           f"{len(regressed)} lost.", "",
           "## Running footers and headers removed, by company", "",
           "Digits shown as #. A shape is removed only where it repeats at least 3 times in a filing.", ""]
    by_ticker = defaultdict(list)
    for (ticker, shape), n in shapes.items():
        by_ticker[ticker].append((n, shape))
    for ticker in sorted(by_ticker):
        listed = "; ".join(f"`{s}` ({n:,})" for n, s in sorted(by_ticker[ticker], reverse=True))
        out.append(f"- **{ticker}**: {listed}")
    out += ["", f"## A random sample of {SAMPLE_SIZE} removed lines (seed {SAMPLE_SEED}), with context", ""]
    sample = random.Random(SAMPLE_SEED).sample(removed_lines, min(SAMPLE_SIZE, len(removed_lines)))
    for doc, before_text, line, after_text in sorted(sample):
        out.append(f"- `{doc}`: …{before_text} **[{line}]** {after_text}…".replace("|", "\\|"))
    return problems, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_dir", type=Path)
    ap.add_argument("new_dir", type=Path)
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args(argv)
    try:
        old, new = load(args.old_dir), load(args.new_dir)
        items = [json.loads(line) for path in (GOLD_FILE, DEV_FILE)
                 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    try:
        furniture_of = filing_furniture(company_names())
    except OSError as err:
        print(f"ERROR: {err} (the downloaded filings are needed: ingest/01_download.py)", file=sys.stderr)
        return 2
    problems, report = compare(old, new, furniture_of, items)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(report[:12]))
    for p in problems:
        print(f"CHECK FAILED: {p}")
    print(f"report: {args.report}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
