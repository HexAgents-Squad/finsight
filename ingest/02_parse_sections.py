"""Step 2 of the corpus: split every downloaded filing into Item-level section documents.

Usage:
    python ingest/02_parse_sections.py                         # all filings in data/raw/manifest.jsonl
    python ingest/02_parse_sections.py --tickers AAPL --out-dir /tmp/sec   # a subset, for debugging
    python ingest/02_parse_sections.py --sample AAPL JPM       # also write the committed WP5 sample

Input : data/raw/manifest.jsonl + the downloaded .htm files (ingest/01_download.py)
Output (in --out-dir, default data/sections; each file is written whole or not at all):
        data/sections/sections.jsonl     1 line = 1 document = 1 Item of 1 filing (week0_plan §2.1)
        data/sections/parse_report.csv   per filing: Items found, kept, relocated, unassigned characters
        data/sections/corpus_stats.csv   per company and form: documents and characters
        data/sample/sections_sample.jsonl  (--sample, always this path; full runs only) the sample tickers'
                                           sections from the latest 10-K and the latest 10-Q, small enough to commit
A --tickers run skips the whole-corpus checks and must write to its own --out-dir, so it can never replace
the full corpus with a subset.
Exit codes: 0 = corpus checks pass, 1 = a check failed (printed), 2 = setup error (no manifest, unknown ticker,
a subset run into data/sections, a downloaded file missing).
The splitting rules are in ingest/sections.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest import DATA_DIR, MANIFEST_FILE, SAMPLE_DIR, SECTIONS_DIR  # noqa: E402
from ingest.sections import FS_KEY, MDNA_KEY, corpus_problems, html_to_text, parse_filing, section_records  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    tmp.replace(path)                         # an interrupted run never leaves a half-written corpus


def write_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def sample_docs(docs: list[dict], tickers: set[str]) -> list[dict]:
    """Sections of each sample ticker's latest 10-K and latest 10-Q."""
    out = []
    for ticker in sorted(tickers):
        mine = [d for d in docs if d["ticker"] == ticker]
        for form in ("10-K", "10-Q"):
            latest = max(((d["fiscal_year"], d["fiscal_period"]) for d in mine if d["form"] == form), default=None)
            out += [d for d in mine if d["form"] == form and (d["fiscal_year"], d["fiscal_period"]) == latest]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", help="only these tickers (skips the whole-corpus checks; needs "
                                                  "--out-dir)")
    ap.add_argument("--sample", nargs="+", metavar="TICKER", help="write data/sample/sections_sample.jsonl")
    ap.add_argument("--out-dir", type=Path, default=SECTIONS_DIR, help="where the three output files go")
    args = ap.parse_args(argv)

    if not MANIFEST_FILE.exists():
        print("ERROR: data/raw/manifest.jsonl not found; run ingest/01_download.py first", file=sys.stderr)
        return 2
    manifest = [json.loads(line) for line in MANIFEST_FILE.read_text(encoding="utf-8").splitlines() if line]
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers}
        if unknown := wanted - {m["ticker"] for m in manifest}:
            print(f"ERROR: not in the manifest: {sorted(unknown)}", file=sys.stderr)
            return 2
        if args.out_dir.resolve() == SECTIONS_DIR.resolve():
            print("ERROR: a --tickers run would replace the full corpus in data/sections; give --out-dir",
                  file=sys.stderr)
            return 2
        if args.sample:                           # the committed sample must come from the full corpus
            print("ERROR: --sample needs a full run (no --tickers)", file=sys.stderr)
            return 2
        manifest = [m for m in manifest if m["ticker"] in wanted]

    docs, report = [], []
    for m in manifest:
        try:
            main_text = html_to_text((DATA_DIR / m["path"]).read_bytes())
            exhibit_text = (html_to_text((DATA_DIR / m["exhibit_path"]).read_bytes()) if m.get("exhibit_path")
                            else None)
        except OSError as err:
            print(f"ERROR: downloaded filing missing or unreadable ({err}); run ingest/01_download.py",
                  file=sys.stderr)
            return 2
        parsed = parse_filing(main_text, m["form"], exhibit_text)
        records = section_records(m, parsed)
        docs += records
        size = {r["item"]: r["n_chars"] for r in records}
        report.append({
            "ticker": m["ticker"], "form": m["form"], "fiscal_year": m["fiscal_year"],
            "fiscal_period": m["fiscal_period"], "item_headings_found": parsed.headings_found,
            "sections_kept": len(records), "mdna_chars": size.get(MDNA_KEY[m["form"]], 0),
            "financial_statements_chars": size.get(FS_KEY[m["form"]], 0),
            "relocated": " ".join(parsed.relocated), "from_exhibit": sum(r["source_location"] != "main document"
                                                                          for r in records),
            "unassigned_chars": parsed.unassigned_chars, "total_chars": len(main_text) + len(exhibit_text or ""),
        })
        print(f"  {m['ticker']:5} {m['form']} FY{m['fiscal_year']} {m['fiscal_period']:2}  "
              f"headings={parsed.headings_found:2d} kept={len(records):2d}"
              f"{'  relocated: ' + ', '.join(parsed.relocated) if parsed.relocated else ''}")

    if not docs:                                   # nothing parsed: keep whatever corpus is there
        print("ERROR: no section documents were parsed; nothing written", file=sys.stderr)
        return 1
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "sections.jsonl", docs)
    write_csv(out / "parse_report.csv", report)
    stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for d in docs:
        stats[(d["ticker"], d["form"])].update(documents=1, characters=d["n_chars"])
    write_csv(out / "corpus_stats.csv",
              [{"ticker": t, "form": f, **c} for (t, f), c in sorted(stats.items())])
    print(f"\n{len(docs)} section documents from {len(manifest)} filings -> {out / 'sections.jsonl'}")

    if args.sample:
        sample = sample_docs(docs, {t.upper() for t in args.sample})
        if not sample:
            print(f"ERROR: no documents for sample tickers {args.sample}; sample not written", file=sys.stderr)
            return 1
        write_jsonl(SAMPLE_DIR / "sections_sample.jsonl", sample)
        print(f"sample: {len(sample)} documents -> data/sample/sections_sample.jsonl")

    if args.tickers:
        print("(subset run: whole-corpus checks skipped)")
        return 0
    expected_10k = sum(1 for m in manifest if m["form"] == "10-K")
    problems = corpus_problems(docs, expected_10k)
    for p in problems:
        print(f"CHECK FAILED: {p}")
    if not problems:
        print("OK: all corpus checks pass (document count, unique ids, Item 7 and Item 8 in every 10-K)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
