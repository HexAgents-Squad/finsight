"""W1-04: compare a re-parsed corpus with the current one before promoting it (offline, independent of the rules).

Run:  python ingest/02_parse_sections.py --out-dir data/cache/sections_new      # re-parse beside the corpus
      python scripts/compare_sections.py data/sections data/cache/sections_new  # -> eval/results/furniture_removal.md

The checks look at what actually changed, not at what the furniture rules say should change, and share no
pattern with ingest/sections.py (the checker's idea of furniture is its own, plainer one), so a wrong rule shows
up here (each is a reason to stop, exit code 1):
  1. the same documents: no doc_id lost or added (every Item of every filing is still there);
  2. per document, old and new lines are aligned (difflib): nothing is inserted; every removed line looks like
     page furniture on its own (a short line starting "Table of Contents"; at most three digits with only spaces,
     dots and dashes; the company's or a report's name with exactly one small number besides form, quarter and
     year; the company's name with "subsidiar"; no repetition required); and where lines changed, the new text
     is the old text without those lines, character for character (a sentence joined across a page break adds
     or loses no character);
  3. every removed page number fits the filing's page sequence (within 4 of the page number removed before or
     after it, across Items, so an Item with a single page number is checked too), and no removed line has a
     kept table row directly above or below it (it was a row or a row's value, not furniture);
  4. no sentence is joined onto a line that repeats in the filing (a running title such as "Notes to
     Consolidated Financial Statements" is no sentence);
  5. the gold verbatim-figure check (ADR-012 check 2): every dollar figure of a gold or development answer that
     appears verbatim in its gold source sections in the old corpus still does in the new one.
The report counts the removed lines by kind, lists every running footer and header shape, and shows a fixed
random sample of removed lines in context, for review. Exit codes: 0 = all checks pass, 1 = a check failed,
2 = input missing.
"""
from __future__ import annotations

import argparse
import difflib
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
from ingest.edgar import company_names  # noqa: E402

REPORT = ROOT / "eval" / "results" / "furniture_removal.md"
SAMPLE_SIZE = 40
SAMPLE_SEED = 4                          # W1-04: the same sample on every run
CONTEXT_CHARS = 70
PAGE_STEP = 4                            # a page number is within this of its neighbours (blank pages, inserts)
REPEATED_LINE = 3                        # a line this frequent in a filing is a running title, not a sentence
SHORT_LINE = 80                          # page furniture is a short line
LEADING_NUMBER = re.compile(r"\d+")
DIGIT = re.compile(r"\d")
# The checker's own, deliberately plain idea of furniture; it shares no pattern with ingest/sections.py, so a
# wrong removal rule is not confirmed by itself
PAGE_NUMBER_CHARS = set(" .-‐‑‒–—−")
NOT_A_PAGE = re.compile(r"\b10-[kq]\b|\bq[1-4]\b|\b(?:19|20)\d\d\b")    # form, quarter and year tokens
REPORT_WORDS = ("10-k", "10-q", "annual report", "quarterly report")


def load(directory: Path) -> dict[str, dict]:
    path = directory / "sections.jsonl"
    return {d["doc_id"]: d for d in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                                     if line.strip())}


def filing_key(doc: dict) -> tuple:
    return doc["ticker"], doc["form"], doc["fiscal_year"], doc["fiscal_period"]


class Looks:
    """Whether a line looks like page furniture on its own, for one company (no repetition, no context), by
    plain tests of its own: a short line that starts with "table of contents"; at most three digits and nothing
    but spaces, dots and dashes; the company's name or a report name with exactly one number of at most three
    digits once form, quarter and year tokens are set aside; the company's name with "subsidiar" and no digit."""

    def __init__(self, names: tuple[str, ...]):
        self.names = tuple(n.lower() for n in names)

    def kind(self, line: str) -> str | None:
        s = line.strip()
        low = s.lower()
        if not s or len(s) > SHORT_LINE:
            return None
        if low.startswith("table of contents"):
            return "table of contents"
        digits = sum(c.isdigit() for c in s)
        if 1 <= digits <= 3 and all(c.isdigit() or c in PAGE_NUMBER_CHARS for c in s):
            return "page number"
        named = any(n in low for n in self.names)
        numbers = LEADING_NUMBER.findall(NOT_A_PAGE.sub(" ", low))
        if (named or any(w in low for w in REPORT_WORDS)) and len(numbers) == 1 and len(numbers[0]) <= 3:
            return "running footer"
        if named and "subsidiar" in low and not digits:
            return "running header"
        return None


def squeeze(lines: list[str]) -> str:
    return "".join("".join(lines).split())


def page_value(line: str) -> int:
    return int(LEADING_NUMBER.search(line).group())


def doc_changes(old_text: str, new_text: str, looks: Looks) -> dict:
    """What changed in one document: removed lines (index, text, kind or None), joins (left line, right line),
    inserted text, and blocks whose characters differ beyond the removed lines."""
    raw = old_text.split("\n")                          # blank lines kept: "directly under" means no blank between
    at = [i for i, ln in enumerate(raw) if ln.strip()]  # raw index of each non-blank old line
    old = [raw[i] for i in at]
    new = [ln for ln in new_text.split("\n") if ln.strip()]
    out: dict = {"removed": [], "joins": [], "inserted": [], "altered": [], "lines": raw}
    matcher = difflib.SequenceMatcher(None, [ln.strip() for ln in old], [ln.strip() for ln in new], autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        if op == "insert":
            out["inserted"] += new[j1:j2]
            continue
        removed = [(at[i], old[i], looks.kind(old[i])) for i in range(i1, i2)]
        kept = [text for _, text, k in removed if k is None]
        if op == "delete":
            out["removed"] += removed
            continue
        # replace: furniture-looking lines removed, the rest must be the new lines (joined across the break)
        furniture = [r for r in removed if r[2] is not None]
        out["removed"] += furniture
        if squeeze(kept) != squeeze(new[j1:j2]):
            out["altered"].append((old[i1:i2], new[j1:j2]))
            continue
        out["joins"] += joined_pairs(kept, new[j1:j2])
    return out


def joined_pairs(pieces: list[str], new_lines: list[str]) -> list[tuple[str, str]]:
    """(left, right) old lines that became one new line, given that the pieces squeeze to the new lines."""
    out, k = [], 0
    for line in new_lines:
        target, taken = len(squeeze([line])), []
        while k < len(pieces) and len(squeeze(taken)) < target:
            taken.append(pieces[k])
            k += 1
        out += list(zip(taken, taken[1:], strict=False))
    return out


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


def compare(old: dict[str, dict], new: dict[str, dict], names: dict[str, tuple[str, ...]],
            items: list[dict]) -> tuple[list[str], list[str]]:
    """(problems, report lines)."""
    problems: list[str] = []
    lost, added = sorted(old.keys() - new.keys()), sorted(new.keys() - old.keys())
    if lost:
        problems.append(f"{len(lost)} documents lost: {lost[:10]}")
    if added:
        problems.append(f"{len(added)} documents added: {added[:10]}")
    line_counts: dict[tuple, Counter] = defaultdict(Counter)
    for d in old.values():
        line_counts[filing_key(d)].update(ln.strip() for ln in d["text"].split("\n") if ln.strip())
    by_kind: Counter = Counter()
    shapes: Counter = Counter()
    sample_pool: list[tuple[str, str, str, str]] = []
    issues: dict[str, list[str]] = defaultdict(list)
    filing_pages: dict[tuple, list[tuple[str, int]]] = defaultdict(list)    # in document order (file order)
    for doc_id in [k for k in old if k in new]:
        d = old[doc_id]
        looks = Looks(names.get(d["ticker"], ()))
        ch = doc_changes(d["text"], new[doc_id]["text"], looks)
        lines = ch["lines"]
        if ch["inserted"]:
            issues["text inserted"].append(f"{doc_id}: {ch['inserted'][0][:60]!r}")
        for old_block, _ in ch["altered"]:
            issues["text changed beyond the removed lines"].append(f"{doc_id}: {' / '.join(old_block)[:80]!r}")
        gone = {i for i, _, _ in ch["removed"]}
        for i, text, k in sorted(ch["removed"]):
            if k == "page number":
                filing_pages[filing_key(d)].append((doc_id, page_value(text)))
            # a table row right above or below (no blank line between) that stays: the line was a row or a value
            rows = [j for j in (i - 1, i + 1) if 0 <= j < len(lines) and j not in gone and "|" in lines[j]]
            if rows:
                issues["removed line next to a table row"].append(f"{doc_id}: {lines[rows[0]][-50:]!r} / "
                                                                  f"{text.strip()[:30]!r}")
        for i, text, k in ch["removed"]:
            if k is None:
                issues["removed line that does not look like furniture"].append(f"{doc_id}: {text.strip()[:70]!r}")
                continue
            by_kind[k] += 1
            if k.startswith("running"):
                shapes[(d["ticker"], DIGIT.sub("#", text.strip()))] += 1
            before = " ".join(lines[max(0, i - 2):i])[-CONTEXT_CHARS:]
            after = " ".join(lines[i + 1:i + 3])[:CONTEXT_CHARS]
            sample_pool.append((doc_id, before, text.strip(), after))
        for left, right in ch["joins"]:
            if line_counts[filing_key(d)][left.strip()] >= REPEATED_LINE:
                issues["sentence joined onto a repeated line"].append(f"{doc_id}: {left.strip()[:50]!r} + "
                                                                      f"{right.strip()[:30]!r}")
    # every removed page number fits the filing's sequence: within PAGE_STEP of the page number removed before or
    # after it, across Items, so a document with a single page number is checked too
    for pages in filing_pages.values():
        for n, (doc_id, value) in enumerate(pages):
            neighbours = [v for _, v in pages[max(0, n - 1):n] + pages[n + 1:n + 2]]
            if neighbours and min(abs(value - v) for v in neighbours) > PAGE_STEP:
                issues["page number out of the page sequence"].append(f"{doc_id}: {value} (next to {neighbours})")
    for name, found in issues.items():
        problems.append(f"{len(found)} {name}: {found[:5]}")
    before, after = gold_figures(items, old), gold_figures(items, new)
    regressed = sorted(k for k, found in before.items() if found and not after.get(k))
    if regressed:
        problems.append(f"gold figures no longer found verbatim in their sources: {regressed}")

    old_chars, new_chars = sum(d["n_chars"] for d in old.values()), sum(d["n_chars"] for d in new.values())
    out = ["# Page furniture removed from the section corpus (W1-04)", "",
           "Generated by `scripts/compare_sections.py` (old corpus vs the re-parse; the checks align old and new "
           "lines and share no pattern with the removal rules); decision in ADR-023.", "",
           f"- Documents: {len(old):,} before, {len(new):,} after; lost {len(lost)}, added {len(added)}.",
           f"- Characters: {old_chars:,} before, {new_chars:,} after ({100 * (old_chars - new_chars) / old_chars:.2f}% "
           f"removed).",
           f"- Lines removed: {sum(by_kind.values()):,}: "
           + ", ".join(f"{k} {n:,}" for k, n in by_kind.most_common()) + ".",
           "- Removed lines that do not look like furniture, text inserted or changed beyond the removed lines, "
           "page numbers out of the filing's sequence, removed lines next to a table row, sentences joined onto a "
           "repeated line: "
           f"{sum(len(v) for v in issues.values())}.",
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
    sample = random.Random(SAMPLE_SEED).sample(sample_pool, min(SAMPLE_SIZE, len(sample_pool)))
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
    problems, report = compare(old, new, company_names(), items)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(report[:12]))
    for p in problems:
        print(f"CHECK FAILED: {p}")
    print(f"report: {args.report}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
