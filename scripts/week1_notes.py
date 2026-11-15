"""W1-11: the Week 1 results notes, generated from committed repo data (docs/week1_plan.txt, W1-11).

Usage:
    python scripts/week1_notes.py            # rewrite the generated tables of both notes
    python scripts/week1_notes.py --check    # exit 1 if a committed note's tables are stale (tests do the same)

Writes docs/report/week1_results_note.md (one page) and docs/report/week1_detailed_results.md. Only the tables
between the GENERATED markers are rebuilt; the text around them is written by hand and kept, as in the other
reports. Every number in the tables is read from a committed file, so a fresh clone (CI) regenerates them exactly:
    eval/results/model_comparison.md     W1-08 scorer, model runs M1-M3 (W1-13/14/15)
    eval/results/prompting_baselines.md  W1-08 scorer, B0 = M1 against S1-S3 (W1-09)
    eval/results/retrieval_summary.md    eval/run_retrieval.py (W1-04, W1-05)
    eval/results/embedding_comparison.md scripts/compare_embedding_runs.py (W1-16)
    eval/results/endpoint_probe.md       scripts/probe_endpoint.py (W1-03)
    eval/gold_set.jsonl, eval/dev_set.jsonl, eval/refusal_tests.jsonl, eval/injection_tests.jsonl
    docs/decisions.md                    the ADRs dated in Week 1
A source table that is missing or changed shape is an error naming the file, never a silently empty cell.
Exit codes: 0 = written (or current with --check), 1 = stale (--check) or a source is unreadable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "eval" / "results"
REPORT_DIR = REPO / "docs" / "report"
NOTE = REPORT_DIR / "week1_results_note.md"
DETAILED = REPORT_DIR / "week1_detailed_results.md"
START, END = "<!-- GENERATED:START (scripts/week1_notes.py) -->", "<!-- GENERATED:END -->"
WEEK1_FIRST_DAY = (11, 8)                          # ADRs dated 8 Nov 2026 or later are Week 1's
MONTHS = {"Oct": 10, "Nov": 11, "Dec": 12}

# Short, model-blind-free names for the run ids (the reports use the full ids)
RUN_NAMES = {"M1_qwen3-8b_rag_think-off_b0-v1": "Qwen3-8B, passages (B0)",
             "M2_llama3.2-3b_rag_think-off_b0-v1": "Llama-3.2-3B, passages",
             "M3_qwen3-8b_closed_think-off_b0-v1": "Qwen3-8B, closed book",
             "S1_qwen3-8b_rag_think-off_s1-v1": "S1 few-shot",
             "S2_qwen3-8b_rag_think-off_s2-v1": "S2 reasoning + 5-sample vote",
             "S3_qwen3-8b_rag_think-off_s3-v1": "S3 decomposition"}
B0 = "M1_qwen3-8b_rag_think-off_b0-v1"


class SourceError(ValueError):
    """A committed source file is missing, or a table in it is not where or how this script expects it."""


@dataclass
class Table:
    heading: str
    header: list[str]
    rows: list[list[str]]

    def row(self, first: str) -> list[str]:
        for r in self.rows:
            if r[0] == first:
                return r
        raise SourceError(f"table {self.heading!r} has no row {first!r}")

    def column(self, name: str) -> int:
        if name not in self.header:
            raise SourceError(f"table {self.heading!r} has no column {name!r}")
        return self.header.index(name)


# --------------------------------------------------------------------------------------------- reading
def shown(path: Path) -> str:
    """The path as messages show it: relative to the repo when inside it."""
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)


def read(path: Path) -> str:
    if not path.exists():
        raise SourceError(f"{shown(path)} is missing")
    return path.read_text(encoding="utf-8")


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def tables(text: str) -> list[Table]:
    """Every markdown table with the heading (or sentence) right above it."""
    lines, out, heading = text.splitlines(), [], ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|") and i + 1 < len(lines) and re.fullmatch(r"\|[-:| ]+\|", lines[i + 1].strip()):
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append(cells(lines[j]))
                j += 1
            out.append(Table(heading, cells(line), rows))
            i = j
            continue
        if line.strip():
            heading = line.strip().lstrip("#").strip()
        i += 1
    return out


def table(path: Path, heading_start: str) -> Table:
    found = [t for t in tables(read(path)) if t.heading.startswith(heading_start)]
    if len(found) != 1:
        raise SourceError(f"{shown(path)}: {len(found)} tables under {heading_start!r}")
    return found[0]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in read(path).splitlines() if line.strip()]


def short(cell: str) -> str:
    """'16/48 = 33% [22%–47%]' -> '16/48 (33%)'; '6/8 = 75% [41%–93%], 2 pending' keeps the pending note."""
    m = re.match(r"(\d+/\d+) = (\d+%) \[[^\]]*\](.*)", cell)
    return f"{m.group(1)} ({m.group(2)}){m.group(3)}" if m else cell


def md(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


# --------------------------------------------------------------------------------------------- sources
def question_sets() -> list[list[str]]:
    gold = jsonl(REPO / "eval" / "gold_set.jsonl")
    dev = jsonl(REPO / "eval" / "dev_set.jsonl")
    refusal = jsonl(REPO / "eval" / "refusal_tests.jsonl")
    injection = jsonl(REPO / "eval" / "injection_tests.jsonl")
    cats = Counter(i["category"] for i in gold)
    kinds = Counter(t["kind"] for t in refusal)
    return [["gold-v1 (frozen)", str(len(gold)), ", ".join(f"{c} {n}" for c, n in cats.items())],
            ["development set (tuning only)", str(len(dev)), ", ".join(f"{c} {n}" for c, n in
                                                                     Counter(i["category"] for i in dev).items())],
            ["refusal tests", str(len(refusal)), ", ".join(f"{k} {n}" for k, n in kinds.items())],
            ["prompt-injection tests", str(len(injection)), "every test verified by a second team member"]]


def model_rows(path: Path, runs: list[str] | None = None) -> tuple[list[str], list[list[str]]]:
    """Accuracy by answer type (+ unsupported numbers) for the runs in `path`, short cells, short run names."""
    acc = table(path, "1. Accuracy by answer type")
    hal = table(path, "3. Hallucination and grounding checks")
    runs = runs or acc.header[1:]
    idx = [acc.column(r) for r in runs]
    header = ["Answer type", *[RUN_NAMES.get(r, r) for r in runs]]
    rows = [[r[0], *[short(r[i]) for i in idx]] for r in acc.rows]
    unsupported = hal.row("unsupported numbers")
    rows.append(["Answers with a number not in the passages", *[unsupported[hal.column(r)] for r in runs]])
    return header, rows


def baseline_rows() -> tuple[list[str], list[list[str]]]:
    path = RESULTS / "prompting_baselines.md"
    header, rows = model_rows(path)
    paired = table(path, "10. Paired with")
    cost = table(path, "9. Cost per question")
    all_row = paired.row("All decided answers")
    rows.append(["Right only with the strategy / only with B0 (sign test)", "–",
                 *[all_row[paired.column(r)] for r in paired.header[1:]]])
    secs = cost.row("seconds per question p50 / p95")
    rows.append(["Seconds per question, median / 95th percentile", *secs[1:]])
    return header, rows


def retrieval_rows() -> tuple[list[str], list[list[str]]]:
    path = RESULTS / "retrieval_summary.md"
    grid_tables = [t for t in tables(read(path)) if t.header[:3] == ["Variant", "Retriever", "Filters"]]
    trials = table(path, "Trials on C3")
    if len(grid_tables) != 1:
        raise SourceError("retrieval_summary.md: the grid table is missing or repeated")
    grid = grid_tables[0]
    want = [("grid", ("C3", "hybrid", "none"), "no filters"),
            ("grid", ("C3", "hybrid", "company+year"), "company + year (W1-04 index)"),
            ("trials", ("hybrid", "question+period"), "company, year, period read from the question"),
            ("trials", ("hybrid", "question+period+statement"), "+ financial statements first (default, ADR-017)"),
            ("trials", ("hybrid+rerank", "question+period+statement"), "+ cross-encoder reranker (opt-in)")]
    out = []
    for source, key, label in want:
        t, width = (grid, 3) if source == "grid" else (trials, 2)
        match = [r for r in t.rows if tuple(r[:width]) == key]
        if len(match) != 1:
            raise SourceError(f"retrieval_summary.md: no single row {key}")
        r = match[0]
        cols = {name: r[t.column(name)] for name in ("recall@5", "recall@10", "MRR", "figure@5", "s/query")}
        out.append([label, cols["recall@5"], cols["recall@10"], cols["MRR"], cols["figure@5"], cols["s/query"]])
    return ["Setting (C3 table-aware chunks)", "recall@5", "recall@10", "MRR", "answer figure in top 5", "s/query"], out


def embedding_rows() -> tuple[list[str], list[list[str]]]:
    path = RESULTS / "embedding_comparison.md"
    found = [t for t in tables(read(path)) if t.header[:2] == ["Retriever", "Filters"] and "sign test p" in t.header]
    if len(found) != 1:
        raise SourceError(f"embedding_comparison.md: {len(found)} comparison tables")
    t = found[0]
    rows = []
    for r in t.rows:
        old, new = r[2].split(" / "), r[3].split(" / ")
        rows.append([f"{r[0]}, {r[1]}", old[1], new[1], r[4], r[5]])
    return ["Retriever, filters (C3)", "MiniLM recall@5", "bge-small recall@5", "better / worse / same", "sign test p"], rows


def week1_adrs() -> list[list[str]]:
    rows = []
    for line in read(REPO / "docs" / "decisions.md").splitlines():
        if not line.startswith("| ADR-") or line.startswith("| ADR-0xx"):
            continue
        c = cells(line)
        m = re.match(r"(\d+) (\w{3})", c[1])
        if not m or (MONTHS.get(m.group(2), 0), int(m.group(1))) < WEEK1_FIRST_DAY:
            continue
        decision = re.split(r"(?<=\.)\s(?=[A-Z(`])", c[3], maxsplit=1)[0]      # the first full sentence
        rows.append([c[0], c[1], decision if len(decision) <= 160 else decision[:157].rstrip() + "...",
                     c[6].split(";")[0].split("(")[0].strip()])
    if not rows:
        raise SourceError("docs/decisions.md: no ADR dated in Week 1")
    return sorted(rows)


# --------------------------------------------------------------------------------------------- notes
def note_block() -> str:
    out = ["**Model comparison** (all 80 gold questions + 22 refusal tests, the same retrieved passages for every "
           "model; `eval/results/model_comparison.md`)", ""]
    out += md(*model_rows(RESULTS / "model_comparison.md"))
    out += ["", "**Prompting baselines on Qwen3-8B** (same passages and questions; B0 is the zero-shot run above; "
                "`eval/results/prompting_baselines.md`)", ""]
    out += md(*baseline_rows())
    out += ["", "**Retrieval** (73 gold questions with sources; recall@k checks the right filing section, the answer "
                "figure the exact passage; `eval/results/retrieval_summary.md`)", ""]
    out += md(*retrieval_rows())
    return "\n".join(out) + "\n"


def detailed_block() -> str:
    out = ["### Question sets", ""]
    out += md(["Set", "Items", "Composition"], question_sets())
    out += ["", "### Model comparison by answer type", ""]
    out += md(*model_rows(RESULTS / "model_comparison.md"))
    out += ["", "### Prompting baselines by answer type", ""]
    out += md(*baseline_rows())
    out += ["", "### Retrieval settings", ""]
    out += md(*retrieval_rows())
    out += ["", "### Embedding model (W1-16): MiniLM (local) against bge-small (course endpoint), before W1-04", ""]
    out += md(*embedding_rows())
    out += ["", "### Decisions recorded in Week 1 (docs/decisions.md)", ""]
    out += md(["ADR", "Date", "Decision (first sentence)", "Status"], week1_adrs())
    return "\n".join(out) + "\n"


def fill(path: Path, block: str, title: str) -> str:
    """The note with its generated block replaced (a new note gets a title and empty hand-written parts)."""
    old = path.read_text(encoding="utf-8") if path.exists() else f"# {title}\n\n{START}\n{END}\n\n(written by hand)\n"
    if START not in old or END not in old:
        raise SourceError(f"{shown(path)} lost its GENERATED markers; restore them")
    head, rest = old.split(START, 1)
    return head + f"{START}\n{block}{END}" + rest.split(END, 1)[1]


NOTES = ((NOTE, note_block, "FinSight Week 1 Results Note"),
         (DETAILED, detailed_block, "FinSight Week 1 Detailed Results"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report whether the committed notes are current")
    args = ap.parse_args(argv)
    try:
        new = {path: fill(path, build(), title) for path, build, title in NOTES}
    except SourceError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    stale = [p for p, text in new.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
    if args.check:
        for p in stale:
            print(f"stale: {shown(p)} (run python scripts/week1_notes.py)", file=sys.stderr)
        return 1 if stale else 0
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in new.items():
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"written: {shown(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
