"""W2-14: the Week 2 results notes, generated from committed repo data (docs/week2_plan.txt, W2-14).

Usage:
    python scripts/week2_notes.py            # rewrite the generated tables of both notes
    python scripts/week2_notes.py --check    # exit 1 if a committed note's tables are stale (tests do the same)

Writes docs/report/week2_results_note.md (one page) and docs/report/week2_detailed_results.md. As in W1-11
(scripts/week1_notes.py, whose table readers this reuses), only the tables between the GENERATED markers are rebuilt;
the text around them is written by hand and kept. Every number is read from a committed file, so a fresh clone (CI)
regenerates them exactly, without a key or data/:
    eval/results/agent_baselines.md        W1-08 scorer: the agent run A1 next to B0 and S1-S3 (W2-11, W2-12)
    eval/results/demo/W2-07_demo.jsonl     the 10 demo queries through the real graph (W2-07)
    eval/results/injection_results.md      the 12 injection tests, defences off and on (W2-13)
    docs/smoke_tests.md                    the smoke test on each team member's laptop (W1-10, W2-12)
    docs/decisions.md                      the Week 2 ADRs (ADR-024 to ADR-035, dated 16-22 Nov)
A source table that is missing or changed shape is an error naming the file, never a silently empty cell. The notes
are a record of Week 2, so nothing Week 3 adds may change them: the runs are named (B0, S1-S3, A1; a later run such
as A5 in the same report is not shown), the ADRs stop at ADR-035, and the injection detector's held-out rates are
quoted in the text from eval/results/injection_results.md rather than re-scored on the current patterns.
Exit codes: 0 = written (or current with --check), 1 = stale (--check) or a source is unreadable.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.week1_notes import (  # noqa: E402  (the W1-11 readers: one parser for both weeks' notes)
    RUN_NAMES,
    SourceError,
    Table,
    adrs_between,
    jsonl,
    md,
    read,
    short,
    shown,
    table,
    tables,
)

RESULTS = REPO / "eval" / "results"
REPORT_DIR = REPO / "docs" / "report"
NOTE = REPORT_DIR / "week2_results_note.md"
DETAILED = REPORT_DIR / "week2_detailed_results.md"
START, END = "<!-- GENERATED:START (scripts/week2_notes.py) -->", "<!-- GENERATED:END -->"
AGENT_REPORT = RESULTS / "agent_baselines.md"
DEMO = RESULTS / "demo" / "W2-07_demo.jsonl"
INJECTION = RESULTS / "injection_results.md"
SMOKE = REPO / "docs" / "smoke_tests.md"
WEEK2_FIRST_DAY, WEEK2_LAST_DAY = (11, 16), (11, 22)    # ADR-024 onwards; W2-12/13 merged on 22 Nov
WEEK2_LAST_ADR = 35                                       # ADR-036 onwards are Week 3's, whatever their date

A1 = "A1_qwen3-8b_agent_think-off_a-v1"
B0 = "M1_qwen3-8b_rag_think-off_b0-v1"
S1, S2, S3 = ("S1_qwen3-8b_rag_think-off_s1-v1", "S2_qwen3-8b_rag_think-off_s2-v1", "S3_qwen3-8b_rag_think-off_s3-v1")
WEEK2_RUNS = [B0, S1, S2, S3, A1]                        # the runs Week 2 compared (a later run is not shown)
NAMES = {**RUN_NAMES, A1: "A1 agent"}


def name(run: str) -> str:
    return NAMES.get(run, run)


# --------------------------------------------------------------------------------------------- the agent run
def agent_rows(runs: list[str] | None = None) -> tuple[list[str], list[list[str]]]:
    """Accuracy by answer type, the grounding check (both rules), the paired sign test against B0 and the cost,
    for `runs` (default: the Week 2 runs), with short cells and short run names."""
    acc = table(AGENT_REPORT, "1. Accuracy by answer type")
    hal = table(AGENT_REPORT, "3. Hallucination and grounding checks")
    paired = table(AGENT_REPORT, "10. Paired with")
    cost = table(AGENT_REPORT, "9. Cost per question")
    runs = runs or WEEK2_RUNS
    header = ["Answer type", *[name(r) for r in runs]]
    rows = [[r[0], *[short(r[acc.column(run)]) for run in runs]] for r in acc.rows]
    own = hal.row("unsupported numbers")
    rows.append(["Answers with an unsupported number (each run's own evidence)", *[own[hal.column(r)] for r in runs]])
    strict = [r for r in hal.rows if r[0].startswith("unsupported numbers, no calculation result")]
    if len(strict) != 1:
        raise SourceError(f"{shown(AGENT_REPORT)}: no single row for B0's grounding rule")
    rows.append(["... no calculation result counted as evidence (B0's rule)",
                 *[strict[0][hal.column(r)] for r in runs]])
    everything = paired.row("All decided answers")
    rows.append(["Right only with the run / only with B0 (sign test)",
                 *["–" if r == B0 else everything[paired.column(r)] for r in runs]])
    for label, key in (("Tokens per question (mean)", "tokens per question (mean, prompt + completion)"),
                       ("Seconds per question, median / 95th percentile", "seconds per question p50 / p95")):
        row = cost.row(key)
        rows.append([label, *[row[cost.column(r)] for r in runs]])
    return header, rows


def agent_table(heading: str, runs: list[str], label: str, keep=None) -> tuple[list[str], list[list[str]]]:
    """One table of the agent report for `runs`, short cells; `keep` filters its rows by their first cell."""
    t = table(AGENT_REPORT, heading)
    rows = [[r[0], *[short(r[t.column(run)]) for run in runs]] for r in t.rows if keep is None or keep(r[0])]
    return [label, *[name(r) for r in runs]], rows


def agent_checks() -> tuple[list[str], list[list[str]]]:
    t = table(AGENT_REPORT, "11. The agent's own checks and cost")
    return ["The agent's own checks (A1)", "Value"], [[r[0], r[t.column(A1)]] for r in t.rows]


# --------------------------------------------------------------------------------------------- the demo
DEMO_FIELDS = ("id", "category", "question", "verdict", "met", "under_time_limit", "seconds", "tokens", "tool_calls")


def demo() -> list[dict]:
    try:
        rows = jsonl(DEMO)
    except ValueError as err:                       # a line that is not JSON
        raise SourceError(f"{shown(DEMO)}: {err}") from err
    if not rows:
        raise SourceError(f"{shown(DEMO)} is empty")
    for n, row in enumerate(rows, start=1):
        missing = [f for f in DEMO_FIELDS if f not in row]
        if missing:
            raise SourceError(f"{shown(DEMO)} line {n}: no {', '.join(missing)}")
    return rows


def demo_summary() -> tuple[list[str], list[list[str]]]:
    rows = demo()
    secs = [r["seconds"] for r in rows]
    verdicts = Counter(r["verdict"] for r in rows)
    return (["Demo queries (W2-07, real graph, tools and data)", "Value"],
            [["queries that met their expectation", f"{sum(r['met'] for r in rows)}/{len(rows)}"],
             ["verdicts", "; ".join(f"{v} {n}" for v, n in verdicts.most_common())],     # a verdict may hold ","
             ["under the 60 s limit", f"{sum(r['under_time_limit'] for r in rows)}/{len(rows)}"],
             ["seconds per query, median / max", f"{statistics.median(secs):.1f} / {max(secs):.1f}"],
             ["tokens per query (mean)", f"{statistics.mean(r['tokens'] for r in rows):,.0f}"],
             ["tool calls (total)", str(sum(len(r['tool_calls']) for r in rows))]])


def demo_rows() -> tuple[list[str], list[list[str]]]:
    out = []
    for r in demo():
        tools = ", ".join(dict.fromkeys(c["tool"] for c in r["tool_calls"])) or "none"
        question = r["question"] if len(r["question"]) <= 90 else r["question"][:87].rstrip() + "..."
        out.append([r["id"], r["category"], question, r["verdict"], "yes" if r["met"] else "NO", tools,
                    f"{r['seconds']:.1f}"])
    return ["Query", "Category", "Question", "Verdict", "Met", "Tools", "s"], out


# --------------------------------------------------------------------------------------------- safety
def injection_rows() -> tuple[list[str], list[list[str]]]:
    found = [t for t in tables(read(INJECTION)) if t.header and t.header[0] == "Measure"]
    if len(found) != 1:
        raise SourceError(f"{shown(INJECTION)}: {len(found)} tables headed 'Measure'")
    t: Table = found[0]
    return ["The 12 injection tests (W2-13)", "Defences off", "Defences on"], [[r[0], r[1], r[2]] for r in t.rows]


# --------------------------------------------------------------------------------------------- the team
def smoke_rows() -> tuple[list[str], list[list[str]]]:
    found = [t for t in tables(read(SMOKE)) if t.header[:2] == ["Member", "Date"]]
    if len(found) != 1:
        raise SourceError(f"{shown(SMOKE)}: {len(found)} smoke-test tables")
    t = found[0]
    result = t.column("Result")
    return (["Member", "OS, Python", "Result"],
            [[r[0], r[t.column("OS, Python")], r[result].split(":")[0] or "pending"] for r in t.rows])


def week2_adrs() -> list[list[str]]:
    return [r for r in adrs_between(WEEK2_FIRST_DAY, WEEK2_LAST_DAY, "Week 2") if int(r[0][4:]) <= WEEK2_LAST_ADR]


# --------------------------------------------------------------------------------------------- notes
def note_block() -> str:
    out = ["**The agent next to Week 1's runs** (all 80 gold questions + 22 refusal tests, the same scorer; B0 and "
           "S1-S3 read a fixed context pack, A1 chose its own tools; `eval/results/agent_baselines.md`)", ""]
    out += md(*agent_rows())
    out += ["", "**The 10 demo queries** (`eval/results/demo/W2-07_demo.md`) and **the injection tests** "
                "(`eval/results/injection_results.md`)", ""]
    out += md(*demo_summary())
    out += [""]
    out += md(*injection_rows())
    return "\n".join(out) + "\n"


def detailed_block() -> str:
    out = ["### The agent next to Week 1's runs", ""]
    out += md(*agent_rows())
    out += ["", "### By gold category and refusal-test kind (B0 and A1)", ""]
    out += md(*agent_table("2. Accuracy by gold category", [B0, A1], "Category"))
    out += ["", "### Refusals (B0 and A1)", ""]
    out += md(*agent_table("4. Refusals", [B0, A1], "Measure"))
    out += ["", "### Failure labels of wrong answers (B0 and A1; confirmed in the grading sheet, else suggested)", ""]
    out += md(*agent_table("6. Failure labels", [B0, A1], "Label"))
    out += ["", "### Human grading, second round (W2-12)", ""]
    out += md(*agent_table("7. Human grades", [S1, S2, S3, A1], "Measure"))
    out += ["", "### The agent's own checks and cost", ""]
    out += md(*agent_checks())
    out += ["", "### The 10 demo queries (W2-07)", ""]
    out += md(*demo_summary())
    out += [""]
    out += md(*demo_rows())
    out += ["", "### Prompt-injection and PII defences (W2-13)", ""]
    out += md(*injection_rows())
    out += ["", "### Smoke test on every laptop (W1-10, closed in W2-12)", ""]
    out += md(*smoke_rows())
    out += ["", "### Decisions recorded in Week 2 (docs/decisions.md)", ""]
    out += md(["ADR", "Date", "Decision (first sentence)", "Status"], week2_adrs())
    return "\n".join(out) + "\n"


def fill(path: Path, block: str, title: str) -> str:
    """The note with its generated block replaced (a new note gets a title and empty hand-written parts)."""
    old = path.read_text(encoding="utf-8") if path.exists() else f"# {title}\n\n{START}\n{END}\n\n(written by hand)\n"
    if START not in old or END not in old or old.index(END) < old.index(START):
        raise SourceError(f"{shown(path)} lost its GENERATED markers; restore them")
    head, rest = old.split(START, 1)
    return head + f"{START}\n{block}{END}" + rest.split(END, 1)[1]


NOTES = ((NOTE, note_block, "FinSight Week 2 Results Note"),
         (DETAILED, detailed_block, "FinSight Week 2 Detailed Results"))


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
            print(f"stale: {shown(p)} (run python scripts/week2_notes.py)", file=sys.stderr)
        return 1 if stale else 0
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in new.items():
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"written: {shown(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
