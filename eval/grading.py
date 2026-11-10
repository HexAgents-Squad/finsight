"""W1-14: tools around the human grading of the model comparison (eval/GRADING.md is the graders' guide).

Usage:
    python eval/grading.py copies --names Chandra Amresh --out DIR   # one blind copy of the sheet per grader
    python eval/grading.py handout --out FILE                        # passages for the "grade" rows (markdown)
    python eval/grading.py check --grader 1 FILE                     # one returned copy, as soon as it arrives
    python eval/grading.py merge --g1 FILE --g2 FILE [--write]       # both copies -> the repo's sheet
    python eval/grading.py examples                                  # rows noted "example" -> failure_examples.md

Roles: grader 1 fills `grader_1` on every "grade" row and `failure_label` on the first half of the "label" rows
(in sheet order); grader 2 fills `grader_2` and the second half. Nobody fills `final` before the merge: the merge
sets it where both grades agree and lists the disagreements, which the graders settle together (a `final` of 2 on
a label row overturns the automatic verdict). Then `python eval/score_answers.py` reads the sheet.

Every file is read with the scorer's reader (eval.score_answers.read_sheet_rows: what a spreadsheet adds
harmlessly is undone, anything else is an error naming line and column). Grades are bound to the row key,
so the merge needs the keys and tasks unchanged and in order; a spreadsheet re-formatting an answer cell ("7,367"
-> "7367") only gives a warning, since the merged sheet takes the text columns from the repo's sheet.
Exit codes: 0 = done, 1 = invalid input (nothing written).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.settings import REPO_ROOT  # noqa: E402
from eval.context_pack import GOLD_PACK  # noqa: E402
from eval.score_answers import (  # noqa: E402
    GRADE,
    GRADING_DIR,
    KEY_NAME,
    LABEL,
    RESULTS_DIR,
    SHEET_COLUMNS,
    SHEET_NAME,
    kappa,
    read_sheet_rows,
    replace_file,
)
from eval.validate_gold import DEFAULT_PATH as GOLD_FILE  # noqa: E402
from eval.validate_gold import validate_file  # noqa: E402

SHEET, KEY = GRADING_DIR / SHEET_NAME, GRADING_DIR / KEY_NAME
EXAMPLES_FILE = RESULTS_DIR / "failure_examples.md"
TEXT_COLUMNS = ("question", "gold_answer", "model_answer", "suggested_label")
GRADER_COLUMNS = ("grader_1", "grader_2", "final", "failure_label")
# The guide asks for "example:" and "overturn?", so "for example" or "no overturn" in a note is not a mark
EXAMPLE_MARK = re.compile(r"\bexample\s*:", re.IGNORECASE)
OVERTURN_MARK = re.compile(r"\boverturn\s*\?", re.IGNORECASE)
MAX_LISTED = 20                                      # problems shown at once; the rest are counted
SAFE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,30}")


class GradingError(ValueError):
    """An input the tools refuse; the message says what to fix."""


def load(path: Path) -> list[dict]:
    """The sheet's rows in file order, validated and cleaned by the scorer's reader."""
    if not path.exists():
        raise GradingError(f"{path} does not exist")
    return read_sheet_rows(path)


def listed(problems: list[str]) -> str:
    extra = len(problems) - MAX_LISTED
    return "\n  ".join(problems[:MAX_LISTED] + ([f"... and {extra} more"] if extra > 0 else []))


def copy_problems(base: list[dict], rows: list[dict], grader: int) -> tuple[list[str], list[str]]:
    """What is wrong (problems) or odd (warnings) in one grader's copy, compared with the repo's sheet."""
    name, other = f"grader {grader}", f"grader_{3 - grader}"
    if [(r["key"], r["task"]) for r in rows] != [(r["key"], r["task"]) for r in base]:
        return [f"{name}'s file has rows added, removed or re-ordered (sorted?) or a changed key or task: "
                f"ask for the copy again, filled without sorting"], []
    problems, warnings, split = [], [], split_row(base)
    for n, (r, b) in enumerate(zip(rows, base, strict=True), start=2):
        if r[other] or r["final"]:
            problems.append(f"{name}'s row {n}: {other} or final filled (only grader_{grader} and labels)")
        if b["task"] == LABEL and owner(n, split) != grader and r["failure_label"]:
            warnings.append(f"row {n}: {name} labelled the other grader's row (ignored)")
    edited = [n for n, (a, b) in enumerate(zip(rows, base, strict=True), start=2)
              if any(a[c] != b[c] for c in TEXT_COLUMNS)]
    if edited:
        warnings.append(f"{name}: text re-formatted or edited on rows {edited[:12]}"
                        f"{' ...' if len(edited) > 12 else ''} (the repo's text is kept)")
    return problems, warnings


def check(base_path: Path, path: Path, grader: int) -> list[str]:
    """One returned copy checked on its own, so a mistake is fixed while the grader is still around: its
    problems raise GradingError; the result lists what is missing and any warnings."""
    base, rows = load(base_path), load(path)
    problems, warnings = copy_problems(base, rows, grader)
    if problems:
        raise GradingError(listed(problems))
    split, col = split_row(base), f"grader_{grader}"
    missing = [f"row {n}: no grade in {col}" for n, r in enumerate(rows, start=2) if r["task"] == GRADE and not r[col]]
    missing += [f"row {n}: no failure_label" for n, r in enumerate(rows, start=2)
                if r["task"] == LABEL and owner(n, split) == grader and not r["failure_label"]]
    return [f"grader {grader}: {len(missing)} missing", *missing, *warnings]


def split_row(rows: list[dict]) -> int:
    """The spreadsheet row number (header = row 1) where grader 2's label rows start: grader 1 takes the first
    half of the label rows in sheet order (the larger half when the count is odd)."""
    labels = [n for n, r in enumerate(rows, start=2) if r["task"] == LABEL]
    return labels[(len(labels) + 1) // 2] if len(labels) > 1 else len(rows) + 2


def owner(row_number: int, split: int) -> int:
    return 1 if row_number < split else 2


def has_grades(rows: list[dict]) -> bool:
    return any(r[c].strip() for r in rows for c in GRADER_COLUMNS)


def csv_text(rows: list[dict]) -> str:
    """The rows as the scorer writes its sheet (same columns, LF line ends), so a merge changes only cells."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SHEET_COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows({c: r[c] for c in SHEET_COLUMNS} for r in rows)
    return buf.getvalue()


# --------------------------------------------------------------------------------------------- copies
def copies(sheet: Path, names: list[str], out_dir: Path) -> list[str]:
    """One untouched copy of the sheet per grader, plus assignments.txt saying who fills what. Refuses a sheet
    that already holds grades (a copy would show them to the other grader)."""
    if len(names) != 2 or len(set(n.lower() for n in names)) != 2:
        raise GradingError("give exactly two different grader names (grader 1 first)")
    bad = [n for n in names if not SAFE_NAME.fullmatch(n)]
    if bad:
        raise GradingError(f"names become file names: letters, digits, - and _ only ({bad})")
    rows = load(sheet)
    if has_grades(rows):
        raise GradingError(f"{sheet} already holds grades or labels: copies would show them to the graders")
    split = split_row(rows)
    grade_rows = sum(r["task"] == GRADE for r in rows)
    labels = [n for n, r in enumerate(rows, start=2) if r["task"] == LABEL]
    mine = {1: [n for n in labels if n < split], 2: [n for n in labels if n >= split]}
    out_dir.mkdir(parents=True, exist_ok=True)
    text = sheet.read_bytes()
    lines = [f"Grading sheet: {len(rows)} rows ({grade_rows} grade, {len(labels)} label). Guide: eval/GRADING.md", ""]
    for i, name in enumerate(names, start=1):
        (out_dir / f"grading_sheet_{name}.csv").write_bytes(text)
        rng = f"{mine[i][0]}-{mine[i][-1]}" if mine[i] else "none"
        lines += [f"Grader {i} = {name}: file grading_sheet_{name}.csv",
                  f"  column grader_{i} on all {grade_rows} grade rows",
                  f"  failure_label on the {len(mine[i])} label rows in spreadsheet rows {rng}",
                  "  leave `final` empty; never sort, add or delete rows", ""]
    (out_dir / "assignments.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return lines


# --------------------------------------------------------------------------------------------- handout
def handout(sheet: Path, key: Path, pack_file: Path, gold_file: Path) -> str:
    """Markdown: for every question with a "grade" row, the question, the gold answer and the passages a RAG run
    saw, passages from the gold source first. It names no model: a question's passages are the same for every run
    (the closed-book run saw none), and questions appear in sheet order."""
    rows = load(sheet)
    ids = {r["key"]: r["id"] for r in csv.DictReader(key.open(encoding="utf-8"))}
    gold = {i.id: i for i in validate_file(gold_file).items}
    packs = {p["id"]: p for p in (json.loads(x) for x in pack_file.read_text(encoding="utf-8").splitlines() if x)}
    seen, out = [], ["# Passages for the grade rows (W1-14)", "",
                     "The passages the models with passages were given (the closed-book model saw none). Use them to "
                     "check claims an answer adds beyond the gold answer. Passages marked ★ come from the filing "
                     "section the gold answer was written from.", ""]
    for r in rows:
        if r["task"] != GRADE:
            continue
        if r["key"] not in ids:
            raise GradingError(f"key {r['key']} is not in {key.name}: re-run eval/score_answers.py")
        qid = ids[r["key"]]
        if qid in seen:
            continue
        seen.append(qid)
        sources = {(s.ticker, s.fiscal_year, s.item) for s in gold[qid].gold_sources} if qid in gold else set()
        passages = packs.get(qid, {}).get("passages", [])
        ordered = sorted(passages, key=lambda p: (p["ticker"], p["fiscal_year"], p["item"]) not in sources)
        out += [f"## {r['question']}", "", f"**Gold answer:** {r['gold_answer']}", ""]
        for p in ordered:
            star = "★ " if (p["ticker"], p["fiscal_year"], p["item"]) in sources else ""
            out += [f"**{star}{p['citation']}**", "", p["text"].strip(), ""]
        if not passages:
            out += ["(no passages)", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------- merge
def merge(base_path: Path, g1_path: Path, g2_path: Path) -> tuple[list[dict], dict]:
    """The repo sheet with grader 1's and grader 2's entries, `final` set where both grades agree. Returns the
    merged rows and a report (pairs, disagreements, missing entries, overturn proposals, warnings)."""
    base, g1, g2 = load(base_path), load(g1_path), load(g2_path)
    if has_grades(base):
        raise GradingError(f"{base_path} already holds grades. To merge corrected copies again, first restore the "
                           f"ungraded sheet: git checkout -- {SHEET.relative_to(REPO_ROOT).as_posix()}")
    problems, warnings = [], []
    for grader, rows in ((1, g1), (2, g2)):
        p, w = copy_problems(base, rows, grader)
        problems, warnings = problems + p, warnings + w
    if problems:
        raise GradingError(listed(problems))

    split = split_row(base)
    merged, pairs, disagree, missing, overturn = [], [], [], [], []
    for n, (b, a1, a2) in enumerate(zip(base, g1, g2, strict=True), start=2):
        row = dict(b)
        row["note"] = " | ".join(f"G{i}: {r['note'].strip()}" for i, r in ((1, a1), (2, a2)) if r["note"].strip())
        if b["task"] == GRADE:
            row["grader_1"], row["grader_2"] = a1["grader_1"], a2["grader_2"]
            labels = {r["failure_label"] for r in (a1, a2) if r["failure_label"]}   # optional on grade rows
            row["failure_label"] = labels.pop() if len(labels) == 1 else ""
            if labels:
                disagree.append(f"row {n}: the graders gave different labels ({a1['failure_label']} / "
                                f"{a2['failure_label']}): set one in failure_label")
            if not row["grader_1"] or not row["grader_2"]:
                missing.append(f"row {n}: grade missing (grader 1: {row['grader_1'] or '-'}, "
                               f"grader 2: {row['grader_2'] or '-'})")
            else:
                pairs.append((int(row["grader_1"]), int(row["grader_2"])))
                if row["grader_1"] == row["grader_2"]:
                    row["final"] = row["grader_1"]
                else:
                    disagree.append(f"row {n}: grader 1 = {row['grader_1']}, grader 2 = {row['grader_2']}: "
                                    f"{b['question'][:80]}")
        elif b["task"] == LABEL:
            row["failure_label"] = (a1 if owner(n, split) == 1 else a2)["failure_label"]
            if not row["failure_label"]:
                missing.append(f"row {n}: no failure_label (suggested: {b['suggested_label'] or 'none'})")
        if OVERTURN_MARK.search(row["note"]):
            overturn.append(f"row {n}: {row['note'][:100]}")
        merged.append(row)
    k = kappa(pairs)
    return merged, {"pairs": len(pairs), "agree": sum(x == y for x, y in pairs), "kappa": k,
                    "disagree": disagree, "missing": missing, "overturn": overturn, "warnings": warnings}


# --------------------------------------------------------------------------------------------- examples
def examples(sheet: Path, key: Path) -> str:
    """Markdown for the rows the graders noted "example" (after grading; names the model, so the sheet is no
    longer blind at this point): question, model, answer, gold answer, label and the graders' reason."""
    rows = load(sheet)
    runs = {r["key"]: (r["run_id"], r["id"]) for r in csv.DictReader(key.open(encoding="utf-8"))}
    out = ["# Failure examples (W1-14)", "",
           "Rows the graders marked `example` in eval/results/grading/grading_sheet.csv, in sheet order.", ""]
    picked = [r for r in rows if EXAMPLE_MARK.search(r["note"])]
    for r in picked:
        run_id, qid = runs.get(r["key"], ("(not in the key)", "?"))
        label = r["failure_label"] or r["suggested_label"] or ("graded " + r["final"] if r["final"] else "-")
        out += [f"## {qid}: {label}", "", f"- **Run:** {run_id}", f"- **Question:** {r['question']}",
                f"- **Model answer:** {r['model_answer']}", f"- **Gold answer:** {r['gold_answer']}",
                f"- **Why:** {' '.join(r['note'].split())}", ""]
    if not picked:
        out += ["(no row is marked `example` yet)", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("copies", help="one blind copy of the sheet per grader")
    c.add_argument("--names", nargs="+", required=True, help="grader 1 then grader 2, e.g. Chandra Amresh")
    c.add_argument("--out", type=Path, required=True, help="a folder outside the repo (copies are not committed)")
    h = sub.add_parser("handout", help="passages for the grade rows, as markdown")
    h.add_argument("--out", type=Path, required=True)
    k = sub.add_parser("check", help="one grader's returned copy, on its own")
    k.add_argument("--grader", type=int, choices=(1, 2), required=True)
    k.add_argument("file", type=Path)
    m = sub.add_parser("merge", help="both graders' copies -> the repo's sheet")
    m.add_argument("--g1", type=Path, required=True)
    m.add_argument("--g2", type=Path, required=True)
    m.add_argument("--write", action="store_true", help="write the merged sheet (default: check and report only)")
    e = sub.add_parser("examples", help="rows noted 'example' -> eval/results/failure_examples.md")
    e.add_argument("--out", type=Path, default=EXAMPLES_FILE)
    for p in (c, h, k, m, e):
        p.add_argument("--sheet", type=Path, default=SHEET)
    for p in (h, e):
        p.add_argument("--key", type=Path, default=KEY)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "copies":
            print("\n".join(copies(args.sheet, args.names, args.out)))
        elif args.cmd == "handout":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            replace_file(args.out, handout(args.sheet, args.key, RESULTS_DIR / GOLD_PACK, GOLD_FILE))
            print(f"written: {args.out}")
        elif args.cmd == "check":
            print("\n".join(check(args.sheet, args.file, args.grader)))
        elif args.cmd == "merge":
            merged, rep = merge(args.sheet, args.g1, args.g2)
            k = f", Cohen's kappa {rep['kappa']:.2f}" if rep["kappa"] is not None else ""
            print(f"grade rows: {rep['pairs']} graded by both, {rep['agree']} agree{k}")
            for title, items in (("disagreements to settle together (set `final`)", rep["disagree"]),
                                 ("missing", rep["missing"]),
                                 ("overturn proposals (decide together; final = 2 overturns)", rep["overturn"]),
                                 ("warnings", rep["warnings"])):
                print(f"{title}: {len(items)}")
                for line in items:
                    print(f"  {line}")
            if args.write:
                replace_file(args.sheet, csv_text(merged))
                print(f"written: {args.sheet} (settle the disagreements in `final`, then run eval/score_answers.py)")
            else:
                print("check only: add --write to write the merged sheet")
        else:
            replace_file(args.out, examples(args.sheet, args.key))
            print(f"written: {args.out}")
    except (GradingError, ValueError, OSError) as err:
        hint = " (is the file open in Excel? close it and run again)" if isinstance(err, PermissionError) else ""
        print(f"ERROR: {err}{hint}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
