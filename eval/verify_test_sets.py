"""Second-person verification of the safety test sets (W1-10): a sheet for the verifier, then `verified_by`.

Usage:
    python eval/verify_test_sets.py sheet verification.csv              # one row per refusal and injection test
    python eval/verify_test_sets.py apply verification.csv --dry-run    # check the filled sheet, write nothing
    python eval/verify_test_sets.py apply verification.csv              # set verified_by from the 'agree' rows

The sheet shows each test as the evaluation will use it: the question, what the agent must do (with the spec §6
rule and its reason), the rationale, and for an injection test the planted text, the phrases that must not
appear, the correct figures and the SEC filing to check them in. The verifier fills three columns per row:
    verifier   their name (must not be the test's author: the two-person rule, ADR-009)
    verdict    agree     the test is right as written; the expected behaviour follows spec §6, and every figure
                         was re-checked in the filing
               disagree  something must change first; `comment` says what (the author fixes the test, makes a new
                         sheet, and the verifier checks that row again)
    comment    optional for agree
Rows left blank stay unverified. `apply` writes nothing if any row disagrees, names the test's author as the
verifier, belongs to a test that changed since the sheet was made (its fingerprint differs: make a new sheet),
or names an unknown test. Otherwise it sets `verified_by` in eval/refusal_tests.jsonl and
eval/injection_tests.jsonl (every other byte stays as it was) and says whether `validate_test_sets.py --final`
now passes. Open and save the sheet in Excel as "CSV UTF-8" (or in Google Sheets: File → Download → CSV).
Exit codes: 0 = applied (or would be with --dry-run), 1 = problems in the sheet or the test files, 2 = a file is
missing or unreadable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.contract import REFUSAL_REASONS  # noqa: E402
from eval.schema import _norm  # noqa: E402
from eval.validate_test_sets import SETS, validate_set  # noqa: E402

SECTIONS = ROOT / "data" / "sections" / "sections.jsonl"
COLUMNS = ["set", "id", "kind", "question", "expected", "rationale", "planted_text", "must_not_contain",
           "must_contain_any", "filing", "written_by", "fingerprint", "verifier", "verdict", "comment"]
FILLED = ("verifier", "verdict", "comment")
VERDICTS = ("agree", "disagree")
R3_REASON = "outside the indexed filings (the R3 sentence)"


def fingerprint(record: dict) -> str:
    """A short hash of everything the verifier checks (all fields but verified_by)."""
    body = {k: v for k, v in record.items() if k != "verified_by"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def rule_text(rule: str | None) -> str:
    return f"{rule}, {REFUSAL_REASONS.get(rule, R3_REASON)}" if rule else ""


def expected(set_name: str, rec: dict) -> str:
    """What the agent must do, in words."""
    if set_name == "refusal":
        return {"refusal": f"refuse ({rule_text(rec['expected_rule'])})",
                "partial": f"answer the factual part, decline the rest in one sentence ({rule_text(rec['expected_rule'])})",
                "control": "answer normally: must NOT be refused"}[rec["kind"]]
    refused = f"refuse the user's question ({rule_text(rec['expected_rule'])}); " if rec["expected_refused"] else ""
    return refused + rec["expected_behavior"]


def filing_links(path: Path = SECTIONS) -> dict[tuple, str]:
    """(ticker, form, fiscal year, fiscal period, Item) -> SEC URL, from the section corpus when it is built."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[(d["ticker"], d["form"], d["fiscal_year"], d["fiscal_period"], d["item"])] = d["source_url"]
    return out


def sheet_rows(records: dict[str, list[dict]], links: dict[tuple, str]) -> list[dict]:
    rows = []
    for set_name, recs in records.items():
        for rec in recs:
            row = dict.fromkeys(COLUMNS, "")
            row.update({"set": set_name, "id": rec["id"], "question": rec["question"],
                        "expected": expected(set_name, rec), "written_by": rec["written_by"],
                        "fingerprint": fingerprint(rec), "verifier": rec.get("verified_by") or ""})
            if row["verifier"]:
                row["verdict"] = "agree"                 # already verified: shown as such, applied again as is
            if set_name == "refusal":
                row.update(kind=rec["kind"], rationale=rec["rationale"])
            else:
                src = rec["planted_in"]
                key = (src["ticker"], src["form"], src["fiscal_year"], src["fiscal_period"], src["item"])
                where = f"{src['ticker']} {src['form']} FY{src['fiscal_year']} {src['fiscal_period']} {src['item']}"
                row.update(kind=rec["injection_type"], rationale=rec.get("notes") or "",
                           planted_text=rec["planted_text"], must_not_contain="; ".join(rec["must_not_contain"]),
                           must_contain_any="; ".join(rec["must_contain_any"]),
                           filing=f"{where} {links[key]}" if key in links else where)
            rows.append(row)
    return rows


def read_sets(paths: dict[str, Path]) -> dict[str, list[dict]]:
    return {name: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for name, path in paths.items()}


def write_sheet(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:      # utf-8-sig: Excel reads the accents right
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_sheet(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("set", "id", "fingerprint", *FILLED) if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"sheet has no column(s) {missing}; make it with 'sheet'")
        return [{k: (v or "").strip() for k, v in row.items() if k} for row in reader]


def apply_sheet(rows: list[dict], records: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], list[str], int]:
    """(records with verified_by set, problems, rows applied). Problems mean nothing may be written."""
    by_key = {(name, rec["id"]): rec for name, recs in records.items() for rec in recs}
    updated = {name: [dict(rec) for rec in recs] for name, recs in records.items()}
    target = {(name, rec["id"]): rec for name, recs in updated.items() for rec in recs}
    problems, applied, seen = [], 0, set()
    for n, row in enumerate(rows, start=2):                          # line 1 is the header
        key = (row.get("set", ""), row.get("id", ""))
        label = f"row {n} ({key[1] or '?'})"
        if key in seen:
            problems.append(f"{label}: the same test appears twice")
            continue
        seen.add(key)
        if key not in by_key:
            problems.append(f"{label}: no {key[0] or '?'} test with this id")
            continue
        verifier, verdict = row["verifier"], row["verdict"].lower()
        if not verifier and not verdict:
            continue                                                 # not checked yet: stays unverified
        rec = by_key[key]
        if row["fingerprint"] != fingerprint(rec):
            problems.append(f"{label}: the test changed after the sheet was made; make a new sheet")
        elif verdict not in VERDICTS:
            problems.append(f"{label}: verdict must be 'agree' or 'disagree', not {row['verdict']!r}")
        elif not verifier:
            problems.append(f"{label}: verdict without a verifier name")
        elif _norm(verifier) == _norm(rec["written_by"]):
            problems.append(f"{label}: {verifier} wrote this test, so a different person must verify it")
        elif verdict == "disagree":
            problems.append(f"{label}: {verifier} disagrees: {row['comment'] or '(no comment)'}")
        else:
            target[key]["verified_by"] = verifier
            applied += 1
    return updated, problems, applied


def check_records(records: dict[str, list[dict]]) -> tuple[list[str], list[str]]:
    """(errors under drafting rules, errors under final rules) of the updated sets, validated as they would be
    written (through validate_test_sets.py's own reader and rules)."""
    draft, final = [], []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, recs in records.items():
            path = Path(tmp_dir) / f"{name}.jsonl"
            path.write_text(dumps(recs), encoding="utf-8", newline="\n")
            draft += [f"{name}: {e}" for e in validate_set(name, path)[1]]
            final += [f"{name}: {e}" for e in validate_set(name, path, final=True)[1]]
    return draft, final


def dumps(records: list[dict]) -> str:
    return "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sheet", "apply"])
    ap.add_argument("sheet", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="apply: check the sheet, write nothing")
    args = ap.parse_args(argv)
    paths = {name: spec[3] for name, spec in SETS.items()}
    try:
        records = read_sets(paths)
        if args.command == "sheet":
            rows = sheet_rows(records, filing_links())
            write_sheet(args.sheet, rows)
            print(f"{len(rows)} tests -> {args.sheet}; fill in verifier, verdict (agree/disagree) and comment")
            return 0
        rows = read_sheet(args.sheet)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    updated, problems, applied = apply_sheet(rows, records)
    draft, final = check_records(updated)
    for p in problems + draft:
        print(f"PROBLEM: {p}")
    if problems or draft:
        print("nothing written")
        return 1
    if not args.dry_run:
        for name, recs in updated.items():
            if recs != records[name]:
                write_atomic(paths[name], dumps(recs))
    verified = sum(bool(r.get("verified_by")) for recs in updated.values() for r in recs)
    total = sum(len(recs) for recs in updated.values())
    print(f"{applied} rows applied{' (dry run, nothing written)' if args.dry_run else ''}; "
          f"{verified} of {total} tests verified")
    print("OK: validate_test_sets.py --final passes" if not final else
          f"--final not yet: {len(final)} problem(s), e.g. {final[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
