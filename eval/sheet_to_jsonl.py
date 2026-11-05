"""Convert the team's Google Sheet of gold questions (File → Download → CSV) into eval/gold_set.jsonl.

Usage:
    python eval/sheet_to_jsonl.py gold_sheet.csv                     # writes eval/gold_set.jsonl if valid
    python eval/sheet_to_jsonl.py gold_sheet.csv -o other.jsonl --final
    python eval/sheet_to_jsonl.py gold_sheet.csv --dry-run           # validate only, write nothing

Sheet columns (header row, any order, case-insensitive; template: eval/gold_sheet_template.csv):
    id, category, question, companies, fiscal_years, gold_answer, gold_value, unit, tolerance_pct,
    gold_sources, must_refuse, refusal_rule, previous_turn_id, written_by, verified_by, notes
Cell formats:
    category      fact | ratio | period_comparison | multi_company | qualitative | multi_turn | refuse,
                  or the letter A–G, or "period comparison", "multi-company", "multi-turn", "refusal"
    companies     tickers separated by commas:            MSFT, GOOGL
    fiscal_years  4-digit years separated by commas ("FY" ok):    2023, FY2024
    gold_value    number; "$", "%" and thousands commas are ignored:   44.6   or   391,035   or (1,200)
    gold_sources  one source per line or separated by ";", each "TICKER | FORM | YEAR | PERIOD | ITEM";
                  PERIOD may be omitted for a 10-K (FY):   MSFT | 10-K | 2024 | Item 8
                                                            AAPL | 10-Q | 2025 | Q2 | Part I Item 2
    must_refuse   TRUE/FALSE, yes/no, 1/0 (empty = FALSE)
The output is written ONLY if every row passes the validator (eval/validate_gold.py rules), so the
committed file is always valid. Rows are sorted by id for clean diffs.
Exit codes: 0 = valid (written), 1 = problems in the sheet, 2 = file missing or not UTF-8.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.schema import CATEGORY_LETTER, DEFAULT_TOLERANCE_PCT  # noqa: E402
from eval.validate_gold import safe_console, summary, validate_records  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "gold_set.jsonl"
LIST_SEPARATORS = re.compile(r"[,;\n]")         # companies / fiscal_years cells
SOURCE_SEPARATORS = re.compile(r"[;\n]")        # gold_sources cell: one source per line or ";"
SOURCE_FIELDS = 5                                # TICKER | FORM | YEAR | PERIOD | ITEM
REQUIRED = ("id", "category", "question", "gold_answer", "written_by")
KNOWN = REQUIRED + ("companies", "fiscal_years", "gold_value", "unit", "tolerance_pct", "gold_sources",
                    "must_refuse", "refusal_rule", "previous_turn_id", "verified_by", "notes")
FIELD_ORDER = ("id", "category", "question", "companies", "fiscal_years", "gold_answer", "gold_value", "unit",
               "tolerance_pct", "gold_sources", "must_refuse", "refusal_rule", "previous_turn_id",
               "written_by", "verified_by", "notes")

_LETTER_TO_CATEGORY = {v: k for k, v in CATEGORY_LETTER.items()}
_CATEGORY_ALIASES = {"period comparison": "period_comparison", "period": "period_comparison",
                     "multi-company": "multi_company", "multi company": "multi_company",
                     "multi-turn": "multi_turn", "multi turn": "multi_turn", "follow-up": "multi_turn",
                     "refusal": "refuse", "refused": "refuse", "not answerable": "refuse"}
_TRUE, _FALSE = {"true", "yes", "y", "1"}, {"false", "no", "n", "0", ""}


class CellError(ValueError):
    """A cell that cannot be parsed; reported with its row and column."""


def _blank(value: str | None) -> bool:
    return value is None or not value.strip()


def header_key(header: str) -> str:
    """'Gold Answer', ' gold_answer ' and 'GOLD-ANSWER' all name the gold_answer column."""
    return "_".join(header.replace("-", " ").replace("_", " ").split()).casefold()


def parse_category(value: str) -> str:
    v = value.strip()
    if v.upper() in _LETTER_TO_CATEGORY:
        return _LETTER_TO_CATEGORY[v.upper()]
    low = v.lower()
    return _CATEGORY_ALIASES.get(low, low.replace(" ", "_").replace("-", "_"))


def parse_list(value: str) -> list[str]:
    return [p.strip() for p in LIST_SEPARATORS.split(value or "") if p.strip()]


def parse_year(value: str) -> int:
    v = value.strip().upper().removeprefix("FY").strip()
    if not (v.isdigit() and len(v) == 4):              # "FY24" or "2,024" must not silently become 24
        raise CellError(f"'{value}' is not a year")
    return int(v)


def parse_number(value: str) -> float | None:
    if _blank(value):
        return None
    v = value.strip().replace(",", "").replace("$", "").replace("%", "").replace("−", "-").strip()
    if v.startswith("(") and v.endswith(")"):           # accounting negative: (1,234) -> -1234
        v = "-" + v[1:-1]
    try:
        number = float(v)
    except ValueError as err:
        raise CellError(f"'{value}' is not a number") from err
    # Checked only once the cell parses as a number, so "none" is "not a number"; float() also accepts every
    # exponent form (3.91E+05, +3.91e5, 3.E5), which Excel writes for narrow columns and which loses digits
    if "e" in v.lower():
        raise CellError(f"'{value}' is in scientific notation; format the column as a plain number and export again")
    if not math.isfinite(number):
        raise CellError(f"'{value}' is not a finite number")
    return number


def parse_bool(value: str) -> bool:
    v = (value or "").strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise CellError(f"'{value}' is not TRUE/FALSE")


def parse_sources(value: str) -> list[dict]:
    sources = []
    for entry in SOURCE_SEPARATORS.split(value or ""):
        if not entry.strip():
            continue
        parts = [p.strip() for p in entry.split("|")]
        if len(parts) == SOURCE_FIELDS - 1 and parts[1].upper() == "10-K":
            parts.insert(3, "FY")                         # a 10-K always covers the full fiscal year
        if len(parts) != SOURCE_FIELDS:
            raise CellError(f"source '{entry.strip()}' must be 'TICKER | FORM | YEAR | PERIOD | ITEM' "
                            "(PERIOD may be omitted only for a 10-K)")
        ticker, form, year, period, item = parts
        sources.append({"ticker": ticker.upper(), "form": form.upper(), "fiscal_year": parse_year(year),
                        "fiscal_period": period.upper(), "item": " ".join(item.split())})
    return sources


def row_to_record(row: dict[str, str]) -> dict:
    def cell(name: str) -> str:
        return (row.get(name) or "").strip()

    rec: dict = {
        "id": cell("id").upper(),
        "category": parse_category(cell("category")),
        "question": cell("question"),
        "companies": [t.upper() for t in parse_list(cell("companies"))],
        "fiscal_years": [parse_year(y) for y in parse_list(cell("fiscal_years"))],
        "gold_answer": cell("gold_answer"),
        "gold_value": parse_number(cell("gold_value")),
        "unit": cell("unit") or None,
        "tolerance_pct": parse_number(cell("tolerance_pct")),
        "gold_sources": parse_sources(cell("gold_sources")),
        "must_refuse": parse_bool(cell("must_refuse")),
        "refusal_rule": cell("refusal_rule").upper() or None,
        "previous_turn_id": cell("previous_turn_id").upper() or None,
        "written_by": cell("written_by"),
        "verified_by": cell("verified_by") or None,
        "notes": cell("notes") or None,
    }
    if rec["tolerance_pct"] is None:
        rec["tolerance_pct"] = DEFAULT_TOLERANCE_PCT
    return {k: rec[k] for k in FIELD_ORDER}


def read_sheet(path: Path) -> tuple[list[tuple[int, dict]], list[str]]:
    """Return (records with their sheet row numbers, errors). Row 1 is the header."""
    errors: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = [header_key(h) for h in (reader.fieldnames or [])]
        if duplicated := sorted({h for h in headers if h and headers.count(h) > 1}):
            return [], [f"header row names column(s) twice (after case/space normalisation): {duplicated}"]
        reader.fieldnames = headers
        missing = [c for c in REQUIRED if c not in headers]
        if missing:
            return [], [f"header row is missing required column(s): {missing}"]
        unknown = [h for h in headers if h and h not in KNOWN]
        if unknown:
            errors.append(f"unknown column(s) ignored: {unknown}")
        records = []
        for row_no, row in enumerate(reader, start=2):
            if all(_blank(v) for v in row.values() if isinstance(v, str)):
                continue                                  # skip empty spreadsheet rows
            if _blank(row.get("id")):
                errors.append(f"row {row_no}: id is empty (fill it in, or delete the row if it is not a question)")
                continue
            try:
                records.append((row_no, row_to_record(row)))
            except CellError as err:
                errors.append(f"row {row_no} ({(row.get('id') or '').strip() or 'no id'}): {err}")
    return records, errors


def main(argv: list[str] | None = None) -> int:
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--final", action="store_true", help="apply the freeze rules before writing")
    ap.add_argument("--dry-run", action="store_true", help="validate only; do not write the output file")
    args = ap.parse_args(argv)

    try:
        records, sheet_errors = read_sheet(args.csv_path)
    except FileNotFoundError:
        print(f"ERROR: {args.csv_path} not found", file=sys.stderr)
        return 2
    except UnicodeDecodeError:
        print(f"ERROR: {args.csv_path} is not UTF-8; in Excel use 'CSV UTF-8', in Google Sheets "
              "File > Download > CSV", file=sys.stderr)
        return 2
    blocking = [e for e in sheet_errors if not e.startswith("unknown column")]
    for e in sheet_errors:
        print(("ERROR: " if e in blocking else "WARNING: ") + e)
    rep = validate_records(records, final=args.final)
    print(summary(rep.items))
    for w in rep.warnings:
        print(f"WARNING: {w.replace('line ', 'row ', 1)}")
    for e in rep.errors:
        print(f"ERROR: {e.replace('line ', 'row ', 1)}")
    if blocking or not rep.ok:
        print(f"NOT WRITTEN: fix {len(blocking) + len(rep.errors)} problem(s) in the sheet and export again")
        return 1
    if args.dry_run:
        print("OK: sheet valid (dry run, nothing written)")
        return 0
    ordered = sorted((rec for _, rec in records), key=lambda r: r["id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for rec in ordered:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"OK: wrote {len(ordered)} items to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
