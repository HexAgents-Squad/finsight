"""Validate the gold evaluation set (eval/gold_set.jsonl).

Usage:
    python eval/validate_gold.py                          # default file, drafting rules
    python eval/validate_gold.py eval/gold_set.jsonl --final   # freeze rules (before tagging gold-v1)
    python eval/validate_gold.py --allow-missing          # CI: succeed if the file does not exist yet

Exit codes: 0 = valid, 1 = problems found, 2 = file missing or unreadable.
Rules: eval/schema.py (types + meaning per item, plus set-level and --final rules).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError  # noqa: E402

from eval.schema import (  # noqa: E402
    CATEGORY_LETTER,
    TARGET_COUNTS,
    GoldItem,
    item_problems,
    item_warnings,
    set_problems,
)

DEFAULT_PATH = Path(__file__).resolve().parent / "gold_set.jsonl"


@dataclass
class Report:
    items: list[GoldItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def format_validation_error(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"]) or "record"
        parts.append(f"{loc}: {e['msg']}")
    return "; ".join(parts)


def validate_records(records: list[tuple[int, dict]], final: bool = False,
                     model: type[GoldItem] = GoldItem) -> Report:
    """Validate parsed records given as (line_number, dict). Used by the CLI, the converter and tests; `model` is
    GoldItem, or DevItem for the development set (eval/validate_dev_set.py)."""
    rep = Report()
    for line_no, rec in records:
        label = f"line {line_no}" + (f" ({rec.get('id')})" if isinstance(rec, dict) and rec.get("id") else "")
        try:
            item = model.model_validate(rec)
        except ValidationError as err:
            rep.errors.append(f"{label}: {format_validation_error(err)}")
            continue
        rep.items.append(item)
        rep.errors += [f"{label}: {p}" for p in item_problems(item)]
        rep.warnings += [f"{label}: {w}" for w in item_warnings(item)]
    set_rep = set_problems(rep.items, final=final)
    rep.errors += set_rep.errors
    rep.warnings += set_rep.warnings
    return rep


def read_jsonl(path: Path) -> tuple[list[tuple[int, dict]], list[str]]:
    records, errors = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as err:
            errors.append(f"line {n}: invalid JSON ({err.msg} at column {err.colno})")
            continue
        if not isinstance(rec, dict):
            errors.append(f"line {n}: each line must be a JSON object")
            continue
        records.append((n, rec))
    return records, errors


def validate_file(path: Path, final: bool = False, model: type[GoldItem] = GoldItem) -> Report:
    records, parse_errors = read_jsonl(path)
    rep = validate_records(records, final=final, model=model)
    rep.errors = parse_errors + rep.errors
    return rep


def summary(items: list[GoldItem]) -> str:
    counts = Counter(i.category for i in items)
    cats = ", ".join(f"{CATEGORY_LETTER[c]} {c}: {counts.get(c, 0)}/{t}" for c, t in TARGET_COUNTS.items())
    refusals = Counter(i.refusal_rule for i in items if i.must_refuse)
    traps = sum(1 for i in items if i.is_trap)
    verified = sum(1 for i in items if i.verified_by)
    return (f"{len(items)} items read | {cats}\n"
            f"refusal rules: {dict(sorted(refusals.items()))} | 'not reported' traps: {traps} | "
            f"verified: {verified}/{len(items)}")


def safe_console() -> None:
    """Windows consoles (cp1252) cannot print characters like '–'; replace them instead of crashing."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def main(argv: list[str] | None = None) -> int:
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--final", action="store_true", help="apply the freeze rules (counts, verification, R1–R7)")
    ap.add_argument("--allow-missing", action="store_true", help="exit 0 if the file does not exist (CI)")
    args = ap.parse_args(argv)

    if not args.path.exists():
        if args.allow_missing:
            print(f"SKIP: {args.path} does not exist yet")
            return 0
        print(f"ERROR: {args.path} not found", file=sys.stderr)
        return 2
    try:
        rep = validate_file(args.path, final=args.final)
    except (OSError, UnicodeDecodeError) as err:
        print(f"ERROR: cannot read {args.path}: {err}", file=sys.stderr)
        return 2

    print(summary(rep.items))
    for w in rep.warnings:
        print(f"WARNING: {w}")
    for e in rep.errors:
        print(f"ERROR: {e}")
    mode = "final" if args.final else "draft"
    if rep.ok:
        print(f"OK: gold set valid ({mode} rules)")
        return 0
    print(f"FAILED: {len(rep.errors)} problem(s) ({mode} rules)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
