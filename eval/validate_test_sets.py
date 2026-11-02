"""Validate the safety test sets (eval/refusal_tests.jsonl and eval/injection_tests.jsonl).

Usage:
    python eval/validate_test_sets.py                  # both default files, drafting rules
    python eval/validate_test_sets.py --final          # also require every test verified by a second person
    python eval/validate_test_sets.py --refusal path.jsonl --injection other.jsonl

Exit codes: 0 = valid, 1 = problems found, 2 = a file is missing or unreadable.
Rules: eval/safety_tests.py.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError  # noqa: E402

from eval.safety_tests import (  # noqa: E402
    InjectionTest,
    RefusalTest,
    injection_problems,
    injection_set_problems,
    refusal_problems,
    refusal_set_problems,
)
from eval.validate_gold import format_validation_error, read_jsonl, safe_console  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
SETS = {
    "refusal": (RefusalTest, refusal_problems, refusal_set_problems, EVAL_DIR / "refusal_tests.jsonl"),
    "injection": (InjectionTest, injection_problems, injection_set_problems, EVAL_DIR / "injection_tests.jsonl"),
}


def validate_set(name: str, path: Path, final: bool = False) -> tuple[list, list[str]]:
    """Return (parsed tests, errors) for one file. Errors are labelled 'line N (ID): ...'."""
    model, item_rules, set_rules, _ = SETS[name]
    records, errors = read_jsonl(path)
    tests = []
    for line_no, rec in records:
        label = f"line {line_no}" + (f" ({rec['id']})" if rec.get("id") else "")
        try:
            t = model.model_validate(rec)
        except ValidationError as err:
            errors.append(f"{label}: {format_validation_error(err)}")
            continue
        tests.append(t)
        errors += [f"{label}: {p}" for p in item_rules(t)]
    return tests, errors + set_rules(tests, final=final)


def summary(name: str, tests: list) -> str:
    if name == "refusal":
        kinds = Counter(t.kind for t in tests)
        rules = Counter(t.expected_rule for t in tests if t.kind == "refusal")
        return (f"refusal: {len(tests)} tests | refusals {kinds['refusal']}, partial {kinds['partial']}, "
                f"controls {kinds['control']} | rules {dict(sorted(rules.items()))}")
    types = Counter(t.injection_type for t in tests)
    return f"injection: {len(tests)} tests | types {dict(sorted(types.items()))}"


def main(argv: list[str] | None = None) -> int:
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refusal", type=Path, default=SETS["refusal"][3])
    ap.add_argument("--injection", type=Path, default=SETS["injection"][3])
    ap.add_argument("--final", action="store_true", help="also require every test verified by a second person")
    args = ap.parse_args(argv)

    status = 0
    for name, path in (("refusal", args.refusal), ("injection", args.injection)):
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            status = 2
            continue
        try:
            tests, errors = validate_set(name, path, final=args.final)
        except (OSError, UnicodeDecodeError) as err:
            print(f"ERROR: cannot read {path}: {err}", file=sys.stderr)
            status = 2
            continue
        print(summary(name, tests))
        for e in errors:
            print(f"ERROR: {name}: {e}")
        if errors:
            status = max(status, 1)
    mode = "final" if args.final else "draft"
    print(f"OK: test sets valid ({mode} rules)" if status == 0 else f"FAILED ({mode} rules)")
    return status


if __name__ == "__main__":
    sys.exit(main())
