"""Validate the development set (eval/dev_set.jsonl, W1-06) and show that it does not overlap gold-v1.

Usage:
    python eval/validate_dev_set.py              # format + overlap (CI)
    python eval/validate_dev_set.py --final      # also the split and a second person's verification on every item
    python eval/validate_dev_set.py eval/dev_set_v2.jsonl [--final]    # the development set v2 (W3-02)

The development set is where the prompting baselines S1-S3 (W1-09) are tuned, so it must teach nothing about
the gold questions (CLAUDE.md: prompts are never tuned on gold-v1). Rules:
- Format: every gold rule (eval/schema.py item_problems / set_problems) with "dev-" ids (DevItem).
- No shared filing period: no dev item cites a (company, fiscal year, period) that a gold item cites, and no dev
  item asks about a company and fiscal year that a gold item asks about (by its companies x fiscal_years fields;
  refusal items leave them empty). This keeps facts apart, not filings: a later gold-cited 10-K still shows a dev
  figure in its prior-year column.
- No near-duplicate wording: each dev question's word overlap (Jaccard over lower-case word sets) with every gold
  question and every refusal / injection test question stays below NEAR_DUPLICATE (0.8, just above the 0.79 of
  gold's own closest pair).
- At least one item (an emptied file must not pass).
- --final: the split DEV_TARGET_COUNTS and verified_by (a second person) on every item.
The development set v2 (W3-02, eval/dev_set_v2.jsonl: questions for each failure class A1 showed, to tune the A5
fixes on) passes the same rules with its own split (DEV2_TARGET_COUNTS) and ids numbered from 21, and is also kept
apart from v1: no id of v1 and no question near a v1 question. v1 stays as it is, so the W1-09 tuning runs and the
planner's development runs on it remain reproducible.
Limit: the word check catches a reworded gold question, not the same idea in other words (a refusal on the same
template, e.g. a forecast question about another company, scores low).
Exit codes: 0 = valid, 1 = problems found, 2 = a file is missing or unreadable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.schema import CATEGORY_LETTER, DEV_TARGET_COUNTS, DevItem, GoldItem  # noqa: E402
from eval.validate_gold import DEFAULT_PATH as GOLD_FILE  # noqa: E402
from eval.validate_gold import safe_console, validate_file  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DEV_FILE = EVAL_DIR / "dev_set.jsonl"
DEV2_FILE = EVAL_DIR / "dev_set_v2.jsonl"           # W3-02
# v2's split: line items outside the facts tool, a reported ratio and partial refusals (A); ratios and period changes
# on them (B, C); 'what drove' summaries (E, at least 8); a margin follow-up (F); rule edge cases and traps (G)
DEV2_TARGET_COUNTS: dict[str, int] = {"fact": 8, "ratio": 3, "period_comparison": 2, "multi_company": 1,
                                      "qualitative": 8, "multi_turn": 1, "refuse": 7}
DEV2_FIRST_NUMBER = 21                               # v2 ids run from dev-A21, so they never collide with v1's
OTHER_QUESTION_FILES = (EVAL_DIR / "refusal_tests.jsonl", EVAL_DIR / "injection_tests.jsonl")
# Word-set overlap at which a dev question counts as a gold question reworded. Measured on gold-v1 itself: two
# different gold questions built on one template ("By what percentage did Alphabet's revenues / AbbVie's net
# revenues grow from fiscal 2024 to fiscal 2025?", C03/C06) reach 0.79, so the limit sits just above: a shared
# template is fine (the dev set should look like the gold set, and the period rule above already keeps the facts
# apart); a question closer than any two gold questions is not.
NEAR_DUPLICATE = 0.8
WORD = re.compile(r"[a-z0-9&']+")
CURLY_APOSTROPHE = "’"             # pasted text: "Costco’s" must tokenize like "Costco's"


def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower().replace(CURLY_APOSTROPHE, "'")))


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of the two questions' word sets (1.0 = the same words)."""
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def gold_periods(gold: list[GoldItem]) -> tuple[set[tuple[str, int, str]], set[tuple[str, int]]]:
    """The filing periods gold cites (ticker, fiscal_year, fiscal_period) and the company-years it asks about."""
    cited = {(s.ticker, s.fiscal_year, s.fiscal_period) for g in gold for s in g.gold_sources}
    asked = {(t, y) for g in gold for t in g.companies for y in g.fiscal_years}
    return cited, asked


def overlap_problems(dev: list[GoldItem], gold: list[GoldItem], other_questions: list[str]) -> tuple[list[str], float]:
    """Every overlap of the dev items with gold-v1 (shared filing periods, near-duplicate questions), and the
    highest question similarity found (for the report)."""
    cited, asked = gold_periods(gold)
    problems, highest = [], 0.0
    references = [(g.id, g.question) for g in gold] + [(f"test {n}", q) for n, q in enumerate(other_questions, 1)]
    for d in dev:
        for s in d.gold_sources:
            if (s.ticker, s.fiscal_year, s.fiscal_period) in cited:
                problems.append(f"{d.id}: cites {s.ticker} {s.form} {s.fiscal_year} {s.fiscal_period}, a filing period "
                                f"gold-v1 cites")
        shared = sorted({(t, y) for t in d.companies for y in d.fiscal_years} & asked)
        if shared:
            problems.append(f"{d.id}: asks about {', '.join(f'{t} fiscal {y}' for t, y in shared)}, as gold-v1 does")
        if d.category == "multi_turn":          # a follow-up's text ("And in the third quarter?") is context-free
            continue
        for ref, question in references:
            score = similarity(d.question, question)
            highest = max(highest, score)
            if score >= NEAR_DUPLICATE:
                problems.append(f"{d.id}: question {score:.2f} similar to {ref}: {question!r}")
    return problems, highest


def other_test_questions(paths: tuple[Path, ...] = OTHER_QUESTION_FILES) -> list[str]:
    out = []
    for path in paths:
        if path.exists():
            out += [json.loads(line)["question"] for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    return out


def v1_problems(dev: list[GoldItem], v1: list[GoldItem]) -> list[str]:
    """v2 against v1: ids numbered from DEV2_FIRST_NUMBER (so none is a v1 id) and no question near a v1 question."""
    out = []
    v1_ids = {d.id for d in v1}
    for d in dev:
        if d.id in v1_ids or int(d.id[-2:]) < DEV2_FIRST_NUMBER:
            out.append(f"{d.id}: v2 ids are numbered from {DEV2_FIRST_NUMBER} (dev-A{DEV2_FIRST_NUMBER}), apart from v1's")
        if d.category == "multi_turn":
            continue
        for old in v1:
            score = similarity(d.question, old.question)
            if score >= NEAR_DUPLICATE:
                out.append(f"{d.id}: question {score:.2f} similar to v1's {old.id}: {old.question!r}")
    return out


def targets_for(path: Path) -> dict[str, int]:
    """The split a dev file must have under --final: v2's for eval/dev_set_v2.jsonl, else v1's."""
    return DEV2_TARGET_COUNTS if path.name == DEV2_FILE.name else DEV_TARGET_COUNTS


def final_problems(dev: list[GoldItem], targets: dict[str, int] = DEV_TARGET_COUNTS) -> list[str]:
    out = []
    counts = Counter(d.category for d in dev)
    for cat, target in targets.items():
        if counts.get(cat, 0) != target:
            out.append(f"category {CATEGORY_LETTER[cat]} {cat}: {counts.get(cat, 0)} items, target {target}")
    unverified = [d.id for d in dev if not d.verified_by]
    if unverified:
        out.append(f"not verified by a second person: {unverified}")
    return out


def main(argv: list[str] | None = None) -> int:
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", type=Path, default=DEV_FILE)
    ap.add_argument("--gold", type=Path, default=GOLD_FILE)
    ap.add_argument("--final", action="store_true", help="also require the split and second-person verification")
    args = ap.parse_args(argv)
    for path in (args.path, args.gold):
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 2
    try:
        dev_rep = validate_file(args.path, model=DevItem)
        gold_rep = validate_file(args.gold)
        others = other_test_questions()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as err:
        print(f"ERROR: cannot read the sets: {err}", file=sys.stderr)
        return 2
    if not gold_rep.ok:
        print(f"ERROR: {args.gold} is invalid; validate it first (eval/validate_gold.py)", file=sys.stderr)
        return 1

    errors = list(dev_rep.errors) + ([] if dev_rep.items else [f"{args.path} has no items"])
    overlap, highest = overlap_problems(dev_rep.items, gold_rep.items, others)
    errors += overlap
    targets = targets_for(args.path)
    if targets is DEV2_TARGET_COUNTS:                    # v2: also apart from v1
        v1_rep = validate_file(DEV_FILE, model=DevItem)
        errors += v1_problems(dev_rep.items, v1_rep.items)
    if args.final:
        errors += final_problems(dev_rep.items, targets)
    counts = Counter(d.category for d in dev_rep.items)
    verified = sum(1 for d in dev_rep.items if d.verified_by)
    print(f"{len(dev_rep.items)} dev items | " + ", ".join(
        f"{CATEGORY_LETTER[c]} {counts.get(c, 0)}/{t}" for c, t in targets.items())
        + f" | verified {verified}/{len(dev_rep.items)}")
    print(f"overlap with gold-v1: {len(overlap)} problem(s); highest question similarity {highest:.2f} "
          f"(limit {NEAR_DUPLICATE}) against {len(gold_rep.items)} gold questions and {len(others)} test questions")
    for w in dev_rep.warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    mode = "final" if args.final else "draft"
    if errors:
        print(f"FAILED: {len(errors)} problem(s) ({mode} rules)")
        return 1
    print(f"OK: development set valid and separate from gold-v1 ({mode} rules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
