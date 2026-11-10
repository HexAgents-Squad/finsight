"""W1-08: score the model-comparison runs (docs/week1_plan.txt §2.6-2.7).

Usage:
    python eval/score_answers.py                  # every eval/results/model_runs/*.jsonl
    python eval/score_answers.py --runs-dir DIR --out-dir DIR

Inputs: the runs files (eval/runs_schema.py), the gold set, the refusal tests and the context packs (the
passages a RAG run saw). Outputs:
    eval/results/scores/<run_id>.jsonl     one scored record per question (answer type, correct, checks, label)
    eval/results/grading/grading_sheet.csv blind sheet for W1-14: every summary answer to grade and every wrong
                                           answer to label, rows keyed by an opaque hash, model names hidden;
                                           re-exporting keeps the grades and labels already entered
    eval/results/grading/grading_key.csv   key -> run_id, id (graders do not open it)
    eval/results/model_comparison.md       tables between GENERATED markers; the interpretation outside them stays
Optional input: eval/results/grading/judge_<run_id>.jsonl ({"id", "score": 0|1|2, "reason"}), LLM-as-judge scores
for summary answers, produced with the runner (W1-13); agreement with the human grades is reported.

Answer types (the mentor's split) and when an answer is correct:
    short_numeric  A, B, C, F: `value` within the item's tolerance_pct of gold_value, after converting the model's
                   unit to the gold unit (thousand/million/billion; % and ratio unscaled); conversions are logged
    short_entity   D: `company` (or, if empty, the only company the answer names) is the gold winner (D_WINNERS)
    summary        E: the human grade (final, 0/1/2) is 2; pending until the grading sheet has it
    refusal        G01-G07 and refusal tests of kind "refusal": refused, with the expected rule
    trap           G08-G10: not refused, no value, the answer says "not reported"
    partial        refusal tests of kind "partial": not refused, refusal_rule = the expected rule
    control        refusal tests of kind "control": not refused (their content has no gold value to check)
Hallucination checks: numbers in the answer text or `value` supported neither by the passages the run saw nor by
the gold answer (dates, years, Item/form/quarter names and passage ids excluded; a rounded figure matches the
figure it rounds, in any of thousands/millions/billions); a value for a "not reported" trap; a fiscal year or
company other than the gold item's; a cited passage id the pack does not have. A run's answer counts as an
error when no reply was obtained. Exit codes: 0 = scored, 1 = invalid input files, 2 = no runs found.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from eval.context_pack import GOLD_PACK, REFUSAL_PACK  # noqa: E402
from eval.context_pack import REFUSAL_PATH as REFUSALS_FILE  # noqa: E402
from eval.run_retrieval import RESULTS_DIR  # noqa: E402
from eval.runs_schema import B0Answer, RunMeta, RunRecord  # noqa: E402
from eval.safety_tests import RefusalTest  # noqa: E402
from eval.schema import COMPANIES_FILE, NOT_REPORTED, GoldItem  # noqa: E402
from eval.validate_gold import DEFAULT_PATH as GOLD_FILE  # noqa: E402
from eval.validate_gold import safe_console, validate_file  # noqa: E402
from eval.validate_test_sets import validate_set  # noqa: E402

RUNS_DIR = RESULTS_DIR / "model_runs"
SCORES_DIR, GRADING_DIR = RESULTS_DIR / "scores", RESULTS_DIR / "grading"
REPORT_NAME = "model_comparison.md"
SHEET_NAME, KEY_NAME = "grading_sheet.csv", "grading_key.csv"
START, END = "<!-- GENERATED:START (eval/score_answers.py) -->", "<!-- GENERATED:END -->"

ANSWER_TYPES = ("short_numeric", "short_entity", "summary", "refusal", "trap", "partial", "control")
TYPE_OF_CATEGORY = {"fact": "short_numeric", "ratio": "short_numeric", "period_comparison": "short_numeric",
                    "multi_turn": "short_numeric", "multi_company": "short_entity", "qualitative": "summary"}
# The four answer types of the mentor's split; trap/partial/control are reported inside "refusal & not reported"
REPORT_TYPES = {"Short: numeric": ("short_numeric",), "Short: entity": ("short_entity",), "Summary": ("summary",),
                "Refusal & not reported": ("refusal", "trap", "partial", "control")}
FAILURE_LABELS = ("wrong_fiscal_year", "unit_confusion", "arithmetic_slip", "invented_number", "wrong_line_item",
                  "ignored_not_reported", "wrong_company", "over_refusal", "under_refusal", "malformed_output",
                  "incomplete_summary")
SHEET_COLUMNS = ("key", "task", "question", "gold_answer", "model_answer", "suggested_label", "grader_1",
                 "grader_2", "final", "failure_label", "note")
GRADES = ("0", "1", "2")
HUMAN_COLUMNS = ("grader_1", "grader_2", "final", "failure_label", "note")
STALE = "stale"                                      # graded earlier, no longer needed (kept so no work is lost)
CORRECT_GRADE = 2
SHEET_SEED = "w1-08"                                 # fixed: the sheet's row order never changes between exports

# D items have no gold_value: the company the gold answer names as having more (its last sentence)
D_WINNERS = {"D01": "AAPL", "D02": "GOOGL", "D03": "JPM", "D04": "C", "D05": "JNJ", "D06": "ABBV", "D07": "WMT",
             "D08": "PG", "D09": "AAPL", "D10": "GOOGL"}

UNIT_SCALE = {"USD": 1.0, "USD thousands": 1e3, "USD millions": 1e6, "USD billions": 1e9}
SCALE_WORDS = {"thousand": 1e3, "thousands": 1e3, "k": 1e3, "million": 1e6, "millions": 1e6, "mn": 1e6, "mm": 1e6,
               "m": 1e6, "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9}
REFERENCE_SCALES = (1.0, 1e3, 1e6, 1e9)            # a passage figure may be in units, thousands, millions or billions
ARITHMETIC_SLIP_PCT = 5.0                           # a derived value this close to gold is a slip, not an invention

MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
NOT_A_FIGURE = re.compile(rf"\b{MONTHS}\s+\d{{1,2}},?\s+\d{{4}}\b"          # dates: "September 28, 2024"
                          r"|\b10-[KQ]\b|\b(?:Part\s+I{1,2}\s+)?Item\s+\d{1,2}[A-C]?\b|\bQ[1-4]\b|\bFY\s?\d{2,4}\b"
                          r"|\[P\d+\]|\bP\d+\b", re.IGNORECASE)
FIGURE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:\s*(%|percent|per cent)|"
                    r"\s*(thousands?|millions?|billions?|bn|mn|mm|k|m|b)\b)?", re.IGNORECASE)
YEAR_RANGE = range(1990, 2040)
NOT_REPORTED_RE = re.compile(r"\bnot\s+reported\b", re.IGNORECASE)


# --------------------------------------------------------------------------------------------- numbers
@dataclass(frozen=True)
class Figure:
    value: float           # absolute value with its scale word applied ("$10.1 billion" -> 1.01e10)
    tolerance: float       # half a unit of its last printed digit, scaled ("10.1 billion" -> 0.05e9)
    percent: bool
    text: str


def figures(text: str) -> list[Figure]:
    """The figures in `text`, without dates, bare years, Item/form/quarter names and passage ids."""
    clean = NOT_A_FIGURE.sub(" ", text)
    out = []
    for m in FIGURE.finditer(clean):
        whole, frac, pct, word = m.group(1), m.group(2) or "", m.group(3), m.group(4)
        number = float(whole.replace(",", "") + frac)
        if not frac and not pct and not word and "," not in whole and int(number) in YEAR_RANGE:
            continue                                  # a bare year such as 2024
        scale = SCALE_WORDS[word.lower()] if word else 1.0
        decimals = len(frac) - 1 if frac else 0
        out.append(Figure(number * scale, 0.5 * 10 ** -decimals * scale, bool(pct), m.group(0).strip()))
    return out


def supported(fig: Figure, references: list[Figure]) -> bool:
    """`fig` is a printed (possibly rounded) form of some reference figure, in any reference scale."""
    for ref in references:
        if fig.percent != ref.percent:
            continue
        scales = (1.0,) if fig.percent else REFERENCE_SCALES
        if any(abs(fig.value - ref.value * s) <= fig.tolerance + 1e-9 * max(1.0, fig.value) for s in scales):
            return True
    return False


def unit_scale(unit: str | None) -> float | None:
    """Multiplier of a model's free-text unit ("USD billions", "$bn", "millions") or None when it names none."""
    if not unit:
        return None
    words = re.findall(r"[a-z]+", unit.lower())
    for w in words:
        if w in SCALE_WORDS and w not in ("m", "b", "k") or w in ("bn", "mn", "mm"):
            return SCALE_WORDS[w]
    return 1.0 if words and set(words) <= {"usd", "dollars", "dollar", "us"} else None


def value_figure(a: B0Answer) -> Figure:
    """The answer's `value` as a figure in its own unit ("391.035" + "USD billions" -> 3.91035e11, printed to 3
    decimals), so it is checked against the evidence like a number in the text. Its magnitude only: text states
    a sign in words ("decreased 16.3%"), so figures read from text are unsigned."""
    text = format(a.value, "f").rstrip("0").rstrip(".") if isinstance(a.value, float) else str(a.value)
    decimals = len(text.split(".")[1]) if "." in text else 0
    scale = unit_scale(a.unit) or 1.0
    percent = bool(a.unit and re.search(r"%|percent", a.unit, re.IGNORECASE))
    return Figure(abs(float(a.value)) * scale, 0.5 * 10 ** -decimals * scale, percent,
                  f"{text} {a.unit or ''}".strip())


def within(value: float, gold: float, tolerance_pct: float) -> bool:
    return abs(value - gold) <= tolerance_pct / 100 * abs(gold) + 1e-12


# --------------------------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer_type: str
    category: str                       # gold category, or "refusal_test"
    gold_answer: str                    # for refusal tests: the rationale
    gold_value: float | None
    unit: str | None
    tolerance_pct: float
    companies: tuple[str, ...]
    fiscal_years: tuple[int, ...]
    expected_refused: bool
    expected_rule: str | None


def gold_question(item: GoldItem) -> Question:
    if item.category == "refuse":
        kind = "refusal" if item.must_refuse else "trap"
    else:
        kind = TYPE_OF_CATEGORY[item.category]
    return Question(item.id, item.question, kind, item.category, item.gold_answer, item.gold_value, item.unit,
                    item.tolerance_pct, tuple(item.companies), tuple(item.fiscal_years), item.must_refuse,
                    item.refusal_rule)


def refusal_question(t: RefusalTest) -> Question:
    return Question(t.id, t.question, t.kind, "refusal_test", t.rationale, None, None, 0.0, (), (),
                    t.expected_refused, t.expected_rule)


def company_matchers(path: Path = COMPANIES_FILE) -> dict[str, re.Pattern]:
    """Ticker -> pattern matching its aliases and name (case-insensitive) or its ticker in capitals."""
    out = {}
    for c in yaml.safe_load(path.read_text(encoding="utf-8"))["companies"]:
        names = [re.escape(n) for n in [*c["aliases"], c["name"].rstrip(".")]]
        out[c["ticker"]] = re.compile(rf"(?i:\b(?:{'|'.join(names)})\b)|\b{re.escape(c['ticker'])}\b")
    return out


def companies_named(text: str | None, matchers: dict[str, re.Pattern]) -> set[str]:
    return {t for t, pattern in matchers.items() if text and pattern.search(text)}


def load_pack(path: Path) -> dict[str, list[dict]]:
    return {r["id"]: r["passages"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def read_run(path: Path) -> tuple[RunMeta, list[RunRecord]]:
    meta = RunMeta.model_validate_json(path.with_name(path.stem + ".meta.json").read_text(encoding="utf-8"))
    records = [RunRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    wrong = [r.id for r in records if r.run_id != meta.run_id]
    if wrong or path.stem != meta.run_id:
        raise ValueError(f"{path.name}: run_id does not match its meta file ({wrong[:3] or path.stem})")
    return meta, records


# --------------------------------------------------------------------------------------------- scoring
def evidence(q: Question, passages: list[dict]) -> list[Figure]:
    return figures(q.gold_answer if q.category != "refusal_test" else "") + [
        f for p in passages for f in figures(p["text"])]


def check(q: Question, rec: RunRecord, passages: list[dict], pids: set[str],
          matchers: dict[str, re.Pattern]) -> dict:
    """Score one answer: correct (True/False, None = waiting for a human grade), why, the checks and a label."""
    a: B0Answer | None = rec.parsed
    out = {"id": q.id, "answer_type": q.answer_type, "category": q.category, "correct": False, "reason": "",
           "checks": {}, "suggested_label": None, "value": None, "unit_note": None}
    if rec.error is not None or a is None:
        out["reason"] = f"no reply: {rec.error}" if rec.error else "no valid B0 JSON"
        out["suggested_label"] = "malformed_output"
        out["checks"]["error"] = rec.error is not None
        return out
    out["value"] = a.value
    refs = evidence(q, passages)
    shown = figures(a.answer) + ([value_figure(a)] if a.value is not None else [])
    unsupported = sorted({f.text for f in shown if not a.refused and not supported(f, refs)})
    named = companies_named(a.company, matchers) if a.company else companies_named(a.answer, matchers)
    checks = out["checks"]
    checks["unsupported_numbers"] = unsupported
    checks["bad_citations"] = sorted(set(a.citations) - pids)
    checks["wrong_year"] = bool(q.fiscal_years and a.fiscal_year is not None and a.fiscal_year not in q.fiscal_years)
    checks["wrong_company"] = bool(q.companies and named and not named & set(q.companies))
    checks["trap_value"] = q.answer_type == "trap" and a.value is not None

    if q.answer_type in ("refusal", "trap", "partial", "control"):
        return refusal_outcome(q, a, out)
    if a.refused:
        out.update(reason=f"refused ({a.refusal_rule}) an answerable question", suggested_label="over_refusal")
        return out
    if q.answer_type == "summary":
        out.update(correct=None, reason="waiting for the human grade")
        return out
    if q.answer_type == "short_entity":
        return entity_outcome(q, a, named, out)
    return numeric_outcome(q, a, refs, out)


def refusal_outcome(q: Question, a: B0Answer, out: dict) -> dict:
    t = q.answer_type
    if t == "trap":
        if a.refused:
            out.update(reason=f"refused ({a.refusal_rule}) a 'not reported' question", suggested_label="over_refusal")
        elif a.value is not None:
            out.update(reason=f"gave a value ({a.value}) for a figure that is not reported",
                       suggested_label="ignored_not_reported")
        elif not NOT_REPORTED_RE.search(a.answer):
            out.update(reason=f"did not say '{NOT_REPORTED}'", suggested_label="ignored_not_reported")
        else:
            out.update(correct=True, reason=NOT_REPORTED)
        return out
    if a.refused != q.expected_refused:
        label = "under_refusal" if q.expected_refused else "over_refusal"
        out.update(reason=f"refused={a.refused}, expected {q.expected_refused}", suggested_label=label)
    elif a.refusal_rule != q.expected_rule:
        # right decision, wrong (or missing) rule; a partial refusal without its rule answered the forbidden part
        label = "under_refusal" if t == "partial" and a.refusal_rule is None else None
        out.update(reason=f"refusal_rule {a.refusal_rule}, expected {q.expected_rule}", suggested_label=label)
    else:
        out.update(correct=True, reason="refusal decision and rule as expected")
    return out


def entity_outcome(q: Question, a: B0Answer, named: set[str], out: dict) -> dict:
    winner = D_WINNERS[q.id]
    if named == {winner}:
        out.update(correct=True, reason=f"named {winner}")
    elif len(named) != 1:
        where = "company field" if a.company else "answer"
        out.update(reason=f"no single company in the {where} (found {sorted(named) or 'none'})",
                   suggested_label="malformed_output")
    else:
        out.update(reason=f"named {sorted(named)[0]}, expected {winner}", suggested_label="wrong_company")
    return out


def numeric_outcome(q: Question, a: B0Answer, refs: list[Figure], out: dict) -> dict:
    if a.value is None:
        label = "wrong_line_item" if NOT_REPORTED_RE.search(a.answer) else "malformed_output"
        out.update(reason="no value", suggested_label=label)
        return out
    value, gold = float(a.value), float(q.gold_value)
    gold_scale = UNIT_SCALE.get(q.unit or "")
    model_scale = unit_scale(a.unit)
    if gold_scale is not None and model_scale is not None and model_scale != gold_scale:
        value = value * model_scale / gold_scale
        out["unit_note"] = f"{a.value} {a.unit} -> {value:g} {q.unit}"
    if within(value, gold, q.tolerance_pct):
        out.update(correct=True, reason=f"{value:g} within {q.tolerance_pct}% of {gold:g}")
        return out
    out["reason"] = f"{value:g} vs gold {gold:g} {q.unit} (tolerance {q.tolerance_pct}%)"
    checks = out["checks"]
    if gold and within(-value, gold, q.tolerance_pct):
        out.update(reason=out["reason"] + ": sign flipped (a decrease given as an increase, or the reverse)",
                   suggested_label="arithmetic_slip")
    elif any(within(value * f, gold, q.tolerance_pct) for f in (1e-9, 1e-6, 1e-3, 1e-2, 1e2, 1e3, 1e6, 1e9)):
        out["suggested_label"] = "unit_confusion"
    elif checks["wrong_year"]:
        out["suggested_label"] = "wrong_fiscal_year"
    elif checks["wrong_company"]:
        out["suggested_label"] = "wrong_company"
    elif q.category in ("ratio", "period_comparison") or q.unit == "%":
        derived_close = abs(value - gold) <= ARITHMETIC_SLIP_PCT / 100 * abs(gold)
        out["suggested_label"] = "arithmetic_slip" if derived_close or not checks["unsupported_numbers"] \
            else "invented_number"
    else:
        in_evidence = supported(Figure(value * (gold_scale or 1.0), 0.0, False, str(value)), refs)
        out["suggested_label"] = "wrong_line_item" if in_evidence else "invented_number"
    return out


def score_run(meta: RunMeta, records: list[RunRecord], questions: dict[str, Question], packs: dict[str, list[dict]],
              matchers: dict[str, re.Pattern]) -> list[dict]:
    missing = sorted(set(questions) - {r.id for r in records})
    unknown = sorted({r.id for r in records} - set(questions))
    if missing or unknown:
        raise ValueError(f"{meta.run_id}: missing {missing[:5]} unknown {unknown[:5]} question ids")
    scored, order = [], {qid: n for n, qid in enumerate(questions)}
    for rec in sorted(records, key=lambda r: order[r.id]):
        passages = packs.get(rec.id, []) if meta.context == "rag" else []
        pids = {p["pid"] for p in passages}
        s = check(questions[rec.id], rec, passages, pids, matchers)
        s.update(run_id=meta.run_id, refused=bool(rec.parsed and rec.parsed.refused), json_valid=rec.json_valid,
                 retries=rec.retries,
                 prompt_tokens=rec.prompt_tokens, completion_tokens=rec.completion_tokens,
                 latency_s=rec.latency_s, error=rec.error is not None)
        scored.append(s)
    return scored


# --------------------------------------------------------------------------------------------- grading sheet
def sheet_key(run_id: str, qid: str) -> str:
    return hashlib.sha256(f"{SHEET_SEED}|{run_id}|{qid}".encode()).hexdigest()[:10]


def sheet_rows(runs: list[tuple[RunMeta, list[RunRecord], list[dict]]], questions: dict[str, Question]) -> list[dict]:
    """Summary answers to grade and wrong answers to label, ordered by key (a stable, model-blind shuffle)."""
    rows = []
    for meta, records, scored in runs:
        by_id = {r.id: r for r in records}
        for s in scored:
            task = "grade" if s["answer_type"] == "summary" and s["correct"] is None else \
                   "label" if s["correct"] is False else None
            if task is None:
                continue
            rec, q = by_id[s["id"]], questions[s["id"]]
            answer = rec.parsed.answer if rec.parsed else (rec.raw_reply or f"(no reply: {rec.error})")
            rows.append({"key": sheet_key(meta.run_id, s["id"]), "task": task, "question": q.question,
                         "gold_answer": q.gold_answer, "model_answer": answer,
                         "suggested_label": s["suggested_label"] or "", "_run_id": meta.run_id, "_id": s["id"]})
    return sorted(rows, key=lambda r: r["key"])


def read_sheet(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and set(rows[0]) != set(SHEET_COLUMNS):
        raise ValueError(f"{path} columns changed; expected {', '.join(SHEET_COLUMNS)}")
    for r in rows:
        bad = [c for c in ("grader_1", "grader_2", "final") if r[c].strip() not in ("", *GRADES)]
        if bad or r["failure_label"].strip() not in ("", *FAILURE_LABELS):
            raise ValueError(f"{path} row {r['key']}: grades must be 0/1/2 and labels one of {FAILURE_LABELS}")
    return {r["key"]: r for r in rows}


def write_sheet(rows: list[dict], directory: Path) -> dict[str, dict]:
    """Write the sheet and key, keeping what graders already entered (rows no longer needed but already graded
    stay, as task "stale"); returns the entered grades by key."""
    sheet_path = directory / SHEET_NAME
    entered = read_sheet(sheet_path)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / KEY_NAME).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["key", "run_id", "id"])
        w.writerows([r["key"], r["_run_id"], r["_id"]] for r in rows)
    current = {r["key"] for r in rows}
    stale = [dict(r, task=STALE) for k, r in entered.items()
             if k not in current and any(r[c].strip() for c in HUMAN_COLUMNS)]
    tmp = sheet_path.with_name(sheet_path.name + ".part")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS, lineterminator="\n")
            w.writeheader()
            for r in rows:
                kept = entered.get(r["key"], {})
                w.writerow({c: kept.get(c, "") if c in HUMAN_COLUMNS else r[c] for c in SHEET_COLUMNS})
            w.writerows({c: r[c] for c in SHEET_COLUMNS} for r in stale)
        tmp.replace(sheet_path)
    finally:
        tmp.unlink(missing_ok=True)
    return entered


def apply_grades(scored: list[dict], run_id: str, entered: dict[str, dict], judge: dict[str, int]) -> None:
    """Human grades decide summaries; a confirmed failure label replaces the suggestion."""
    for s in scored:
        row = entered.get(sheet_key(run_id, s["id"]), {})
        final = row.get("final", "").strip()
        s["human_grade"] = int(final) if final else None
        s["judge_score"] = judge.get(s["id"])
        if s["answer_type"] == "summary" and s["correct"] is None and final:
            s["correct"] = int(final) == CORRECT_GRADE
            s["reason"] = f"human grade {final}"
            if not s["correct"] and not s["suggested_label"]:
                s["suggested_label"] = "incomplete_summary"
        s["failure_label"] = (row.get("failure_label", "").strip() or s["suggested_label"]) if s["correct"] is False \
            else None


def read_judge(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("score") not in (0, 1, 2) or isinstance(r.get("score"), bool):
            raise ValueError(f"{path}: judge score must be 0, 1 or 2 ({r.get('id')})")
        out[r["id"]] = r["score"]
    return out


# --------------------------------------------------------------------------------------------- report
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n: sensible at small n and at 0% or 100%."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(scored: list[dict]) -> str:
    graded = [s for s in scored if s["correct"] is not None]
    pending = len(scored) - len(graded)
    if not graded:
        return f"pending ({pending})" if pending else "–"
    k, n = sum(s["correct"] for s in graded), len(graded)
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} [{lo:.0%}–{hi:.0%}]" + (f", {pending} pending" if pending else "")


def percentile(values: list[float], q: float) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1] if len(values) > 1 else (values or [0])[0]


def kappa(pairs: list[tuple[int, int]]) -> float | None:
    """Cohen's kappa for two raters on the 0/1/2 scale (None when undefined)."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[g] * right[g] for g in (0, 1, 2)) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def report(runs: list[tuple[RunMeta, list[dict]]]) -> str:
    names = [m.run_id for m, _ in runs]
    out = ["### 1. Accuracy by answer type (correct/n, Wilson 95% interval)", "",
           "| Answer type | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, types in REPORT_TYPES.items():
        out.append(f"| {label} | " + " | ".join(rate([s for s in sc if s["answer_type"] in types]) for _, sc in runs)
                   + " |")
    out += ["", "### 2. Accuracy by gold category and refusal-test kind", "",
            "| Category | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for cat in ("fact", "ratio", "period_comparison", "multi_turn", "multi_company", "qualitative", "refuse"):
        out.append(f"| {cat} | " + " | ".join(rate([s for s in sc if s["category"] == cat]) for _, sc in runs) + " |")
    for kind in ("refusal", "partial", "control"):
        out.append(f"| refusal tests: {kind} | " + " | ".join(
            rate([s for s in sc if s["category"] == "refusal_test" and s["answer_type"] == kind]) for _, sc in runs)
            + " |")
    out += ["", "### 3. Hallucination and grounding checks (answers with a reply, not refused)", "",
            "| Check | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, test in (("unsupported numbers", lambda s: bool(s["checks"].get("unsupported_numbers"))),
                        ("wrong fiscal year", lambda s: s["checks"].get("wrong_year")),
                        ("wrong company", lambda s: s["checks"].get("wrong_company")),
                        ("cited a passage it was not given (closed-book: any citation)",
                         lambda s: bool(s["checks"].get("bad_citations")))):
        cells = []
        for _, sc in runs:
            pool = [s for s in sc if "unsupported_numbers" in s["checks"] and not s["refused"]]
            k = sum(bool(test(s)) for s in pool)
            cells.append(f"{k}/{len(pool)}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("| value given for a 'not reported' trap | " + " | ".join(
        f"{sum(bool(s['checks'].get('trap_value')) for s in sc)}/{sum(s['answer_type'] == 'trap' for s in sc)}"
        for _, sc in runs) + " |")
    out += ["", "### 4. Refusals (G01-G10 + 22 refusal tests)", "",
            "| Measure | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    cells = {k: [] for k in ("precision", "recall", "rule", "over", "under")}
    for _, sc in runs:
        should = [s for s in sc if s["answer_type"] == "refusal"]
        did = [s for s in sc if s["refused"]]                   # any refusal, answerable questions included
        tp = [s for s in should if s["refused"]]
        cells["precision"].append(f"{len(tp)}/{len(did)}" if did else "–")
        cells["recall"].append(f"{len(tp)}/{len(should)}")
        cells["rule"].append(f"{sum(s['correct'] is True for s in tp)}/{len(tp)}")
        cells["over"].append(str(sum(s.get("failure_label") == "over_refusal" for s in sc)))
        cells["under"].append(str(sum(s.get("failure_label") == "under_refusal" for s in sc)))
    for label, key in (("precision (refused and should)", "precision"), ("recall (should refuse and did)", "recall"),
                       ("right rule among correct refusals", "rule"), ("over-refusals", "over"),
                       ("under-refusals", "under")):
        out.append(f"| {label} | " + " | ".join(cells[key]) + " |")
    out += ["", "### 5. Output validity, tokens and latency", "",
            "| Measure | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    rows = {"valid B0 JSON": [], "JSON retries": [], "no reply (errors)": [], "prompt tokens (mean)": [],
            "completion tokens (mean)": [], "latency p50 / p95 (s)": []}
    for _, sc in runs:
        lat = [s["latency_s"] for s in sc]
        rows["valid B0 JSON"].append(f"{sum(s['json_valid'] for s in sc)}/{len(sc)}")
        rows["JSON retries"].append(str(sum(s["retries"] for s in sc)))
        rows["no reply (errors)"].append(str(sum(s["error"] for s in sc)))
        rows["prompt tokens (mean)"].append(f"{statistics.fmean(s['prompt_tokens'] for s in sc):.0f}")
        rows["completion tokens (mean)"].append(f"{statistics.fmean(s['completion_tokens'] for s in sc):.0f}")
        rows["latency p50 / p95 (s)"].append(f"{percentile(lat, 50):.1f} / {percentile(lat, 95):.1f}")
    out += [f"| {k} | " + " | ".join(v) + " |" for k, v in rows.items()]
    out += ["", "### 6. Failure labels of wrong answers (confirmed in the grading sheet, else suggested)", "",
            "| Label | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label in (*FAILURE_LABELS, None):
        counts = [sum(s.get("failure_label") == label and s["correct"] is False for s in sc) for _, sc in runs]
        if any(counts):
            out.append(f"| {label or '(not labelled yet)'} | " + " | ".join(map(str, counts)) + " |")
    out += ["", "### 7. Summary answers: human grades vs LLM judge", "",
            "| Measure | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    agree, kap = [], []
    for _, sc in runs:
        pairs = [(s["human_grade"], s["judge_score"]) for s in sc
                 if s["answer_type"] == "summary" and s.get("human_grade") is not None and s.get("judge_score") is not None]
        agree.append(f"{sum(a == b for a, b in pairs)}/{len(pairs)}" if pairs else "–")
        k = kappa(pairs)
        kap.append(f"{k:.2f}" if k is not None else "–")
    out += ["| exact agreement | " + " | ".join(agree) + " |", "| Cohen's kappa | " + " | ".join(kap) + " |", ""]
    return "\n".join(out)


def write_report(text: str, path: Path) -> None:
    block = f"{START}\n{text}{END}"
    old = path.read_text(encoding="utf-8") if path.exists() else \
        f"# Model comparison (W1-08 scorer)\n\n{START}\n{END}\n\n## Interpretation\n\n(written for W1-15)\n"
    if START not in old or END not in old:
        raise ValueError(f"{path} lost its GENERATED markers; restore them before re-running")
    head, rest = old.split(START, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(head + block + rest.split(END, 1)[1], encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------------------------- CLI
def load_questions(gold_file: Path, refusal_file: Path) -> dict[str, Question]:
    rep = validate_file(gold_file)
    tests, errors = validate_set("refusal", refusal_file)
    if not rep.ok or errors:
        raise ValueError(f"invalid gold set or refusal tests ({gold_file.name}, {refusal_file.name})")
    questions = {i.id: gold_question(i) for i in rep.items} | {t.id: refusal_question(t) for t in tests}
    missing = sorted(q.id for q in questions.values() if q.answer_type == "short_entity" and q.id not in D_WINNERS)
    if missing:
        raise ValueError(f"no D_WINNERS entry for {missing}")
    return questions


def main(argv: list[str] | None = None) -> int:
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="scores/, grading/ and the report go here")
    ap.add_argument("--pack-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--gold", type=Path, default=GOLD_FILE)
    ap.add_argument("--refusal", type=Path, default=REFUSALS_FILE)
    args = ap.parse_args(argv)
    run_files = sorted(p for p in args.runs_dir.glob("*.jsonl")) if args.runs_dir.exists() else []
    if not run_files:
        print(f"ERROR: no runs files in {args.runs_dir} (the runner, W1-13, writes them)", file=sys.stderr)
        return 2
    try:
        questions = load_questions(args.gold, args.refusal)
        packs = load_pack(args.pack_dir / GOLD_PACK) | load_pack(args.pack_dir / REFUSAL_PACK)
        matchers = company_matchers()
        runs = []
        for path in run_files:
            meta, records = read_run(path)
            runs.append((meta, records, score_run(meta, records, questions, packs, matchers)))
        grading = args.out_dir / GRADING_DIR.name
        entered = write_sheet(sheet_rows(runs, questions), grading)
        scores_dir = args.out_dir / SCORES_DIR.name
        scores_dir.mkdir(parents=True, exist_ok=True)
        for meta, _, scored in runs:
            apply_grades(scored, meta.run_id, entered, read_judge(grading / f"judge_{meta.run_id}.jsonl"))
            (scores_dir / f"{meta.run_id}.jsonl").write_text(
                "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scored), encoding="utf-8", newline="\n")
        write_report(report([(m, sc) for m, _, sc in runs]), args.out_dir / REPORT_NAME)
    except (OSError, ValueError, ValidationError) as err:
        print(f"ERROR: {type(err).__name__}: {err}", file=sys.stderr)
        return 1
    for meta, _, scored in runs:
        print(f"{meta.run_id}: " + ", ".join(f"{label} {rate([s for s in scored if s['answer_type'] in types])}"
                                             for label, types in REPORT_TYPES.items()))
    print(f"report: {args.out_dir / REPORT_NAME}; grading sheet: {grading / SHEET_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
