"""Schema and rules for the FinSight gold evaluation set (eval/gold_set.jsonl).

Single source of truth for the contracts in docs/week0_plan.txt §2.4 and §3 and docs/spec.md §3–§7 (units,
fiscal periods, refusal rules and Item names come from core/spec.py, shared with the tools and the agent).
Pydantic checks field types; the rule functions below check meaning and return ALL problems at once
(so a writer can fix every issue in one pass instead of one error at a time).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.spec import (  # noqa: F401  (re-exported: eval code and tests import these names from here)
    ALLOWED_UNITS,
    NOT_REPORTED,
    REFUSAL_RULES,
    TENK_ITEMS,
    TENQ_ITEMS,
    FiscalPeriod,
    RefusalRule,
)

ROOT = Path(__file__).resolve().parents[1]
COMPANIES_FILE = ROOT / "config" / "companies.yaml"

Category = Literal["fact", "ratio", "period_comparison", "multi_company", "qualitative", "multi_turn", "refuse"]
CATEGORY_LETTER: dict[str, str] = {"fact": "A", "ratio": "B", "period_comparison": "C", "multi_company": "D",
                                   "qualitative": "E", "multi_turn": "F", "refuse": "G"}
# docs/week0_plan.txt §3: target split of the 80 questions
TARGET_COUNTS: dict[str, int] = {"fact": 15, "ratio": 15, "period_comparison": 12, "multi_company": 10,
                                 "qualitative": 12, "multi_turn": 6, "refuse": 10}
TRAPS_MIN, TRAPS_MAX = 2, 3                      # "not reported" traps among the G items (week0 §3 rule 5)

GOLD_ID_PATTERN = r"^[A-G]\d{2}$"                # category letter + 2 digits, e.g. "B07"
# Development set (W1-06): the same format with a "dev-" prefix, so a dev id can never be read as a gold id
DEV_PREFIX = "dev-"
DEV_ID_PATTERN = r"^dev-[A-G]\d{2}$"
DEV_TARGET_COUNTS: dict[str, int] = {"fact": 3, "ratio": 2, "period_comparison": 2, "multi_company": 2,
                                     "qualitative": 2, "multi_turn": 2, "refuse": 2}
MIN_QUESTION_CHARS = 10                         # catches placeholders like "tbd" or "?"
DEFAULT_TOLERANCE_PCT = 1.0                     # week0 §2.4 example; a question can set its own

# ----------------------------------------------------------------------------------------------- scope
@dataclass(frozen=True)
class CorpusScope:
    tickers: frozenset[str]
    tenk_years: frozenset[int]
    tenq_years: frozenset[int]
    tenq_periods: frozenset[str]


@cache
def corpus_scope(path: Path = COMPANIES_FILE) -> CorpusScope:
    """In-scope companies, forms and periods from config/companies.yaml (schema v2)."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    forms = cfg["forms"]
    return CorpusScope(
        tickers=frozenset(c["ticker"] for c in cfg["companies"]),
        tenk_years=frozenset(forms["10-K"]["fiscal_years"]),
        tenq_years=frozenset(forms["10-Q"]["fiscal_years"]),
        tenq_periods=frozenset(forms["10-Q"]["fiscal_periods"]),
    )


# ----------------------------------------------------------------------------------------------- models
STRICT = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def not_blank(v: str) -> str:
    """Pydantic validator: min_length alone lets whitespace-only text through."""
    if not v.strip():
        raise ValueError("must not be blank")
    return v


class GoldSource(BaseModel):
    """Where the evidence for an answer lives: one Item of one filing (used for recall@k)."""
    model_config = STRICT

    ticker: str
    form: Literal["10-K", "10-Q"]
    fiscal_year: int
    fiscal_period: FiscalPeriod
    item: str

    def key(self) -> tuple[str, int, str, str]:
        """Matching key for retrieval metrics: (ticker, fiscal_year, fiscal_period, item)."""
        return (self.ticker, self.fiscal_year, self.fiscal_period, self.item)


class GoldItem(BaseModel):
    """One gold question (docs/week0_plan.txt §2.4)."""
    model_config = STRICT

    id: str = Field(pattern=GOLD_ID_PATTERN)
    category: Category
    question: str = Field(min_length=MIN_QUESTION_CHARS)
    companies: list[str] = []
    fiscal_years: list[int] = []
    gold_answer: str = Field(min_length=1)
    gold_value: float | None = None
    unit: str | None = None
    tolerance_pct: float = Field(default=DEFAULT_TOLERANCE_PCT, ge=0, le=100)
    gold_sources: list[GoldSource] = []
    must_refuse: bool = False
    refusal_rule: RefusalRule | None = None
    previous_turn_id: str | None = None
    written_by: str = Field(min_length=1)
    verified_by: str | None = None
    notes: str | None = None

    _not_blank = field_validator("question", "gold_answer", "written_by")(not_blank)

    @property
    def is_trap(self) -> bool:
        """A G item that is answerable in scope but whose figure is not reported."""
        return self.category == "refuse" and not self.must_refuse


class DevItem(GoldItem):
    """One development-set question (W1-06): the gold format and rules, with a "dev-" id."""
    id: str = Field(pattern=DEV_ID_PATTERN)


# ----------------------------------------------------------------------------------------------- rules
def _norm(name: str | None) -> str:
    return " ".join((name or "").split()).casefold()


def two_person_problems(written_by: str, verified_by: str | None) -> list[str]:
    """The verifier must be a different person (names compared ignoring case and spacing)."""
    if verified_by and _norm(verified_by) == _norm(written_by):
        return ["two-person rule: verified_by must be a different person than written_by"]
    return []


def source_problems(src: GoldSource, scope: CorpusScope) -> list[str]:
    where = f"source {src.ticker} {src.form} FY{src.fiscal_year} {src.fiscal_period} '{src.item}'"
    out = []
    if src.ticker not in scope.tickers:
        out.append(f"{where}: ticker not in config/companies.yaml")
    if src.form == "10-K":
        if src.fiscal_period != "FY":
            out.append(f"{where}: a 10-K source must have fiscal_period 'FY'")
        if src.fiscal_year not in scope.tenk_years:
            out.append(f"{where}: 10-K fiscal year outside {sorted(scope.tenk_years)}")
        if src.item not in TENK_ITEMS:
            out.append(f"{where}: not a valid 10-K Item name (expected e.g. 'Item 7', 'Item 1A')")
    else:
        if src.fiscal_period not in scope.tenq_periods:
            out.append(f"{where}: a 10-Q source must have fiscal_period in {sorted(scope.tenq_periods)}")
        if src.fiscal_year not in scope.tenq_years:
            out.append(f"{where}: 10-Q fiscal year outside {sorted(scope.tenq_years)}")
        if src.item not in TENQ_ITEMS:
            out.append(f"{where}: not a valid 10-Q Item name (expected e.g. 'Part I Item 2')")
    return out


def item_problems(item: GoldItem, scope: CorpusScope | None = None) -> list[str]:
    """Meaning-level rules for one gold item. Returns every problem found (empty list = valid)."""
    scope = scope or corpus_scope()
    out: list[str] = []
    letter = CATEGORY_LETTER[item.category]
    if item.id.removeprefix(DEV_PREFIX)[0] != letter:
        out.append(f"id '{item.id}' must start with '{letter}' for category '{item.category}'")

    # --- refusals, traps and partial refusals (spec §5, §6; week0 §2.4)
    if item.must_refuse:
        if item.category != "refuse":
            out.append("must_refuse is true, so the category must be 'refuse' (G)")
        if item.refusal_rule is None:
            out.append("must_refuse is true, so refusal_rule must be one of R1–R7")
        if item.gold_sources:
            out.append("refusal items must not have gold_sources")
        if item.gold_value is not None:
            out.append("refusal items must not have a gold_value")
    else:
        if not item.gold_sources:
            out.append("non-refused items need at least one gold_source (recall@k and citations depend on it)")
        if item.is_trap:
            if item.refusal_rule is not None:
                out.append("'not reported' trap (G, must_refuse false) must have refusal_rule null")
            if not _norm(item.gold_answer).startswith(NOT_REPORTED):
                out.append(f"'not reported' trap: gold_answer must start with '{NOT_REPORTED}'")
            if item.gold_value is not None:
                out.append("'not reported' trap must not have a gold_value")
        # refusal_rule with must_refuse false on A–F = partial refusal: allowed (spec §6)

    # --- numeric answers (spec §7)
    if item.category in ("ratio", "period_comparison") and item.gold_value is None:
        out.append(f"category '{item.category}' needs a numeric gold_value")
    if item.gold_value is not None:
        if not item.unit:
            out.append("gold_value is set, so unit is required")
        elif item.unit not in ALLOWED_UNITS:
            out.append(f"unit '{item.unit}' not allowed; use one of {sorted(ALLOWED_UNITS)}")
    elif item.unit:
        out.append("unit is set but gold_value is empty")

    # --- entities and periods
    if not item.must_refuse:
        if not item.companies:
            out.append("companies must list the ticker(s) the question is about")
        if not item.fiscal_years:
            out.append("fiscal_years must list the fiscal year(s) the question is about")
        unknown = [t for t in item.companies if t not in scope.tickers]
        if unknown:
            out.append(f"companies not in config/companies.yaml: {unknown}")
        if len(set(item.companies)) != len(item.companies):
            out.append("companies contains duplicates")
        corpus_years = scope.tenk_years | scope.tenq_years
        outside = [y for y in item.fiscal_years if y not in corpus_years]
        if outside:
            out.append(f"fiscal_years {outside} outside the corpus {sorted(corpus_years)} (that is an R3 refusal)")
        if len(set(item.fiscal_years)) != len(item.fiscal_years):
            out.append("fiscal_years contains duplicates")
        if item.gold_sources:
            sourced = {s.ticker for s in item.gold_sources}
            unsourced = [t for t in item.companies if t not in sourced]
            if unsourced:
                out.append(f"no gold_source for compan(ies) {unsourced}: every listed company needs one")
        for src in item.gold_sources:
            out.extend(source_problems(src, scope))
            if src.ticker not in item.companies:
                out.append(f"source ticker {src.ticker} is not listed in companies")
            if src.fiscal_year not in item.fiscal_years:
                out.append(f"source fiscal year {src.fiscal_year} is not listed in fiscal_years")
    if item.category == "multi_company" and len(set(item.companies)) < 2:
        out.append("multi_company questions need at least 2 companies")

    # --- follow-ups
    if item.category == "multi_turn" and not item.previous_turn_id:
        out.append("multi_turn questions need previous_turn_id")
    if item.category != "multi_turn" and item.previous_turn_id:
        out.append("only multi_turn (F) questions may set previous_turn_id")

    out.extend(two_person_problems(item.written_by, item.verified_by))
    return out


def item_warnings(item: GoldItem) -> list[str]:
    """Non-blocking hints."""
    out = []
    if item.category == "fact" and item.gold_value is None:
        out.append("fact question without gold_value: fine for text answers, but numeric facts should set it")
    return out


@dataclass
class SetReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def set_problems(items: list[GoldItem], final: bool = False) -> SetReport:
    """Rules across the whole set. `final=True` adds the freeze rules (week0 §3 / §8 WP8)."""
    rep = SetReport()
    ids = [i.id for i in items]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        rep.errors.append(f"duplicate id {dup}")
    questions: dict[str, str] = {}
    for i in items:
        if i.category == "multi_turn":                   # follow-ups may share text ("And in fiscal 2023?")
            continue
        key = _norm(i.question)
        if key in questions:
            rep.errors.append(f"{i.id}: same question text as {questions[key]}")
        questions.setdefault(key, i.id)

    by_id = {i.id: i for i in items}
    for i in items:
        if i.previous_turn_id is None:
            continue
        if i.previous_turn_id == i.id:
            rep.errors.append(f"{i.id}: previous_turn_id points to itself")
        elif i.previous_turn_id not in by_id:
            rep.errors.append(f"{i.id}: previous_turn_id {i.previous_turn_id} does not exist")
        else:                                            # detect cycles in follow-up chains
            seen, cur = {i.id}, by_id[i.previous_turn_id]
            while cur.previous_turn_id and cur.previous_turn_id in by_id:
                if cur.id in seen:
                    rep.errors.append(f"{i.id}: previous_turn_id chain has a cycle")
                    break
                seen.add(cur.id)
                cur = by_id[cur.previous_turn_id]

    if final:
        unverified = [i.id for i in items if not i.verified_by]
        if unverified:
            rep.errors.append(f"not verified by a second person: {unverified}")
        counts = {c: sum(1 for i in items if i.category == c) for c in TARGET_COUNTS}
        for cat, target in TARGET_COUNTS.items():
            if counts[cat] != target:
                rep.errors.append(f"category {CATEGORY_LETTER[cat]} {cat}: {counts[cat]} questions, target {target}")
        covered = {i.refusal_rule for i in items if i.must_refuse}
        missing = [r for r in REFUSAL_RULES if r not in covered]
        if missing:
            rep.errors.append(f"refusal rules without a G question: {missing}")
        traps = sum(1 for i in items if i.is_trap)
        if not TRAPS_MIN <= traps <= TRAPS_MAX:
            rep.errors.append(f"'not reported' traps: {traps}, expected {TRAPS_MIN}–{TRAPS_MAX}")
    return rep
