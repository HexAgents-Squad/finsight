"""Select reported figures from SEC XBRL company facts (WP3; used by WP6's get_financial_fact).

A company-facts file (data/xbrl/<TICKER>.json) holds, per concept and unit, one row per (filing, period):
{start?, end, val, accn, fy, fp, form, filed}. A 10-K also repeats the prior year's figures under its own fy
label, and later filings repeat (and sometimes restate) earlier periods, so a value is chosen in two steps:

1. The PERIOD comes from the company's own label (spec §4.1): among the rows of the filing labelled
   (fiscal_year, fiscal_period) with the right form and duration, the one with the LATEST end date.
   Annual: 10-K, fp FY, 330-400 days. Quarterly: 10-Q, fp Q1-Q3, 3M/6M/9M windows (3M is the default).
   Balance-sheet items are instants (no start date; duration None). Q4 is never derived (spec §4.2).
   If the concept is missing from its own filing, the dates come from a concept every filing has (net income,
   total assets) in that same labelled filing, so the period is still the company's label, never a calendar guess.
2. The VALUE is the most recently filed row for exactly that period that changed it, from any 10-K/10-Q or
   amendment (spec §7.7: restated or amended figures use the latest filed value; `restated` says so and
   `original_value` keeps the first). A later repeat that merely rounds the figure is not a restatement. The
   citation is the filing that carries the value: the original filing unless the value was restated.

Currency is scaled to the unit the statements use ("USD millions" for all 20 companies, checked in WP3);
per-share values stay "USD per share". A metric not reported is None, never a substitute (spec §7.5).
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from ingest import ROOT

CONCEPT_MAP_FILE = ROOT / "config" / "concept_map.yaml"
TAXONOMY = "us-gaap"
ANNUAL_FORMS = frozenset({"10-K", "10-K/A"})
QUARTERLY_FORMS = frozenset({"10-Q", "10-Q/A"})
ALL_FORMS = ANNUAL_FORMS | QUARTERLY_FORMS
QUARTERS = ("Q1", "Q2", "Q3")
# Duration windows in days (end - start), week0_plan WP3 step 4. The 6M/9M lower bounds are wider than the plan's
# 170/260: Costco's fiscal quarters are 12 weeks, so its year-to-date periods are 168 and 252 days (ADR-011).
DURATION_DAYS = {"12M": (330, 400), "3M": (80, 100), "6M": (160, 190), "9M": (245, 280)}
# Every filing of all 20 companies reports these, so they fix a fiscal period's dates (fiscal_period_dates)
PERIOD_ANCHOR_DURATION = "NetIncomeLoss"
PERIOD_ANCHOR_INSTANT = "Assets"
DEFAULT_DURATION = {"FY": "12M", "Q1": "3M", "Q2": "3M", "Q3": "3M"}          # spec §4.2 / week0 §2.3
YTD_DURATION = {"FY": "12M", "Q1": "3M", "Q2": "6M", "Q3": "9M"}
# The durations a fiscal period can have. Anything else would return a different period (FY + 3M = a standalone
# Q4, which spec §4.2 forbids; Q3 + 12M = the prior fiscal year), so it is an error.
VALID_DURATIONS = {"FY": frozenset({"12M"}), "Q1": frozenset({"3M"}), "Q2": frozenset({"3M", "6M"}),
                   "Q3": frozenset({"3M", "9M"})}
# A later filing that repeats a figure rounded (e.g. a summary table: 2,411,834 -> 2,412,000) has not restated it.
# Only roundings that keep a few significant digits count, so a real change is never mistaken for one.
ROUNDING_DIGITS = range(1, 10)              # rounded to 10^1 .. 10^9 of the reported unit
ROUNDING_MIN_SIGNIFICANT_DIGITS = 3
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
CURRENCY_UNIT = "USD millions"         # all 20 companies present their statements in millions (ADR-011)
CURRENCY_SCALE = 1_000_000
PER_SHARE_UNIT = "USD per share"
XBRL_UNITS = {"currency": "USD", "per_share": "USD/shares"}
OUTPUT_UNITS = {"currency": CURRENCY_UNIT, "per_share": PER_SHARE_UNIT}
PERIOD_TYPES = ("duration", "instant")
METRIC_KEYS = frozenset({"period", "kind", "concepts", "quarterly_ytd_only"})
CONCEPT_KEYS = frozenset({"concept", "label", "sectors", "unless_reported"})


@dataclass(frozen=True)
class Concept:
    name: str
    label: str                          # the line item as the statements call it ("total net revenue")
    # A fallback concept may mean the same thing only for some filers, so it can be limited to sectors, or to
    # periods where other concepts are absent (equity incl. minority interest = equity when there is none)
    sectors: tuple[str, ...] = ()
    unless_reported: tuple[str, ...] = ()


@dataclass(frozen=True)
class Metric:
    name: str
    period: str                         # "duration" (income / cash flow) or "instant" (balance sheet)
    kind: str                           # "currency" or "per_share"
    concepts: tuple[Concept, ...]       # tried in order; only concepts that mean the same thing
    # Cash-flow statements in 10-Qs are year-to-date (6M at Q2, 9M at Q3); only MSFT, AMZN and KO also report 3M
    quarterly_ytd_only: bool = False

    def quarterly_duration(self, fiscal_period: str) -> str:
        """The duration a 10-Q reports for this metric: 3M, or year-to-date for cash-flow metrics."""
        return YTD_DURATION[fiscal_period] if self.quarterly_ytd_only else DEFAULT_DURATION[fiscal_period]


@dataclass(frozen=True)
class Fact:
    ticker: str
    metric: str
    fiscal_year: int
    fiscal_period: str
    duration: str | None                # "12M"/"3M"/"6M"/"9M", None for balance-sheet instants
    value: float                        # in `unit`
    unit: str
    concept: str
    label: str
    period_start: str | None
    period_end: str
    # The filing to cite (week0 §2.3): the one that first reported the period (the indexed corpus filing), or,
    # when the value was restated, the later filing that carries the cited value
    accession_no: str
    form: str
    filed: str
    url: str | None                     # SEC folder of that filing (None if no CIK was given)
    restated: bool                      # a later filing changed the value (not just rounded it)
    original_value: float               # as first reported, in `unit` (what the original filing's text shows)
    original_accession_no: str


def load_concept_map(path: Path = CONCEPT_MAP_FILE) -> dict[str, Metric]:
    """The metrics in config/concept_map.yaml. Unknown keys and values are errors: a typo, or a comma inside an
    unquoted label in a {...} entry, would otherwise be dropped silently."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["metrics"]
    for name, spec in raw.items():
        _check_keys(f"metric {name}", spec, METRIC_KEYS)
        if spec["period"] not in PERIOD_TYPES or spec.get("kind", "currency") not in XBRL_UNITS:
            raise ValueError(f"metric {name}: period must be one of {PERIOD_TYPES}, kind one of {list(XBRL_UNITS)}")
        for c in spec["concepts"]:
            _check_keys(f"metric {name} concept {c.get('concept')}", c, CONCEPT_KEYS)
    return {name: Metric(name, spec["period"], spec.get("kind", "currency"),
                         tuple(Concept(c["concept"], c["label"], tuple(c.get("sectors", ())),
                                       tuple(c.get("unless_reported", ()))) for c in spec["concepts"]),
                         bool(spec.get("quarterly_ytd_only", False)))
            for name, spec in raw.items()}


def _check_keys(where: str, spec: dict, allowed: frozenset[str]) -> None:
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"config/concept_map.yaml {where}: unknown key(s) {sorted(unknown)} "
                         f"(quote labels that contain a comma)")


@cache
def _days(start: str, end: str) -> int:
    return (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days


def _rows(company_facts: dict, concept: str, kind: str) -> list[dict[str, Any]]:
    units = company_facts.get("facts", {}).get(TAXONOMY, {}).get(concept, {}).get("units", {})
    return units.get(XBRL_UNITS[kind], [])


def _in_window(row: dict, duration: str | None) -> bool:
    if duration is None:
        return "start" not in row
    if "start" not in row:
        return False
    low, high = DURATION_DAYS[duration]
    return low <= _days(row["start"], row["end"]) <= high


Period = tuple[str | None, str]          # (start or None for an instant, end)


def labelled_period(rows: list[dict], fiscal_year: int, fiscal_period: str, duration: str | None) -> Period | None:
    """Step 1: the period the filing labelled (fiscal_year, fiscal_period) reports, i.e. its row with the right
    form and duration and the LATEST end date (a 10-K/10-Q also repeats earlier periods under the same label)."""
    forms = ANNUAL_FORMS if fiscal_period == "FY" else QUARTERLY_FORMS
    labelled = [r for r in rows if r.get("fy") == fiscal_year and str(r.get("fp", "")).upper() == fiscal_period
                and r.get("form") in forms and _in_window(r, duration)]
    if not labelled:
        return None
    latest = max(labelled, key=lambda r: (r["end"], r["filed"]))
    return latest.get("start"), latest["end"]


def select_row(rows: list[dict], period: Period) -> tuple[dict, dict] | None:
    """Step 2: (row to use, first filed row) for exactly `period`, from any 10-K/10-Q or amendment. The row to use
    is the latest filed one that changed the value; if no later filing changed it (or only rounded it), the
    first filed row, so the citation points at the filing that reported the period."""
    start, end = period
    same = [r for r in rows if r["end"] == end and r.get("start") == start and r.get("form") in ALL_FORMS]
    if not same:
        return None
    first = min(same, key=lambda r: (r["filed"], r["accn"]))
    changed = [r for r in same if r["val"] != first["val"] and not is_rounding_of(r["val"], first["val"])]
    return (max(changed, key=lambda r: (r["filed"], r["accn"])) if changed else first), first


def is_rounding_of(value: float, exact: float) -> bool:
    """True if `value` is `exact` rounded to a power of ten (a summary table repeating 2,411,834 as 2,412,000)."""
    return value != exact and any(value == round(exact, -digits) for digits in ROUNDING_DIGITS
                                  if 10 ** (digits + ROUNDING_MIN_SIGNIFICANT_DIGITS - 1) <= abs(exact))


def filing_url(cik: str | None, accession_no: str) -> str | None:
    return f"{SEC_ARCHIVES}/{int(cik)}/{accession_no.replace('-', '')}/" if cik else None


def fiscal_period_dates(company_facts: dict, fiscal_year: int, fiscal_period: str,
                        duration: str | None) -> Period | None:
    """The dates of a fiscal period, from a concept every filing reports. Used when the concept asked for is
    missing from its own filing but reported for that period elsewhere (NVIDIA tagged its FY2023 capex
    differently in the FY2023 10-K; the FY2024 10-K repeats it under the mapped concept)."""
    anchor = PERIOD_ANCHOR_INSTANT if duration is None else PERIOD_ANCHOR_DURATION
    return labelled_period(_rows(company_facts, anchor, "currency"), fiscal_year, fiscal_period, duration)


def _reported_for(company_facts: dict, concept: str, kind: str, row: dict) -> bool:
    """True if `concept` has a non-zero value for exactly the period of `row` (same start and end)."""
    return any(r["end"] == row["end"] and r.get("start") == row.get("start") and r["val"] != 0
               for r in _rows(company_facts, concept, kind))


def find_fact(company_facts: dict, ticker: str, metric: Metric, fiscal_year: int, fiscal_period: str = "FY",
              duration: str | None = None, sector: str | None = None, cik: str | None = None) -> Fact | None:
    """The reported value of `metric`, or None if the company does not report it for that period.

    `duration` defaults to 12M for FY and 3M for quarters and must suit the period (VALID_DURATIONS, else
    ValueError); balance-sheet metrics are instants whatever is asked. Cash-flow metrics are reported
    year-to-date in 10-Qs (a 3M figure only by a few companies): when the 3M figure is None,
    `metric.quarterly_duration(fiscal_period)` gives the duration that exists (the answer then says it is
    year-to-date, spec §4.2). `sector` (config/companies.yaml) enables sector-specific concepts such as banks'
    "long-term debt"; `cik` is used for the filing URL."""
    fiscal_period = fiscal_period.upper()
    if fiscal_period not in VALID_DURATIONS:
        raise ValueError(f"fiscal_period must be one of {sorted(VALID_DURATIONS)}; Q4 is never derived")
    if metric.period == "instant":
        duration = None
    elif duration is None:
        duration = DEFAULT_DURATION[fiscal_period]
    elif duration not in VALID_DURATIONS[fiscal_period]:
        raise ValueError(f"duration {duration!r} is not valid for {fiscal_period}; use one of "
                         f"{sorted(VALID_DURATIONS[fiscal_period])} (Q4 is never derived)")
    scale = CURRENCY_SCALE if metric.kind == "currency" else 1
    anchor_period = fiscal_period_dates(company_facts, fiscal_year, fiscal_period, duration)
    candidates = [c for c in metric.concepts if not c.sectors or sector in c.sectors]
    # Pass 1: concepts the period's own filing used (so original_value is what that filing said). Pass 2: the
    # same period found through the anchor, for a concept the filing tagged differently (NVIDIA FY2023 capex).
    for use_anchor in (False, True):
        for concept in candidates:
            rows = _rows(company_facts, concept.name, metric.kind)
            period = anchor_period if use_anchor else labelled_period(rows, fiscal_year, fiscal_period, duration)
            found = select_row(rows, period) if period else None
            if found is None:
                continue
            row, first = found
            if any(_reported_for(company_facts, other, metric.kind, row) for other in concept.unless_reported):
                continue
            return Fact(ticker, metric.name, fiscal_year, fiscal_period, duration, row["val"] / scale,
                        OUTPUT_UNITS[metric.kind], concept.name, concept.label, row.get("start"), row["end"],
                        row["accn"], row["form"], row["filed"], filing_url(cik, row["accn"]),
                        row is not first, first["val"] / scale, first["accn"])
    return None


# ----------------------------------------------------------------------------------------------- coverage and checks


@dataclass(frozen=True)
class Gap:
    metric: str
    tickers: frozenset[str]
    periods: frozenset[str]            # fiscal periods the gap covers ("FY", "Q1"...); all when the map lists none
    reason: str


def load_gaps(path: Path = CONCEPT_MAP_FILE) -> list[Gap]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")).get("gaps", {})
    return [Gap(spec.get("metric", name), frozenset(spec["tickers"]),
                frozenset(spec.get("periods", ("FY", *QUARTERS))), spec["reason"]) for name, spec in raw.items()]


def gap_problems(missing: set[tuple[str, str, int, str]], gaps: list[Gap],
                 periods: list[tuple[int, str]]) -> list[str]:
    """Compare real coverage with the declared gaps, year by year. `missing` holds (metric, ticker, fiscal_year,
    fiscal_period) for every in-scope period without a value; `periods` lists the in-scope (fiscal_year,
    fiscal_period). Undeclared gaps and declared gaps that now have values are both problems."""
    declared = {(g.metric, t, y, p) for g in gaps for t in g.tickers for y, p in periods if p in g.periods}
    out = [f"undeclared gap: {m} not found for {t} FY{y} {p}" for m, t, y, p in sorted(missing - declared)]
    out += [f"stale gap entry: {m} for {t} FY{y} {p} is now reported" for m, t, y, p in sorted(declared - missing)]
    return out


# Metrics that are positive and accrue steadily through the year, so a fiscal year's value must be in proportion
# to the same year's quarterly values. A concept that holds a different line item (JNJ's in-process R&D tagged as
# R&D: 109 for the year vs about 3,600 a quarter) fails this.
PLAUSIBILITY_METRICS = frozenset({"revenue", "gross_profit", "rd_expense"})
DURATION_MONTHS = {"3M": 3, "6M": 6, "9M": 9, "12M": 12}
PLAUSIBLE_ANNUALISED_RATIO = (0.5, 2.0)      # annual value / (quarterly value scaled to 12 months)


def implausible(facts: list[Fact]) -> list[str]:
    """Annual values out of proportion with the same fiscal year's quarterly values (see PLAUSIBILITY_METRICS)."""
    annual = {(f.ticker, f.metric, f.fiscal_year): f for f in facts if f.fiscal_period == "FY"}
    low, high = PLAUSIBLE_ANNUALISED_RATIO
    out = []
    for f in facts:
        year = annual.get((f.ticker, f.metric, f.fiscal_year))
        if f.metric not in PLAUSIBILITY_METRICS or f.fiscal_period == "FY" or year is None or f.value <= 0:
            continue
        ratio = year.value / (f.value * DURATION_MONTHS["12M"] / DURATION_MONTHS[f.duration])
        if not low <= ratio <= high:
            out.append(f"implausible: {f.ticker} {f.metric} FY{f.fiscal_year} = {year.value:,.0f} but "
                       f"{f.fiscal_period} {f.duration} = {f.value:,.0f} ({year.concept} vs {f.concept})")
    return out


def display_forms(value: float, unit: str) -> list[str]:
    """How a value is printed in the statements: "391,035" (USD millions), "6.08" (per share); a negative value
    may be printed in parentheses, so only the magnitude is searched."""
    magnitude = Decimal(str(abs(value)))
    if unit == PER_SHARE_UNIT:
        return [f"{magnitude.quantize(Decimal('0.01'), ROUND_HALF_UP)}"]
    whole = magnitude.quantize(Decimal(1), ROUND_HALF_UP)     # statements round half up; round() would not
    forms = [f"{int(whole):,}"]
    if magnitude != whole:
        forms.append(f"{magnitude.quantize(Decimal('0.1'), ROUND_HALF_UP):,}")
    return forms


def value_in_text(value: float, unit: str, text: str) -> bool:
    """True if the value appears in `text` as a whole number (not inside a longer one, e.g. 1,391,035)."""
    return any(re.search(rf"(?<![\d,.]){re.escape(form)}(?![\d]|,\d)", text) for form in display_forms(value, unit))


# Spot check: words that a statement line carrying the metric contains ("Total net sales | 391,035"), so a number
# elsewhere in the filing that merely equals the value does not count as a match
STATEMENT_LINE_WORDS = {name: re.compile(words, re.IGNORECASE) for name, words in {
    "revenue": r"revenue|sales",
    "operating_income": r"operating (?:income|profit|loss|earnings)|(?:income|earnings) from operations",
    "net_income": r"net (?:income|earnings|loss)",
    "gross_profit": r"gross (?:profit|margin)",
    "rd_expense": r"research",
    "capex": r"property|productive assets|capital expenditure",
    "operating_cash_flow": r"operating activities|cash from operations",
    # PFE prints its diluted figures under a "diluted" heading, on lines named "... common shareholders"
    "eps_diluted": r"dilut|common shareholders",
    "total_assets": r"total assets",
    "total_liabilities": r"total liabilities",
    "stockholders_equity": r"equity",
    "cash": r"cash",
    "long_term_debt": r"debt|borrowings",
}.items()}


def value_on_statement_line(value: float, unit: str, metric: str, text: str) -> bool:
    """True if `value` appears on a line of `text` that carries the metric's words (STATEMENT_LINE_WORDS)."""
    words = STATEMENT_LINE_WORDS[metric]
    return any(words.search(line) and value_in_text(value, unit, line) for line in text.splitlines())
