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
2. The VALUE is the most recently filed row for exactly that period, from any 10-K/10-Q or amendment
   (spec §7.7: restated or amended figures use the latest filed value; `restated` says if it changed).

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
    # Cash-flow statements in 10-Qs are year-to-date only (6M at Q2, 9M at Q3): a 3-month figure is not reported
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
    accession_no: str
    form: str
    filed: str
    restated: bool                      # the latest filed value differs from the one first reported
    original_value: float               # as first reported, in `unit` (what the original filing's text shows)


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
    """Step 2: (latest filed row, first filed row) for exactly `period`, from any 10-K/10-Q or amendment."""
    start, end = period
    same = [r for r in rows if r["end"] == end and r.get("start") == start and r.get("form") in ALL_FORMS]
    if not same:
        return None
    return max(same, key=lambda r: (r["filed"], r["accn"])), min(same, key=lambda r: (r["filed"], r["accn"]))


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
              duration: str | None = None, sector: str | None = None) -> Fact | None:
    """The reported value of `metric`, or None if the company does not report it for that period. `duration`
    defaults to 12M for FY and 3M for quarters; balance-sheet metrics are instants whatever is asked. `sector`
    (config/companies.yaml) enables sector-specific concepts such as banks' "long-term debt"."""
    fiscal_period = fiscal_period.upper()
    if fiscal_period not in DEFAULT_DURATION:
        raise ValueError(f"fiscal_period must be one of {sorted(DEFAULT_DURATION)}; Q4 is never derived")
    if metric.period == "instant":
        duration = None
    elif duration is None:
        duration = DEFAULT_DURATION[fiscal_period]
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
                        row["accn"], row["form"], row["filed"], row["val"] != first["val"], first["val"] / scale)
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


def gap_problems(missing: set[tuple[str, str, str]], gaps: list[Gap]) -> list[str]:
    """Compare real coverage with the declared gaps. `missing` holds (metric, ticker, fiscal_period) for every
    in-scope period without a value. Undeclared gaps and declared gaps that now have values are both problems."""
    declared = {(g.metric, t, p) for g in gaps for t in g.tickers for p in g.periods}
    out = [f"undeclared gap: {m} not found for {t} {p}" for m, t, p in sorted(missing - declared)]
    out += [f"stale gap entry: {m} for {t} {p} is now reported" for m, t, p in sorted(declared - missing)]
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
