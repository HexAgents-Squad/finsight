"""Select reported figures from SEC XBRL company facts (WP3; used by WP6's get_financial_fact).

A company-facts file (data/xbrl/<TICKER>.json) holds, per concept and unit, one row per (filing, period):
{start?, end, val, accn, fy, fp, form, filed}. A 10-K also repeats the prior year's figures under its own fy
label, and later filings repeat (and sometimes restate) earlier periods, so a value is chosen in two steps:

1. The PERIOD comes from the company's own label (spec §4.1): among the rows of the filing labelled
   (fiscal_year, fiscal_period) with the right form and duration, the one with the LATEST end date.
   Annual: 10-K, fp FY, 330-400 days. Quarterly: 10-Q, fp Q1-Q3, 3M/6M/9M windows (3M is the default).
   Balance-sheet items are instants (no start date; duration None). Q4 is never derived (spec §4.2).
   The dates must agree with a concept every filing has (net income, total assets) in that labelled filing.
   If the concept is missing from its own filing, those anchor dates are used, so the period is still the
   company's label, never a calendar guess.
2. The VALUE is the latest filed for exactly that period, from any 10-K/10-Q or amendment (spec §7.7:
   restated or amended figures use the latest filed value; `restated` says so and `original_value` keeps the
   first). A later repeat that merely rounds the figure is not a restatement. The citation is the earliest
   filing that carries the final value (the original filing unless the value was restated);
   `period_accession_no` is the filing labelled with the period itself (the indexed corpus filing).

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
FISCAL_PERIODS = ("FY", *QUARTERS)
# Duration windows in days (end - start), week0_plan WP3 step 4. The 6M/9M lower bounds are wider than the plan's
# 170/260: Costco's fiscal quarters are 12 weeks, so its year-to-date periods are 168 and 252 days (ADR-011).
DURATION_DAYS = {"12M": (330, 400), "3M": (80, 100), "6M": (160, 190), "9M": (245, 280)}
# Every filing of all 20 companies reports these, so they fix a fiscal period's dates (fiscal_period_row)
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
ROUNDING_DIGITS = range(1, 13)         # rounded to 10^1 .. 10^12 of the reported unit (bank totals are ~10^12)
ROUNDING_MIN_SIGNIFICANT_DIGITS = 3
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
CURRENCY_UNIT = "USD millions"         # all 20 companies present their statements in millions (ADR-011)
CURRENCY_SCALE = 1_000_000
PER_SHARE_UNIT = "USD per share"
XBRL_UNITS = {"currency": "USD", "per_share": "USD/shares"}
OUTPUT_UNITS = {"currency": CURRENCY_UNIT, "per_share": PER_SHARE_UNIT}
PERIOD_TYPES = ("duration", "instant")
TOP_KEYS = frozenset({"schema_version", "metrics", "gaps", "notes"})
METRIC_KEYS = frozenset({"period", "kind", "concepts", "quarterly_ytd_only", "ticker_labels"})
CONCEPT_KEYS = frozenset({"concept", "label", "sectors", "tickers", "unless_reported"})
GAP_KEYS = frozenset({"metric", "tickers", "periods", "reason"})
NOTE_KEYS = frozenset({"metric", "tickers", "note"})


@dataclass(frozen=True)
class Concept:
    name: str
    label: str                          # the line item as the statements call it ("total net revenue")
    # A fallback concept may mean the same thing only for some filers, so it can be limited to sectors or
    # tickers, or to periods where other concepts are absent (equity incl. minority interest = equity when
    # there is none)
    sectors: tuple[str, ...] = ()
    tickers: tuple[str, ...] = ()
    unless_reported: tuple[str, ...] = ()

    def applies_to(self, ticker: str, sector: str | None) -> bool:
        return (not self.sectors or sector in self.sectors) and (not self.tickers or ticker in self.tickers)


@dataclass(frozen=True)
class Metric:
    name: str
    period: str                         # "duration" (income / cash flow) or "instant" (balance sheet)
    kind: str                           # "currency" or "per_share"
    concepts: tuple[Concept, ...]       # tried in order; only concepts that mean the same thing
    # Cash-flow statements in 10-Qs are year-to-date (6M at Q2, 9M at Q3); only MSFT and AMZN also print 3M
    quarterly_ytd_only: bool = False
    # The company's own name for the line where it differs from the concept's label ("operating earnings")
    ticker_labels: tuple[tuple[str, str], ...] = ()

    def quarterly_duration(self, fiscal_period: str) -> str:
        """The duration every 10-Q reports for this metric: 3M, or year-to-date for cash-flow metrics."""
        return YTD_DURATION[fiscal_period] if self.quarterly_ytd_only else DEFAULT_DURATION[fiscal_period]

    def label_for(self, ticker: str, concept: Concept) -> str:
        return dict(self.ticker_labels).get(ticker, concept.label)


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
    # The filing to cite (week0 §2.3): the earliest one that carries the value, i.e. the filing that first
    # reported it, or, when it was restated, the first filing with the restated value
    accession_no: str
    form: str
    filed: str
    url: str | None                     # SEC folder of that filing (None if no CIK was given)
    restated: bool                      # a later filing changed the value (not just rounded it)
    original_value: float               # as first reported under the concept, in `unit`
    original_accession_no: str
    # The filing labelled with this fiscal period (the corpus filing). It differs from original_accession_no
    # when that filing showed the figure under another tag or in a note, and a later filing repeated it
    # under the mapped concept (NVIDIA FY2023 capex, KO FY2025 quarterly capex)
    period_accession_no: str


def _fail(where: str, problem: str) -> ValueError:
    return ValueError(f"config/concept_map.yaml {where}: {problem}")


def _check_keys(where: str, spec: Any, allowed: frozenset[str], required: frozenset[str] = frozenset()) -> None:
    if not isinstance(spec, dict):
        raise _fail(where, "must be a mapping")
    if unknown := set(spec) - allowed:
        raise _fail(where, f"unknown key(s) {sorted(unknown)} (quote labels that contain a comma)")
    if missing := required - set(spec):
        raise _fail(where, f"missing key(s) {sorted(missing)}")


def _text(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(where, f"must be a non-empty string, not {value!r}")
    return value.strip()


def _texts(where: str, value: Any) -> tuple[str, ...]:
    """A YAML list of strings; a bare string would otherwise be split into its letters."""
    if not isinstance(value, list) or not value:
        raise _fail(where, f"must be a non-empty list, not {value!r}")
    return tuple(_text(where, v) for v in value)


def _raw_map(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _check_keys("top level", raw, TOP_KEYS, frozenset({"metrics"}))
    return raw


def load_concept_map(path: Path = CONCEPT_MAP_FILE) -> dict[str, Metric]:
    """The metrics in config/concept_map.yaml. Unknown keys, missing keys and wrong types are errors: a typo,
    or a comma inside an unquoted label in a {...} entry, would otherwise be dropped silently."""
    metrics = {}
    for name, spec in _raw_map(path)["metrics"].items():
        where = f"metric {name}"
        _check_keys(where, spec, METRIC_KEYS, frozenset({"period", "concepts"}))
        kind = spec.get("kind", "currency")
        if spec["period"] not in PERIOD_TYPES or kind not in XBRL_UNITS:
            raise _fail(where, f"period must be one of {PERIOD_TYPES}, kind one of {list(XBRL_UNITS)}")
        if not isinstance(spec["concepts"], list) or not spec["concepts"]:
            raise _fail(where, "concepts must be a non-empty list")
        concepts = []
        for c in spec["concepts"]:
            cwhere = f"{where} concept {c.get('concept') if isinstance(c, dict) else c!r}"
            _check_keys(cwhere, c, CONCEPT_KEYS, frozenset({"concept", "label"}))
            concepts.append(Concept(
                _text(cwhere, c["concept"]), _text(cwhere, c["label"]),
                *(_texts(f"{cwhere} {key}", c[key]) if key in c else ()
                  for key in ("sectors", "tickers", "unless_reported"))))
        labels = spec.get("ticker_labels", {})
        if not isinstance(labels, dict):
            raise _fail(where, "ticker_labels must be a mapping of ticker to label")
        metrics[name] = Metric(name, spec["period"], kind, tuple(concepts),
                               bool(spec.get("quarterly_ytd_only", False)),
                               tuple((_text(where, t), _text(f"{where} {t}", label)) for t, label in labels.items()))
    return metrics


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


def _period(row: dict) -> Period:
    return row.get("start"), row["end"]


def labelled_row(rows: list[dict], fiscal_year: int, fiscal_period: str, duration: str | None) -> dict | None:
    """Step 1: the row of the filing labelled (fiscal_year, fiscal_period) with the right form and duration and
    the LATEST end date (a 10-K/10-Q also repeats earlier periods under the same label)."""
    forms = ANNUAL_FORMS if fiscal_period == "FY" else QUARTERLY_FORMS
    labelled = [r for r in rows if r.get("fy") == fiscal_year and str(r.get("fp", "")).upper() == fiscal_period
                and r.get("form") in forms and _in_window(r, duration)]
    return max(labelled, key=lambda r: (r["end"], r["filed"])) if labelled else None


def select_row(rows: list[dict], period: Period) -> tuple[dict, dict] | None:
    """Step 2: (row to cite, first filed row) for exactly `period`, from any 10-K/10-Q or amendment.

    Rows are taken in filing order; a row changes the value unless it repeats the current one or merely rounds
    it. The row to cite is the earliest filing that carries the final value (the first row if the value ends
    where it started), so the citation is the earliest, usually indexed, filing with that value."""
    same = sorted((r for r in rows if _period(r) == period and r.get("form") in ALL_FORMS),
                  key=lambda r: (r["filed"], r["accn"]))
    if not same:
        return None
    first = current = same[0]
    for r in same[1:]:
        if r["val"] != current["val"] and not is_rounding_of(r["val"], current["val"]):
            current = r
    return (first if current["val"] == first["val"] else current), first


def is_rounding_of(value: float, exact: float) -> bool:
    """True if `value` is `exact` rounded to a power of ten (a summary table repeating 2,411,834 as 2,412,000)."""
    return value != exact and any(value == round(exact, -digits) for digits in ROUNDING_DIGITS
                                  if 10 ** (digits + ROUNDING_MIN_SIGNIFICANT_DIGITS - 1) <= abs(exact))


def filing_url(cik: str | None, accession_no: str) -> str | None:
    return f"{SEC_ARCHIVES}/{int(cik)}/{accession_no.replace('-', '')}/" if cik else None


def fiscal_period_row(company_facts: dict, fiscal_year: int, fiscal_period: str, duration: str | None) -> dict | None:
    """The labelled row of a concept every filing reports (net income, total assets): it fixes the period's dates
    and its own filing. Used to check the dates a concept gives, and when the concept asked for is missing from
    its own filing but reported for that period elsewhere (NVIDIA tagged its FY2023 capex with its own tag in
    the FY2023 10-K; the FY2024 10-K repeats it under the mapped concept)."""
    anchor = PERIOD_ANCHOR_INSTANT if duration is None else PERIOD_ANCHOR_DURATION
    return labelled_row(_rows(company_facts, anchor, "currency"), fiscal_year, fiscal_period, duration)


def _reported_for(company_facts: dict, concept: str, kind: str, row: dict) -> bool:
    """True if `concept` has a non-zero value for exactly the period of `row` (same start and end)."""
    return any(_period(r) == _period(row) and r["val"] != 0 for r in _rows(company_facts, concept, kind))


def find_fact(company_facts: dict, ticker: str, metric: Metric, fiscal_year: int, fiscal_period: str = "FY",
              duration: str | None = None, sector: str | None = None, cik: str | None = None) -> Fact | None:
    """The reported value of `metric`, or None if the company does not report it for that period.

    `fiscal_year` must be an int (a string would silently match nothing). `fiscal_period` ("FY", "Q1"-"Q3") and
    `duration` are case- and space-insensitive. `duration` defaults to 12M for FY and 3M for quarters and must
    suit the period (VALID_DURATIONS, else ValueError); balance-sheet metrics are instants whatever is asked.
    Cash-flow metrics are reported year-to-date in 10-Qs (a 3M figure only by a few companies): when the 3M
    figure is None, `metric.quarterly_duration(fiscal_period)` gives the duration that exists (the answer then
    says it is year-to-date, spec §4.2). `sector` (config/companies.yaml) enables sector-specific concepts such
    as banks' "long-term debt"; `cik` is used for the filing URL."""
    if not isinstance(fiscal_year, int) or isinstance(fiscal_year, bool):
        raise TypeError(f"fiscal_year must be an int, not {fiscal_year!r}")
    ticker = ticker.strip().upper()
    fiscal_period = str(fiscal_period).strip().upper()
    if fiscal_period not in VALID_DURATIONS:
        raise ValueError(f"fiscal_period must be one of {sorted(VALID_DURATIONS)}, not {fiscal_period!r}; "
                         f"Q4 is never derived")
    if metric.period == "instant":
        duration = None
    elif duration is None:
        duration = DEFAULT_DURATION[fiscal_period]
    else:
        duration = str(duration).strip().upper()
        if duration not in VALID_DURATIONS[fiscal_period]:
            raise ValueError(f"duration {duration!r} is not valid for {fiscal_period}; use one of "
                             f"{sorted(VALID_DURATIONS[fiscal_period])} (Q4 is never derived)")
    scale = CURRENCY_SCALE if metric.kind == "currency" else 1
    anchor = fiscal_period_row(company_facts, fiscal_year, fiscal_period, duration)
    candidates = [c for c in metric.concepts if c.applies_to(ticker, sector)]
    # Pass 1: concepts the period's own filing used (so original_value is what that filing said); their dates
    # must match the anchor's, or a concept the filing reports only for a comparative column would give the
    # prior period. Pass 2: the anchor's period, for a concept the filing showed under another tag or in a note.
    for use_anchor in (False, True):
        for concept in candidates:
            rows = _rows(company_facts, concept.name, metric.kind)
            own = anchor if use_anchor else labelled_row(rows, fiscal_year, fiscal_period, duration)
            if own is None or (anchor is not None and _period(own) != _period(anchor)):
                continue
            found = select_row(rows, _period(own))
            if found is None:
                continue
            row, first = found
            if any(_reported_for(company_facts, other, metric.kind, row) for other in concept.unless_reported):
                continue
            return Fact(ticker, metric.name, fiscal_year, fiscal_period, duration, row["val"] / scale,
                        OUTPUT_UNITS[metric.kind], concept.name, metric.label_for(ticker, concept),
                        row.get("start"), row["end"], row["accn"], row["form"], row["filed"],
                        filing_url(cik, row["accn"]), row["val"] != first["val"], first["val"] / scale,
                        first["accn"], own["accn"])
    return None


# ----------------------------------------------------------------------------------------------- coverage and checks


@dataclass(frozen=True)
class Gap:
    metric: str
    tickers: frozenset[str]
    periods: frozenset[str]            # fiscal periods the gap covers ("FY", "Q1"...); all when the map lists none
    reason: str


def _entries(path: Path, section: str, keys: frozenset[str], required: frozenset[str]) -> list[tuple[str, dict]]:
    """Validated (metric, spec) entries of the gaps or notes section; `metric` defaults to the entry name."""
    raw = _raw_map(path)
    metrics = set(raw["metrics"])
    out = []
    for name, spec in (raw.get(section) or {}).items():
        where = f"{section} {name}"
        _check_keys(where, spec, keys, required)
        metric = spec.get("metric", name)
        if metric not in metrics:
            raise _fail(where, f"unknown metric {metric!r}")
        out.append((metric, spec))
    return out


def load_gaps(path: Path = CONCEPT_MAP_FILE) -> list[Gap]:
    gaps = []
    for metric, spec in _entries(path, "gaps", GAP_KEYS, frozenset({"tickers", "reason"})):
        periods = _texts(f"gaps {metric} periods", spec["periods"]) if "periods" in spec else FISCAL_PERIODS
        if unknown := set(periods) - set(FISCAL_PERIODS):
            raise _fail(f"gaps {metric}", f"unknown period(s) {sorted(unknown)}")
        gaps.append(Gap(metric, frozenset(_texts(f"gaps {metric} tickers", spec["tickers"])), frozenset(periods),
                        _text(f"gaps {metric} reason", spec["reason"])))
    return gaps


def load_notes(path: Path = CONCEPT_MAP_FILE) -> dict[tuple[str, str], str]:
    """(metric, ticker) -> a caveat the answer must carry (e.g. a value the filing gives only rounded)."""
    return {(metric, t): _text(f"notes {metric} note", spec["note"])
            for metric, spec in _entries(path, "notes", NOTE_KEYS, frozenset({"tickers", "note"}))
            for t in _texts(f"notes {metric} tickers", spec["tickers"])}


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
# annual value / (quarterly value scaled to 12 months). The upper bound allows real swings: NVIDIA FY2024 revenue
# (Q1 7,192, year 60,922: 2.1) and MRK FY2023 R&D (an acquired-R&D charge in Q2: 2.3); a wrong concept is far
# outside (JNJ's in-process line: 0.008)
PLAUSIBLE_ANNUALISED_RATIO = (0.5, 3.0)


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


def value_pattern(value: float, unit: str) -> re.Pattern[str]:
    """The value as printed, as a whole number (not inside a longer one, e.g. 1,391,035)."""
    return re.compile("|".join(rf"(?<![\d,.]){re.escape(form)}(?!\d|,\d)" for form in display_forms(value, unit)))


def value_in_text(value: float, unit: str, text: str) -> bool:
    return value_pattern(value, unit).search(text) is not None


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
    words, number = STATEMENT_LINE_WORDS[metric], value_pattern(value, unit)
    return any(words.search(line) and number.search(line) for line in text.splitlines())


def missing_ticker_labels(metrics: dict[str, Metric], statements: dict[str, str]) -> list[str]:
    """Company-specific labels that do not appear in that company's statements (`statements`: ticker -> text),
    so a label in the concept map is always the company's own wording."""
    return [f"label not in {t}'s statements: {m.name} {label!r}" for m in metrics.values()
            for t, label in m.ticker_labels
            if t in statements and label.lower() not in statements[t].lower()]
