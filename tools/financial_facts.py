"""get_financial_fact: one reported figure from the cached SEC XBRL company facts (WP3 selection rules).

    get_financial_fact(ticker, metric, fiscal_year, fiscal_period="FY", duration=None)
        -> {value, unit, ticker, fiscal_year, fiscal_period, duration, period_start, period_end, concept,
            accession_no, filed, url, metric, label, form, restated, original_value, original_accession_no,
            period_accession_no, cited_filing_indexed, later_filings_exist, notes}

The selection itself is ingest/xbrl.py find_fact (ADR-011): the company's own fiscal-year/period labels, never
calendar dates; the period's own values, never a prior-year comparative column; the latest filed value with the
original kept when restated; currency in USD millions, per-share values unscaled, balance-sheet values as
point-in-time (duration null). Duration defaults to 12M for FY and 3M for quarters (spec §4.2).

Citation fields: `accession_no`/`form`/`filed`/`url` describe the filing that carries the value used (the
earliest with the final value; a later filing when the value was restated), while `fiscal_year`/`fiscal_period`
are always the period of the value, as the answer's citation needs (spec §8 invariant 2). `period_accession_no`
is the period's own filing (the one in the text corpus). `later_filings_exist` (spec §4.3.3) is set for the
latest indexed fiscal year only: true when the company has already filed a later 10-K, so an answer to "latest"
must say that later filings exist but are not indexed; null for other periods.

Every way a figure can be unavailable is a typed error result, checked in this order, so the agent can tell a
policy refusal from a data finding (spec §5 G):
    invalid_input   wrong types, unknown metric, a duration the period cannot have
    not_reported    Q4: never filed separately and never derived (spec §4.2.3)
    out_of_corpus   ticker not among the 20 companies, or a period outside config/companies.yaml (R3)
    not_reported    a gap declared in config/concept_map.yaml (with its reason), or no value for the period
                    (a cash-flow metric reported only year-to-date names the duration that exists)
    data_missing    the XBRL cache, concept map, company list or manifest is missing or unreadable
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator

from ingest import COMPANIES_FILE, MANIFEST_FILE, XBRL_DIR
from ingest.edgar import load_scope
from ingest.xbrl import (
    ANNUAL_FORMS,
    CONCEPT_MAP_FILE,
    PERIOD_ANCHOR_DURATION,
    TAXONOMY,
    VALID_DURATIONS,
    Fact,
    find_fact,
    load_concept_map,
    load_gaps,
    load_notes,
)
from tools.base import (
    DATA_MISSING,
    INVALID_INPUT,
    NOT_REPORTED,
    OUT_OF_CORPUS,
    FactNotFound,
    ToolArgs,
    ToolError,
    tool_boundary,
    tool_schema,
)

# Q4 is accepted so it can be answered "not reported separately" (spec §4.2.3) rather than "fix your call"
FiscalPeriod = Literal["FY", "Q1", "Q2", "Q3", "Q4"]
Duration = Literal["3M", "6M", "9M", "12M"]
ANNUAL = "FY"
FOURTH_QUARTER = "Q4"
THREE_MONTHS = "3M"
# The output fields, in order; compare_periods fills the same keys with null for a year without a value
FACT_FIELDS = ("value", "unit", "ticker", "fiscal_year", "fiscal_period", "duration", "period_start", "period_end",
               "concept", "accession_no", "filed", "url", "metric", "label", "form", "restated", "original_value",
               "original_accession_no", "period_accession_no", "cited_filing_indexed", "later_filings_exist",
               "notes")
Q4_MESSAGE = ("Q4 is not reported separately: 10-Qs cover Q1-Q3 and the 10-K the whole fiscal year, and Q4 is "
              "never derived from them (spec §4.2)")
YEAR_TO_DATE_NOTE = "year-to-date figure ({duration}), not the three-month quarter (spec §4.2)"
RESTATED_NOTE = ("restated: first reported as {original} {unit} (accession {accession}); the latest filed "
                 "value is used (spec §7.7)")
INSTANT_NOTE = "point-in-time (balance-sheet) value as of {period_end}; a duration does not apply"
OUTSIDE_CORPUS_NOTE = ("cited from a {form} filed {filed} that is not in the indexed text corpus: it carries the "
                       "latest filed value (spec §7.7)")
# What can go wrong while building the store from the config files and the manifest: setup, not a bad call
SETUP_ERRORS = (OSError, ValueError, KeyError, TypeError, yaml.YAMLError)


class FactArgs(ToolArgs):
    ticker: str = Field(min_length=1, max_length=10, description="Ticker of one of the 20 companies, e.g. AAPL")
    metric: str = Field(min_length=1, description="Metric name from config/concept_map.yaml, e.g. revenue")
    fiscal_year: int = Field(description="The company's own fiscal-year label (FY2025 -> 2025), not calendar year")
    fiscal_period: FiscalPeriod = Field(default="FY", description=(
        "FY, or a fiscal quarter Q1-Q3; Q4 is never reported separately (returns not_reported)"))
    duration: Duration | None = Field(default=None, description=(
        "Length of the period: 12M for FY; 3M (default) or the year-to-date 6M (Q2) / 9M (Q3) for quarters. "
        "Ignored for balance-sheet metrics"))

    @field_validator("ticker", "fiscal_period", "duration", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("metric", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


def period_text(fiscal_year: int, fiscal_period: str) -> str:
    """"FY2024" for a fiscal year, "FY2025 Q2" for a quarter (as answers and errors name periods)."""
    return f"FY{fiscal_year}" + ("" if fiscal_period == ANNUAL else f" {fiscal_period}")


class FactStore:
    """The concept map, the corpus scope and the cached company facts (each ticker's JSON loaded once)."""

    def __init__(self, xbrl_dir: Path = XBRL_DIR, concept_map: Path = CONCEPT_MAP_FILE,
                 companies: Path = COMPANIES_FILE, manifest: Path = MANIFEST_FILE) -> None:
        self.xbrl_dir = Path(xbrl_dir)
        self.metrics = load_concept_map(concept_map)
        self.gaps = load_gaps(concept_map)
        self.notes = load_notes(concept_map)
        self.scope = load_scope(companies)
        self.companies = {c.ticker: c for c in self.scope.companies}
        self.latest_year = max(self.scope.tenk_years)
        self._facts: dict[str, dict] = {}
        # Accession numbers of the downloaded (indexed) filings; None when the manifest is absent, in which case
        # whether a cited filing is in the text corpus is unknown rather than guessed
        self.indexed: frozenset[str] | None = None
        if Path(manifest).exists():
            lines = Path(manifest).read_text(encoding="utf-8").splitlines()
            self.indexed = frozenset(json.loads(line)["accession_no"] for line in lines if line.strip())

    def company_facts(self, ticker: str) -> dict:
        if ticker not in self._facts:
            path = self.xbrl_dir / f"{ticker}.json"
            if not path.exists():
                raise ToolError(DATA_MISSING, f"XBRL facts for {ticker} are not cached ({path}); run "
                                              f"ingest/05_xbrl_facts.py")
            try:
                self._facts[ticker] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:   # broken data, not a bad call
                raise ToolError(DATA_MISSING, f"XBRL facts for {ticker} cannot be read ({path}: {err}); run "
                                              f"ingest/05_xbrl_facts.py --refresh") from err
        return self._facts[ticker]

    def check_company_metric(self, ticker: str, metric: str) -> None:
        """Errors that apply to every period of the request (compare_periods fails the whole call on these)."""
        if metric not in self.metrics:
            raise ToolError(INVALID_INPUT, f"unknown metric {metric!r}; use one of {sorted(self.metrics)}")
        if ticker not in self.companies:
            raise ToolError(OUT_OF_CORPUS, f"{ticker} is not one of the {len(self.companies)} companies in "
                                           f"config/companies.yaml")

    def check_request(self, ticker: str, metric: str, fiscal_year: int, fiscal_period: str,
                      duration: str | None = None) -> None:
        """Raise the right typed error for a request the corpus cannot answer, before any lookup, in the order of
        the module docstring."""
        spec = self.metrics.get(metric)
        if fiscal_period != FOURTH_QUARTER and spec is not None and spec.period == "duration" \
                and duration is not None and duration not in VALID_DURATIONS[fiscal_period]:
            raise ToolError(INVALID_INPUT, f"duration {duration!r} is not valid for {fiscal_period}; use one of "
                                           f"{sorted(VALID_DURATIONS[fiscal_period])}")
        if spec is not None and fiscal_period == FOURTH_QUARTER:
            raise FactNotFound(NOT_REPORTED, Q4_MESSAGE, ticker=ticker, metric=metric, fiscal_year=fiscal_year,
                               fiscal_period=fiscal_period)
        self.check_company_metric(ticker, metric)
        in_scope = (fiscal_year in self.scope.tenk_years if fiscal_period == ANNUAL else
                    fiscal_year in self.scope.tenq_years and fiscal_period in self.scope.tenq_periods)
        if not in_scope:
            scope = self.scope
            raise ToolError(OUT_OF_CORPUS, f"{period_text(fiscal_year, fiscal_period)} is outside the indexed "
                                           f"filings (10-K fiscal {sorted(scope.tenk_years)}, 10-Q fiscal "
                                           f"{sorted(scope.tenq_years)} {sorted(scope.tenq_periods)})")
        for gap in self.gaps:
            if gap.metric == metric and ticker in gap.tickers and fiscal_period in gap.periods:
                raise FactNotFound(NOT_REPORTED, f"{metric} is not reported by {ticker}: {gap.reason}",
                                   ticker=ticker, metric=metric, fiscal_year=fiscal_year,
                                   fiscal_period=fiscal_period)

    def lookup(self, ticker: str, metric: str, fiscal_year: int, fiscal_period: str = ANNUAL,
               duration: str | None = None) -> Fact:
        """The Fact, or a typed ToolError/FactNotFound. Arguments must be normalised (FactArgs does that)."""
        self.check_request(ticker, metric, fiscal_year, fiscal_period, duration)
        spec, company = self.metrics[metric], self.companies[ticker]
        facts = self.company_facts(ticker)
        # Arguments are validated above, so an exception from find_fact is a bug and surfaces as internal_error
        fact = find_fact(facts, ticker, spec, fiscal_year, fiscal_period, duration, company.sector, company.cik)
        if fact is not None:
            return fact
        where = dict(ticker=ticker, metric=metric, fiscal_year=fiscal_year, fiscal_period=fiscal_period)
        period = period_text(fiscal_year, fiscal_period)
        asked = duration or (THREE_MONTHS if fiscal_period != ANNUAL else None)
        ytd = spec.quarterly_duration(fiscal_period) if fiscal_period != ANNUAL else None
        # Suggest the year-to-date duration only when it differs from the one asked and does have a value
        if spec.period == "duration" and ytd and ytd != asked and find_fact(
                facts, ticker, spec, fiscal_year, fiscal_period, ytd, company.sector):
            raise FactNotFound(NOT_REPORTED, f"{ticker} reports {metric} for {period} only year-to-date; ask "
                                             f"for duration {ytd}", available_duration=ytd, **where)
        raise FactNotFound(NOT_REPORTED, f"no reported {metric} for {ticker} {period}"
                                         + (f" ({duration})" if duration else ""), **where)

    def later_filings_exist(self, fact: Fact) -> bool | None:
        """For the latest indexed fiscal year: has the company already filed a 10-K for a later fiscal year
        (spec §4.3.3)? None for any other period. Read from the net-income rows every filing carries."""
        if fact.fiscal_period != ANNUAL or fact.fiscal_year != self.latest_year:
            return None
        rows = (self.company_facts(fact.ticker).get("facts", {}).get(TAXONOMY, {})
                .get(PERIOD_ANCHOR_DURATION, {}).get("units", {}).get("USD", []))
        return any(r.get("form") in ANNUAL_FORMS and isinstance(r.get("fy"), int) and r["fy"] > self.latest_year
                   for r in rows)

    def notes_for(self, fact: Fact, duration_asked: str | None) -> list[str]:
        """The caveats the answer must carry with this value (restatement, year-to-date, concept-map notes)."""
        notes = []
        if fact.restated:
            notes.append(RESTATED_NOTE.format(original=_figure(fact.original_value), unit=fact.unit,
                                              accession=fact.original_accession_no))
        if fact.fiscal_period != ANNUAL and fact.duration not in (None, THREE_MONTHS):
            notes.append(YEAR_TO_DATE_NOTE.format(duration=fact.duration))
        if fact.duration is None and duration_asked is not None:
            notes.append(INSTANT_NOTE.format(period_end=fact.period_end))
        if self.indexed is not None and fact.accession_no not in self.indexed:
            notes.append(OUTSIDE_CORPUS_NOTE.format(form=fact.form, filed=fact.filed))
        if note := self.notes.get((fact.metric, fact.ticker)):
            notes.append(note)
        return notes

    def result(self, fact: Fact, duration_asked: str | None) -> dict[str, Any]:
        """The §2.3 output fields plus what the answer contract also needs, in FACT_FIELDS order."""
        out = {"value": fact.value, "unit": fact.unit, "ticker": fact.ticker, "fiscal_year": fact.fiscal_year,
               "fiscal_period": fact.fiscal_period, "duration": fact.duration, "period_start": fact.period_start,
               "period_end": fact.period_end, "concept": f"{TAXONOMY}:{fact.concept}",
               "accession_no": fact.accession_no, "filed": fact.filed, "url": fact.url, "metric": fact.metric,
               "label": fact.label, "form": fact.form, "restated": fact.restated,
               "original_value": fact.original_value, "original_accession_no": fact.original_accession_no,
               "period_accession_no": fact.period_accession_no,
               "cited_filing_indexed": None if self.indexed is None else fact.accession_no in self.indexed,
               "later_filings_exist": self.later_filings_exist(fact),
               "notes": self.notes_for(fact, duration_asked)}
        return {k: out[k] for k in FACT_FIELDS}


def _figure(value: float) -> str:
    """391035.0 -> "391,035"; 6.08 -> "6.08" (as the statements print it)."""
    return f"{value:,.0f}" if value.is_integer() else f"{value:,}"


@functools.cache
def _default_store() -> FactStore:
    return FactStore()


def default_store() -> FactStore:
    """The store over data/xbrl, built on first use (the concept map is validated then). A broken config file
    or manifest is a data_missing error, and the failure is not cached, so a fixed file works without a restart."""
    try:
        return _default_store()
    except SETUP_ERRORS as err:
        raise ToolError(DATA_MISSING, f"the XBRL fact store cannot be built: {type(err).__name__}: {err}") from err


@tool_boundary
def get_financial_fact(ticker: str, metric: str, fiscal_year: int, fiscal_period: str = ANNUAL,
                       duration: str | None = None, *, store: FactStore | None = None) -> dict[str, Any]:
    """Tool entry point: the fact, or {"error": {...}} (never raises). `store` is for tests."""
    args = FactArgs(ticker=ticker, metric=metric, fiscal_year=fiscal_year, fiscal_period=fiscal_period,
                    duration=duration)
    store = store or default_store()
    fact = store.lookup(args.ticker, args.metric, args.fiscal_year, args.fiscal_period, args.duration)
    return store.result(fact, args.duration)


SCHEMA = tool_schema(
    "get_financial_fact",
    "Look up one reported figure (revenue, net_income, total_assets, eps_diluted, ...) for a company and fiscal "
    "period from its SEC XBRL filings. Returns value and unit (USD millions; per-share in USD), the period dates, "
    "the XBRL concept and the filing to cite. A figure the company does not report returns a not_reported error "
    "with the reason; never substitute another line item.",
    FactArgs)
