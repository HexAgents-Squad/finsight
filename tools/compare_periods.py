"""compare_periods: one metric across several fiscal years, with year-over-year growth from the calculator.

    compare_periods(ticker, metric, years)
        -> [{fiscal_year, value, unit, period_end, yoy_growth_pct, yoy_calculation, ...get_financial_fact fields}]

Years are de-duplicated and sorted (oldest first); values are annual (FY, 12M or the year-end balance). A year
that cannot be answered (not reported, outside the corpus) comes back as a row with value null and the reason in
`notes`, not as an error, so the other years are still usable. An unknown metric or company fails the whole call.

yoy_growth_pct compares a year with the PREVIOUS fiscal year only (FY2025 vs FY2024), computed by
tools/calculator.py (growth, half-up to 1 decimal) and returned with its calculation so the answer's
`calculations` can copy it; when the previous year is missing or not requested, it is null and `notes` says
why (the oldest requested year simply has no comparison). A comparison between a restated and an unrestated
year is noted (ADR-011).
"""
from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from ingest.xbrl import OUTPUT_UNITS
from tools.base import NOT_REPORTED, OUT_OF_CORPUS, ToolArgs, ToolError, tool_boundary, tool_schema
from tools.calculator import compute
from tools.financial_facts import ANNUAL, FactStore, default_store, fact_result

MAX_YEARS = 10                          # the corpus has 3 fiscal years; a bound keeps a bad call cheap
PER_YEAR_ERRORS = frozenset({NOT_REPORTED, OUT_OF_CORPUS})
MISSING_FIELDS = ("period_start", "period_end", "duration", "concept", "accession_no", "filed", "url", "label",
                  "form", "restated", "original_value", "original_accession_no", "cited_filing_indexed")
NOT_REQUESTED_NOTE = "no year-over-year growth: FY{previous} was not requested"
PREVIOUS_MISSING_NOTE = "no year-over-year growth: FY{previous} has no value"
YOY_UNDEFINED_NOTE = "year-over-year growth is not meaningful: {reason}"
MIXED_RESTATEMENT_NOTE = ("FY{restated} is a restated value and FY{other} is not, so the comparison mixes a restated "
                          "and an originally reported figure")


class CompareArgs(ToolArgs):
    ticker: str = Field(min_length=1, max_length=10, description="Ticker of one of the 20 companies, e.g. NVDA")
    metric: str = Field(min_length=1, description="Metric name from config/concept_map.yaml, e.g. revenue")
    years: list[int] = Field(min_length=1, max_length=MAX_YEARS, description=(
        "The company's own fiscal-year labels to compare, e.g. [2024, 2025]"))

    @field_validator("years", mode="before")
    @classmethod
    def _unique(cls, value: Any) -> Any:
        """Duplicates are dropped before the MAX_YEARS bound applies ([2024] * 11 is one year)."""
        if isinstance(value, list) and all(isinstance(y, int) and not isinstance(y, bool) for y in value):
            return sorted(set(value))
        return value

    @field_validator("ticker", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("metric", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


def _row(store: FactStore, ticker: str, metric: str, year: int) -> dict[str, Any]:
    try:
        fact = store.lookup(ticker, metric, year, ANNUAL)
    except ToolError as err:                            # FactNotFound included
        # Only per-year findings become a row; a setup failure (cache missing or unreadable) fails the call, or
        # the agent would report "no value" for data that exists
        if err.error_type not in PER_YEAR_ERRORS:
            raise
        unit = OUTPUT_UNITS[store.metrics[metric].kind]
        return {"value": None, "unit": unit, "ticker": ticker, "fiscal_year": year, "fiscal_period": ANNUAL,
                **dict.fromkeys(MISSING_FIELDS), "metric": metric, "missing": err.error_type,
                "notes": [err.message]}
    return fact_result(fact, store.notes_for(fact, None), store.indexed) | {"missing": None}


def _add_yoy(rows: list[dict[str, Any]]) -> None:
    by_year = {r["fiscal_year"]: r for r in rows}
    for row in rows:
        row["yoy_growth_pct"], row["yoy_calculation"] = None, None
        previous = by_year.get(row["fiscal_year"] - 1)
        if row["value"] is None or (row is rows[0] and previous is None):
            continue                                   # the oldest requested year has nothing to compare with
        if previous is None:
            row["notes"].append(NOT_REQUESTED_NOTE.format(previous=row["fiscal_year"] - 1))
            continue
        if previous["value"] is None:
            row["notes"].append(PREVIOUS_MISSING_NOTE.format(previous=previous["fiscal_year"]))
            continue
        try:
            calc = compute("growth", [previous["value"], row["value"]])
        except ToolError as err:                        # growth from 0 is undefined (spec §7.2)
            row["notes"].append(YOY_UNDEFINED_NOTE.format(reason=err.message))
            continue
        row["yoy_growth_pct"], row["yoy_calculation"] = calc["result"], calc
        row["notes"] += calc["notes"]
        if row["restated"] != previous["restated"]:
            restated, other = (row, previous) if row["restated"] else (previous, row)
            row["notes"].append(MIXED_RESTATEMENT_NOTE.format(restated=restated["fiscal_year"],
                                                              other=other["fiscal_year"]))


@tool_boundary
def compare_periods(ticker: str, metric: str, years: list[int], *,
                    store: FactStore | None = None) -> list[dict[str, Any]] | dict[str, Any]:
    """Tool entry point: one row per requested year (oldest first), or {"error": {...}} for an unknown metric or
    company or invalid arguments (never raises). `store` is for tests."""
    args = CompareArgs(ticker=ticker, metric=metric, years=years)
    store = store or default_store()
    store.check_company_metric(args.ticker, args.metric)
    rows = [_row(store, args.ticker, args.metric, y) for y in sorted(set(args.years))]
    _add_yoy(rows)
    return rows


SCHEMA = tool_schema(
    "compare_periods",
    "Get one metric for a company across several fiscal years (oldest first) with year-over-year growth in % "
    "computed by the calculator. A year without a value is returned with value null and the reason in notes.",
    CompareArgs)
