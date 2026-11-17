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
from tools import financial_facts
from tools.base import NOT_REPORTED, OUT_OF_CORPUS, ToolArgs, ToolError, tool_boundary, tool_schema
from tools.calculator import compute
from tools.financial_facts import ANNUAL, FACT_FIELDS, FactStore, Metric, metric_name

MAX_YEARS = 10                          # distinct years; the corpus has 3, and a bound keeps a bad call cheap
PER_YEAR_ERRORS = frozenset({NOT_REPORTED, OUT_OF_CORPUS})
NOT_REQUESTED_NOTE = "no year-over-year growth: FY{previous} was not requested"
PREVIOUS_MISSING_NOTE = "no year-over-year growth: FY{previous} has no value"
YOY_UNDEFINED_NOTE = "year-over-year growth is not meaningful: {reason}"
MIXED_RESTATEMENT_NOTE = ("FY{restated} is a restated value and FY{other} is not (they come from different "
                          "filings), so the comparison mixes a restated and an originally reported figure")


class CompareArgs(ToolArgs):
    ticker: str = Field(min_length=1, max_length=10, description="Ticker of one of the 20 companies, e.g. NVDA")
    metric: Metric = Field(description="One of get_financial_fact's 13 metric names, e.g. revenue")
    years: list[int] = Field(min_length=1, description=(
        f"The company's own fiscal-year labels to compare, e.g. [2024, 2025]; at most {MAX_YEARS} distinct years"))

    @field_validator("metric", mode="before")
    @classmethod
    def _metric(cls, value: Any) -> Any:
        return metric_name(value)

    @field_validator("years")
    @classmethod
    def _distinct(cls, value: list[int]) -> list[int]:
        """Sorted and de-duplicated; the MAX_YEARS bound counts distinct years ([2024] * 11 is one year)."""
        years = sorted(set(value))
        if len(years) > MAX_YEARS:
            raise ValueError(f"at most {MAX_YEARS} distinct years, got {len(years)}")
        return years

    @field_validator("ticker", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value


def _row(store: FactStore, ticker: str, metric: str, year: int) -> dict[str, Any]:
    try:
        fact = store.lookup(ticker, metric, year, ANNUAL)
    except ToolError as err:                            # FactNotFound included
        # Only per-year findings become a row; a setup failure (cache missing or unreadable) fails the call, or
        # the agent would report "no value" for data that exists
        if err.error_type not in PER_YEAR_ERRORS:
            raise
        known = {"unit": OUTPUT_UNITS[store.metrics[metric].kind], "ticker": ticker, "fiscal_year": year,
                 "fiscal_period": ANNUAL, "metric": metric, "notes": [err.message]}
        return {k: known.get(k) for k in FACT_FIELDS} | {"missing": err.error_type}
    return store.result(fact, None) | {"missing": None}


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
        # Both years from the same (restating) filing are like for like; only a restated and an original figure
        # from different filings are mixed (ADR-011)
        if row["restated"] != previous["restated"] and row["accession_no"] != previous["accession_no"]:
            restated, other = (row, previous) if row["restated"] else (previous, row)
            row["notes"].append(MIXED_RESTATEMENT_NOTE.format(restated=restated["fiscal_year"],
                                                              other=other["fiscal_year"]))


@tool_boundary
def compare_periods(ticker: str, metric: str, years: list[int], *,
                    store: FactStore | None = None) -> list[dict[str, Any]] | dict[str, Any]:
    """Tool entry point: one row per requested year (oldest first), or {"error": {...}} for an unknown metric or
    company or invalid arguments (never raises). `store` is for tests."""
    args = CompareArgs(ticker=ticker, metric=metric, years=years)
    store = store or financial_facts.default_store()        # via the module, so tests can replace it
    store.check_company_metric(args.ticker, args.metric)
    rows = [_row(store, args.ticker, args.metric, y) for y in args.years]
    _add_yoy(rows)
    return rows


SCHEMA = tool_schema(
    "compare_periods",
    "Get one metric for a company across several fiscal years (oldest first) with year-over-year growth in % "
    "computed by the calculator. A year without a value is returned with value null and the reason in notes.",
    CompareArgs)
