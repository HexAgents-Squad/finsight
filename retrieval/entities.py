"""Companies, fiscal years and fiscal periods named in a question (W1-05), for question-derived search filters.

The rules are lexical and come from config/companies.yaml, so they need no model and no gold data:
- a company is named by its full name or an alias, matched case-insensitively as whole words (a one-word name
  only when not all in lower case, so "chase growth" names no company), or by its ticker
  written in capitals (so the ticker "C" never matches the letter "c"; the config's documented planner rule), and
  not after "Class" or "Series" ("Class C shares");
- a fiscal year is a four-digit year from 1993 on ("fiscal 2024", "FY2024", "2024") or "FY24"; "latest" or "most
  recent" fiscal year reads as the newest indexed 10-K year and "latest quarter" as the newest indexed quarter,
  the readings the context pack uses for the questions that list no companies (eval/context_pack.py);
- a fiscal period is a quarter ("Q2", "second quarter", "2Q25", "Q2FY25"; "six months" or "first half" = Q2,
  "nine months" = Q3). The fourth quarter reads as the annual report (FY): the 10-Ks cover it and no 10-Q does
  (spec §4.2). A question that names no
  quarter asks about a fiscal year, so its period is FY; "10-Q", "quarterly report" or "three months ended" (a
  date, whose fiscal quarter depends on the company) without a quarter means every indexed quarter.
A follow-up ("And in the third quarter?") names only what changes: `from_turns` takes what the last turn names
and fills what it does not name from the earlier turns, the latest first; a turn that names a year but no period
keeps no earlier quarter. Known limits: a calendar date's year is read as the fiscal year ("quarter ended
September 30, 2024" is Microsoft's fiscal 2025); any number from 1993 to 2099 standing alone reads as a year
("2051 units"; a year outside the corpus makes search_filings answer out_of_corpus); "FY 2024-25" reads as 2024.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass

import yaml

from ingest.edgar import COMPANIES_FILE, load_scope
from retrieval.common import MIN_FISCAL_YEAR, Filters

YEAR = re.compile(r"(?<![\d.,$])(?:19|20)\d{2}(?!\d|[.,]\d)")      # "2024" and "2024,", not "$2,024", "2024.5"
SHORT_YEAR = re.compile(r"(?<![A-Za-z])FY\s?'?(\d{2})\b", re.IGNORECASE)   # "FY24", "Q2FY25"
QUARTER_FIRST = re.compile(r"\b([1-4])Q(\d{4}|\d{2})?\b", re.IGNORECASE)        # "1Q25", "1Q2025", "3Q"
LATEST = re.compile(r"\b(?:latest|most\s+recent)\b", re.IGNORECASE)
QUARTER_WORD = re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+(?:fiscal\s+)?quarter\b", re.IGNORECASE)
QUARTER_TOKEN = re.compile(r"(?<![A-Za-z])Q([1-4])(?:\b|(?=FY))", re.IGNORECASE)   # "Q2", "2025Q2", "Q2FY25"
YEAR_TO_DATE = {"six months": 2, "first half": 2, "nine months": 3}    # year to date: in that quarter's 10-Q
YEAR_TO_DATE_WORDS = re.compile(r"\b(six\s+months|first\s+half|nine\s+months)\b", re.IGNORECASE)
# a quarter named by its end date ("three months ended March 31, 2025") is some quarter of a fiscal year the date
# alone does not give (fiscal years differ by company): every indexed quarter, as for "10-Q"
QUARTERLY_FORM = re.compile(r"\b(?:10-Q|quarterly\s+report|three\s+months\s+ended)\b", re.IGNORECASE)
ANNUAL_FORM = re.compile(r"\b(?:10-K|annual\s+report)\b", re.IGNORECASE)
QUARTER_NUMBER = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
ANNUAL = "FY"


@dataclass(frozen=True)
class Entities:
    """What a question names: tickers in the order they first appear, fiscal years in that order, and the fiscal
    periods it asks about ("FY", "Q1", "Q2", "Q3"; empty when it names none)."""
    tickers: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    periods: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Tuples whatever sequence is given: a checkpoint (W2-06) restores them as lists, which would make an
        equal value compare unequal and unhashable."""
        for name in ("tickers", "years", "periods"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def filters(self, period: bool = False) -> list[Filters]:
        """One search filter per (company, year), and per period when `period` is set (FY when the question names
        none). A period the corpus does not index for that year (fiscal 2024 Q2: only fiscal-2025 10-Qs are
        indexed) is dropped for that year, which keeps the company and year. No company and no year: no filter."""
        periods = (self.periods or (ANNUAL,)) if period else (None,)
        out: list[Filters] = []
        for ticker in self.tickers or (None,):
            for year in self.years or (None,):
                kept = [p for p in periods if p is None or indexed(year, p)] or [None]
                out += [Filters.of(ticker=ticker, fiscal_year=year, fiscal_period=p) for p in kept]
        return [f for f in dict.fromkeys(out) if f.ticker or f.fiscal_year]     # a period alone narrows nothing


@dataclass(frozen=True)
class _Names:
    pattern: re.Pattern              # names and aliases, case-insensitive, longest first
    tickers: re.Pattern              # tickers in capitals only
    by_name: dict[str, str]          # casefolded name or alias -> ticker
    latest_year: int                 # newest indexed 10-K fiscal year
    latest_quarter: tuple[int, str]  # newest indexed 10-Q (fiscal year, quarter)
    quarter_years: frozenset[int]


@functools.cache
def _names() -> _Names:
    cfg = yaml.safe_load(COMPANIES_FILE.read_text(encoding="utf-8"))
    scope = load_scope()
    by_name: dict[str, str] = {}
    for c in cfg["companies"]:
        for name in (c["name"], *(c.get("aliases") or ())):
            by_name[name.casefold()] = c["ticker"]
    names = sorted(by_name, key=len, reverse=True)       # "JPMorgan Chase" before "JPMorgan" before "Chase"
    pattern = re.compile(r"(?<![\w&-])(?:" + "|".join(re.escape(n) for n in names) + r")(?![\w&-])", re.IGNORECASE)
    # a share class is not a ticker ("Class C shares" is not Citigroup)
    tickers = re.compile(r"(?<![\w&.-])(?<![Cc]lass )(?<![Ss]eries )(?:" + "|".join(sorted((c["ticker"] for c in cfg["companies"]), key=len,
                                                              reverse=True)) + r")(?![\w&-])")
    quarter_year = max(scope.tenq_years)
    return _Names(pattern, tickers, by_name, max(scope.tenk_years), (quarter_year, max(scope.tenq_periods)),
                  scope.tenq_years)


def indexed(year: int | None, period: str) -> bool:
    """Whether the corpus indexes a filing for this fiscal year and period (any year when the year is unknown)."""
    if year is None or period == ANNUAL:
        return True
    return year in _names().quarter_years


def companies(text: str) -> tuple[str, ...]:
    """Tickers named in the text, in the order they first appear."""
    names = _names()
    # a one-word name in lower case is an ordinary word ("chase growth", "meta trends"), not the company
    found = [(m.start(), names.by_name[m.group().casefold()]) for m in names.pattern.finditer(text)
             if " " in m.group() or not m.group().islower()]
    found += [(m.start(), m.group()) for m in names.tickers.finditer(text)]
    return tuple(dict.fromkeys(t for _, t in sorted(found)))


def years(text: str) -> tuple[int, ...]:
    found = [(m.start(), int(m.group())) for m in YEAR.finditer(text)]
    found += [(m.start(), 2000 + int(m.group(1))) for m in SHORT_YEAR.finditer(text)]
    found += [(m.start(), 2000 + int(m.group(2))) for m in QUARTER_FIRST.finditer(text)      # "1Q25" (YEAR reads
              if m.group(2) and len(m.group(2)) == 2]                                       # "1Q2025")
    return tuple(dict.fromkeys(y for _, y in sorted(found) if y >= MIN_FISCAL_YEAR))


def periods(text: str) -> tuple[str, ...]:
    """The fiscal periods asked about; empty when the text names no quarter, form or 'latest quarter'."""
    found = [(m.start(), QUARTER_NUMBER[m.group(1).lower()]) for m in QUARTER_WORD.finditer(text)]
    found += [(m.start(), int(m.group(1))) for m in QUARTER_TOKEN.finditer(text)]
    found += [(m.start(), int(m.group(1))) for m in QUARTER_FIRST.finditer(text)]
    found += [(m.start(), YEAR_TO_DATE[" ".join(m.group(1).lower().split())])
              for m in YEAR_TO_DATE_WORDS.finditer(text)]
    out = [ANNUAL if q == 4 else f"Q{q}" for _, q in sorted(found)]
    if not out and LATEST.search(text) and re.search(r"\bquarter\b", text, re.IGNORECASE):
        out = [_names().latest_quarter[1]]
    if not out and QUARTERLY_FORM.search(text):
        out = sorted(load_scope().tenq_periods)
    if not out and ANNUAL_FORM.search(text):
        out = [ANNUAL]
    return tuple(dict.fromkeys(out))


def extract(question: str) -> Entities:
    """What one question names."""
    found_periods = periods(question)
    found_years = years(question)
    if not found_years and LATEST.search(question):
        names = _names()
        quarterly = bool(found_periods) and all(p != ANNUAL for p in found_periods)
        found_years = (names.latest_quarter[0] if quarterly else names.latest_year,)
    return Entities(companies(question), found_years, found_periods)


def from_turns(questions: list[str]) -> Entities:
    """What the last question names, with each field it leaves out taken from the latest earlier turn that names
    it ("And in the third quarter?" keeps the company and year of the turn before). A turn that names a year and
    no period asks about that fiscal year, so it keeps no earlier quarter ("Apple's Q3 2025 revenue?", "And fiscal
    2024?" asks about the 2024 10-K)."""
    if not questions:
        return Entities()
    out = extract(questions[-1])
    tickers, years, periods = out.tickers, out.years, out.periods
    open_period = not periods and not years          # the period is still to be found in an earlier turn
    for earlier in reversed(questions[:-1]):
        prev = extract(earlier)
        tickers, years = tickers or prev.tickers, years or prev.years
        if open_period and (prev.periods or prev.years):
            periods, open_period = prev.periods, False   # a turn naming only a year asks about the fiscal year
    return Entities(tickers, years, periods)

