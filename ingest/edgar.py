"""SEC EDGAR access and filing selection for the download step (ingest/01_download.py).

- SecClient: polite HTTP client (identifying User-Agent, < 10 requests/second, retries on 429/5xx).
- fiscal_labels(): accession number -> the company's own (fiscal year, fiscal period) from XBRL company facts.
  The fiscal year is NEVER derived from a calendar date (docs/spec.md §4.1): filings without a label are
  reported, not guessed.
- select_filings(): one original 10-K per in-scope fiscal year and the in-scope 10-Qs (config/companies.yaml).
  CIKs also come from config/companies.yaml (verified in WP1a) rather than SEC's company_tickers.json, so the
  scope file stays the single source of truth and each run needs one request less.
- find_annual_report(): the annual-report exhibit (e.g. Exhibit 13) that some banks file next to the 10-K.
"""
from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import requests
import yaml

from ingest import COMPANIES_FILE

SEC_WWW = "https://www.sec.gov"
SEC_DATA = "https://data.sec.gov"
MIN_INTERVAL_S = 0.15              # ~6-7 requests/second, safely under the SEC's limit of 10
TIMEOUT_S = 60
MAX_ATTEMPTS = 4
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
BACKOFF_BASE_S = 2.0               # waits 2, 4, 8 s between attempts
MAX_RETRY_AFTER_S = 60.0           # never wait longer than this, whatever Retry-After says
PLACEHOLDER_CONTACT = "example.com"
EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")

# A filing's annual-report exhibit: named like "ex13.htm", "ex-13", "exhibit13" (case-insensitive)
EXHIBIT_13_NAME = re.compile(r"ex(?:hibit)?[-_]?13", re.IGNORECASE)
XBRL_VIEWER_PAGE = re.compile(r"^R\d+\.htm$", re.IGNORECASE)
OTHER_EXHIBIT_NAME = re.compile(r"ex(?:hibit)?[-_]?\d", re.IGNORECASE)
# Without an "ex13" name, a document much larger than the primary one is the annual report (e.g. Wells Fargo
# files a thin 10-K wrapper plus the full report); "much larger" = this many times the primary and at least:
ANNUAL_REPORT_SIZE_RATIO = 3
ANNUAL_REPORT_MIN_BYTES = 1_000_000
EXHIBIT_LABEL = "Exhibit 13"


# ----------------------------------------------------------------------------------------------- scope
@dataclass(frozen=True)
class Company:
    ticker: str
    cik: str
    sector: str


@dataclass(frozen=True)
class Scope:
    companies: tuple[Company, ...]
    tenk_years: frozenset[int]
    tenq_years: frozenset[int]
    tenq_periods: frozenset[str]

    @property
    def earliest_year(self) -> int:
        return min(self.tenk_years | self.tenq_years)


def company_names(path=COMPANIES_FILE) -> dict[str, tuple[str, ...]]:
    """Per ticker, the names its filings print in running footers and headers: the full name and every alias
    of more than one word (a one-word alias such as "Chase" or "Meta" is too likely to start a table row)."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {c["ticker"]: (c["name"], *(a for a in c.get("aliases", []) if " " in a.strip()))
            for c in cfg["companies"]}


def load_scope(path=COMPANIES_FILE) -> Scope:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    forms = cfg["forms"]
    return Scope(
        companies=tuple(Company(c["ticker"], c["cik"], c["sector"]) for c in cfg["companies"]),
        tenk_years=frozenset(forms["10-K"]["fiscal_years"]),
        tenq_years=frozenset(forms["10-Q"]["fiscal_years"]),
        tenq_periods=frozenset(forms["10-Q"]["fiscal_periods"]),
    )


def check_user_agent(user_agent: str) -> str | None:
    """Problem with SEC_USER_AGENT, or None. The SEC blocks clients without a real contact."""
    if not EMAIL.search(user_agent):
        return "SEC_USER_AGENT must contain a contact email, e.g. 'HexAgents_Squad IITKGP you@domain.com'"
    if PLACEHOLDER_CONTACT in user_agent.lower():
        return "SEC_USER_AGENT still has the example.com placeholder; put your own email in .env"
    return None


# ----------------------------------------------------------------------------------------------- HTTP
class SecClient:
    """GET with an identifying User-Agent, a minimum gap between requests and retries on 429/5xx."""

    def __init__(self, user_agent: str, session: Any = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._sleep, self._clock = sleep, clock
        self._last_request = float("-inf")
        self.requests = 0

    def get(self, url: str) -> requests.Response:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            wait = MIN_INTERVAL_S - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = self._clock()
            self.requests += 1
            try:
                response = self.session.get(url, timeout=TIMEOUT_S)
            except requests.RequestException:        # connection reset, timeout, truncated body
                if attempt == MAX_ATTEMPTS:
                    raise
                self._sleep(BACKOFF_BASE_S ** attempt)
                continue
            if response.status_code == requests.codes.ok:
                return response
            if response.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
                wait = retry_after(response)          # 0 is a real answer ("retry now"), so test for None
                self._sleep(BACKOFF_BASE_S ** attempt if wait is None else wait)
                continue
            response.raise_for_status()
        raise RuntimeError(f"unreachable: {url}")

    def json(self, url: str) -> Any:
        return self.get(url).json()


def retry_after(response: Any) -> float | None:
    """Seconds the server asked us to wait (Retry-After header in seconds), capped; None if absent."""
    value = (getattr(response, "headers", None) or {}).get("Retry-After", "")
    return min(float(value), MAX_RETRY_AFTER_S) if str(value).strip().isdigit() else None


def submissions_url(cik: str) -> str:
    return f"{SEC_DATA}/submissions/CIK{cik}.json"


def companyfacts_url(cik: str) -> str:
    return f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik}.json"


def filing_folder_url(cik: str, accession_no: str) -> str:
    return f"{SEC_WWW}/Archives/edgar/data/{int(cik)}/{accession_no.replace('-', '')}"


# ----------------------------------------------------------------------------------------------- fiscal labels
def fiscal_labels(company_facts: dict) -> dict[str, tuple[int, str]]:
    """accession number -> (fiscal year, fiscal period) as the company tags it in XBRL (e.g. Apple's FY2025 =
    Oct 2024 - Sep 2025). Every fact row of a filing carries the filing's fy/fp; the most common pair wins in
    case a row is mis-tagged."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for taxonomy in company_facts.get("facts", {}).values():
        for concept in taxonomy.values():
            for rows in concept.get("units", {}).values():
                for row in rows:
                    if row.get("accn") and row.get("fy") and row.get("fp"):
                        votes[row["accn"]][(int(row["fy"]), str(row["fp"]).upper())] += 1
    return {accn: counter.most_common(1)[0][0] for accn, counter in votes.items()}


# ----------------------------------------------------------------------------------------------- selection
def filing_rows(submissions: dict, older_pages: list[dict]) -> list[dict]:
    """All filings of a company as row dicts, newest first. `filings.recent` holds about the last 1,000 filings;
    heavy filers (banks) need the older pages listed in `filings.files`."""
    rows = []
    for page in [submissions["filings"]["recent"], *older_pages]:
        columns = list(page)
        rows += [dict(zip(columns, values, strict=True)) for values in zip(*page.values(), strict=True)]
    return sorted(rows, key=lambda r: (r.get("filingDate", ""), r.get("accessionNumber", "")), reverse=True)


def older_page_names(submissions: dict, earliest_year: int) -> list[str]:
    """Older submission pages that can contain in-scope filings (a fiscal year's 10-K is filed up to a year
    after the year starts, so pages ending before `earliest_year - 1` are not needed)."""
    cutoff = f"{earliest_year - 1}-01-01"
    return [p["name"] for p in submissions["filings"].get("files", []) if p.get("filingTo", "") >= cutoff]


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: str
    sector: str
    form: str
    fiscal_year: int
    fiscal_period: str
    period_end: str
    filing_date: str
    accession_no: str
    primary_document: str

    @property
    def source_url(self) -> str:
        return f"{filing_folder_url(self.cik, self.accession_no)}/{self.primary_document}"

    @property
    def stem(self) -> str:
        period = "" if self.fiscal_period == "FY" else self.fiscal_period
        return f"{self.form}_{self.fiscal_year}{period}_{self.accession_no}"


@dataclass
class Selection:
    filings: list[Filing]
    unlabelled: list[str]          # in-scope-looking filings without an XBRL fiscal label (reported, not guessed)
    missing: list[str]             # expected (form, year, period) not found


def select_filings(company: Company, rows: list[dict], labels: dict[str, tuple[int, str]], scope: Scope) -> Selection:
    """One original 10-K per in-scope fiscal year and one 10-Q per in-scope fiscal (year, quarter); the newest
    filing wins if a period was filed twice. Amendments (10-K/A, 10-Q/A) are never selected."""
    chosen: dict[tuple[str, int, str], Filing] = {}
    unlabelled = []
    for r in rows:                                       # newest first
        form = r.get("form")
        if form not in ("10-K", "10-Q"):
            continue
        label = labels.get(r["accessionNumber"])
        if label is None:
            # only filings that could be in scope are worth reporting (the report date is used for this
            # filter only, never as a fiscal label)
            if r.get("reportDate", "")[:4] >= str(scope.earliest_year - 1):
                unlabelled.append(f"{form} {r['accessionNumber']} (period {r.get('reportDate', '?')})")
            continue
        fy, fp = label
        if form == "10-K":
            in_scope = fy in scope.tenk_years and fp == "FY"
        else:
            in_scope = fy in scope.tenq_years and fp in scope.tenq_periods
        key = (form, fy, fp)
        if in_scope and key not in chosen:
            chosen[key] = Filing(company.ticker, company.cik, company.sector, form, fy, fp, r.get("reportDate", ""),
                                 r.get("filingDate", ""), r["accessionNumber"], r["primaryDocument"])
    expected = [("10-K", y, "FY") for y in sorted(scope.tenk_years)]
    expected += [("10-Q", y, p) for y in sorted(scope.tenq_years) for p in sorted(scope.tenq_periods)]
    missing = [f"{form} {y} {p}" for form, y, p in expected if (form, y, p) not in chosen]
    return Selection(sorted(chosen.values(), key=lambda f: (f.form, f.fiscal_year, f.fiscal_period)),
                     unlabelled, missing)


# ----------------------------------------------------------------------------------------------- exhibits
def find_annual_report(index_items: list[dict], primary_document: str) -> str | None:
    """Name of the annual-report document filed next to the primary 10-K document, if any: an "ex13" file, or
    else an HTML document much larger than the primary one (other numbered exhibits excluded)."""
    html = [i for i in index_items
            if i["name"].lower().endswith((".htm", ".html")) and not XBRL_VIEWER_PAGE.match(i["name"])
            and "index" not in i["name"].lower() and i["name"] != primary_document]
    named = sorted(i["name"] for i in html if EXHIBIT_13_NAME.search(i["name"]))
    if named:
        return named[0]
    sizes = {i["name"]: int(i["size"]) for i in html if str(i.get("size", "")).isdigit()}
    primary_size = next((int(i["size"]) for i in index_items
                         if i["name"] == primary_document and str(i.get("size", "")).isdigit()), 0)
    big = [n for n, s in sizes.items() if not OTHER_EXHIBIT_NAME.search(n)
           and s > max(ANNUAL_REPORT_SIZE_RATIO * primary_size, ANNUAL_REPORT_MIN_BYTES)]
    return max(big, key=lambda n: sizes[n]) if big else None


def manifest_record(filing: Filing, path: str, exhibit_name: str | None, exhibit_path: str | None) -> dict:
    """One line of data/raw/manifest.jsonl (paths relative to data/)."""
    record = asdict(filing)
    record.pop("primary_document")
    record.update(source_url=filing.source_url, path=path, exhibit=EXHIBIT_LABEL if exhibit_name else None,
                  exhibit_url=f"{filing_folder_url(filing.cik, filing.accession_no)}/{exhibit_name}"
                  if exhibit_name else None,
                  exhibit_path=exhibit_path)
    return record
