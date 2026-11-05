"""WP3: the reported financial figures of every in-scope company and period, with coverage and spot checks.

Usage:
    python ingest/05_xbrl_facts.py                   # uses the cached company facts (downloads missing ones)
    python ingest/05_xbrl_facts.py --refresh         # download all company facts again first

Input : data/xbrl/<TICKER>.json (SEC company facts; ingest/01_download.py caches them too),
        config/concept_map.yaml, data/sections/sections.jsonl (optional, for the spot and label checks)
Output: data/xbrl/facts_table.csv   one row per (company, metric, fiscal period, duration) found, with whether
                                    the cited filing is in the indexed corpus and any caveat from the concept map
        data/xbrl/coverage.csv      metric x company x period: the line item found, or empty for a gap
        data/xbrl/spot_check.csv    every original value searched in the financial statements of the filing that
                                    first reported it or of the period's own filing, and every restated value in
                                    the filing it cites, on a line carrying the metric's words
Checks (exit 1 if one fails): at least MIN_METRICS metrics mapped; real coverage equals the gaps declared in
config/concept_map.yaml, year by year; annual values in proportion with the same year's quarterly values (for
every in-scope year, using out-of-scope 10-Qs where needed); every company-specific label appears in that
company's statements; at least MIN_SPOT_CHECK_SHARE of the values found on their statement line, and no miss
without a known reason.
Exit codes: 0 = all checks pass, 1 = a check failed, 2 = setup error (settings, network, cache files).
Selection rules: ingest/xbrl.py. Figures are as of the company-facts download: a --refresh can add restatements.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from core.settings import get_sec_settings  # noqa: E402
from ingest import SECTIONS_DIR, XBRL_DIR  # noqa: E402
from ingest.edgar import SecClient, check_user_agent, companyfacts_url, load_scope  # noqa: E402
from ingest.xbrl import (  # noqa: E402
    PLAUSIBILITY_METRICS,
    QUARTERS,
    Fact,
    find_fact,
    gap_problems,
    implausible,
    load_concept_map,
    load_gaps,
    load_notes,
    missing_ticker_labels,
    value_on_statement_line,
)

MIN_METRICS = 10                    # Jira UPGRAD-26 acceptance: at least 10 metrics mapped
MIN_SPOT_CHECK_SHARE = 0.95         # share of values that must appear on their line of the original statements
# Where the financial statements are (data/sections): 10-K Item 8, or Item 15 for filers that put them after the
# exhibits list; 10-Q Part I Item 1
STATEMENT_ITEMS = frozenset({"Item 8", "Item 15", "Part I Item 1"})
# Checked misses, with the reason (each value was confirmed by hand in the filing), keyed by (ticker, metric,
# fiscal year or None for every year). Any other miss fails the run.
KNOWN_TEXT_MISSES = {
    ("PG", "rd_expense", None): 'the 10-K states R&D only in words, rounded to $0.1 billion ("$2.0/2.1 billion")',
    ("MRK", "total_assets", None): "the balance sheet prints the total-assets row without a label",
    ("MRK", "eps_diluted", 2023): "printed under an 'Assuming Dilution' heading, on the line 'Net Income | 0.14'",
    ("BAC", "stockholders_equity", 2023): "the restated value is only in the changes-in-equity statement "
                                           "('Balance, December 31, 2023 | ... | 290,209')",
}
FACT_EXTRA_COLUMNS = ["cited_filing_indexed", "note"]
SPOT_COLUMNS = ["ticker", "metric", "fiscal_year", "fiscal_period", "duration", "checked", "value", "unit",
                "searched_accession_nos", "found_on_statement_line", "miss_reason"]


class SetupError(Exception):
    """Settings, network or cache problem: exit code 2."""


@dataclass
class Corpus:
    """The parsed filings (data/sections): statement text per filing and per company, and every filing id."""
    statements_by_filing: dict[str, str]
    statements_by_ticker: dict[str, str]
    filings: frozenset[str]


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ensure_facts(tickers_ciks: list[tuple[str, str]], refresh: bool) -> None:
    """Download the company facts that are missing (or all, with refresh)."""
    todo = [(t, cik) for t, cik in tickers_ciks if refresh or not (XBRL_DIR / f"{t}.json").exists()]
    if not todo:
        return
    try:
        user_agent = get_sec_settings().SEC_USER_AGENT
    except ValidationError as err:
        raise SetupError(f"settings are incomplete; copy .env.example to .env and fill it in.\n{err}") from err
    if problem := check_user_agent(user_agent):
        raise SetupError(problem)
    client = SecClient(user_agent)
    XBRL_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, cik in todo:
        tmp = XBRL_DIR / f"{ticker}.json.part"
        try:
            tmp.write_bytes(client.get(companyfacts_url(cik)).content)
        except requests.RequestException as err:
            raise SetupError(f"could not download the company facts for {ticker}: {err}") from err
        tmp.replace(XBRL_DIR / f"{ticker}.json")
        print(f"  downloaded company facts for {ticker}")


def load_company_facts(ticker: str) -> dict:
    path = XBRL_DIR / f"{ticker}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise SetupError(f"cannot read {path} ({err}); run with --refresh") from err


def load_corpus() -> Corpus:
    by_filing: dict[str, list[str]] = defaultdict(list)
    by_ticker: dict[str, list[str]] = defaultdict(list)
    filings = set()
    path = SECTIONS_DIR / "sections.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            filings.add(d["accession_no"])
            if d["item"] in STATEMENT_ITEMS:
                by_filing[d["accession_no"]].append(d["text"])
                by_ticker[d["ticker"]].append(d["text"])
    return Corpus({k: "\n".join(v) for k, v in by_filing.items()}, {k: "\n".join(v) for k, v in by_ticker.items()},
                  frozenset(filings))


def miss_reason(fact: Fact) -> str:
    return (KNOWN_TEXT_MISSES.get((fact.ticker, fact.metric, fact.fiscal_year))
            or KNOWN_TEXT_MISSES.get((fact.ticker, fact.metric, None)) or "UNEXPLAINED")


def spot_checks(fact: Fact, corpus: Corpus) -> list[dict]:
    """The original value in the filing that first reported it or in the period's own filing (whichever are
    indexed), and a restated value in the filing it cites."""
    checks = [("original", fact.original_value, [fact.original_accession_no, fact.period_accession_no])]
    if fact.restated:
        checks.append(("restated", fact.value, [fact.accession_no]))
    out = []
    for checked, value, accessions in checks:
        indexed = [a for a in dict.fromkeys(accessions) if a in corpus.statements_by_filing]
        if not indexed:
            continue
        found = any(value_on_statement_line(value, fact.unit, fact.metric, corpus.statements_by_filing[a])
                    for a in indexed)
        out.append({"ticker": fact.ticker, "metric": fact.metric, "fiscal_year": fact.fiscal_year,
                    "fiscal_period": fact.fiscal_period, "duration": fact.duration, "checked": checked,
                    "value": value, "unit": fact.unit, "searched_accession_nos": " ".join(indexed),
                    "found_on_statement_line": found,
                    "miss_reason": "" if found else miss_reason(fact)})
    return out


def run(refresh: bool) -> int:
    scope, metrics, gaps, notes = load_scope(), load_concept_map(), load_gaps(), load_notes()
    ensure_facts([(c.ticker, c.cik) for c in scope.companies], refresh)

    periods = [(y, "FY") for y in sorted(scope.tenk_years)]
    periods += [(y, q) for y in sorted(scope.tenq_years) for q in QUARTERS if q in scope.tenq_periods]
    # years whose quarters are out of scope: their 10-Q values are read only for the plausibility check
    check_only_years = sorted(set(scope.tenk_years) - set(scope.tenq_years))
    corpus = load_corpus()
    facts, check_only, coverage, missing = [], [], [], set()
    for company in scope.companies:
        company_facts = load_company_facts(company.ticker)
        for metric in metrics.values():
            row = {"metric": metric.name, "ticker": company.ticker}
            for year, period in periods:
                # quarterly durations to tabulate: the 3M default and, for duration metrics, the year-to-date figure
                durations = [None] if period == "FY" or metric.period == "instant" else sorted(
                    {metric.quarterly_duration(period), "3M"})
                found = [f for d in durations if (f := find_fact(company_facts, company.ticker, metric, year, period,
                                                                  d, company.sector, company.cik))]
                row[f"FY{year}" + ("" if period == "FY" else period)] = "; ".join(
                    f"{f.duration or 'instant'}: {f.label}" for f in found)
                if not found:
                    missing.add((metric.name, company.ticker, year, period))
                facts += found
            if metric.name in PLAUSIBILITY_METRICS:
                check_only += [f for y in check_only_years for q in QUARTERS
                               if (f := find_fact(company_facts, company.ticker, metric, y, q, None, company.sector))]
            coverage.append(row)

    spot = [s for f in facts for s in spot_checks(f, corpus)]
    XBRL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(XBRL_DIR / "facts_table.csv",
              [asdict(f) | {"cited_filing_indexed": f.accession_no in corpus.filings,
                            "note": notes.get((f.metric, f.ticker), "")} for f in facts],
              [f.name for f in fields(Fact)] + FACT_EXTRA_COLUMNS)
    write_csv(XBRL_DIR / "coverage.csv", coverage, ["metric", "ticker"] + [
        f"FY{y}" + ("" if p == "FY" else p) for y, p in periods])
    write_csv(XBRL_DIR / "spot_check.csv", spot, SPOT_COLUMNS)

    print(f"{len(facts)} facts for {len(scope.companies)} companies x {len(metrics)} metrics "
          f"({len(periods)} periods) -> data/xbrl/facts_table.csv, coverage.csv")
    print(f"{'metric':22}" + "".join(f"{c.ticker:>6}" for c in scope.companies)
          + f"   (periods found of {len(periods)})")
    for metric in metrics:
        counts = [sum(1 for (m, t, _y, _p) in missing if m == metric and t == c.ticker) for c in scope.companies]
        print(f"{metric:22}" + "".join(f"{len(periods) - n:>6}" for n in counts))
    restated = [f for f in facts if f.restated]
    outside = [f for f in facts if f.accession_no not in corpus.filings]
    print(f"{len(restated)} facts restated by a later filing; {len(outside)} cite a filing outside the indexed "
          f"corpus (column cited_filing_indexed)")

    problems = []
    if len(metrics) < MIN_METRICS:
        problems.append(f"{len(metrics)} metrics mapped; at least {MIN_METRICS} required")
    problems += gap_problems(missing, gaps, periods)
    problems += implausible(facts + check_only)
    problems += missing_ticker_labels(metrics, corpus.statements_by_ticker)
    if spot:
        matched = sum(s["found_on_statement_line"] for s in spot)
        share = matched / len(spot)
        print(f"spot check: {matched}/{len(spot)} values ({share:.1%}) found on their line of the financial "
              f"statements -> data/xbrl/spot_check.csv")
        for s in spot:
            if not s["found_on_statement_line"]:
                print(f"  not found: {s['ticker']} {s['metric']} FY{s['fiscal_year']} {s['fiscal_period']} "
                      f"{s['duration'] or 'instant'} {s['checked']} {s['value']:,}: {s['miss_reason']}")
        if share < MIN_SPOT_CHECK_SHARE:
            problems.append(f"only {share:.1%} of values found on their statement line "
                            f"(need {MIN_SPOT_CHECK_SHARE:.0%})")
        problems += [f"spot check miss without a known reason: {s['ticker']} {s['metric']} FY{s['fiscal_year']} "
                     f"{s['fiscal_period']}" for s in spot if s["miss_reason"] == "UNEXPLAINED"]
    else:
        print("spot and label checks skipped: data/sections/sections.jsonl not found "
              "(run ingest/02_parse_sections.py)")
    for p in problems:
        print(f"CHECK FAILED: {p}")
    if not problems:
        print("OK: all checks pass (metrics mapped, coverage equals the declared gaps, plausibility, labels, "
              "spot check)")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="download all company facts again")
    args = ap.parse_args(argv)
    try:
        return run(args.refresh)
    except SetupError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
