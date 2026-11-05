"""WP3: the reported financial figures of every in-scope company and period, with coverage and spot checks.

Usage:
    python ingest/05_xbrl_facts.py                   # uses the cached company facts (downloads missing ones)
    python ingest/05_xbrl_facts.py --refresh         # download all company facts again first

Input : data/xbrl/<TICKER>.json (SEC company facts; ingest/01_download.py caches them too), config/concept_map.yaml,
        data/sections/sections.jsonl (optional, for the spot checks)
Output: data/xbrl/facts_table.csv   one row per (company, metric, fiscal period, duration) found
        data/xbrl/coverage.csv      metric x company x period: the line item found, or empty for a gap
        data/xbrl/spot_check.csv    every value searched in the financial statements of the filing that first
                                    reported it, on a line carrying the metric's words
Checks (exit 1 if one fails): at least MIN_METRICS metrics mapped; real coverage equals the gaps declared in
config/concept_map.yaml, year by year; annual values in proportion with the same year's quarterly values; at least
MIN_SPOT_CHECK_SHARE of the values found on their statement line.
Exit codes: 0 = all checks pass, 1 = a check failed, 2 = setup error. Selection rules: ingest/xbrl.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError  # noqa: E402

from core.settings import get_settings  # noqa: E402
from ingest import SECTIONS_DIR, XBRL_DIR  # noqa: E402
from ingest.edgar import SecClient, check_user_agent, companyfacts_url, load_scope  # noqa: E402
from ingest.xbrl import (  # noqa: E402
    QUARTERS,
    find_fact,
    gap_problems,
    implausible,
    load_concept_map,
    load_gaps,
    value_on_statement_line,
)

MIN_METRICS = 10                    # Jira UPGRAD-26 acceptance: at least 10 metrics mapped
MIN_SPOT_CHECK_SHARE = 0.95         # share of values that must appear on their line of the original statements
# Where the financial statements are (data/sections): 10-K Item 8, or Item 15 for filers that put them after the
# exhibits list; 10-Q Part I Item 1
STATEMENT_ITEMS = frozenset({"Item 8", "Item 15", "Part I Item 1"})
# Checked misses, with the reason (each value was confirmed by hand in the filing), so a reader knows why they miss
KNOWN_TEXT_MISSES = {
    ("PG", "rd_expense"): "the 10-K states R&D only in words, rounded to $0.1 billion (\"$2.0 billion\")",
    ("MRK", "total_assets"): "the balance sheet prints the total-assets row without a label",
    ("MRK", "eps_diluted"): "printed under an 'Assuming Dilution' heading, on the line 'Net Income | 0.14'",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ensure_facts(tickers_ciks: list[tuple[str, str]], refresh: bool) -> str | None:
    """Download the company facts that are missing (or all, with refresh). Returns an error message or None."""
    todo = [(t, cik) for t, cik in tickers_ciks if refresh or not (XBRL_DIR / f"{t}.json").exists()]
    if not todo:
        return None
    try:
        user_agent = get_settings().SEC_USER_AGENT
    except ValidationError as err:
        return f"settings are incomplete; copy .env.example to .env and fill it in.\n{err}"
    if problem := check_user_agent(user_agent):
        return problem
    client = SecClient(user_agent)
    XBRL_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, cik in todo:
        tmp = XBRL_DIR / f"{ticker}.json.part"
        tmp.write_bytes(client.get(companyfacts_url(cik)).content)
        tmp.replace(XBRL_DIR / f"{ticker}.json")
        print(f"  downloaded company facts for {ticker}")
    return None


def statement_texts() -> dict[str, str]:
    """accession number -> the financial-statement sections of that filing, for the spot checks."""
    texts: dict[str, list[str]] = defaultdict(list)
    path = SECTIONS_DIR / "sections.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d["item"] in STATEMENT_ITEMS:
                texts[d["accession_no"]].append(d["text"])
    return {k: "\n".join(v) for k, v in texts.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="download all company facts again")
    args = ap.parse_args(argv)

    scope, metrics, gaps = load_scope(), load_concept_map(), load_gaps()
    if error := ensure_facts([(c.ticker, c.cik) for c in scope.companies], args.refresh):
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    periods = [(y, "FY") for y in sorted(scope.tenk_years)]
    periods += [(y, q) for y in sorted(scope.tenq_years) for q in QUARTERS if q in scope.tenq_periods]
    texts = statement_texts()
    facts, coverage, spot, missing = [], [], [], set()
    for company in scope.companies:
        company_facts = json.loads((XBRL_DIR / f"{company.ticker}.json").read_text(encoding="utf-8"))
        for metric in metrics.values():
            row = {"metric": metric.name, "ticker": company.ticker}
            for year, period in periods:
                # quarterly durations to tabulate: the 3M default and, for duration metrics, the year-to-date figure
                durations = [None] if period == "FY" or metric.period == "instant" else sorted(
                    {metric.quarterly_duration(period), "3M"})
                found = [f for d in durations if (f := find_fact(company_facts, company.ticker, metric, year, period,
                                                                  d, company.sector, company.cik))]
                key = f"FY{year}" + ("" if period == "FY" else period)
                row[key] = "; ".join(f"{f.duration or 'instant'}: {f.label}" for f in found)
                if not found:
                    missing.add((metric.name, company.ticker, year, period))
                facts += found
                for fact in found:
                    # the original value, in the filing that first reported it (usually the period's own filing)
                    text = texts.get(fact.original_accession_no)
                    if text is not None:
                        found_on_line = value_on_statement_line(fact.original_value, fact.unit, metric.name, text)
                        spot.append({"ticker": company.ticker, "metric": metric.name, "fiscal_year": year,
                                     "fiscal_period": period, "duration": fact.duration,
                                     "original_value": fact.original_value, "unit": fact.unit,
                                     "accession_no": fact.original_accession_no,
                                     "found_on_statement_line": found_on_line,
                                     "note": "" if found_on_line else KNOWN_TEXT_MISSES.get(
                                         (company.ticker, metric.name), "UNEXPLAINED")})
            coverage.append(row)

    XBRL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(XBRL_DIR / "facts_table.csv", [asdict(f) for f in facts])
    write_csv(XBRL_DIR / "coverage.csv", coverage)
    if spot:
        write_csv(XBRL_DIR / "spot_check.csv", spot)

    print(f"{len(facts)} facts for {len(scope.companies)} companies x {len(metrics)} metrics "
          f"({len(periods)} periods) -> data/xbrl/facts_table.csv, coverage.csv")
    print(f"{'metric':22}" + "".join(f"{c.ticker:>6}" for c in scope.companies) + "   (periods found of "
          f"{len(periods)})")
    for metric in metrics:
        counts = [sum(1 for (m, t, _y, _p) in missing if m == metric and t == c.ticker) for c in scope.companies]
        print(f"{metric:22}" + "".join(f"{len(periods) - n:>6}" for n in counts))

    problems = []
    if len(metrics) < MIN_METRICS:
        problems.append(f"{len(metrics)} metrics mapped; at least {MIN_METRICS} required")
    problems += gap_problems(missing, gaps, periods)
    problems += implausible(facts)
    if spot:
        matched = sum(s["found_on_statement_line"] for s in spot)
        share = matched / len(spot)
        print(f"spot check: {matched}/{len(spot)} values ({share:.1%}) found on their line of the original financial "
              f"statements -> data/xbrl/spot_check.csv")
        for s in spot:
            if not s["found_on_statement_line"]:
                print(f"  not found: {s['ticker']} {s['metric']} FY{s['fiscal_year']} {s['fiscal_period']} "
                      f"{s['duration'] or 'instant'} {s['original_value']:,} {s['note']}".rstrip())
        if share < MIN_SPOT_CHECK_SHARE:
            problems.append(f"only {share:.1%} of values found on their statement line "
                            f"(need {MIN_SPOT_CHECK_SHARE:.0%})")
    else:
        print("spot check skipped: data/sections/sections.jsonl not found (run ingest/02_parse_sections.py)")
    for p in problems:
        print(f"CHECK FAILED: {p}")
    if not problems:
        print("OK: all checks pass (metrics mapped, coverage equals the declared gaps, plausibility, spot check)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
