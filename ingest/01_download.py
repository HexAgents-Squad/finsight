"""Step 1 of the corpus: download the in-scope 10-K/10-Q filings from SEC EDGAR and write the manifest.

Usage:
    python ingest/01_download.py                      # all companies in config/companies.yaml
    python ingest/01_download.py --tickers AAPL JPM   # a subset
    python ingest/01_download.py --dry-run            # show the selection, download no filings

Output (git-ignored; files already on disk are not downloaded again):
    data/xbrl/<TICKER>.json                    XBRL company facts (source of the fiscal-year labels; shared with WP3)
    data/raw/<TICKER>/submissions.json         filing index (+ older pages for heavy filers)
    data/raw/<TICKER>/<form>_<year><period>_<accession>.htm        primary document
    data/raw/<TICKER>/<form>_<year><period>_<accession>_EX13.htm   annual-report exhibit, if the filing has one
    data/raw/manifest.jsonl                    one line per selected filing (a --tickers run replaces only
                                               those companies' lines)

Exit codes: 0 = every expected filing found, 1 = some missing (see PROBLEM lines), 2 = setup error.
SEC fair-access policy: SEC_USER_AGENT in .env must name you and give a contact email; < 10 requests/second.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError  # noqa: E402

from core.settings import get_settings  # noqa: E402
from ingest import DATA_DIR, MANIFEST_FILE, RAW_DIR, XBRL_DIR  # noqa: E402
from ingest.edgar import (  # noqa: E402
    Filing,
    SecClient,
    check_user_agent,
    companyfacts_url,
    filing_folder_url,
    filing_rows,
    find_annual_report,
    fiscal_labels,
    load_scope,
    manifest_record,
    older_page_names,
    select_filings,
    submissions_url,
)


def download(client: SecClient, url: str, path: Path) -> None:
    """Fetch `url` into `path` unless it is already there (written via a temp file, so an interrupted run
    never leaves a truncated file that later runs would trust)."""
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(client.get(url).content)
    tmp.replace(path)


def annual_report_exhibit(client: SecClient, filing: Filing, folder: Path) -> str | None:
    """Name of the filing's annual-report exhibit, or None. The answer (also "none") is cached in a small
    .exhibit file, so re-runs do not fetch every filing index again."""
    cache = folder / f"{filing.stem}.exhibit"
    if cache.exists():
        return cache.read_text(encoding="utf-8").strip() or None
    index = client.json(f"{filing_folder_url(filing.cik, filing.accession_no)}/index.json")
    name = find_annual_report(index.get("directory", {}).get("item", []), filing.primary_document)
    cache.write_text(name or "", encoding="utf-8")
    return name


def write_manifest(records: list[dict], replaced_tickers: set[str]) -> int:
    """Rewrite the manifest: this run's records plus earlier records of tickers not in this run."""
    kept = []
    if MANIFEST_FILE.exists():
        kept = [r for r in map(json.loads, MANIFEST_FILE.read_text(encoding="utf-8").splitlines())
                if r["ticker"] not in replaced_tickers]
    rows = sorted(kept + records, key=lambda r: (r["ticker"], r["form"], r["fiscal_year"], r["fiscal_period"]))
    MANIFEST_FILE.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="*", help="only these tickers (default: all in config/companies.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="print the selection; download no filing documents")
    args = ap.parse_args(argv)

    try:
        user_agent = get_settings().SEC_USER_AGENT
    except ValidationError as err:
        print(f"ERROR: settings are incomplete; copy .env.example to .env and fill it in.\n{err}", file=sys.stderr)
        return 2
    if problem := check_user_agent(user_agent):
        print(f"ERROR: {problem}", file=sys.stderr)
        return 2

    scope = load_scope()
    wanted = {t.upper() for t in args.tickers or []}
    unknown = wanted - {c.ticker for c in scope.companies}
    if unknown:
        print(f"ERROR: not in config/companies.yaml: {sorted(unknown)}", file=sys.stderr)
        return 2
    companies = [c for c in scope.companies if not wanted or c.ticker in wanted]

    client = SecClient(user_agent)
    XBRL_DIR.mkdir(parents=True, exist_ok=True)
    records, problems = [], []
    for company in companies:
        folder = RAW_DIR / company.ticker
        folder.mkdir(parents=True, exist_ok=True)
        facts_path = XBRL_DIR / f"{company.ticker}.json"
        download(client, companyfacts_url(company.cik), facts_path)
        labels = fiscal_labels(json.loads(facts_path.read_text(encoding="utf-8")))

        submissions = client.json(submissions_url(company.cik))
        older = [client.json(f"https://data.sec.gov/submissions/{name}")
                 for name in older_page_names(submissions, scope.earliest_year)]
        (folder / "submissions.json").write_text(json.dumps({"recent": submissions, "older_pages": older}),
                                                 encoding="utf-8")
        selection = select_filings(company, filing_rows(submissions, older), labels, scope)
        problems += [f"{company.ticker}: missing {m}" for m in selection.missing]
        if selection.missing:          # unlabelled filings only matter when they could be the missing ones
            problems += [f"{company.ticker}: (hint) no XBRL fiscal label for {u}" for u in selection.unlabelled]

        for filing in selection.filings:
            path = folder / f"{filing.stem}.htm"
            exhibit_name = exhibit_path = None
            if not args.dry_run:
                download(client, filing.source_url, path)
                exhibit_name = annual_report_exhibit(client, filing, folder)
                if exhibit_name:
                    exhibit_file = folder / f"{filing.stem}_EX13.htm"
                    download(client, f"{filing_folder_url(filing.cik, filing.accession_no)}/{exhibit_name}",
                             exhibit_file)
                    exhibit_path = exhibit_file.relative_to(DATA_DIR).as_posix()
            records.append(manifest_record(filing, path.relative_to(DATA_DIR).as_posix(), exhibit_name, exhibit_path))
            print(f"  {company.ticker:5} {filing.form} FY{filing.fiscal_year} {filing.fiscal_period:2}  "
                  f"period {filing.period_end}  filed {filing.filing_date}"
                  f"{'  + ' + exhibit_name if exhibit_name else ''}")

    if not args.dry_run:
        total = write_manifest(records, {c.ticker for c in companies})
        print(f"\nmanifest: {len(records)} filings from this run, {total} in total "
              f"({MANIFEST_FILE.relative_to(DATA_DIR.parent).as_posix()})")
    print(f"{client.requests} requests to sec.gov")
    for p in problems:
        print(f"PROBLEM: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
