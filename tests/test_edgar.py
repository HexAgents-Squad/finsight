"""ingest/edgar.py: filing selection, fiscal labels, exhibits and the polite HTTP client (offline)."""
from types import SimpleNamespace

import pytest

from ingest.edgar import (
    MIN_INTERVAL_S,
    Company,
    SecClient,
    check_user_agent,
    filing_rows,
    find_annual_report,
    fiscal_labels,
    load_scope,
    manifest_record,
    older_page_names,
    select_filings,
)

APPLE = Company("AAPL", "0000320193", "tech")
SCOPE = load_scope()


def page(*rows):
    """A submissions page in the SEC's column layout."""
    columns = ["accessionNumber", "form", "reportDate", "filingDate", "primaryDocument"]
    return {c: [r[i] for r in rows] for i, c in enumerate(columns)}


def facts(*rows):
    """Company facts with one concept whose rows carry (accn, fy, fp)."""
    return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"accn": a, "fy": fy, "fp": fp, "val": 1} for a, fy, fp in rows]}}}}}


# --- scope and settings ---------------------------------------------------------------------------------

def test_scope_matches_companies_yaml():
    assert len(SCOPE.companies) == 20
    assert SCOPE.tenk_years == {2023, 2024, 2025} and SCOPE.tenq_years == {2025}
    assert SCOPE.tenq_periods == {"Q1", "Q2", "Q3"} and SCOPE.earliest_year == 2023


@pytest.mark.parametrize("agent, ok", [
    ("HexAgents_Squad IITKGP someone@iitkgp.ac.in", True),
    ("HexAgents_Squad IITKGP your-name@example.com", False),
    ("HexAgents_Squad", False),
])
def test_user_agent_needs_a_real_contact(agent, ok):
    assert (check_user_agent(agent) is None) is ok


# --- fiscal labels --------------------------------------------------------------------------------------

def test_fiscal_labels_use_the_companys_own_year_and_majority_vote():
    labels = fiscal_labels(facts(("A1", 2025, "Q1"), ("A1", 2025, "Q1"), ("A1", 2024, "FY"), ("A2", "2024", "fy"),
                                 ("A3", None, "Q2")))
    assert labels == {"A1": (2025, "Q1"), "A2": (2024, "FY")}       # A3 has no year: no label


# --- selection ------------------------------------------------------------------------------------------

def apple_rows():
    return filing_rows({"filings": {"recent": page(
        ("K25", "10-K", "2025-09-27", "2025-10-31", "k25.htm"),
        ("Q3", "10-Q", "2025-06-28", "2025-08-01", "q3.htm"),
        ("Q2", "10-Q", "2025-03-29", "2025-05-02", "q2.htm"),
        ("Q1", "10-Q", "2024-12-28", "2025-01-31", "q1.htm"),          # calendar 2024, fiscal 2025 Q1
        ("K24A", "10-K/A", "2024-09-28", "2025-01-15", "k24a.htm"),   # amendment: never selected
        ("K24", "10-K", "2024-09-28", "2024-11-01", "k24.htm"),
        ("Q3OLD", "10-Q", "2024-06-29", "2024-08-02", "q3old.htm"),    # fiscal 2024: out of scope
        ("K23", "10-K", "2023-09-30", "2023-11-03", "k23.htm"),
        ("8K", "8-K", "2024-05-02", "2024-05-02", "8k.htm"),
    )}}, older_pages=[])


APPLE_LABELS = {"K25": (2025, "FY"), "Q3": (2025, "Q3"), "Q2": (2025, "Q2"), "Q1": (2025, "Q1"),
                "K24A": (2024, "FY"), "K24": (2024, "FY"), "Q3OLD": (2024, "Q3"), "K23": (2023, "FY")}


def test_selection_uses_fiscal_labels_and_skips_amendments():
    sel = select_filings(APPLE, apple_rows(), APPLE_LABELS, SCOPE)
    assert [(f.form, f.fiscal_year, f.fiscal_period, f.accession_no) for f in sel.filings] == [
        ("10-K", 2023, "FY", "K23"), ("10-K", 2024, "FY", "K24"), ("10-K", 2025, "FY", "K25"),
        ("10-Q", 2025, "Q1", "Q1"), ("10-Q", 2025, "Q2", "Q2"), ("10-Q", 2025, "Q3", "Q3")]
    assert sel.missing == [] and sel.unlabelled == []
    q1 = next(f for f in sel.filings if f.fiscal_period == "Q1")
    assert q1.period_end == "2024-12-28" and q1.stem == "10-Q_2025Q1_Q1"


def test_unlabelled_filings_are_reported_never_guessed_from_the_calendar():
    labels = {k: v for k, v in APPLE_LABELS.items() if k != "K24"}
    sel = select_filings(APPLE, apple_rows(), labels, SCOPE)
    assert "10-K 2024 FY" in sel.missing
    assert sel.unlabelled == ["10-K K24 (period 2024-09-28)"]
    assert all(f.accession_no != "K24" for f in sel.filings)


def test_newest_filing_wins_when_a_period_is_filed_twice():
    rows = filing_rows({"filings": {"recent": page(("NEW", "10-K", "2024-09-28", "2024-12-01", "new.htm"),
                                                   ("OLD", "10-K", "2024-09-28", "2024-11-01", "old.htm"))}}, [])
    sel = select_filings(APPLE, rows, {"NEW": (2024, "FY"), "OLD": (2024, "FY")}, SCOPE)
    assert [f.accession_no for f in sel.filings] == ["NEW"]


def test_older_pages_are_read_for_heavy_filers():
    recent = {"filings": {"recent": page(("NEW", "10-Q", "2025-03-31", "2025-05-01", "q.htm")),
                          "files": [{"name": "CIK-001.json", "filingTo": "2024-12-31"},
                                    {"name": "CIK-002.json", "filingTo": "2019-06-30"}]}}
    assert older_page_names(recent, earliest_year=2023) == ["CIK-001.json"]
    older = page(("K23", "10-K", "2023-12-31", "2024-02-16", "k.htm"))
    assert [r["accessionNumber"] for r in filing_rows(recent, [older])] == ["NEW", "K23"]


# --- exhibits and manifest ------------------------------------------------------------------------------

def item(name, size=1000):
    return {"name": name, "size": str(size)}


@pytest.mark.parametrize("items, expected", [
    ([item("main.htm"), item("ex13.htm"), item("ex21.htm")], "ex13.htm"),
    ([item("main.htm"), item("wfc-2024xex13.htm"), item("R1.htm")], "wfc-2024xex13.htm"),
    ([item("main.htm", 200_000), item("annualreport.htm", 4_000_000), item("ex99.htm", 5_000_000)],
     "annualreport.htm"),                                          # big unnamed report; other exhibits ignored
    ([item("main.htm", 2_000_000), item("other.htm", 3_000_000)], None),   # not much larger than primary
    ([item("main.htm"), item("R7.htm", 9_000_000), item("FilingSummary.htm")], None),
])
def test_find_annual_report(items, expected):
    assert find_annual_report(items, "main.htm") == expected


def test_manifest_record_has_urls_and_relative_paths():
    q1 = next(f for f in select_filings(APPLE, apple_rows(), APPLE_LABELS, SCOPE).filings if f.fiscal_period == "Q1")
    rec = manifest_record(q1, "raw/AAPL/10-Q_2025Q1_Q1.htm", "ex13.htm", "raw/AAPL/10-Q_2025Q1_Q1_EX13.htm")
    assert rec["source_url"] == "https://www.sec.gov/Archives/edgar/data/320193/Q1/q1.htm"
    assert rec["exhibit"] == "Exhibit 13" and rec["exhibit_url"].endswith("/Q1/ex13.htm")
    assert rec["fiscal_year"] == 2025 and rec["fiscal_period"] == "Q1" and "primary_document" not in rec
    plain = manifest_record(q1, "raw/x.htm", None, None)
    assert plain["exhibit"] is None and plain["exhibit_url"] is None


# --- HTTP client ----------------------------------------------------------------------------------------

class FakeSession:
    def __init__(self, *statuses):
        self.statuses, self.headers, self.urls = list(statuses), {}, []

    def get(self, url, timeout):
        self.urls.append(url)
        status = self.statuses.pop(0)

        def raise_for_status():
            raise RuntimeError(f"HTTP {status}")
        return SimpleNamespace(status_code=status, raise_for_status=raise_for_status, json=lambda: {})


class FakeClock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(round(seconds, 3))
        self.now += seconds


def test_client_identifies_itself_and_keeps_a_gap_between_requests():
    clock = FakeClock()
    client = SecClient("Team you@x.org", session=FakeSession(200, 200), sleep=clock.sleep, clock=clock.time)
    client.get("u1")
    client.get("u2")
    assert client.session.headers["User-Agent"] == "Team you@x.org"
    assert clock.slept == [MIN_INTERVAL_S] and client.requests == 2


def test_client_retries_429_and_5xx_then_gives_up_on_others():
    clock = FakeClock()
    client = SecClient("Team you@x.org", session=FakeSession(429, 503, 200), sleep=clock.sleep, clock=clock.time)
    assert client.get("u").status_code == 200 and client.requests == 3
    client = SecClient("Team you@x.org", session=FakeSession(404), sleep=clock.sleep, clock=clock.time)
    with pytest.raises(RuntimeError, match="404"):
        client.get("u")
    assert client.requests == 1
