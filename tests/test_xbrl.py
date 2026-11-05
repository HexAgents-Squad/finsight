"""ingest/xbrl.py: selecting reported figures from SEC company facts, concept map, gaps and spot checks (offline).

Each case uses a small synthetic company-facts structure reproducing a situation found in the real data.
"""
import pytest

from ingest.xbrl import (
    CONCEPT_MAP_FILE,
    Concept,
    Gap,
    Metric,
    display_forms,
    find_fact,
    gap_problems,
    load_concept_map,
    load_gaps,
    value_in_text,
)

M = 1_000_000


def row(val, end, fy, fp, form="10-K", start=None, filed="2024-11-01", accn=None):
    r = {"val": val, "end": end, "fy": fy, "fp": fp, "form": form, "filed": filed, "accn": accn or f"A-{filed}"}
    if start:
        r["start"] = start
    return r


def facts(**concepts):
    """{concept: [rows]} -> company facts (USD; per-share concepts end in 'PerShare')."""
    return {"facts": {"us-gaap": {c: {"units": {"USD/shares" if c.endswith("PerShare") else "USD": rows}}
                                  for c, rows in concepts.items()}}}


def metric(*concepts, period="duration", kind="currency", ytd=False):
    return Metric("m", period, kind, tuple(c if isinstance(c, Concept) else Concept(c, c.lower()) for c in concepts),
                  ytd)


FY24 = ("2023-10-01", "2024-09-28")             # a 52-week fiscal year (Apple-like)
FY23 = ("2022-09-25", "2023-09-30")


def annual(val, period, fy, **kw):
    return row(val, period[1], fy, "FY", start=period[0], **kw)


# --- annual selection --------------------------------------------------------------------------------------

def test_annual_value_is_the_latest_period_in_the_labelled_10k():
    # the FY2024 10-K also repeats FY2023 under fy=2024; quarterly rows inside a 10-K are not annual
    data = facts(Revenues=[annual(391_035 * M, FY24, 2024), annual(383_285 * M, FY23, 2024),
                           row(94_930 * M, "2024-09-28", 2024, "FY", start="2024-06-30")])
    f = find_fact(data, "AAPL", metric("Revenues"), 2024)
    assert (f.value, f.unit, f.duration) == (391_035, "USD millions", "12M")
    assert (f.period_start, f.period_end) == FY24 and not f.restated


def test_restated_value_uses_the_latest_filing_and_keeps_the_original():
    # BAC re-presented revenue in a later 10-K: latest filed value, flagged, original kept (spec §7.7)
    data = facts(Revenues=[annual(101_887 * M, FY24, 2024, filed="2025-02-25"),
                           annual(105_856 * M, FY24, 2025, filed="2026-02-24", accn="LATER")])
    f = find_fact(data, "BAC", metric("Revenues"), 2024)
    assert (f.value, f.original_value, f.restated, f.accession_no) == (105_856, 101_887, True, "LATER")


def test_amendment_is_the_most_recently_filed_value():
    data = facts(NetIncomeLoss=[annual(10 * M, FY24, 2024, filed="2024-11-01"),
                                annual(12 * M, FY24, 2024, form="10-K/A", filed="2025-01-15")])
    f = find_fact(data, "X", metric("NetIncomeLoss"), 2024)
    assert (f.value, f.form, f.restated) == (12, "10-K/A", True)


def test_not_reported_is_none_never_a_substitute():
    data = facts(Revenues=[annual(1 * M, FY24, 2024)])
    assert find_fact(data, "JPM", metric("OperatingIncomeLoss"), 2024) is None
    assert find_fact(data, "JPM", metric("Revenues"), 2022) is None


# --- quarters and instants -----------------------------------------------------------------------------------

Q2 = [row(95_359 * M, "2025-03-29", 2025, "Q2", "10-Q", start="2024-12-29"),            # 3M
      row(219_659 * M, "2025-03-29", 2025, "Q2", "10-Q", start="2024-09-29"),           # 6M year to date
      row(90_753 * M, "2024-03-30", 2025, "Q2", "10-Q", start="2023-12-31")]            # prior-year quarter


def test_quarter_defaults_to_three_months_and_year_to_date_on_request():
    data = facts(Revenues=Q2)
    assert find_fact(data, "AAPL", metric("Revenues"), 2025, "Q2").value == 95_359
    six = find_fact(data, "AAPL", metric("Revenues"), 2025, "q2", duration="6M")
    assert (six.value, six.duration, six.fiscal_period) == (219_659, "6M", "Q2")
    with pytest.raises(ValueError, match="Q4 is never derived"):
        find_fact(data, "AAPL", metric("Revenues"), 2025, "Q4")


def test_cash_flow_metrics_report_year_to_date_in_10qs():
    m = metric("NetCashProvidedByUsedInOperatingActivities", ytd=True)
    assert [m.quarterly_duration(q) for q in ("Q1", "Q2", "Q3")] == ["3M", "6M", "9M"]
    assert metric("Revenues").quarterly_duration("Q2") == "3M"


def test_twelve_week_quarters_fit_the_year_to_date_windows():
    # Costco: 24- and 36-week year-to-date periods (168 and 252 days)
    data = facts(OCF=[row(6_008 * M, "2025-02-16", 2025, "Q2", "10-Q", start="2024-09-02"),
                      row(9_468 * M, "2025-05-11", 2025, "Q3", "10-Q", start="2024-09-02")])
    m = metric("OCF", ytd=True)
    assert find_fact(data, "COST", m, 2025, "Q2", duration="6M").value == 6_008
    assert find_fact(data, "COST", m, 2025, "Q3", duration="9M").value == 9_468


def test_balance_sheet_values_are_instants():
    data = facts(Assets=[row(364_980 * M, "2024-09-28", 2024, "FY"), row(352_583 * M, "2023-09-30", 2024, "FY")])
    f = find_fact(data, "AAPL", metric("Assets", period="instant"), 2024, duration="12M")
    assert (f.value, f.duration, f.period_start) == (364_980, None, None)


def test_per_share_values_are_not_scaled():
    data = facts(EarningsPerSharePerShare=[annual(6.08, FY24, 2024)])
    f = find_fact(data, "AAPL", metric("EarningsPerSharePerShare", kind="per_share"), 2024)
    assert (f.value, f.unit) == (6.08, "USD per share")


# --- concept rules -------------------------------------------------------------------------------------------

def test_concepts_are_tried_in_order():
    # Walmart tags both: "Revenues" (incl. membership income) must win over contract revenue (net sales)
    data = facts(Revenues=[annual(648_125 * M, FY24, 2024)],
                 RevenueFromContractWithCustomerExcludingAssessedTax=[annual(642_637 * M, FY24, 2024)])
    m = metric("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")
    assert find_fact(data, "WMT", m, 2024).value == 648_125


def test_sector_specific_concept():
    data = facts(LongTermDebt=[row(401 * M, "2024-12-31", 2024, "FY")])
    m = metric(Concept("LongTermDebt", "long-term debt", sectors=("financials",)), period="instant")
    assert find_fact(data, "JPM", m, 2024, sector="financials").value == 401
    assert find_fact(data, "AAPL", m, 2024, sector="tech") is None


def test_unless_reported_checks_the_same_period_only():
    # JNJ: equity incl. minority interest equals equity only when no minority interest is reported for that date
    m = metric(Concept("EquityIncl", "total shareholders' equity", unless_reported=("MinorityInterest",)),
               period="instant")
    no_nci = facts(EquityIncl=[row(71 * M, "2024-12-29", 2024, "FY")],
                   MinorityInterest=[row(5 * M, "2020-12-31", 2020, "FY")])           # another period only
    assert find_fact(no_nci, "JNJ", m, 2024).value == 71
    with_nci = facts(EquityIncl=[row(98 * M, "2024-12-31", 2024, "FY")],
                     MinorityInterest=[row(5 * M, "2024-12-31", 2024, "FY")])
    assert find_fact(with_nci, "UNH", m, 2024) is None


def test_period_anchor_finds_a_value_tagged_differently_in_its_own_filing():
    # NVIDIA: FY2023 capex is under the mapped concept only in the FY2024 10-K (comparative column)
    fy23 = ("2022-01-31", "2023-01-29")
    data = facts(NetIncomeLoss=[annual(4_368 * M, fy23, 2023, filed="2023-02-24")],
                 Capex=[annual(1_833 * M, fy23, 2024, filed="2024-02-21")])
    f = find_fact(data, "NVDA", metric("Capex"), 2023)
    assert (f.value, f.period_end) == (1_833, "2023-01-29")


def test_a_concept_used_by_the_original_filing_wins_over_the_anchor():
    # Google FY2023 long-term debt: the FY2023 10-K used the second concept (13,253, later restated to 11,870);
    # the first concept appears only in the FY2024 10-K and must not hide that
    data = facts(Assets=[row(402 * M, "2023-12-31", 2023, "FY", filed="2024-01-31")],
                 Noncurrent=[row(11_870 * M, "2023-12-31", 2024, "FY", filed="2025-02-05")],
                 WithLeases=[row(13_253 * M, "2023-12-31", 2023, "FY", filed="2024-01-31"),
                             row(11_870 * M, "2023-12-31", 2024, "FY", filed="2025-02-05")])
    f = find_fact(data, "GOOGL", metric("Noncurrent", "WithLeases", period="instant"), 2023)
    assert (f.concept, f.value, f.original_value, f.restated) == ("WithLeases", 11_870, 13_253, True)


# --- concept map, gaps and spot checks -------------------------------------------------------------------------

PLAN_METRICS = {"revenue", "operating_income", "net_income", "gross_profit", "rd_expense", "capex",
                "operating_cash_flow", "eps_diluted", "total_assets", "total_liabilities", "stockholders_equity",
                "cash", "long_term_debt"}


def test_concept_map_covers_the_plan_metrics():
    metrics = load_concept_map()
    assert set(metrics) == PLAN_METRICS                                   # week0_plan WP3 step 3
    assert all(c.label for m in metrics.values() for c in m.concepts)
    assert metrics["eps_diluted"].kind == "per_share" and metrics["cash"].period == "instant"
    assert metrics["capex"].quarterly_ytd_only and not metrics["revenue"].quarterly_ytd_only
    gaps = load_gaps()
    assert {g.metric for g in gaps} <= PLAN_METRICS


def test_concept_map_rejects_unknown_keys(tmp_path):
    bad = CONCEPT_MAP_FILE.read_text(encoding="utf-8").replace(
        'label: "purchases of property, equipment and other productive assets"',
        "label: purchases of property, equipment and other productive assets")
    path = tmp_path / "concept_map.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_concept_map(path)


def test_gap_problems():
    gaps = [Gap("operating_income", frozenset({"JPM"}), frozenset({"FY", "Q1"}), "no line")]
    assert gap_problems({("operating_income", "JPM", "FY"), ("operating_income", "JPM", "Q1")}, gaps) == []
    assert gap_problems({("operating_income", "JPM", "FY"), ("operating_income", "JPM", "Q1"),
                         ("revenue", "AAPL", "FY")}, gaps) == ["undeclared gap: revenue not found for AAPL FY"]
    assert gap_problems({("operating_income", "JPM", "FY")}, gaps) == [
        "stale gap entry: operating_income for JPM Q1 is now reported"]


@pytest.mark.parametrize("value, unit, text, found", [
    (391_035.0, "USD millions", "Total net sales | 391,035 | 383,285", True),
    (391_035.0, "USD millions", "Total | 1,391,035", False),
    (391_035.0, "USD millions", "391,035,000", False),
    (-1_234.0, "USD millions", "Net loss | (1,234)", True),
    (6.08, "USD per share", "Diluted | $ 6.08 | $ 6.13", True),
    (6.08, "USD per share", "16.08", False),
])
def test_value_in_text(value, unit, text, found):
    assert value_in_text(value, unit, text) is found


def test_display_forms():
    assert display_forms(1_234.5, "USD millions") == ["1,235", "1,234.5"]   # half up, like the statements
    assert display_forms(6.081, "USD per share") == ["6.08"] and display_forms(0.125, "USD per share") == ["0.13"]
