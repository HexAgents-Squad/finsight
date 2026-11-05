"""ingest/xbrl.py: selecting reported figures from SEC company facts, concept map, gaps and spot checks (offline).

Each case uses a small synthetic company-facts structure reproducing a situation found in the real data.
"""
import importlib.util
import sys
from dataclasses import replace

import pytest

from ingest.xbrl import (
    CONCEPT_MAP_FILE,
    STATEMENT_LINE_WORDS,
    Concept,
    Fact,
    Gap,
    Metric,
    display_forms,
    find_fact,
    gap_problems,
    implausible,
    is_rounding_of,
    load_concept_map,
    load_gaps,
    load_notes,
    missing_ticker_labels,
    select_row,
    value_in_text,
    value_on_statement_line,
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


def test_unrestated_value_cites_the_filing_that_first_reported_it():
    # the FY2025 10-K repeats FY2024 unchanged: cite the FY2024 10-K (the corpus filing), with its SEC folder
    data = facts(Revenues=[annual(391_035 * M, FY24, 2024, filed="2024-11-01", accn="0000320193-24-000123"),
                           annual(391_035 * M, FY24, 2025, filed="2025-10-31", accn="0000320193-25-000079")])
    f = find_fact(data, "AAPL", metric("Revenues"), 2024, cik="0000320193")
    assert (f.accession_no, f.original_accession_no, f.restated) == ("0000320193-24-000123",) * 2 + (False,)
    assert f.url == "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/"
    assert find_fact(data, "AAPL", metric("Revenues"), 2024).url is None


def test_a_rounded_repeat_is_not_a_restatement():
    # Citi: a later 10-K repeats FY2023 total assets 2,411,834 as 2,412,000 (a summary table); a real change is kept
    rounded = facts(Assets=[row(2_411_834 * M, "2023-12-31", 2023, "FY", filed="2024-02-23", accn="ORIG"),
                            row(2_412_000 * M, "2023-12-31", 2024, "FY", filed="2025-02-21", accn="LATER")])
    f = find_fact(rounded, "C", metric("Assets", period="instant"), 2023)
    assert (f.value, f.restated, f.accession_no) == (2_411_834, False, "ORIG")
    changed = facts(Assets=[row(2_411_834 * M, "2023-12-31", 2023, "FY", filed="2024-02-23", accn="ORIG"),
                            row(2_411_900 * M, "2023-12-31", 2024, "FY", filed="2025-02-21", accn="LATER")])
    f = find_fact(changed, "C", metric("Assets", period="instant"), 2023)
    assert (f.value, f.restated, f.original_accession_no) == (2_411_900, True, "ORIG")


def test_restated_then_reverted_is_the_original_value():
    rows = [annual(100 * M, FY24, 2024, filed="2024-11-01", accn="A"), annual(105 * M, FY24, 2025, filed="2025-11-01",
            accn="B"), annual(100 * M, FY24, 2026, filed="2026-11-01", accn="C")]
    row, first = select_row(rows, FY24)
    assert (row["accn"], first["accn"]) == ("A", "A")                  # latest filed value = the original


def test_a_rounded_repeat_of_a_restated_value_keeps_the_restated_value():
    rows = [row(2_411_834 * M, "2023-12-31", 2023, "FY", filed="2024-02-23", accn="A"),
            row(2_420_123 * M, "2023-12-31", 2024, "FY", filed="2025-02-21", accn="B"),
            row(2_420_000 * M, "2023-12-31", 2025, "FY", filed="2026-02-20", accn="C")]
    f = find_fact(facts(Assets=rows), "C", metric("Assets", period="instant"), 2023)
    assert (f.value, f.accession_no, f.restated, f.original_value) == (2_420_123, "B", True, 2_411_834)


def test_a_restated_value_cites_the_earliest_filing_that_carries_it():
    # NVIDIA FY2024 EPS: split-adjusted 1.19 first appears in the FY2025 10-K (indexed) and is repeated in the
    # FY2026 10-K (not indexed): cite the FY2025 10-K
    data = facts(EarningsPerSharePerShare=[annual(11.93, FY24, 2024, filed="2024-02-21", accn="K24"),
                                           annual(1.19, FY24, 2025, filed="2025-02-26", accn="K25"),
                                           annual(1.19, FY24, 2026, filed="2026-02-25", accn="K26")])
    f = find_fact(data, "NVDA", metric("EarningsPerSharePerShare", kind="per_share"), 2024)
    assert (f.value, f.accession_no, f.original_accession_no, f.period_accession_no) == (1.19, "K25", "K24", "K24")


@pytest.mark.parametrize("value, exact, rounding", [
    (2_412_000, 2_411_834, True),
    (2_411_800, 2_411_834, True),
    (2_000_000, 2_411_834, False),       # keeps fewer than 3 significant digits: a different value
    (2_411_834, 2_411_834, False),       # unchanged
    (6.1, 6.08, False),                  # per share: a split or restatement, never a rounding
    (2_410_000_000_000, 2_411_834_000_000, True),     # a bank total rounded to $10 billion
])
def test_is_rounding_of(value, exact, rounding):
    assert is_rounding_of(value, exact) is rounding


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


def test_inputs_are_normalised():
    data = facts(Revenues=Q2)
    f = find_fact(data, " aapl ", metric("Revenues"), 2025, " q2 ", " 6m ")
    assert (f.ticker, f.fiscal_period, f.duration, f.value) == ("AAPL", "Q2", "6M", 219_659)


@pytest.mark.parametrize("year", ["2025", 2025.0, True])
def test_fiscal_year_must_be_an_int(year):
    # a string would silently match nothing and read as "not reported"
    with pytest.raises(TypeError, match="fiscal_year must be an int"):
        find_fact(facts(Revenues=Q2), "AAPL", metric("Revenues"), year, "Q2")


@pytest.mark.parametrize("fiscal_period, duration", [("FY", "3M"), ("Q1", "6M"), ("Q2", "9M"), ("Q3", "12M"),
                                                     ("Q2", "4M")])
def test_a_duration_the_period_cannot_have_is_an_error(fiscal_period, duration):
    # FY + 3M would be a standalone Q4 (never derived); Q3 + 12M the prior fiscal year
    with pytest.raises(ValueError, match="not valid"):
        find_fact(facts(Revenues=Q2), "AAPL", metric("Revenues"), 2025, fiscal_period, duration)


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


def test_rd_uses_the_income_statement_line_before_the_in_process_line():
    # JNJ tags its separate acquired in-process R&D line as ResearchAndDevelopmentExpense (1,841) and the
    # income-statement R&D line as the "excluding acquired in-process cost" concept (17,232)
    data = facts(ResearchAndDevelopmentExpense=[annual(1_841 * M, FY24, 2024)],
                 ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost=[annual(17_232 * M, FY24, 2024)])
    assert find_fact(data, "JNJ", load_concept_map()["rd_expense"], 2024).value == 17_232


def test_bank_cash_is_the_cash_and_equivalents_total_where_the_balance_sheet_has_one():
    # BAC prints "Cash and cash equivalents | 290,114" (tagged with the restricted-cash concept) and, as a part of
    # it, "Cash and due from banks | 26,003"; JPM has only the latter line
    cash = load_concept_map()["cash"]
    rows = dict(CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents=[row(290_114 * M, "2024-12-31", 2024,
                                                                                   "FY")],
                CashAndDueFromBanks=[row(26_003 * M, "2024-12-31", 2024, "FY")])
    bac = find_fact(facts(**rows), "BAC", cash, 2024, sector="financials")
    assert (bac.value, bac.label) == (290_114, "cash and cash equivalents")
    jpm = find_fact(facts(**rows), "JPM", cash, 2024, sector="financials")
    assert (jpm.value, jpm.label) == (26_003, "cash and due from banks")


def test_company_labels_override_the_concept_label():
    m = Metric("operating_income", "duration", "currency", (Concept("OperatingIncomeLoss", "operating income"),),
               ticker_labels=(("ABBV", "operating earnings"),))
    data = facts(OperatingIncomeLoss=[annual(9_137 * M, FY24, 2024)])
    assert find_fact(data, "ABBV", m, 2024).label == "operating earnings"
    assert find_fact(data, "AAPL", m, 2024).label == "operating income"
    assert missing_ticker_labels({"operating_income": m}, {"ABBV": "Operating earnings | 9,137"}) == []
    assert missing_ticker_labels({"operating_income": m}, {"ABBV": "Operating income | 9,137"}) == [
        "label not in ABBV's statements: operating_income 'operating earnings'"]


def test_a_concept_reported_only_for_the_comparative_column_is_not_this_period():
    # the FY2024 10-K reports the fallback concept only for FY2023 (its comparative column): pass 1 must not take
    # FY2023 as FY2024; the anchor fixes FY2024 and the concept has no FY2024 value
    data = facts(NetIncomeLoss=[annual(10 * M, FY24, 2024)], Other=[annual(7 * M, FY23, 2024)])
    assert find_fact(data, "X", metric("Other"), 2024) is None


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
    # cited: the FY2024 10-K that carries the mapped concept; the period's own filing is the FY2023 10-K
    assert (f.accession_no, f.period_accession_no) == ("A-2024-02-21", "A-2023-02-24")


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
    assert set(STATEMENT_LINE_WORDS) == PLAN_METRICS                     # every metric can be spot-checked


def test_concept_map_rejects_unknown_keys(tmp_path):
    bad = CONCEPT_MAP_FILE.read_text(encoding="utf-8").replace(
        'label: "purchases of property, equipment and other productive assets"',
        "label: purchases of property, equipment and other productive assets")
    path = tmp_path / "concept_map.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_concept_map(path)


@pytest.mark.parametrize("old, new, error", [
    ("sectors: [financials]}   # JPM", "sectors: financials}   # JPM", "must be a non-empty list"),
    ("      - {concept: GrossProfit, label: gross profit}", "      - {concept: GrossProfit}", "missing key"),
    ("    tickers: [GS]\n", "    tickers: [GS]\n    period: [FY]\n", "unknown key"),
    ("    periods: [Q1, Q2, Q3]", "    periods: [Q1, Q4]", "unknown period"),
    ("  capex:\n    tickers: [JPM", "  capx:\n    tickers: [JPM", "unknown metric"),
])
def test_concept_map_is_validated_strictly(tmp_path, old, new, error):
    text = CONCEPT_MAP_FILE.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path = tmp_path / "concept_map.yaml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_concept_map(path)
        load_gaps(path)


def test_notes_carry_caveats():
    assert "approximate" in load_notes()[("rd_expense", "PG")]


def test_gap_problems_are_checked_year_by_year():
    gaps = [Gap("operating_income", frozenset({"JPM"}), frozenset({"FY", "Q1"}), "no line")]
    periods = [(2024, "FY"), (2025, "FY"), (2025, "Q1")]
    every = {("operating_income", "JPM", 2024, "FY"), ("operating_income", "JPM", 2025, "FY"),
             ("operating_income", "JPM", 2025, "Q1")}
    assert gap_problems(every, gaps, periods) == []
    assert gap_problems(every | {("revenue", "AAPL", 2024, "FY")}, gaps, periods) == [
        "undeclared gap: revenue not found for AAPL FY2024 FY"]
    # reported for one year only: the gap is stale for that year, not hidden by the other year's gap
    assert gap_problems(every - {("operating_income", "JPM", 2024, "FY")}, gaps, periods) == [
        "stale gap entry: operating_income for JPM FY2024 FY is now reported"]


def fact(metric_name, period, duration, value, concept="C", ticker="JNJ", year=2025, restated=False):
    return Fact(ticker, metric_name, year, period, duration, value, "USD millions", concept, "l", None, "e", "a",
                "10-K", "f", None, restated, value, "a", "a")


def test_implausible_annual_values_are_reported():
    # JNJ before the R&D fix: FY 109 (the in-process line) against about 3,600 a quarter
    wrong = [fact("rd_expense", "FY", "12M", 109, "InProcess"), fact("rd_expense", "Q1", "3M", 3_225)]
    assert implausible(wrong) == ["implausible: JNJ rd_expense FY2025 = 109 but Q1 3M = 3,225 (InProcess vs C)"]
    right = [fact("rd_expense", "FY", "12M", 14_665), fact("rd_expense", "Q1", "3M", 3_225),
             fact("capex", "FY", "12M", 10), fact("capex", "Q3", "9M", 900),     # capex: lumpy, not checked
             fact("revenue", "FY", "12M", 60_922), fact("revenue", "Q1", "3M", 7_192)]   # NVDA FY2024 growth
    assert implausible(right) == []


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


@pytest.mark.parametrize("metric_name, value, unit, line, found", [
    ("revenue", 391_035.0, "USD millions", "Total net sales | 391,035 | 383,285", True),
    ("revenue", 391_035.0, "USD millions", "Total assets | 391,035", False),   # the value, on another line item
    ("operating_cash_flow", 118_548.0, "USD millions", "Net cash from operations | 118,548 | 87,582", True),
    ("operating_income", 9_137.0, "USD millions", "Operating earnings | 9,137 | 12,757", True),
    ("eps_diluted", 6.74, "USD per share", "Earnings per Common Share Assuming Dilution | 6.74 | 0.14", True),
])
def test_value_on_statement_line(metric_name, value, unit, line, found):
    assert value_on_statement_line(value, unit, metric_name, "Revenue\n391,035\n" + line) is found


def test_display_forms():
    assert display_forms(1_234.5, "USD millions") == ["1,235", "1,234.5"]   # half up, like the statements
    assert display_forms(6.081, "USD per share") == ["6.08"] and display_forms(0.125, "USD per share") == ["0.13"]


def load_facts_script():
    spec = importlib.util.spec_from_file_location("xbrl_facts_script", CONCEPT_MAP_FILE.parents[1] / "ingest"
                                                  / "05_xbrl_facts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module        # @dataclass looks the module up
    spec.loader.exec_module(module)
    return module


def test_known_misses_are_year_specific_where_declared():
    script = load_facts_script()
    assert "Assuming Dilution" in script.miss_reason(fact("eps_diluted", "FY", "12M", 0.14, ticker="MRK", year=2023))
    assert script.miss_reason(fact("eps_diluted", "FY", "12M", 6.74, ticker="MRK", year=2024)) == "UNEXPLAINED"
    assert "without a label" in script.miss_reason(fact("total_assets", "Q1", None, 1, ticker="MRK", year=2025))


def test_spot_check_searches_original_and_restated_values():
    script = load_facts_script()
    corpus = script.Corpus({"a": "Total revenue, net of interest expense | 105,856 | 101,887"}, {}, frozenset({"a"}))
    f = fact("revenue", "FY", "12M", 105_856, ticker="BAC", restated=True)
    f = replace(f, original_value=101_887)
    checks = script.spot_checks(f, corpus)
    assert [(c["checked"], c["value"], c["found_on_statement_line"]) for c in checks] == [
        ("original", 101_887, True), ("restated", 105_856, True)]


def test_a_corrupt_cache_file_is_a_setup_error(tmp_path, monkeypatch, capsys):
    script = load_facts_script()
    monkeypatch.setattr(script, "XBRL_DIR", tmp_path)
    for company in script.load_scope().companies:
        (tmp_path / f"{company.ticker}.json").write_text("{not json", encoding="utf-8")
    assert script.main([]) == 2
    assert "run with --refresh" in capsys.readouterr().err
