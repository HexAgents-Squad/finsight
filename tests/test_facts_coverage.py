"""W3-04 (ADR-040): the facts tool's four new metrics against the concept map and, where data/xbrl is present, the real
SEC company facts (skipped without it, as in CI).

deposits and noninterest_expense are the banks' lines; current_assets and current_liabilities are the classified
balance sheet's totals, which the banks do not present. Each is one us-gaap concept with the same meaning everywhere.
"""
import pytest

from ingest import XBRL_DIR
from ingest.edgar import load_scope
from ingest.xbrl import STATEMENT_LINE_WORDS, find_fact, load_concept_map, load_gaps
from tools.financial_facts import METRIC_HELP, FactStore, Metric, get_financial_fact

NEW = {"deposits": ("Deposits", "instant"), "noninterest_expense": ("NoninterestExpense", "duration"),
       "current_assets": ("AssetsCurrent", "instant"), "current_liabilities": ("LiabilitiesCurrent", "instant")}
BANKS = frozenset({"JPM", "BAC", "WFC", "C", "GS"})
TICKERS = frozenset(c.ticker for c in load_scope().companies)
HAS_DATA = all((XBRL_DIR / f"{t}.json").exists() for t in sorted(TICKERS))
needs_data = pytest.mark.skipif(not HAS_DATA, reason="needs data/xbrl (the SEC company facts)")


# ----------------------------------------------------------------------------------------------- the concept map
@pytest.mark.parametrize("name", sorted(NEW))
def test_each_new_metric_is_one_concept_with_its_period(name):
    spec = load_concept_map()[name]
    concept, period = NEW[name]
    assert [c.name for c in spec.concepts] == [concept] and spec.period == period
    assert name in Metric.__args__ and name in METRIC_HELP and name in STATEMENT_LINE_WORDS


def test_the_new_metrics_gaps_are_exactly_the_companies_without_the_line():
    gaps = {g.metric: g for g in load_gaps()}
    for name in ("deposits", "noninterest_expense"):
        assert gaps[name].tickers == TICKERS - BANKS and gaps[name].reason
    for name in ("current_assets", "current_liabilities"):
        assert gaps[name].tickers == BANKS and "unclassified" in gaps[name].reason


def test_the_bank_lines_are_sector_restricted_and_named_as_each_bank_names_them():
    metrics = load_concept_map()
    assert all(c.sectors == ("financials",) for n in ("deposits", "noninterest_expense") for c in metrics[n].concepts)
    spec = metrics["noninterest_expense"]
    assert spec.label_for("C", spec.concepts[0]) == "total operating expenses"
    assert spec.label_for("WFC", spec.concepts[0]) == "total noninterest expense"
    row = {"val": 5.0e9, "end": "2024-12-31", "fy": 2024, "fp": "FY", "form": "10-K", "filed": "2025-02-01",
           "accn": "A-1"}
    data = {"facts": {"us-gaap": {"Deposits": {"units": {"USD": [row]}}}}}
    assert find_fact(data, "BAC", metrics["deposits"], 2024, sector="financials").value == 5000
    assert find_fact(data, "AAPL", metrics["deposits"], 2024, sector="tech") is None


# ----------------------------------------------------------------------------------------------- the real facts
@needs_data
@pytest.mark.parametrize("ticker, metric, year, period, value, label", [
    ("BAC", "deposits", 2023, "FY", 1923827.0, "total deposits"),
    ("BAC", "deposits", 2025, "Q2", 2011613.0, "total deposits"),
    ("WFC", "noninterest_expense", 2024, "FY", 54598.0, "total noninterest expense"),
    ("C", "noninterest_expense", 2024, "FY", 53567.0, "total operating expenses"),
    ("BAC", "noninterest_expense", 2025, "Q1", 17770.0, "total noninterest expense"),
    ("PFE", "current_assets", 2023, "FY", 43333.0, "total current assets"),
    ("PFE", "current_liabilities", 2023, "FY", 47794.0, "total current liabilities"),
    ("AAPL", "current_assets", 2024, "FY", 152987.0, "total current assets"),
    ("AAPL", "current_liabilities", 2024, "FY", 176392.0, "total current liabilities"),
])
def test_the_tool_returns_the_statement_figure(ticker, metric, year, period, value, label):
    out = get_financial_fact(ticker, metric, year, period, store=FactStore())
    assert (out["value"], out["unit"], out["label"]) == (value, "USD millions", label)
    assert out["concept"] == f"us-gaap:{NEW[metric][0]}"
    assert out["duration"] == ({"FY": "12M", "Q1": "3M"}[period] if NEW[metric][1] == "duration" else None)


@needs_data
def test_every_company_and_period_is_reported_or_a_declared_gap():
    store = FactStore()
    scope = store.scope
    periods = [(y, "FY") for y in sorted(scope.tenk_years)] + [
        (y, p) for y in sorted(scope.tenq_years) for p in sorted(scope.tenq_periods)]
    for metric in sorted(NEW):
        gap = next(g for g in store.gaps if g.metric == metric)
        for ticker in sorted(TICKERS):
            for year, period in periods:
                out = get_financial_fact(ticker, metric, year, period, store=store)
                if ticker in gap.tickers:
                    assert out["error"]["type"] == "not_reported", (ticker, metric, year, period)
                else:
                    assert "error" not in out and out["value"] > 0, (ticker, metric, year, period, out)


@needs_data
def test_a_current_ratio_from_the_tool_matches_the_statements():
    store = FactStore()
    assets = get_financial_fact("AAPL", "current_assets", 2024, store=store)["value"]
    liabilities = get_financial_fact("AAPL", "current_liabilities", 2024, store=store)["value"]
    assert round(assets / liabilities, 2) == 0.87                         # Apple 10-K FY2024: 152,987 / 176,392
