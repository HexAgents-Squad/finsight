"""tools/: calculator, financial facts, compare_periods, search_filings and the registry (offline, synthetic data).

tests/fixtures/xbrl/AAPL.json is synthetic company-facts data (not Apple's figures) built so that each rule has a
case: 10-Ks that repeat the prior year as a comparative column, a restated FY2023 net income, a FY2025 Q1 revenue
restated by a fiscal-2026 10-Q (outside the corpus manifest), 3M and year-to-date quarters, balance-sheet instants,
per-share values and a zero gross profit.
"""
import asyncio
import json
import math
from pathlib import Path

import pytest

from core.exceptions import FinSightError
from retrieval.common import NO_FILTERS, Filters, SearchResult
from tools import compare_periods as compare_periods_module
from tools import financial_facts, registry, search_filings
from tools.base import CalculationError, ToolError
from tools.calculator import calculate, compute, round_half_up
from tools.compare_periods import compare_periods
from tools.financial_facts import FactStore, get_financial_fact

FIXTURES = Path(__file__).parent / "fixtures"
FACT_KEYS = {"value", "unit", "ticker", "fiscal_year", "fiscal_period", "duration", "period_start", "period_end",
             "concept", "accession_no", "filed", "url", "metric", "label", "form", "restated", "original_value",
             "original_accession_no", "period_accession_no", "cited_filing_indexed", "later_filings_exist", "notes"}
K24, K25, Q1_26 = "0000320193-24-000001", "0000320193-25-000001", "0000320193-26-000101"


@pytest.fixture(scope="module")
def store():
    return FactStore(xbrl_dir=FIXTURES / "xbrl", manifest=FIXTURES / "manifest.jsonl")


def fact(store, *args, **kwargs):
    return get_financial_fact(*args, store=store, **kwargs)


def error_type(result):
    assert isinstance(result, dict) and set(result) == {"error"}, result
    return result["error"]["type"]


# --- calculate -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("operation, inputs, extra, result, unit, expression", [
    ("ratio", [55624, 54007], {}, 1.03, "ratio", "55624 / 54007"),                   # gold B15
    ("percent", [109433, 245122], {}, 44.6, "%", "109433 / 245122 * 100"),           # gold B01
    ("growth", [60922, 130497], {}, 114.2, "%", "(130497 - 60922) / abs(60922) * 100"),   # gold C01
    ("growth", [14405, 12056], {}, -16.3, "%", "(12056 - 14405) / abs(14405) * 100"),     # gold C12
    ("difference", [383285, 391035], {"unit": "USD millions"}, 7750, "USD millions", "391035 - 383285"),
    ("sum", [0.1, 0.2], {"unit": "USD millions"}, 0.3, "USD millions", "0.1 + 0.2"),       # no float noise
    ("difference", [6.08, 6.11], {"unit": "USD per share"}, 0.03, "USD per share", "6.11 - 6.08"),
    ("cagr", [100, 121], {"years": 2}, 10.0, "%", "((121 / 100) ** (1 / 2) - 1) * 100"),
    ("cagr", [100, 0], {"years": 2}, -100.0, "%", "((0 / 100) ** (1 / 2) - 1) * 100"),     # end = 0 is defined
])
def test_calculator_operations_are_pinned(operation, inputs, extra, result, unit, expression):
    out = calculate(operation, inputs, **extra)
    assert out == {"operation": operation, "inputs": [float(x) for x in inputs], "result": result, "unit": unit,
                   "expression": expression, "notes": []}


@pytest.mark.parametrize("value, decimals, rounded", [(0.125, 2, 0.13), (12.25, 1, 12.3), (-2.25, 1, -2.3),
                                                      (2.675, 2, 2.68)])
def test_rounding_is_half_up_on_the_written_value(value, decimals, rounded):
    assert round_half_up(value, decimals) == rounded           # round() gives 0.12, 12.2, -2.2, 2.67


def test_calculator_rounds_half_up_like_the_gold_set():
    assert calculate("ratio", [1, 8])["result"] == 0.13
    assert calculate("percent", [1, 8])["result"] == 12.5


@pytest.mark.parametrize("operation, inputs, result", [
    ("percent", [23, 80], 28.8),        # exactly 28.75; binary floats give 28.749999... -> 28.7
    ("growth", [80, 103], 28.8),        # exactly 28.75
    ("percent", [9, 2000], 0.5),        # exactly 0.45
    ("ratio", [1, 40], 0.03),           # exactly 0.025
])
def test_exact_halfway_results_round_up(operation, inputs, result):
    assert calculate(operation, inputs)["result"] == result


def test_difference_and_sum_without_a_unit_keep_the_inputs_unit():
    out = calculate("difference", [383285, 391035])                  # the week0 §2.3 call, without `unit`
    assert (out["result"], out["unit"], out["notes"]) == (7750.0, None,
                                                          ["no unit given: the result is in the inputs' unit"])


def test_very_large_but_finite_results_are_numbers():
    assert calculate("ratio", [1e15, 1e-10])["result"] == 1e25          # beyond 28 decimal digits: still exact
    assert calculate("sum", [1e15, 1], unit="USD")["result"] == 1000000000000001.0


def test_negative_growth_base_is_computed_and_noted():
    out = calculate("growth", [-50, 25])
    assert out["result"] == 150.0 and out["expression"] == "(25 - (-50)) / abs(-50) * 100"
    assert out["notes"] == ["the base (old value) is negative, so growth is measured against its absolute value"]


@pytest.mark.parametrize("operation, inputs, years", [
    ("ratio", [1, 0], None), ("percent", [5, 0], None), ("growth", [0, 10], None),
    ("cagr", [0, 10], 2), ("cagr", [-1, 10], 2), ("cagr", [10, -1], 2), ("cagr", [10, 20], 0), ("cagr", [10, 20], -1)])
def test_undefined_calculations_are_errors_not_numbers(operation, inputs, years):
    assert error_type(calculate(operation, inputs, years=years)) == "undefined"
    with pytest.raises(CalculationError):
        compute(operation, inputs, years)


# Inputs are bounded (MAX_ABS_INPUT), but a tiny denominator or root can still leave the float range
@pytest.mark.parametrize("kwargs", [{"operation": "ratio", "inputs": [1e15, 1e-300]},            # 1e315
                                    {"operation": "percent", "inputs": [-1e15, 1e-310]},
                                    {"operation": "cagr", "inputs": [1e-300, 1e15], "years": 0.001}])
def test_an_overflowing_result_is_undefined_not_a_crash(kwargs):
    out = calculate(**kwargs)
    assert error_type(out) == "undefined" and "not a finite number" in out["error"]["message"]


@pytest.mark.parametrize("kwargs, message", [
    ({"operation": "ratio", "inputs": [1, 2, 3]}, "exactly 2 inputs"),
    ({"operation": "sum", "inputs": [1]}, "inputs: List should have at least 2 items"),
    ({"operation": "sum", "inputs": [2.0 ** 53 + 2, 1], "unit": "USD"}, "inputs.0: Input should be less than or equal"),
    ({"operation": "sum", "inputs": [10 ** 400, 1], "unit": "USD"}, "inputs.0: Input should be"),
    ({"operation": "cagr", "inputs": [1, 2]}, "years is required for cagr"),
    ({"operation": "growth", "inputs": [1, 2], "years": 1}, "years is required for cagr"),
    ({"operation": "ratio", "inputs": [1, 2], "unit": "USD millions"}, "unit is only for difference and sum"),
    ({"operation": "sum", "inputs": [1, 2], "unit": "dollars"}, "unit: Input should be"),
    ({"operation": "pow", "inputs": [1, 2]}, "operation: Input should be"),
    ({"operation": "ratio", "inputs": ["1", 2]}, "inputs.0: Input should be a valid number"),
    ({"operation": "ratio", "inputs": [True, 2]}, "inputs.0: Input should be a valid number"),
    ({"operation": "ratio", "inputs": [math.nan, 2]}, "inputs.0: Input should be a finite number"),
    ({"operation": "ratio", "inputs": [math.inf, 2]}, "inputs.0: Input should be a finite number"),
    ({"operation": "sum", "inputs": [1.0] * 51, "unit": "count"}, "inputs: List should have at most 50"),
])
def test_bad_calculator_calls_are_invalid_input(kwargs, message):
    out = calculate(**kwargs)
    assert error_type(out) == "invalid_input" and message in out["error"]["message"]


# --- get_financial_fact ----------------------------------------------------------------------------------------------

def test_fact_output_contract(store):
    out = fact(store, "AAPL", "revenue", 2025)
    assert set(out) == FACT_KEYS
    assert out | {"notes": None} == {
        "value": 1210.0, "unit": "USD millions", "ticker": "AAPL", "fiscal_year": 2025, "fiscal_period": "FY",
        "duration": "12M", "period_start": "2024-09-29", "period_end": "2025-09-27",
        "concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "accession_no": K25,
        "filed": "2025-10-31", "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019325000001/",
        "metric": "revenue", "label": "total net sales", "form": "10-K", "restated": False,
        "original_value": 1210.0, "original_accession_no": K25, "period_accession_no": K25,
        "cited_filing_indexed": True, "later_filings_exist": False, "notes": None}
    assert list(out) == list(financial_facts.FACT_FIELDS)                     # stable field order


def test_current_year_not_the_prior_year_comparative(store):
    # The FY2025 10-K also carries FY2024 (1,100) labelled fy=2025; the FY2025 figure is 1,210
    assert fact(store, "AAPL", "revenue", 2025)["value"] == 1210.0
    assert fact(store, "AAPL", "revenue", 2024)["value"] == 1100.0


def test_restated_value_is_the_latest_filed_with_the_original_kept(store):
    out = fact(store, "AAPL", "net_income", 2023)
    assert (out["value"], out["original_value"], out["restated"], out["accession_no"]) == (105.0, 100.0, True, K24)
    assert out["period_accession_no"] == "0000320193-23-000001"        # the period's own (corpus) filing
    assert (out["fiscal_year"], out["later_filings_exist"]) == (2023, None)  # the value's period, not the filing's
    assert out["notes"] == ["restated: first reported as 100 USD millions (accession 0000320193-23-000001); "
                            "the latest filed value is used (spec §7.7)"]


def test_a_citation_outside_the_indexed_corpus_is_flagged(store):
    out = fact(store, "AAPL", "revenue", 2025, "Q1")
    assert (out["value"], out["accession_no"], out["cited_filing_indexed"]) == (310.0, Q1_26, False)
    assert any("not in the indexed text corpus" in n for n in out["notes"])


def test_without_a_manifest_indexing_is_unknown_not_guessed():
    bare = FactStore(xbrl_dir=FIXTURES / "xbrl", manifest=FIXTURES / "no_such_manifest.jsonl")
    out = fact(bare, "AAPL", "revenue", 2025, "Q1")
    assert out["cited_filing_indexed"] is None and not any("indexed" in n for n in out["notes"])


def test_balance_sheet_values_are_instants(store):
    out = fact(store, "AAPL", "total_assets", 2025, "Q1")
    assert (out["value"], out["duration"], out["period_start"], out["period_end"]) == \
        (5600.0, None, None, "2024-12-28")
    asked = fact(store, "AAPL", "total_assets", 2025, "FY", "12M")
    assert asked["duration"] is None and asked["notes"] == [
        "point-in-time (balance-sheet) value as of 2025-09-27; a duration does not apply"]


def test_quarters_default_to_three_months_and_give_year_to_date_on_request(store):
    q2 = fact(store, "AAPL", "revenue", 2025, "Q2")
    assert (q2["value"], q2["duration"], q2["notes"]) == (280.0, "3M", [])
    ytd = fact(store, "AAPL", "revenue", 2025, "Q2", "6M")
    assert (ytd["value"], ytd["period_start"]) == (580.0, "2024-09-29")
    assert ytd["notes"] == ["year-to-date figure (6M), not the three-month quarter (spec §4.2)"]
    assert fact(store, "AAPL", "revenue", 2025, "Q3", "9M")["value"] == 870.0


def test_a_cash_flow_metric_reported_only_year_to_date_says_which_duration_exists(store):
    out = fact(store, "AAPL", "capex", 2025, "Q2")
    assert error_type(out) == "not_reported" and out["error"]["available_duration"] == "6M"
    assert fact(store, "AAPL", "capex", 2025, "Q2", "6M")["value"] == 32.0


def test_per_share_values_are_not_scaled(store):
    out = fact(store, "AAPL", "eps_diluted", 2025)
    assert (out["value"], out["unit"]) == (2.25, "USD per share")


@pytest.mark.parametrize("args, kind, message", [
    (("JPM", "gross_profit", 2024), "not_reported", "only AAPL, MSFT, NVDA, JNJ and KO present a gross-profit"),
    (("PG", "rd_expense", 2025, "Q1"), "not_reported", "10-K only"),
    (("AAPL", "rd_expense", 2024), "not_reported", "no reported rd_expense for AAPL FY2024"),
    (("TSLA", "revenue", 2024), "out_of_corpus", "TSLA is not one of the 20 companies"),
    (("AAPL", "revenue", 2022), "out_of_corpus", "FY2022 is outside the indexed filings"),
    (("AAPL", "revenue", 2024, "Q2"), "out_of_corpus", "FY2024 Q2 is outside"),
    (("MSFT", "revenue", 2024), "data_missing", "XBRL facts for MSFT are not cached"),
    (("AAPL", "ebitda", 2024), "invalid_input", "unknown metric 'ebitda'"),
    (("AAPL", "revenue", 2025, "Q4"), "not_reported", "Q4 is not reported separately"),
    (("AAPL", "revenue", 2025, "q4 "), "not_reported", "never derived"),
    (("PG", "rd_expense", 2025), "data_missing", "XBRL facts for PG are not cached"),   # the gap is Q1-Q3 only
    (("AAPL", "revenue", "2024"), "invalid_input", "fiscal_year: Input should be a valid integer"),
    (("AAPL", "revenue", True), "invalid_input", "fiscal_year: Input should be a valid integer"),
    (("AAPL", "revenue", 2024, "FY", "3M"), "invalid_input", "duration '3M' is not valid for FY"),
    (("AAPL", "revenue", 2025, "Q1", "6M"), "invalid_input", "duration '6M' is not valid for Q1"),
    (("AAPL", "revenue", 2025, "H1"), "invalid_input", "fiscal_period: Input should be"),
])
def test_missing_data_and_bad_requests_are_typed_errors(store, args, kind, message):
    out = fact(store, *args)
    assert error_type(out) == kind and message in out["error"]["message"]


@pytest.mark.parametrize("args", [("JPM", "gross_profit", 2024, "FY", "3M"), ("AAPL", "revenue", 2026, "FY", "3M"),
                                  ("TSLA", "revenue", 2024, "FY", "3M")])
def test_an_impossible_duration_is_reported_before_scope_or_gaps(store, args):
    assert error_type(fact(store, *args)) == "invalid_input"


def test_an_unreadable_cache_is_data_missing_not_invalid_input(tmp_path):
    (tmp_path / "AAPL.json").write_text("{not json", encoding="utf-8")
    out = fact(FactStore(xbrl_dir=tmp_path, manifest=tmp_path / "none.jsonl"), "AAPL", "revenue", 2024)
    assert error_type(out) == "data_missing" and "cannot be read" in out["error"]["message"]


def test_fact_inputs_are_normalised(store):
    assert fact(store, " aapl ", "Revenue ", 2025, "q1")["value"] == 310.0


# --- compare_periods -------------------------------------------------------------------------------------------------

def test_compare_sorts_years_and_computes_yoy_with_the_calculator(store):
    rows = compare_periods("AAPL", "revenue", [2025, 2023, 2024, 2025], store=store)
    assert [(r["fiscal_year"], r["value"], r["yoy_growth_pct"]) for r in rows] == \
        [(2023, 1000.0, None), (2024, 1100.0, 10.0), (2025, 1210.0, 10.0)]
    assert rows[2]["yoy_calculation"] == compute("growth", [1100.0, 1210.0])
    assert rows[0]["notes"] == [] and rows[0]["yoy_calculation"] is None
    assert {"unit", "period_end"} <= set(rows[0]) and rows[0]["period_end"] == "2023-09-30"


def test_a_missing_year_comes_back_with_a_note(store):
    rows = compare_periods("AAPL", "revenue", [2022, 2023, 2025], store=store)
    assert [r["value"] for r in rows] == [None, 1000.0, 1210.0]
    assert rows[0]["missing"] == "out_of_corpus" and rows[0]["unit"] == "USD millions"
    assert set(rows[0]) == set(rows[1])                                      # same shape for every row
    assert rows[1]["notes"] == ["no year-over-year growth: FY2022 has no value"]
    assert rows[2]["notes"] == ["no year-over-year growth: FY2024 was not requested"]


def test_yoy_from_a_zero_base_is_not_meaningful(store):
    rows = compare_periods("AAPL", "gross_profit", [2023, 2024], store=store)
    assert rows[1]["yoy_growth_pct"] is None
    assert rows[1]["notes"] == ["year-over-year growth is not meaningful: growth is undefined: the old value is 0"]


def test_a_restated_year_from_the_same_filing_is_like_for_like(store):
    # FY2023 (restated) and FY2024 both come from the FY2024 10-K: same basis, no mixing note
    rows = compare_periods("AAPL", "net_income", [2023, 2024], store=store)
    assert rows[1]["yoy_growth_pct"] == 14.3                                  # (120 - 105) / 105 = 14.29%
    assert rows[0]["accession_no"] == rows[1]["accession_no"] and not any("mixes" in n for n in rows[1]["notes"])


def yoy_rows(*values):
    """Minimal compare rows for _add_yoy: (fiscal_year, value, restated, accession_no)."""
    return [{"fiscal_year": y, "value": v, "restated": r, "accession_no": a, "notes": []} for y, v, r, a in values]


def test_a_restated_and_an_original_figure_from_different_filings_are_noted():
    rows = yoy_rows((2023, 100.0, True, "later-10-K"), (2024, 110.0, False, "own-10-K"))
    compare_periods_module._add_yoy(rows)
    assert rows[1]["notes"] == ["FY2023 is a restated value and FY2024 is not (they come from different filings), "
                                "so the comparison mixes a restated and an originally reported figure"]


def test_a_negative_base_note_reaches_the_comparison_row():
    rows = yoy_rows((2023, -50.0, False, "a"), (2024, 25.0, False, "b"))
    compare_periods_module._add_yoy(rows)
    assert rows[1]["yoy_growth_pct"] == 150.0
    assert rows[1]["notes"] == ["the base (old value) is negative, so growth is measured against its absolute value"]


def test_a_declared_gap_gives_rows_with_the_reason(store):
    rows = compare_periods("JPM", "gross_profit", [2024, 2025], store=store)
    assert [r["missing"] for r in rows] == ["not_reported", "not_reported"]


def test_a_missing_cache_fails_the_whole_comparison(store):
    out = compare_periods("MSFT", "revenue", [2024, 2025], store=store)        # no MSFT file in the fixture
    assert error_type(out) == "data_missing"


def test_duplicate_years_count_once(store):
    assert [r["fiscal_year"] for r in compare_periods("AAPL", "revenue", [2024] * 11, store=store)] == [2024]


@pytest.mark.parametrize("args, kind", [(("TSLA", "revenue", [2024]), "out_of_corpus"),
                                        (("AAPL", "ebitda", [2024]), "invalid_input"),
                                        (("AAPL", "revenue", []), "invalid_input"),
                                        (("AAPL", "revenue", list(range(2015, 2026))), "invalid_input"),
                                        (("AAPL", "revenue", ["2024"]), "invalid_input"),
                                        (("AAPL", "revenue", 2024), "invalid_input")])
def test_compare_call_level_errors(store, args, kind):
    assert error_type(compare_periods(*args, store=store)) == kind


# --- search_filings --------------------------------------------------------------------------------------------------

def hit(cid, form="10-K", period="FY", item="Item 7", score=0.0123456789):
    return SearchResult(cid, "text", "AAPL", form, 2025, period, item, "https://www.sec.gov/x.htm", score,
                        "2025-09-27")


class FakeRetriever:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, query, k=5, filters=NO_FILTERS):
        self.calls.append((query, k, filters))
        return self.hits[:k]


def test_search_passes_normalised_filters_and_returns_citations():
    fake = FakeRetriever([hit("a"), hit("b", "10-Q", "Q2", "Part I Item 2")])
    out = search_filings.search_filings(" growth drivers ", ticker=" aapl", fiscal_year=2025, fiscal_period="q2",
                                        form="10-q", item="part i  item 2", k=2, retriever=fake)
    assert fake.calls == [("growth drivers", 2, Filters("AAPL", 2025, "Q2", "10-Q", "Part I Item 2"))]
    assert out[0] == {"chunk_id": "a", "text": "text", "ticker": "AAPL", "form": "10-K", "fiscal_year": 2025,
                      "fiscal_period": "FY", "item": "Item 7", "source_url": "https://www.sec.gov/x.htm",
                      "score": 0.012346, "citation": "[AAPL 10-K FY2025 · Item 7]", "period_end": "2025-09-27"}
    assert out[1]["citation"] == "[AAPL 10-Q FY2025 Q2 · Part I Item 2]"


def test_search_without_filters_and_default_k():
    fake = FakeRetriever([hit(str(i)) for i in range(10)])
    assert len(search_filings.search_filings("revenue", retriever=fake)) == search_filings.DEFAULT_K
    assert fake.calls[0][2] == NO_FILTERS


def test_search_with_nothing_found_is_a_no_results_error():
    out = search_filings.search_filings("revenue", ticker="AAPL", retriever=FakeRetriever([]))
    assert error_type(out) == "no_results" and out["error"]["filters"] == {"ticker": "AAPL"}


@pytest.mark.parametrize("kwargs, message", [
    ({"query": ""}, "query: String should have at least 1 character"),
    ({"query": "   "}, "query: String should have at least 1 character"),
    ({"query": "x", "k": 0}, "k: Input should be greater than or equal to 1"),
    ({"query": "x", "k": 21}, "k: Input should be less than or equal to 20"),
    ({"query": "x", "fiscal_year": "2024"}, "fiscal_year: Input should be a valid integer"),
    ({"query": "x", "fiscal_period": "Q4"}, "fiscal_period: Input should be"),
    ({"query": "x", "form": "8-K"}, "form: Input should be"),
    ({"query": "x", "ticker": "A A"}, "invalid ticker"),
    ({"query": "x", "item": "MD&A"}, "invalid item"),
    ({"query": "x", "fiscal_period": "FY", "form": "10-Q"}, "form 10-Q does not fit fiscal_period FY"),
    ({"query": "x", "fiscal_period": "Q1", "form": "10-K"}, "form 10-K does not fit fiscal_period Q1"),
])
def test_search_rejects_bad_arguments_and_filters(kwargs, message):
    out = search_filings.search_filings(**kwargs, retriever=FakeRetriever([hit("a")]))
    assert error_type(out) == "invalid_input" and message in out["error"]["message"]


@pytest.mark.parametrize("kwargs, message", [
    ({"ticker": "TSLA"}, "TSLA is not one of the 20 companies"),
    ({"fiscal_year": 2026}, "fiscal 2026 is outside the indexed filings"),
    ({"fiscal_year": 2024, "fiscal_period": "Q1"}, "fiscal 2024 is outside"),          # 10-Qs: fiscal 2025 only
    ({"fiscal_year": 2024, "form": "10-Q"}, "fiscal 2024 is outside"),
    ({"fiscal_year": 24}, "fiscal 24 is outside"),                    # R3 like get_financial_fact, not "fix the call"
    ({"fiscal_year": 1990}, "fiscal 1990 is outside"),
])
def test_search_outside_the_corpus_is_out_of_corpus_like_the_fact_tool(kwargs, message):
    fake = FakeRetriever([hit("a")])
    out = search_filings.search_filings("revenue", **kwargs, retriever=fake)
    assert error_type(out) == "out_of_corpus" and message in out["error"]["message"] and fake.calls == []


def test_search_inside_the_corpus_passes_the_scope_check():
    fake = FakeRetriever([hit("a")])
    assert search_filings.search_filings("revenue", fiscal_year=2024, form="10-K", retriever=fake)[0]["chunk_id"] == "a"
    assert search_filings.search_filings("revenue", fiscal_year=2025, fiscal_period="Q3", retriever=fake)


@pytest.mark.parametrize("error", [FileNotFoundError("data/index/bm25_C3.pkl not found; run ingest/04_index.py"),
                                   ImportError("No module named 'chromadb'")])
def test_an_unavailable_index_is_a_data_missing_error(monkeypatch, error):
    import retrieval.bm25

    def unavailable(path):
        raise error

    monkeypatch.setattr(retrieval.bm25.BM25Index, "load", staticmethod(unavailable))
    search_filings.default_retriever.cache_clear()
    try:
        out = search_filings.search_filings("revenue")
    finally:
        search_filings.default_retriever.cache_clear()
    assert error_type(out) == "data_missing" and str(error) in out["error"]["message"]


def test_an_unexpected_failure_is_an_internal_error_not_an_exception():
    class Broken:
        def search(self, query, k=5, filters=NO_FILTERS):
            raise RuntimeError("boom")

    out = search_filings.search_filings("revenue", retriever=Broken())
    assert out == {"error": {"type": "internal_error", "message": "RuntimeError: boom"}}


# --- registry ---------------------------------------------------------------------------------------------------------

def test_schemas_match_the_week0_signatures():
    params = {s["name"]: (list(s["parameters"]["properties"]), s["parameters"].get("required", []))
              for s in registry.SCHEMAS}
    assert params == {
        "calculate": (["operation", "inputs", "years", "unit"], ["operation", "inputs"]),
        "get_financial_fact": (["ticker", "metric", "fiscal_year", "fiscal_period", "duration"],
                               ["ticker", "metric", "fiscal_year"]),
        "compare_periods": (["ticker", "metric", "years"], ["ticker", "metric", "years"]),
        "search_filings": (["query", "ticker", "fiscal_year", "fiscal_period", "form", "item", "k"], ["query"]),
    }
    for schema in registry.SCHEMAS:
        assert schema["description"] and schema["parameters"]["additionalProperties"] is False
        assert "title" not in schema["parameters"]
        assert all("description" in p for p in schema["parameters"]["properties"].values())


def test_call_tool_validates_before_calling(store, monkeypatch):
    monkeypatch.setattr(financial_facts, "default_store", lambda: store)
    assert registry.call_tool("calculate", {"operation": "percent", "inputs": [1, 4]})["result"] == 25.0
    assert registry.call_tool("get_financial_fact", {"ticker": "AAPL", "metric": "revenue",
                                                     "fiscal_year": 2024})["value"] == 1100.0
    assert error_type(registry.call_tool("plot", {})) == "invalid_input"
    assert error_type(registry.call_tool(["calculate"], {})) == "invalid_input"         # unhashable name
    assert error_type(registry.call_tool("calculate", ["percent", [1, 4]])) == "invalid_input"
    assert error_type(registry.call_tool("calculate", None)) == "invalid_input"           # required args missing
    extra = registry.call_tool("calculate", {"operation": "percent", "inputs": [1, 4], "precision": 3})
    assert "precision: Extra inputs are not permitted" in extra["error"]["message"]
    injected = registry.call_tool("get_financial_fact", {"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024,
                                                         "store": None})
    assert "store: Extra inputs are not permitted" in injected["error"]["message"]


def test_warm_up_reports_what_is_missing(monkeypatch):
    def unavailable():
        raise ToolError("data_missing", "the search index is not available")

    monkeypatch.setattr(financial_facts, "default_store", lambda: None)
    monkeypatch.setattr(search_filings, "default_retriever", unavailable)
    assert registry.warm_up() == {"error": {"type": "data_missing", "message": "the search index is not available"}}


# --- review follow-ups: rounding per unit, notes, later filings, setup and service errors, async, JSON args ------

def test_per_share_and_cagr_results_use_their_own_decimals():
    assert calculate("sum", [1.005, 2], unit="USD per share")["result"] == 3.01      # 2 decimals, half-up
    assert calculate("sum", [1.005, 2], unit="USD millions")["result"] == 3.005      # other units keep the value
    assert calculate("cagr", [100, 101], years=4)["result"] == 0.2                   # 0.249%: needs a precise root


def test_an_instant_without_a_duration_carries_no_note(store):
    assert fact(store, "AAPL", "total_assets", 2024)["notes"] == []
    rows = compare_periods("AAPL", "total_assets", [2024, 2025], store=store)
    assert [r["duration"] for r in rows] == [None, None] and rows[0]["notes"] == []
    assert rows[1]["notes"] == [] and rows[1]["yoy_growth_pct"] == 9.1               # 5,500 -> 6,000


def test_an_explicit_duration_miss_does_not_suggest_itself(store):
    out = fact(store, "AAPL", "eps_diluted", 2025, "Q2", "6M")                      # no quarterly EPS in the fixture
    assert error_type(out) == "not_reported" and "available_duration" not in out["error"]
    assert out["error"]["message"] == "no reported eps_diluted for AAPL FY2025 Q2 (6M)"


def test_a_missing_per_share_year_keeps_the_per_share_unit(store):
    rows = compare_periods("AAPL", "eps_diluted", [2022, 2025], store=store)
    assert [(r["value"], r["unit"]) for r in rows] == [(None, "USD per share"), (2.25, "USD per share")]
    fact_keys = [k for k in rows[0] if k not in ("missing", "yoy_growth_pct", "yoy_calculation")]
    assert fact_keys == list(financial_facts.FACT_FIELDS)                         # same keys and order as a fact


def test_too_many_distinct_years_are_rejected(store):
    out = compare_periods("AAPL", "revenue", list(range(2015, 2026)), store=store)
    assert error_type(out) == "invalid_input" and "at most 10 distinct years, got 11" in out["error"]["message"]


def test_a_later_annual_filing_is_flagged_for_the_latest_year_only(tmp_path):
    facts = json.loads((FIXTURES / "xbrl" / "AAPL.json").read_text(encoding="utf-8"))
    facts["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"].append(
        {"start": "2025-09-28", "end": "2026-09-26", "val": 1, "accn": "0000320193-26-000001", "fy": 2026,
         "fp": "FY", "form": "10-K", "filed": "2026-10-30"})
    (tmp_path / "AAPL.json").write_text(json.dumps(facts), encoding="utf-8")
    later = FactStore(xbrl_dir=tmp_path, manifest=FIXTURES / "manifest.jsonl")
    assert fact(later, "AAPL", "revenue", 2025)["later_filings_exist"] is True
    assert fact(later, "AAPL", "revenue", 2024)["later_filings_exist"] is None
    assert fact(later, "AAPL", "revenue", 2025, "Q3")["later_filings_exist"] is None


def test_a_cache_that_cannot_be_read_is_data_missing(tmp_path):
    (tmp_path / "AAPL.json").mkdir()                                     # reading a directory raises an OSError
    out = fact(FactStore(xbrl_dir=tmp_path, manifest=tmp_path / "none.jsonl"), "AAPL", "revenue", 2024)
    assert error_type(out) == "data_missing" and "cannot be read" in out["error"]["message"]


@pytest.mark.parametrize("broken", ["concept_map", "manifest"])
def test_a_broken_setup_file_is_data_missing(tmp_path, monkeypatch, broken):
    kwargs = {"xbrl_dir": FIXTURES / "xbrl", "manifest": FIXTURES / "manifest.jsonl"}
    if broken == "concept_map":
        (tmp_path / "map.yaml").write_text("metrics: [not, a, mapping", encoding="utf-8")
        kwargs["concept_map"] = tmp_path / "map.yaml"
    else:
        (tmp_path / "manifest.jsonl").write_text('{"ticker": "AAPL"}\n', encoding="utf-8")   # no accession_no
        kwargs["manifest"] = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(financial_facts, "_default_store", lambda: FactStore(**kwargs))
    out = registry.call_tool("get_financial_fact", {"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024})
    assert error_type(out) == "data_missing" and "fact store cannot be built" in out["error"]["message"]


def test_incomplete_settings_for_the_search_index_are_data_missing(monkeypatch):
    from pydantic import BaseModel

    import retrieval.bm25
    import retrieval.dense

    class Settings(BaseModel):
        EMBED_BACKEND: int

    class Client:
        def list_collections(self):
            return ["chunks_C3"]

    def incomplete():
        Settings(EMBED_BACKEND="x")

    monkeypatch.setattr(retrieval.bm25.BM25Index, "load", staticmethod(lambda path: object()))
    monkeypatch.setattr(retrieval.dense, "open_client", lambda *a: Client())
    monkeypatch.setattr(retrieval.dense, "embedder_from_settings", incomplete)
    search_filings.default_retriever.cache_clear()
    try:
        out = search_filings.search_filings("revenue")
    finally:
        search_filings.default_retriever.cache_clear()
    assert error_type(out) == "data_missing" and out["error"]["message"].startswith("settings are incomplete")


def test_an_embedding_outage_is_service_unavailable_not_a_bug():
    class Down:
        def search(self, query, k=5, filters=NO_FILTERS):
            raise FinSightError("LLM request to all-MiniLM failed: connection refused", error_type="NetworkError")

    out = search_filings.search_filings("revenue", retriever=Down())
    assert out == {"error": {"type": "service_unavailable", "cause": "NetworkError",
                             "message": "LLM request to all-MiniLM failed: connection refused"}}


class PrivateLoopRetriever:
    """Searches the way the real embedder does: on its own event loop (retrieval/dense.py)."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()

    def search(self, query, k=5, filters=NO_FILTERS):
        async def embed():
            return [hit("a")]
        return self.loop.run_until_complete(embed())


# The wrong-way call leaves the embed coroutine un-awaited, exactly as the real embedder would
@pytest.mark.filterwarnings("ignore:coroutine .* was never awaited:RuntimeWarning")
def test_async_callers_use_call_tool_async(monkeypatch):
    retriever = PrivateLoopRetriever()
    monkeypatch.setattr(search_filings, "default_retriever", lambda: retriever)

    async def agent_step():
        inside_loop = registry.call_tool("search_filings", {"query": "revenue"})           # wrong way: fails
        threaded = await registry.call_tool_async("search_filings", {"query": "revenue"})  # right way
        return inside_loop, threaded

    try:
        inside_loop, threaded = asyncio.run(agent_step())
    finally:
        retriever.loop.close()
    assert error_type(inside_loop) == "internal_error"
    assert threaded[0]["citation"] == "[AAPL 10-K FY2025 · Item 7]"


@pytest.mark.parametrize("arguments, expected", [
    ('{"operation": "percent", "inputs": [1, 4]}', 25.0),                 # OpenAI tool_calls arguments string
    (b'{"operation": "ratio", "inputs": [1, 4]}', 0.25),
])
def test_call_tool_accepts_json_string_arguments(arguments, expected):
    assert registry.call_tool("calculate", arguments)["result"] == expected


@pytest.mark.parametrize("arguments, message", [('{"operation": "percent", ', "arguments are not valid JSON"),
                                                ('["percent", [1, 4]]', "must be a JSON object, not list")])
def test_call_tool_rejects_broken_json_arguments(arguments, message):
    out = registry.call_tool("calculate", arguments)
    assert error_type(out) == "invalid_input" and message in out["error"]["message"]


def test_warm_up_runs_one_search_so_the_model_loads(monkeypatch, store):
    fake = FakeRetriever([hit("a")])
    monkeypatch.setattr(financial_facts, "_default_store", lambda: store)
    monkeypatch.setattr(search_filings, "default_retriever", lambda: fake)
    assert registry.warm_up() == {"ok": True}
    assert fake.calls == [(registry.WARM_UP_QUERY, 1, NO_FILTERS)]


def test_the_calculator_units_are_the_spec_units_the_gold_schema_checks():
    from typing import get_args

    from eval.schema import ALLOWED_UNITS
    from tools.calculator import Unit
    assert set(get_args(Unit)) == ALLOWED_UNITS                    # one list in spec §7, defined twice in code
