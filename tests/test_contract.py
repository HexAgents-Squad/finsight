"""agent/contract.py: the spec §8 answer contract as a strict pydantic model (offline)."""
import copy
import json
import re

import pytest
import yaml
from pydantic import ValidationError

from agent.contract import (
    POLICY_SENTENCE,
    R3_SENTENCE,
    REFUSAL_REASONS,
    SCOPE_TEXT,
    AnswerContract,
    Calculation,
    Citation,
    Value,
    contract_json_schema,
    refusal_answer,
)
from core.spec import ALLOWED_UNITS, REFUSAL_RULES
from eval.schema import COMPANIES_FILE, ROOT

PERIOD = {"ticker": "MSFT", "fiscal_year": 2024, "fiscal_period": "FY"}
VALUE = {"name": "revenue", "value": 245122, "unit": "USD millions", **PERIOD, "duration": "12M",
         "period_end": "2024-06-30"}
XBRL = {"source_type": "xbrl", "form": "10-K", **PERIOD, "concept": "us-gaap:OperatingIncomeLoss",
        "accession_no": "0000950170-24-087843", "url": "https://www.sec.gov/Archives/edgar/data/789019/"}
FILING = {"source_type": "filing", "form": "10-K", **PERIOD, "item": "Item 8", "chunk_id": "MSFT_10-K_2024_Item8_C3_0012",
          "url": "https://www.sec.gov/Archives/edgar/data/789019/msft-10k.htm"}
CALC = {"operation": "percent", "inputs": [109433, 245122], "result": 44.6, "unit": "%",
        "expression": "109433 / 245122 * 100"}


def spec_example() -> dict:
    """docs/spec.md §8's example contract (figures checked against SEC XBRL in the spec)."""
    margin = {**VALUE, "name": "operating_margin", "value": 44.6, "unit": "%"}
    return {"answer": "Microsoft's operating margin for fiscal 2024 (ended 30 June 2024) was 44.6% (operating income "
                      "$109.4 billion ÷ revenue $245.1 billion).",
            "values": [{**VALUE, "name": "operating_income", "value": 109433}, dict(VALUE), margin],
            "citations": [dict(XBRL), dict(FILING)], "calculations": [dict(CALC)], "verified": True,
            "refused": False, "refusal_rule": None, "notes": ["Microsoft's fiscal year ends on 30 June."]}


def with_(base: dict, **changes) -> dict:
    out = copy.deepcopy(base)
    out.update(changes)
    return out


def test_the_spec_example_is_a_valid_contract_and_round_trips():
    c = AnswerContract.model_validate_json(json.dumps(spec_example()))
    assert AnswerContract.model_validate_json(c.model_dump_json()) == c
    assert c.values[2].unit == "%" and c.citations[1].chunk_id == "MSFT_10-K_2024_Item8_C3_0012"


@pytest.mark.parametrize("field, bad", [
    ("fiscal_year", "2024"),                  # strict: a string is not a year
    ("value", True),                          # nor a bool a number
    ("value", "245122"),
    ("unit", "USD bn"),                       # not a spec §7 unit
    ("fiscal_period", "Q4"),                  # Q4 is the 10-K's FY
    ("duration", "12 months"),
    ("ticker", "msft"),
    ("period_end", "30/06/2024"),
    ("period_end", "2024-02-30"),             # the right shape, not a date
    ("name", "   "),
])
def test_a_value_with_a_wrong_field_is_rejected(field, bad):
    with pytest.raises(ValidationError):
        Value(**with_(VALUE, **{field: bad}))


def test_unknown_fields_and_nan_are_rejected():
    with pytest.raises(ValidationError, match="Extra inputs"):
        Value(**with_(VALUE, currency="USD"))
    with pytest.raises(ValidationError):
        AnswerContract.model_validate_json('{"answer": "x", "verified": true, "refused": false, '
                                           '"values": [' + json.dumps(VALUE).replace("245122", "NaN") + "]}")


@pytest.mark.parametrize("period, duration, ok", [
    ("FY", "12M", True), ("FY", None, True), ("FY", "3M", False),
    ("Q1", "3M", True), ("Q1", "6M", False), ("Q2", "6M", True), ("Q2", "9M", False),
    ("Q3", "9M", True), ("Q3", "3M", True), ("Q3", "12M", False),
])
def test_a_values_duration_fits_its_fiscal_period(period, duration, ok):
    args = with_(VALUE, fiscal_year=2025, fiscal_period=period, duration=duration)
    if ok:
        assert Value(**args).duration == duration
    else:
        with pytest.raises(ValidationError, match=f"a {period} value cannot last"):
            Value(**args)


@pytest.mark.parametrize("changes, error", [
    ({"concept": None}, "needs concept and accession_no"),
    ({"accession_no": None}, "needs concept and accession_no"),
    ({"item": "Item 8"}, "has no item or chunk_id"),
    ({"accession_no": "0000950170-24-87843"}, "String should match pattern"),
    ({"concept": "OperatingIncomeLoss"}, "String should match pattern"),
    ({"fiscal_period": "Q2"}, "a 10-K is not cited for Q2"),
    ({"form": "10-Q"}, "a 10-Q is not cited for FY"),
    ({"url": "http://www.sec.gov/x"}, "String should match pattern"),
    ({"url": "https://www.sec.gov.example.com/x"}, "String should match pattern"),
])
def test_an_inconsistent_xbrl_citation_is_rejected(changes, error):
    with pytest.raises(ValidationError, match=error):
        Citation(**with_(XBRL, **changes))


@pytest.mark.parametrize("changes, error", [
    ({"chunk_id": None}, "needs item and chunk_id"),
    ({"accession_no": "0000950170-24-087843"}, "has no concept or accession_no"),
    ({"item": "Part I Item 2"}, "is not an Item of a 10-K"),
    ({"item": "Item 99"}, "is not an Item of a 10-K"),
    ({"chunk_id": "MSFT_10-K_2023_Item8_C3_0012"}, "is not a chunk of MSFT_10-K_2024_Item8"),    # another year
    ({"chunk_id": "MSFT_10-K_2024_Item7_C3_0012"}, "is not a chunk of"),                         # another Item
    ({"chunk_id": "AAPL_10-K_2024_Item8_C3_0012"}, "is not a chunk of"),                         # another company
    ({"chunk_id": "MSFT_10-K_2024_Item8_C3_0012x"}, "is not a chunk of"),
    ({"chunk_id": "MSFT_10-K_2024_Item8_C3_12"}, "is not a chunk of"),
])
def test_an_inconsistent_filing_citation_is_rejected(changes, error):
    with pytest.raises(ValidationError, match=error):
        Citation(**with_(FILING, **changes))


def test_a_10q_passage_is_cited_with_its_part_and_quarter():
    c = Citation(**with_(FILING, form="10-Q", fiscal_year=2025, fiscal_period="Q2", item="Part I Item 2",
                         chunk_id="MSFT_10-Q_2025Q2_PartIItem2_C1_0003"))
    assert c.item == "Part I Item 2"


def test_values_and_citations_are_built_from_tool_outputs():
    fact = {"value": 245122, "unit": "USD millions", "ticker": "MSFT", "fiscal_year": 2024, "fiscal_period": "FY",
            "duration": "12M", "period_start": "2023-07-01", "period_end": "2024-06-30",
            "concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "accession_no": "0000950170-24-087843", "filed": "2024-07-30", "url": XBRL["url"], "metric": "revenue",
            "label": "Revenue", "form": "10-K", "restated": False, "notes": []}
    assert Value.from_fact(fact) == Value(**VALUE)
    assert Value.from_fact(fact, name="total_revenue").name == "total_revenue"
    assert Citation.from_fact(fact).concept == fact["concept"]
    hit = {"chunk_id": FILING["chunk_id"], "text": "...", "ticker": "MSFT", "form": "10-K", "fiscal_year": 2024,
           "fiscal_period": "FY", "item": "Item 8", "source_url": FILING["url"], "score": 0.5, "citation": "..."}
    assert Citation.from_search(hit) == Citation(**FILING)


def test_calculations_come_from_the_calculate_tool():
    from tools.calculator import calculate
    out = calculate("cagr", [100, 121], years=2)
    calc = Calculation.from_tool(out)
    assert (calc.operation, calc.years, calc.result, calc.unit) == ("cagr", 2, 10.0, "%")      # notes left out
    assert Calculation.from_tool(calculate("percent", [109433, 245122])) == Calculation(**CALC)
    with pytest.raises(ValueError, match="calculate failed: percent is undefined"):
        Calculation.from_tool(calculate("percent", [1, 0]))
    with pytest.raises(ValidationError):
        Calculation(**with_(CALC, inputs=[1]))                       # every operation has two inputs or more


def test_refusal_answers_are_the_spec_sentences():
    assert refusal_answer("R1", "Apple's net sales in fiscal 2025?") == (
        "I can't help with that because it is investment advice. I can share facts from the indexed SEC filings "
        "instead, for example Apple's net sales in fiscal 2025.")
    assert refusal_answer("R3", "Apple's revenue in fiscal 2024").startswith(
        "That is outside the filings FinSight has indexed (20 companies,")
    with pytest.raises(ValueError, match="needs a suggestion"):
        refusal_answer("R4", " ")


def test_the_refusal_texts_are_spec_6_word_for_word():
    spec = (ROOT / "docs" / "spec.md").read_text(encoding="utf-8")
    assert set(REFUSAL_REASONS) == set(REFUSAL_RULES) - {"R3"}
    for rule, reason in REFUSAL_REASONS.items():
        row = next(line for line in spec.splitlines() if line.startswith(f"| **{rule} "))
        assert row.rstrip().endswith(f'| "{reason}" |'), rule
    assert f'"{POLICY_SENTENCE}"' in spec and f'"{R3_SENTENCE}"' in spec


def test_the_r3_sentence_states_the_configured_corpus():
    cfg = yaml.safe_load(COMPANIES_FILE.read_text(encoding="utf-8"))
    tenk, tenq = cfg["forms"]["10-K"], cfg["forms"]["10-Q"]
    expected = (f"({len(cfg['companies'])} companies, 10-K fiscal {min(tenk['fiscal_years'])}–"
                f"{max(tenk['fiscal_years'])} and 10-Q fiscal {tenq['fiscal_years'][0]} "
                f"{tenq['fiscal_periods'][0]}–{tenq['fiscal_periods'][-1]})")
    assert SCOPE_TEXT == expected


def test_the_allowed_units_are_spec_7():
    spec = (ROOT / "docs" / "spec.md").read_text(encoding="utf-8")
    line = next(line for line in spec.splitlines() if line.startswith("Allowed `unit` values:"))
    assert set(re.findall(r'"([^"]+)"', line)) == ALLOWED_UNITS


def test_the_json_schema_suits_guided_decoding():
    schema = contract_json_schema()
    text = json.dumps(schema)
    assert '"title"' not in text and {"answer", "verified", "refused"} <= set(schema["required"])
    # guided decoding matches a pattern against the whole string: every pattern must be a full match
    patterns = re.findall(r'"pattern": "((?:[^"\\]|\\.)*)"', text)
    assert patterns and all(p.startswith("^") and p.endswith("$") for p in patterns)
    assert "Value" in schema["$defs"] and "Citation" in schema["$defs"]
