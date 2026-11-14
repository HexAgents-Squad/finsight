"""agent/critic_checks.py: each spec §8 invariant passing and failing, offline."""
import json

import pytest

from agent.contract import AnswerContract, refusal_answer
from agent.critic_checks import (
    QuestionScope,
    Violation,
    check_calculations,
    check_citations_present,
    check_contract,
    check_numbers_in_values,
    check_operation,
    check_partial_refusal,
    check_refusal,
    check_scope,
    check_values_supported,
    check_verified,
    expected_operations,
    with_verification,
)
from eval.validate_dev_set import DEV_FILE as DEV_PATH
from eval.validate_gold import DEFAULT_PATH as GOLD_PATH
from tests.test_contract import CALC, FILING, VALUE, XBRL, spec_example, with_

MSFT_2024 = QuestionScope(frozenset({"MSFT"}), frozenset({2024}))
MARGIN_QUESTION = "What was Microsoft's operating margin in fiscal 2024?"


def make(**changes) -> AnswerContract:
    return AnswerContract(**with_(spec_example(), **changes))


def rules(violations: list[Violation]) -> list[str]:
    return [v.rule for v in violations]


def refusal(rule="R1", **changes) -> AnswerContract:
    base = {"answer": refusal_answer(rule, "Apple's net sales in fiscal 2025"), "verified": True, "refused": True,
            "refusal_rule": rule}
    return AnswerContract(**{**base, **changes})


# ----------------------------------------------------------------------------------------------- all together
def test_the_spec_example_passes_every_check():
    assert check_contract(make(), MSFT_2024, MARGIN_QUESTION) == []


def test_a_refusal_passes_every_check():
    for rule in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        assert check_contract(refusal(rule), QuestionScope(frozenset())) == [], rule


# ----------------------------------------------------------------------------------------------- invariant 1
@pytest.mark.parametrize("shown", [
    "$109.4 billion", "$109,433 million", "109,433", "$109 billion", "USD 109.43 billion",
])
def test_invariant_1_accepts_a_value_as_displayed(shown):
    c = make(answer=f"Operating income was {shown}; the margin was 44.6% of revenue of $245.1 billion.")
    assert check_numbers_in_values(c) == []


@pytest.mark.parametrize("answer, figure", [
    ("The operating margin was 45.2%.", "45.2%"),                     # a number that is no value
    ("Operating income was $110.0 billion.", "110.0 billion"),        # rounds to a different figure
    ("The operating margin was 44.6.", "44.6"),                       # a percentage value shown without %
    ("Revenue was 245,122 and 2,000 employees.", "2,000"),
])
def test_invariant_1_flags_a_number_with_no_value(answer, figure):
    assert [v.message for v in check_numbers_in_values(make(answer=answer))] == [
        f"the answer states {figure!r} but no value in values matches it"]


def test_invariant_1_ignores_dates_years_items_and_the_r3_scope_text():
    c = make(answer="Microsoft's fiscal 2024 ended June 30, 2024 (Item 8 of the 10-K); margin 44.6%.")
    assert check_numbers_in_values(c) == []
    assert check_numbers_in_values(refusal("R3")) == []


def test_invariant_1_reads_a_sign_stated_in_words():
    v = with_(VALUE, name="revenue_growth", value=-16.3, unit="%")
    assert check_numbers_in_values(make(answer="Net earnings decreased 16.3%.", values=[v])) == []


# ----------------------------------------------------------------------------------------------- invariant 2
def test_invariant_2_needs_a_citation_for_the_values_company_and_period():
    other_year = with_(XBRL, fiscal_year=2023)
    other_chunk = with_(FILING, fiscal_year=2023, chunk_id="MSFT_10-K_2023_Item8_C3_0012")
    c = make(citations=[other_year, other_chunk])
    assert rules(check_values_supported(c)) == ["2", "2"]          # the two cited inputs; the margin is computed
    assert "operating_income = 109433 USD millions (MSFT FY 2024)" in check_values_supported(c)[0].message


def test_invariant_2_accepts_a_computed_value_only_when_its_inputs_are_values():
    # Citations of another year leave the margin supported by its calculation alone (spec: a citation with the
    # value's ticker, fiscal year and period supports any value, a computed one too)
    other_year = [with_(XBRL, fiscal_year=2023)]
    names = [v.message.split(" =")[0] for v in check_values_supported(make(citations=other_year))]
    assert names == ["value operating_income", "value revenue"]
    for bad in ({"inputs": [109433, 245000]},                          # an input that is not a value
                {"unit": "ratio"},                                     # not the value's unit
                {"result": 44.5}):                                     # not the value
        found = check_values_supported(make(citations=other_year, calculations=[with_(CALC, **bad)]))
        assert [v.message.split(" =")[0] for v in found][-1] == "value operating_margin", bad
    assert check_values_supported(make(calculations=[])) == []       # all three cited for FY2024


# ----------------------------------------------------------------------------------------------- invariant 3
def test_invariant_3_recomputes_every_calculation():
    assert check_calculations(make()) == []
    wrong = make(calculations=[with_(CALC, result=44.7)])
    assert [v.message for v in check_calculations(wrong)] == [
        "percent of [109433, 245122] is 44.6, not 44.7 (109433 / 245122 * 100)"]
    assert "gives unit '%', not 'ratio'" in check_calculations(make(calculations=[with_(CALC, unit="ratio")]))[0].message


@pytest.mark.parametrize("calc, ok", [
    ({"operation": "cagr", "inputs": [100, 121], "years": 2, "result": 10.0, "unit": "%"}, True),
    ({"operation": "growth", "inputs": [14405, 12056], "result": -16.3, "unit": "%"}, True),       # gold C12
    ({"operation": "difference", "inputs": [6.08, 6.11], "result": 0.03, "unit": "USD per share"}, True),
    ({"operation": "sum", "inputs": [1, 2, 3], "result": 6, "unit": None}, True),
    ({"operation": "ratio", "inputs": [3, 2], "result": 1.5, "unit": "ratio"}, True),
    ({"operation": "cagr", "inputs": [100, 121], "result": 10.0, "unit": "%"}, False),            # no years
    ({"operation": "percent", "inputs": [1, 0], "result": 0, "unit": "%"}, False),               # undefined
    ({"operation": "growth", "inputs": [1, 2, 3], "result": 100.0, "unit": "%"}, False),         # three inputs
    ({"operation": "percent", "inputs": [1, 3], "result": 33.33, "unit": "%"}, False),           # not rounded
])
def test_invariant_3_operations(calc, ok):
    c = make(calculations=[{**calc, "expression": "as returned"}])
    assert (check_calculations(c) == []) is ok


# ----------------------------------------------------------------------------------------------- invariant 4
def test_invariant_4_cited_companies_and_periods_are_the_questions():
    assert check_scope(make(), MSFT_2024) == [] and check_scope(make(), None) == []
    assert [v.message for v in check_scope(make(), QuestionScope(frozenset({"AAPL"})))] == [
        "citation MSFT 10-K FY 2024: MSFT is not a company the question asks about",
        "citation MSFT 10-K FY 2024: MSFT is not a company the question asks about",
        "AAPL is in the question but has no citation"]
    assert rules(check_scope(make(), QuestionScope(frozenset({"MSFT"}), frozenset({2025})))) == ["4", "4"]
    q2 = QuestionScope(frozenset({"MSFT"}), frozenset({2024}), frozenset({"Q2"}))
    assert "FY is not a period the question asks about" in check_scope(make(), q2)[0].message


def test_invariant_4_every_company_of_a_comparison_is_cited():
    both = QuestionScope(frozenset({"MSFT", "AAPL"}), frozenset({2024}))
    assert [v.message for v in check_scope(make(), both)] == ["AAPL is in the question but has no citation"]
    assert check_scope(refusal(), both) == []                       # a refusal cites nothing


# ----------------------------------------------------------------------------------------------- invariant 5
def test_invariant_5_a_refusal_has_a_rule_no_values_and_the_spec_sentence():
    assert check_refusal(refusal()) == [] and check_refusal(make()) == []
    assert [v.message for v in check_refusal(refusal(refusal_rule=None))] == [
        "refused is true but refusal_rule is not set"]
    bad = refusal(values=[VALUE], calculations=[CALC])
    assert [v.message.split(",")[0] for v in check_refusal(bad)] == [
        "a refusal has no values", "a refusal has no calculations"]


@pytest.mark.parametrize("answer, ok", [
    ("I can’t help with that because it is investment advice. I can share facts from the indexed SEC filings "
     "instead, for example Apple's net sales in fiscal 2025.", True),               # curly apostrophe
    ("I can't help with that because it is investment advice. I can share facts from the indexed SEC filings "
     "instead, for example What were Apple's net sales in fiscal 2025?", True),     # suggestion as a question
    ("I can't help with that because it is investment advice. I can share facts from the indexed SEC filings "
     "instead, for example {suggestion}.", False),                                  # placeholder left in (measured)
    ("I can't help with that because it is investment advice. I can share facts from the indexed SEC filings "
     "instead, for example .", False),                                              # no suggestion
    ("I can't help with that because it is a valuation opinion. I can share facts from the indexed SEC filings "
     "instead, for example Apple's net sales in fiscal 2025.", False),              # R4's reason under R1
    ("Sorry, I cannot give investment advice.", False),
])
def test_invariant_5_the_refusal_sentence(answer, ok):
    assert (check_refusal(refusal(answer=answer)) == []) is ok


def test_invariant_5_r3_accepts_typographic_dashes():
    plain = refusal_answer("R3", "Apple's revenue in fiscal 2024").replace("–", "-")
    assert check_refusal(refusal("R3", answer=plain)) == []


# ----------------------------------------------------------------------------------------------- invariant 6
def partial(answer: str, rule: str = "R1") -> AnswerContract:
    return make(answer=answer, refusal_rule=rule)


@pytest.mark.parametrize("answer, rule, ok", [
    ("Microsoft's operating margin in fiscal 2024 was 44.6%. I can't help with whether to buy the stock because "
     "it is investment advice.", "R1", True),                                       # RF14's shape
    ("Microsoft's operating margin in fiscal 2024 was 44.6%. Whether it will rise is a forecast that the filings "
     "do not state.", "R2", True),                                                  # the rule's reason, no "can't"
    ("Microsoft's operating margin in fiscal 2024 was 44.6%. Fiscal 2026 is outside the filings FinSight has "
     "indexed.", "R3", True),
    ("Microsoft's operating margin in fiscal 2024 was 44.6%.", "R1", False),        # nothing declined
    ("I can't help with that because it is investment advice. I cannot recommend stocks.", "R1", False),  # two
    ("I can't help with buying the stock because it is investment advice.", "R1", False),  # no factual part
])
def test_invariant_6_a_partial_refusal(answer, rule, ok):
    assert (check_partial_refusal(partial(answer, rule)) == []) is ok


def test_invariant_6_only_applies_to_partial_refusals():
    assert check_partial_refusal(make()) == [] and check_partial_refusal(refusal()) == []


# ----------------------------------------------------------------------------------------------- invariant 7
def test_invariant_7_verified_and_its_note():
    assert check_verified(make(), []) == []
    assert rules(check_verified(make(verified=False), [])) == ["7"]
    assert check_verified(make(verified=False, notes=["could not verify: a figure has no source"]), []) == []
    assert "notes says it could not verify" in check_verified(make(notes=["Could not verify: x"]), [])[0].message
    assert "verified is true but 1 check(s) fail" in check_verified(make(), [Violation("1", "x")])[0].message


def test_citations_are_required_unless_refused():
    assert rules(check_citations_present(make(citations=[]))) == ["citations"]
    assert check_citations_present(refusal()) == [] and check_citations_present(make()) == []


# ----------------------------------------------------------------------------------------------- operation
def test_the_operation_must_fit_the_question():
    assert check_operation(make(), MARGIN_QUESTION) == []
    growth = {"operation": "growth", "inputs": [109433, 245122], "result": 124.0, "unit": "%",
              "expression": "x"}
    assert [v.message for v in check_operation(make(calculations=[growth]), MARGIN_QUESTION)] == [
        "the question asks for percent, but the calculations use growth"]
    assert check_operation(make(calculations=[growth]), "What was Microsoft's revenue in fiscal 2024?") == []
    assert check_operation(make(calculations=[growth]), None) == []


@pytest.mark.parametrize("question, operations", [
    ("By what percentage did NVIDIA's revenue change from fiscal 2024 to fiscal 2025?", {"growth", "cagr"}),
    ("What was the compound annual growth rate of Apple's revenue from fiscal 2023 to 2025?", {"cagr"}),
    ("What was NVIDIA's gross margin in fiscal 2023, in percent?", {"percent"}),     # "in percent" is no change
    ("What was Johnson & Johnson's current ratio at the end of fiscal 2025?", {"ratio"}),
    ("How much more net income did JPMorgan earn than Bank of America in fiscal 2024?", {"difference"}),
    ("What were Apple's and Microsoft's combined revenues in fiscal 2024?", {"sum"}),
    ("How did Wells Fargo's net income change from fiscal 2024 to fiscal 2025?", set()),   # % or $: either
    ("What were Apple's diluted earnings per share in fiscal 2025?", set()),         # "per share" is no share of
])
def test_expected_operations(question, operations):
    assert expected_operations(question) == operations


def test_the_operation_rules_fit_the_gold_and_dev_questions():
    items = [json.loads(line) for path in (GOLD_PATH, DEV_PATH)
             for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for item in items:
        ops = expected_operations(item["question"])
        if item["category"] == "ratio":
            assert ops in ({"percent"}, {"ratio"}), item["id"]
        elif item["category"] == "period_comparison":
            assert "growth" in ops, item["id"]
        elif item["category"] in ("fact", "multi_company", "qualitative", "multi_turn"):
            assert ops == set(), item["id"]


# ----------------------------------------------------------------------------------------------- verification
def test_with_verification_marks_an_answer_that_fails():
    draft = make(calculations=[with_(CALC, result=44.7)])
    found = check_contract(draft, MSFT_2024, MARGIN_QUESTION)
    assert rules(found) == ["3", "7"]
    final = with_verification(draft, found)
    assert final.verified is False and final.notes == [
        "Microsoft's fiscal year ends on 30 June.",
        "could not verify: invariant 3: percent of [109433, 245122] is 44.6, not 44.7 (109433 / 245122 * 100)"]
    assert rules(check_contract(final, MSFT_2024, MARGIN_QUESTION)) == ["3"]      # invariant 7 now holds


def test_with_verification_marks_a_clean_answer_verified_and_drops_stale_notes():
    stale = make(verified=False, notes=["could not verify: old", "Microsoft's fiscal year ends on 30 June."])
    final = with_verification(stale, check_contract(stale, MSFT_2024))
    assert final.verified is True and final.notes == ["Microsoft's fiscal year ends on 30 June."]
    assert check_contract(final, MSFT_2024) == [] and with_verification(final, []) == final


def test_violations_read_as_messages_for_the_model():
    assert str(Violation("3", "x")) == "invariant 3: x" and str(Violation("operation", "y")) == "operation: y"
