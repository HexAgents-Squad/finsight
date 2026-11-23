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
    messages = [v.message for v in check_scope(make(), QuestionScope(frozenset({"AAPL"})))]
    assert messages[:2] == ["citation MSFT 10-K FY 2024: MSFT is not a company the question asks about"] * 2
    assert messages[2] == "value operating_income (MSFT FY 2024): MSFT is not a company the question asks about"
    assert len(messages) == 6 and messages[-1] == "AAPL is in the question but has no citation"
    assert rules(check_scope(make(), QuestionScope(frozenset({"MSFT"}), frozenset({2025})))) == ["4"] * 5
    q2 = QuestionScope(frozenset({"MSFT"}), frozenset({2024}), frozenset({"Q2"}))
    assert "FY is not a period the question asks about" in check_scope(make(), q2)[0].message


def test_invariant_4_checks_a_computed_values_own_period():
    # a growth value labelled with a year the question does not ask about, though its calculation is right
    growth = with_(VALUE, name="revenue_growth", value=124.0, unit="%", fiscal_year=2026, period_end="2026-06-30")
    c = make(values=[*spec_example()["values"], growth])
    assert [v.message for v in check_scope(c, MSFT_2024)] == [
        "value revenue_growth (MSFT FY 2026): fiscal 2026 is not a year the question asks about"]


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
    # variants a model writes (W1-16 review): a comma or colon after "for example", a quoted suggestion, "cannot"
    ("I can't help with that because it is investment advice. I can share facts from the indexed SEC filings "
     "instead, for example, Costco's net income in fiscal 2025.", True),
    ("I can't help with that because it is investment advice. I can share facts from the indexed SEC filings "
     "instead, for example: Costco's net income in fiscal 2025.", True),
    ('I can\'t help with that because it is investment advice. I can share facts from the indexed SEC filings '
     'instead, for example "What was Costco\'s net income in fiscal 2025?"', True),
    ("I cannot help with that because it is investment advice. I can share facts from the indexed SEC filings "
     "instead, for example Costco's net income in fiscal 2025.", True),
    ("I can't help with that because it is investment advice. I can share facts from the indexed SEC filings "
     "instead, for example Costco's net income in fiscal 2025. Buy it anyway.", False),   # more than the sentence
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
    ("Microsoft's operating margin in fiscal 2024 was 44.6%, but I can't help with whether to buy the stock "
     "because it is investment advice.", "R1", True),                               # one sentence, both parts
    ("Microsoft's operating margin in fiscal 2024 was 44.6%. The filings do not provide a fiscal 2026 figure. "
     "I can't help with that because it is a forecast that the filings do not state.", "R2", True),  # a fact
])
def test_invariant_6_a_partial_refusal(answer, rule, ok):
    assert (check_partial_refusal(partial(answer, rule)) == []) is ok


def test_invariant_6_a_partial_refusal_that_only_declines():
    only = make(answer="I can't help with buying the stock because it is investment advice.", refusal_rule="R1",
                values=[], calculations=[])
    assert [v.message for v in check_partial_refusal(only)] == [
        "a partial refusal answers the factual part too; the answer only declines"]


def test_invariant_6_rf15_with_the_full_policy_sentence():
    answer = ("Amazon's net sales in fiscal 2024 were $638.0 billion. " + refusal_answer("R2", "Amazon's net sales "
              "in fiscal 2023"))
    v = with_(VALUE, ticker="AMZN", name="net_sales", value=637959, fiscal_year=2024, period_end="2024-12-31")
    c = make(answer=answer, refusal_rule="R2", values=[v], calculations=[],
             citations=[with_(XBRL, ticker="AMZN", url="https://www.sec.gov/x")])
    assert check_contract(c, QuestionScope(frozenset({"AMZN"}), frozenset({2024, 2026}))) == []


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
    ("What was Johnson & Johnson's current ratio at the end of fiscal 2025?", {"ratio", "percent"}),
    ("What was the difference in net income between JPMorgan and Bank of America in fiscal 2024?", {"difference"}),
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
            assert ops and ops <= {"percent", "ratio"}, item["id"]
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


# ----------------------------------------------------------------------------------------------- review findings
def test_a_wrong_margin_without_its_calculation_is_not_verified():
    # W1-07 review B1: 44.7% instead of 44.6%, no calculation, only an XBRL citation
    margin = with_(VALUE, name="operating_margin", value=44.7, unit="%")
    c = make(answer="Microsoft's operating margin for fiscal 2024 was 44.7% (operating income $109.4 billion / "
                    "revenue $245.1 billion).", values=[*spec_example()["values"][:2], margin],
             citations=[XBRL], calculations=[])
    found = check_contract(c, MSFT_2024, MARGIN_QUESTION)
    assert rules(found) == ["2", "operation", "7"]
    assert found[0].message.startswith("value operating_margin = 44.7 % (MSFT FY 2024) has no filing passage for "
                                       "that company and period")
    assert found[1].message.startswith("the question asks for percent, but no calculation computes it")


def test_a_percentage_stated_in_a_cited_passage_needs_no_calculation():
    # gold E02 style: "rose 142%" read from MD&A text, with a filing citation of that period
    rise = with_(VALUE, name="revenue_growth", value=142.0, unit="%")
    c = make(answer="Revenue rose 142% in fiscal 2024.", values=[rise], citations=[FILING], calculations=[])
    assert check_contract(c, MSFT_2024, "What drove Microsoft's revenue in fiscal 2024?") == []


def test_an_undefined_or_unreported_measure_has_no_calculation_to_show():
    inputs = spec_example()["values"][:2]
    for answer in ("The operating margin is not meaningful: revenue was $245.1 billion and operating income "
                   "$109.4 billion.", "The operating margin is not separately reported; revenue was $245.1 billion "
                   "and operating income $109.4 billion."):
        c = make(answer=answer, values=inputs, calculations=[])
        assert check_operation(c, MARGIN_QUESTION) == [], answer
    assert check_operation(refusal(), MARGIN_QUESTION) == []


@pytest.mark.parametrize("answer", [
    "Net income attributable to Costco was $109,433 million in fiscal 2024 (52 weeks ended September 1, 2024).",
    "Revenue was $245.1 billion in the 53-week fiscal year.",
    "Operating income was USD109,433 million.",
    "The margin was a 44.6-percent share of revenue.",
])
def test_invariant_1_reads_weeks_multiples_and_spelled_units(answer):
    assert check_numbers_in_values(make(answer=answer)) == []


def test_invariant_1_a_ratio_written_as_a_multiple():
    ratio = with_(VALUE, name="current_ratio", value=1.03, unit="ratio")
    assert check_numbers_in_values(make(answer="The current ratio was 1.03x.", values=[ratio])) == []
    assert rules(check_numbers_in_values(make(answer="The current ratio was 1.3x.", values=[ratio]))) == ["1"]


def test_invariant_1_ignores_the_scope_text_in_a_partial_r3_answer():
    answer = ("Microsoft's operating margin in fiscal 2024 was 44.6%. Fiscal 2026 is outside the filings FinSight "
              "has indexed (20 companies, 10-K fiscal 2023–2025 and 10-Q fiscal 2025 Q1–Q3).")
    assert check_numbers_in_values(make(answer=answer, refusal_rule="R3")) == []


def test_invariant_3_an_input_the_calculator_refuses():
    c = make(calculations=[with_(CALC, inputs=[2e15, 1])])               # beyond the calculator's input range
    assert "cannot be recomputed" in check_calculations(c)[0].message


# ----------------------------------------------------------------------------------------------- PR #28 review
@pytest.mark.parametrize("answer", [
    "Operating income was $109.4 billion for the 6 months ended December 31, 2024.",
    "Operating income was $109.4 billion in the 9-month period and $245.1 billion over 12 months ended June 30.",
    "Operating income was $109.4 billion in the 13 weeks ended March 30, 2025.",
    "Operating income was $109.4 billion in fiscal 2024-25 (FY2024/25).",
    "After the 10-for-1 split, operating income was $109.4 billion (see Note 12, page 28).",
])
def test_invariant_1_period_lengths_spans_splits_and_references_are_not_figures(answer):
    assert check_numbers_in_values(make(answer=answer)) == []


def test_invariant_1_a_cagr_years_count_is_backed_by_its_calculation():
    cagr = {"operation": "cagr", "inputs": [100, 121], "years": 2, "result": 10.0, "unit": "%", "expression": "x"}
    values = [with_(VALUE, value=100), with_(VALUE, value=121), with_(VALUE, name="revenue_cagr", value=10.0, unit="%")]
    c = make(answer="Revenue grew at a 2-year CAGR of 10.0%.", values=values, calculations=[cagr])
    assert check_numbers_in_values(c) == []
    assert rules(check_numbers_in_values(make(answer="Revenue grew at a 3-year CAGR of 10.0%.", values=values,
                                              calculations=[cagr]))) == ["1"]


def test_invariant_1_a_ratio_may_also_be_shown_as_a_percentage():
    ratio = with_(VALUE, name="efficiency_ratio", value=0.45, unit="ratio")
    assert check_numbers_in_values(make(answer="The ratio was 0.45, i.e. 45%.", values=[ratio])) == []
    assert rules(check_numbers_in_values(make(answer="The ratio was 46%.", values=[ratio]))) == ["1"]


def test_invariant_6_the_full_refusal_sentence_is_not_a_partial_answer():
    # the §6 sentence's offer ("I can share facts ... for example ...") is no factual part
    c = make(answer=refusal_answer("R1", "What was Apple's total net sales in fiscal 2025?"), refusal_rule="R1",
             values=[], calculations=[])
    assert [v.message for v in check_partial_refusal(c)] == [
        "a partial refusal answers the factual part too; the answer only declines"]


def test_the_operation_check_accepts_a_percentage_the_filing_states():
    # gold B13: Apple's gross margin percentage is stated in Item 7; read from the passage, not computed
    margin = with_(VALUE, ticker="AAPL", name="gross_margin", value=46.9, unit="%", fiscal_year=2025,
                   period_end="2025-09-27")
    passage = with_(FILING, ticker="AAPL", fiscal_year=2025, item="Item 7", chunk_id="AAPL_10-K_2025_Item7_C3_0040")
    c = make(answer="Apple's total gross margin percentage was 46.9% in fiscal 2025 (Item 7).", values=[margin],
             citations=[passage], calculations=[])
    question = "What was Apple's gross margin percentage in fiscal 2025?"
    assert check_contract(c, QuestionScope(frozenset({"AAPL"}), frozenset({2025})), question) == []
    xbrl_only = make(answer=c.answer, values=[margin], citations=[with_(XBRL, ticker="AAPL", fiscal_year=2025)],
                     calculations=[])
    assert "operation" in rules(check_contract(xbrl_only, QuestionScope(frozenset({"AAPL"})), question))


@pytest.mark.parametrize("ending", [
    "'What was NVIDIA's revenue in fiscal 2025?'",              # single quotes, an apostrophe inside
    "NVIDIA's revenue under U.S. GAAP in fiscal 2025.",          # an abbreviation is no sentence end
    "segment revenue, e.g. Data Center revenue in fiscal 2025.",
])
def test_invariant_5_quoted_suggestions_and_abbreviations(ending):
    answer = ("I can't help with that because it is investment advice. I can share facts from the indexed SEC "
              "filings instead, for example " + ending)
    assert check_refusal(refusal(answer=answer)) == []


def test_the_gold_refusal_sentences_pass_every_check():
    # each gold G01-G07 answer quotes the §6 sentence the agent must give (Refuse (R1): "...")
    items = [json.loads(line) for line in GOLD_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    refusals = [i for i in items if i["must_refuse"]]
    assert len(refusals) == 7
    for item in refusals:
        sentence = item["gold_answer"].split('"', 1)[1].rsplit('"', 1)[0]
        c = AnswerContract(answer=sentence, verified=True, refused=True, refusal_rule=item["refusal_rule"])
        assert check_contract(c, QuestionScope(frozenset(item["companies"]))) == [], item["id"]


def test_a_degenerate_reply_neither_crashes_nor_stalls_the_critic():
    import time
    for answer in ("Revenue was " + "9" * 20_000 + ".", "Revenue was " + "0" * 8_000 + ".",
                   "I can't " * 5_000 + "help.", "Revenue rose. " * 5_000):
        c = make(answer=answer, refusal_rule="R1")
        started = time.perf_counter()
        check_contract(c, MSFT_2024, MARGIN_QUESTION)
        assert time.perf_counter() - started < 2.0, answer[:30]


# W3-05: a drivers question asks for the drivers, not an operation
def test_a_drivers_question_expects_no_operation_even_when_it_names_a_ratio_or_a_margin():
    from agent.critic_checks import expected_operations
    assert expected_operations("Why did UnitedHealth Group's medical care ratio rise in fiscal 2024?") == frozenset()
    assert expected_operations("What drove the increase in Apple's gross margin in fiscal 2024?") == frozenset()
    assert expected_operations("What was UnitedHealth Group's medical care ratio in fiscal 2024?") == {"ratio", "percent"}



@pytest.mark.parametrize(("question", "expected"), [
    ("What drove the gross margin in fiscal 2023, and what was it?", {"percent"}),
    ("What was the percentage change in revenue in fiscal 2023 and what drove it?", {"growth", "cagr"}),
    ("What was Apple's revenue CAGR from fiscal 2023 to 2025 and what drove it?", {"cagr"}),
    ("What were the drivers of Apple's gross margin change in fiscal 2024?", set()),
])
def test_a_drivers_question_that_also_asks_for_a_figure_keeps_its_operation(question, expected):
    """PR #58 review: only a question that asks for the drivers alone expects no operation."""
    from agent.critic_checks import expected_operations
    assert expected_operations(question) == expected
