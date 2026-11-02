"""Rules of the gold-set validator (eval/schema.py): one test per rule, plus 'valid stays valid'."""
import pytest

from eval.validate_gold import validate_records
from tests.factories import BASE, full_gold_set, make


def errors_for(*records: dict, final: bool = False) -> list[str]:
    return validate_records(list(enumerate(records, start=1)), final=final).errors


@pytest.mark.parametrize("kind", list(BASE))
def test_every_valid_template_passes(kind):
    records = [make(kind)]
    if kind == "multi_turn":
        records.insert(0, make("fact"))          # its previous turn must exist
    assert errors_for(*records) == []


def test_partial_refusal_on_a_normal_question_is_allowed():
    assert errors_for(make("fact", refusal_rule="R1")) == []


def test_full_80_item_set_passes_final_rules():
    rep = validate_records(list(enumerate(full_gold_set(), start=1)), final=True)
    assert rep.errors == []
    assert len(rep.items) == 80


@pytest.mark.parametrize("kind, overrides, expected", [
    ("fact", {"id": "B05"}, "must start with 'A'"),
    ("refuse", {"refusal_rule": None}, "refusal_rule must be one of R1"),
    ("refuse", {"gold_sources": [BASE["fact"]["gold_sources"][0]]}, "must not have gold_sources"),
    ("refuse", {"gold_value": 1.0, "unit": "%"}, "must not have a gold_value"),
    ("fact", {"must_refuse": True, "refusal_rule": "R1", "gold_sources": [], "gold_value": None, "unit": None},
     "category must be 'refuse'"),
    ("trap", {"refusal_rule": "R3"}, "must have refusal_rule null"),
    ("trap", {"gold_answer": "Unknown"}, "must start with 'not reported'"),
    ("trap", {"gold_value": 5.0, "unit": "%"}, "trap must not have a gold_value"),
    ("trap", {"gold_sources": []}, "need at least one gold_source"),
    ("fact", {"gold_sources": []}, "need at least one gold_source"),
    ("ratio", {"gold_value": None, "unit": None}, "needs a numeric gold_value"),
    ("period_comparison", {"gold_value": None, "unit": None}, "needs a numeric gold_value"),
    ("fact", {"unit": None}, "unit is required"),
    ("fact", {"unit": "dollars"}, "not allowed"),
    ("qualitative", {"unit": "%"}, "unit is set but gold_value is empty"),
    ("fact", {"companies": ["TSLA"]}, "not in config/companies.yaml"),
    ("fact", {"companies": []}, "companies must list"),
    ("fact", {"fiscal_years": []}, "fiscal_years must list"),
    ("multi_company", {"companies": ["MSFT", "MSFT"]}, "duplicates"),
    ("multi_company", {"companies": ["MSFT"], "gold_sources": [BASE["ratio"]["gold_sources"][0]]},
     "at least 2 companies"),
    ("multi_turn", {"previous_turn_id": None}, "need previous_turn_id"),
    ("fact", {"previous_turn_id": "A09"}, "only multi_turn"),
    ("fact", {"verified_by": "  anirudh "}, "two-person rule"),
])
def test_item_rule_violations(kind, overrides, expected):
    errors = errors_for(make(kind, **overrides))
    assert any(expected in e for e in errors), errors


@pytest.mark.parametrize("source_override, expected", [
    ({"fiscal_period": "Q2"}, "10-K source must have fiscal_period 'FY'"),
    ({"form": "10-Q", "fiscal_period": "FY", "fiscal_year": 2025, "item": "Part I Item 1"}, "10-Q source must have"),
    ({"fiscal_year": 2022}, "fiscal year outside"),
    ({"form": "10-Q", "fiscal_period": "Q1", "fiscal_year": 2024, "item": "Part I Item 1"}, "10-Q fiscal year outside"),
    ({"item": "Item 7a"}, "not a valid 10-K Item name"),
    ({"form": "10-Q", "fiscal_period": "Q1", "fiscal_year": 2025, "item": "Item 2"}, "not a valid 10-Q Item name"),
    ({"ticker": "GOOGL"}, "is not listed in companies"),
])
def test_source_rule_violations(source_override, expected):
    rec = make("fact", fiscal_years=[2024, 2025])
    rec["gold_sources"] = [{**rec["gold_sources"][0], **source_override}]
    errors = errors_for(rec)
    assert any(expected in e for e in errors), errors


def test_source_year_must_be_listed_in_fiscal_years():
    rec = make("fact")
    rec["gold_sources"][0]["fiscal_year"] = 2023
    assert any("not listed in fiscal_years" in e for e in errors_for(rec))


def test_type_errors_and_unknown_fields_are_reported_with_line_numbers():
    errors = errors_for(make("fact", gold_value="lots"), make("ratio", colour="blue"))
    assert any(e.startswith("line 1 (A01): gold_value") for e in errors), errors
    assert any(e.startswith("line 2 (B01)") and "colour" in e for e in errors), errors


def test_bad_id_and_category_are_rejected():
    assert errors_for(make("fact", id="A1"))
    assert errors_for(make("fact", category="facts"))


def test_set_rules_duplicates_and_follow_up_links():
    errors = errors_for(make("fact"), make("fact"))
    assert any("duplicate id A01" in e for e in errors)
    assert any("same question text" in e for e in errors)
    assert any("does not exist" in e for e in errors_for(make("multi_turn", previous_turn_id="A07")))
    assert any("points to itself" in e for e in errors_for(make("multi_turn", previous_turn_id="F01")))
    cycle = errors_for(make("multi_turn", id="F01", previous_turn_id="F02"),
                       make("multi_turn", id="F02", previous_turn_id="F01", question="And in fiscal 2025 then?"))
    assert any("cycle" in e for e in cycle), cycle


def test_final_rules_counts_verification_coverage_and_traps():
    items = full_gold_set()
    items[0]["verified_by"] = None                                          # unverified
    items = [i for i in items if i["id"] != "B15"]                           # ratio count 14/15

    def as_refusal(rec: dict, rule: str) -> dict:                             # replace the whole record
        return make("refuse", id=rec["id"], refusal_rule=rule, question=rec["question"])

    items = [as_refusal(i, "R1") if i["id"] == "G07" else i for i in items]   # R7 no longer covered
    items = [as_refusal(i, "R2") if i["id"] in ("G08", "G09", "G10") else i for i in items]  # 0 traps
    errors = errors_for(*items, final=True)
    joined = "\n".join(errors)
    assert "not verified by a second person: ['A01']" in joined
    assert "ratio: 14 questions, target 15" in joined
    assert "refusal rules without a G question: ['R7']" in joined
    assert "'not reported' traps: 0" in joined
    assert errors_for(*items, final=False) == []                             # draft mode tolerates all four


@pytest.mark.parametrize("overrides, expected", [
    ({"must_refuse": "yes"}, "must_refuse"),                # strict types: strings are not bools/ints
    ({"fiscal_years": ["2024"]}, "fiscal_years"),
    ({"gold_value": True}, "gold_value"),
    ({"gold_value": float("nan")}, "gold_value"),           # NaN/Infinity are not valid JSON numbers
    ({"question": "            "}, "must not be blank"),
    ({"written_by": "  "}, "must not be blank"),
])
def test_strict_types_and_blank_text_are_rejected(overrides, expected):
    errors = errors_for(make("fact", **overrides))
    assert any(expected in e for e in errors), errors


def test_fiscal_years_must_be_in_the_corpus_and_unique():
    assert any("fiscal_years [2019] outside the corpus" in e
               for e in errors_for(make("fact", fiscal_years=[2024, 2019])))
    assert any("fiscal_years contains duplicates" in e for e in errors_for(make("fact", fiscal_years=[2024, 2024])))


def test_every_listed_company_needs_a_source():
    d = make("multi_company")
    errors = errors_for({**d, "gold_sources": d["gold_sources"][:1]})
    assert any("no gold_source for compan(ies) ['GOOGL']" in e for e in errors), errors


def test_follow_ups_may_share_question_text():
    f1 = make("multi_turn", question="And what was it in fiscal 2023?")
    f2 = {**f1, "id": "F02"}
    assert not any("same question text" in e for e in errors_for(make("fact"), f1, f2))


def test_default_tolerance_is_shared_with_the_converter():
    from eval.schema import DEFAULT_TOLERANCE_PCT, GoldItem
    rec = {k: v for k, v in make("fact").items() if k != "tolerance_pct"}
    assert GoldItem.model_validate(rec).tolerance_pct == DEFAULT_TOLERANCE_PCT
