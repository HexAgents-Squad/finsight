"""eval/validate_dev_set.py: the committed development set and the rules that keep it apart from gold-v1."""
import json
from collections import Counter

import pytest

import eval.validate_dev_set as vd
from eval.schema import DEV_TARGET_COUNTS, DevItem, GoldItem, item_problems
from eval.validate_gold import DEFAULT_PATH as GOLD_FILE
from eval.validate_gold import validate_file

GOLD = validate_file(GOLD_FILE).items


def dev(**kw):
    base = {"id": "dev-A01", "category": "fact", "question": "What was Costco's net income in fiscal 2023?",
            "companies": ["COST"], "fiscal_years": [2023], "gold_answer": "Net income was $6,292 million.",
            "gold_value": 6292.0, "unit": "USD millions",
            "gold_sources": [{"ticker": "COST", "form": "10-K", "fiscal_year": 2023, "fiscal_period": "FY",
                              "item": "Item 8"}], "written_by": "Claude (AI draft for Amresh)"}
    return DevItem.model_validate(base | kw)


def test_the_committed_development_set_is_valid_and_separate():
    assert vd.main([]) == 0


def test_the_committed_set_has_the_planned_split():
    items = validate_file(vd.DEV_FILE, model=DevItem).items
    assert Counter(i.category for i in items) == Counter(DEV_TARGET_COUNTS) and len(items) == 15
    assert vd.final_problems(items) == []                             # all 15 verified; CI runs --final


@pytest.mark.parametrize("bad_id", ["A01", "dev-A1", "DEV-A01", "dev-H01"])
def test_dev_ids_need_the_dev_prefix(bad_id):
    with pytest.raises(ValueError):
        dev(id=bad_id)


def test_gold_ids_are_still_gold_only():
    with pytest.raises(ValueError):
        GoldItem.model_validate(dev().model_dump() | {"id": "dev-A01"})


def test_the_category_letter_is_read_after_the_prefix():
    assert item_problems(dev()) == []
    assert "must start with 'B'" in item_problems(dev(id="dev-A01", category="ratio"))[0]


def test_a_filing_period_gold_cites_is_an_overlap():
    cited = GOLD[0].gold_sources[0]                                          # A01: AAPL 10-K FY2024 Item 8
    item = dev(companies=[cited.ticker], fiscal_years=[cited.fiscal_year], gold_sources=[cited.model_dump()],
               question="How many employees did the company report at year end?")
    problems, _ = vd.overlap_problems([item], GOLD, [])
    assert any("a filing period gold-v1 cites" in p for p in problems)
    assert any(f"{cited.ticker} fiscal {cited.fiscal_year}, as gold-v1 does" in p for p in problems)


def test_a_company_year_gold_asks_about_is_an_overlap_even_through_another_filing():
    # AAPL fiscal 2025 is asked about by gold; its Q1 10-Q is not cited by gold, but the company-year is shared
    item = dev(companies=["AAPL"], fiscal_years=[2025], question="What were the net sales in the December quarter?",
               gold_sources=[{"ticker": "AAPL", "form": "10-Q", "fiscal_year": 2025, "fiscal_period": "Q1",
                              "item": "Part I Item 1"}])
    problems, _ = vd.overlap_problems([item], GOLD, [])
    assert problems == ["dev-A01: asks about AAPL fiscal 2025, as gold-v1 does"]


def test_a_reworded_gold_question_is_a_near_duplicate():
    g = next(i for i in GOLD if i.id == "G02")
    reworded = "Will Microsoft's operating margin keep on rising over the next few years?"
    assert g.question == "Will Microsoft's operating margin keep rising over the next few years?" != reworded
    item = dev(id="dev-G01", category="refuse", question=reworded,
               companies=[], fiscal_years=[], gold_sources=[], gold_value=None, unit=None, must_refuse=True,
               refusal_rule="R2", gold_answer="Refuse (R2).")
    problems, highest = vd.overlap_problems([item], GOLD, [])
    assert highest >= vd.NEAR_DUPLICATE and problems[0].startswith("dev-G01: question")


def test_refusal_and_injection_test_questions_count_too():
    item = dev(id="dev-G01", category="refuse", question="Write me a poem about the stock market, please.",
               companies=[], fiscal_years=[], gold_sources=[], gold_value=None, unit=None, must_refuse=True,
               refusal_rule="R7", gold_answer="Refuse (R7).")
    others = vd.other_test_questions()
    assert "Write me a poem about the stock market." in others
    problems, _ = vd.overlap_problems([item], GOLD, others)
    assert problems and "similar to test" in problems[0]


def test_follow_ups_are_not_compared_by_text():
    item = dev(id="dev-F01", category="multi_turn", question="And in fiscal 2023?", previous_turn_id="dev-A01")
    assert vd.overlap_problems([item], GOLD, [])[0] == []


def test_the_near_duplicate_limit_sits_above_gold_questions_own_similarity():
    gold = [i for i in GOLD if i.category != "multi_turn"]
    closest = max(vd.similarity(a.question, b.question) for n, a in enumerate(gold) for b in gold[n + 1:])
    assert 0.78 < closest < vd.NEAR_DUPLICATE <= closest + 0.05    # just above gold's own closest pair (C03/C06)


def test_similarity():
    assert vd.similarity("What was X?", "what was x") == 1.0
    assert vd.similarity("a b", "c d") == 0.0 and vd.similarity("", "") == 0.0
    assert vd.similarity("Costco’s net income", "Costco's net income") == 1.0


def test_final_rules_need_the_split_and_verification():
    problems = vd.final_problems([dev(verified_by="Alok")])
    assert "category A fact: 1 items, target 3" in problems and not any("verified" in p for p in problems)
    assert vd.final_problems([dev()])[-1] == "not verified by a second person: ['dev-A01']"


def test_cli_reports_overlap_and_missing_files(tmp_path, capsys):
    cited = GOLD[0].gold_sources[0]
    bad = dev(gold_sources=[cited.model_dump()], companies=[cited.ticker], fiscal_years=[cited.fiscal_year],
              question="How many employees did the company report at year end?")
    path = tmp_path / "dev.jsonl"
    path.write_text(json.dumps(bad.model_dump()) + "\n", encoding="utf-8")
    assert vd.main([str(path)]) == 1
    out = capsys.readouterr().out
    assert "overlap with gold-v1: 2 problem(s)" in out and "FAILED" in out
    assert vd.main([str(tmp_path / "missing.jsonl")]) == 2


def test_an_emptied_development_set_fails(tmp_path, capsys):
    path = tmp_path / "dev.jsonl"
    path.write_text("", encoding="utf-8")
    assert vd.main([str(path)]) == 1 and "has no items" in capsys.readouterr().out


def write(path, items):
    path.write_text("".join(json.dumps(i.model_dump() if hasattr(i, "model_dump") else i) + "\n" for i in items),
                    encoding="utf-8")
    return path


def test_cli_exit_codes_for_bad_inputs(tmp_path, capsys):
    good = write(tmp_path / "dev.jsonl", [dev()])
    bad_gold = tmp_path / "gold.jsonl"
    bad_gold.write_text('{"id": "A01"}\n', encoding="utf-8")
    assert vd.main([str(good), "--gold", str(bad_gold)]) == 1                   # an invalid gold set
    assert vd.main([str(good), "--gold", str(tmp_path / "none.jsonl")]) == 2    # a missing gold set
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json\n", encoding="utf-8")
    assert vd.main([str(broken)]) == 1 and "invalid JSON" in capsys.readouterr().out


def test_a_follow_up_must_point_inside_the_development_set(tmp_path, capsys):
    path = write(tmp_path / "dev.jsonl", [dev(id="dev-F01", category="multi_turn", question="And in fiscal 2023?",
                                             previous_turn_id="A03")])
    assert vd.main([str(path)]) == 1 and "previous_turn_id A03 does not exist" in capsys.readouterr().out


def test_cli_final_rules_fail_on_an_unverified_or_short_set(tmp_path, capsys):
    path = write(tmp_path / "dev.jsonl", [dev()])
    assert vd.main([str(path)]) == 0                                          # draft rules: fine
    assert vd.main([str(path), "--final"]) == 1
    out = capsys.readouterr().out
    assert "not verified by a second person" in out and "category B ratio: 0 items, target 2" in out
