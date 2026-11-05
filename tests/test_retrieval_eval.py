"""recall@k / MRR harness (eval/retrieval_eval.py)."""
import json

import pytest

from eval.retrieval_eval import (
    build_query,
    hit_key,
    main,
    read_runs,
    recall_at_k,
    reciprocal_rank,
    score_runs,
    to_markdown,
)
from eval.schema import GoldItem
from tests.factories import make

MSFT = ("MSFT", 2024, "FY", "Item 8")
GOOGL = ("GOOGL", 2024, "FY", "Item 8")
OTHER = ("AAPL", 2024, "FY", "Item 7")


def test_recall_and_reciprocal_rank():
    retrieved = [OTHER, MSFT, OTHER, GOOGL]
    assert recall_at_k(retrieved, [MSFT], 1) == 0.0
    assert recall_at_k(retrieved, [MSFT], 2) == 1.0
    assert recall_at_k(retrieved, [MSFT, GOOGL], 2) == 0.5          # multi-company: half the sources found
    assert recall_at_k(retrieved, [MSFT, GOOGL], 10) == 1.0
    assert reciprocal_rank(retrieved, [MSFT, GOOGL]) == 0.5          # first hit at rank 2
    assert reciprocal_rank([OTHER], [MSFT]) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(retrieved, [], 5)


def test_hit_key_normalises_case_and_spacing():
    assert hit_key({"ticker": "msft", "fiscal_year": "2024", "fiscal_period": "fy", "item": "Item  8"}) == MSFT


def items(*records):
    return [GoldItem.model_validate(r) for r in records]


def test_score_runs_skips_refusals_and_counts_missing_runs():
    gold = items(make("ratio"), make("multi_company"), make("refuse"))
    result = score_runs(gold, {"B01": [MSFT], "D01": [GOOGL, OTHER]})
    assert result["n"] == 2 and result["missing"] == []
    assert result["overall"]["recall@1"] == 0.75                     # (1 + 0.5) / 2
    assert result["overall"]["mrr"] == 1.0
    assert set(result["by_category"]) == {"ratio", "multi_company"}
    missing = score_runs(gold, {"B01": [MSFT]})
    assert missing["missing"] == ["D01"] and missing["overall"]["recall@10"] == 0.5


def test_build_query_prepends_previous_turns_and_survives_cycles():
    a, f1 = make("fact"), make("multi_turn")
    f2 = make("multi_turn", id="F02", previous_turn_id="F01", question="What about its net income then?")
    by_id = {i.id: i for i in items(a, f1, f2)}
    assert build_query(by_id["A01"], by_id) == a["question"]
    assert build_query(by_id["F02"], by_id) == " ".join([a["question"], f1["question"], f2["question"]])
    loop = {i.id: i for i in items(make("multi_turn", id="F01", previous_turn_id="F02"),
                                   make("multi_turn", id="F02", previous_turn_id="F01", question="Again for 2025?"))}
    assert build_query(loop["F01"], loop).count("Again for 2025?") == 1


def test_read_runs_and_markdown(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps({"id": "B01", "retriever": "hybrid", "filters": "company+year",
                                "retrieved": [{"ticker": "MSFT", "fiscal_year": 2024, "fiscal_period": "FY",
                                               "item": "Item 8", "chunk_id": "x", "score": 0.9}]}) + "\n",
                    encoding="utf-8")
    runs, meta = read_runs(path)
    assert runs == {"B01": [MSFT]} and meta == {"retriever": "hybrid", "filters": "company+year"}
    table = to_markdown(score_runs(items(make("ratio")), runs), meta)
    assert "Retriever: hybrid · filters: company+year" in table and "| **overall** | 1 | 1.000" in table
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "B01", "retrieved": [{"ticker": "MSFT"}]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad run record"):
        read_runs(bad)


def test_cli_end_to_end(tmp_path, capsys):
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps(make("ratio")) + "\n", encoding="utf-8")
    runs = tmp_path / "runs.jsonl"
    runs.write_text(json.dumps({"id": "B01", "retrieved": [{"ticker": "MSFT", "fiscal_year": 2024,
                                                            "fiscal_period": "FY", "item": "Item 8"}]}) + "\n",
                    encoding="utf-8")
    out = tmp_path / "results.md"
    assert main(["--gold", str(gold), "--runs", str(runs), "--out", str(out)]) == 0
    assert "| **overall** | 1 | 1.000 | 1.000 | 1.000 | 1.000 |" in out.read_text(encoding="utf-8")
    bad_gold = tmp_path / "bad.jsonl"
    bad_gold.write_text(json.dumps(make("ratio", gold_value=None)) + "\n", encoding="utf-8")
    assert main(["--gold", str(bad_gold), "--runs", str(runs)]) == 1


def test_read_runs_rejects_bad_json_and_duplicate_ids(tmp_path):
    line = json.dumps({"id": "B01", "retrieved": []})
    for name, text in (("json", "{not json\n"), ("dup", f"{line}\n{line}\n")):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="bad run record"):
            read_runs(path)


def test_quarter_must_match_for_a_10q_hit():
    q2 = make("quarterly")
    item = GoldItem.model_validate(q2)
    src = q2["gold_sources"][0]
    q1_chunk = hit_key({**src, "fiscal_period": "Q1"})
    assert score_runs([item], {item.id: [q1_chunk]})["overall"]["recall@10"] == 0.0
    assert score_runs([item], {item.id: [hit_key(src)]})["overall"]["recall@10"] == 1.0


def test_hit_key_strips_every_part():
    assert hit_key({"ticker": " msft ", "fiscal_year": "2024", "fiscal_period": " fy", "item": "Item  8"}) == MSFT


@pytest.mark.parametrize("year", [2024.9, True, "24", "FY2024", None, 24, 99999, "\u0662\u0660\u0662\u0664"])
def test_hit_key_rejects_years_that_would_match_by_accident(year):
    with pytest.raises(ValueError, match="fiscal_year must be a year"):
        hit_key({"ticker": "MSFT", "fiscal_year": year, "fiscal_period": "FY", "item": "Item 8"})
