"""eval/run_retrieval.py: per-entity filters, interleaving and the summary's generated block (offline)."""
import pytest

from eval.run_retrieval import END, START, interleave, item_filters, run, write_summary
from eval.schema import GoldItem
from retrieval.common import NO_FILTERS, Filters, SearchResult


def item(id="D01", companies=("AAPL", "MSFT"), years=(2024,), previous=None, sources=True):
    return GoldItem.model_validate({
        "id": id, "category": "multi_turn" if previous else "multi_company", "question": f"Question {id}?",
        "companies": list(companies), "fiscal_years": list(years), "gold_answer": "x",
        "gold_sources": [{"ticker": t, "form": "10-K", "fiscal_year": y, "fiscal_period": "FY", "item": "Item 8"}
                         for t in companies for y in years] if sources else [],
        "previous_turn_id": previous, "written_by": "w"})


def hit(cid, ticker="AAPL"):
    return SearchResult(cid, "t", ticker, "10-K", 2024, "FY", "Item 8", "u", 1.0)


def test_one_filter_per_company_and_year():
    assert item_filters(item(companies=("AAPL", "MSFT"), years=(2023, 2024))) == [
        Filters.of(ticker="AAPL", fiscal_year=2023), Filters.of(ticker="AAPL", fiscal_year=2024),
        Filters.of(ticker="MSFT", fiscal_year=2023), Filters.of(ticker="MSFT", fiscal_year=2024)]


def test_interleave_alternates_and_drops_repeats():
    merged = interleave([[hit("a1"), hit("a2"), hit("s")], [hit("b1"), hit("s"), hit("b3")]], k=5)
    assert [h.chunk_id for h in merged] == ["a1", "b1", "a2", "s", "b3"]


class Recorder:
    def __init__(self):
        self.calls = []

    def search(self, query, k=5, filters=NO_FILTERS):
        self.calls.append((query, filters))
        return [hit(f"{filters.ticker}-{k}", filters.ticker or "AAPL")]


def test_run_uses_previous_turns_filters_and_skips_refusals():
    items = [item("A01", ("AAPL",)), item("F01", ("MSFT",), previous="A01"),
             item("G01", (), (), sources=False)]
    rec = Recorder()
    records, _ = run(rec, items, "company+year")
    assert [r["id"] for r in records] == ["A01", "F01"]
    assert rec.calls == [("Question A01?", Filters.of(ticker="AAPL", fiscal_year=2024)),
                         ("Question A01? Question F01?", Filters.of(ticker="MSFT", fiscal_year=2024))]
    assert records[1]["retrieved"][0]["ticker"] == "MSFT"
    run(rec, items, "none")
    assert rec.calls[-1] == ("Question A01? Question F01?", NO_FILTERS)


def test_summary_keeps_text_outside_the_generated_block(tmp_path):
    path = tmp_path / "summary.md"
    path.write_text(f"# Title\n\nIntro.\n\n{START}\nold table\n{END}\n\n## Interpretation\nKeep me.\n",
                    encoding="utf-8")
    write_summary("new table\n", path)
    assert path.read_text(encoding="utf-8") == \
        f"# Title\n\nIntro.\n\n{START}\nnew table\n{END}\n\n## Interpretation\nKeep me.\n"
    path.write_text("markers removed", encoding="utf-8")
    with pytest.raises(ValueError, match="GENERATED markers"):
        write_summary("x", path)
