"""eval/run_retrieval.py: per-entity filters, interleaving, the summary's generated block, the output contract
and the command line's exit codes (offline)."""
import json

import pytest
from pydantic import BaseModel

from eval import run_retrieval
from eval.retrieval_eval import read_runs
from eval.run_retrieval import (
    END,
    START,
    SUMMARY_NAME,
    best_row,
    interleave,
    item_filters,
    main,
    run,
    run_file,
    tables,
    write_runs,
    write_summary,
)
from eval.schema import GoldItem
from retrieval.bm25 import BM25Index, index_path
from retrieval.common import NO_FILTERS, Chunk, Filters, SearchResult, chunks_path, write_chunks


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


# --- output contract and the command line -------------------------------------------------------------------------

def row(variant, retriever, filters, r5, mrr, by_category=None):
    overall = {"recall@1": 0.1, "recall@5": r5, "recall@10": 0.9, "mrr": mrr}
    return {"variant": variant, "retriever": retriever, "filters": filters, "seconds": 0.25,
            "result": {"overall": overall, "n": 73, "by_category": by_category or {}}}


def test_best_row_is_recall_at_5_then_mrr_then_first():
    rows = [row("C1", "bm25", "none", 0.7, 0.5), row("C2", "bm25", "none", 0.7, 0.6),
            row("C3", "bm25", "none", 0.7, 0.6)]
    assert best_row(rows)["variant"] == "C2"


def test_tables_are_pinned():
    cats = {"fact": {"n": 2, "recall@1": 0.5, "recall@5": 1.0, "recall@10": 1.0, "mrr": 0.75},
            "unknown_category": {"n": 1, "recall@1": 0, "recall@5": 0, "recall@10": 0, "mrr": 0}}
    best = row("C3", "hybrid", "company+year", 0.747, 0.539, cats)
    assert tables([best], best) == (
        "| Variant | Retriever | Filters | recall@1 | recall@5 | recall@10 | MRR | s/query |\n"
        "|---|---|---|---:|---:|---:|---:|---:|\n"
        "| C3 | hybrid | company+year | 0.100 | 0.747 | 0.900 | 0.539 | 0.25 |\n"
        "\n"
        "By question category, best setting (C3 hybrid, filters company+year):\n"
        "\n"
        "| Category | n | recall@1 | recall@5 | recall@10 | MRR |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        "| fact | 2 | 0.500 | 1.000 | 1.000 | 0.750 |\n"     # categories outside CATEGORY_ORDER are not listed
        "\n"
        "73 gold questions with gold sources (gold-v1); items without sources (out-of-scope refusals) are skipped.\n")


def test_run_files_are_named_and_shaped_for_retrieval_eval(tmp_path):
    path = run_file(tmp_path, "C3", "hybrid", "company+year")
    assert path == tmp_path / "runs" / "C3_hybrid_company_year.jsonl"
    retrieved = [{"chunk_id": "c", "ticker": "AAPL", "fiscal_year": 2024, "fiscal_period": "FY", "item": "Item 8",
                  "score": 0.5}]
    write_runs(path, [{"id": "A01", "retrieved": retrieved}], "hybrid", "company+year")
    runs, meta = read_runs(path)
    assert runs == {"A01": [("AAPL", 2024, "FY", "Item 8")]}
    assert json.loads(path.read_text(encoding="utf-8")) == {"id": "A01", "retriever": "hybrid",
                                                            "filters": "company+year", "retrieved": retrieved}


@pytest.fixture
def bm25_dirs(tmp_path):
    """A one-chunk C1 BM25 index; no dense index (so dense/hybrid runs find nothing)."""
    c = Chunk("AAPL_10-K_2024_Item8_C1_0000", "AAPL_10-K_2024_Item8", "C1", "text",
              "[AAPL 10-K FY2024 · Item 8 · Financial Statements]", "Total net sales 391,035", "AAPL", "10-K",
              2024, "FY", "Item 8", "2024-09-28", "u")
    BM25Index([c]).save(index_path("C1", tmp_path / "index"))
    write_chunks([c], chunks_path("C1", tmp_path / "chunks"))
    return tmp_path


def cli(dirs, *extra):
    return main(["--index-dir", str(dirs / "index"), "--chunks-dir", str(dirs / "chunks"),
                 "--out-dir", str(dirs / "out"), *extra])


def test_a_partial_run_writes_run_files_but_not_the_summary(bm25_dirs):
    (bm25_dirs / "out").mkdir()
    summary = bm25_dirs / "out" / SUMMARY_NAME
    summary.write_text(f"{START}\nfull table\n{END}\n", encoding="utf-8")
    assert cli(bm25_dirs, "--variants", "C1", "--retrievers", "bm25") == 0
    assert summary.read_text(encoding="utf-8") == f"{START}\nfull table\n{END}\n"
    assert run_file(bm25_dirs / "out", "C1", "bm25", "company+year").exists()


def test_exit_codes(bm25_dirs, monkeypatch, capsys):
    assert cli(bm25_dirs, "--variants", "C2", "--retrievers", "bm25") == 2           # no C2 index
    bad_gold = bm25_dirs / "gold.jsonl"
    bad_gold.write_text('{"id": "A01"}\n', encoding="utf-8")
    assert cli(bm25_dirs, "--retrievers", "bm25", "--gold", str(bad_gold)) == 1

    class Strict(BaseModel):
        EMBED_BACKEND: int

    def incomplete_settings():
        Strict(EMBED_BACKEND="x")                    # raises pydantic.ValidationError, as get_settings() would

    monkeypatch.setattr(run_retrieval, "embedder_from_settings", incomplete_settings)
    capsys.readouterr()
    assert cli(bm25_dirs, "--variants", "C1", "--retrievers", "dense") == 2
    assert "settings are incomplete; copy .env.example to .env" in capsys.readouterr().err


def test_a_summary_without_markers_is_exit_2(bm25_dirs, monkeypatch, capsys):
    (bm25_dirs / "out").mkdir()
    (bm25_dirs / "out" / SUMMARY_NAME).write_text("markers removed\n", encoding="utf-8")
    monkeypatch.setattr(run_retrieval, "evaluate", lambda *a, **k: [row("C1", "bm25", "none", 0.5, 0.4)])
    assert cli(bm25_dirs) == 2
    assert "GENERATED markers" in capsys.readouterr().err


def test_a_dense_only_run_does_not_need_the_bm25_index(tmp_path):
    out = run_retrieval.load_retrievers("C1", ["dense"], None, tmp_path / "no_index", tmp_path / "no_chunks")
    assert out == {}                                            # no embedder: nothing to load, and no BM25 error
