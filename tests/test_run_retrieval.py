"""eval/run_retrieval.py: per-entity filters, interleaving, the summary's generated block, the output contract
and the command line's exit codes (offline)."""
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from eval import run_retrieval
from eval.retrieval_eval import read_runs
from eval.run_retrieval import (
    END,
    GRID,
    RERANKED,
    SETTINGS,
    START,
    SUMMARY_NAME,
    TRIALS,
    Setting,
    best_row,
    entity_accuracy,
    interleave,
    item_filters,
    main,
    run,
    run_file,
    setting_filters,
    settings_for,
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
        "| Variant | Retriever | Filters | recall@1 | recall@5 | recall@10 | MRR | figure@5 | s/query |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|\n"
        "| C3 | hybrid | company+year | 0.100 | 0.747 | 0.900 | 0.539 | - | 0.25 |\n"
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
    # a full run builds the embedder first; a stub keeps the test independent of any .env
    monkeypatch.setattr(run_retrieval, "embedder_from_settings",
                        lambda: SimpleNamespace(embed=lambda texts: [], model_id="m", close=lambda: None))
    monkeypatch.setattr(run_retrieval, "load_reranker", lambda: (lambda query, passages: []))
    assert cli(bm25_dirs, "--rerank") == 2
    assert "GENERATED markers" in capsys.readouterr().err


def test_a_dense_only_run_does_not_need_the_bm25_index(tmp_path):
    out = run_retrieval.load_retrievers("C1", ["dense"], None, tmp_path / "no_index", tmp_path / "no_chunks")
    assert out == {}                                            # no embedder: nothing to load, and no BM25 error


# --- W1-05: filter settings, trials, entity reading -----------------------------------------------------------


def asked(id, question, companies, years, periods=("FY",), previous=None):
    return GoldItem.model_validate({
        "id": id, "category": "multi_turn" if previous else "fact", "question": question,
        "companies": list(companies), "fiscal_years": list(years), "gold_answer": "x",
        "gold_sources": [{"ticker": t, "form": "10-K" if p == "FY" else "10-Q", "fiscal_year": y,
                          "fiscal_period": p, "item": "Item 8" if p == "FY" else "Part I Item 1"}
                         for t in companies for y in years for p in periods],
        "previous_turn_id": previous, "written_by": "w"})


def test_setting_names_and_which_runs_where():
    assert [s.name for s in GRID] == ["none", "question", "company+year"]
    assert [s.name for s in TRIALS] == ["question+period", "question+period+statement", "company+year+period",
                                        "company+year+period+statement"]
    assert settings_for("C1", "bm25") == GRID and settings_for("C3", "dense") == GRID
    assert settings_for("C3", "hybrid") == GRID + TRIALS
    assert settings_for("C3", RERANKED) == GRID[1:] + TRIALS
    with pytest.raises(ValueError):
        Setting("question", statement=True)                         # the statement trial needs the period
    with pytest.raises(ValueError):
        Setting("none", period=True)
    with pytest.raises(ValueError):
        Setting("x")


def test_question_derived_filters_read_the_turns_and_gold_filters_take_the_question_period():
    a = asked("A01", "What was Meta's revenue in the second quarter of fiscal 2025?", ("META",), (2025,), ("Q2",))
    f = asked("F01", "And in the third quarter?", ("META",), (2025,), ("Q3",), previous="A01")
    by_id = {"A01": a, "F01": f}
    assert setting_filters(f, by_id, SETTINGS["question"]) == [Filters.of(ticker="META", fiscal_year=2025)]
    assert setting_filters(f, by_id, SETTINGS["question+period"]) == [
        Filters.of(ticker="META", fiscal_year=2025, fiscal_period="Q3")]
    assert setting_filters(f, by_id, SETTINGS["company+year+period"]) == [
        Filters.of(ticker="META", fiscal_year=2025, fiscal_period="Q3")]
    assert setting_filters(f, by_id, SETTINGS["none"]) == []
    nothing = asked("A02", "What was the total revenue?", ("AAPL",), (2024,))
    assert setting_filters(nothing, {"A02": nothing}, SETTINGS["question"]) == []     # one unfiltered search
    assert setting_filters(nothing, {"A02": nothing}, SETTINGS["company+year"]) == [
        Filters.of(ticker="AAPL", fiscal_year=2024)]


def test_run_accepts_a_setting_or_its_name_and_names_trial_files(tmp_path):
    items = [asked("A01", "What was Apple's revenue in fiscal 2024?", ("AAPL",), (2024,))]
    rec = Recorder()
    assert run(rec, items, "question+period")[0] == run(rec, items, SETTINGS["question+period"])[0]
    assert rec.calls[-1] == ("What was Apple's revenue in fiscal 2024?",
                             Filters.of(ticker="AAPL", fiscal_year=2024, fiscal_period="FY"))
    assert run_file(tmp_path, "C3", RERANKED, "company+year+period+statement") == \
        tmp_path / "runs" / "C3_hybrid_rerank_company_year_period_statement.jsonl"


def test_entity_accuracy_counts_exact_matches():
    items = [asked("A01", "What was Apple's revenue in fiscal 2024?", ("AAPL",), (2024,)),
             asked("A02", "What was Meta's revenue in fiscal 2025?", ("META",), (2025,), ("Q2",)),   # period missed
             asked("A03", "What was the revenue?", ("KO",), (2023,))]                               # nothing named
    assert entity_accuracy(items) == {"n": 3, "companies": 2, "years": 2, "periods": 2, "all": 1}


def test_tables_add_the_trials_the_realistic_best_and_the_entity_table():
    cats = {"fact": {"n": 2, "recall@1": 0.5, "recall@5": 1.0, "recall@10": 1.0, "mrr": 0.75}}
    rows = [row("C3", "hybrid", "question", 0.70, 0.5, cats), row("C3", "hybrid", "company+year", 0.77, 0.55, cats),
            row("C3", "hybrid", "question+period", 0.72, 0.52, cats), row("C3", RERANKED, "question", 0.75, 0.6, cats)]
    text = tables(rows, best_row(rows), {"gold-v1": {"n": 73, "companies": 70, "years": 69, "periods": 68,
                                                    "all": 66}})
    assert "| C3 | hybrid+rerank |" not in text                       # the reranker is a trial, not a grid row
    assert "| hybrid | question+period | 0.100 | 0.720 | 0.900 | 0.520 | - | 0.25 |" in text
    assert "| hybrid+rerank | question | 0.100 | 0.750 | 0.900 | 0.600 | - | 0.25 |" in text
    assert ("best setting with question-derived filters and no reranker (C3 hybrid, filters question+period)"
            in text)
    assert "| gold-v1 | 73 | 70 | 69 | 68 | 66 |" in text
    best = row("C3", "hybrid", "question", 0.80, 0.6, cats)               # the best is already that setting
    assert tables([best, rows[1]], best).count("By question category") == 1


def test_the_pairs_table_compares_recall_at_5_per_question():
    items = [asked("A01", "Apple question one?", ("AAPL",), (2024,)),
             asked("A02", "Microsoft question two?", ("MSFT",), (2024,))]
    good = {"A01": [("AAPL", 2024, "FY", "Item 8")], "A02": [("MSFT", 2024, "FY", "Item 8")]}
    bad = {"A01": [("AAPL", 2024, "FY", "Item 7")], "A02": [("MSFT", 2024, "FY", "Item 8")]}
    rows = [{**row("C3", "hybrid", "question", 0.5, 0.5), "keys": bad},
            {**row("C3", "hybrid", "company+year", 1.0, 1.0), "keys": good}]
    text = tables(rows, best_row(rows), items=items)
    assert "| hybrid, question | hybrid, company+year | 1 / 0 / 1 | 1.000 | - | - |" in text
    rows[0]["figures"], rows[1]["figures"] = set(), {"A01"}
    for r in rows:
        r["figure_n"] = 2
    text = tables(rows, best_row(rows), items=items)
    assert "| hybrid, question | hybrid, company+year | 1 / 0 / 1 | 1.000 | 1 / 0 | 1.000 |" in text
    assert "| C3 | hybrid | company+year | 0.100 | 1.000 | 0.900 | 1.000 | 1/2 | 0.25 |" in text
    assert "Paired on C3" not in tables(rows, best_row(rows))                  # no items: no pairs table


def test_the_figure_check_needs_the_figure_in_the_first_five_chunks():
    from eval.run_retrieval import figure_hits, figure_items
    items = [GoldItem.model_validate({**asked("A01", "What was Apple's revenue in fiscal 2024?", ("AAPL",),
                                              (2024,)).model_dump(), "gold_answer": "Total net sales were $391,035 million."}),
             GoldItem.model_validate({**asked("A02", "What drove Apple's sales in fiscal 2024?", ("AAPL",),
                                              (2024,)).model_dump(), "gold_answer": "Strong iPhone demand."})]
    assert figure_items(items) == {"A01"}                              # no figure in the answer: not counted
    texts = {f"c{n}": "nothing" for n in range(6)} | {"c5": "Total net sales | 391,035", "big": "1,391,035"}
    late = [{"id": "A01", "retrieved": [{"chunk_id": f"c{n}"} for n in range(6)]}]          # 6th chunk: too late
    assert figure_hits(items, late, texts) == set()
    early = [{"id": "A01", "retrieved": [{"chunk_id": "c5"}]}]
    assert figure_hits(items, early, texts) == {"A01"}
    inside = [{"id": "A01", "retrieved": [{"chunk_id": "big"}]}]                            # not a longer number
    assert figure_hits(items, inside, texts) == set()


def test_a_statement_setting_runs_through_the_statement_first_wrapper():
    class ByItem:
        def search(self, query, k=5, filters=NO_FILTERS):
            return [SearchResult(f"{filters.item or 'any'}-0", "t", "AAPL", "10-K", 2024, "FY",
                                 filters.item or "Item 7", "u", 1.0)]
    items = [asked("A01", "What was Apple's net income in fiscal 2024?", ("AAPL",), (2024,))]
    with_statements = run(ByItem(), items, "question+period+statement")[0][0]["retrieved"]
    without = run(ByItem(), items, "question+period")[0][0]["retrieved"]
    assert [h["chunk_id"] for h in with_statements] == ["Item 8-0", "any-0"]
    assert [h["chunk_id"] for h in without] == ["any-0"]
