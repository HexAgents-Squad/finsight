"""eval/context_pack.py and core/tokens.py, offline: fake retriever, fake token counter, real gold/refusal files."""
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

import core.tokens
import eval.context_pack as cp
from core.exceptions import FinSightError
from eval.retrieval_eval import build_query
from eval.schema import CorpusScope, corpus_scope
from eval.validate_gold import validate_file
from eval.validate_test_sets import validate_set
from retrieval.common import NO_FILTERS, Filters, SearchResult

ROOT = Path(__file__).resolve().parents[1]
GOLD = validate_file(cp.GOLD_PATH).items
REFUSALS, _ = validate_set("refusal", cp.REFUSAL_PATH)
SCOPE = corpus_scope()
COMPANY_NAMES = {"NFLX": ["Netflix"], "TSLA": ["Tesla"]}          # the out-of-corpus companies questions name
ALIASES = {c["ticker"]: c["aliases"] for c in yaml.safe_load(
    (ROOT / "config" / "companies.yaml").read_text(encoding="utf-8"))["companies"]} | COMPANY_NAMES
NAME_PATTERNS = {t: re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b", re.IGNORECASE)
                 for t, names in ALIASES.items()}
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
QUARTER = re.compile(r"\bQ[1-4]\b")


def words(text: str) -> int:
    """Deterministic stand-in for the Qwen tokenizer: one token per whitespace-separated word."""
    return len(text.split())


def hit(chunk_id: str, ticker: str = "AAPL", year: int = 2024, period: str = "FY", item: str = "Item 8",
        text: str = "net sales 391,035") -> SearchResult:
    form = "10-K" if period == "FY" else "10-Q"
    return SearchResult(chunk_id, text, ticker, form, year, period, item, "https://sec.test/doc", 0.5, "2024-09-28")


class FakeRetriever:
    """Returns hits for the filter's ticker, three per search, and records every search."""

    def __init__(self):
        self.searches = []

    def search(self, query, k=5, filters=NO_FILTERS):
        self.searches.append((query, k, filters))
        t = filters.ticker or "AAPL"
        return [hit(f"{t}_{n}", ticker=t, item="Item 8" if n == 0 else "Item 7") for n in range(3)]


def question(qid: str):
    return next(q for q in cp.gold_questions(GOLD) + cp.refusal_questions(REFUSALS) if q.id == qid)


# --- rendering ---------------------------------------------------------------------------------------------

def test_passages_render_exactly_as_the_prompt_will_show_them():
    p1 = cp.Passage("P1", "c1", "[AAPL 10-K FY2024 · Item 8]", "AAPL", "10-K", 2024, "FY", "Item 8", "Net sales")
    p2 = cp.Passage("P2", "c2", "[AAPL 10-K FY2024 · Item 7]", "AAPL", "10-K", 2024, "FY", "Item 7", "MD&A")
    assert cp.render_passages([p1, p2]) == "[P1] [AAPL 10-K FY2024 · Item 8]\nNet sales\n\n[P2] [AAPL 10-K FY2024 · Item 7]\nMD&A"


# --- filters -------------------------------------------------------------------------------------------------

def test_every_question_without_companies_is_annotated_and_nothing_else():
    need = {i.id for i in GOLD if not i.companies} | {t.id for t in REFUSALS}
    assert set(cp.QUESTION_FILTERS) == need                 # no missing and no stale entries


@pytest.mark.parametrize("qid", sorted(cp.QUESTION_FILTERS))
def test_annotations_name_exactly_what_the_question_says(qid):
    """Both directions: every annotated company, year and quarter is in the question, and every one the question
    names is annotated, except the documented readings ("latest" = FY2025, "Q4" = the annual report)."""
    text = next(x.question for x in [*GOLD, *REFUSALS] if x.id == qid)
    tickers, years, period = cp.QUESTION_FILTERS[qid]
    named = (sorted(t for t, pattern in NAME_PATTERNS.items() if pattern.search(text)),
             sorted(int(y) for y in YEAR.findall(text)), QUARTER.findall(text))
    annotated = (sorted(tickers), sorted(years), [period] if period else [])
    if qid in cp.ANNOTATION_READINGS:
        assert named != annotated, f"{qid} needs no reading any more; remove it from ANNOTATION_READINGS"
        assert named[0] == annotated[0]                  # a reading changes a year or quarter, never the company
    else:
        assert named == annotated


def test_the_readings_are_the_spec_rules():
    assert cp.QUESTION_FILTERS["RF22"] == (("MSFT",), (2025,), None)     # latest indexed 10-K (spec §4.3)
    assert cp.QUESTION_FILTERS["RF21"] == (("AAPL",), (2024,), None)     # Q4 -> the annual report (spec §4.2)
    assert set(cp.ANNOTATION_READINGS) == {"RF21", "RF22"}


def test_annotated_filters_cover_every_name_year_and_quarter():
    assert cp.annotated_filters("RF02") == (Filters.of(ticker="JPM"), Filters.of(ticker="BAC"))
    assert cp.annotated_filters("RF07") == (Filters.of(ticker="MRK", fiscal_year=2024, fiscal_period="Q2"),)
    assert cp.annotated_filters("RF15") == (Filters.of(ticker="AMZN", fiscal_year=2024),
                                            Filters.of(ticker="AMZN", fiscal_year=2026))
    assert cp.annotated_filters("G06") == ()


def test_gold_questions_use_the_items_own_companies_and_years():
    d01 = question("D01")
    assert d01.filters == (Filters.of(ticker="AAPL", fiscal_year=2024), Filters.of(ticker="MSFT", fiscal_year=2024))
    assert d01.gold_keys == {("AAPL", 2024, "FY", "Item 8"), ("MSFT", 2024, "FY", "Item 8")}
    assert question("G01").filters == (Filters.of(ticker="COST", fiscal_year=2025),)    # annotated
    assert question("G01").gold_keys is None and question("RF17").gold_keys is None


def test_a_follow_up_carries_the_earlier_turn_with_its_gold_answer():
    items = {i.id: i for i in GOLD}
    f04 = question("F04")
    first = items[items["F04"].previous_turn_id]
    assert f04.history == ((first.question, first.gold_answer),)
    assert f04.query == build_query(items["F04"], items) and first.question in f04.query
    assert question("A01").history == ()
    retriever = FakeRetriever()
    cp.build_record(f04, retriever, words, SCOPE)
    assert [s[0] for s in retriever.searches] == [f04.query]          # searched with the earlier turn included


@pytest.mark.parametrize("f, reason", [
    (Filters.of(ticker="TSLA", fiscal_year=2024), "TSLA is not in config/companies.yaml"),
    (Filters.of(ticker="MSFT", fiscal_year=2026), "MSFT fiscal 2026 is not indexed"),
    (Filters.of(ticker="MRK", fiscal_year=2024, fiscal_period="Q2"), "MRK fiscal 2024 Q2 is not indexed"),
    (Filters.of(ticker="MRK", fiscal_year=2025, fiscal_period="Q2"), None),
    (Filters.of(ticker="AAPL", fiscal_year=2023), None),
    (Filters.of(ticker="MSFT"), None),
])
def test_out_of_corpus_matches_what_search_filings_would_refuse(f, reason):
    assert cp.out_of_corpus(f, SCOPE) == reason


def test_the_period_decides_the_form_whose_years_count():
    """As search_filings.check_period: FY means the 10-K years, a quarter the 10-Q years, no period either."""
    scope = CorpusScope(frozenset({"AAPL"}), frozenset({2024}), frozenset({2025}), frozenset({"Q1", "Q2", "Q3"}))
    assert cp.out_of_corpus(Filters.of(ticker="AAPL", fiscal_year=2025, fiscal_period="FY"), scope) == \
        "AAPL fiscal 2025 FY is not indexed"
    assert cp.out_of_corpus(Filters.of(ticker="AAPL", fiscal_year=2024, fiscal_period="Q1"), scope) == \
        "AAPL fiscal 2024 Q1 is not indexed"
    assert cp.out_of_corpus(Filters.of(ticker="AAPL", fiscal_year=2025), scope) is None


# --- packing -------------------------------------------------------------------------------------------------

def test_passages_fill_the_budget_in_rank_order_and_count_the_rendered_block():
    hits = [hit(f"c{n}", text="word " * 10) for n in range(5)]        # 17 "tokens" per rendered passage
    passages, tokens = cp.pack_passages(hits, words, budget=51)       # exactly three passages: kept
    assert [p.chunk_id for p in passages] == ["c0", "c1", "c2"] and [p.pid for p in passages] == ["P1", "P2", "P3"]
    assert tokens == words(cp.render_passages(passages)) == 51
    assert len(cp.pack_passages(hits, words, budget=50)[0]) == 2      # one token short: the third is left out
    assert passages[0].citation == "[AAPL 10-K FY2024 · Item 8]"


def test_packing_stops_at_the_first_passage_that_does_not_fit():
    hits = [hit("small1", text="a b"), hit("huge", text="w " * 100), hit("small2", text="c d")]
    passages, _ = cp.pack_passages(hits, words, budget=30)
    assert [p.chunk_id for p in passages] == ["small1"]          # a prefix of the ranking, never reordered
    assert cp.pack_passages([], words, 30) == ([], 0)


def test_record_searches_each_filter_interleaves_and_reports_evidence():
    retriever = FakeRetriever()
    record = cp.build_record(question("D01"), retriever, words, SCOPE)
    assert [s[2] for s in retriever.searches] == list(question("D01").filters)
    assert all(s[1] == cp.HITS_PER_FILTER for s in retriever.searches)
    assert [p["chunk_id"] for p in record["passages"]] == ["AAPL_0", "MSFT_0", "AAPL_1", "MSFT_1", "AAPL_2",
                                                         "MSFT_2"]
    assert record["evidence_in_pack"] is True and record["out_of_corpus"] == []
    assert list(record) == ["id", "question", "history", "filters", "out_of_corpus", "passages", "n_tokens",
                            "evidence_in_pack"]
    assert record["n_tokens"] == words(cp.render_passages(cp.Passage(**p) for p in record["passages"]))


def test_evidence_is_false_when_no_passage_is_from_a_gold_source():
    class MdaOnly(FakeRetriever):
        def search(self, query, k=5, filters=NO_FILTERS):
            return [hit("mda", ticker=filters.ticker, item="Item 7")]

    assert cp.build_record(question("D01"), MdaOnly(), words, SCOPE)["evidence_in_pack"] is False


def test_out_of_corpus_filters_are_not_searched_and_are_recorded():
    retriever = FakeRetriever()
    record = cp.build_record(question("RF15"), retriever, words, SCOPE)     # AMZN 2024 and 2026
    assert [s[2] for s in retriever.searches] == [Filters.of(ticker="AMZN", fiscal_year=2024)]
    assert record["out_of_corpus"] == ["AMZN fiscal 2026 is not indexed"] and record["passages"]
    assert record["evidence_in_pack"] is None                              # refusal tests have no gold sources
    none = cp.build_record(question("RF04"), retriever, words, SCOPE)       # Tesla
    assert none["passages"] == [] and none["n_tokens"] == 0 and len(retriever.searches) == 1


def test_a_question_naming_no_company_gets_no_search_and_no_passages():
    retriever = FakeRetriever()
    record = cp.build_record(question("G06"), retriever, words, SCOPE)
    assert retriever.searches == [] and record["passages"] == [] and record["filters"] == []


# --- files ---------------------------------------------------------------------------------------------------

def test_pack_files_are_deterministic_and_written_atomically(tmp_path):
    records = [cp.build_record(question(q), FakeRetriever(), words, SCOPE) for q in ("A01", "D01", "G06")]
    chunks = tmp_path / "chunks_C3.jsonl"
    chunks.write_text("{}\n", encoding="utf-8")
    bm25 = tmp_path / "bm25_C3.pkl"
    bm25.write_bytes(b"index")
    info = cp.meta(records, cp.GOLD_PATH, chunks, bm25, "sentence-transformers:test", "tok@rev")
    out = tmp_path / "out" / cp.GOLD_PACK
    cp.write_pack(records, out, info)
    first = (out.read_bytes(), out.with_name("context_pack_gold.meta.json").read_bytes())
    cp.write_pack(records, out, cp.meta(records, cp.GOLD_PATH, chunks, bm25, "sentence-transformers:test", "tok@rev"))
    assert (out.read_bytes(), out.with_name("context_pack_gold.meta.json").read_bytes()) == first
    assert sorted(p.name for p in out.parent.iterdir()) == ["context_pack_gold.jsonl", "context_pack_gold.meta.json"]
    saved = json.loads(first[1])
    assert saved["inputs"] == {"gold_set.jsonl": cp.sha256(cp.GOLD_PATH), "chunks_C3.jsonl": cp.sha256(chunks),
                               "bm25_C3.pkl": cp.sha256(bm25)}
    assert (saved["questions"], saved["with_passages"], saved["with_gold_sources"]) == (3, 2, 2)
    assert set(saved) == {"pack_version", "questions", "retrieval", "token_budget", "tokenizer", "history",
                          "with_passages", "evidence_in_pack", "with_gold_sources", "max_n_tokens", "inputs"}


# --- CLI -----------------------------------------------------------------------------------------------------

def test_an_invalid_gold_set_exits_1(tmp_path):
    bad = tmp_path / "gold.jsonl"
    bad.write_text('{"id": "Z99"}\n', encoding="utf-8")
    assert cp.main(["--gold", str(bad), "--out-dir", str(tmp_path)]) == 1


def test_a_failed_write_keeps_the_old_file_and_leaves_no_temporary_file(monkeypatch, tmp_path):
    target = tmp_path / "pack.jsonl"
    target.write_text("old\n", encoding="utf-8")

    def disk_full(self, other):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", disk_full)
    with pytest.raises(OSError):
        cp.write_atomic("new\n", target)
    assert target.read_text(encoding="utf-8") == "old\n" and [p.name for p in tmp_path.iterdir()] == ["pack.jsonl"]


def test_a_question_without_an_annotation_exits_1(monkeypatch, tmp_path, capsys):
    monkeypatch.delitem(cp.QUESTION_FILTERS, "G01")
    assert cp.main(["--out-dir", str(tmp_path)]) == 1
    assert "no QUESTION_FILTERS entry for G01" in capsys.readouterr().err


def test_a_missing_tokenizer_or_index_exits_2(monkeypatch, tmp_path, capsys):
    def offline(text):
        raise FinSightError("Qwen tokenizer unavailable", error_type="TokenizerUnavailable")

    monkeypatch.setattr(core.tokens, "count_tokens", offline)
    assert cp.main(["--out-dir", str(tmp_path)]) == 2
    assert "tokenizer unavailable" in capsys.readouterr().err

    import retrieval.dense
    closed = []
    monkeypatch.setattr(core.tokens, "count_tokens", words)
    monkeypatch.setattr(retrieval.dense, "embedder_from_settings",
                        lambda: retrieval.dense.Embedder(lambda texts: [], "fake:model", lambda: closed.append(1)))
    assert cp.main(["--out-dir", str(tmp_path), "--index-dir", str(tmp_path / "no-index")]) == 2
    assert closed == [1] and not (tmp_path / cp.GOLD_PACK).exists()      # embedder closed, nothing written


@pytest.mark.parametrize("failure", [ValueError("collection chunks_C3 was built with ollama:nomic-embed-text"),
                                     RuntimeError("vector count 10 != chunk count 12"),
                                     ImportError("No module named 'chromadb'")])
def test_a_mismatched_or_unloadable_index_exits_2_without_writing(monkeypatch, tmp_path, capsys, failure):
    import retrieval.dense

    def broken(*args, **kwargs):
        raise failure

    monkeypatch.setattr(core.tokens, "count_tokens", words)
    monkeypatch.setattr(retrieval.dense, "embedder_from_settings",
                        lambda: retrieval.dense.Embedder(lambda texts: [], "fake:model"))
    monkeypatch.setattr(cp, "load_retrievers", broken)
    assert cp.main(["--out-dir", str(tmp_path)]) == 2
    assert type(failure).__name__ in capsys.readouterr().err and list(tmp_path.iterdir()) == []


def test_a_failure_while_building_the_second_pack_writes_neither(monkeypatch, tmp_path):
    import retrieval.dense

    class FailsOnRefusals(FakeRetriever):
        def search(self, query, k=5, filters=NO_FILTERS):
            if query.startswith("Should I buy"):                  # RF01, the first refusal test searched
                raise RuntimeError("index went away")
            return super().search(query, k, filters)

    monkeypatch.setattr(core.tokens, "count_tokens", words)
    monkeypatch.setattr(retrieval.dense, "embedder_from_settings",
                        lambda: retrieval.dense.Embedder(lambda texts: [], "fake:model"))
    monkeypatch.setattr(cp, "load_retrievers", lambda *args: {cp.RETRIEVER: FailsOnRefusals()})
    assert cp.main(["--out-dir", str(tmp_path)]) == 2
    assert list(tmp_path.iterdir()) == []


# --- core/tokens.py ------------------------------------------------------------------------------------------

def test_count_tokens_counts_without_special_tokens(monkeypatch):
    calls = []

    class FakeTokenizer:
        def encode(self, text, add_special_tokens):
            calls.append(add_special_tokens)
            return type("Encoding", (), {"ids": text.split()})()

    monkeypatch.setattr(core.tokens, "qwen_tokenizer", lambda: FakeTokenizer())
    assert core.tokens.count_tokens("Net sales 391,035") == 3 and calls == [False]


def test_an_unavailable_tokenizer_is_a_typed_error_and_not_cached(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    def no_network(name, revision):
        raise Exception("couldn't reach huggingface.co")

    fake = ModuleType("tokenizers")               # CI installs no tokenizers package; never touch the network
    fake.Tokenizer = SimpleNamespace(from_pretrained=no_network)
    monkeypatch.setitem(sys.modules, "tokenizers", fake)
    core.tokens.qwen_tokenizer.cache_clear()
    try:
        with pytest.raises(FinSightError) as err:
            core.tokens.qwen_tokenizer()
        assert err.value.error_type == "TokenizerUnavailable" and core.tokens.QWEN_TOKENIZER_REVISION[:12] in err.value.message
        assert core.tokens.qwen_tokenizer.cache_info().currsize == 0
    finally:
        core.tokens.qwen_tokenizer.cache_clear()


# --- the committed packs ---------------------------------------------------------------------------------------

def sha256_unverified(path: Path) -> str:
    """sha256 of a JSONL file with every verified_by set back to null, written as the test sets are."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    text = "".join(json.dumps({**r, "verified_by": None} if "verified_by" in r else r, ensure_ascii=False) + "\n"
                   for r in records)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("name, source, ids", [
    (cp.GOLD_PACK, cp.GOLD_PATH, [i.id for i in GOLD]),
    (cp.REFUSAL_PACK, cp.REFUSAL_PATH, [t.id for t in REFUSALS]),
])
def test_committed_packs_match_their_inputs_and_the_budget(name, source, ids):
    """The packs are committed, so everything that does not need the index is checked against the current code
    and inputs: a change to the gold set, the refusal tests, QUESTION_FILTERS, out_of_corpus, the history rule,
    the citation format or the meta counts fails here until the pack is rebuilt (python eval/context_pack.py)."""
    path = cp.RESULTS_DIR / name
    assert path.exists(), f"{name} missing: run python eval/context_pack.py"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    info = json.loads(cp.meta_path(path).read_text(encoding="utf-8"))
    assert [r["id"] for r in records] == ids
    # a second person's verified_by is no input of the pack (it reads ids and questions): the pack was built from
    # the same tests before they were verified (W1-10), so the file with verified_by unset must give the recorded hash
    assert info["inputs"][source.name] in (cp.sha256(source), sha256_unverified(source))
    questions = {q.id: q for q in cp.gold_questions(GOLD) + cp.refusal_questions(REFUSALS)}
    for r in records:
        q = questions[r["id"]]
        assert r["question"] == q.question
        assert r["history"] == [{"question": hq, "answer": ha} for hq, ha in q.history]
        assert r["filters"] == [f.active() for f in q.filters]
        assert r["out_of_corpus"] == [x for f in q.filters if (x := cp.out_of_corpus(f, SCOPE))]
        searchable = [f for f in q.filters if cp.out_of_corpus(f, SCOPE) is None]
        assert [p["pid"] for p in r["passages"]] == [f"P{n}" for n in range(1, len(r["passages"]) + 1)]
        for p in r["passages"]:
            hit_ = hit(p["chunk_id"], p["ticker"], p["fiscal_year"], p["fiscal_period"], p["item"])
            assert p["citation"] == cp.citation(hit_) and p["form"] == hit_.form
            assert any(f.matches(hit_) for f in searchable)        # Chunk-like: ticker/year/period/item
        assert (r["n_tokens"] == 0) == (r["passages"] == []) and r["n_tokens"] <= cp.TOKEN_BUDGET
        keys = q.gold_keys
        assert r["evidence_in_pack"] == (None if keys is None else any(
            (p["ticker"], p["fiscal_year"], p["fiscal_period"], p["item"]) in keys for p in r["passages"]))
    assert info["token_budget"] == cp.TOKEN_BUDGET and info["questions"] == len(records)
    assert info["with_passages"] == sum(bool(r["passages"]) for r in records)
    assert info["evidence_in_pack"] == sum(r["evidence_in_pack"] is True for r in records)
    assert info["with_gold_sources"] == sum(r["evidence_in_pack"] is not None for r in records)
    assert info["max_n_tokens"] == max(r["n_tokens"] for r in records)
