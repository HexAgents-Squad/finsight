"""W1-12: the context pack, i.e. the same retrieved passages for every model in the model comparison.

Usage:
    python eval/context_pack.py                                   # writes both packs to eval/results/
    python eval/context_pack.py --out-dir /tmp/packs --index-dir /tmp/idx --chunks-dir /tmp/chunks

Every model in the comparison (docs/week1_plan.txt §3) answers from exactly the passages in these files, so a
difference between models is a difference in the models, not in retrieval, and a run needs no index on the
machine that makes it. Output (docs/week1_plan.txt §2.3):
    eval/results/context_pack_gold.jsonl       80 gold questions (gold-v1)
    eval/results/context_pack_refusal.jsonl    22 refusal tests
    <pack>.meta.json                           settings, tokenizer and input fingerprints (sha256)
One line per question:
    {"id", "question", "history": [{"question", "answer"}], "filters": [{"ticker", "fiscal_year", ...}],
     "out_of_corpus": [...], "passages": [{"pid", "chunk_id", "citation", "ticker", "form", "fiscal_year",
     "fiscal_period", "item", "text"}], "n_tokens", "evidence_in_pack"}

Retrieval: C3 chunks + hybrid search (ADR-013), one search per filter, rankings interleaved (as
eval/run_retrieval.py does for its company+year setting), then passages in rank order while the rendered block
(render_passages, the exact text a prompt shows) stays within TOKEN_BUDGET Qwen3 tokens (counted on the block
alone; inside a full prompt the count can differ by a token or two at the block's edges).
Filters are ORACLE filters, i.e. the right answer's company and fiscal year, so the comparison measures the
models rather than retrieval (the report must say so):
    - gold items: the item's `companies` x `fiscal_years`;
    - questions that list none (G01-G07 and the refusal tests): QUESTION_FILTERS, annotated by hand from the
      question text; tests check the annotation names exactly the companies, years and quarters the question
      does, except the readings in ANNOTATION_READINGS ("latest" = FY2025, "Q4" = the annual report).
A filter outside config/companies.yaml (a company, or a year/quarter not indexed) gets no search, as
search_filings would answer out_of_corpus; it is listed in `out_of_corpus`. A question naming no company gets no
passages. `history` holds the earlier turns of a follow-up, oldest first, with their GOLD answers (the same
history for every model). `evidence_in_pack` says whether a passage comes from one of the item's gold sources (null without
sources), so a wrong answer can be told apart: evidence missing (retrieval) vs evidence ignored (model).

Deterministic: same index, chunks and inputs -> byte-identical files (no timestamps; the meta file records the
sha256 of the questions file, chunks_C3.jsonl and the C3 BM25 index instead of a commit; the dense collection is
checked against the chunk ids when it loads). Both packs are built before anything is written, and every file is
replaced atomically. Exit codes: 0 = written, 1 = gold set, refusal tests or annotations invalid, 2 = index,
chunks, settings, search packages or tokenizer missing or mismatched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError  # noqa: E402

from core.exceptions import FinSightError  # noqa: E402
from eval.retrieval_eval import build_query, hit_key  # noqa: E402
from eval.run_retrieval import RESULTS_DIR, interleave, load_retrievers  # noqa: E402
from eval.safety_tests import RefusalTest  # noqa: E402
from eval.schema import CorpusScope, GoldItem, corpus_scope  # noqa: E402
from eval.validate_gold import DEFAULT_PATH as GOLD_PATH  # noqa: E402
from eval.validate_gold import safe_console, validate_file  # noqa: E402
from eval.validate_test_sets import SETS, validate_set  # noqa: E402
from retrieval import CHUNKS_DIR, INDEX_DIR  # noqa: E402
from retrieval.bm25 import index_path  # noqa: E402
from retrieval.common import Filters, SearchResult, chunks_path  # noqa: E402
from retrieval.hybrid import CANDIDATES, RRF_K, Retriever  # noqa: E402
from tools.search_filings import citation  # noqa: E402

PACK_VERSION = "cp-v1"
VARIANT, RETRIEVER = "C3", "hybrid"                 # the search_filings default (ADR-013)
HITS_PER_FILTER = 10                                # candidates per search before the token budget cuts
# 8,192 tokens per call (docs/week1_plan.txt §2.2) - 1,024 reply - 4,000 passages leaves ~3,100 for the system
# prompt, chat template, question and history (today's longest question + history: under 400 characters)
TOKEN_BUDGET = 4_000
PASSAGE_SEPARATOR = "\n\n"
GOLD_PACK, REFUSAL_PACK = "context_pack_gold.jsonl", "context_pack_refusal.jsonl"
META_SUFFIX = ".meta.json"
QUARTERS = frozenset({"Q1", "Q2", "Q3"})
ANNUAL = "FY"
REFUSAL_PATH = SETS["refusal"][3]

# (tickers, fiscal years, fiscal period) named by questions whose item lists no companies, annotated from the
# question text. Out-of-corpus names are kept (NFLX, TSLA, years 2026/2027) so the pack records why there are no
# passages. A quarter outside FISCAL_PERIODS is not annotated: Q4 is never derived (spec §4.2), so "fiscal 2024
# Q4" searches the fiscal 2024 annual report, as the agent would.
QUESTION_FILTERS: dict[str, tuple[tuple[str, ...], tuple[int, ...], str | None]] = {
    "G01": (("COST",), (2025,), None),
    "G02": (("MSFT",), (), None),
    "G03": (("NFLX",), (2024,), None),
    "G04": (("KO",), (), None),
    "G05": (("PFE",), (), None),
    "G06": ((), (), None),
    "G07": ((), (), None),
    "RF01": (("NVDA",), (), None),
    "RF02": (("JPM", "BAC"), (), None),
    "RF03": (("AAPL",), (2027,), None),
    "RF04": (("TSLA",), (2024,), None),
    "RF05": (("MSFT",), (2026,), None),
    "RF06": (("JPM",), (), None),
    "RF07": (("MRK",), (2024,), "Q2"),
    "RF08": (("GS",), (), None),
    "RF09": (("WMT",), (), None),
    "RF10": ((), (), None),
    "RF11": ((), (), None),
    "RF12": ((), (), None),
    "RF13": (("META",), (), None),
    "RF14": (("AAPL",), (2025,), None),
    "RF15": (("AMZN",), (2024, 2026), None),
    "RF16": (("NVDA",), (2025,), None),
    "RF17": (("WMT",), (2025,), None),
    "RF18": (("KO",), (2024,), None),
    "RF19": (("MSFT",), (), None),
    "RF20": (("UNH",), (2024,), None),
    "RF21": (("AAPL",), (2024,), None),
    "RF22": (("MSFT",), (2025,), None),
}
# Annotations that are a reading of the question, not its literal words; the annotation test allows exactly these
ANNOTATION_READINGS: dict[str, str] = {
    "RF21": "'fiscal 2024 Q4': Q4 is not derived (spec §4.2), so the fiscal 2024 annual report is searched",
    "RF22": "'latest fiscal year' = FY2025, the latest indexed 10-K (spec §4.3; the test's rationale)",
}

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    query: str                                          # what is searched (a follow-up carries the earlier turns)
    history: tuple[tuple[str, str], ...]                # (question, gold answer) of the earlier turns, oldest first
    filters: tuple[Filters, ...]
    gold_keys: frozenset[tuple[str, int, str, str]] | None    # None: no gold sources (refusal tests, G refusals)


@dataclass(frozen=True)
class Passage:
    pid: str                                            # "P1", "P2", ... in rank order: what citations refer to
    chunk_id: str
    citation: str
    ticker: str
    form: str
    fiscal_year: int
    fiscal_period: str
    item: str
    text: str


def render_passage(pid: str, cite: str, text: str) -> str:
    """How one passage appears in a prompt. The B0 runner (W1-13) renders with this function, so the token
    count recorded here is the count the model sees."""
    return f"[{pid}] {cite}\n{text}"


def render_passages(passages: Iterable[Passage]) -> str:
    return PASSAGE_SEPARATOR.join(render_passage(p.pid, p.citation, p.text) for p in passages)


def annotated_filters(qid: str) -> tuple[Filters, ...]:
    """Filters for a question that lists no companies: every annotated (ticker, year), quarter included."""
    tickers, years, period = QUESTION_FILTERS[qid]
    return tuple(Filters.of(ticker=t, fiscal_year=y, fiscal_period=period) for t in tickers for y in years or (None,))


def gold_questions(items: list[GoldItem]) -> list[Question]:
    by_id = {i.id: i for i in items}
    out = []
    for item in items:
        history, cur, seen = [], item, {item.id}
        while cur.previous_turn_id and cur.previous_turn_id in by_id and cur.previous_turn_id not in seen:
            cur = by_id[cur.previous_turn_id]
            seen.add(cur.id)
            history.insert(0, (cur.question, cur.gold_answer))
        if item.companies:
            filters = tuple(Filters.of(ticker=t, fiscal_year=y) for t in item.companies
                            for y in item.fiscal_years or (None,))
        else:
            filters = annotated_filters(item.id)
        keys = frozenset(s.key() for s in item.gold_sources) or None
        out.append(Question(item.id, item.question, build_query(item, by_id), tuple(history), filters, keys))
    return out


def refusal_questions(tests: list[RefusalTest]) -> list[Question]:
    return [Question(t.id, t.question, t.question, (), annotated_filters(t.id), None) for t in tests]


def out_of_corpus(f: Filters, scope: CorpusScope) -> str | None:
    """Why search_filings would refuse this filter (R3), or None if it is searchable."""
    if f.ticker not in scope.tickers:
        return f"{f.ticker} is not in config/companies.yaml"
    if f.fiscal_year is None:
        return None
    # the same years search_filings.check_period allows: FY = 10-K, a quarter = 10-Q, no period = either form
    if f.fiscal_period == ANNUAL:
        years = scope.tenk_years
    elif f.fiscal_period in QUARTERS:
        years = scope.tenq_years
    else:
        years = scope.tenk_years | scope.tenq_years
    if f.fiscal_year not in years:
        period = f" {f.fiscal_period}" if f.fiscal_period else ""
        return f"{f.ticker} fiscal {f.fiscal_year}{period} is not indexed"
    return None


def pack_passages(hits: list[SearchResult], count: TokenCounter, budget: int) -> tuple[list[Passage], int]:
    """Passages in rank order while the rendered block fits `budget`; stops at the first that does not, so the
    pack is always a prefix of the ranking. Counting the whole block (not summing parts) keeps the count exact
    across the separators."""
    kept: list[Passage] = []
    tokens = 0
    for hit in hits:
        candidate = [*kept, Passage(f"P{len(kept) + 1}", hit.chunk_id, citation(hit), hit.ticker, hit.form,
                                    hit.fiscal_year, hit.fiscal_period, hit.item, hit.text)]
        n = count(render_passages(candidate))
        if n > budget:
            break
        kept, tokens = candidate, n
    return kept, tokens


def build_record(q: Question, retriever: Retriever, count: TokenCounter, scope: CorpusScope,
                 budget: int = TOKEN_BUDGET) -> dict:
    searchable = [f for f in q.filters if out_of_corpus(f, scope) is None]
    rankings = [retriever.search(q.query, HITS_PER_FILTER, f) for f in searchable]
    hits = interleave(rankings, HITS_PER_FILTER * len(rankings)) if rankings else []
    passages, tokens = pack_passages(hits, count, budget)
    evidence = None
    if q.gold_keys is not None:
        evidence = any(hit_key(asdict(p)) in q.gold_keys for p in passages)
    return {"id": q.id, "question": q.question,
            "history": [{"question": hq, "answer": ha} for hq, ha in q.history],
            "filters": [f.active() for f in q.filters],
            "out_of_corpus": [reason for f in q.filters if (reason := out_of_corpus(f, scope))],
            "passages": [asdict(p) for p in passages], "n_tokens": tokens, "evidence_in_pack": evidence}


def write_atomic(text: str, path: Path) -> None:
    """Write via a temporary file and a rename: a reader never sees half a file, and a failed write leaves the
    old file in place and no temporary file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def meta(records: list[dict], questions_file: Path, chunks_file: Path, bm25_file: Path, embed_model: str,
         tokenizer: str) -> dict:
    return {"pack_version": PACK_VERSION, "questions": len(records),
            "retrieval": {"variant": VARIANT, "retriever": RETRIEVER, "rrf_k": RRF_K, "candidates": CANDIDATES,
                          "hits_per_filter": HITS_PER_FILTER, "embedding_model": embed_model,
                          "filters": "oracle: gold companies x fiscal years; QUESTION_FILTERS where none"},
            "token_budget": TOKEN_BUDGET, "tokenizer": tokenizer, "history": "earlier turns with gold answers",
            "with_passages": sum(bool(r["passages"]) for r in records),
            "evidence_in_pack": sum(r["evidence_in_pack"] is True for r in records),
            "with_gold_sources": sum(r["evidence_in_pack"] is not None for r in records),
            "max_n_tokens": max((r["n_tokens"] for r in records), default=0),
            "inputs": {questions_file.name: sha256(questions_file), chunks_file.name: sha256(chunks_file),
                       bm25_file.name: sha256(bm25_file)}}


def meta_path(path: Path) -> Path:
    return path.with_name(path.stem + META_SUFFIX)


def write_pack(records: list[dict], path: Path, info: dict) -> None:
    write_atomic(jsonl(records), path)
    write_atomic(json.dumps(info, indent=2, ensure_ascii=False) + "\n", meta_path(path))


def main(argv: list[str] | None = None) -> int:
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=GOLD_PATH)
    ap.add_argument("--refusal", type=Path, default=REFUSAL_PATH)
    ap.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    ap.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)

    rep = validate_file(args.gold)
    if not rep.ok:
        print(f"ERROR: gold set {args.gold} is invalid; run eval/validate_gold.py first", file=sys.stderr)
        return 1
    tests, errors = validate_set("refusal", args.refusal)       # the rules validate_test_sets.py (CI) applies
    if errors:
        print(f"ERROR: refusal tests {args.refusal} are invalid; run eval/validate_test_sets.py", file=sys.stderr)
        return 1
    missing = sorted(({i.id for i in rep.items if not i.companies} | {t.id for t in tests}) - set(QUESTION_FILTERS))
    if missing:
        print(f"ERROR: no QUESTION_FILTERS entry for {', '.join(missing)}", file=sys.stderr)
        return 1

    from core.tokens import QWEN_TOKENIZER, QWEN_TOKENIZER_REVISION, count_tokens
    from retrieval.dense import embedder_from_settings
    embedder = None
    try:
        count_tokens("warm-up")                         # fail fast if the tokenizer cannot be loaded
        embedder = embedder_from_settings()
        retriever = load_retrievers(VARIANT, [RETRIEVER], embedder, args.index_dir, args.chunks_dir)[RETRIEVER]
        scope = corpus_scope()
        tokenizer = f"{QWEN_TOKENIZER}@{QWEN_TOKENIZER_REVISION}"
        chunks_file, bm25_file = chunks_path(VARIANT, args.chunks_dir), index_path(VARIANT, args.index_dir)
        packs = []                                      # both built first: a failure writes neither
        for questions, source, name in ((gold_questions(rep.items), args.gold, GOLD_PACK),
                                        (refusal_questions(tests), args.refusal, REFUSAL_PACK)):
            records = [build_record(q, retriever, count_tokens, scope) for q in questions]
            packs.append((records, args.out_dir / name,
                          meta(records, source, chunks_file, bm25_file, embedder.model_id, tokenizer)))
        for records, path, info in packs:
            write_pack(records, path, info)
            print(f"{path.name}: {info['questions']} questions, {info['with_passages']} with passages, evidence in "
                  f"pack for {info['evidence_in_pack']}/{info['with_gold_sources']}, max {info['max_n_tokens']} tokens")
    except ValidationError as err:                      # before ValueError, which it subclasses
        print(f"ERROR: settings are incomplete; copy .env.example to .env and fill it in.\n{err}", file=sys.stderr)
        return 2
    # ValueError: index built with another embedding model or an old format; RuntimeError: vector count differs
    # from the chunk count; ImportError: chromadb or the tokenizers package missing (as eval/run_retrieval.py)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, FinSightError) as err:
        print(f"ERROR: {type(err).__name__}: {err}", file=sys.stderr)
        return 2
    finally:
        if embedder is not None:
            embedder.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
