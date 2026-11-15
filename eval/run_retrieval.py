"""WP5 step 5, W1-05: recall@k and MRR of every retriever and filter setting on the frozen gold set (gold-v1).

Usage:
    python eval/run_retrieval.py --rerank                 # everything, and the summary (needs every index)
    python eval/run_retrieval.py --variants C3 --retrievers bm25 hybrid
    python eval/run_retrieval.py --index-dir /tmp/idx --chunks-dir /tmp/chunks --out-dir /tmp/results

For each chunk variant (C1/C2/C3), retriever (bm25, dense, hybrid) and filter setting it runs every answerable
gold question and writes the ranked chunks to eval/results/runs/<variant>_<retriever>_<setting>.jsonl in the
format eval/retrieval_eval.py scores. Queries come from retrieval_eval.build_query (a follow-up carries the
previous question). Filter settings (one filtered search per filter; the result lists are interleaved, as the
agent issues one search per entity):
    none                  the question only;
    question              the companies and fiscal years the question names (retrieval/entities.py; a follow-up
                          keeps what it does not name from the turn before): what the agent can do;
    company+year          the gold item's `companies` and `fiscal_years` (correct by construction: an upper bound);
and on C3 (the default variant, ADR-013) with hybrid search, the trials:
    ...+period            also the fiscal period the question names (FY when it names no quarter), which also
                          picks the form (FY = 10-K, a quarter = 10-Q);
    ...+period+statement  for a question about a reported figure, the statement Item's chunks (Item 8, or Part I
                          Item 1 of a 10-Q) interleaved first (the Item 8 preference trial);
--rerank adds hybrid+rerank on C3: the hybrid top 30 reordered by a cross-encoder (retrieval/rerank.py), for the
question and company+year settings and every trial. A full run (every variant and retriever, and --rerank)
writes the tables to <out-dir>/retrieval_summary.md between the GENERATED markers, keeping the text outside them
(the interpretation, which cites every row). A partial run writes its run files and prints its scores but leaves
the summary alone, so it never replaces the full table. Query embeddings are cached for the run (the same
question is searched once per variant, setting and filter).
Exit codes: 0 = done, 1 = gold set invalid, 2 = an index, a chunk file, the reranker or the settings are
missing, or the summary lost its GENERATED markers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError  # noqa: E402

from core.figures import GROUPED_FIGURE, contains_figure  # noqa: E402
from eval.retrieval_eval import KS, build_query, hit_key, paired_recall, score_runs, sign_test  # noqa: E402
from eval.schema import GoldItem  # noqa: E402
from eval.validate_gold import DEFAULT_PATH, safe_console, validate_file  # noqa: E402
from retrieval import CHROMA_DIR, CHUNKS_DIR, INDEX_DIR  # noqa: E402
from retrieval.bm25 import BM25Index, index_path  # noqa: E402
from retrieval.chunking import VARIANTS  # noqa: E402
from retrieval.common import NO_FILTERS, Filters, chunks_path, read_chunks  # noqa: E402
from retrieval.dense import Embedder, cached, embedder_from_settings  # noqa: E402
from retrieval.entities import Entities, from_turns  # noqa: E402
from retrieval.hybrid import HybridIndex, Retriever, interleave  # noqa: E402
from retrieval.statement import StatementFirstIndex  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_NAME = "retrieval_summary.md"
SUMMARY_FILE = RESULTS_DIR / SUMMARY_NAME
RETRIEVERS = ("bm25", "dense", "hybrid")
RERANKED = "hybrid+rerank"
TRIAL_VARIANT = "C3"
TOP_K = max(KS)
FIGURE_K = 5                     # the answer figure must be in one of the first 5 chunks
START, END = "<!-- GENERATED:START (eval/run_retrieval.py) -->", "<!-- GENERATED:END -->"
CATEGORY_ORDER = ("fact", "ratio", "period_comparison", "multi_company", "qualitative", "multi_turn", "refuse")
SOURCE_NAMES = {"none": "none", "question": "question", "gold": "company+year"}


@dataclass(frozen=True)
class Setting:
    """Where the filters come from ("none", "question" or "gold"), and the two trials on top."""
    source: str
    period: bool = False             # also the fiscal period the question names
    statement: bool = False          # figure questions: the statement Item's chunks interleaved first

    def __post_init__(self) -> None:
        if self.source not in SOURCE_NAMES or (self.source == "none" and (self.period or self.statement)) \
                or (self.statement and not self.period):
            raise ValueError(f"invalid setting {self!r}")

    @property
    def name(self) -> str:
        return SOURCE_NAMES[self.source] + "+period" * self.period + "+statement" * self.statement


GRID = (Setting("none"), Setting("question"), Setting("gold"))
TRIALS = (Setting("question", True), Setting("question", True, True), Setting("gold", True),
          Setting("gold", True, True))
SETTINGS = {s.name: s for s in GRID + TRIALS}
# The paired comparisons the summary reports (recall@5 per question, sign test), all on C3, fixed before the first
# run so the table cannot be chosen after seeing the results: (retriever, setting) a -> b
PAIRS = (
    (("hybrid", "question"), ("hybrid", "company+year")),                         # reading filters vs the gold ones
    (("hybrid", "question"), ("hybrid", "question+period")),                      # the period filter
    (("hybrid", "question+period"), ("hybrid", "question+period+statement")),     # statement Item first
    (("hybrid", "question"), (RERANKED, "question")),                             # reranking
    (("hybrid", "question+period"), (RERANKED, "question+period")),
    (("hybrid", "question+period+statement"), (RERANKED, "question+period+statement")),
    (("hybrid", "company+year"), ("hybrid", "company+year+period+statement")),    # both trials, gold filters
)


def settings_for(variant: str, retriever: str) -> tuple[Setting, ...]:
    """The grid for every variant and retriever; the trials on C3 hybrid and hybrid+rerank."""
    if retriever == RERANKED:
        return GRID[1:] + TRIALS
    return GRID + (TRIALS if variant == TRIAL_VARIANT and retriever == "hybrid" else ())


def turns(item: GoldItem, by_id: dict[str, GoldItem]) -> list[str]:
    """The questions of the conversation up to this item, oldest first (as build_query joins them)."""
    out, cur, seen = [item.question], item, {item.id}
    while cur.previous_turn_id and cur.previous_turn_id in by_id and cur.previous_turn_id not in seen:
        cur = by_id[cur.previous_turn_id]
        seen.add(cur.id)
        out.insert(0, cur.question)
    return out


def setting_filters(item: GoldItem, by_id: dict[str, GoldItem], setting: Setting) -> list[Filters]:
    """The filters of one search per entity; empty = one unfiltered search (the question names nothing)."""
    if setting.source == "none":
        return []
    read = from_turns(turns(item, by_id))
    if setting.source == "gold":
        read = Entities(tuple(item.companies), tuple(item.fiscal_years), read.periods)
    return read.filters(period=setting.period)


def run(retriever: Retriever, items: list[GoldItem], setting: Setting | str) -> tuple[list[dict], float]:
    """Ranked results for every answerable item, and the mean seconds per question."""
    setting = SETTINGS[setting] if isinstance(setting, str) else setting
    by_id = {i.id: i for i in items}
    records, start = [], time.perf_counter()
    answerable = [i for i in items if i.gold_sources]
    for item in answerable:
        query = build_query(item, by_id)
        filters = setting_filters(item, by_id, setting) or [NO_FILTERS]
        searcher = StatementFirstIndex(retriever) if setting.statement else retriever
        hits = interleave([searcher.search(query, TOP_K, f) for f in filters], TOP_K)
        records.append({"id": item.id, "retrieved": [
            {"chunk_id": h.chunk_id, "ticker": h.ticker, "fiscal_year": h.fiscal_year,
             "fiscal_period": h.fiscal_period, "item": h.item, "score": round(h.score, 6)} for h in hits]})
    return records, (time.perf_counter() - start) / max(len(answerable), 1)


def figure_hits(items: list[GoldItem], records: list[dict], texts: dict[str, str], k: int = FIGURE_K) -> set[str]:
    """The items whose gold answer states a figure as the gold set writes it ("391,035") that one of the first k
    retrieved chunks contains verbatim: the chunk that answers, not only the right Item (W1-05). Items without such
    a figure are not counted (figure_items)."""
    wanted = {i.id: GROUPED_FIGURE.findall(i.gold_answer or "") for i in items if i.gold_sources}
    out = set()
    for r in records:
        top = "\n".join(texts.get(h["chunk_id"], "") for h in r["retrieved"][:k])
        if any(contains_figure(top, f) for f in wanted.get(r["id"], [])):
            out.add(r["id"])
    return out


def figure_items(items: list[GoldItem]) -> set[str]:
    return {i.id for i in items if i.gold_sources and GROUPED_FIGURE.search(i.gold_answer or "")}


def entity_accuracy(items: list[GoldItem]) -> dict[str, int]:
    """How often the question-derived reading equals the item's companies, fiscal years and gold-source periods
    (FY when the question names no period), over the items with gold sources."""
    by_id = {i.id: i for i in items}
    answerable = [i for i in items if i.gold_sources]
    out = {"n": len(answerable), "companies": 0, "years": 0, "periods": 0, "all": 0}
    for item in answerable:
        read = from_turns(turns(item, by_id))
        ok = (set(read.tickers) == set(item.companies), set(read.years) == set(item.fiscal_years),
              set(read.periods or ("FY",)) == {s.fiscal_period for s in item.gold_sources})
        for key, good in zip(("companies", "years", "periods"), ok, strict=True):
            out[key] += good
        out["all"] += all(ok)
    return out


def load_retrievers(variant: str, wanted: list[str], embedder: Embedder | None,
                    index_dir: Path = INDEX_DIR, chunks_dir: Path = CHUNKS_DIR) -> dict[str, Retriever]:
    """The wanted retrievers for one variant. A missing index or chunk file is a FileNotFoundError."""
    out: dict[str, Retriever] = {}
    # BM25 is needed for bm25 and hybrid runs only, so a dense-only run works without the BM25 index
    bm25 = BM25Index.load(index_path(variant, index_dir)) if {"bm25", "hybrid"} & set(wanted) else None
    if "bm25" in wanted:
        out["bm25"] = bm25
    if embedder is not None:
        from retrieval.dense import DenseIndex, collection_name, collection_names, open_client
        client, name = open_client(index_dir / CHROMA_DIR.name), collection_name(variant)
        # Why check first: Chroma's "not found" error type differs between versions (ValueError in 0.x,
        # chromadb.errors.NotFoundError in 1.x), so test for the name instead of catching it.
        if name not in collection_names(client):
            raise FileNotFoundError(f"dense collection {name} not found in {index_dir / CHROMA_DIR.name}; "
                                    f"run ingest/04_index.py")
        dense = DenseIndex(client.get_collection(name), read_chunks(chunks_path(variant, chunks_dir)), embedder)
        if "dense" in wanted:
            out["dense"] = dense
        if "hybrid" in wanted:
            out["hybrid"] = HybridIndex(bm25, dense)
    return out


def write_runs(path: Path, records: list[dict], retriever: str, setting: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps({"id": r["id"], "retriever": retriever, "filters": setting,
                                "retrieved": r["retrieved"]}) + "\n")


def best_row(rows: list[dict]) -> dict:
    """Highest recall@5, then highest MRR; on a full tie the first row in run order (max keeps the first)."""
    return max(rows, key=lambda r: (r["result"]["overall"]["recall@5"], r["result"]["overall"]["mrr"]))


def run_file(out_dir: Path, variant: str, retriever: str, setting: str) -> Path:
    """'+' is written '_' ('company+year' -> 'company_year'): a '+' in a file name needs quoting in some shells."""
    return out_dir / "runs" / f"{variant}_{retriever}_{setting}.jsonl".replace("+", "_")


def _metric_row(cells: list[str], m: dict, seconds: float | None = None, figures: str | None = None) -> str:
    ks = [f"recall@{k}" for k in KS]
    tail = " | ".join(f"{m[k]:.3f}" for k in ks) + f" | {m['mrr']:.3f}"
    tail += f" | {figures}" if figures is not None else ""
    return "| " + " | ".join(cells) + " | " + tail + (f" | {seconds:.2f} |" if seconds is not None else " |")


def _figures(row: dict) -> str:
    """'31/59': items whose answer figure is in the first FIGURE_K chunks, of the items with such a figure."""
    return f"{len(row['figures'])}/{row['figure_n']}" if "figures" in row else "-"


def _category_table(best: dict, label: str) -> list[str]:
    ks = [f"recall@{k}" for k in KS]
    out = ["", f"By question category, {label} ({best['variant']} {best['retriever']}, filters {best['filters']}):",
           "", "| Category | n | " + " | ".join(ks) + " | MRR |", "|---|---:|" + "---:|" * (len(ks) + 1)]
    by_cat = best["result"]["by_category"]
    return out + [_metric_row([cat, str(by_cat[cat]["n"])], by_cat[cat]) for cat in CATEGORY_ORDER if cat in by_cat]


def _p(p: float) -> str:
    """A sign-test p as the table prints it: '<0.001' rather than a rounded '0.000'."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _pairs_table(rows: list[dict], items: list[GoldItem]) -> list[str]:
    """The PAIRS whose two runs are in `rows` (with their ranked keys)."""
    by_run = {(r["retriever"], r["filters"]): r for r in rows if r["variant"] == TRIAL_VARIANT and "keys" in r}
    found = [(a, b) for a, b in PAIRS if a in by_run and b in by_run]
    if not found or not items:
        return []
    out = ["", f"Paired on {TRIAL_VARIANT}: questions better / worse with b than with a, by recall@5 (all questions "
               f"with gold sources) and by the answer figure in the first {FIGURE_K} chunks (questions whose gold "
               "answer states one), with a two-sided sign test on the questions that differ:", "",
           "| a | b | recall@5 better / worse / same | p | figure better / worse | p |",
           "|---|---|---|---:|---|---:|"]
    for a, b in found:
        ra, rb = by_run[a], by_run[b]
        c = paired_recall(items, ra["keys"], rb["keys"])
        cells = [f"{a[0]}, {a[1]}", f"{b[0]}, {b[1]}", f"{c['better']} / {c['worse']} / {c['same']}",
                 _p(c["sign_test_p"])]
        if "figures" in ra and "figures" in rb:
            win, loss = len(rb["figures"] - ra["figures"]), len(ra["figures"] - rb["figures"])
            cells += [f"{win} / {loss}", _p(sign_test(win, loss))]
        else:
            cells += ["-", "-"]
        out.append("| " + " | ".join(cells) + " |")
    return out


def tables(rows: list[dict], best: dict, entities: dict[str, dict[str, int]] | None = None,
           items: list[GoldItem] | None = None) -> str:
    ks = [f"recall@{k}" for k in KS]
    fig = f"figure@{FIGURE_K}"
    head = "| " + " | ".join(["Variant", "Retriever", "Filters", *ks, "MRR", fig, "s/query"]) + " |"
    grid = [r for r in rows if r["retriever"] in RETRIEVERS and r["filters"] in {s.name for s in GRID}]
    out = [head, "|---|---|---|" + "---:|" * (len(ks) + 3)]
    out += [_metric_row([r["variant"], r["retriever"], r["filters"]], r["result"]["overall"], r["seconds"],
                        _figures(r)) for r in grid]
    trials = [r for r in rows if r["variant"] == TRIAL_VARIANT and r["retriever"] in ("hybrid", RERANKED)
              and r["filters"] != "none"]
    if any(r["retriever"] == RERANKED or "+period" in r["filters"] for r in trials):     # the grid rows alone: no
        order = [s.name for s in GRID[1:] + TRIALS]
        trials.sort(key=lambda r: (r["retriever"] == RERANKED, order.index(r["filters"])))
        out += ["", f"Trials on {TRIAL_VARIANT} (period filter, statement Item first, cross-encoder reranking):", "",
                "| " + " | ".join(["Retriever", "Filters", *ks, "MRR", fig, "s/query"]) + " |",
                "|---|---|" + "---:|" * (len(ks) + 3)]
        out += [_metric_row([r["retriever"], r["filters"]], r["result"]["overall"], r["seconds"], _figures(r))
                for r in trials]
    out += _pairs_table(rows, items or [])
    out += _category_table(best, "best setting")
    # the setting the agent can use without the cross-encoder's latency (ADR-017), when it is not the best already
    realistic = [r for r in rows if r["filters"].startswith("question") and r["retriever"] != RERANKED]
    if realistic and best_row(realistic) is not best:
        out += _category_table(best_row(realistic), "best setting with question-derived filters and no reranker")
    if entities:
        out += ["", "Question-derived filters against the items' own (retrieval/entities.py; exact matches, items "
                    "with gold sources; a period counts as FY when the question names none):", "",
                "| Set | n | companies | fiscal years | periods | all three |", "|---|---:|---:|---:|---:|---:|"]
        out += [f"| {name} | {e['n']} | {e['companies']} | {e['years']} | {e['periods']} | {e['all']} |"
                for name, e in entities.items()]
    n = best["result"]["n"]
    # Why: refusals whose concept is absent from a real statement (e.g. a bank's "gross profit") keep that
    # statement as their gold source, so only source-less items are skipped, not the whole refuse category.
    return "\n".join(out) + (f"\n\n{n} gold questions with gold sources (gold-v1); items without sources "
                             "(out-of-scope refusals) are skipped.\n")


def write_summary(text: str, path: Path = SUMMARY_FILE) -> None:
    block = f"{START}\n{text}{END}"
    old = path.read_text(encoding="utf-8") if path.exists() else f"# Retrieval results (WP5)\n\n{START}\n{END}\n"
    if START not in old or END not in old:
        raise ValueError(f"{path} lost its GENERATED markers; restore them before re-running")
    head, rest = old.split(START, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(head + block + rest.split(END, 1)[1], encoding="utf-8", newline="\n")


def load_reranker():
    """The cross-encoder scoring function (heavy, so imported only for --rerank)."""
    from retrieval.rerank import cross_encoder
    return cross_encoder()


def main(argv: list[str] | None = None) -> int:
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--retrievers", nargs="+", choices=RETRIEVERS, default=list(RETRIEVERS))
    ap.add_argument("--rerank", action="store_true", help=f"also {RERANKED} on {TRIAL_VARIANT} (cross-encoder)")
    ap.add_argument("--gold", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    ap.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="run files and the summary go here")
    args = ap.parse_args(argv)

    rep = validate_file(args.gold)
    if not rep.ok:
        print(f"ERROR: gold set {args.gold} is invalid; run eval/validate_gold.py first", file=sys.stderr)
        return 1
    embedder = None
    try:
        # one embedder (the model is loaded once) for every variant's dense index, with the queries cached
        if {"dense", "hybrid"} & set(args.retrievers):
            embedder = cached(embedder_from_settings())
        try:
            reranker = load_reranker() if args.rerank and "hybrid" in args.retrievers else None
        except OSError as err:                     # the model is not downloaded and the Hub is not reachable
            raise FileNotFoundError(f"the reranker model is not available: {err}") from err
        rows = evaluate(args.variants, args.retrievers, rep.items, embedder, args.index_dir, args.chunks_dir,
                        args.out_dir, reranker)
    except ValidationError as err:
        print(f"ERROR: settings are incomplete; copy .env.example to .env and fill it in.\n{err}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, ImportError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    finally:
        if embedder is not None:
            embedder.close()
    if set(args.variants) != set(VARIANTS) or set(args.retrievers) != set(RETRIEVERS) or not args.rerank:
        print(f"partial run: run files written to {args.out_dir / 'runs'}; {SUMMARY_NAME} left unchanged")
        return 0
    try:
        write_summary(tables(rows, best_row(rows), entity_tables(rep.items), rep.items),
                      args.out_dir / SUMMARY_NAME)
    except ValueError as err:                      # the summary lost its GENERATED markers
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    print(f"tables written to {args.out_dir / SUMMARY_NAME}")
    return 0


def entity_tables(gold: list[GoldItem]) -> dict[str, dict[str, int]]:
    """Entity reading on the development set (where the rules were checked first) and on the gold set."""
    from eval.validate_dev_set import DEV_FILE, DevItem
    dev = validate_file(DEV_FILE, model=DevItem)
    out = {"development set": entity_accuracy(dev.items)} if dev.ok else {}
    return out | {"gold-v1": entity_accuracy(gold)}


def evaluate(variants: list[str], wanted: list[str], items: list[GoldItem], embedder: Embedder | None,
             index_dir: Path = INDEX_DIR, chunks_dir: Path = CHUNKS_DIR, out_dir: Path = RESULTS_DIR,
             reranker=None) -> list[dict]:
    rows, with_figure = [], figure_items(items)
    for variant in variants:
        retrievers = load_retrievers(variant, wanted, embedder, index_dir, chunks_dir)
        texts = {c.chunk_id: c.text for c in read_chunks(chunks_path(variant, chunks_dir))} if retrievers else {}
        if reranker is not None and variant == TRIAL_VARIANT and "hybrid" in retrievers:
            from retrieval.rerank import RerankedIndex
            retrievers[RERANKED] = RerankedIndex(retrievers["hybrid"], reranker)
        for name, retriever in retrievers.items():
            for setting in settings_for(variant, name):
                records, seconds = run(retriever, items, setting)
                write_runs(run_file(out_dir, variant, name, setting.name), records, name, setting.name)
                keys = {r["id"]: [hit_key(h) for h in r["retrieved"]] for r in records}
                result = score_runs(items, keys)
                figures = figure_hits(items, records, texts)
                rows.append({"variant": variant, "retriever": name, "filters": setting.name, "result": result,
                             "seconds": seconds, "keys": keys, "figures": figures, "figure_n": len(with_figure)})
                m = result["overall"]
                print(f"{variant} {name:13} {setting.name:30} " + " ".join(f"{k}={v:.3f}" for k, v in m.items())
                      + f" figure@{FIGURE_K}={len(figures)}/{len(with_figure)} ({seconds:.2f} s/query)", flush=True)
    return rows


if __name__ == "__main__":
    sys.exit(main())
