"""WP5 step 5: recall@k and MRR of every retriever on the frozen gold set (gold-v1).

Usage:
    python eval/run_retrieval.py                          # every variant with an index, all retrievers
    python eval/run_retrieval.py --variants C3 --retrievers bm25 hybrid

For each chunk variant (C1/C2/C3), retriever (bm25, dense, hybrid) and filter setting it runs every answerable
gold question and writes the ranked chunks to eval/results/runs/<variant>_<retriever>_<filters>.jsonl in the
format eval/retrieval_eval.py scores. Queries come from retrieval_eval.build_query (a follow-up carries the
previous question). Filter settings:
    none          the question only;
    company+year  metadata filters from the gold item's companies and fiscal years, as the planner will set
                  them. An item about several companies or years gets one filtered search per (company, year),
                  and the result lists are interleaved, as the agent issues one search per entity.
The tables are written to eval/results/retrieval_summary.md between the GENERATED markers; text outside the
markers (the interpretation) is kept.
Exit codes: 0 = done, 1 = gold set invalid, 2 = an index is missing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.retrieval_eval import KS, build_query, hit_key, score_runs  # noqa: E402
from eval.schema import GoldItem  # noqa: E402
from eval.validate_gold import DEFAULT_PATH, safe_console, validate_file  # noqa: E402
from retrieval.bm25 import BM25Index, index_path  # noqa: E402
from retrieval.chunking import VARIANTS  # noqa: E402
from retrieval.common import Filters, SearchResult, chunks_path, read_chunks  # noqa: E402
from retrieval.dense import Embedder, embedder_from_settings  # noqa: E402
from retrieval.hybrid import HybridIndex, Retriever  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARY_FILE = RESULTS_DIR / "retrieval_summary.md"
RETRIEVERS = ("bm25", "dense", "hybrid")
FILTER_SETTINGS = ("none", "company+year")
TOP_K = max(KS)
START, END = "<!-- GENERATED:START (eval/run_retrieval.py) -->", "<!-- GENERATED:END -->"
CATEGORY_ORDER = ("fact", "ratio", "period_comparison", "multi_company", "qualitative", "multi_turn", "refuse")


def item_filters(item: GoldItem) -> list[Filters]:
    """One filter per (company, fiscal year) of the gold item, in the order they are listed."""
    return [Filters.of(ticker=t, fiscal_year=y) for t in item.companies for y in item.fiscal_years]


def interleave(rankings: list[list[SearchResult]], k: int) -> list[SearchResult]:
    """Round-robin merge (1st of each list, then 2nd of each...), without repeating a chunk."""
    out, seen = [], set()
    for tier in zip_longest(*rankings):
        for hit in tier:
            if hit is not None and hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                out.append(hit)
    return out[:k]


def run(retriever: Retriever, items: list[GoldItem], setting: str) -> tuple[list[dict], float]:
    """Ranked results for every answerable item, and the mean seconds per question."""
    by_id = {i.id: i for i in items}
    records, start = [], time.perf_counter()
    answerable = [i for i in items if i.gold_sources]
    for item in answerable:
        query = build_query(item, by_id)
        if setting == "none":
            hits = retriever.search(query, TOP_K)
        else:
            hits = interleave([retriever.search(query, TOP_K, f) for f in item_filters(item)], TOP_K)
        records.append({"id": item.id, "retrieved": [
            {"chunk_id": h.chunk_id, "ticker": h.ticker, "fiscal_year": h.fiscal_year,
             "fiscal_period": h.fiscal_period, "item": h.item, "score": round(h.score, 6)} for h in hits]})
    return records, (time.perf_counter() - start) / max(len(answerable), 1)


def load_retrievers(variant: str, wanted: list[str], embedder: Embedder | None) -> dict[str, Retriever]:
    out: dict[str, Retriever] = {}
    bm25 = BM25Index.load(index_path(variant))
    if "bm25" in wanted:
        out["bm25"] = bm25
    if embedder is not None:
        from retrieval.dense import DenseIndex, collection_name, open_client
        collection = open_client().get_collection(collection_name(variant))
        dense = DenseIndex(collection, read_chunks(chunks_path(variant)), embedder)
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


def tables(rows: list[dict], best: dict) -> str:
    ks = [f"recall@{k}" for k in KS]
    out = ["| Variant | Retriever | Filters | " + " | ".join(ks) + " | MRR | s/query |",
           "|---|---|---|" + "---:|" * (len(ks) + 2)]
    for r in rows:
        m = r["result"]["overall"]
        out.append(f"| {r['variant']} | {r['retriever']} | {r['filters']} | "
                   + " | ".join(f"{m[k]:.3f}" for k in ks) + f" | {m['mrr']:.3f} | {r['seconds']:.2f} |")
    by_cat = best["result"]["by_category"]
    out += ["", f"By question category, best setting ({best['variant']} {best['retriever']}, filters "
                f"{best['filters']}):", "", "| Category | n | " + " | ".join(ks) + " | MRR |",
            "|---|---:|" + "---:|" * (len(ks) + 1)]
    for cat in [c for c in CATEGORY_ORDER if c in by_cat]:
        m = by_cat[cat]
        out.append(f"| {cat} | {m['n']} | " + " | ".join(f"{m[k]:.3f}" for k in ks) + f" | {m['mrr']:.3f} |")
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


def main(argv: list[str] | None = None) -> int:
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--retrievers", nargs="+", choices=RETRIEVERS, default=list(RETRIEVERS))
    ap.add_argument("--gold", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args(argv)

    rep = validate_file(args.gold)
    if not rep.ok:
        print(f"ERROR: gold set {args.gold} is invalid; run eval/validate_gold.py first", file=sys.stderr)
        return 1
    # one embedder (the model is loaded once) for every variant's dense index
    embedder = embedder_from_settings() if {"dense", "hybrid"} & set(args.retrievers) else None
    try:
        rows = evaluate(args.variants, args.retrievers, rep.items, embedder)
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    finally:
        if embedder is not None:
            embedder.close()
    best = max(rows, key=lambda r: (r["result"]["overall"]["recall@5"], r["result"]["overall"]["mrr"]))
    write_summary(tables(rows, best))
    print(f"tables written to {SUMMARY_FILE}")
    return 0


def evaluate(variants: list[str], wanted: list[str], items: list[GoldItem], embedder: Embedder | None) -> list[dict]:
    rows = []
    for variant in variants:
        retrievers = load_retrievers(variant, wanted, embedder)
        for name, retriever in retrievers.items():
            for setting in FILTER_SETTINGS:
                records, seconds = run(retriever, items, setting)
                tag = setting.replace("+", "_")
                write_runs(RESULTS_DIR / "runs" / f"{variant}_{name}_{tag}.jsonl", records, name, setting)
                result = score_runs(items, {r["id"]: [hit_key(h) for h in r["retrieved"]] for r in records})
                rows.append({"variant": variant, "retriever": name, "filters": setting, "result": result,
                             "seconds": seconds})
                m = result["overall"]
                print(f"{variant} {name:6} {setting:12} " + " ".join(f"{k}={v:.3f}" for k, v in m.items())
                      + f" ({seconds:.2f} s/query)", flush=True)
    return rows


if __name__ == "__main__":
    sys.exit(main())
