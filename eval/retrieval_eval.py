"""Retrieval metrics on the gold set: recall@k and MRR, measured separately from answer quality.

Contract with WP5 (retrieval):
  1. For every answerable gold item, build the query with `build_query(item, items_by_id)` (follow-up
     questions get the previous question prepended, exactly as the agent's memory will).
  2. Run the retriever and write one JSON line per item to a "runs" file:
        {"id": "B07", "retriever": "hybrid", "filters": "company+year",
         "retrieved": [{"ticker": "MSFT", "fiscal_year": 2024, "fiscal_period": "FY", "item": "Item 8"}, ...]}
     (ranked best first; extra fields such as chunk_id or score are allowed and ignored)
  3. Score it:  python eval/retrieval_eval.py --runs eval/results/runs_c2_hybrid.jsonl [--out table.md]

A retrieved chunk is a HIT for a gold source when (ticker, fiscal_year, fiscal_period, item) match
(fiscal_period separates the 10-Q quarters, which share Item names; docs/decisions.md ADR-009).
recall@k = share of the item's gold sources found in the top k (multi-company items have several);
MRR = 1 / rank of the first hit (0 if none in the list). Refusal items (no gold sources) are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.schema import GoldItem  # noqa: E402
from eval.validate_gold import DEFAULT_PATH, safe_console, validate_file  # noqa: E402

KS = (1, 5, 10)
METRIC_DECIMALS = 3
SourceKey = tuple[str, int, str, str]


def build_query(item: GoldItem, items_by_id: dict[str, GoldItem]) -> str:
    """The text to retrieve with. Follow-ups carry the previous turn(s), oldest first."""
    parts, cur, seen = [item.question], item, {item.id}
    while cur.previous_turn_id and cur.previous_turn_id in items_by_id and cur.previous_turn_id not in seen:
        cur = items_by_id[cur.previous_turn_id]
        seen.add(cur.id)
        parts.insert(0, cur.question)
    return " ".join(parts)


def hit_key(record: dict) -> SourceKey:
    return (str(record["ticker"]).strip().upper(), int(record["fiscal_year"]),
            str(record["fiscal_period"]).strip().upper(), " ".join(str(record["item"]).split()))


def recall_at_k(retrieved: list[SourceKey], gold: list[SourceKey], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("recall@k is undefined without gold sources")
    return len(gold_set & set(retrieved[:k])) / len(gold_set)


def reciprocal_rank(retrieved: list[SourceKey], gold: list[SourceKey]) -> float:
    gold_set = set(gold)
    return next((1.0 / rank for rank, key in enumerate(retrieved, start=1) if key in gold_set), 0.0)


def score_runs(items: list[GoldItem], runs: dict[str, list[SourceKey]], ks: tuple[int, ...] = KS) -> dict:
    """Per-item and aggregated metrics. Items with no run count as 0 (and are listed as missing)."""
    answerable = [i for i in items if i.gold_sources]
    per_item, missing = [], []
    for item in answerable:
        retrieved = runs.get(item.id)
        if retrieved is None:
            missing.append(item.id)
            retrieved = []
        gold = [s.key() for s in item.gold_sources]
        row = {"id": item.id, "category": item.category, "mrr": reciprocal_rank(retrieved, gold)}
        row.update({f"recall@{k}": recall_at_k(retrieved, gold, k) for k in ks})
        per_item.append(row)

    def mean(rows: list[dict], key: str) -> float:
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    metrics = [f"recall@{k}" for k in ks] + ["mrr"]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in per_item:
        by_cat[r["category"]].append(r)
    return {
        "n": len(per_item),
        "missing": missing,
        "overall": {m: round(mean(per_item, m), METRIC_DECIMALS) for m in metrics},
        "by_category": {c: {"n": len(rows), **{m: round(mean(rows, m), METRIC_DECIMALS) for m in metrics}}
                        for c, rows in sorted(by_cat.items())},
        "per_item": per_item,
    }


def read_runs(path: Path) -> tuple[dict[str, list[SourceKey]], dict]:
    runs, meta = {}, {}
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if rec["id"] in runs:
                raise ValueError(f"duplicate id {rec['id']}")
            runs[rec["id"]] = [hit_key(r) for r in rec["retrieved"]]
        except (KeyError, TypeError, ValueError) as err:     # JSONDecodeError is a ValueError
            raise ValueError(f"{path}:{n}: bad run record ({err})") from err
        meta = {k: rec.get(k) for k in ("retriever", "filters")} if not meta else meta
    return runs, meta


def to_markdown(result: dict, meta: dict) -> str:
    ks = [k for k in result["overall"] if k.startswith("recall@")]
    head = "| Scope | n | " + " | ".join(ks) + " | MRR |\n|---|---|" + "---|" * (len(ks) + 1) + "\n"
    rows = [f"| **overall** | {result['n']} | " + " | ".join(f"{result['overall'][k]:.3f}" for k in ks)
            + f" | {result['overall']['mrr']:.3f} |"]
    for cat, m in result["by_category"].items():
        rows.append(f"| {cat} | {m['n']} | " + " | ".join(f"{m[k]:.3f}" for k in ks) + f" | {m['mrr']:.3f} |")
    title = f"Retriever: {meta.get('retriever') or '?'} · filters: {meta.get('filters') or '?'}"
    note = f"\nMissing runs (scored 0): {result['missing']}\n" if result["missing"] else ""
    return f"### {title}\n\n{head}" + "\n".join(rows) + "\n" + note


def main(argv: list[str] | None = None) -> int:
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="append the markdown table to this file")
    args = ap.parse_args(argv)

    rep = validate_file(args.gold)
    if not rep.ok:
        print(f"ERROR: gold set {args.gold} is invalid; run eval/validate_gold.py first", file=sys.stderr)
        return 1
    try:
        runs, meta = read_runs(args.runs)
    except (OSError, ValueError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    unknown = sorted(set(runs) - {i.id for i in rep.items})
    if unknown:
        print(f"WARNING: run ids not in the gold set (ignored): {unknown}", file=sys.stderr)
    result = score_runs(rep.items, runs)
    table = to_markdown(result, meta)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as f:
            f.write(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
