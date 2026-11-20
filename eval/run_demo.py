"""Run the W2-07 demo queries (eval/demo_queries.jsonl) through the whole agent graph and write the transcripts.

    python eval/run_demo.py                      # all 10; writes eval/results/demo/W2-07_demo.{jsonl,md}
    python eval/run_demo.py --max-requests 40    # stop before the 41st live chat request (cache hits are free)
    python eval/run_demo.py --no-cache           # every request live: the timings a user would see

The real nodes (agent.nodes.real_nodes) on the shared endpoint through the reply cache, the tools on data/, one
checkpointer thread per conversation: a follow-up (previous_turn_id) runs after its first turn on that turn's
thread, so memory supplies the company and year (W2-06). Each record is agent.transcript.run_record plus the
demo item's id, category, source and expectation, whether the run met it, and whether it stayed under the time
limit. The demo set: one or more questions per category, from the development set ("source": "dev-…") and new
ones written for the demo ("source": "new") on company-years gold-v1 does not ask about (tests/test_run_demo.py
checks the separation). Never the gold set.
Exit codes: 0 = every query met its expectation within the time limit, 1 = not, 2 = setup or budget stopped it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEMO_FILE = ROOT / "eval" / "demo_queries.jsonl"
DEV_FILE = ROOT / "eval" / "dev_set.jsonl"
OUT_DIR = ROOT / "eval" / "results" / "demo"
EXPECTATIONS = ("answer", "refuse", "partial", "not_reported")
NOT_REPORTED_WORDS = ("not report", "does not disclose", "not disclosed", "not separately")


class BudgetExceeded(RuntimeError):
    pass


class BudgetedLLM:
    """An LLM client that refuses to start a chat request once `max_requests` live ones were sent (cache hits are
    free): the shared key's budget for one demo run. A request already started may still pass it by its own HTTP
    retries (core/llm.py counts them)."""

    def __init__(self, inner: Any, max_requests: int):
        self.inner, self.max_requests = inner, max_requests

    @property
    def usage(self) -> Any:
        return self.inner.usage

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        if self.inner.usage.calls >= self.max_requests:
            raise BudgetExceeded(f"the budget of {self.max_requests} live requests is used up")
        return await self.inner.complete(messages, **kwargs)


def load_items(path: Path = DEMO_FILE) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def conversations(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Each first turn with its follow-ups in order (a follow-up's previous turn must come earlier in the file)."""
    by_id = {d["id"]: d for d in items}
    chains: dict[str, list[dict[str, Any]]] = {}
    for d in items:
        prev = d.get("previous_turn_id")
        if prev is None:
            chains[d["id"]] = [d]
            continue
        root = prev
        while by_id[root].get("previous_turn_id"):
            root = by_id[root]["previous_turn_id"]
        chains[root].append(d)
    return list(chains.values())


def dev_values() -> dict[str, dict[str, Any]]:
    return {d["id"]: d for d in load_items(DEV_FILE)}


def expectation_met(item: dict[str, Any], record: dict[str, Any], dev_value_ok: bool | None = None) -> tuple[bool, str]:
    """Whether the answer is what the demo item expects: a verified answer (with the development value, where the
    item has one), a refusal under its rule, a verified partial refusal with the rule, or a verified answer that
    says the figure is not reported (spec §5, category G)."""
    expect, rule = item["expect"], item.get("refusal_rule")
    if expect in ("answer", "partial", "not_reported") and not record["refused"]:
        from agent.composer import company
        unnamed = [t for t in item.get("companies") or [] if company(t).lower() not in record["answer"].lower()]
        if unnamed:
            return False, f"the answer does not name {', '.join(company(t) for t in unnamed)}"
    if expect == "answer":
        if dev_value_ok is False:
            return False, "verified, but no value equals the development value"
        ok = record["verified"] and not record["refused"] and not record["refusal_rule"]
        return ok, "verified answer" if ok else f"expected a verified answer, got: {record['verdict']}"
    if expect == "refuse":
        ok = record["refused"] and record["refusal_rule"] == rule
        return ok, f"refused under {rule}" if ok else f"expected a refusal under {rule}, got: {record['verdict']} " \
                                                      f"({record['refusal_rule']})"
    if expect == "partial":
        ok = not record["refused"] and record["refusal_rule"] == rule and record["verified"]
        return ok, f"partial refusal under {rule}" if ok else f"expected a verified partial refusal under {rule}, " \
                                                              f"got: {record['verdict']} ({record['refusal_rule']})"
    said = any(w in record["answer"].lower() for w in NOT_REPORTED_WORDS)
    ok = said and not record["refused"] and record["verified"]
    return ok, "verified: says it is not reported" if ok else ("expected a verified answer saying the figure is not "
                                                               f"reported, got: {record['verdict']}")


def dev_value_check(item: dict[str, Any], record: dict[str, Any], dev: dict[str, dict[str, Any]]) -> bool | None:
    """For a development item with a numeric value: whether a value in the answer equals it, sign included, within
    its tolerance_pct (1% and at least 0.05 when the item gives none)."""
    source = dev.get(item["source"])
    if source is None or source.get("gold_value") is None:
        return None
    target, pct = source["gold_value"], source.get("tolerance_pct")
    tol = max(abs(target) / 100, 0.05) if pct is None else abs(target) * pct / 100
    return any(abs(v["value"] - target) <= tol for v in record["values"])


async def run_items(items: list[dict[str, Any]], nodes: Any, limits: Any) -> tuple[list[dict[str, Any]], str | None]:
    """The records of the queries run, in file order, and why the run stopped early (the request budget), if it
    did: the queries answered before it are kept."""
    from agent.graph import build_graph, in_memory_checkpointer, run, thread
    from agent.transcript import run_record
    graph = build_graph(nodes, checkpointer=in_memory_checkpointer())
    dev = dev_values()
    records: list[dict[str, Any]] = []
    order = {d["id"]: n for n, d in enumerate(items)}
    for chain in conversations(items):
        for item in chain:
            try:
                result = await run(item["question"], graph=graph, limits=limits, config=thread(chain[0]["id"]))
            except BudgetExceeded as err:
                return sorted(records, key=lambda r: order[r["id"]]), f"stopped before {item['id']}: {err}"
            record = run_record(result, item["question"])
            value_ok = dev_value_check(item, record, dev)
            met, why = expectation_met(item, record, value_ok)
            records.append({"id": item["id"], "category": item["category"], "source": item["source"],
                            "expect": item["expect"], "expected_rule": item.get("refusal_rule"), "met": met,
                            "why": why, "dev_value_ok": value_ok,
                            "under_time_limit": result.seconds < limits.timeout_s, **record})
            print(f"{item['id']}: {record['verdict']:30} met={met!s:5} {record['seconds']:5.1f}s "
                  f"{record['tokens']:6} tokens {record['steps']:2} tool calls", flush=True)
    return sorted(records, key=lambda r: order[r["id"]]), None


async def run_real(items: list[dict[str, Any]], settings: Any, max_requests: int,
                   limits: Any) -> tuple[list[dict[str, Any]], str | None, Any]:
    """run_items on the real nodes, with an LLM client opened and closed here (core/llm.py) behind the budget."""
    from agent.nodes import real_nodes
    from core.llm import LLMClient
    async with LLMClient(settings) as client:
        budgeted = BudgetedLLM(client, max_requests)
        records, stopped = await run_items(items, real_nodes(budgeted), limits)
        return records, stopped, budgeted.usage


def write_outputs(records: list[dict[str, Any]], out_dir: Path, stem: str, header: str) -> tuple[Path, Path]:
    from agent.transcript import markdown
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl, md = out_dir / f"{stem}.jsonl", out_dir / f"{stem}.md"
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8",
                     newline="\n")
    met = sum(r["met"] for r in records)
    dev_cell = {True: "yes", False: "NO", None: "-"}
    lines = [f"# W2-07 demo: {met}/{len(records)} met their expectation", "", header, "",
             "| id | category | source | verdict | met | dev value | seconds | tokens | tool calls |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['category']} | {r['source']} | {r['verdict']} | {'yes' if r['met'] else 'NO'} | "
              f"{dev_cell[r['dev_value_ok']]} | {r['seconds']} | {r['tokens']} | {r['steps']} |" for r in records]
    lines.append("")
    for r in records:
        lines.append(markdown(r, title=f"{r['id']} ({r['category']}, {r['source']}): {r['why']}"))
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return jsonl, md


def main(argv: list[str] | None = None, *, nodes: Any = None, limits: Any = None) -> int:
    from eval.validate_gold import safe_console
    safe_console()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-requests", type=int, default=40, help="live chat requests this run may send")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--stem", default="W2-07_demo")
    parser.add_argument("--no-cache", action="store_true",
                        help="send every chat request (honest latency); the reply cache is neither read nor written")
    args = parser.parse_args(argv)
    items = load_items()
    from agent.graph import Limits
    if nodes is None:
        try:
            from core.settings import get_settings
            settings = get_settings()
        except (ValueError, OSError) as err:        # pydantic's ValidationError: an incomplete .env
            print(f"setup incomplete (.env): {str(err).splitlines()[0]}", file=sys.stderr)
            return 2
        if args.no_cache:
            settings = settings.model_copy(update={"LLM_CACHE_DIR": None})
        records, stopped, usage = asyncio.run(run_real(items, settings, args.max_requests,
                                                       limits or Limits.from_settings(settings)))
        header = (f"Model {settings.LLM_MODEL}; {len(records)} queries; reply cache "
                  f"{'off' if args.no_cache else 'on'}; live chat requests {usage.calls}, cache hits "
                  f"{usage.cache_hits}.")
    else:                                           # given nodes (a test or a script); the default limits
        records, stopped = asyncio.run(run_items(items, nodes, limits or Limits()))
        header = f"{len(records)} queries."
    if stopped:
        header += f" Incomplete: {stopped}."
    jsonl, md = write_outputs(records, args.out_dir, args.stem, header)
    print(f"wrote {jsonl.name} and {md.name}; " + header)
    if stopped:
        print(stopped, file=sys.stderr)
        return 2
    return 0 if all(r["met"] and r["under_time_limit"] for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
