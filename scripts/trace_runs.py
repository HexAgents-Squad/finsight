"""Trace a set of questions in Langfuse, one run id for the set (W3-09): the trace dashboard's real runs.

    python scripts/trace_runs.py --run-id W3-09-demo-a5 --demo --tag version:A5 --max-requests 40
    python scripts/trace_runs.py --run-id W3-09-a1-failures-a5 --gold A06 B04 --tag version:A5 --cache-only

Each question goes through the agent graph on the real nodes (agent.nodes.real_nodes) with tracing on, as one trace
tagged q:<id>, run:<run_id> and the --tag values (core/tracing.py masks e-mail addresses and phone numbers in
everything sent). The demo queries (eval/demo_queries.jsonl) run as conversations, a follow-up on its first turn's
thread, as eval/run_demo.py runs them; gold questions (--gold ids, from the gold and refusal context packs) each run
with their earlier questions as history, as eval/run_agent.py runs them, never their answers. Replies come through
the reply cache: `--cache-only` sends no chat request at all (a reply that is not cached raises), for a set whose
replies a committed run already made; otherwise at most --max-requests live chat requests. A summary line per
question goes to eval/results/traces/<run_id>.jsonl; the table is scripts/trace_dashboard.py's, read back from
Langfuse. Needs LANGFUSE_ENABLED=true and the keys in .env.

The script uses only the graph's, the client's and the runners' long-standing entry points, so it also runs from a
checkout of an earlier version (W3-09 traced A1's own code from a worktree at its commit).
Exit codes: 0 every question traced, 1 a question raised, 2 setup, budget or a reply not in the cache.
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

OUT_DIR = ROOT / "eval" / "results" / "traces"


class NotCached(RuntimeError):
    """A chat reply that is not in the reply cache, in a --cache-only run."""


def cache_only(client: Any) -> Any:
    """`client` (core.llm.LLMClient) answering from its reply cache only: a miss raises NotCached."""
    async def refuse(request: dict[str, Any], target_model: str, path: Path | None) -> Any:
        raise NotCached(f"a {target_model} reply is not in the reply cache (--cache-only sends no request)")
    client._fetch = refuse
    return client


def gold_items(ids: list[str]) -> list[dict[str, Any]]:
    """The gold questions and refusal tests with these ids, with their earlier questions, in the order given."""
    from eval.run_agent import GOLD_PACK, REFUSAL_PACK, RESULTS_DIR, load_items
    by_id = {i.id: i for i in load_items([RESULTS_DIR / GOLD_PACK, RESULTS_DIR / REFUSAL_PACK])}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"not gold questions or refusal tests: {', '.join(missing)}")
    return [{"id": i, "question": by_id[i].question, "history": list(by_id[i].history)} for i in ids]


def summary(item_id: str, question: str, result: Any) -> dict[str, Any]:
    """What the run answered, for the local record (the trace in Langfuse has everything)."""
    state = result.state
    return {"id": item_id, "question": question, "answer": result.answer.answer, "verified": result.answer.verified,
            "refused": result.answer.refused, "refusal_rule": result.answer.refusal_rule,
            "seconds": round(result.seconds, 3), "tokens": int(state.get("tokens") or 0),
            "tool_calls": len(state.get("tool_results") or []), "visits": dict(state.get("visits") or {}),
            "stop_reason": state.get("stop_reason")}


async def trace_demo(nodes: Any, limits: Any, info: Any, tracer: Any = None) -> list[dict[str, Any]]:
    """The demo queries as conversations, each question traced; a follow-up runs on its first turn's thread."""
    from agent.graph import build_graph, in_memory_checkpointer, run, thread
    from eval.run_demo import DEMO_FILE, conversations, load_items
    graph = build_graph(nodes, checkpointer=in_memory_checkpointer())
    out = []
    for chain in conversations(load_items(DEMO_FILE)):
        for item in chain:
            result = await run(item["question"], graph=graph, limits=limits, config=thread(chain[0]["id"]),
                               trace=info(item["id"], chain[0]["id"]), tracer=tracer)
            out.append(summary(item["id"], item["question"], result))
            print(f"{item['id']}: verified={result.answer.verified} refused={result.answer.refused}", flush=True)
    return out


async def trace_gold(nodes: Any, limits: Any, info: Any, items: list[dict[str, Any]],
                     tracer: Any = None) -> list[dict[str, Any]]:
    """Each gold question with its earlier questions as history (never an answer), each traced."""
    from agent.graph import build_graph, run
    graph = build_graph(nodes)
    out = []
    for item in items:
        result = await run(item["question"], item["history"], graph=graph, limits=limits, trace=info(item["id"]),
                           tracer=tracer)
        out.append(summary(item["id"], item["question"], result))
        print(f"{item['id']}: verified={result.answer.verified} refused={result.answer.refused}", flush=True)
    return out


async def trace_set(args: argparse.Namespace, settings: Any, items: list[dict[str, Any]] | None) -> list[dict]:
    from agent.graph import Limits
    from agent.nodes import real_nodes
    from core.llm import LLMClient
    from core.tracing import TraceInfo
    from eval.run_demo import BudgetedLLM

    def info(question_id: str, session_id: str | None = None) -> Any:
        return TraceInfo(question_id=question_id, run_id=args.run_id, session_id=session_id, tags=tuple(args.tag))

    limits = Limits.from_settings(settings)
    async with LLMClient(settings) as client:
        llm = cache_only(client) if args.cache_only else BudgetedLLM(client, args.max_requests)
        nodes = real_nodes(llm)
        records = await (trace_demo(nodes, limits, info) if items is None else trace_gold(nodes, limits, info, items))
        print(f"chat requests sent: {client.usage.calls}, cache hits: {client.usage.cache_hits}")
        return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-id", required=True, help="becomes the tag run:<id> (scripts/trace_dashboard.py --run)")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--demo", action="store_true", help="the 10 demo queries (eval/demo_queries.jsonl)")
    which.add_argument("--gold", nargs="+", metavar="ID", help="gold question or refusal test ids")
    ap.add_argument("--tag", action="append", default=[], help="an extra trace tag, e.g. version:A5 (repeatable)")
    ap.add_argument("--cache-only", action="store_true", help="send no chat request (every reply must be cached)")
    ap.add_argument("--max-requests", type=int, default=40, help="live chat requests at most (without --cache-only)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    from core import tracing
    from core.settings import get_settings
    try:
        settings = get_settings()
        if not settings.LANGFUSE_ENABLED:
            print("LANGFUSE_ENABLED is not true in .env: nothing would be traced", file=sys.stderr)
            return 2
        items = None if args.demo else gold_items(args.gold)
    except (ValueError, OSError) as err:
        print(f"setup: {str(err).splitlines()[0]}", file=sys.stderr)
        return 2
    from eval.run_demo import BudgetExceeded
    try:
        records = asyncio.run(trace_set(args, settings, items))
    except (NotCached, BudgetExceeded) as err:
        print(f"{err}; stopped, nothing written (the traces sent so far stay in Langfuse)", file=sys.stderr)
        return 2
    finally:
        tracing.flush()                              # the last traces of the process are sent before it exits
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.run_id}.jsonl"
    path.write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8"))
    print(f"wrote {path}; Langfuse: Tracing, filter tags = run:{args.run_id} (a few minutes after the run), then "
          f"python scripts/trace_dashboard.py --run {args.run_id} --out eval/results/traces/{args.run_id}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
