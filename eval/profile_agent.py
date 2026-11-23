"""W3-07: where an agent run's time and tokens go, per node and per tool, from the reply cache (no request).

    python eval/profile_agent.py eval/results/dev_v2/model_runs/<run>.jsonl --questions dev2
    python eval/profile_agent.py <runs file> --questions dev2 --label after     # the report's name: <run>-after

The run's questions go through the real graph again (agent.nodes.real_nodes, the tools on data/) with a recording
tracer (core/tracing.Tracer): every node, tool call and LLM call is a span with its wall time. Every chat reply must
come from the reply cache, where each entry keeps the seconds its request took when it was sent: a cache miss
raises instead of sending (`cache_only`), so profiling never spends the shared key. A question's time is then
- `llm`: the endpoint seconds of its LLM calls, as recorded when they were sent (the live latency), per node;
- `local`: the wall time of everything else in the replay (retrieval, the tools, the rules and checks), per node and
  per tool. Retrieval embeds the query with this machine's embedder (EMBED_BACKEND), so a search's time is this
  machine's, not the hosted endpoint's.
Their sum is the estimated latency, set beside the latency the runs file recorded. Tokens are the replies' recorded
counts. The report goes to eval/results/profile/<run id>[-<label>].md with one JSON line per question beside it.
Also, with `--runs-only`, the runs file's own figures (latency, tokens, LLM calls per question) without a replay:
for a run that no longer replays (A1, after the W3-04 and W3-06 planner prompts).
Exit codes: 0 = written, 2 = setup, a cache miss or an answer that differs from the runs file.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_agent import QUESTION_SETS, Item, load_items  # noqa: E402
from eval.run_retrieval import RESULTS_DIR  # noqa: E402

PROFILE_DIR = RESULTS_DIR / "profile"
NODES = ("planner", "executor", "composer", "critic", "finish")
LLM_SPAN = "llm"
TOOL_PREFIX = "tool:"


class CacheMiss(RuntimeError):
    """A chat request that is not in the reply cache: profiling sends nothing."""


# ----------------------------------------------------------------------------------------------- the tracer
@dataclass
class SpanRecord:
    name: str
    parent: str | None          # the enclosing node's name (a tool or LLM span), None for a node
    seconds: float = 0.0        # wall time in this process
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] | None = None


class _Recorded:
    def __init__(self, record: SpanRecord) -> None:
        self.record = record

    def tag(self, *tags: str) -> None:
        return None

    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None, level: str | None = None,
               status_message: str | None = None, usage: dict[str, int] | None = None) -> None:
        self.record.metadata |= metadata or {}
        if usage is not None:
            self.record.usage = usage


class ProfilingTracer:
    """A core.tracing.Tracer that keeps every span of a run in memory with its wall time (nothing is sent)."""

    enabled = True

    def __init__(self) -> None:
        self.spans: list[SpanRecord] = []
        self._node: list[str] = []

    @contextlib.contextmanager
    def span(self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None) -> Iterator[_Recorded]:
        record = SpanRecord(name, self._node[-1] if self._node else None, metadata=dict(metadata or {}))
        is_node = name in NODES
        if is_node:
            self._node.append(name)
        start = time.perf_counter()
        try:
            yield _Recorded(record)
        finally:
            record.seconds = time.perf_counter() - start
            if is_node:
                self._node.pop()
            self.spans.append(record)

    def generation(self, name: str, *, model: str, input: Any = None,
                   metadata: dict[str, Any] | None = None) -> contextlib.AbstractContextManager[_Recorded]:
        return self.span(name, input=input, metadata=metadata)

    @contextlib.contextmanager
    def trace(self, info: Any, *, input: Any = None) -> Iterator[_Recorded]:
        self.spans, self._node = [], []
        yield _Recorded(SpanRecord("trace", None))

    def flush(self) -> None:
        return None


# ----------------------------------------------------------------------------------------------- one question
def llm_tokens(record: SpanRecord) -> tuple[int, int]:
    """(prompt, completion) tokens of an LLM span: Langfuse usage when sent, the cached counts on a cache hit."""
    counts = record.usage or record.metadata.get("cached_tokens") or {}
    return int(counts.get("input", 0)), int(counts.get("output", 0))


def question_profile(spans: list[SpanRecord]) -> dict[str, Any]:
    """Per node: runs, LLM calls, endpoint seconds (llm), the replay's other wall time (local), tokens; per tool:
    calls and wall seconds. A node's local time is its wall time less its LLM spans' (their replay is a cache read)."""
    nodes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    tools: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for s in spans:
        if s.name in NODES:
            nodes[s.name]["runs"] += 1
            nodes[s.name]["local"] += s.seconds
        elif s.name == LLM_SPAN:
            node = nodes[s.parent or "other"]
            prompt, completion = llm_tokens(s)
            node["llm_calls"] += 1
            node["llm"] += float(s.metadata.get("seconds", 0.0))
            node["local"] -= s.seconds
            node["prompt_tokens"] += prompt
            node["completion_tokens"] += completion
        elif s.name.startswith(TOOL_PREFIX):
            tool = tools[s.name.removeprefix(TOOL_PREFIX)]
            tool["calls"] += 1
            tool["seconds"] += s.seconds
    as_dict = {k: {m: round(v, 4) for m, v in d.items()} for k, d in nodes.items()}
    llm = sum(d.get("llm", 0.0) for d in as_dict.values())
    local = sum(max(d.get("local", 0.0), 0.0) for d in as_dict.values())
    return {"nodes": as_dict, "tools": {k: {m: round(v, 4) for m, v in d.items()} for k, d in tools.items()},
            "llm_s": round(llm, 3), "local_s": round(local, 3), "estimated_s": round(llm + local, 3),
            "tokens": int(sum(d.get("prompt_tokens", 0) + d.get("completion_tokens", 0) for d in as_dict.values()))}


# ----------------------------------------------------------------------------------------------- the replay
def cache_only(client: Any) -> Any:
    """`client` (core.llm.LLMClient) answering from its reply cache only: a miss raises CacheMiss, nothing is sent."""
    async def refuse(request: dict[str, Any], target_model: str, path: Path | None) -> Any:
        raise CacheMiss(f"a {target_model} reply is not in the reply cache (profiling sends no request)")
    client._fetch = refuse
    return client


async def replay(items: list[Item], nodes: Any, limits: Any) -> list[tuple[Item, Any, dict[str, Any]]]:
    """Each item through the graph with a ProfilingTracer: (item, the run, its profile)."""
    from agent.graph import build_graph, run
    from core.tracing import TraceInfo
    graph, tracer, out = build_graph(nodes), ProfilingTracer(), []
    for item in items:
        result = await run(item.question, list(item.history), graph=graph, limits=limits,
                           trace=TraceInfo(item.id, "profile"), tracer=tracer)
        out.append((item, result, question_profile(tracer.spans)))
    return out


def warm() -> float:
    """The API's start-up warm-up (api.main.real_warm_up), so the questions are profiled as a running server answers
    them; returns its seconds."""
    from api.main import real_warm_up
    start = time.perf_counter()
    real_warm_up()
    return time.perf_counter() - start


async def replay_real(items: list[Item], settings: Any, limits: Any) -> list[tuple[Item, Any, dict[str, Any]]]:
    from agent.nodes import real_nodes
    from core.llm import LLMClient
    async with LLMClient(settings) as client:
        return await replay(items, real_nodes(cache_only(client)), limits)


# ----------------------------------------------------------------------------------------------- the report
def load_records(path: Path) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r.get("id")}


def pct(values: list[float], q: float) -> float:
    """The q-quantile (nearest rank), 0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def runs_summary(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The runs file's own figures per question: latency, tokens, LLM calls, tool calls (errors left out)."""
    ok = [r for r in records.values() if not r.get("error")]
    agent = [r.get("agent") or {} for r in ok]
    latency = [float(r.get("latency_s") or 0.0) for r in ok]
    tokens = [int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0) for r in ok]
    return {"questions": len(records), "errors": len(records) - len(ok),
            "latency_p50": round(statistics.median(latency), 2) if latency else 0.0,
            "latency_p95": round(pct(latency, 0.95), 2), "latency_mean": round(statistics.fmean(latency), 2)
            if latency else 0.0,
            "tokens_p50": int(statistics.median(tokens)) if tokens else 0,
            "tokens_mean": round(statistics.fmean(tokens)) if tokens else 0,
            "llm_calls_mean": round(statistics.fmean(a.get("llm_calls", 0) for a in agent), 2) if agent else 0.0,
            "tool_calls_mean": round(statistics.fmean(a.get("tool_steps", 0) for a in agent), 2) if agent else 0.0}


def summary_lines(name: str, s: dict[str, Any]) -> list[str]:
    return [f"| {name} | {s['questions']} | {s['latency_p50']:.2f} | {s['latency_p95']:.2f} | {s['tokens_p50']} | "
            f"{s['tokens_mean']} | {s['llm_calls_mean']:.2f} | {s['tool_calls_mean']:.2f} |"]


SUMMARY_HEADER = ["| Run | Questions | Latency p50 (s) | Latency p95 (s) | Tokens p50 | Tokens mean | LLM calls / q | "
                  "Tool calls / q |", "|---|---|---|---|---|---|---|---|"]


def report(run_id: str, records: dict[str, dict[str, Any]], profiles: list[dict[str, Any]] | None) -> str:
    """The Markdown report: the runs file's figures, then (after a replay) where the time and tokens go."""
    lines = [f"# Profile: {run_id}", "", "Generated by `eval/profile_agent.py` (W3-07). The runs file's figures, then "
             "where the time goes in a replay from the reply cache (no request): `llm` is the endpoint seconds each "
             "LLM call took when it was sent, `local` the replay's other wall time on this machine (retrieval embeds "
             "with the local embedder).", "", "## The runs file", "", *SUMMARY_HEADER,
             *summary_lines(run_id, runs_summary(records))]
    if not profiles:
        return "\n".join(lines) + "\n"
    n = len(profiles)
    total = sum(p["estimated_s"] for p in profiles) or 1.0
    estimated = [p["estimated_s"] for p in profiles]
    recorded = [float(records.get(p["id"], {}).get("latency_s") or 0.0) for p in profiles]
    tokens = [p["tokens"] for p in profiles]
    llm_calls = [int(sum(d.get("llm_calls", 0) for d in p["nodes"].values())) for p in profiles]
    lines += ["", "## Where the time goes (replay)", "",
              f"{n} questions. Estimated latency p50 {statistics.median(estimated):.2f} s (recorded p50 "
              f"{statistics.median(recorded):.2f} s); LLM seconds are {sum(p['llm_s'] for p in profiles) / total:.0%} "
              f"of the estimated total. Tokens per question p50 {statistics.median(tokens):.0f}, mean "
              f"{statistics.fmean(tokens):.0f}; LLM calls {sum(llm_calls) / n:.2f} per question ({sum(llm_calls)} in "
              "all).", "",
              "| Node | Runs / q | LLM calls / q | LLM s / q | Local s / q | Share of time | Prompt tokens / q | "
              "Completion tokens / q |", "|---|---|---|---|---|---|---|---|"]
    for node in NODES:
        rows = [p["nodes"].get(node, {}) for p in profiles]
        llm = sum(r.get("llm", 0.0) for r in rows)
        local = sum(max(r.get("local", 0.0), 0.0) for r in rows)
        lines.append(f"| {node} | {sum(r.get('runs', 0) for r in rows) / n:.2f} | "
                     f"{sum(r.get('llm_calls', 0) for r in rows) / n:.2f} | {llm / n:.2f} | {local / n:.3f} | "
                     f"{(llm + local) / total:.0%} | {sum(r.get('prompt_tokens', 0) for r in rows) / n:.0f} | "
                     f"{sum(r.get('completion_tokens', 0) for r in rows) / n:.0f} |")
    tools: dict[str, list[float]] = defaultdict(list)
    for p in profiles:
        for name, t in p["tools"].items():
            tools[name] += [t["seconds"] / t["calls"]] * int(t["calls"])
    lines += ["", "| Tool | Calls / q | Local ms per call, p50 | p95 | Local s / q |", "|---|---|---|---|---|"]
    for name in sorted(tools):
        per = tools[name]
        lines.append(f"| {name} | {len(per) / n:.2f} | {statistics.median(per) * 1000:.0f} | "
                     f"{pct(per, 0.95) * 1000:.0f} | {sum(per) / n:.3f} |")
    slowest = sorted(profiles, key=lambda p: -p["estimated_s"])[:5]
    lines += ["", "Slowest questions (estimated): " + ", ".join(f"{p['id']} {p['estimated_s']:.1f} s ("
                                                                 f"{p['nodes'].get('composer', {}).get('llm_calls', 0):.0f} composer, "
                                                                 f"{p['nodes'].get('critic', {}).get('llm_calls', 0):.0f} critic LLM calls)"
                                                                 for p in slowest) + "."]
    return "\n".join(lines) + "\n"


def same_answer(record: dict[str, Any], result: Any) -> bool:
    """The replay answered as the runs file did (the profile describes that run, not another one)."""
    recorded = json.loads(record.get("raw_reply") or "null") or {}
    return recorded.get("answer") == result.answer.answer and recorded.get("verified") == result.answer.verified


def main(argv: list[str] | None = None, *, nodes: Any = None, settings: Any = None, limits: Any = None) -> int:
    from eval.validate_gold import safe_console
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", type=Path, help="a runs file (eval/run_agent.py)")
    ap.add_argument("--questions", choices=sorted(QUESTION_SETS), default="dev2")
    ap.add_argument("--label", help="appended to the report's name (e.g. before, after)")
    ap.add_argument("--runs-only", action="store_true", help="the runs file's figures only, no replay")
    ap.add_argument("--profiles", type=Path, help="rebuild the report from a profile's saved JSON lines (no replay)")
    ap.add_argument("--out-dir", type=Path, default=PROFILE_DIR)
    ap.add_argument("--pack-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)
    try:
        records = load_records(args.runs)
    except (OSError, ValueError) as err:
        print(f"{args.runs}: {err}", file=sys.stderr)
        return 2
    run_id = next(iter(records.values()), {}).get("run_id") or args.runs.stem
    name = f"{run_id}-{args.label}" if args.label else run_id
    profiles: list[dict[str, Any]] | None = None
    if args.profiles:
        profiles = [json.loads(line) for line in args.profiles.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif not args.runs_only:
        packs, _ = QUESTION_SETS[args.questions]
        # a question the run recorded as an error has no answer to compare with: left out of the profile, said so
        errored = sorted(qid for qid, r in records.items() if r.get("error"))
        items = [i for i in load_items([args.pack_dir / p for p in packs]) if i.id in records and i.id not in errored]
        if errored:
            print(f"left out (an error in the runs file): {', '.join(errored)}", file=sys.stderr)
        try:
            from agent.graph import Limits
            if settings is None and nodes is None:
                from core.settings import get_settings
                settings = get_settings()
            limits = limits or Limits.from_settings(settings)
            warm_s = warm() if nodes is None else 0.0
            done = asyncio.run(replay(items, nodes, limits) if nodes is not None
                               else replay_real(items, settings, limits))
        except CacheMiss as err:
            print(f"{err}; nothing written", file=sys.stderr)
            return 2
        except (ValueError, OSError) as err:
            print(f"setup incomplete: {str(err).splitlines()[0]}", file=sys.stderr)
            return 2
        changed = [item.id for item, result, _ in done if not same_answer(records[item.id], result)]
        if changed:
            print(f"the replay answered {', '.join(changed)} differently from {args.runs}: the code is not that run's; "
                  "nothing written", file=sys.stderr)
            return 2
        profiles = [{"id": item.id, **profile} for item, _, profile in done]
        print(f"warm-up {warm_s:.1f} s; first question local {profiles[0]['local_s'] if profiles else 0:.2f} s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = args.out_dir / f"{name}.md"
    md.write_bytes(report(run_id, records, profiles).encode("utf-8"))
    if profiles is not None:
        (args.out_dir / f"{name}.jsonl").write_bytes("".join(json.dumps(p, sort_keys=True) + "\n"
                                                             for p in profiles).encode("utf-8"))
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
