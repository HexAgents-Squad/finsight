"""The trace dashboard (W2-08): latency, steps, tokens and failures per question for one traced run.

    python scripts/trace_dashboard.py --run <run_id> [--out eval/results/traces/<run_id>.md]

Reads the run's traces from Langfuse by their tag `run:<run_id>` (each traced question carries it, with `q:<id>` and
`verdict:<verified|unverified|refused>`; core/tracing.py) and writes one Markdown table: a row per question, then the
totals (verdicts, latency p50/p95, mean steps, tool calls and tokens, failures by reason). Needs LANGFUSE_ENABLED=true
and the keys in .env, and the `langfuse` package. The numbers come from each trace's metadata, which agent.graph.run
writes from the run's own state, so the table matches what the agent did, not what Langfuse inferred.

The same view in the Langfuse UI: Traces, filter `tags = run:<run_id>`, columns latency, tokens and tags (ADR-029).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAGE_SIZE = 50
MAX_PAGES = 200                                     # 10,000 traces: a safety stop, far above any run of ours


def fetch_traces(client: Any, run_id: str, page_size: int = PAGE_SIZE) -> list[Any]:
    """Every trace tagged run:<run_id>, oldest first (`client` is a langfuse.Langfuse, or a fake in tests)."""
    traces: list[Any] = []
    for page in range(1, MAX_PAGES + 1):
        batch = client.api.trace.list(tags=[f"run:{run_id}"], page=page, limit=page_size, order_by="timestamp.asc")
        traces.extend(batch.data)
        if page >= (batch.meta.total_pages or 0):
            break
    return traces


def _tag(tags: list[str] | None, prefix: str) -> str | None:
    return next((t[len(prefix):] for t in tags or [] if t.startswith(prefix)), None)


def _decoded(value: Any) -> Any:
    """A metadata value as the agent wrote it. The Langfuse SDK sends each non-string metadata value as JSON text
    (OpenTelemetry attributes), so the API may return "3.2", "null" or '{"planner": 1}': decode those, keep text."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _number(value: Any) -> float | int | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def row(trace: Any) -> dict[str, Any]:
    """One question's numbers from its trace (tolerant of a trace whose run failed before writing a summary)."""
    raw = trace.metadata if isinstance(trace.metadata, dict) else {}
    meta = {key: _decoded(value) for key, value in raw.items()}
    visits = meta.get("visits") if isinstance(meta.get("visits"), dict) else {}
    notes = meta.get("notes") if isinstance(meta.get("notes"), list) else []
    verdict = _tag(trace.tags, "verdict:")
    seconds = _number(meta.get("seconds"))
    if verdict is None:
        failure = "no verdict (the run raised)"
    elif verdict != "unverified":                       # verified or refused: not a failure, even if a limit broke
        failure = ""                                    # after the critic passed the answer (ADR-024)
    elif meta.get("stop_reason"):
        failure = str(meta["stop_reason"])
    else:
        failure = "; ".join(str(n) for n in notes) or "unverified"
    return {"question": str(raw.get("question_id") or _tag(trace.tags, "q:") or trace.id),
            "verdict": verdict or "none", "seconds": seconds if seconds is not None else trace.latency,
            "steps": sum(v for v in visits.values() if _number(v) is not None),
            "tool_calls": _number(meta.get("tool_calls")), "tokens": _number(meta.get("tokens")), "failure": failure}


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def _num(value: Any, digits: int = 0) -> str:
    if not isinstance(value, int | float):
        return "–"
    return f"{value:,.{digits}f}"


def _cell(text: str, limit: int = 90) -> str:
    text = text.replace("|", "/").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render(rows: list[dict[str, Any]], run_id: str) -> str:
    """The dashboard as Markdown: per-question table, then totals."""
    lines = [f"# Trace dashboard: run {run_id}", "",
             f"{len(rows)} traced question(s). Source: Langfuse traces tagged `run:{run_id}` (scripts/trace_dashboard.py).",
             "", "| Question | Verdict | Seconds | Steps | Tool calls | Tokens | Failure |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['question']} | {r['verdict']} | {_num(r['seconds'], 1)} | {r['steps']} | "
                     f"{_num(r['tool_calls'])} | {_num(r['tokens'])} | {_cell(r['failure'])} |")
    seconds = [r["seconds"] for r in rows if isinstance(r["seconds"], int | float)]
    verdicts = Counter(r["verdict"] for r in rows)
    failures = Counter(r["failure"].split(":")[0] for r in rows if r["failure"])

    def mean(key: str) -> str:
        vals = [r[key] for r in rows if isinstance(r[key], int | float)]
        return _num(sum(vals) / len(vals), 1) if vals else "–"

    lines += ["", "## Totals", "",
              "- Verdicts: " + (", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())) or "none"),
              f"- Seconds per question: p50 {_num(_percentile(seconds, 50), 1)}, p95 {_num(_percentile(seconds, 95), 1)}, "
              f"max {_num(max(seconds) if seconds else None, 1)}",
              f"- Mean per question: {mean('steps')} node runs, {mean('tool_calls')} tool calls, {mean('tokens')} tokens",
              "- Failures by reason: " + (", ".join(f"{_cell(k, 60)} ({v})" for k, v in failures.most_common()) or "none")]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="the run id given to TraceInfo(run_id=...)")
    parser.add_argument("--out", type=Path, help="write the Markdown here (default: print it)")
    args = parser.parse_args(argv)
    from core.settings import get_settings
    from core.tracing import LangfuseTracer
    settings = get_settings()
    if not settings.LANGFUSE_ENABLED:
        print("LANGFUSE_ENABLED is false in .env: nothing to read (see .env.example).", file=sys.stderr)
        return 2
    client = LangfuseTracer.from_settings(settings).client
    rows = [row(t) for t in fetch_traces(client, args.run)]
    text = render(rows, args.run)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"written: {args.out} ({len(rows)} questions)")
    else:
        print(text)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
