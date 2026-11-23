"""The trace dashboard (W2-08): latency, steps, tokens and failures per question for one traced run.

    python scripts/trace_dashboard.py --run <run_id> [--days 30] [--out eval/results/traces/<run_id>.md]

Reads the run from Langfuse and writes one Markdown table: a row per question, then the totals (verdicts, latency
p50/p95, mean steps, tool calls and tokens, failures by reason). Needs LANGFUSE_ENABLED=true and the keys in .env.

Source: GET /api/public/v2/observations, the API Langfuse Cloud supports (the v1 trace list, GET /api/public/traces,
answers 410 Gone since 19 Nov 2026). Each traced question is a trace whose spans are named `finsight.ask` and tagged
`q:<id>`, `run:<run_id>` and `verdict:<verified|unverified|refused>`; Langfuse copies the trace's metadata (the run's
summary that agent.graph.run writes from the agent's own state) onto its observations, so one `finsight.ask` row per
trace carries every number: the table matches what the agent did, not what Langfuse inferred. Traces sent by the v3
SDK become readable a few minutes after the run (measured: about 8), so run this after a short wait.

The same view in the Langfuse UI: Tracing, filter `tags = run:<run_id>` (ADR-029).

Exit codes: 0 table written, 1 no trace tagged run:<run_id> yet, 2 tracing is off in .env, 3 Langfuse answered an
HTTP error or could not be reached.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROOT_SPAN = "finsight.ask"                          # core/tracing.py names each question's root span so
OBSERVATIONS = "/api/public/v2/observations"
PAGE_SIZE = 200
MAX_PAGES = 100                                     # 20,000 rows: a safety stop, far above any run of ours
DEFAULT_DAYS = 30                                   # the Hobby plan keeps 30 days of data


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def fetch_traces(http: Any, run_id: str, *, days: int = DEFAULT_DAYS, now: datetime | None = None,
                 page_size: int = PAGE_SIZE) -> list[dict[str, Any]]:
    """One `finsight.ask` row per trace tagged run:<run_id>, oldest first. `http` is an httpx.Client with the
    project's base URL and keys (a client on a mock transport in tests); paging follows `meta.cursor`."""
    end = now or datetime.now(UTC)
    params: dict[str, Any] = {
        "name": ROOT_SPAN, "fields": "core,basic,metadata,metrics,trace_context", "limit": page_size,
        "fromStartTime": _iso(end - timedelta(days=days)), "toStartTime": _iso(end + timedelta(minutes=1)),
        "expandMetadata": "notes,visits,stop_reason",
        "filter": json.dumps([{"type": "arrayOptions", "column": "tags", "operator": "any of", "value": [f"run:{run_id}"]}]),
    }
    rows: dict[str, dict[str, Any]] = {}
    for _ in range(MAX_PAGES):
        response = http.get(OBSERVATIONS, params=params)
        response.raise_for_status()
        body = response.json()
        for obs in body.get("data") or []:       # newest first; an empty page can still carry a cursor
            trace_id = obs.get("traceId")
            # a v3-SDK trace can show two finsight.ask rows (an outer row without latency, and the span); keep the
            # span, whichever arrives first: both carry the same trace metadata
            if trace_id and (trace_id not in rows
                             or (rows[trace_id].get("latency") is None and obs.get("latency") is not None)):
                rows[trace_id] = obs
        cursor = (body.get("meta") or {}).get("cursor")
        if not cursor:
            break
        params["cursor"] = cursor
    else:
        print(f"warning: stopped after {MAX_PAGES} pages with more rows left; the table is incomplete",
              file=sys.stderr)
    return sorted(rows.values(), key=_started)


def _started(obs: dict[str, Any]) -> datetime:
    """The row's start time as a datetime: the API's timestamps vary in precision ("10:00:00Z", "10:00:00.5Z"), so
    sorting the strings would misorder them."""
    try:
        return datetime.fromisoformat(str(obs.get("startTime")).replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


def _tag(tags: list[str] | None, prefix: str) -> str | None:
    return next((t[len(prefix):] for t in tags or [] if t.startswith(prefix)), None)


def _decoded(value: Any) -> Any:
    """A metadata value as the agent wrote it. The Langfuse SDK sends each non-string metadata value as JSON text
    (OpenTelemetry attributes), so the API may return "3.2", "null" or '{"planner": 1}': decode those, keep text."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value, parse_constant=lambda _: value)    # "NaN" / "Infinity" stay text, never numbers
    except ValueError:
        return value


def _number(value: Any) -> float | int | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def row(obs: dict[str, Any]) -> dict[str, Any]:
    """One question's numbers from its `finsight.ask` observation row (tolerant of a trace whose run failed before
    writing a summary: then only the tags and the span's latency are known)."""
    raw = obs.get("metadata") if isinstance(obs.get("metadata"), dict) else {}
    meta = {key: _decoded(value) for key, value in raw.items()}
    visits = meta.get("visits") if isinstance(meta.get("visits"), dict) else {}
    notes = meta.get("notes") if isinstance(meta.get("notes"), list) else []
    tags = obs.get("tags") if isinstance(obs.get("tags"), list) else []
    verdict = _tag(tags, "verdict:")
    seconds = _number(meta.get("seconds"))
    if verdict is None:
        failure = "no verdict (the run raised)"
    elif verdict != "unverified":                       # verified or refused: not a failure, even if a limit broke
        failure = ""                                    # after the critic passed the answer (ADR-024)
    elif meta.get("stop_reason"):
        failure = str(meta["stop_reason"])
    else:
        failure = "; ".join(str(n) for n in notes) or "unverified"
    # the q: tag arrives whole; a metadata value over 200 characters would come back cut
    return {"question": str(_tag(tags, "q:") or raw.get("question_id") or obs.get("traceId")),
            "verdict": verdict or "none", "seconds": seconds if seconds is not None else _number(obs.get("latency")),
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
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="how far back to look (default 30)")
    parser.add_argument("--out", type=Path, help="write the Markdown here (default: print it)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):              # a Windows console (cp1252) cannot print "…" or "·"
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import httpx

    from core.settings import get_settings
    settings = get_settings()
    if not settings.LANGFUSE_ENABLED:
        print("LANGFUSE_ENABLED is false in .env: nothing to read (see .env.example).", file=sys.stderr)
        return 2
    auth = (settings.LANGFUSE_PUBLIC_KEY.get_secret_value(), settings.LANGFUSE_SECRET_KEY.get_secret_value())
    try:
        with httpx.Client(base_url=settings.LANGFUSE_HOST, auth=auth, timeout=60) as http:
            rows = [row(obs) for obs in fetch_traces(http, args.run, days=args.days)]
    except httpx.HTTPStatusError as err:
        status = err.response.status_code
        hint = {401: "check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY and that LANGFUSE_HOST is the project's region",
                403: "the API key lacks read access", 410: "Langfuse removed this API: see ADR-029",
                429: "rate limited: try again in a minute"}.get(status, "see the Langfuse status page")
        print(f"Langfuse answered HTTP {status} ({hint}).", file=sys.stderr)
        return 3
    except httpx.HTTPError as err:                     # no connection, timeout
        print(f"could not reach Langfuse at {settings.LANGFUSE_HOST}: {type(err).__name__}", file=sys.stderr)
        return 3
    if not rows:
        print(f"no traces tagged run:{args.run} yet (traces from the v3 SDK take a few minutes to appear; "
              f"looked back {args.days} days)", file=sys.stderr)
        return 1
    text = render(rows, args.run)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")      # LF on Windows too (W3-09)
        print(f"written: {args.out} ({len(rows)} questions)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
