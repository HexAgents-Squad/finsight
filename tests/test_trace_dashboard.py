"""W2-08: the trace dashboard and the demo script (offline: a mock HTTP transport for Langfuse's v2 observations API,
no network, no .env), plus one check against the real Langfuse SDK that runs only where `langfuse` is installed (CI
does not install it)."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

import scripts.trace_dashboard as td
import scripts.trace_demo as demo


def trace(qid, verdict="verified", seconds=3.2, visits=None, tool_calls=2, tokens=1900, stop=None, notes=(),
          latency=9.9, metadata=True, start="2026-11-20T10:00:00Z", trace_id=None):
    """A `finsight.ask` row as GET /api/public/v2/observations returns it (shape seen on the team's project)."""
    meta = {"question_id": qid, "run_id": "r1", "seconds": seconds, "tokens": tokens, "tool_calls": tool_calls,
            "visits": visits or {"planner": 1, "executor": 1, "composer": 1, "critic": 1}, "verified": verdict == "verified",
            "refused": verdict == "refused", "stop_reason": stop, "notes": list(notes),
            "scope": {"name": "langfuse-sdk"}, "resourceAttributes": {"telemetry.sdk.language": "python"}}
    tags = [f"q:{qid}", "run:r1"] + ([f"verdict:{verdict}"] if verdict else [])
    return {"id": "o-" + qid, "traceId": trace_id or "t-" + qid, "name": "finsight.ask", "type": "SPAN",
            "startTime": start, "latency": latency, "tags": tags, "traceName": "finsight.ask",
            "metadata": meta if metadata else None}


def api(rows, page_size=2):
    """A mock transport for the v2 observations endpoint: tag filter, name filter and cursor paging."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/public/v2/observations"
        params = dict(request.url.params)
        calls.append(params)
        wanted = json.loads(params["filter"])[0]["value"][0]
        matching = [r for r in rows if wanted in r["tags"] and r["name"] == params["name"]]
        start = int(params.get("cursor") or 0)
        page = matching[start:start + page_size]
        more = start + page_size < len(matching)
        return httpx.Response(200, json={"data": page, "meta": {"cursor": str(start + page_size) if more else None}})

    return httpx.Client(base_url="https://us.cloud.langfuse.com", transport=httpx.MockTransport(handler)), calls


# ----------------------------------------------------------------------------------------------- dashboard
def test_fetch_reads_every_page_of_the_run_by_its_tag():
    rows = [trace(f"Q{i}", start=f"2026-11-20T10:0{i}:00Z") for i in range(5)]
    other = dict(trace("Z1"), tags=["q:Z1", "run:other"])
    http, calls = api(rows + [other])
    got = td.fetch_traces(http, "r1", now=datetime(2026, 11, 20, 12, tzinfo=UTC))
    assert [r["traceId"] for r in got] == [f"t-Q{i}" for i in range(5)]           # run:other is not read
    assert [c.get("cursor") for c in calls] == [None, "2", "4"]
    first = calls[0]
    assert first["name"] == "finsight.ask" and "trace_context" in first["fields"] and "metadata" in first["fields"]
    assert json.loads(first["filter"]) == [{"type": "arrayOptions", "column": "tags", "operator": "any of",
                                            "value": ["run:r1"]}]
    assert first["fromStartTime"] == "2026-10-21T12:00:00Z" and first["toStartTime"] == "2026-11-20T12:01:00Z"


def test_fetch_keeps_one_row_per_trace_the_one_with_the_span_latency():
    """A v3-SDK trace shows two finsight.ask rows on Langfuse v4: the trace's own row (no latency) and the span."""
    outer = trace("B07", latency=None, trace_id="t-1", start="2026-11-20T10:00:00Z")
    inner = dict(trace("B07", latency=4.5, trace_id="t-1", start="2026-11-20T10:00:00.1Z"), id="o-inner")
    http, _ = api([outer, inner])
    got = td.fetch_traces(http, "r1", now=datetime(2026, 11, 20, 12, tzinfo=UTC))
    assert len(got) == 1 and got[0]["id"] == "o-inner"


def test_fetch_raises_on_an_http_error():
    http = httpx.Client(base_url="https://x", transport=httpx.MockTransport(lambda r: httpx.Response(410, json={})))
    with pytest.raises(httpx.HTTPStatusError):
        td.fetch_traces(http, "r1")


def test_a_row_takes_the_agents_own_numbers():
    r = td.row(trace("B07", visits={"planner": 2, "executor": 2, "composer": 1, "critic": 1}, tool_calls=4, tokens=3536))
    assert r == {"question": "B07", "verdict": "verified", "seconds": 3.2, "steps": 6, "tool_calls": 4,
                 "tokens": 3536, "failure": ""}


@pytest.mark.parametrize(("kwargs", "failure"), [
    ({"verdict": "unverified", "notes": ["could not verify: a number is not in any tool result"]},
     "could not verify: a number is not in any tool result"),
    ({"verdict": "unverified", "stop": "step limit reached (10 tool calls)"}, "step limit reached (10 tool calls)"),
    ({"verdict": "unverified"}, "unverified"),
    ({"verdict": "refused"}, ""),
    ({"verdict": "verified", "stop": "limit reached after the answer was checked: token budget"}, ""),
    ({"verdict": None}, "no verdict (the run raised)"),
])
def test_a_row_names_the_failure(kwargs, failure):
    assert td.row(trace("X", **kwargs))["failure"] == failure


def test_a_row_decodes_metadata_the_sdk_sent_as_json_text():
    """The SDK sends every non-string metadata value as JSON text (langfuse._client.attributes._serialize)."""
    t = trace("B07", verdict="unverified", stop="step limit reached (10 tool calls)", tokens=3536, tool_calls=4,
              visits={"planner": 2, "executor": 2, "composer": 1, "critic": 1})
    t["metadata"] = {k: v if isinstance(v, str) else json.dumps(v) for k, v in t["metadata"].items()}
    assert td.row(t) == {"question": "B07", "verdict": "unverified", "seconds": 3.2, "steps": 6, "tool_calls": 4,
                         "tokens": 3536, "failure": "step limit reached (10 tool calls)"}
    t["metadata"]["stop_reason"], t["metadata"]["notes"] = "null", '["could not verify: x"]'
    assert td.row(t)["failure"] == "could not verify: x"
    t["metadata"]["tokens"], t["metadata"]["seconds"] = "NaN", "Infinity"      # never read as numbers
    assert td.row(t)["tokens"] is None and td.row(t)["seconds"] == 9.9       # falls back to the span latency


def test_a_trace_without_a_summary_still_gives_a_row():
    r = td.row(trace("Q9", verdict=None, metadata=False, latency=12.5))
    assert r["question"] == "Q9" and r["seconds"] == 12.5 and r["steps"] == 0 and r["tokens"] is None
    assert td.render([r], "r1").count("| Q9 | none | 12.5 | 0 | – | – |") == 1


def test_render_gives_the_table_and_totals():
    rows = [td.row(trace("A01", seconds=2.0, tokens=1000)), td.row(trace("A02", seconds=4.0, tokens=3000)),
            td.row(trace("A03", verdict="unverified", seconds=60.0, stop="time limit (60 s)", tokens=2000)),
            td.row(trace("G01", verdict="refused", seconds=1.0, tool_calls=0, tokens=500))]
    text = td.render(rows, "r1")
    assert "| A03 | unverified | 60.0 | 4 | 2 | 2,000 | time limit (60 s) |" in text
    assert "- Verdicts: refused 1, unverified 1, verified 2" in text
    assert "p50 2.0, p95 60.0, max 60.0" in text
    assert "- Mean per question: 4.0 node runs, 1.5 tool calls, 1,625.0 tokens" in text
    assert "- Failures by reason: time limit (60 s) (1)" in text


def test_render_keeps_the_table_intact_with_pipes_and_long_notes():
    r = td.row(trace("A01", verdict="unverified", notes=["a | b " + "x" * 200]))
    line = next(ln for ln in td.render([r], "r1").splitlines() if ln.startswith("| A01"))
    assert line.count("|") == 8 and line.endswith("… |")


def test_dashboard_cli_refuses_when_tracing_is_off(monkeypatch, capsys):
    import core.settings
    monkeypatch.setattr(core.settings, "get_settings", lambda: SimpleNamespace(LANGFUSE_ENABLED=False))
    assert td.main(["--run", "r1"]) == 2
    assert "LANGFUSE_ENABLED is false" in capsys.readouterr().err


def test_dashboard_cli_reads_the_run_and_writes_the_file(monkeypatch, tmp_path, capsys):
    import core.settings
    secret = SimpleNamespace(get_secret_value=lambda: "x")
    monkeypatch.setattr(core.settings, "get_settings", lambda: SimpleNamespace(
        LANGFUSE_ENABLED=True, LANGFUSE_PUBLIC_KEY=secret, LANGFUSE_SECRET_KEY=secret,
        LANGFUSE_HOST="https://us.cloud.langfuse.com"))
    mock, _ = api([trace("A01")])
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(base_url=kw["base_url"], transport=mock._transport))
    out = tmp_path / "r1.md"
    assert td.main(["--run", "r1", "--out", str(out)]) == 0
    assert "| A01 | verified | 3.2 | 4 | 2 | 1,900 |  |" in out.read_text(encoding="utf-8")
    assert td.main(["--run", "missing"]) == 1                                  # nothing tagged run:missing yet
    assert "no traces tagged run:missing yet" in capsys.readouterr().err


# ----------------------------------------------------------------------------------------------- demo script
def test_demo_uses_the_real_planner_executor_and_critic_with_the_composer_stub():
    from agent.stubs import compose_stub
    nodes = demo.demo_nodes(llm=SimpleNamespace())
    assert nodes.composer is compose_stub
    assert all(callable(getattr(nodes, n)) for n in ("planner", "executor", "critic"))


def test_demo_traces_the_run_it_makes():
    from agent.contract import AnswerContract
    from agent.graph import Nodes
    from agent.state import Plan, Verdict
    from core.tracing import TraceInfo
    from tests.test_tracing import RecordingTracer

    async def planner(state):
        return {"plan": Plan(reasoning="x", refuse=False, refusal_rule=None, steps=[])}

    async def executor(state):
        return {"tool_results": []}

    async def composer(state):
        return {"draft": AnswerContract(answer="A draft.", verified=False, refused=False)}

    async def critic(state):
        return {"verdict": Verdict()}

    from agent.graph import Limits
    tracer = RecordingTracer()
    result = asyncio.run(demo.traced_run("q", TraceInfo("DEMO-1", "demo-x"), Nodes(planner, executor, composer, critic),
                                         tracer=tracer, limits=Limits()))      # never the .env (CI has none)
    assert result.answer.answer == "A draft." and tracer.events[0]["info"].run_id == "demo-x"
    assert "verdict:verified" in tracer.tags


def test_demo_cli_refuses_when_tracing_is_off(monkeypatch, capsys):
    import core.settings
    monkeypatch.setattr(core.settings, "get_settings", lambda: SimpleNamespace(LANGFUSE_ENABLED=False))
    assert demo.main(["What was Apple's revenue in fiscal 2024?"]) == 2
    assert "LANGFUSE_ENABLED is false" in capsys.readouterr().err


# ----------------------------------------------------------------------------------------------- the real SDK
def test_with_the_real_langfuse_sdk_a_graph_run_is_one_trace_with_nested_spans(monkeypatch):
    """Only where `langfuse` is installed: spans go to an in-memory OpenTelemetry exporter, Langfuse's own OTLP export
    is stubbed and its client shut down before the test ends (else the SDK retries a flush to the closed port at
    interpreter exit and pytest hangs), so nothing leaves the machine. Checks what the fakes cannot: the SDK accepts
    our calls, and LangGraph's node tasks keep the parent span (one trace, every node under the root, tools and LLM
    calls nested)."""
    pytest.importorskip("langfuse")
    from opentelemetry import trace as otel
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(OTLPSpanExporter, "export", lambda self, spans: SpanExportResult.SUCCESS)

    exporter = InMemorySpanExporter()
    provider = otel.get_tracer_provider()
    if not isinstance(provider, TracerProvider):          # the first test in the process to set one
        provider = TracerProvider()
        otel.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    logging.getLogger("langfuse").setLevel(logging.CRITICAL)
    from langfuse import Langfuse

    from agent import executor as ex
    from agent import graph as g
    from agent.contract import AnswerContract
    from agent.critic import make_critic
    from agent.state import Plan, PlanStep
    from core.tracing import LangfuseTracer, TraceInfo
    from tests.test_llm import client, reply

    # a short flush interval: the SDK's consumer threads wait on it, and shutdown() joins them (3600 s hung pytest)
    lf = Langfuse(public_key="pk-lf-test", secret_key="sk-lf-test", host="http://127.0.0.1:9", flush_interval=0.2)
    llm = client(reply("ok", 12, 3))
    step = PlanStep(tool="get_financial_fact", purpose="x", arguments={
        "ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024, "fiscal_period": "FY", "duration": None})

    async def planner(state):
        await llm.chat([{"role": "user", "content": "plan; mail jane@corp.com"}])
        return {"plan": Plan(reasoning="x", refuse=False, refusal_rule=None, steps=[step])}

    async def tools(name, arguments):
        return {"value": 391035.0, "unit": "USD millions", "ticker": "AAPL", "metric": "revenue",
                "fiscal_year": 2024, "fiscal_period": "FY", "duration": "12M"}

    async def composer(state):
        return {"draft": AnswerContract(answer="A draft.", verified=False, refused=False)}

    graph = g.build_graph(g.Nodes(planner, ex.make_executor(tools, lambda t: len(t.split())), composer, make_critic()))
    asyncio.run(g.run("Apple revenue fiscal 2024? call 415-555-2671", graph=graph, limits=g.Limits(),
                      trace=TraceInfo("A01", "sdk-test"), tracer=LangfuseTracer(lf)))
    provider.force_flush()
    lf.shutdown()                                           # flush (stubbed) and stop: nothing left for atexit
    spans = [s for s in exporter.get_finished_spans()]
    by_id = {s.context.span_id: s for s in spans}
    root = next(s for s in spans if s.name == "finsight.ask")
    assert {s.context.trace_id for s in spans} == {root.context.trace_id}
    parent = {s.name: by_id[s.parent.span_id].name for s in spans if s.parent is not None and s.parent.span_id in by_id}
    assert parent["planner"] == parent["executor"] == parent["critic"] == parent["finish"] == "finsight.ask"
    assert parent["llm"] == "planner" and parent["tool:get_financial_fact"] == "executor"
    assert set(root.attributes["langfuse.trace.tags"]) == {"q:A01", "run:sdk-test", "verdict:unverified"}
    blob = " ".join(str(v) for s in spans for v in s.attributes.values())
    assert "jane@corp.com" not in blob and "415-555-2671" not in blob and "[email]" in blob
