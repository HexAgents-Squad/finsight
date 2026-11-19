"""W2-08: the trace dashboard and the demo script (offline: fake Langfuse API objects, no network, no .env), plus one
check against the real Langfuse SDK that runs only where `langfuse` is installed (CI does not install it)."""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

import scripts.trace_dashboard as td
import scripts.trace_demo as demo


def trace(qid, verdict="verified", seconds=3.2, visits=None, tool_calls=2, tokens=1900, stop=None, notes=(),
          latency=9.9, metadata=True):
    meta = {"question_id": qid, "run_id": "r1", "seconds": seconds, "tokens": tokens, "tool_calls": tool_calls,
            "visits": visits or {"planner": 1, "executor": 1, "composer": 1, "critic": 1}, "verified": verdict == "verified",
            "refused": verdict == "refused", "stop_reason": stop, "notes": list(notes)}
    tags = [f"q:{qid}", "run:r1"] + ([f"verdict:{verdict}"] if verdict else [])
    return SimpleNamespace(id="t-" + qid, tags=tags, latency=latency, metadata=meta if metadata else None)


class FakeTraceAPI:
    def __init__(self, traces, page_size):
        self.traces, self.page_size, self.calls = traces, page_size, []

    def list(self, **kw):
        self.calls.append(kw)
        start = (kw["page"] - 1) * kw["limit"]
        pages = max(1, -(-len(self.traces) // kw["limit"]))
        return SimpleNamespace(data=self.traces[start:start + kw["limit"]], meta=SimpleNamespace(total_pages=pages))


def fake_client(traces, page_size=2):
    api = FakeTraceAPI(traces, page_size)
    return SimpleNamespace(api=SimpleNamespace(trace=api)), api


# ----------------------------------------------------------------------------------------------- dashboard
def test_fetch_reads_every_page_of_the_run_by_its_tag():
    traces = [trace(f"Q{i}") for i in range(5)]
    client, api = fake_client(traces)
    got = td.fetch_traces(client, "r1", page_size=2)
    assert [t.id for t in got] == [t.id for t in traces]
    assert [c["page"] for c in api.calls] == [1, 2, 3]
    assert all(c["tags"] == ["run:r1"] and c["order_by"] == "timestamp.asc" for c in api.calls)


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
    import json
    t = trace("B07", verdict="unverified", stop="step limit reached (10 tool calls)", tokens=3536, tool_calls=4,
              visits={"planner": 2, "executor": 2, "composer": 1, "critic": 1})
    t.metadata = {k: v if isinstance(v, str) else json.dumps(v) for k, v in t.metadata.items()}
    assert td.row(t) == {"question": "B07", "verdict": "unverified", "seconds": 3.2, "steps": 6, "tool_calls": 4,
                         "tokens": 3536, "failure": "step limit reached (10 tool calls)"}
    t.metadata["stop_reason"], t.metadata["notes"] = "null", '["could not verify: x"]'
    assert td.row(t)["failure"] == "could not verify: x"


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
def test_with_the_real_langfuse_sdk_a_graph_run_is_one_trace_with_nested_spans():
    """Only where `langfuse` is installed: spans go to an in-memory OpenTelemetry exporter and the client points at a
    closed local port, so nothing leaves the machine. Checks what the fakes cannot: the SDK accepts our calls, and
    LangGraph's node tasks keep the parent span (one trace, every node under the root, tools and LLM calls nested)."""
    pytest.importorskip("langfuse")
    from opentelemetry import trace as otel
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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

    lf = Langfuse(public_key="pk-lf-test", secret_key="sk-lf-test", host="http://127.0.0.1:9", flush_at=1000,
                  flush_interval=3600)
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
