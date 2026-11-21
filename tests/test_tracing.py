"""W2-08: Langfuse tracing. Offline: a recording fake tracer and a fake Langfuse client (the `langfuse` package is not
imported, and nothing here reaches the network). Tracing is off unless a test turns it on."""
from __future__ import annotations

import asyncio
import contextlib
import sys
from types import SimpleNamespace

import pytest

from core import tracing as tr
from core.tracing import (
    NOOP_TRACER,
    LangfuseTracer,
    TraceInfo,
    generation,
    make_tracer,
    scrub,
    span,
    trace_run,
)

INFO = TraceInfo(question_id="B07", run_id="r1", session_id="s1")


# ----------------------------------------------------------------------------------------------- fakes
class RecordingTracer:
    """A Tracer that records the tree of spans: (kind, name, input, updates), in the order they were opened."""
    enabled = True

    def __init__(self):
        self.events: list[dict] = []
        self.stack: list[dict] = []
        self.tags: list[str] = []

    @contextlib.contextmanager
    def _open(self, kind, name, **fields):
        event = {"kind": kind, "name": name, "depth": len(self.stack), "updates": [], "error": None, **fields}
        self.events.append(event)
        self.stack.append(event)
        handle = SimpleNamespace(update=lambda **kw: event["updates"].append(kw),
                                 tag=lambda *t: self.tags.extend(t))
        try:
            yield handle
        except BaseException as err:
            event["error"] = type(err).__name__
            raise
        finally:
            self.stack.pop()

    def span(self, name, *, input=None, metadata=None):
        return self._open("span", name, input=input, metadata=metadata)

    def generation(self, name, *, model, input=None, metadata=None):
        return self._open("generation", name, model=model, input=input, metadata=metadata)

    def trace(self, info, *, input=None):
        return self._open("trace", "finsight.ask", info=info, input=input)

    def flush(self):
        pass

    def names(self):
        return [e["name"] for e in self.events]


class FakeLangfuse:
    """The five calls LangfuseTracer makes on a langfuse.Langfuse client."""

    def __init__(self, fail_on=()):
        self.calls, self.fail_on, self.exits, self.raws = [], set(fail_on), [], []

    def _cm(self, kind, **kw):
        if kind in self.fail_on:
            raise RuntimeError("langfuse is down")
        self.calls.append((kind, kw))
        fake = self
        raw = SimpleNamespace(updates=[], update=lambda **u: raw.updates.append(u))
        self.raws.append(raw)

        class CM:
            def __enter__(self):
                return raw

            def __exit__(self, *exc):
                fake.exits.append(exc[0])

        return CM()

    def start_as_current_span(self, **kw):
        return self._cm("span", **kw)

    def start_as_current_generation(self, **kw):
        return self._cm("generation", **kw)

    def update_current_trace(self, **kw):
        if "trace" in self.fail_on:
            raise RuntimeError("langfuse is down")
        self.calls.append(("trace", kw))

    def flush(self):
        self.calls.append(("flush", {}))


# ----------------------------------------------------------------------------------------------- off by default
def test_no_tracer_means_no_op_and_no_settings_read():
    """Calls outside a traced run do nothing, and run(trace=None) never reads the settings (no .env in CI)."""
    with span("x", input={"a": 1}) as sp, generation("y", model="m") as gen:
        sp.update(output=1)
        gen.update(output=2, usage={"input": 1, "output": 1})
    with trace_run(None, "a question") as root:
        root.tag("verdict:verified")
        root.update(output="x")


def test_langfuse_is_off_by_default_in_settings():
    from core.settings import Settings
    s = Settings(_env_file=None, LLM_BASE_URL="http://x/v1", LLM_API_KEY="k", LLM_MODEL="m", SEC_USER_AGENT="t a@b.co")
    assert s.LANGFUSE_ENABLED is False and s.LANGFUSE_PUBLIC_KEY is None and s.LANGFUSE_SECRET_KEY is None
    assert make_tracer(s) is NOOP_TRACER


def test_make_tracer_off_does_not_import_langfuse(monkeypatch):
    monkeypatch.setitem(sys.modules, "langfuse", None)         # an import would raise ImportError
    assert make_tracer(SimpleNamespace(LANGFUSE_ENABLED=False)) is NOOP_TRACER


def test_make_tracer_on_without_the_package_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "langfuse", None)
    with pytest.raises(ImportError, match="langfuse"):
        make_tracer(SimpleNamespace(LANGFUSE_ENABLED=True))


def test_enabled_needs_both_keys_and_the_message_hides_them():
    from pydantic import ValidationError

    from core.settings import Settings
    base = {"LLM_BASE_URL": "http://x/v1", "LLM_API_KEY": "k", "LLM_MODEL": "m", "SEC_USER_AGENT": "t a@b.co"}
    with pytest.raises(ValidationError, match="LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY") as err:
        Settings(_env_file=None, LANGFUSE_ENABLED=True, LANGFUSE_PUBLIC_KEY="pk-lf-visible", **base)
    assert "pk-lf-visible" not in str(err.value)
    with pytest.raises(ValidationError, match="not both set") as err:          # only the secret: never quoted
        Settings(_env_file=None, LANGFUSE_ENABLED=True, LANGFUSE_SECRET_KEY="sk-lf-q7Zx9", **base)
    assert "q7Zx9" not in str(err.value) and "q7Zx9" not in repr(err.value)
    keys = {"LANGFUSE_PUBLIC_KEY": "pk-lf-1", "LANGFUSE_SECRET_KEY": "sk-lf-2"}
    with pytest.raises(ValidationError, match="https://") as err:
        Settings(_env_file=None, LANGFUSE_ENABLED=True, LANGFUSE_HOST="cloud.langfuse.com", **keys, **base)
    assert "sk-lf-2" not in str(err.value)
    local = Settings(_env_file=None, LANGFUSE_ENABLED=True, LANGFUSE_HOST="http://localhost:3000/", **keys, **base)
    assert local.LANGFUSE_HOST == "http://localhost:3000"
    off = Settings(_env_file=None, LANGFUSE_HOST="", **base)                      # `LANGFUSE_HOST=` while off: fine
    assert off.LANGFUSE_HOST == "https://cloud.langfuse.com"
    ok = Settings(_env_file=None, LANGFUSE_ENABLED=True, LANGFUSE_PUBLIC_KEY="pk-lf-1", LANGFUSE_SECRET_KEY="sk-lf-2", **base)
    assert "sk-lf-2" not in repr(ok) and "pk-lf-1" not in repr(ok)
    empty = Settings(_env_file=None, LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY=" ", **base)    # `KEY=` in .env = unset
    assert empty.LANGFUSE_SECRET_KEY is None


# ----------------------------------------------------------------------------------------------- scrubbing
def test_scrub_masks_emails_and_phones_but_not_financial_figures():
    text = ("mail jane.doe+x@corp.example.com, call +91 98765 43210, (415) 555-2671 or 415-555-2671; "
            "revenue 383285000000 on 2024-09-28, accession 0000320193-24-000123, FY 2023 2024 2025, 109.4 / 245.1 = 44.6%")
    out = scrub(text)
    assert out.count("[email]") == 1 and out.count("[phone]") == 3
    for kept in ("383285000000", "2024-09-28", "0000320193-24-000123", "2023 2024 2025", "109.4 / 245.1 = 44.6%"):
        assert kept in out


@pytest.mark.parametrize("phone", ["415-555-2671", "415.555.2671", "(415) 555-2671", "(415)555-2671", "1-800-555-1234",
                                   "1 800 555 1234", "1 (800) 555-1234", "+1 415 555 2671", "+1-415-555-2671",
                                   "+91 98765 43210", "+44 20 7946 0958", "+33 1 23 45 67 89", "+1 (415) 555-2671",
                                   # compact E.164 (W2-13); a signed figure of 11+ digits without separators reads
                                   # the same and is masked too: filings and the tools write figures with commas
                                   "+14155552671", "+383285000000", "+93736000000"])
def test_scrub_masks_phone_shapes(phone):
    assert scrub(f"call {phone} today") == "call [phone] today"


@pytest.mark.parametrize("text", [
    "4155552671", "(123) 456 789", "(35) 133 148", "change +12 345 678", "+5 2023 2024", "growth +1.25 34.6",
    "| 164,501 | 134,902 |", "2023-2025", "FY 2023 2024 2025", "0000320193-24-000123", "2024-09-28", "EIN 94-2404110",
    "File No. 001-36743", "CUSIP 037833100", "revenue 1,234.5 567.8 9,012.3", "123 456 7890.5",
    "net income 383285000000", "+9373600000", "+3.5 12.25 2024 2023", "shares 391 035 2024",
    "Net sales 245 122 2023 2022", "segments 100 200 3000 4000", "123\n456\n7890", "ratio 123-456-7890.12",
    "415 555 2671",                     # bare spaces: indistinguishable from a row of figures, so left alone
])
def test_scrub_never_masks_financial_text(text):
    assert scrub(text) == text


def test_scrub_cuts_long_texts_and_big_collections_and_handles_models():
    assert len(scrub("x" * 10_000)) < tr.MAX_TEXT_CHARS + 60
    assert len(scrub(list(range(500)))) == tr.MAX_ITEMS + 1
    model = SimpleNamespace(model_dump=lambda mode: {"who": "a@b.co"})
    assert scrub({"m": model}) == {"m": {"who": "[email]"}}
    assert scrub({"n": None, "b": True, "f": 1.5, "o": object})["n"] is None


# ----------------------------------------------------------------------------------------------- the Langfuse adapter
def test_adapter_opens_nested_spans_a_generation_and_tags_the_trace():
    client = FakeLangfuse()
    tracer = LangfuseTracer(client)
    with trace_run(INFO, "Mail me at a@b.co: Apple FY2024 revenue?", tracer=tracer) as root:
        with span("planner", input={"question": "q"}) as sp:
            with generation("llm", model="Qwen/Qwen3-8B", input=[{"role": "user", "content": "hi"}]) as gen:
                gen.update(output="ok", usage={"input": 7, "output": 3}, metadata={"cached": False})
            sp.update(output={"refuse": False})
        root.update(output="answer", metadata={"tokens": 10})
        root.tag("verdict:verified")
    kinds = [c[0] for c in client.calls]
    # root, its tags, planner, llm, the root's summary copied to the trace, the verdict tag
    assert kinds == ["span", "trace", "span", "generation", "trace", "trace"]
    assert client.calls[0][1]["input"].startswith("Mail me at [email]")        # scrubbed before it leaves the process
    first_trace, summary, last_trace = client.calls[1][1], client.calls[4][1], client.calls[5][1]
    assert first_trace["tags"] == ["q:B07", "run:r1"] and first_trace["session_id"] == "s1"
    assert summary == {"output": "answer", "metadata": {"question_id": "B07", "run_id": "r1", "tokens": 10}}
    assert last_trace["tags"] == ["q:B07", "run:r1", "verdict:verified"]
    assert client.calls[3][1]["model"] == "Qwen/Qwen3-8B"
    assert client.exits == [None, None, None]                                  # all three contexts closed cleanly


def test_adapter_marks_a_failed_block_and_still_closes_the_span():
    client = FakeLangfuse()
    with pytest.raises(ValueError):
        with trace_run(INFO, "q", tracer=LangfuseTracer(client)):
            with span("executor"):
                raise ValueError("no filing for jane@corp.example, call 415-555-2671")
    failed = {"level": "ERROR", "status_message": "ValueError: no filing for [email], call [phone]"}
    assert all(failed in raw.updates for raw in client.raws)                   # both spans failed, the text masked
    assert client.exits == [None, None]       # closed without the exception: the SDK would record its raw text


@pytest.mark.parametrize("fail_on", [{"span"}, {"generation"}, {"trace"}])
def test_a_langfuse_failure_never_changes_the_run(fail_on):
    client = FakeLangfuse(fail_on=fail_on)
    ran = []
    with trace_run(INFO, "q", tracer=LangfuseTracer(client)) as root:
        with span("planner") as sp, generation("llm", model="m") as gen:
            ran.append("body")
            sp.update(output=1)
            gen.update(output="x", usage={"input": 1, "output": 1})
        root.update(output="a")
        root.tag("verdict:verified")
    assert ran == ["body"]


def test_a_block_cancelled_by_the_time_limit_is_marked_as_an_error():
    client = FakeLangfuse()

    async def go():
        with trace_run(INFO, "q", tracer=LangfuseTracer(client)):
            with span("executor"):
                await asyncio.sleep(10)

    async def main():
        await asyncio.wait_for(go(), timeout=0.05)

    with pytest.raises(TimeoutError):
        asyncio.run(main())
    executor = client.raws[1]                                                 # the root is raws[0]
    assert {"level": "ERROR", "status_message": "cancelled"} in executor.updates
    assert client.exits[0] is None


def test_a_generation_uses_the_current_sdk_call_when_the_client_has_it():
    client = FakeLangfuse()
    client.start_as_current_observation = lambda **kw: client._cm("observation", **kw)
    with trace_run(INFO, "q", tracer=LangfuseTracer(client)), generation("llm", model="Qwen/Qwen3-8B"):
        pass
    kinds = [(c[0], c[1].get("as_type")) for c in client.calls]
    assert ("observation", "generation") in kinds and not any(k == "generation" for k, _ in kinds)


def test_a_cached_reply_reports_no_usage(tmp_path):
    from tests.test_llm import client, reply
    llm = client(reply("hello", 11, 4), LLM_CACHE_DIR=str(tmp_path))
    tracer = RecordingTracer()

    async def go():
        with trace_run(INFO, "q", tracer=tracer):
            await llm.chat([{"role": "user", "content": "Reply with: hello"}])
            await llm.chat([{"role": "user", "content": "Reply with: hello"}])     # the same request: from the cache

    asyncio.run(go())
    first, second = [e["updates"][0] for e in tracer.events if e["kind"] == "generation"]
    assert first["usage"] == {"input": 11, "output": 4} and first["metadata"]["cached"] is False
    assert second["usage"] is None and second["metadata"]["cached"] is True
    assert second["metadata"]["cached_tokens"] == {"input": 11, "output": 4}


def test_after_a_traced_run_nothing_more_is_traced():
    """The tracer is reset when the run ends, also when it raises: later work in the same context is not traced."""
    tracer = RecordingTracer()
    with trace_run(INFO, "q", tracer=tracer):
        pass
    with pytest.raises(RuntimeError), trace_run(INFO, "q", tracer=tracer):
        raise RuntimeError("boom")
    before = len(tracer.events)
    with span("later"), generation("llm", model="m"):
        pass
    assert len(tracer.events) == before == 2 and tr._current.get() is NOOP_TRACER


def test_flush_reaches_the_client():
    client = FakeLangfuse()
    LangfuseTracer(client).flush()
    assert client.calls == [("flush", {})]


def test_trace_info_needs_ids_and_dedupes_tags():
    with pytest.raises(ValueError):
        TraceInfo(question_id=" ", run_id="r")
    assert TraceInfo("B07", "r1", tags=("q:B07", "dev")).all_tags("verdict:verified") == [
        "q:B07", "run:r1", "dev", "verdict:verified"]


# ----------------------------------------------------------------------------------------------- the wiring
def test_graph_run_traces_every_node_and_tags_the_verdict():
    from agent import graph as g
    from agent.stubs import stub_nodes
    tracer = RecordingTracer()
    limits = g.Limits(max_steps=10, max_tokens=12_000, timeout_s=5)
    result = asyncio.run(g.run("What was Apple's total revenue in fiscal 2024?", graph=g.build_graph(stub_nodes()),
                               limits=limits, trace=INFO, tracer=tracer))
    assert tracer.events[0]["kind"] == "trace" and tracer.events[0]["info"] == INFO
    seen = list(dict.fromkeys(n for n in tracer.names() if n != "finsight.ask"))    # first visit of each node
    assert seen == ["planner", "executor", "composer", "critic", "finish"]
    assert all(e["depth"] == 1 for e in tracer.events[1:])                  # every node sits under the root
    root = tracer.events[0]
    summary = root["updates"][-1]["metadata"]
    assert summary["verified"] == result.answer.verified and summary["visits"]["planner"] == 1
    assert tracer.tags == ["verdict:" + ("verified" if result.answer.verified else "unverified")]
    assert not any(e["error"] for e in tracer.events)


def test_graph_run_without_trace_touches_no_tracer():
    from agent import graph as g
    from agent.stubs import stub_nodes
    tracer = RecordingTracer()                                              # given, but no TraceInfo: not used
    limits = g.Limits(max_steps=10, max_tokens=12_000, timeout_s=5)
    asyncio.run(g.run("What was Apple's total revenue in fiscal 2024?", graph=g.build_graph(stub_nodes()),
                      limits=limits, tracer=tracer))
    assert tracer.events == []


def test_executor_records_a_span_per_tool_call_with_its_error_type():
    from agent import executor as ex
    from agent.state import ToolCall

    async def fake_tools(name, arguments):
        if name == "get_financial_fact":
            return {"error": {"type": "not_reported", "message": "Q4 is not reported"}}
        if name == "compare_periods":
            return {"error": {"type": "data_missing", "message": "no facts table"}}
        return {"results": []}

    tracer = RecordingTracer()

    async def go():
        with trace_run(INFO, "q", tracer=tracer):
            await ex._run(fake_tools, ToolCall(name="search_filings", arguments={"query": "revenue"}))
            await ex._run(fake_tools, ToolCall(name="get_financial_fact", arguments={"ticker": "AAPL"}))
            await ex._run(fake_tools, ToolCall(name="compare_periods", arguments={"ticker": "AAPL"}))

    asyncio.run(go())
    tools = [e for e in tracer.events if e["name"].startswith("tool:")]
    assert [e["name"] for e in tools] == ["tool:search_filings", "tool:get_financial_fact", "tool:compare_periods"]
    assert tools[0]["updates"][0]["metadata"] == {"error_type": None} and tools[0]["updates"][0]["level"] is None
    # not_reported is an answer (ADR-028): recorded, but not a WARNING; a typed error is
    assert tools[1]["updates"][0]["metadata"] == {"error_type": "not_reported"}
    assert tools[1]["updates"][0]["level"] is None
    assert tools[2]["updates"][0]["metadata"] == {"error_type": "data_missing"}
    assert tools[2]["updates"][0]["level"] == "WARNING"


def test_llm_client_records_a_generation_with_tokens_and_cache_flag():
    from tests.test_llm import client, reply
    llm = client(reply("hello", 11, 4))
    tracer = RecordingTracer()

    async def go():
        with trace_run(INFO, "q", tracer=tracer):
            return await llm.chat([{"role": "user", "content": "Reply with: hello"}])

    assert asyncio.run(go()) == "hello"
    gen = next(e for e in tracer.events if e["kind"] == "generation")
    assert gen["model"] == "chat-model" and gen["name"] == "llm"
    update = gen["updates"][0]
    assert update["usage"] == {"input": 11, "output": 4} and update["output"] == "hello"
    assert update["metadata"]["cached"] is False and update["metadata"]["retries"] == 0
