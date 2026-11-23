"""W3-09: scripts/trace_runs.py. Offline: stub nodes and a recording tracer (no Langfuse, no endpoint)."""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import graph as g
from agent.stubs import stub_nodes

SPEC = importlib.util.spec_from_file_location("trace_runs", Path(__file__).resolve().parents[1] / "scripts" /
                                              "trace_runs.py")
tr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tr)
LIMITS = g.Limits(max_steps=10, max_tokens=12_000, timeout_s=10)


class Recorder:
    """A core.tracing.Tracer that keeps each trace's info (what Langfuse would be sent), nothing else."""
    enabled = True

    def __init__(self):
        self.traces = []

    @contextlib.contextmanager
    def span(self, name, **_):
        yield SimpleNamespace(tag=lambda *a: None, update=lambda **k: None)

    generation = span

    @contextlib.contextmanager
    def trace(self, info, **_):
        self.traces.append(info)
        yield SimpleNamespace(tag=lambda *a: None, update=lambda **k: None)

    def flush(self):
        return None


def info_for(run_id, tags=()):
    from core.tracing import TraceInfo
    return lambda qid, session=None: TraceInfo(question_id=qid, run_id=run_id, session_id=session, tags=tags)


def test_gold_questions_run_with_their_earlier_questions_and_each_is_one_trace():
    items = tr.gold_items(["A01", "F02"])
    assert items[1]["history"] and not items[0]["history"]               # F02 is a follow-up
    rec = Recorder()
    out = asyncio.run(tr.trace_gold(stub_nodes(), LIMITS, info_for("W3-09-test", ("version:A5",)), items, tracer=rec))
    assert [r["id"] for r in out] == ["A01", "F02"] and {"answer", "verified", "seconds", "tokens"} <= set(out[0])
    assert [(t.question_id, t.run_id) for t in rec.traces] == [("A01", "W3-09-test"), ("F02", "W3-09-test")]
    assert "version:A5" in rec.traces[0].all_tags() and "run:W3-09-test" in rec.traces[0].all_tags()


def test_the_demo_queries_run_as_conversations():
    rec = Recorder()
    out = asyncio.run(tr.trace_demo(stub_nodes(), LIMITS, info_for("W3-09-demo"), tracer=rec))
    assert len(out) == 10 and len(rec.traces) == 10
    assert all(t.session_id for t in rec.traces)                          # a follow-up shares its first turn's
    assert len({t.session_id for t in rec.traces}) < 10                    # thread


def test_an_unknown_gold_id_is_refused():
    with pytest.raises(ValueError, match="XX99"):
        tr.gold_items(["A01", "XX99"])


def test_cache_only_never_sends(tmp_path):
    client = SimpleNamespace(_fetch=None)
    tr.cache_only(client)
    with pytest.raises(tr.NotCached):
        asyncio.run(client._fetch({}, "model", None))


def test_nothing_is_traced_when_tracing_is_off(monkeypatch, capsys):
    import core.settings
    monkeypatch.setattr(core.settings, "get_settings", lambda: SimpleNamespace(LANGFUSE_ENABLED=False))
    assert tr.main(["--run-id", "x", "--demo"]) == 2 and "LANGFUSE_ENABLED" in capsys.readouterr().err


def test_cache_only_still_answers_from_the_cache(tmp_path):
    """Review: --cache-only must keep the cache read (core.llm reads the cache before _fetch)."""
    from core.llm import LLMClient
    from core.settings import Settings

    class Chat:
        def __init__(self):
            self.requests = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **request):
            self.requests.append(request)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="cached"),
                                                            finish_reason="stop")],
                                   usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1))

    settings = Settings(_env_file=None, LLM_BASE_URL="http://llm.test/v1", LLM_API_KEY="k", LLM_MODEL="m",
                        SEC_USER_AGENT="HexAgents_Squad test@example.com", LLM_CACHE_DIR=str(tmp_path),
                        LLM_CONTEXT_TOKENS="", LLM_RETRY_BASE_S=0.0, LLM_MIN_INTERVAL_S=0.0)
    messages = [{"role": "user", "content": "q"}]
    asyncio.run(LLMClient(settings, chat_client=Chat()).complete(messages))           # fills the cache
    chat = Chat()
    replay = tr.cache_only(LLMClient(settings, chat_client=chat))
    assert asyncio.run(replay.complete(messages)).text == "cached" and chat.requests == []
    with pytest.raises(tr.NotCached):
        asyncio.run(replay.complete([{"role": "user", "content": "another"}]))
