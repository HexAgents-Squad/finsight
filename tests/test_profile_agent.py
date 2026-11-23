"""W3-07: eval/profile_agent.py. Offline: stub nodes and a fake chat client (no data/, no endpoint)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from agent import graph as g
from agent.stubs import stub_nodes
from core import tracing
from core.llm import LLMClient
from core.settings import Settings
from eval import profile_agent as pa
from eval.run_agent import Item

LIMITS = g.Limits(max_steps=10, max_tokens=12_000, timeout_s=10)
ITEMS = [Item("dev-A21", "What were Bank of America's total deposits at the end of fiscal 2023?", ()),
         Item("dev-A22", "What was Wells Fargo's noninterest expense in fiscal 2023?", ())]


def traced_nodes():
    """The stubs, with one LLM call in the planner (as the real planner's, a cache hit of 2.5 s) and one tool call
    in the executor, each in the span the real code opens."""
    stubs = stub_nodes()

    async def planner(state):
        with tracing.generation("llm", model="m") as gen:
            gen.update(metadata={"cached": True, "seconds": 2.5, "cached_tokens": {"input": 100, "output": 10}})
        return await stubs.planner(state)

    async def executor(state):
        with tracing.span("tool:search_filings"):
            pass
        return await stubs.executor(state)

    return g.Nodes(planner=planner, executor=executor, composer=stubs.composer, critic=stubs.critic)


def test_a_replay_profiles_each_node_and_tool():
    done = asyncio.run(pa.replay(ITEMS, traced_nodes(), LIMITS))
    assert [item.id for item, _, _ in done] == ["dev-A21", "dev-A22"]
    profile = done[0][2]
    planner = profile["nodes"]["planner"]
    assert (planner["runs"], planner["llm_calls"], planner["llm"]) == (1, 1, 2.5)
    assert (planner["prompt_tokens"], planner["completion_tokens"], profile["tokens"]) == (100, 10, 110)
    assert profile["llm_s"] == 2.5 and profile["estimated_s"] >= 2.5 and profile["local_s"] >= 0
    assert set(profile["nodes"]) >= {"planner", "composer", "critic", "finish"}
    assert profile["tools"] == {} or set(profile["tools"]) == {"search_filings"}     # a refusal plan skips the executor


def test_the_spans_of_one_question_do_not_leak_into_the_next():
    tracer = pa.ProfilingTracer()
    with tracer.trace(None):
        with tracer.span("planner"), tracer.generation("llm", model="m") as gen:
            gen.update(usage={"input": 5, "output": 1}, metadata={"seconds": 1.0})
    assert pa.question_profile(tracer.spans)["tokens"] == 6
    with tracer.trace(None):
        pass
    assert tracer.spans == []


def test_a_sent_reply_counts_its_usage_and_a_cached_one_its_recorded_tokens():
    sent = pa.SpanRecord("llm", "planner", usage={"input": 7, "output": 3})
    cached = pa.SpanRecord("llm", "composer", metadata={"cached": True, "cached_tokens": {"input": 11, "output": 4}})
    assert (pa.llm_tokens(sent), pa.llm_tokens(cached), pa.llm_tokens(pa.SpanRecord("llm", None))) == ((7, 3), (11, 4),
                                                                                                      (0, 0))


class FakeChat:
    def __init__(self, text):
        self.text, self.requests = text, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text),
                                                        finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=9, completion_tokens=2))


def llm_settings(tmp_path) -> Settings:
    return Settings(_env_file=None, LLM_BASE_URL="http://llm.test/v1", LLM_API_KEY="key", LLM_MODEL="m",
                    SEC_USER_AGENT="HexAgents_Squad test@example.com", LLM_CACHE_DIR=str(tmp_path),
                    LLM_CONTEXT_TOKENS="", LLM_RETRY_BASE_S=0.0, LLM_MIN_INTERVAL_S=0.0)


def test_cache_only_answers_from_the_cache_and_never_sends(tmp_path):
    messages = [{"role": "user", "content": "hi"}]
    live = LLMClient(llm_settings(tmp_path), chat_client=FakeChat("cached answer"))
    assert asyncio.run(live.complete(messages)).text == "cached answer"           # the run that filled the cache
    replay_chat = FakeChat("never")
    replayed = pa.cache_only(LLMClient(llm_settings(tmp_path), chat_client=replay_chat))
    assert asyncio.run(replayed.complete(messages)).text == "cached answer"
    with pytest.raises(pa.CacheMiss):
        asyncio.run(replayed.complete([{"role": "user", "content": "another question"}]))
    assert replay_chat.requests == [] and replayed.usage.calls == 0


def runs_file(tmp_path, answers, latency=(4.0, 2.0)):
    path = tmp_path / "A5_qwen3-8b_agent_think-off_dev-test.jsonl"
    lines = [json.dumps({"id": item.id, "run_id": "A5_qwen3-8b_agent_think-off_dev-test", "latency_s": lat,
                         "prompt_tokens": 3000, "completion_tokens": 100, "error": None,
                         "raw_reply": json.dumps({"answer": answer, "verified": verified}),
                         "agent": {"llm_calls": 2, "tool_steps": 1}})
             for item, (answer, verified), lat in zip(ITEMS, answers, latency, strict=True)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_runs_only_reports_the_runs_files_figures(tmp_path):
    path = runs_file(tmp_path, [("a", True), ("b", True)])
    assert pa.main([str(path), "--runs-only", "--out-dir", str(tmp_path / "out")]) == 0
    md = (tmp_path / "out" / "A5_qwen3-8b_agent_think-off_dev-test.md").read_text(encoding="utf-8")
    assert "| A5_qwen3-8b_agent_think-off_dev-test | 2 | 3.00 | 4.00 | 3100 | 3100 | 2.00 | 1.00 |" in md
    assert "Where the time goes" not in md and not list((tmp_path / "out").glob("*.jsonl"))


def test_a_replay_that_answers_differently_writes_nothing(tmp_path, capsys):
    path = runs_file(tmp_path, [("another answer", True), ("another answer", True)])
    out = tmp_path / "out"
    assert pa.main([str(path), "--out-dir", str(out)], nodes=traced_nodes(), limits=LIMITS) == 2
    assert not out.exists() and "answered dev-A21, dev-A22 differently" in capsys.readouterr().err


def test_a_replay_of_the_same_answers_writes_the_report_and_its_lines(tmp_path):
    done = asyncio.run(pa.replay(ITEMS, traced_nodes(), LIMITS))
    path = runs_file(tmp_path, [(r.answer.answer, r.answer.verified) for _, r, _ in done])
    out = tmp_path / "out"
    assert pa.main([str(path), "--label", "after", "--out-dir", str(out)], nodes=traced_nodes(), limits=LIMITS) == 0
    md = (out / "A5_qwen3-8b_agent_think-off_dev-test-after.md").read_text(encoding="utf-8")
    lines = [json.loads(x) for x in (out / "A5_qwen3-8b_agent_think-off_dev-test-after.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert [p["id"] for p in lines] == ["dev-A21", "dev-A22"] and "| planner | 1.00 | 1.00 | 2.50 |" in md
    assert "Tokens per question p50 110" in md
    rebuilt = out / "rebuilt"                          # --profiles: the same report from the saved lines
    assert pa.main([str(path), "--label", "after", "--out-dir", str(rebuilt), "--profiles",
                    str(out / "A5_qwen3-8b_agent_think-off_dev-test-after.jsonl")]) == 0
    assert (rebuilt / "A5_qwen3-8b_agent_think-off_dev-test-after.md").read_text(encoding="utf-8") == md


def test_quantiles_use_the_nearest_rank():
    assert (pa.pct([], 0.95), pa.pct([1.0], 0.95), pa.pct([1.0, 2.0, 3.0, 4.0], 0.95)) == (0.0, 1.0, 4.0)


def test_a_question_the_run_recorded_as_an_error_is_left_out(tmp_path, capsys):
    done = asyncio.run(pa.replay(ITEMS[:1], traced_nodes(), LIMITS))
    path = runs_file(tmp_path, [(done[0][1].answer.answer, done[0][1].answer.verified), ("x", True)])
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    lines[1].update(error="RuntimeError: a node bug", raw_reply=None)
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    assert pa.main([str(path), "--out-dir", str(out)], nodes=traced_nodes(), limits=LIMITS) == 0
    profiled = [json.loads(x)["id"] for x in next(out.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert profiled == ["dev-A21"] and "left out (an error in the runs file): dev-A22" in capsys.readouterr().err
