"""scripts/probe_endpoint.py offline: a fake OpenAI client that answers with real openai response types, so every
probe, check and the report run without the endpoint (and never print the key or the host)."""
import asyncio
import json

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

import scripts.probe_endpoint as pe
from core.settings import Settings

SECRET_HOST, SECRET_KEY = "secret-group-host.example", "sk-team-secret-123"
VALID_B0 = json.dumps({"answer": "Apple's total net sales were $391,035 million.", "value": 391035,
                       "unit": "USD millions", "company": "Apple", "fiscal_year": 2024, "citations": ["P1"],
                       "refused": False, "refusal_rule": None})


@pytest.fixture(autouse=True)
def no_gap(monkeypatch):
    monkeypatch.setattr(pe, "MIN_GAP_S", 0.0)


def settings(**kw):
    env = {"LLM_BASE_URL": f"https://{SECRET_HOST}/v1", "LLM_API_KEY": SECRET_KEY, "LLM_MODEL": "Qwen/Qwen3-8B",
           "SEC_USER_AGENT": "a b@c.d", "LLM_ENABLE_THINKING": False} | kw
    return Settings(_env_file=None, **env)


def completion(content=None, tool_calls=None, finish="stop", prompt=40, completion_tokens=20, reasoning=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                                  "function": {"name": n, "arguments": a}} for i, (n, a) in enumerate(tool_calls)]
    if reasoning:
        message["reasoning_content"] = reasoning
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion_tokens,
                  "total_tokens": prompt + completion_tokens}})


def status_error(cls, code, headers=None):
    request = httpx.Request("POST", f"https://{SECRET_HOST}/v1/chat/completions")
    response = httpx.Response(code, request=request, headers=headers or {})
    return cls(f"Error code: {code} from https://{SECRET_HOST}/v1 with key {SECRET_KEY}", response=response,
               body=None)


class Endpoint:
    """Answers like a vLLM server with tool calling and guided decoding; `busy` refuses overlapping requests."""

    def __init__(self, tools=True, schema=True, busy=True):
        self.tools, self.schema, self.busy, self.active, self.sent = tools, schema, busy, 0, []
        self.chat = self
        self.completions = self

    async def create(self, **request):
        self.sent.append(request)
        if self.busy and self.active:
            raise status_error(openai.RateLimitError, 429, {"retry-after": "1"})
        self.active += 1
        try:
            await asyncio.sleep(0.01)
            return self.answer(request)
        finally:
            self.active -= 1

    def answer(self, r):
        if r["max_tokens"] > 1024:
            raise status_error(openai.BadRequestError, 400)
        if "tools" in r and not self.tools:
            raise status_error(openai.BadRequestError, 400)
        tool_msgs = [m for m in r["messages"] if m["role"] == "tool"]
        if tool_msgs and "invalid_input" in tool_msgs[-1]["content"]:       # corrects itself after an error
            return completion(tool_calls=[("calculate", '{"operation": "growth", "inputs": [383285, 391035]}')],
                              finish="tool_calls", prompt=600)
        if tool_msgs:
            return completion("Net sales grew 2.0%.")
        choice = r.get("tool_choice")
        if choice and choice != "auto" and isinstance(choice, dict):
            return completion(tool_calls=[("calculate", '{"operation": "growth", "inputs": [383285, 391035]}')],
                              finish="tool_calls", prompt=540)
        if choice == "auto" and "grow" in r["messages"][0]["content"]:         # first try: a wrong extra argument
            return completion(tool_calls=[("calculate", '{"operation": "growth", "inputs": [383285, 391035], '
                                                        '"unit": "USD millions"}')], finish="tool_calls", prompt=540)
        if choice in ("auto", "required"):
            args = ('{"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024}' if choice == "auto"
                    else '{"company": "AAPL"}')                                 # required: a wrong argument name
            return completion(tool_calls=[("get_financial_fact", args)], finish="tool_calls", prompt=540)
        fmt = r.get("response_format")
        if fmt and fmt["type"] == "json_schema" and self.schema:
            return completion(VALID_B0)
        if fmt and fmt["type"] == "json_object":
            return completion(VALID_B0.replace('"refused": false, ', ""))           # JSON, not valid B0
        if "Reply in plain prose" in r["messages"][-1]["content"]:
            return completion("Apple's net sales were $391,035 million.")
        if r["messages"][0]["content"] == pe.JSON_INSTRUCTION:
            return completion(VALID_B0)
        n = min(r["max_tokens"], 600)
        return completion("word " * n, finish="length" if n == r["max_tokens"] else "stop",
                          completion_tokens=n, prompt=40 + (500 if "tools" in r else 0))


def probe(name, endpoint=None, **kw):
    ep = endpoint or Endpoint()
    return asyncio.run(pe.PROBES[name](pe.Session(ep, settings(**kw)))), ep


def test_tool_probe_checks_calls_against_the_registry_and_runs_the_round_trip():
    result, ep = probe("tools")
    f = result["findings"]
    assert f["auto"] == {"tool_call": True, "name": "get_financial_fact", "finish_reason": "tool_calls",
                         "arguments_valid": True, "expected_tool": True, "tool_accepts": True}
    assert f["required"]["arguments_valid"] is False and f["required"]["why"].startswith(("company:", "ticker:"))
    assert f["required"]["tool_accepts"] is False and f["named"]["tool_accepts"] is True
    assert f["named"]["arguments_valid"] is True and f["named"]["expected_tool"] is True
    assert f["schema_prompt_tokens"] == 500
    assert f["loop"] == {"rounds": [
        {"call": 'calculate({"operation": "growth", "inputs": [383285, 391035], "unit": "USD millions"})',
         "result": "unit is only for difference and sum; growth returns '%'"},
        {"call": 'calculate({"operation": "growth", "inputs": [383285, 391035]})', "result": 2.0}],
        "replies": 3, "answered": True, "answer": "Net sales grew 2.0%.", "correct": True}
    tool_msg = ep.sent[-1]["messages"][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["result"] == 2.0
    assert ep.sent[1]["tools"] == pe.TOOLS_SPEC and ep.sent[1]["tools"][0]["type"] == "function"


def test_an_endpoint_without_tool_calling_is_reported_not_crashed():
    result, _ = probe("tools", Endpoint(tools=False))
    assert result["findings"]["auto"] == {"tool_call": False, "why": "BadRequestError"}
    assert result["findings"]["loop"]["answered"] is False


def test_json_probe_scores_each_mode_with_the_strict_b0_model():
    f = probe("json")[0]["findings"]
    assert f["prompt_only"] == {"accepted": True, "valid_b0": True}
    assert f["json_object"] == {"accepted": True, "valid_b0": False, "why": "refused: Field required"}
    assert f["json_schema"] == {"accepted": True, "valid_b0": True}
    assert f["json_schema_vs_prose"] == {"accepted": True, "valid_b0": True}
    prose = probe("json", Endpoint(schema=False))[0]["findings"]["json_schema_vs_prose"]
    assert prose["valid_b0"] is False


def test_thinking_probe_sends_on_off_and_no_switch():
    result, ep = probe("thinking")
    switches = [r.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") for r in ep.sent]
    assert switches == [True, False, None] and "extra_body" not in ep.sent[2]
    f = result["findings"]
    assert f["switch_honoured"] is False and f["identical_replies"] is True


def test_a_reasoning_field_or_think_block_counts_as_thinking():
    assert pe.outcome_of("x", completion("ok", reasoning="step 1"), 1.0).reasoning_chars == 6
    assert pe.thinking_check(pe.outcome_of("x", completion("<think>hm</think>20%"), 1.0))["think_block"] is True


def test_latency_fit_recovers_overhead_and_rate():
    assert pe.linear_fit([(0, 0.5), (100, 2.5), (200, 4.5)]) == {"overhead_s": 0.5, "tokens_per_s": 50.0}
    assert pe.linear_fit([(100, 1.0), (100, 1.2)]) is None
    f = probe("latency")[0]["findings"]
    assert [p["completion_tokens"] for p in f["points"]] == [16, 128, 512, 600]


def test_limits_probe_reports_refused_max_tokens():
    f = probe("limits")[0]["findings"]
    assert f["max_tokens_1025"]["accepted"] is False and f["max_tokens_1025"]["status"] == 400
    assert "over_context" not in f                         # tests set no context limit (and load no tokenizer)


def test_determinism_probe_groups_repeats():
    f = probe("determinism")[0]["findings"]
    assert f["temperature_0"] == {"identical": True, "distinct_replies": 1}
    assert set(f) == {"temperature_0", "temperature_0.9_seed_1234", "temperature_0.9_no_seed",
                      "temperature_0.9_seeds_1_2_3"}


def test_concurrency_probe_records_the_429_and_retry_after():
    f = probe("concurrency")[0]["findings"]
    assert sorted(f["statuses"], key=str) == [200, 429] and f["both_ok"] is False
    assert f["refused"] == [{"status": 429, "error_type": "RateLimitError", "retry_after": "1"}]


def test_requests_are_spaced(monkeypatch):
    monkeypatch.setattr(pe, "MIN_GAP_S", 0.05)
    s, starts = pe.Session(Endpoint(), settings()), []
    real = s.client.create

    async def timed(**r):
        starts.append(asyncio.get_running_loop().time())
        return await real(**r)

    s.client.create = timed
    asyncio.run(pe.probe_determinism(s))
    assert all(b - a >= 0.045 for a, b in zip(starts, starts[1:], strict=False))


def test_main_writes_both_reports_without_the_key_or_the_host(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pe, "get_settings", lambda: settings())

    class Ctx:
        def __init__(self, s):
            self.client = Endpoint()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(pe, "LLMClient", Ctx)
    assert pe.main(["--out-dir", str(tmp_path)]) == 0
    texts = [p.read_text(encoding="utf-8") for p in (tmp_path / "endpoint_probe.json", tmp_path / "endpoint_probe.md")]
    texts.append(capsys.readouterr().out)
    for text in texts:
        assert SECRET_KEY not in text and SECRET_HOST not in text
    report = json.loads(texts[0])
    assert set(report["probes"]) == set(pe.PROBES) and report["meta"]["endpoint"] == "hosted"
    assert report["meta"]["script_changed_since_commit"] in (True, False, None)
    refused = report["probes"]["concurrency"]["outcomes"]
    assert any("<endpoint>" in (o["error"] or "") and "<key>" in o["error"] for o in refused)
    assert not list(tmp_path.glob("*.part"))


def test_nothing_is_written_when_the_endpoint_is_unreachable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pe, "get_settings", lambda: settings())

    class Down(Endpoint):
        async def create(self, **request):
            raise openai.APIConnectionError(request=httpx.Request("POST", f"https://{SECRET_HOST}"))

    class Ctx:
        def __init__(self, s):
            self.client = Down()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(pe, "LLMClient", Ctx)
    assert pe.main(["--out-dir", str(tmp_path / "out"), "--only", "thinking"]) == 1
    assert "nothing written" in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_the_loop_stops_after_max_rounds_and_flags_a_wrong_answer():
    class Looping(Endpoint):
        def answer(self, r):
            if r.get("tools"):
                return completion(tool_calls=[("calculate", '{"operation": "percent", "inputs": [383285, 391035]}')],
                                  finish="tool_calls")
            return super().answer(r)

    outs, loop = asyncio.run(pe.tool_loop(pe.Session(Looping(), settings()), [{"role": "user", "content": "q"}]))
    assert len(outs) == pe.MAX_TOOL_ROUNDS and loop["answered"] is False and loop["rounds"][0]["result"] == 98.0
    for right in ("grew 2.02%", "2.0%", "about 2%", "2.0 percent", "2.024%"):
        assert pe.GROWTH_ANSWER.search(right), right
    for wrong in ("12.0%", "98.0%", "2.09%", "2.05%", "grew 2.5%"):
        assert not pe.GROWTH_ANSWER.search(wrong), wrong


# --- added after the review of W1-03 -----------------------------------------------------------------------------

def fake_main(monkeypatch, endpoint):
    monkeypatch.setattr(pe, "get_settings", lambda: settings())

    class Ctx:
        def __init__(self, s):
            self.client = endpoint

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(pe, "LLMClient", Ctx)


def test_a_crashing_probe_keeps_the_others(tmp_path, monkeypatch, capsys):
    fake_main(monkeypatch, Endpoint())

    async def boom(s):
        raise RuntimeError(f"failed at https://{SECRET_HOST}/v1")

    monkeypatch.setitem(pe.PROBES, "json", boom)
    assert pe.main(["--out-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "endpoint_probe.json").read_text(encoding="utf-8"))
    assert report["probes"]["json"] == {"outcomes": [], "findings": {"crashed": "RuntimeError: failed at "
                                                                                  "https://<endpoint>/v1"}}
    assert report["probes"]["tools"]["outcomes"] and SECRET_HOST not in capsys.readouterr().out


def test_a_bad_key_writes_nothing(tmp_path, monkeypatch, capsys):
    class Unauthorised(Endpoint):
        async def create(self, **request):
            raise status_error(openai.AuthenticationError, 401)

    fake_main(monkeypatch, Unauthorised())
    assert pe.main(["--out-dir", str(tmp_path / "out")]) == 1
    assert "nothing written" in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_a_subset_never_overwrites_the_committed_report(monkeypatch, capsys):
    fake_main(monkeypatch, Endpoint())
    assert pe.main(["--only", "thinking"]) == 1
    assert "must not overwrite the full report" in capsys.readouterr().err


def test_the_loop_runs_tools_through_the_async_registry_and_redacts_results(monkeypatch):
    seen = []

    async def fake_tool(name, arguments):
        seen.append(name)
        return {"error": {"type": "internal_error", "message": f"index missing at https://{SECRET_HOST}/x"}}

    monkeypatch.setattr(pe, "call_tool_async", fake_tool)
    _, loop = asyncio.run(pe.tool_loop(pe.Session(Endpoint(), settings()), [{"role": "user", "content": "grow"}]))
    assert seen and loop["rounds"][0]["result"] == "index missing at https://<endpoint>/x"


def test_the_over_context_prompt_is_sized_by_counting_the_prompt(monkeypatch):
    import core.tokens
    monkeypatch.setattr(core.tokens, "count_tokens", lambda text: len(text) // 5)   # merges repeats like Qwen
    result, ep = probe("limits", LLM_CONTEXT_TOKENS=4096)
    f = result["findings"]["over_context"]
    assert f["prompt_tokens_counted"] >= f["target"] == 4096 - 1024 + pe.CONTEXT_OVERSHOOT
    assert len(ep.sent[-1]["messages"][0]["content"]) // 5 >= f["target"]


def test_a_schema_valid_call_the_tool_rejects_is_flagged(monkeypatch):
    # measured: Qwen asked get_financial_fact for metric "net_sales", which passes the schema but not the tool
    async def tool(name, arguments):
        return {"error": {"type": "invalid_input", "message": "unknown metric 'net_sales'"}}

    monkeypatch.setattr(pe, "call_tool_async", tool)
    outcome = pe.Outcome("x", True, 1.0, tool_calls=[{"name": "get_financial_fact", "arguments": "{}"}])
    assert asyncio.run(pe.tool_verdict(pe.Session(Endpoint(), settings()), outcome)) == {
        "tool_accepts": False, "tool_error": "unknown metric 'net_sales'"}
    assert asyncio.run(pe.tool_verdict(pe.Session(Endpoint(), settings()), pe.Outcome("x", True, 1.0))) == {}
