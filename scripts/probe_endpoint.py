"""W1-03: probe what the LLM endpoint in .env supports, and write the evidence for ADR-016 "Endpoint usage".

Run (about 30 requests, ~2 minutes; one at a time, at least MIN_GAP_S apart, except the deliberate concurrent
pair; use it outside the key freeze, as it competes with anyone else on the team key):
    python scripts/probe_endpoint.py                      # all probes -> eval/results/endpoint_probe.{json,md}
    python scripts/probe_endpoint.py --only tools json    # a subset (names: see PROBES)
    python scripts/probe_endpoint.py --skip concurrency   # never provoke a 429

Probes: native tool calling with the four tools.registry schemas (tool_choice auto / required / named, and a
short agent loop that runs every call through the real registry until the model answers), response_format (none / json_object / json_schema with eval's B0Answer,
also against a prose instruction), the thinking switch (on / off / not sent), latency by output length (seconds
vs completion tokens, with a linear fit), limits (max_tokens above 1,024; a prompt over the context), temperature
and seed determinism, and two concurrent requests (the second should get 429 on a one-request key).

Requests go through core.llm's configured client (the group CA, the timeout below), never through its retries
or cache: a failure is measured as it happens. The output never contains the API key, and a remote host is shown
as <endpoint> (scripts/smoke_llm.py's redaction), so the report can be committed and pasted.
Exit codes: 0 = written (a probe that finds a feature missing still counts), 1 = settings missing or the
endpoint unreachable (nothing written).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:               # run as a script from any folder
    sys.path.insert(0, str(ROOT))

import openai  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from core.llm import LLMClient, extract_json, strip_thinking  # noqa: E402
from core.settings import HTTPS, Settings, get_settings  # noqa: E402
from eval.runs_schema import B0Answer  # noqa: E402
from scripts.smoke_llm import RedactHost, redact  # noqa: E402
from tools.registry import SCHEMAS, TOOLS, call_tool  # noqa: E402

PROBE_VERSION = "p-v1"
OUT_DIR = ROOT / "eval" / "results"
OUT_NAME = "endpoint_probe"
MIN_GAP_S = 1.1                 # 60 requests/min on the key: never start two requests closer than this
PROBE_TIMEOUT_S = 120.0         # a 1,024-token reply must not time out (the .env default of 30 s can)
TEXT_KEEP = 400                 # characters of each reply kept in the report
LATENCY_MAX_TOKENS = (16, 128, 512, 1024)
LIMIT_MAX_TOKENS = (1025, 2048)
CONTEXT_OVERSHOOT = 400         # tokens by which the over-context prompt + max_tokens exceed the limit
FILLER = "The company reported its results for the fiscal year in the annual report on Form 10-K. "
TOOLS_SPEC = [{"type": "function", "function": schema} for schema in SCHEMAS]
FACT_QUESTION = "What were Apple's total net sales in fiscal 2024? Use the tools."
GROWTH_QUESTION = "By what percent did net sales grow from 383,285 to 391,035 (USD millions)? Use the tools."
GROWTH_ANSWER = re.compile(r"(?<![\d.])2\.0\d?(?!\d)")     # (391,035 - 383,285) / 383,285 = 2.02%: "2.0" or "2.02"
MAX_TOOL_ROUNDS = 4             # replies in the agent-loop probe: a call, a corrected call after an error, an answer
PASSAGE = "[P1] Apple's total net sales were $391,035 million in fiscal 2024 (year ended September 28, 2024)."
JSON_INSTRUCTION = ("Answer with one JSON object with the fields answer (string), value (number or null), unit, "
                    "company, fiscal_year (integer), citations (list of passage ids), refused (boolean), "
                    "refusal_rule (null).")
ESSAY = "Write a long, detailed essay on the history of financial reporting standards. Do not stop early."
SENTENCE = "Write one original sentence describing a bank's balance sheet."
THINK_QUESTION = "A company's revenue rose from 80 to 100. What is the growth in percent? Answer briefly."


@dataclass
class Outcome:
    """One request as the report keeps it: what was sent (label), what came back, how long it took."""
    label: str
    ok: bool
    seconds: float
    status: int | None = None
    error_type: str | None = None
    error: str | None = None
    retry_after: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    text: str | None = None
    reasoning_chars: int | None = None
    tool_calls: list[dict[str, str]] = field(default_factory=list)


class Session:
    """Sends probe requests one at a time, MIN_GAP_S apart (except `concurrent`), and records each Outcome."""

    def __init__(self, client: Any, settings: Settings) -> None:
        self.client, self.settings, self.last_start = client, settings, 0.0

    def base(self, messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        request = {"model": self.settings.LLM_MODEL, "messages": messages, "temperature": 0.0,
                   "max_tokens": self.settings.LLM_MAX_COMPLETION_TOKENS}
        if self.settings.LLM_ENABLE_THINKING is not None:          # the same switch every app request sends
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.settings.LLM_ENABLE_THINKING}}
        return {**request, **extra}

    async def send(self, label: str, request: dict[str, Any], spaced: bool = True) -> tuple[Outcome, Any]:
        if spaced:
            wait = self.last_start + MIN_GAP_S - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
        self.last_start = time.monotonic()
        start = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(**request)
        except openai.APIStatusError as err:
            headers = getattr(err.response, "headers", {}) or {}
            return Outcome(label, False, round(time.perf_counter() - start, 3), status=err.status_code,
                           error_type=type(err).__name__, error=self.clean(str(err)),
                           retry_after=headers.get("retry-after")), None
        except openai.APIError as err:                               # connection, timeout
            return Outcome(label, False, round(time.perf_counter() - start, 3), error_type=type(err).__name__,
                           error=self.clean(str(err))), None
        return outcome_of(label, response, time.perf_counter() - start), response

    def clean(self, text: str) -> str:
        """An error message fit for the report: no host, no key, not too long."""
        key = self.settings.LLM_API_KEY.get_secret_value()
        text = redact(text, self.settings.LLM_BASE_URL)
        return (text.replace(key, "<key>") if key else text)[:TEXT_KEEP]


def outcome_of(label: str, response: Any, seconds: float) -> Outcome:
    choice = response.choices[0] if response.choices else None
    message = getattr(choice, "message", None)
    usage = getattr(response, "usage", None)
    extra = getattr(message, "model_extra", None) or {}
    reasoning = getattr(message, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning")
    calls = [{"name": c.function.name, "arguments": c.function.arguments}
             for c in (getattr(message, "tool_calls", None) or [])]
    content = getattr(message, "content", None)
    return Outcome(label, True, round(seconds, 3), status=200, finish_reason=getattr(choice, "finish_reason", None),
                   prompt_tokens=getattr(usage, "prompt_tokens", None),
                   completion_tokens=getattr(usage, "completion_tokens", None),
                   text=content[:TEXT_KEEP] if isinstance(content, str) else None,
                   reasoning_chars=len(reasoning) if isinstance(reasoning, str) else None, tool_calls=calls)


# --------------------------------------------------------------------------------------------- checks
def first_error(err: ValidationError) -> str:
    """The first validation problem as "field: message" (pydantic's own first line names only the model)."""
    e = err.errors()[0]
    return f"{'.'.join(str(x) for x in e['loc']) or '(whole reply)'}: {e['msg']}"


def tool_call_check(outcome: Outcome, expected: str | None = None) -> dict[str, Any]:
    """Whether the reply is a native tool call whose arguments pass the tool's own strict model."""
    if not outcome.ok:
        return {"tool_call": False, "why": outcome.error_type}
    if not outcome.tool_calls:
        return {"tool_call": False, "why": "answered in text", "finish_reason": outcome.finish_reason}
    call = outcome.tool_calls[0]
    result: dict[str, Any] = {"tool_call": True, "name": call["name"], "finish_reason": outcome.finish_reason}
    if call["name"] not in TOOLS:
        return {**result, "arguments_valid": False, "why": "unknown tool"}
    try:
        TOOLS[call["name"]][1].model_validate_json(call["arguments"])
        result["arguments_valid"] = True
    except ValidationError as err:
        result.update(arguments_valid=False, why=first_error(err))
    if expected is not None:
        result["expected_tool"] = call["name"] == expected
    return result


def b0_check(outcome: Outcome) -> dict[str, Any]:
    """Whether the reply is valid JSON for eval's strict B0Answer (the same rule the comparison runs used)."""
    if not outcome.ok:
        return {"accepted": False, "valid_b0": False, "why": outcome.error_type, "status": outcome.status}
    try:
        B0Answer.model_validate_json(extract_json(strip_thinking(outcome.text or "")))
        return {"accepted": True, "valid_b0": True}
    except ValidationError as err:
        return {"accepted": True, "valid_b0": False, "why": first_error(err)}


def thinking_check(outcome: Outcome) -> dict[str, Any]:
    text = outcome.text or ""
    return {"ok": outcome.ok, "think_block": "<think>" in text, "reasoning_field_chars": outcome.reasoning_chars,
            "completion_tokens": outcome.completion_tokens, "seconds": outcome.seconds}


def linear_fit(points: list[tuple[float, float]]) -> dict[str, float] | None:
    """Least-squares seconds = overhead + tokens / rate, from (completion tokens, seconds) pairs."""
    if len({x for x, _ in points}) < 2:
        return None
    n = len(points)
    mx, my = sum(x for x, _ in points) / n, sum(y for _, y in points) / n
    slope = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)
    return {"overhead_s": round(my - slope * mx, 2), "tokens_per_s": round(1 / slope, 1) if slope > 0 else 0.0}


def same_text(outcomes: list[Outcome]) -> bool | None:
    texts = [o.text for o in outcomes if o.ok]
    return None if len(texts) < 2 else len(set(texts)) == 1


# --------------------------------------------------------------------------------------------- probes
async def probe_tools(s: Session) -> dict[str, Any]:
    user = [{"role": "user", "content": FACT_QUESTION}]
    plain, _ = await s.send("no tools (baseline for prompt tokens)", s.base(user))
    auto, _ = await s.send("tools, tool_choice=auto", s.base(user, tools=TOOLS_SPEC, tool_choice="auto"))
    required, _ = await s.send("tools, tool_choice=required", s.base(user, tools=TOOLS_SPEC, tool_choice="required"))
    growth = [{"role": "user", "content": GROWTH_QUESTION}]
    named, _ = await s.send("tools, tool_choice=calculate (named)", s.base(
        growth, tools=TOOLS_SPEC, tool_choice={"type": "function", "function": {"name": "calculate"}}))
    outcomes, findings = [plain, auto, required, named], {
        "auto": tool_call_check(auto, "get_financial_fact"), "required": tool_call_check(required),
        "named": tool_call_check(named, "calculate"),
        "schema_prompt_tokens": (auto.prompt_tokens - plain.prompt_tokens)
        if auto.ok and plain.ok and auto.prompt_tokens and plain.prompt_tokens else None}
    loop_outcomes, findings["loop"] = await tool_loop(s, growth)
    return {"outcomes": outcomes + loop_outcomes, "findings": findings}


async def tool_loop(s: Session, messages: list[dict[str, Any]]) -> tuple[list[Outcome], dict[str, Any]]:
    """A minimal agent loop (tool_choice auto): every tool call runs through the real registry and its result goes
    back, for at most MAX_TOOL_ROUNDS replies, until the model answers in text. Shows whether the Week 2 loop
    works on this endpoint and whether the model picks the right operation (growth, not percent)."""
    working, outcomes, rounds = list(messages), [], []
    for n in range(1, MAX_TOOL_ROUNDS + 1):
        outcome, response = await s.send(f"agent loop, reply {n}", s.base(working, tools=TOOLS_SPEC, tool_choice="auto"))
        outcomes.append(outcome)
        if response is None or not outcome.tool_calls:
            answer = outcome.text or ""
            return outcomes, {"rounds": rounds, "replies": n, "answered": outcome.ok and bool(answer.strip()),
                              "answer": answer[:200], "correct": bool(GROWTH_ANSWER.search(answer))}
        calls = response.choices[0].message.tool_calls
        working.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in calls]})
        for c in calls:
            result = call_tool(c.function.name, c.function.arguments)    # the real tool, as the agent will call it
            working.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result)})
            rounds.append({"call": f"{c.function.name}({c.function.arguments})",
                           "result": result.get("error", {}).get("message") if isinstance(result, dict)
                           and "error" in result else (result.get("result") if isinstance(result, dict) else result)})
    return outcomes, {"rounds": rounds, "replies": MAX_TOOL_ROUNDS, "answered": False, "answer": "",
                      "correct": False}


async def probe_json(s: Session) -> dict[str, Any]:
    ask = [{"role": "system", "content": JSON_INSTRUCTION},
           {"role": "user", "content": f"Passages:\n{PASSAGE}\n\nQuestion: What were Apple's total net sales in "
                                       f"fiscal 2024?"}]
    schema = {"type": "json_schema", "json_schema": {"name": "B0Answer", "schema": B0Answer.model_json_schema(),
                                                     "strict": True}}
    prose = [ask[0], {"role": "user", "content": ask[1]["content"] + " Reply in plain prose, not JSON."}]
    runs = [("prompt only (no response_format)", s.base(ask)),
            ("response_format=json_object", s.base(ask, response_format={"type": "json_object"})),
            ("response_format=json_schema (B0Answer, strict)", s.base(ask, response_format=schema)),
            ("json_schema against a prose instruction", s.base(prose, response_format=schema))]
    outcomes = [(await s.send(label, request))[0] for label, request in runs]
    keys = ("prompt_only", "json_object", "json_schema", "json_schema_vs_prose")
    return {"outcomes": outcomes, "findings": {k: b0_check(o) for k, o in zip(keys, outcomes, strict=True)}}


async def probe_thinking(s: Session) -> dict[str, Any]:
    user = [{"role": "user", "content": THINK_QUESTION}]
    plain = {k: v for k, v in s.base(user).items() if k != "extra_body"}
    runs = [("enable_thinking=true", {**plain, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}),
            ("enable_thinking=false", {**plain, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}),
            ("switch not sent", plain)]
    outcomes = [(await s.send(label, request))[0] for label, request in runs]
    checks = {k: thinking_check(o) for k, o in zip(("on", "off", "not_sent"), outcomes, strict=True)}
    return {"outcomes": outcomes, "findings": {**checks, "identical_replies": same_text(outcomes),
                                               "switch_honoured": checks["on"]["think_block"]
                                               or bool(checks["on"]["reasoning_field_chars"])}}


async def probe_latency(s: Session) -> dict[str, Any]:
    user = [{"role": "user", "content": ESSAY}]
    outcomes = [(await s.send(f"max_tokens={n}", s.base(user, max_tokens=n)))[0] for n in LATENCY_MAX_TOKENS]
    points = [(o.completion_tokens, o.seconds) for o in outcomes if o.ok and o.completion_tokens]
    return {"outcomes": outcomes, "findings": {"points": [{"completion_tokens": x, "seconds": y} for x, y in points],
                                               "fit": linear_fit(points)}}


async def probe_limits(s: Session) -> dict[str, Any]:
    user = [{"role": "user", "content": SENTENCE}]
    outcomes = [(await s.send(f"max_tokens={n}", s.base(user, max_tokens=n)))[0] for n in LIMIT_MAX_TOKENS]
    findings: dict[str, Any] = {f"max_tokens_{n}": {"accepted": o.ok, "status": o.status, "error": o.error}
                                for n, o in zip(LIMIT_MAX_TOKENS, outcomes, strict=True)}
    context = s.settings.LLM_CONTEXT_TOKENS
    if context:
        try:
            from core.tokens import count_tokens
            per = count_tokens(FILLER)
        except Exception as err:            # noqa: BLE001 - no tokenizer: say so in the report, do not stop
            findings["over_context"] = {"skipped": f"Qwen tokenizer unavailable ({type(err).__name__})"}
        else:
            target = context - s.settings.LLM_MAX_COMPLETION_TOKENS + CONTEXT_OVERSHOOT
            prompt = FILLER * (target // per + 1)
            o, _ = await s.send(f"prompt of ~{count_tokens(prompt)} tokens + max_tokens "
                                f"{s.settings.LLM_MAX_COMPLETION_TOKENS} (over {context})",
                                s.base([{"role": "user", "content": prompt + "Summarise in one line."}]))
            outcomes.append(o)
            findings["over_context"] = {"prompt_tokens_counted": count_tokens(prompt), "accepted": o.ok,
                                        "status": o.status, "error": o.error}
    return {"outcomes": outcomes, "findings": findings}


async def probe_determinism(s: Session) -> dict[str, Any]:
    user = [{"role": "user", "content": SENTENCE}]
    groups = {"temperature_0": [s.base(user)] * 3,
              "temperature_0.9_seed_1234": [s.base(user, temperature=0.9, seed=1234)] * 2,
              "temperature_0.9_no_seed": [s.base(user, temperature=0.9)] * 2}
    outcomes, findings = [], {}
    for name, requests in groups.items():
        got = [(await s.send(f"{name} #{i + 1}", r))[0] for i, r in enumerate(requests)]
        outcomes += got
        findings[name] = {"identical": same_text(got), "distinct_replies": len({o.text for o in got if o.ok})}
    return {"outcomes": outcomes, "findings": findings}


async def probe_concurrency(s: Session) -> dict[str, Any]:
    request = s.base([{"role": "user", "content": ESSAY}], max_tokens=256)
    await asyncio.sleep(MIN_GAP_S)
    pair = await asyncio.gather(s.send("concurrent request A", request, spaced=False),
                                s.send("concurrent request B", request, spaced=False))
    outcomes = [o for o, _ in pair]
    return {"outcomes": outcomes, "findings": {
        "statuses": [o.status for o in outcomes], "both_ok": all(o.ok for o in outcomes),
        "refused": [{"status": o.status, "error_type": o.error_type, "retry_after": o.retry_after}
                    for o in outcomes if not o.ok]}}


PROBES = {"tools": probe_tools, "json": probe_json, "thinking": probe_thinking, "latency": probe_latency,
          "limits": probe_limits, "determinism": probe_determinism, "concurrency": probe_concurrency}


# --------------------------------------------------------------------------------------------- report
def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def report_meta(settings: Settings) -> dict[str, Any]:
    return {"probe_version": PROBE_VERSION, "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": settings.LLM_MODEL,
            "endpoint": "hosted" if settings.LLM_BASE_URL.lower().startswith(HTTPS) else "local",
            "thinking_setting": settings.LLM_ENABLE_THINKING, "max_tokens_setting": settings.LLM_MAX_COMPLETION_TOKENS,
            "context_setting": settings.LLM_CONTEXT_TOKENS, "min_gap_s": MIN_GAP_S, "timeout_s": PROBE_TIMEOUT_S,
            "git_commit": git_commit()}


def markdown(report: dict[str, Any]) -> str:
    meta = report["meta"]
    out = [f"# Endpoint probe ({meta['probe_version']}, {meta['date']})", "",
           f"Model {meta['model']} on the {meta['endpoint']} endpoint; generated by `scripts/probe_endpoint.py` "
           f"(code {meta['git_commit'] or 'unknown'}). Decisions: ADR-016 in docs/decisions.md.", ""]
    for name, probe in report["probes"].items():
        out += [f"## {name}", "", "| Request | OK | Status | Seconds | Prompt tok | Completion tok | Finish | Reply / "
                "error |", "|---|---|---|---|---|---|---|---|"]
        for o in probe["outcomes"]:
            shown = (o["error"] or "") if not o["ok"] else (
                "tool call " + ", ".join(f"{c['name']}({c['arguments']})" for c in o["tool_calls"])
                if o["tool_calls"] else (o["text"] or ""))
            shown = " ".join(shown.split())[:140].replace("|", "\\|")
            out.append(f"| {o['label']} | {'yes' if o['ok'] else 'no'} | {o['status'] or '-'} | {o['seconds']} | "
                       f"{o['prompt_tokens'] or '-'} | {o['completion_tokens'] or '-'} | {o['finish_reason'] or '-'} | "
                       f"{shown} |")
        out += ["", "```json", json.dumps(probe["findings"], indent=1, ensure_ascii=False), "```", ""]
    return "\n".join(out)


def write_report(report: dict[str, Any], out_dir: Path) -> list[Path]:
    """The JSON and markdown report, each through a temporary file (never half a file)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, text in ((".json", json.dumps(report, indent=1, ensure_ascii=False) + "\n"), (".md", markdown(report))):
        path, tmp = out_dir / f"{OUT_NAME}{suffix}", out_dir / f"{OUT_NAME}{suffix}.part"
        try:
            tmp.write_text(text, encoding="utf-8", newline="\n")
            tmp.replace(path)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        paths.append(path)
    return paths


async def run(client: Any, settings: Settings, names: list[str]) -> dict[str, Any]:
    session = Session(client, settings)
    probes = {}
    for name in names:
        print(f"[{name}] ...", flush=True)
        result = await PROBES[name](session)
        probes[name] = {"outcomes": [asdict(o) for o in result["outcomes"]], "findings": result["findings"]}
        failed = sum(not o.ok for o in result["outcomes"])
        print(f"[{name}] {len(result['outcomes'])} requests, {failed} refused or failed", flush=True)
    return {"meta": report_meta(settings), "probes": probes}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(PROBES), help="run only these probes")
    ap.add_argument("--skip", nargs="+", choices=list(PROBES), default=[], help="leave these probes out")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        settings = get_settings()
    except ValidationError as err:
        print(f"ERROR: settings are incomplete (copy .env.example to .env):\n{err}", file=sys.stderr)
        return 1
    for handler in logging.getLogger().handlers:
        handler.addFilter(RedactHost(settings.LLM_BASE_URL))
    settings = settings.model_copy(update={"LLM_CACHE_DIR": None, "LLM_TIMEOUT": PROBE_TIMEOUT_S})
    names = [n for n in (args.only or list(PROBES)) if n not in args.skip]

    async def go() -> dict[str, Any]:
        async with LLMClient(settings) as llm:            # its configured client: same TLS, no SDK retries
            return await run(llm.client, settings, names)

    report = asyncio.run(go())
    reachable = any(o["ok"] or o["status"] for p in report["probes"].values() for o in p["outcomes"])
    if not reachable:
        print("ERROR: the endpoint answered no request (network, TLS or key); nothing written", file=sys.stderr)
        return 1
    for path in write_report(report, args.out_dir):
        print(f"written: {path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sys.exit(main())
