"""W2-03: native tool calling on the development questions, measured before and after a tool-schema change.

    python scripts/probe_tool_calls.py --label before      # with the schemas as they are
    python scripts/probe_tool_calls.py --label after       # after editing tools/

Two measures, both with tool_choice auto (the agent's setting, ADR-016) and the tools.registry schemas:
- first-call acceptance: each answerable development question (eval/dev_set.jsonl and eval/planner_dev.jsonl,
  never gold-v1 or the refusal tests) is asked once; the reply's first tool call is accepted when the tool takes its
  arguments: get_financial_fact, compare_periods and calculate run for real on the local XBRL data (an unknown
  metric or a unit calculate does not take is invalid_input, as in the agent), search_filings' arguments are
  validated against its model (a search would load the index). A text answer without a tool call is not accepted.
- the right call: the first call's tool and arguments (company, metric, year, period) against the first step of the planner's committed
  plan for the question (eval/results/planner_runs/P8.jsonl, every plan reviewed as right). Acceptance alone cannot
  see a valid call for the wrong thing, e.g. a metric substituted for a line item that is not one of the 13.
- the calculator: each of those questions whose wording asks for a calculation (agent/critic_checks
  .expected_operations: a margin, a change in percent, ...) is replayed with the figures already fetched (the fact
  steps of the planner's committed run, executed on the local data); the next call should be calculate with an
  expected operation and accepted arguments.

    python scripts/probe_tool_calls.py --rescore after --label after-lenient-unit   # no requests

--rescore re-checks a recorded run's calls against the tools as they are now (the model's calls unchanged, no
request sent): it measures a change to what the tools accept, e.g. calculate ignoring a unit it does not use.
Writes eval/results/tool_calls/<label>.json and eval/results/tool_calls.md (every label present, side by side).
Requests go one at a time through scripts/probe_endpoint.Session (spacing, the key and host kept out of reports),
uncached, so a run measures the schemas as they are.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from agent.critic_checks import expected_operations  # noqa: E402
from eval.run_planner import load_questions  # noqa: E402
from scripts.probe_endpoint import PROBE_TIMEOUT_S, Session, first_error  # noqa: E402

OUT_DIR = ROOT / "eval" / "results" / "tool_calls"
SUMMARY = ROOT / "eval" / "results" / "tool_calls.md"
PLANNER_RUN = ROOT / "eval" / "results" / "planner_runs" / "P8.jsonl"
EXECUTOR_PROMPT = ("You are FinSight's executor. You answer questions about US companies' SEC 10-K and 10-Q filings "
                   "by calling tools. Call the tool that fetches the first thing the answer needs. Never compute a "
                   "number yourself: use calculate.")
LOCAL_TOOLS = ("get_financial_fact", "compare_periods", "calculate")       # cheap and offline: run for real


def tool_specs() -> list[dict[str, Any]]:
    from tools.registry import SCHEMAS
    return [{"type": "function", "function": schema} for schema in SCHEMAS]


def schemas_fingerprint() -> tuple[str, int]:
    """The schemas' sha256 (which schemas a run measured) and their size in tokens (what every call pays)."""
    from core.tokens import count_tokens
    from tools.registry import SCHEMAS
    text = json.dumps(SCHEMAS, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), count_tokens(text)


def questions() -> list[dict[str, Any]]:
    """The answerable development questions (refusals left out: the executor never sees them)."""
    return [q for q in load_questions("all") if q["kind"] != "refusal"]


def user_text(q: dict[str, Any]) -> str:
    earlier = "".join(f"Earlier question in this conversation: {h}\n" for h in q["history"])
    return f"{earlier}Question: {q['question']}"


def accepts(name: str, arguments: str) -> tuple[bool, str | None]:
    """Whether the tool takes these arguments (see the module docstring), and why not."""
    from tools.registry import TOOLS, _parse_arguments, call_tool
    if name not in TOOLS:
        return False, f"unknown tool {name!r}"
    parsed, error = _parse_arguments(arguments)
    if error:
        return False, error["error"]["message"]
    if name in LOCAL_TOOLS:
        result = call_tool(name, parsed)
        if isinstance(result, dict) and result.get("error", {}).get("type") == "invalid_input":
            return False, result["error"]["message"]
        return True, None
    try:
        TOOLS[name][1].model_validate(parsed)
    except ValidationError as err:
        return False, first_error(err)
    return True, None


def calc_cases(qs: list[dict[str, Any]], planner_run: Path = PLANNER_RUN) -> list[dict[str, Any]]:
    """The questions that ask for a calculation, with the fact steps the planner planned for them."""
    plans = {}
    for line in planner_run.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        plans[record["id"]] = record["plan"]
    cases = []
    for q in qs:
        expected = expected_operations(q["question"])
        plan = plans.get(q["id"])
        facts = [s["arguments"] for s in (plan or {}).get("steps", []) if s["tool"] == "get_financial_fact"]
        if expected and len(facts) >= 2:
            cases.append({**q, "expected": sorted(expected), "facts": facts})
    return cases


def calc_messages(case: dict[str, Any], run_tool: Any) -> list[dict[str, Any]] | None:
    """The conversation with the case's figures already fetched (None when a figure cannot be fetched)."""
    calls, results = [], []
    for n, args in enumerate(case["facts"], 1):
        output = run_tool("get_financial_fact", args)
        if isinstance(output, dict) and "error" in output:
            return None
        call_id = f"call_{n}"
        calls.append({"id": call_id, "type": "function",
                      "function": {"name": "get_financial_fact", "arguments": json.dumps(args)}})
        keep = {k: output.get(k) for k in ("value", "unit", "ticker", "metric", "fiscal_year", "fiscal_period")}
        results.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(keep)})
    return [{"role": "system", "content": EXECUTOR_PROMPT}, {"role": "user", "content": user_text(case)},
            {"role": "assistant", "content": None, "tool_calls": calls}, *results]


Reference = tuple[str, dict[str, Any]]


def reference_steps(planner_run: Path = PLANNER_RUN) -> dict[str, Reference]:
    """Per question, the (tool, arguments) of the first step of the planner's reviewed plan."""
    out = {}
    for line in planner_run.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        steps = (record["plan"] or {}).get("steps") or []
        if steps:
            out[record["id"]] = (steps[0]["tool"], steps[0]["arguments"])
    return out


def matches_plan(call: dict[str, Any], reference: Reference | None) -> bool | None:
    """Whether the call fetches what the reviewed plan's first step fetches (None: no reference): the same tool;
    for a fact the same company, metric, fiscal year and period (FY when the call leaves it out); for
    compare_periods the same company, metric and years; for a search the plan's company and year where it set them.
    A call that is accepted can still ask for the wrong year ("most recent" read as the year before)."""
    if reference is None:
        return None
    if not call.get("tool_call"):
        return False
    try:
        args = json.loads(call["arguments"])
    except json.JSONDecodeError:
        return False
    tool, planned = reference
    if call["name"] != tool or not isinstance(args, dict):
        return False
    if tool == "get_financial_fact":
        keys, args = ("ticker", "metric", "fiscal_year"), {**args, "fiscal_period": args.get("fiscal_period") or "FY"}
        return all(args.get(k) == planned.get(k) for k in keys) and args["fiscal_period"] == planned["fiscal_period"]
    if tool == "compare_periods":
        return all(args.get(k) == planned.get(k) for k in ("ticker", "metric")) \
            and sorted(args.get("years") or []) == sorted(planned.get("years") or [])
    return all(args.get(k) == planned[k] for k in ("ticker", "fiscal_year") if planned.get(k) is not None)


def first_call(outcome: Any) -> dict[str, Any]:
    if not outcome.ok:
        return {"tool_call": False, "why": outcome.error_type}
    if not outcome.tool_calls:
        return {"tool_call": False, "why": "answered in text"}
    call = outcome.tool_calls[0]
    accepted, why = accepts(call["name"], call["arguments"])
    return {"tool_call": True, "name": call["name"], "arguments": call["arguments"], "accepted": accepted, "why": why}


async def measure(session: Session, qs: list[dict[str, Any]], cases: list[dict[str, Any]],
                  run_tool: Any) -> dict[str, Any]:
    specs = tool_specs()
    first, calc = [], []
    for q in qs:
        messages = [{"role": "system", "content": EXECUTOR_PROMPT}, {"role": "user", "content": user_text(q)}]
        outcome, _ = await session.send(q["id"], session.base(messages, tools=specs, tool_choice="auto"))
        first.append({"id": q["id"], "question": q["question"], "prompt_tokens": outcome.prompt_tokens,
                      **first_call(outcome)})
    for case in cases:
        messages = calc_messages(case, run_tool)
        if messages is None:
            calc.append({"id": case["id"], "skipped": "a planned figure could not be fetched"})
            continue
        outcome, _ = await session.send(f"{case['id']} calc", session.base(messages, tools=specs, tool_choice="auto"))
        call = first_call(outcome)
        operation = None
        if call.get("name") == "calculate":
            try:
                operation = json.loads(call["arguments"]).get("operation")
            except (json.JSONDecodeError, AttributeError):
                operation = None
        calc.append({"id": case["id"], "question": case["question"], "expected": case["expected"],
                     "operation": operation, "right_operation": operation in case["expected"], **call})
    return {"first_call": first, "calculator": calc}


def summarize(run: dict[str, Any], references: dict[str, Reference] | None = None) -> dict[str, Any]:
    first, calc = run["first_call"], [c for c in run["calculator"] if "skipped" not in c]
    references = reference_steps() if references is None else references
    matched = [matches_plan(f, references.get(f["id"])) for f in first]
    return {"questions": len(first), "tool_calls": sum(f["tool_call"] for f in first),
            "accepted": sum(f.get("accepted", False) for f in first),
            "with_reference": sum(m is not None for m in matched), "matches_plan": sum(m is True for m in matched),
            "calc_cases": len(calc), "calc_called": sum(c.get("name") == "calculate" for c in calc),
            "calc_right_operation": sum(c["right_operation"] for c in calc),
            "calc_accepted": sum(c.get("name") == "calculate" and c.get("accepted", False) for c in calc)}


def plan_text(reference: Reference | None) -> str:
    if reference is None:
        return "-"
    tool, args = reference
    return f"{tool} {json.dumps({k: v for k, v in args.items() if v is not None})}"


def markdown(runs: dict[str, dict[str, Any]]) -> str:
    """Every label side by side, then each run's rejected calls."""
    labels = list(runs)
    rows = [("first call is a tool call", "tool_calls", "questions"), ("first call accepted by the tool", "accepted",
             "questions"), ("first call fetches what the reviewed plan fetches", "matches_plan", "with_reference"), ("calculator: calls calculate", "calc_called", "calc_cases"),
            ("calculator: expected operation", "calc_right_operation", "calc_cases"),
            ("calculator: arguments accepted", "calc_accepted", "calc_cases")]
    lines = ["# Native tool calling on the development questions (W2-03)", "",
             "Generated by `scripts/probe_tool_calls.py`: tool_choice auto, the `tools.registry` schemas, answerable "
             "development questions only (never gold-v1 or the refusal tests).", "",
             "| Measure | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for title, key, total in rows:
        cells = []
        for label in labels:
            s = runs[label]["summary"]
            cells.append(f"{s[key]}/{s[total]}")
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines.append("| schema tokens per call | " + " | ".join(str(runs[x]["schema_tokens"]) for x in labels) + " |")
    lines.append("| schemas sha256 | " + " | ".join(f"`{runs[x]['schemas_sha256'][:12]}`" for x in labels) + " |")
    for label in labels:
        run = runs[label]
        references = reference_steps()
        bad = [f for f in run["first_call"] if not f.get("accepted")
               or matches_plan(f, references.get(f["id"])) is False]
        bad_calc = [c for c in run["calculator"] if "skipped" not in c and not (c["right_operation"] and c.get("accepted"))]
        lines += ["", f"## {label} ({run['model']}, {run['date']})", ""]
        lines += [f"- first call, {f['id']}: {f.get('name') or '-'} {f.get('arguments') or ''} -> "
                  f"{f.get('why') or 'accepted, but not what the reviewed plan fetches: ' + plan_text(references.get(f['id']))}"
                  for f in bad] or ["- every first call accepted and as planned"]
        lines += [f"- calculator, {c['id']}: {c.get('name') or '-'} {c.get('arguments') or ''} (expected "
                  f"{'/'.join(c['expected'])}) -> {c.get('why') or 'wrong operation'}" for c in bad_calc] \
            or ["- every calculator call right and accepted"]
    return "\n".join(lines) + "\n"


def write(label: str, run: dict[str, Any], out_dir: Path = OUT_DIR, summary_file: Path = SUMMARY) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}.json"
    path.write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*.json"))}
    order = sorted(runs, key=lambda x: (x != "before", x))           # "before" first, then by name
    summary_file.write_text(markdown({k: runs[k] for k in order}), encoding="utf-8", newline="\n")
    return [path, summary_file]


def rescore(run: dict[str, Any], label: str) -> dict[str, Any]:
    """The recorded run with every recorded call re-checked against the current tools (no request)."""
    out = {**run, "label": label, "rescored_from": run["label"],
           "first_call": [], "calculator": []}
    for f in run["first_call"]:
        if f.get("tool_call"):
            accepted, why = accepts(f["name"], f["arguments"])
            f = {**f, "accepted": accepted, "why": why}
        out["first_call"].append(f)
    for c in run["calculator"]:
        if c.get("tool_call"):
            accepted, why = accepts(c["name"], c["arguments"])
            c = {**c, "accepted": accepted, "why": why}
        out["calculator"].append(c)
    out["summary"] = summarize(out)
    return out


async def main_async(label: str) -> int:
    from core.llm import LLMClient
    from core.settings import get_settings
    from tools.registry import call_tool
    settings = get_settings().model_copy(update={"LLM_CACHE_DIR": None, "LLM_TIMEOUT": PROBE_TIMEOUT_S})
    qs = questions()
    cases = calc_cases(qs)
    async with LLMClient(settings) as llm:
        result = await measure(Session(llm.client, settings), qs, cases, call_tool)
    sha, tokens = schemas_fingerprint()
    run = {"label": label, "model": settings.LLM_MODEL, "date": datetime.now(UTC).strftime("%Y-%m-%d"),
           "schemas_sha256": sha, "schema_tokens": tokens, "executor_prompt": EXECUTOR_PROMPT, **result}
    run["summary"] = summarize(run)
    for path in write(label, run):
        print(f"written: {path.relative_to(ROOT)}")
    print(run["summary"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--label", required=True, help="e.g. before / after (the run's file name)")
    parser.add_argument("--rescore", metavar="LABEL", help="re-check the recorded run LABEL; send no request")
    args = parser.parse_args(argv)
    if args.rescore:
        run = rescore(json.loads((OUT_DIR / f"{args.rescore}.json").read_text(encoding="utf-8")), args.label)
        for path in write(args.label, run):
            print(f"written: {path.relative_to(ROOT)}")
        print(run["summary"])
        return 0
    return asyncio.run(main_async(args.label))


if __name__ == "__main__":
    sys.exit(main())
