"""W2-02: run the planner node on the development questions and score its refusal decisions and plans.

    python eval/run_planner.py --run P9                 # all 49: eval/dev_set.jsonl (15) + eval/planner_dev.jsonl (34)
    python eval/run_planner.py --run P9 --questions pdev --limit 5

Only development questions: the 15 of the development set and the planner's own tuning questions
(eval/planner_dev.jsonl: refusals for every rule, partial refusals and answerable controls, written for tuning and
kept apart from gold-v1, the refusal tests and the injection tests by tests/test_planner.py). There is no option for
gold-v1 or the refusal tests: the prompt is tuned here, then frozen and pinned (agent/planner.py PROMPT_SHA256), and
only then do the agent's gold runs (W2-11) see it.

Writes eval/results/planner_runs/<run>.jsonl (one record per question: the plan, the decision, tokens, seconds) and
eval/results/planner_dev.md (the summary of that run). Replies are cached (LLM_CACHE_DIR), so a re-run with the same
prompt costs no requests. One request at a time through core/llm.py (the shared key's rules, week1_plan §2.2).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.planner import PROMPT_VERSION, plan_question, prompt_sha256, rules_sha256  # noqa: E402
from agent.state import Plan, plan_json_schema  # noqa: E402
from retrieval.entities import from_turns  # noqa: E402

EVAL_DIR = ROOT / "eval"
DEV_FILE = EVAL_DIR / "dev_set.jsonl"
PLANNER_DEV_FILE = EVAL_DIR / "planner_dev.jsonl"
RUNS_DIR = EVAL_DIR / "results" / "planner_runs"
SUMMARY_FILE = EVAL_DIR / "results" / "planner_dev.md"
SOURCES = {"dev": (DEV_FILE,), "pdev": (PLANNER_DEV_FILE,), "all": (DEV_FILE, PLANNER_DEV_FILE)}
ALLOWED_PREFIXES = ("dev-", "pdev-")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_questions(which: str = "all") -> list[dict[str, Any]]:
    """The questions as {id, kind, question, history, expected_refused, expected_rule}. Development-set items are
    mapped: a G item with a rule is a refusal, every other item (the "not reported" trap included) a control; a
    follow-up's history is the question it follows."""
    out: list[dict[str, Any]] = []
    for path in SOURCES[which]:
        rows = _jsonl(path)
        if path == DEV_FILE:
            by_id = {r["id"]: r for r in rows}
            for r in rows:
                refused = r["category"] == "refuse" and r.get("refusal_rule") is not None
                prev = r.get("previous_turn_id")
                out.append({"id": r["id"], "kind": "refusal" if refused else "control", "question": r["question"],
                            "history": [by_id[prev]["question"]] if prev else [], "expected_refused": refused,
                            "expected_rule": r.get("refusal_rule") if refused else None})
        else:
            out += [{k: r[k] for k in ("id", "kind", "question", "history", "expected_refused", "expected_rule")}
                    for r in rows]
    bad = [q["id"] for q in out if not q["id"].startswith(ALLOWED_PREFIXES)]
    if bad:
        raise SystemExit(f"not development questions: {bad}")
    return out


def entity_filters_used(plan: Plan, tickers: tuple[str, ...]) -> bool | None:
    """Whether the plan's searches use only the question's companies AND search every one of them (None: no search
    step or no company named). Membership alone would pass a plan that searches one of two named companies."""
    searches = [s for s in plan.steps if s.tool == "search_filings"]
    if not searches or not tickers:
        return None
    searched = {s.arguments.get("ticker") for s in searches}
    return searched <= set(tickers) and set(tickers) <= searched


async def plan_one(llm: Any, q: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    state = {"question": q["question"], "history": q["history"], "plan": None, "tool_results": []}
    entities = from_turns([*q["history"], q["question"]])
    result = await plan_question(llm, state, entities, schema=schema)
    plan, entities = result.plan, result.entities
    record = {**q, "entities": {"tickers": list(entities.tickers), "years": list(entities.years),
                                "periods": list(entities.periods)},
              "plan": plan.model_dump() if plan else None, "json_valid": plan is not None, "retries": result.retries,
              "error": result.error, "notes": result.notes, "prompt_tokens": result.prompt_tokens,
              "completion_tokens": result.completion_tokens, "seconds": round(result.seconds, 3),
              "cached": result.cached}
    if plan is None:
        record.update(refused=None, rule=None, decision_correct=False, steps=0, tools=[], calculation=None,
                      entity_filters=None)
    else:
        record.update(refused=plan.refuse, rule=plan.refusal_rule,
                      decision_correct=plan.refuse == q["expected_refused"] and plan.refusal_rule == q["expected_rule"],
                      steps=len(plan.steps), tools=[s.tool for s in plan.steps], calculation=plan.calculation,
                      entity_filters=entity_filters_used(plan, entities.tickers))
    return record


async def run(llm: Any, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schema = plan_json_schema()
    records = []
    for q in questions:                                 # one at a time: the shared key (week1_plan §2.2)
        records.append(await plan_one(llm, q, schema))
    return records


def _pct(n: int, d: int) -> str:
    return f"{n}/{d}" + (f" = {100 * n / d:.0f}%" if d else "")


def summary(records: list[dict[str, Any]], run_id: str, model: str) -> str:
    """The run's summary (markdown): decisions by kind and rule, plan validity, entity filters, cost."""
    by_kind: dict[str, list[dict]] = defaultdict(list)
    by_rule: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_kind[r["kind"]].append(r)
        by_rule[r["expected_rule"] or "none"].append(r)
    correct = sum(r["decision_correct"] for r in records)
    wrong = [r for r in records if not r["decision_correct"]]
    with_filters = [r["entity_filters"] for r in records if r["entity_filters"] is not None]
    seconds = sorted(r["seconds"] for r in records)
    tokens = [r["prompt_tokens"] + r["completion_tokens"] for r in records]
    lines = [f"# Planner on the development questions ({run_id})", "",
             f"Generated by `eval/run_planner.py`. Model `{model}`, prompt `{PROMPT_VERSION}` "
             f"(sha256 `{prompt_sha256()[:12]}`), {len(records)} questions (development set and "
             "`eval/planner_dev.jsonl`; never gold-v1 or the refusal tests).", "",
             "| Measure | Result |", "|---|---|",
             f"| refusal decision correct (refused or not, and the rule) | {_pct(correct, len(records))} |"]
    for kind in ("refusal", "partial", "control"):
        if by_kind[kind]:
            lines.append(f"| {kind} | {_pct(sum(r['decision_correct'] for r in by_kind[kind]), len(by_kind[kind]))} |")
    lines += [f"| valid plan (JSON and the Plan rules) | {_pct(sum(r['json_valid'] for r in records), len(records))} |",
              f"| plans that needed the JSON retry | {sum(1 for r in records if r['retries'])} |",
              f"| plans whose searches cover exactly the question's companies | {_pct(sum(with_filters), len(with_filters))} |",
              f"| tokens per plan, mean | {statistics.mean(tokens):.0f} |" if tokens else "| tokens | - |",
              f"| seconds per plan, median / max | {statistics.median(seconds):.1f} / {seconds[-1]:.1f} |"
              if seconds else "| seconds | - |", "",
              "| Expected rule | Correct |", "|---|---|"]
    lines += [f"| {rule} | {_pct(sum(r['decision_correct'] for r in rs), len(rs))} |"
              for rule, rs in sorted(by_rule.items())]
    tools = Counter(t for r in records for t in r["tools"])
    lines += ["", "Tools planned: " + (", ".join(f"{t} {n}" for t, n in tools.most_common()) or "none") + ".", ""]
    if wrong:
        lines += ["Wrong decisions:", ""]
        lines += [f"- {r['id']} ({r['kind']}): expected refused={r['expected_refused']} rule={r['expected_rule']}, "
                  f"got refused={r['refused']} rule={r['rule']}" + (f" ({r['error'][:80]})" if r["error"] else "")
                  + f": {r['question']}" for r in wrong]
    return "\n".join(lines) + "\n"


def write(records: list[dict[str, Any]], run_id: str, model: str, runs_dir: Path = RUNS_DIR,
          summary_file: Path = SUMMARY_FILE) -> tuple[Path, Path]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs = runs_dir / f"{run_id}.jsonl"
    meta = {"prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_sha256(), "rules_sha256": rules_sha256(),
            "model": model}
    runs.write_text("".join(json.dumps({**meta, **r}, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8", newline="\n")
    summary_file.write_text(summary(records, run_id, model), encoding="utf-8", newline="\n")
    return runs, summary_file


async def main_async(args: argparse.Namespace) -> int:
    from core.llm import LLMClient
    from core.settings import get_settings
    settings = get_settings()
    questions = load_questions(args.questions)[: args.limit]
    async with LLMClient(settings) as llm:
        records = await run(llm, questions)
        usage = llm.usage
    runs, summary_path = write(records, args.run, settings.LLM_MODEL)
    correct = sum(r["decision_correct"] for r in records)
    print(f"{len(records)} questions: {correct} correct decisions; {usage.calls} requests sent, "
          f"{usage.cache_hits} cached; {runs.relative_to(ROOT)}, {summary_path.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="run id, e.g. P1 (the runs file's name)")
    parser.add_argument("--questions", choices=sorted(SOURCES), default="all")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
