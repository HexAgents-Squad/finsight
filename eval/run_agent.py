"""W2-11: the agent graph on the 80 gold questions and 22 refusal tests (A1), as a runs file the W1-08 scorer reads.

    python eval/run_agent.py                       # A1 on gold + refusal tests -> eval/results/model_runs/
    python eval/run_agent.py --max-requests 400    # stop before the 401st live chat request (cache hits are free)
    python eval/run_agent.py --questions dev       # the 15 development questions -> eval/results/dev/model_runs/
    python eval/run_agent.py --questions dev2      # W3-02: the development set v2 -> eval/results/dev_v2/model_runs/
    python eval/run_agent.py --ablation no-critic  # W3-01: A1 with one component off (A2-A4, below)

The questions and their earlier turns come from the context packs (the same 102 records B0 and S1-S3 answered),
but the agent retrieves for itself: the packs' passages are never shown to it. A follow-up (gold category F) gets
its earlier questions as `history`, the way the agent's memory holds a conversation (W2-06: questions, not
answers), so no gold answer reaches the model. The real nodes (agent.nodes.real_nodes) on the shared endpoint
through the reply cache, one question at a time (the plan: one evaluation run at a time on the shared key), the
tools on data/. Each question is a RunRecord (eval/runs_schema.py) with:
- `parsed`: the answer mapped to the B0 fields by b0_answer (below), so the scorer judges it like B0;
- `tool_calls`: every tool call with its arguments, typed error and evidence (the passages and figures it
  returned), which the scorer uses as the run's evidence and its valid citations;
- `agent`: the whole answer contract, plan, critic verdict, node visits and cost (agent/transcript.run_record).
The meta file pins the prompts (planner and composer sha256, tests/test_planner.py and tests/test_composer.py),
the limits, the retrieval settings, the requests sent and the packs read; a run other than A1 also records the
planner's rules (rules_sha256). A gold run starts only when its configuration is pinned (W3-08, pin_problems): both
prompts are their pinned PROMPT_SHA256 and the code (CODE_PATHS) is committed, so the meta's git_commit is the code
that ran. Nothing is written until every question
has a record; a run stopped by the budget is re-run with a larger one and replays the answered questions from
the reply cache for free. After a fix to how a run is recorded or mapped (not to the agent), `--replay-of <the
earlier runs file>` re-derives the file from the cache and keeps the earlier run's timings, requests and dates,
only if every answer and tool call is the same; the meta's `replay` says when and with which code.
The ablations (W3-01, ADR-036): `--ablation` turns one component off on the A1 configuration, everything else
unchanged (prompts, limits, retrieval settings, questions), so each is paired with A1 question by question:
- A2 `no-calculator`: the executor never runs the plan's calculation (the planner still names it; the composer
  gets the fetched figures without a calculated one);
- A3 `no-critic`: no critic node; the composer's draft is the answer, unverified, with no send-back;
- A4 `no-statements-first`: search_filings on plain hybrid retrieval, without the statements-first layer.
Each has its own run id (ABLATIONS); the meta records `ablation` and `ablation_of` (always A1's committed run id,
RUN_ID: an ablation is paired with that run), and a replay is refused across runs, so an ablation never borrows A1's
timings. Gold only: a development set has no committed A1 run to pair with, so `--ablation` with `--questions dev`
or `dev2` is refused. Unchanged calls replay from the reply cache, so a run costs only the requests its switch
changes. Nodes given by a test are used as they are (no switch applied).
Exit codes: 0 = every question answered, 1 = some questions raised (recorded as errors), 2 = setup or budget.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.context_pack import DEV2_PACK, DEV_PACK, GOLD_PACK, REFUSAL_PACK, sha256  # noqa: E402
from eval.run_retrieval import RESULTS_DIR  # noqa: E402
from eval.runs_schema import (  # noqa: E402
    RUN_ID_PATTERN,
    AgentInfo,
    B0Answer,
    Evidence,
    RunMeta,
    RunRecord,
    ToolCallRecord,
)

RUNS_DIR = RESULTS_DIR / "model_runs"
DEV_RUNS_DIR = RESULTS_DIR / "dev" / "model_runs"
DEV2_RUNS_DIR = RESULTS_DIR / "dev_v2" / "model_runs"      # W3-02: the development set v2
# the packs each question set reads (questions and earlier turns only) and where its runs go
QUESTION_SETS = {"gold": ((GOLD_PACK, REFUSAL_PACK), RUNS_DIR), "dev": ((DEV_PACK,), DEV_RUNS_DIR),
                 "dev2": ((DEV2_PACK,), DEV2_RUNS_DIR)}
RUN_ID = "A1_qwen3-8b_agent_think-off_a-v1"
CALCULATOR = "calculate"


@dataclass(frozen=True)
class Ablation:
    """One W3-01 ablation: its run number, what it switches off, and its agent.nodes.real_nodes switch."""
    number: str
    off: str
    switch: str

    def run_id(self, name: str) -> str:
        return f"{self.number}{RUN_ID[2:]}-{name}"          # A1's id with the ablation's number and name


ABLATIONS = {
    "no-calculator": Ablation("A2", "the calculator", "calculator"),
    "no-critic": Ablation("A3", "the critic", "critic"),
    "no-statements-first": Ablation("A4", "statements-first retrieval", "statements_first"),
}


@dataclass(frozen=True)
class Item:
    """One question as the packs hold it: its id, text and the earlier questions of its conversation."""
    id: str
    question: str
    history: tuple[str, ...]


def load_items(pack_files: list[Path]) -> list[Item]:
    items = []
    for path in pack_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                items.append(Item(r["id"], r["question"], tuple(h["question"] for h in r.get("history") or [])))
    return items


# --------------------------------------------------------------------------------------------- evidence
def figure_text(value: Any) -> str:
    """A number as a filing prints it, every digit kept: no exponent, no trailing zeros (60458000000, 0.0000001,
    1234.56789012, 22.1). A float is read through its shortest repr, so nothing is rounded away."""
    if isinstance(value, int):
        return str(value)
    text = format(Decimal(repr(value)), "f")
    return (text.rstrip("0").rstrip(".") if "." in text else text) or "0"


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def percent_text(x: Any) -> str:
    return figure_text(x) + "%"


def fact_evidence(x: dict[str, Any]) -> Evidence:
    """One XBRL fact (a get_financial_fact result, or a compare_periods row): its value, its original value only when
    a later filing restated it, and on a period row the tool's year-on-year growth as a percentage (computed from
    the fetched facts, so it is evidence like them). The label says whose figure it is, for a grader."""
    nums = [figure_text(x["value"])] if is_number(x.get("value")) else []
    if is_number(x.get("original_value")) and x["original_value"] != x.get("value"):
        nums.append(figure_text(x["original_value"]))
    if is_number(x.get("yoy_growth_pct")):
        nums.append(percent_text(x["yoy_growth_pct"]))
    what = x.get("label") or x.get("metric") or x["concept"]
    label = " ".join(str(p) for p in (x.get("ticker"), x.get("form"), f"FY{x.get('fiscal_year')}",
                                      x.get("fiscal_period"), "·", what, f"({x.get('unit')})") if p)
    return Evidence(source=f"{x['concept']} {x.get('accession_no') or ''}".strip(), text=" ".join(nums), label=label)


def calculation_evidence(x: dict[str, Any]) -> list[Evidence]:
    """A calculation's result, never its inputs (the planner chose them): the scorer counts it as evidence only when
    each input is a figure the run fetched (score_answers.agent_texts), so an invented input passed through the
    calculator is not "supported"."""
    if not is_number(x.get("result")):
        return []
    text = percent_text(x["result"]) if x.get("unit") == "%" else figure_text(x["result"])
    inputs = ", ".join(figure_text(i) for i in x.get("inputs") or [] if is_number(i))
    return [Evidence(source=CALCULATOR, text=text, label=f"{x.get('operation', '')} of {inputs}".strip())]


def evidence(tool: str, output: Any) -> list[Evidence]:
    """What a tool call returned, as the scorer reads it: each passage (search_filings) by chunk_id, each XBRL fact
    (a fact or a period comparison's rows) by "<concept> <accession_no>" (fact_evidence), a calculation's result by
    the tool's name (calculation_evidence). An error returned nothing."""
    if isinstance(output, dict) and isinstance(output.get("error"), dict):
        return []
    if tool == CALCULATOR:
        return calculation_evidence(output) if isinstance(output, dict) else []
    found: list[Evidence] = []

    def walk(x: Any) -> None:
        if isinstance(x, list):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            if isinstance(x.get("chunk_id"), str) and isinstance(x.get("text"), str):
                label = " ".join(str(x[k]) for k in ("ticker", "form", "fiscal_year", "fiscal_period", "item")
                                 if x.get(k))
                found.append(Evidence(source=x["chunk_id"], text=x["text"], label=label or None))
            elif isinstance(x.get("concept"), str):
                found.append(fact_evidence(x))
            else:
                for y in x.values():
                    walk(y)

    walk(output)
    return found


def citation_id(c: dict[str, Any]) -> str:
    """How an agent citation names its source: a passage by chunk_id, an XBRL fact as "<concept> <accession_no>"
    (the same names evidence() gives, so the scorer can tell a source the run was given)."""
    if c.get("chunk_id"):
        return str(c["chunk_id"])
    return f"{c.get('concept') or ''} {c.get('accession_no') or ''}".strip() or "?"


# --------------------------------------------------------------------------------------------- B0 fields
def only(items: list[Any]) -> Any:
    """The one distinct item, else None."""
    distinct = list(dict.fromkeys(items))
    return distinct[0] if len(distinct) == 1 else None


def b0_answer(record: dict[str, Any]) -> B0Answer:
    """The agent's answer (agent/transcript.run_record) in the B0 fields the scorer reads, by a fixed rule:
    - answer, refused, refusal_rule: as the agent gave them;
    - value and unit: the first of these the answer's text states: the last calculation's result (a ratio, a
      change), then the one value when the answer has exactly one; else none (several values without a stated
      calculation: the scorer then says "no value" and a grader can overturn it on the label row). B0's model chose
      whether to fill `value`; the mapping must not give an answer a figure it does not state;
    - no value at all when the answer opens (its first clause) by saying the figure asked is not reported: the figure it then offers
      ("it reports current liabilities of $179,431 million") is not its answer. A "not reported" aside after a
      stated answer ("...; the Services split is not reported separately") keeps the value;
    - company: the one ticker of the values (else of the citations), when they name exactly one;
    - fiscal_year: the one fiscal year of the values (else of the citations), when they name exactly one;
    - citations: each source's name (citation_id)."""
    from core.figures import figures
    from eval.score_answers import NOT_REPORTED_RE, supported, value_figure
    calcs, values, cites = record["calculations"], record["values"], record["citations"]
    candidates = ([(calcs[-1].get("result"), calcs[-1].get("unit"))] if calcs else []) + \
        ([(values[0].get("value"), values[0].get("unit"))] if len(values) == 1 else [])
    source = values or cites
    answer = B0Answer(answer=record["answer"], company=only([x.get("ticker") for x in source if x.get("ticker")]),
                      fiscal_year=only([x.get("fiscal_year") for x in source if isinstance(x.get("fiscal_year"), int)]),
                      citations=[citation_id(c) for c in cites],
                      refused=bool(record["refused"]), refusal_rule=record["refusal_rule"])
    opening = re.split(r"(?<=[.!?;])\s+", answer.answer.strip(), maxsplit=1)[0]      # up to the first ; or sentence end
    if NOT_REPORTED_RE.search(opening):
        return answer
    stated = figures(answer.answer)
    for value, unit in candidates:
        if is_number(value):
            candidate = answer.model_copy(update={"value": value, "unit": unit if isinstance(unit, str) else None})
            if supported(value_figure(candidate), stated):
                return candidate
    return answer


# --------------------------------------------------------------------------------------------- one question
@dataclass
class Counts:
    """The LLM client's totals at one moment (core/llm.Usage), to charge each question its own share."""
    calls: int = 0
    http_retries: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    json_retries: int = 0

    @classmethod
    def of(cls, usage: Any) -> Counts:
        if usage is None:
            return cls()
        return cls(usage.calls, usage.http_retries, usage.cache_hits, usage.prompt_tokens, usage.completion_tokens,
                   usage.json_retries)

    def __sub__(self, o: Counts) -> Counts:
        return Counts(self.calls - o.calls, self.http_retries - o.http_retries, self.cache_hits - o.cache_hits,
                      self.prompt_tokens - o.prompt_tokens, self.completion_tokens - o.completion_tokens,
                      self.json_retries - o.json_retries)


def run_record(item: Item, run_id: str, result: Any, spent: Counts) -> RunRecord:
    """A finished agent run as the question's RunRecord."""
    from agent.transcript import run_record as transcript
    rec = transcript(result, item.question, list(item.history))
    calls = [ToolCallRecord(tool=r.call.name, arguments=dict(r.call.arguments), error=r.error_type,
                            evidence=evidence(r.call.name, r.output))
             for r in result.state.get("tool_results") or []]
    replies = spent.calls - spent.http_retries + spent.cache_hits
    info = AgentInfo(verdict=rec["verdict"], verified=rec["verified"], values=rec["values"],
                     calculations=rec["calculations"], citations=rec["citations"], notes=rec["notes"],
                     history=rec["history"], plan=rec["plan"], visits=rec["visits"], stop_reason=rec["stop_reason"],
                     node_notes=[str(n) for n in rec["node_notes"]], tool_steps=rec["steps"], tokens=rec["tokens"],
                     llm_calls=replies, live_requests=spent.calls, cache_hits=spent.cache_hits)
    contract = {k: rec[k] for k in ("answer", "verified", "refused", "refusal_rule", "values", "calculations",
                                    "citations", "notes")}
    return RunRecord(id=item.id, run_id=run_id, raw_reply=json.dumps(contract, ensure_ascii=False),
                     parsed=b0_answer(rec), json_valid=True, retries=spent.json_retries,
                     prompt_tokens=spent.prompt_tokens, completion_tokens=spent.completion_tokens,
                     latency_s=round(result.seconds, 3), cached=spent.calls == 0 and spent.cache_hits > 0,
                     tool_calls=calls, agent=info)


def error_record(item: Item, run_id: str, err: BaseException, spent: Counts) -> RunRecord:
    """A question the run raised on (a bug, or a node's own failure): recorded, so the scorer counts it as no reply."""
    return RunRecord(id=item.id, run_id=run_id, raw_reply=None, parsed=None, json_valid=False,
                     retries=spent.json_retries, prompt_tokens=spent.prompt_tokens,
                     completion_tokens=spent.completion_tokens, latency_s=0.0, cached=False,
                     error=f"{type(err).__name__}: {err}"[:500])


async def run_items(items: list[Item], nodes: Any, limits: Any, run_id: str,
                    usage: Any = None) -> tuple[list[RunRecord], str | None]:
    """Every item through the graph, one at a time, each with its own earlier questions (never a shared thread);
    stops at the request budget (eval.run_demo.BudgetExceeded) with the reason, keeping the records so far."""
    from agent.graph import build_graph, run
    from eval.run_demo import BudgetExceeded
    graph = build_graph(nodes)
    records: list[RunRecord] = []
    for item in items:
        before = Counts.of(usage)
        try:
            result = await run(item.question, list(item.history), graph=graph, limits=limits)
        except BudgetExceeded as err:
            return records, f"stopped before {item.id}: {err}"
        except Exception as err:                    # noqa: BLE001 - recorded and reported, the run goes on
            traceback.print_exc()
            records.append(error_record(item, run_id, err, Counts.of(usage) - before))
            print(f"{item.id}: ERROR {type(err).__name__}: {err}", flush=True)
            continue
        rec = run_record(item, run_id, result, Counts.of(usage) - before)
        records.append(rec)
        a = rec.agent
        print(f"{item.id}: {a.verdict:32} {rec.latency_s:5.1f}s {a.tokens:6} tokens {a.tool_steps:2} tool calls "
              f"{a.live_requests} live", flush=True)
    return records, None


# --------------------------------------------------------------------------------------------- the run
# What a gold run's git_commit must describe (W3-08): the agent, its tools and data code, the config, the evaluation
# code the runner imports (the answer mapping reads eval/score_answers.py) and the dependencies (git pathspecs; the
# results under eval/ are data, not code)
CODE_PATHS = ("agent", "core", "tools", "retrieval", "ingest", "config", "eval/*.py", "requirements.txt",
              "requirements-dev.txt", "pyproject.toml")


def uncommitted(paths: tuple[str, ...] = CODE_PATHS) -> list[str]:
    """Files under `paths` that differ from HEAD (changed, staged or new), so a run's git_commit would not be the
    code it ran; ["(git unavailable)"] when git cannot tell."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ["(git unavailable)"]
    # "XY path", a rename "R  old -> new" (the new path), a quoted path with spaces ("a b.py")
    return sorted(line[3:].split(" -> ")[-1].strip().strip('"') for line in out.splitlines() if line.strip())


def pin_problems() -> list[str]:
    """Why a gold run's configuration is not pinned (W3-08, docs/week3_plan.txt §2: A5 runs on gold once, with its
    configuration pinned before the run), or []: the planner's and composer's prompts must be the pinned ones
    (PROMPT_SHA256, which the meta records), and the code must be committed (the meta's git_commit)."""
    from agent import composer, planner
    problems = [f"the {name} prompt is not its pinned PROMPT_SHA256 (re-measure on the development set, then re-pin)"
                for name, module in (("planner", planner), ("composer", composer))
                if module.prompt_sha256() != module.PROMPT_SHA256]
    if dirty := uncommitted():
        problems.append(f"uncommitted changes in {', '.join(dirty[:5])}{' ...' if len(dirty) > 5 else ''}: commit "
                        "them first, so the run's git_commit is the code it ran")
    return problems


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def run_meta(run_id: str, pack_files: list[Path], settings: Any, limits: Any, started: str, totals: dict[str, int],
             questions: str, ablation: str | None = None) -> RunMeta:
    from agent import composer, planner
    return RunMeta(
        run_id=run_id, model=settings.LLM_MODEL, endpoint="hosted", context="agent",
        thinking=bool(settings.LLM_ENABLE_THINKING), temperature=0.0, max_tokens=settings.LLM_MAX_COMPLETION_TOKENS,
        prompt_version=f"{planner.PROMPT_VERSION} + composer {composer.PROMPT_SHA256[:12]}",
        context_pack=" + ".join(p.name for p in pack_files) + " (questions and earlier turns only)",
        context_pack_sha256={p.name: sha256(p) for p in pack_files},
        planner_prompt_sha256=planner.PROMPT_SHA256, composer_prompt_sha256=composer.PROMPT_SHA256,
        limits={"max_steps": limits.max_steps, "max_tokens": limits.max_tokens, "timeout_s": limits.timeout_s},
        # search_filings' stack (ADR-017): C3 chunks, hybrid RRF, statements first (off only in the A4 ablation)
        retrieval={"embed_model": settings.EMBED_MODEL, "rerank": settings.RETRIEVAL_RERANK, "chunks": "C3",
                   "fusion": "hybrid RRF", "statements_first": ablation != "no-statements-first"},
        **({"ablation": ablation, "ablation_of": RUN_ID} if ablation else {}),     # A1's meta keeps its exact keys
        # W3-08: the planner's deterministic rules, which the prompt's pin does not cover (A1's meta keeps its keys)
        **({"rules_sha256": planner.rules_sha256()} if run_id != RUN_ID else {}),
        reply_cache=settings.LLM_CACHE_DIR is not None, questions=questions, requests=totals,
        git_commit=git_commit(), started=started, finished=datetime.now(UTC).isoformat(timespec="seconds"))


REPLAY_SAME = ("run_id", "model", "questions", "context_pack_sha256", "planner_prompt_sha256",
               "composer_prompt_sha256", "limits", "ablation")      # what makes two files the same run


def carry_over(records: list[RunRecord], previous: Path, meta: RunMeta) -> tuple[list[RunRecord], dict[str, Any]]:
    """`records`, re-derived from the reply cache after a runner fix (`--replay-of`), with the earlier run's timings
    and cost: a replay's cache hits take no time and send nothing, but the answers are the earlier run's, so its
    seconds, requests and tokens are the ones to report. Refused (ValueError) unless `previous` (a runs file, read as
    JSON: it may predate fields added since) is the same run (REPLAY_SAME: run id, model, questions, packs, prompts,
    limits; so an ablation can never borrow A1's timings) and every question has the same answer and tool calls.
    Returns the records and the earlier meta."""
    prev = {r["id"]: r for r in map(json.loads, previous.read_text(encoding="utf-8").splitlines()) if r}
    prev_meta = json.loads(previous.with_name(previous.stem + ".meta.json").read_text(encoding="utf-8"))
    now = json.loads(meta.model_dump_json())
    differ = [k for k in REPLAY_SAME if prev_meta.get(k) != now.get(k)]
    if differ:
        raise ValueError(f"{previous.name} is another run ({', '.join(differ)} differ); not a replay")
    if set(prev) != {r.id for r in records}:
        raise ValueError(f"{previous.name} has other questions")
    out = []
    for r in records:
        p = prev[r.id]
        same_calls = [(c.tool, c.arguments, c.error) for c in r.tool_calls or []] == \
            [(c["tool"], c["arguments"], c.get("error")) for c in p.get("tool_calls") or []]
        if r.error or p.get("error") or json.loads(r.raw_reply) != json.loads(p["raw_reply"]) or not same_calls:
            raise ValueError(f"{r.id}: the replay's answer or tool calls differ from {previous.name}; not a replay")
        pa = p["agent"]
        out.append(r.model_copy(update={
            "latency_s": p["latency_s"], "cached": p["cached"], "retries": p["retries"],
            "prompt_tokens": p["prompt_tokens"], "completion_tokens": p["completion_tokens"],
            # the cost is the earlier run's whole (W3-08: a fix that saves a call, such as W3-07's, would otherwise
            # pair the earlier run's calls with the replay's own token count)
            "agent": r.agent.model_copy(update={"llm_calls": pa["llm_calls"], "live_requests": pa["live_requests"],
                                                "cache_hits": pa.get("cache_hits", 0),
                                                "tokens": pa.get("tokens", r.agent.tokens)})}))
    return out, prev_meta


def ablation_nodes(llm: Any, ablation: str | None, call_tool: Any = None) -> Any:
    """The real nodes on `llm`, with the ablation's one component off (none: A1's nodes)."""
    from agent.nodes import real_nodes
    switches = {ABLATIONS[ablation].switch: False} if ablation else {}
    return real_nodes(llm, call_tool, **switches)


async def run_real(items: list[Item], settings: Any, max_requests: int, limits: Any, run_id: str,
                   ablation: str | None = None) -> tuple[list[RunRecord], str | None, Any]:
    from core.llm import LLMClient
    from eval.run_demo import BudgetedLLM
    async with LLMClient(settings) as client:
        budgeted = BudgetedLLM(client, max_requests)
        records, stopped = await run_items(items, ablation_nodes(budgeted, ablation), limits, run_id,
                                           usage=client.usage)
        return records, stopped, client.usage


def run_id_problem(run_id: str, ablation: str | None) -> str | None:
    """Why `run_id` cannot name this run, or None: it must be an agent run id, and its number is the ablation's
    (A2-A4) exactly when the run is that ablation, so a file's name never hides a switched-off component."""
    if not (re.fullmatch(RUN_ID_PATTERN, run_id) and run_id.startswith("A") and "_agent_" in run_id):
        return f"not an agent run id (like {RUN_ID})"
    number, taken = run_id.split("_", 1)[0], {a.number: name for name, a in ABLATIONS.items()}
    if ablation and number != ABLATIONS[ablation].number:
        return f"the {ablation} ablation's run id starts with {ABLATIONS[ablation].number}"
    if not ablation and number in taken:
        return f"{number} is the {taken[number]} ablation's number; pass --ablation {taken[number]}"
    return None


def main(argv: list[str] | None = None, *, nodes: Any = None, settings: Any = None, limits: Any = None) -> int:
    from eval.validate_gold import safe_console
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--questions", choices=sorted(QUESTION_SETS), default="gold",
                    help="gold (80 + 22 refusal tests), dev (the 15 development questions) or dev2 (the development "
                         "set v2, W3-02)")
    ap.add_argument("--ablation", choices=sorted(ABLATIONS),
                    help="W3-01: switch one component off (A2-A4); the run id defaults to the ablation's")
    ap.add_argument("--run-id", help=f"default {RUN_ID}, or the ablation's (e.g. "
                    f"{ABLATIONS['no-critic'].run_id('no-critic')})")
    ap.add_argument("--max-requests", type=int, default=400, help="live chat requests this run may send")
    ap.add_argument("--out-dir", type=Path, help=f"default {RUNS_DIR} (dev: {DEV_RUNS_DIR}; dev2: {DEV2_RUNS_DIR})")
    ap.add_argument("--pack-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--replay-of", type=Path, help="an earlier runs file of this run: re-derive it from the reply "
                    "cache after a runner fix, keeping its timings and cost (refused unless every answer and tool "
                    "call is the same)")
    args = ap.parse_args(argv)
    ablation = ABLATIONS[args.ablation] if args.ablation else None
    args.run_id = args.run_id or (ablation.run_id(args.ablation) if ablation else RUN_ID)
    if problem := run_id_problem(args.run_id, args.ablation):
        print(f"--run-id {args.run_id!r}: {problem}", file=sys.stderr)
        return 2
    if ablation and args.questions != "gold":
        print(f"--ablation {args.ablation}: an ablation is paired with A1's gold run, so it runs on the gold questions "
              "only", file=sys.stderr)
        return 2
    packs, default_dir = QUESTION_SETS[args.questions]
    out_dir = args.out_dir or default_dir
    pack_files = [args.pack_dir / name for name in packs]
    try:
        items = load_items(pack_files)
        from agent.graph import Limits
        if settings is None:
            from core.settings import get_settings
            settings = get_settings()
    except (ValueError, OSError) as err:           # pydantic's ValidationError: an incomplete .env
        print(f"setup incomplete: {str(err).splitlines()[0]}", file=sys.stderr)
        return 2
    limits = limits or Limits.from_settings(settings)
    if args.questions == "gold" and nodes is None and (problems := pin_problems()):
        print("not pinned, nothing run:\n" + "\n".join(f"- {p}" for p in problems), file=sys.stderr)
        return 2
    started = datetime.now(UTC).isoformat(timespec="seconds")
    # A replay must come from the reply cache: one live request at most (the first miss), and then nothing is written
    budget = 1 if args.replay_of else args.max_requests
    if nodes is None:
        records, stopped, usage = asyncio.run(run_real(items, settings, budget, limits, args.run_id, args.ablation))
        # chat requests only: a search embeds its query through the search tool's own embedder, not this client
        totals = {"live": usage.calls, "cache_hits": usage.cache_hits, "http_retries": usage.http_retries}
    else:                                           # given nodes (a test): no client, no requests
        records, stopped = asyncio.run(run_items(items, nodes, limits, args.run_id))
        totals = {"live": 0, "cache_hits": 0, "http_retries": 0}
    if args.replay_of and (stopped or totals["live"]):
        print(f"--replay-of {args.replay_of}: a reply was not in the cache ({totals['live']} live request(s)); not a "
              f"replay, nothing written", file=sys.stderr)
        return 2
    if stopped:
        print(f"{stopped}; {len(records)}/{len(items)} answered, nothing written (re-run with a larger "
              f"--max-requests: the answered questions replay from the reply cache)", file=sys.stderr)
        return 2
    from eval.run_models import write_run  # the runners' one writer: meta, then records, never half a file
    meta = run_meta(args.run_id, pack_files, settings, limits, started, totals, args.questions, args.ablation)
    if args.replay_of:
        try:
            records, prev = carry_over(records, args.replay_of, meta)
        except (ValueError, OSError, KeyError) as err:
            print(f"--replay-of {args.replay_of}: {err}; nothing written", file=sys.stderr)
            return 2
        replay = {"replayed": meta.finished, "git_commit": meta.git_commit, "requests": totals}
        # chat requests only (an older meta also had embed_calls, which never saw the search tool's embedder)
        totals = {k: v for k, v in prev["requests"].items() if k in ("live", "cache_hits", "http_retries")}
        meta = meta.model_copy(update={"started": prev["started"], "finished": prev["finished"], "requests": totals,
                                       "git_commit": prev["git_commit"], "replay": replay})
    path = write_run(records, meta, out_dir)
    errors = [r.id for r in records if r.error]
    print(f"wrote {path} ({len(records)} questions; live requests {totals['live']}, cache hits "
          f"{totals['cache_hits']}){'; errors: ' + ', '.join(errors) if errors else ''}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
