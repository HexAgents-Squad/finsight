"""The planner node (W2-02): refusal first, entities from the question, a JSON plan of tool steps.

    make_planner(llm)        -> the node: state -> {entities, plan, tokens[, errors]}
    system_prompt()          the frozen prompt (agent/prompts/planner.txt with the corpus scope, tools and metrics)
    user_message(state, e)   the question, the earlier turns, what retrieval/entities.py read, a replan's errors
    apply_entities(plan, e)  the question's companies, years and period in every search_filings step (ADR-017)

One guided call per plan (agent/guided.py, ADR-016): the reply is decoded against plan_json_schema() (one PlanStep
variant per tool, with that tool's own argument schema) and validated as a Plan (a refusal has no steps, ...);
an invalid reply is re-asked once with the error. A refusal (R1-R7, docs/spec.md §6) is decided before any step:
the prompt asks for it first and `reasoning` comes before the decision (ADR-018). Its examples carry no figures
(ADR-018: S1's refusal examples helped refusals but S1 copied a real figure from one).

When the call fails or the reply is still invalid, the node returns no plan with the reason in `errors`, and the
graph gives up ("could not verify: the planner returned no plan"). On a replan (after a typed tool error) the
message shows the previous steps and the errors, so the new plan can avoid them.

The prompt is tuned on the development questions only (eval/dev_set.jsonl and eval/planner_dev.jsonl, never gold-v1
or the refusal tests; eval/run_planner.py) and pinned: PROMPT_SHA256 is prompt_sha256(), over the system prompt, the
decoding schema, the message templates and the rule patterns; a test fails when any of them changes (a prompt edit,
a tool's argument model, the metrics, the scope, a rule). Re-tune, then re-pin.
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, get_args

import yaml

from agent.critic_checks import expected_operations
from agent.guided import CompletionClient, GuidedReply, ask_guided
from agent.state import MAX_PLAN_STEPS, AgentState, Plan, PlannedTool, PlanStep, ToolResult, plan_json_schema
from core.llm import Message
from ingest.edgar import COMPANIES_FILE, load_scope
from ingest.xbrl import load_concept_map
from retrieval.entities import Entities, from_turns, indexed
from tools.registry import TOOLS

PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "planner.txt"
PROMPT_VERSION = "planner-v1"
PROMPT_SHA256 = "fe027517f1c1785fc5c25f443da31a4a435077ab3e3b78702884144ee9c26383"   # measured in P7
SEARCH = "search_filings"


# ----------------------------------------------------------------------------------------------- prompt
def tools_text() -> str:
    """One line per planned tool: its arguments (required first) and the first sentence of its description,
    generated from the tools' own models (tools.registry), so the prompt follows a schema change (W2-03)."""
    lines = []
    for name in get_args(PlannedTool):
        _, args_model, schema = TOOLS[name]
        required = [k for k, f in args_model.model_fields.items() if f.is_required()]
        optional = [k for k, f in args_model.model_fields.items() if not f.is_required()]
        sentence = schema["description"].split(". ")[0].rstrip(".") + "."
        lines.append(f"- {name}({', '.join(required)}; optional: {', '.join(optional) or 'none'}): {sentence}")
    return "\n".join(lines)


@functools.cache
def system_prompt() -> str:
    """The planner's system prompt: agent/prompts/planner.txt with the corpus scope (config/companies.yaml), the
    tools (tools.registry) and the metrics (config/concept_map.yaml) filled in."""
    scope = load_scope()
    names = {c["ticker"]: c["name"] for c in yaml.safe_load(COMPANIES_FILE.read_text(encoding="utf-8"))["companies"]}
    companies = "; ".join(f"{names[c.ticker]} ({c.ticker})" for c in scope.companies)
    tenq = f"{', '.join(str(y) for y in sorted(scope.tenq_years))} {', '.join(sorted(scope.tenq_periods))}"
    return Template(PROMPT_FILE.read_text(encoding="utf-8")).substitute(
        COMPANIES=companies, TENK_YEARS=", ".join(str(y) for y in sorted(scope.tenk_years)), TENQ=tenq,
        TOOLS=tools_text(), METRICS=", ".join(sorted(load_concept_map())))


def prompt_sha256() -> str:
    """The planner's pin: the system prompt, the decoding schema, every message template the model sees (entity
    hints, the replan text, the R3 correction, the JSON re-ask) and the patterns of the deterministic rules. Any
    change to what the model is shown or how its plan is corrected changes it (tests/test_planner.py)."""
    from agent.guided import RETRY_INSTRUCTION
    parts = [system_prompt(), json.dumps(plan_json_schema(), sort_keys=True), RETRY_INSTRUCTION, NO_COMPANY_HINT,
             NOT_INDEXED_LINE, REPLAN_TEXT, CORRECTION_TEXT, FORECAST.pattern, LIVE_OR_NEWS.pattern, YTD_WORDS.pattern,
             NAMES_AN_ITEM.pattern, FISCAL_WORDS.pattern, FOURTH_QUARTER.pattern, str(YEAR_MARGIN), str(SEARCH_K)]
    return hashlib.sha256("\n\0\n".join(parts).encode("utf-8")).hexdigest()


# Message templates (pinned with the prompt): kept verbatim, so cached replies stay valid while the rules change
NO_COMPANY_HINT = " (a company the question names that is not in the indexed list is not indexed: R3)"
NOT_INDEXED_LINE = "- not indexed: {missing}"
REPLAN_TEXT = ("Your previous plan failed. Its steps:\n{steps}\nTool errors:\n{errors}\nMake a new plan that avoids "
               "these errors. If the data is not in the filings, plan no steps.")
CORRECTION_TEXT = ("Check against the indexed filings: the filings for {tickers}, {periods} ARE in the corpus (10-K "
                   "fiscal years {years} are all indexed and already reported), and the question asks about reported "
                   "figures, not a forecast, so R3 does not apply to them. Only "
                   "these companies and periods are checked here: a company the question names that is not in the "
                   "indexed list is still R3. A follow-up keeps the company and the line item of the earlier "
                   "question. Decide again with the same rules and reply with ONE JSON object.")


# ----------------------------------------------------------------------------------------------- fiscal years
FISCAL_WORDS = re.compile(r"\b(?:fiscal(?:\s+year)?|FY)\s*'?(\d{4}|\d{2})\b", re.IGNORECASE)
YEAR_MARGIN = 2             # a bare year this close to the indexed years is read as a fiscal year


def planning_entities(entities: Entities, question: str) -> Entities:
    """The entities with only the years the question uses as fiscal years: written as fiscal ("fiscal 2030",
    "FY2024", "FY24") or a bare year within YEAR_MARGIN of the indexed years. retrieval.entities reads any year,
    so "notes due 2030" or "the 2008 financial crisis" would otherwise be refused as R3 or searched as a fiscal
    year (out_of_corpus, a wasted replan)."""
    scope = load_scope()
    low, high = min(scope.tenk_years | scope.tenq_years) - YEAR_MARGIN, max(scope.tenk_years) + YEAR_MARGIN
    written = {int(y) if len(y) == 4 else 2000 + int(y) for y in FISCAL_WORDS.findall(question)}
    years = tuple(y for y in entities.years if y in written or low <= y <= high)
    return entities if years == entities.years else Entities(entities.tickers, years, entities.periods)


def entities_text(entities: Entities) -> str:
    """What retrieval/entities.py read from the conversation, with the periods the corpus does not index, as a
    hint (the planner decides; R3 covers unindexed periods, R2 a question about the future)."""
    def listed(values: Sequence[Any]) -> str:
        return ", ".join(str(v) for v in values) or "none named"
    companies = listed(entities.tickers)
    if not entities.tickers:
        companies += NO_COMPANY_HINT
    lines = [f"- companies (indexed tickers): {companies}",
             f"- fiscal years: {listed(entities.years)}",
             f"- fiscal periods: {listed(entities.periods)}"]
    missing = unindexed(entities)
    if missing:
        lines.append(NOT_INDEXED_LINE.format(missing=", ".join(missing)))
    return "\n".join(lines)


def unindexed(entities: Entities) -> list[str]:
    """The (fiscal year, period) pairs the question names that the corpus does not index, e.g. 'fiscal 2026 FY'."""
    return [f"fiscal {y} {p}" for y in entities.years for p in entities.periods or ("FY",) if not _indexed(y, p)]


# A question about the future (R2). Word-final forms only: "expected credit losses", "projected benefit
# obligation" and "R&D projects" are reported accounting terms, not forecasts.
FORECAST = re.compile(r"\b(?:will|would\s+be|expects?|expected\s+to|predict(?:s|ed|ion)?|forecasts?|projected\s+to|"
                      r"outlook|next\s+(?:year|quarter)|going\s+to|likely)\b", re.IGNORECASE)


def scope_check(plan: Plan, entities: Entities, question: str) -> tuple[Plan, str | None]:
    """docs/spec.md §6 from the corpus scope, not the model: a plan that answers a question all of whose fiscal
    periods are not indexed is refused, R3, or R2 when the question asks about the future (R2 before R3). Returns
    the plan and a note when the check changed it. A question that also names an indexed period is left to the
    plan (a partial refusal's factual part)."""
    missing = unindexed(entities)
    if plan.refuse or not entities.years or len(missing) < len(entities.years) * len(entities.periods or ("FY",)):
        return plan, None
    rule = "R2" if FORECAST.search(question) else "R3"
    refused = Plan(reasoning=f"Scope check: {', '.join(missing)} is not indexed, so the question is refused ({rule}).",
                   refuse=True, refusal_rule=rule, steps=[], calculation=None)
    return refused, f"planner: scope check refused the plan ({rule}: {', '.join(missing)} not indexed)"


YEAR_TO_DATE = {"Q2": "6M", "Q3": "9M"}
YTD_WORDS = re.compile(r"\b(?:six|nine)\s+months\b|\bfirst\s+half\b|\byear[\s-]+to[\s-]+date\b", re.IGNORECASE)
NAMES_AN_ITEM = re.compile(r"\bItem\s+\d", re.IGNORECASE)


SEARCH_K = 5                # search_filings' own default (ADR-017 was measured at the top 5)
FOURTH_QUARTER = re.compile(r"\bQ4\b|\b(?:fourth|4th)\s+(?:fiscal\s+)?quarter\b", re.IGNORECASE)
FORM_OF_PERIOD = {"FY": "10-K", "Q1": "10-Q", "Q2": "10-Q", "Q3": "10-Q"}


def question_period(entities: Entities) -> str | None:
    """The one fiscal period the question asks about: its quarter, FY when it names none, None when it names
    several (each step then keeps its own)."""
    if not entities.periods:
        return "FY"
    return entities.periods[0] if len(entities.periods) == 1 else None


def normalize_arguments(plan: Plan, question: str, entities: Entities) -> Plan:
    """Arguments the question and the tools decide better than the model. Measured on the development questions,
    where strict decoding (every argument required, the tools' defaults hidden) made the model write k=1 on every
    search, Q4 or Q1 for full-year figures, duration "12M" for quarters and balance-sheet items, and "Item 1" for
    MD&A questions:
    - the fiscal period is the question's (FY when it names none) on every fact and search step, and a search's form
      follows it (a form that contradicts the period is invalid_input in search_filings);
    - get_financial_fact's duration is left to the tool (12M for a year, 3M for a quarter, none for a point in time,
      spec §4.2) unless the question asks for year to date;
    - search_filings returns its own top SEARCH_K and searches every Item (ADR-017's default ranks the statements
      first) unless the question names one;
    - a compare_periods step over fewer than two years, or one that repeats figures other steps already fetch, is
      dropped."""
    if not plan.steps:
        return plan
    ytd = bool(YTD_WORDS.search(question))
    period = question_period(entities)
    fetched = {(s.arguments["ticker"], s.arguments["metric"], s.arguments["fiscal_year"])
               for s in plan.steps if s.tool == "get_financial_fact"}
    steps = []
    for step in plan.steps:
        args = dict(step.arguments)
        if step.tool == "compare_periods":
            years = args.get("years") or []
            if len(years) < 2 or all((args["ticker"], args["metric"], y) in fetched for y in years):
                continue
        if step.tool == "get_financial_fact" and period is not None:
            # Q4 is asked of the tool as Q4: it answers "not reported separately" (spec §4.2), never the year's value
            args["fiscal_period"] = "Q4" if FOURTH_QUARTER.search(question) else period
        elif step.tool == SEARCH and period is not None and indexed(args.get("fiscal_year"), period):
            args["fiscal_period"] = period
        if step.tool == "get_financial_fact":
            args["duration"] = YEAR_TO_DATE.get(args.get("fiscal_period") or "FY") if ytd else None
        elif step.tool == SEARCH:
            args["k"] = SEARCH_K
            if not NAMES_AN_ITEM.search(question):
                args["item"] = None
            if args.get("fiscal_period") is not None:
                args["form"] = FORM_OF_PERIOD[args["fiscal_period"]]
        steps.append(step if args == step.arguments else PlanStep.model_validate({**step.model_dump(),
                                                                                   "arguments": args}))
    return plan.model_copy(update={"steps": steps})


CAGR_WORDS = re.compile(r"\bcagr\b|\bcompound(?:ed)? annual", re.IGNORECASE)    # as critic_checks.OPERATION_RULES
PREFERENCE = ("growth", "ratio", "percent", "difference", "sum", "cagr")


def align_calculation(plan: Plan, question: str) -> Plan:
    """The calculation the critic will expect from the question's wording (agent/critic_checks.expected_operations:
    a margin is "percent", a change in percent "growth" or "cagr", ...), so the plan and the check agree. Where the
    wording allows several, the model's choice stands; a missing or unexpected one becomes the preferred: cagr only
    when the question says so, else growth, ratio before percent."""
    expected = expected_operations(question)
    if not plan.steps or not expected or plan.calculation in expected:
        return plan
    if "cagr" in expected and CAGR_WORDS.search(question):
        choice = "cagr"
    else:
        choice = next(op for op in PREFERENCE if op in expected)
    return plan.model_copy(update={"calculation": choice})


def finalize(plan: Plan, entities: Entities, question: str) -> tuple[Plan, list[str]]:
    """The model's plan after the deterministic rules: the scope check, the entities in every search step, the
    arguments the question and the tools decide, then the expected calculation. `entities` are the planning
    entities (planning_entities)."""
    plan, note = scope_check(plan, entities, question)
    notes = [note] if note else []
    plan, cut = apply_entities(plan, entities)
    if cut:
        notes.append(f"planner: {cut} search step(s) beyond the {MAX_PLAN_STEPS}-step cap were dropped")
    return align_calculation(normalize_arguments(plan, question, entities), question), notes


def _indexed(year: int, period: str) -> bool:
    """Whether the corpus has the filing: a 10-K for FY, a 10-Q for Q1-Q3."""
    if period == "FY":
        return year in load_scope().tenk_years
    return indexed(year, period)


def _error_lines(results: Sequence[ToolResult]) -> list[str]:
    return [f"- {r.call.name}({json.dumps(r.call.arguments, sort_keys=True)}) -> {r.error_type}: "
            f"{str(r.output['error'].get('message', ''))[:200]}" for r in results if r.error_type]


def user_message(state: AgentState, entities: Entities) -> str:
    """The question with the earlier turns, the entities read from them and, on a replan, the previous steps and
    the tool errors."""
    parts = []
    history = state.get("history") or []
    if history:
        parts.append("Earlier questions in this conversation (oldest first):\n" + "\n".join(f"- {q}" for q in history))
    parts.append(f"Question: {state['question']}")
    parts.append("Read from the conversation:\n" + entities_text(entities))
    previous, errors = state.get("plan"), _error_lines(state.get("tool_results") or [])
    if previous is not None and errors:
        steps = [{"tool": s.tool, "arguments": s.arguments} for s in previous.steps]
        parts.append(REPLAN_TEXT.format(steps=json.dumps(steps), errors="\n".join(errors)))
    return "\n\n".join(parts)


def messages(state: AgentState, entities: Entities) -> list[Message]:
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user_message(state, entities)}]


# ----------------------------------------------------------------------------------------------- entities
def _filled(step: PlanStep, field: str, values: Sequence[Any]) -> list[PlanStep]:
    """The step with `field` set to each of `values` when the step leaves it empty (one step per value)."""
    if step.arguments.get(field) is not None or not values:
        return [step]
    return [step.model_copy(update={"arguments": {**step.arguments, field: v}}) for v in values]


def apply_entities(plan: Plan, entities: Entities) -> tuple[Plan, int]:
    """Every search_filings step searches with the companies, fiscal years and period read from the question
    (ADR-017: question-derived filters are the biggest retrieval gain), filled in where the step leaves them empty:
    one step per company (and per year) when the question names several. A period the corpus does not index for
    that year is not set (Entities.filters drops it the same way); normalize_arguments then makes the form follow
    the period. Returns the plan and how many steps the MAX_PLAN_STEPS cap dropped."""
    if plan.refuse or not plan.steps:
        return plan, 0
    steps: list[PlanStep] = []
    for step in plan.steps:
        if step.tool != SEARCH:
            steps.append(step)
            continue
        for s in _filled(step, "ticker", entities.tickers):
            for s2 in _filled(s, "fiscal_year", entities.years):
                period = entities.periods[0] if len(entities.periods) == 1 else None
                year = s2.arguments.get("fiscal_year")
                if period and s2.arguments.get("fiscal_period") is None and indexed(year, period):
                    s2 = s2.model_copy(update={"arguments": {**s2.arguments, "fiscal_period": period}})
                steps.append(PlanStep.model_validate(s2.model_dump()))       # re-validated against SearchArgs
    unique = list({json.dumps(s.model_dump(), sort_keys=True): s for s in steps}.values())
    return plan.model_copy(update={"steps": unique[:MAX_PLAN_STEPS]}), max(0, len(unique) - MAX_PLAN_STEPS)


# ----------------------------------------------------------------------------------------------- scope facts
LIVE_OR_NEWS = re.compile(r"\b(?:price|trading|today|right\s+now|currently|news|latest\s+news)\b", re.IGNORECASE)


def scope_contradiction(plan: Plan, entities: Entities, question: str) -> str | None:
    """A correction when the model refused R3 although every company and period read from the conversation is
    indexed (measured: "What about fiscal 2025?" after a Pfizer question was refused as "fiscal year 2025 is not
    indexed"). Not for a question about the future (R2 territory), live prices or news. The correction says only
    what the scope shows: a company the question names that retrieval.entities does not know (not indexed) is
    still R3, so a correct refusal ("Apple's and Tesla's revenue") is kept by the model. None when consistent."""
    if not (plan.refuse and plan.refusal_rule == "R3" and entities.tickers and entities.years):
        return None
    if unindexed(entities) or LIVE_OR_NEWS.search(question) or FORECAST.search(question):
        return None
    periods = ", ".join(f"fiscal {y} {p}" for y in entities.years for p in entities.periods or ("FY",))
    years = ", ".join(str(y) for y in sorted(load_scope().tenk_years))
    return CORRECTION_TEXT.format(tickers=", ".join(entities.tickers), periods=periods, years=years)


# ----------------------------------------------------------------------------------------------- planning
@dataclass
class PlanResult:
    """One question's planning: the final plan (None when the call failed), notes for the trace, and the cost."""
    plan: Plan | None
    notes: list[str]
    entities: Entities                  # the planning entities (only the years used as fiscal years)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cached: bool = True
    retries: int = 0
    error: str | None = None

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, reply: GuidedReply) -> None:
        self.prompt_tokens += reply.prompt_tokens
        self.completion_tokens += reply.completion_tokens
        self.seconds += reply.seconds
        self.cached = self.cached and reply.cached
        self.retries += reply.retries


async def plan_question(llm: CompletionClient, state: AgentState, entities: Entities, *,
                        schema: dict[str, Any] | None = None, llm_model: str | None = None) -> PlanResult:
    """The planner's work for one question (the node and eval/run_planner.py both use it): one guided call; one
    more when its R3 refusal contradicts the corpus scope; then the deterministic rules (finalize)."""
    schema = schema or plan_json_schema()
    entities = planning_entities(entities, " ".join([*(state.get("history") or []), state["question"]]))
    working = messages(state, entities)
    reply = await ask_guided(llm, working, Plan, schema, llm_model=llm_model)
    result = PlanResult(None, [], entities)
    result.add(reply)
    if reply.parsed is None:
        result.error = reply.error
        return result
    if reply.retries:
        result.notes.append(f"planner: valid plan after {reply.retries} JSON retry")
    plan = typing.cast(Plan, reply.parsed)
    correction = scope_contradiction(plan, entities, state["question"])
    if correction:
        again = await ask_guided(llm, [*working, {"role": "assistant", "content": plan.model_dump_json()},
                                       {"role": "user", "content": correction}], Plan, schema, llm_model=llm_model)
        result.add(again)
        if again.parsed is not None:
            plan = again.parsed
        result.notes.append("planner: an R3 refusal contradicted the indexed scope; asked again"
                            + ("" if again.parsed is not None else f" ({again.error}); kept the refusal"))
    result.plan, notes = finalize(plan, entities, state["question"])
    result.notes += notes
    return result


# ----------------------------------------------------------------------------------------------- node
def make_planner(llm: CompletionClient, *, llm_model: str | None = None) -> Any:
    """The planner node for agent.graph.Nodes(planner=...)."""
    schema = plan_json_schema()

    async def planner(state: AgentState) -> dict[str, Any]:
        entities = from_turns([*(state.get("history") or []), state["question"]])
        result = await plan_question(llm, state, entities, schema=schema, llm_model=llm_model)
        update: dict[str, Any] = {"entities": result.entities, "tokens": result.tokens, "plan": result.plan}
        notes = result.notes if result.plan is not None else [f"planner: {result.error}"]
        if notes:
            update["errors"] = notes
        return update

    return planner
