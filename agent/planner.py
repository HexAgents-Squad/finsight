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
or the refusal tests; eval/run_planner.py) and pinned: PROMPT_SHA256 is prompt_sha256(), over everything the model
sees (the system prompt, the decoding schema, every message template); a test fails when any of them changes (a
prompt edit, a tool's argument model, the metrics, the scope). The rules after the model are tested, and each run
records rules_sha256(). Re-tune after a change to what the model sees, then re-pin.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import io
import json
import re
import tokenize
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
from retrieval.entities import Entities, extract, from_turns, indexed
from tools.registry import TOOLS

PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "planner.txt"
PROMPT_VERSION = "planner-v2"                 # W3-04: the four new metrics (ADR-040)
PROMPT_SHA256 = "c4e5c45a25ae71b6a4834b1b5c82c8aef3d83825d1a1c450492b9eca82ddaebe"   # planner-v2: model inputs of P11 (live), P12 and P13 (replayed; P13 the final rules)
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


def _sha256(parts: Sequence[str]) -> str:
    return hashlib.sha256("\n\0\n".join(parts).encode("utf-8")).hexdigest()


def prompt_sha256() -> str:
    """The planner's pin: everything the model is shown, so an unchanged pin means unchanged requests (and cached
    replies replay exactly): the system prompt, the decoding schema, every message template and the JSON re-ask.
    The rules that correct a plan after the model are code, covered by tests, and recorded per run by
    rules_sha256() (eval/run_planner.py), so two runs with the same pin but different rules are told apart."""
    from agent.guided import RETRY_INSTRUCTION
    return _sha256([system_prompt(), json.dumps(plan_json_schema(), sort_keys=True), RETRY_INSTRUCTION,
                    *MESSAGE_TEMPLATES])


REPO_ROOT = Path(__file__).resolve().parents[1]
RULE_SOURCES = ("agent/planner.py", "agent/guided.py", "agent/critic_checks.py", "retrieval/entities.py",
                "agent/critic.py", "agent/critic_semantic.py")   # W3-04: search_only_steps reads the critic's rules


SKIPPED_TOKENS = frozenset({tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER})
STRUCTURE = {tokenize.NEWLINE: "<nl>", tokenize.INDENT: "<in>", tokenize.DEDENT: "<de>"}


def _code(source: str) -> str:
    """The source's code as its tokens, without docstrings, comments, layout (line breaks inside brackets, the width
    of an indent) or the pin's value, so editing a docstring or a comment does not change rules_sha256 (it did once,
    between run P8 and the merge). Tokens, not ast.dump: the token text is the same on Python 3.12-3.14 (CI runs
    3.12), ast.dump's output changed in 3.13. Docstrings are found by their position in the syntax tree."""
    docstrings = set()
    for node in ast.walk(ast.parse(source)):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            docstrings.add((body[0].lineno, body[0].col_offset))
    out: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in SKIPPED_TOKENS or (tok.type == tokenize.STRING and tok.start in docstrings):
            continue
        out.append(STRUCTURE.get(tok.type, tok.string))
    text = " ".join(out)
    return re.sub(r'PROMPT_SHA256 = "[^"]*"', 'PROMPT_SHA256 = ""', text)


def rules_sha256() -> str:
    """The code that turns the model's reply into the final plan (RULE_SOURCES' tokens, see _code)."""
    return _sha256([_code((REPO_ROOT / name).read_text(encoding="utf-8")) for name in RULE_SOURCES])


# Message templates: every literal of the user messages, pinned with the prompt and kept verbatim
HISTORY_HEADER = "Earlier questions in this conversation (oldest first):\n"
QUESTION_LINE = "Question: {question}"
ENTITIES_HEADER = "Read from the conversation:\n"
COMPANIES_LINE = "- companies (indexed tickers): {companies}"
YEARS_LINE = "- fiscal years: {years}"
PERIODS_LINE = "- fiscal periods: {periods}"
NONE_NAMED = "none named"
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
MESSAGE_TEMPLATES = (HISTORY_HEADER, QUESTION_LINE, ENTITIES_HEADER, COMPANIES_LINE, YEARS_LINE, PERIODS_LINE,
                     NONE_NAMED, NO_COMPANY_HINT, NOT_INDEXED_LINE, REPLAN_TEXT, CORRECTION_TEXT)


# ----------------------------------------------------------------------------------------------- fiscal years
FISCAL_WORDS = re.compile(r"\b(?:fiscal(?:\s+years?)?|FY)\s*'?(\d{4}|\d{2})\b", re.IGNORECASE)
# A year in these contexts is a date in the text, not the fiscal year asked about (a bare year otherwise reads as
# the fiscal-year label, spec §4.1 rule 2, so "Apple's revenue in 2020" is still R3)
NOT_A_FISCAL_YEAR = (r"\b(?:due|by|until|through|maturing(?:\s+in)?|before)\s+{y}\b",
                     r"\b{y}(?:\s+[\w-]+){{0,2}}\s+(?:goals?|targets?|commitments?|crisis|pledges?|plans?|deadlines?|"
                     r"notes|bonds|debentures)\b")


def planning_entities(entities: Entities, text: str) -> Entities:
    """The entities without the years `text` (the conversation) uses as dates rather than fiscal years ("notes due
    2030", "the 2008 financial crisis", "its 2030 goal"), which would otherwise be refused as R3 or searched as a
    fiscal year (out_of_corpus, a wasted replan). A year written as fiscal ("fiscal 2030", "FY24") is always kept."""
    written = {int(y) if len(y) == 4 else 2000 + int(y) for y in FISCAL_WORDS.findall(text)}
    years = tuple(y for y in entities.years if y in written
                  or not any(re.search(p.format(y=y), text, re.IGNORECASE) for p in NOT_A_FISCAL_YEAR))
    if years == entities.years:
        return entities
    if not years:                   # "latest 10-K ... notes due 2030": without the date, "latest" names the year
        dates = "|".join(str(y) for y in entities.years)
        years = extract(re.sub(rf"\b(?:{dates})\b", "", text)).years
    return Entities(entities.tickers, years, entities.periods)


def _indexed(year: int | None, period: str | None) -> bool:
    """Whether the corpus has the filing: a 10-K for FY (and Q4, which the 10-K covers), a 10-Q for Q1-Q3; a step
    without a year is not checked."""
    if year is None:
        return True
    if period in (None, "FY", "Q4"):
        return year in load_scope().tenk_years or (period is None and year in load_scope().tenq_years)
    return indexed(year, period)


def unindexed(entities: Entities) -> list[str]:
    """The (fiscal year, period) pairs the question names that the corpus does not index, e.g. 'fiscal 2026 FY'."""
    return [f"fiscal {y} {p}" for y in entities.years for p in entities.periods or ("FY",) if not _indexed(y, p)]


def entities_text(entities: Entities) -> str:
    """What retrieval/entities.py read from the conversation, with the periods the corpus does not index, as a
    hint (the planner decides; R3 covers unindexed periods, R2 a question about the future)."""
    def listed(values: Sequence[Any]) -> str:
        return ", ".join(str(v) for v in values) or NONE_NAMED
    companies = listed(entities.tickers) + ("" if entities.tickers else NO_COMPANY_HINT)
    lines = [COMPANIES_LINE.format(companies=companies), YEARS_LINE.format(years=listed(entities.years)),
             PERIODS_LINE.format(periods=listed(entities.periods))]
    missing = unindexed(entities)
    if missing:
        lines.append(NOT_INDEXED_LINE.format(missing=", ".join(missing)))
    return "\n".join(lines)


def _error_lines(results: Sequence[ToolResult]) -> list[str]:
    return [f"- {r.call.name}({json.dumps(r.call.arguments, sort_keys=True)}) -> {r.error_type}: "
            f"{str(r.output['error'].get('message', ''))[:200]}" for r in results if r.error_type]


def user_message(state: AgentState, entities: Entities) -> str:
    """The question with the earlier turns, the entities read from them and, on a replan, the previous steps and
    the tool errors."""
    parts = []
    history = state.get("history") or []
    if history:
        parts.append(HISTORY_HEADER + "\n".join(f"- {q}" for q in history))
    parts.append(QUESTION_LINE.format(question=state["question"]))
    parts.append(ENTITIES_HEADER + entities_text(entities))
    previous, errors = state.get("plan"), _error_lines(state.get("tool_results") or [])
    if previous is not None and errors:
        steps = [{"tool": s.tool, "arguments": s.arguments} for s in previous.steps]
        parts.append(REPLAN_TEXT.format(steps=json.dumps(steps), errors="\n".join(errors)))
    return "\n\n".join(parts)


def messages(state: AgentState, entities: Entities) -> list[Message]:
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user_message(state, entities)}]


# ----------------------------------------------------------------------------------------------- scope rules
# A question about the future (R2). Word-final forms only: "expected credit losses", "projected benefit
# obligation" and "R&D projects" are reported accounting terms, not forecasts.
FORECAST = re.compile(r"\b(?:will|would\s+be|expects?|expected\s+to|predict(?:s|ed|ion)?|forecasts?|projected\s+to|"
                      r"outlook|next\s+(?:year|quarter)|going\s+to|likely)\b", re.IGNORECASE)
LIVE_OR_NEWS = re.compile(r"\b(?:price|trading|today|right\s+now|currently|news|latest\s+news)\b", re.IGNORECASE)


def _refusal(reason: str, rule: str) -> Plan:
    return Plan(reasoning=f"Scope check: {reason}, so the question is refused ({rule}).", refuse=True,
                refusal_rule=rule, steps=[], calculation=None)


def scope_check(plan: Plan, entities: Entities, question: str) -> tuple[Plan, str | None]:
    """docs/spec.md §6 from the corpus scope, not the model: a plan that answers a question all of whose fiscal
    periods are not indexed is refused, R3, or R2 when the question asks about the future (R2 before R3). A question
    that also names an indexed period is left to the plan; step_scope then drops the unindexed steps."""
    missing = unindexed(entities)
    if plan.refuse or not entities.years or len(missing) < len(entities.years) * len(entities.periods or ("FY",)):
        return plan, None
    rule = "R2" if FORECAST.search(question) else "R3"
    return (_refusal(f"{', '.join(missing)} is not indexed", rule),
            f"planner: scope check refused the plan ({rule}: {', '.join(missing)} not indexed)")


def company_scope(plan: Plan, entities: Entities, question: str) -> tuple[Plan, str | None]:
    """docs/spec.md §6 R3 from the corpus scope, for companies as scope_check is for periods: a plan whose every step
    asks for a ticker outside config/companies.yaml, for a conversation that names no indexed company, is refused
    (R2 when the question asks about the future). Measured (W3-04, P9): "Did Starbucks report higher free cash flow
    ...?" was planned as two SBUX facts. A conversation that names an indexed company is left to the plan (a
    mistyped ticker for it is the tool's out_of_corpus and the replan's), and so is a ticker that names an indexed
    company another way (GOOG for GOOGL: retrieval.entities reads it as that company)."""
    if plan.refuse or entities.tickers or not plan.steps:
        return plan, None
    indexed = {c.ticker for c in load_scope().companies}
    tickers = [str(s.arguments.get("ticker") or "").strip().upper() for s in plan.steps]
    if any(not t or t in indexed or extract(t).tickers for t in tickers):
        return plan, None
    rule = "R2" if FORECAST.search(question) else "R3"
    names = ", ".join(dict.fromkeys(tickers))
    return (_refusal(f"{names} is not an indexed company", rule),
            f"planner: company scope check refused the plan ({rule}: {names} not indexed)")


def search_only_steps(plan: Plan, question: str, history: Sequence[str] = ()) -> tuple[Plan, str | None]:
    """The critic's rule for a search-only line item (agent.critic.SEARCH_ONLY, e.g. stock-based compensation) applied
    to the plan: a fact step for a metric the question does not name is the substitute the critic rejects, so it
    becomes a search for the line item in that company's filing for each period it asked for (a named metric, e.g.
    the revenue a share of revenue divides by, or net income named with its holders, stays). The turn read is the
    critic's (metric_question_text: a follow-up's earlier turn). Measured (W3-04, P11): "What stock-based compensation
    expense did Meta report ...?" was planned as Meta's stockholders_equity."""
    from agent.critic import metric_question_text, search_only_line_item
    from agent.critic_semantic import question_metrics
    text = metric_question_text(question, list(history))
    item = search_only_line_item(text)
    if plan.refuse or not item:
        return plan, None
    named = question_metrics(text)
    steps, replaced = [], []
    for step in plan.steps:
        args = step.arguments
        if step.tool not in ("get_financial_fact", "compare_periods") or args.get("metric") in named:
            steps.append(step)
            continue
        replaced.append(str(args.get("metric")))
        period = str(args.get("fiscal_period") or "").strip().upper()
        for year in args.get("years") or [args.get("fiscal_year")]:      # compare_periods: one search per year
            steps.append(PlanStep.model_validate({"tool": "search_filings", "purpose": item, "arguments": {
                "query": item, "ticker": args.get("ticker"), "fiscal_year": year,
                "fiscal_period": period if period in FORM_OF_PERIOD else None,   # a search has no Q4 filter
                "form": None, "item": None, "k": SEARCH_K}}))
    if not replaced:
        return plan, None
    calculation = plan.calculation if any(s.tool != "search_filings" for s in steps) else None
    return (plan.model_copy(update={"steps": steps, "calculation": calculation}),
            f"planner: {', '.join(dict.fromkeys(replaced))} replaced by a search for {item} (a search-only line item)")


def step_scope(plan: Plan, question: str) -> tuple[Plan, str | None]:
    """Steps that ask for a filing the corpus does not index (a fact or search for an unindexed fiscal year or
    quarter; compare_periods years outside the 10-K years) are dropped: the tool would answer out_of_corpus and use
    up the one replan. What remains is answered and the rest declined (a partial refusal, R3, or R2 for a question
    about the future); nothing left is a refusal."""
    if plan.refuse or not plan.steps:
        return plan, None
    tenk = load_scope().tenk_years
    kept, dropped = [], []
    for step in plan.steps:
        args = step.arguments
        if step.tool == "compare_periods":
            years = [y for y in args["years"] if y in tenk]
            if len(years) < len(args["years"]):
                dropped.append(f"fiscal {', '.join(str(y) for y in args['years'] if y not in tenk)}")
                if len(years) < 2:
                    continue
                step = PlanStep.model_validate({**step.model_dump(), "arguments": {**args, "years": years}})
        elif not _indexed(args.get("fiscal_year"), args.get("fiscal_period")):
            dropped.append(f"fiscal {args.get('fiscal_year')} {args.get('fiscal_period') or ''}".strip())
            continue
        kept.append(step)
    if not dropped:
        return plan, None
    rule = plan.refusal_rule or ("R2" if FORECAST.search(question) else "R3")
    what = ", ".join(dict.fromkeys(dropped))
    note = f"planner: steps for {what} dropped (not indexed)"
    if not kept:
        return _refusal(f"{what} is not indexed", rule), note
    return plan.model_copy(update={"steps": kept, "refusal_rule": rule}), note + f"; the rest answered ({rule})"


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


# ----------------------------------------------------------------------------------------------- arguments
SEARCH_K = 5                # search_filings' own default (ADR-017 was measured at the top 5)
FORM_OF_PERIOD = {"FY": "10-K", "Q1": "10-Q", "Q2": "10-Q", "Q3": "10-Q"}
YEAR_TO_DATE = {"Q2": "6M", "Q3": "9M"}
YTD_WORDS = re.compile(r"\b(?:six|nine)[\s-]+months?\b|\bfirst[\s-]+half\b|\bH1\b|\byear[\s-]+to[\s-]+date\b|\bYTD\b",
                       re.IGNORECASE)
# Any sign that the conversation asks about part of a year: then the model's period stands (the question may mix
# periods, "fourth-quarter and full-year", "end of fiscal 2024 to Q3 of fiscal 2025"); with none, it is the year
PERIOD_CUE = re.compile(r"\bQ[1-4]\b|\b(?:first|second|third|fourth|1st|2nd|3rd|4th)[\s-]+(?:fiscal[\s-]+)?quarter|"
                        r"\bquarter(?:s|ly)?\b|\b(?:three|six|nine)[\s-]+months?\b|\bhalf\b|\bH[12]\b|"
                        r"\byear[\s-]+to[\s-]+date\b|\bYTD\b|\b10-Q\b", re.IGNORECASE)
NAMES_AN_ITEM = re.compile(r"\bItem\s+\d", re.IGNORECASE)
# Management's discussion and analysis, where a filing explains what drove a change, as the index names its Item
MDNA_ITEM = {"10-K": "Item 7", "10-Q": "Part I Item 2"}


def cue_text(question: str, history: Sequence[str] = ()) -> str:
    """The text whose period cues count: the question, or the whole conversation for a follow-up that names no year
    or period of its own ("And its operating income?" after a fourth-quarter question)."""
    named = extract(question)
    return question if named.years or named.periods else " ".join([*history, question])


def normalize_arguments(plan: Plan, text: str) -> Plan:
    """Arguments the question and the tools decide better than the model. Strict decoding makes every argument
    required and hides the tools' defaults; measured on the development questions, the model then wrote k=1 on every
    search, Q4 or Q1 for full-year figures, duration "12M" for quarters and balance-sheet items, and "Item 1" for
    MD&A questions. `text` is cue_text():
    - with no period cue in it, every fact and search is for the fiscal year (FY); with one, the model's period
      stands, and a search whose period the corpus does not index for its year searches the whole year;
    - a search's form follows its period (a form that contradicts it is invalid_input in search_filings);
    - get_financial_fact's duration is left to the tool (12M, 3M or a point in time, spec §4.2) unless the
      question asks for year to date;
    - search_filings returns its own top SEARCH_K and searches every Item unless the question names one;
    - a compare_periods step over fewer than two years, or one that repeats fetched figures, is dropped."""
    if not plan.steps:
        return plan
    whole_year, ytd = not PERIOD_CUE.search(text), bool(YTD_WORDS.search(text))
    fetched = {(s.arguments["ticker"], s.arguments["metric"], s.arguments["fiscal_year"])
               for s in plan.steps if s.tool == "get_financial_fact"}
    steps = []
    for step in plan.steps:
        args = dict(step.arguments)
        if step.tool == "compare_periods":
            years = args.get("years") or []
            if len(years) < 2 or all((args["ticker"], args["metric"], y) in fetched for y in years):
                continue
        elif step.tool == "get_financial_fact":
            if whole_year:
                args["fiscal_period"] = "FY"
            args["duration"] = YEAR_TO_DATE.get(args.get("fiscal_period") or "FY") if ytd else None
        else:
            if whole_year:
                args["fiscal_period"] = "FY" if _indexed(args.get("fiscal_year"), "FY") else None
            elif args.get("fiscal_period") and not _indexed(args.get("fiscal_year"), args["fiscal_period"]):
                args["fiscal_period"] = None
            args["form"] = FORM_OF_PERIOD.get(args.get("fiscal_period") or "", None) if args.get("fiscal_period") \
                else None
            args["k"] = SEARCH_K
            if not NAMES_AN_ITEM.search(text):
                args["item"] = None
        steps.append(step if args == step.arguments else PlanStep.model_validate({**step.model_dump(),
                                                                                   "arguments": args}))
    return plan.model_copy(update={"steps": steps})


def mdna_searches(plan: Plan, question: str) -> tuple[Plan, str | None]:
    """A "what drove X" question (agent.critic_checks.DRIVERS) is answered from the MD&A of the asked period: each
    search with a form (normalize_arguments sets it from the period) searches that form's MD&A Item (MDNA_ITEM), and
    its query names X (agent.critic_semantic.x_words, in the question's order) when the model's query left a word of
    it out. Measured (W3-05, development questions): unfiltered, the passage that states X's change ranked first for
    two of ten questions and was missing for one, among statement tables and notes (Item 8); with the MD&A filter it
    ranked first for eight and third for two. A question that names an Item keeps it (normalize_arguments)."""
    from agent.critic_checks import DRIVERS
    from agent.critic_semantic import WORD, x_words
    if plan.refuse or not DRIVERS.search(question) or NAMES_AN_ITEM.search(question):
        return plan, None
    words = x_words(question)
    subject = list(dict.fromkeys(w for w in WORD.findall(question.lower().replace("'s", " ")) if w in words))
    steps, changed = [], False
    for step in plan.steps:
        args = dict(step.arguments)
        if step.tool == SEARCH and args.get("form") in MDNA_ITEM:
            args["item"] = MDNA_ITEM[args["form"]]
            asked = set(WORD.findall(str(args.get("query") or "").lower()))
            missing = [w for w in subject if w not in asked]
            if missing:
                args["query"] = f"{args.get('query') or ''} {' '.join(missing)}".strip()
        if args != step.arguments:
            changed = True
            step = PlanStep.model_validate({**step.model_dump(), "arguments": args})
        steps.append(step)
    if not changed:
        return plan, None
    return plan.model_copy(update={"steps": steps}), "planner: the drivers question's searches go to the MD&A"


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


# ----------------------------------------------------------------------------------------------- entities
def _filled(step: PlanStep, field: str, values: Sequence[Any]) -> list[PlanStep]:
    """The step with `field` set to each of `values` when the step leaves it empty (one step per value)."""
    if step.arguments.get(field) is not None or not values:
        return [step]
    return [step.model_copy(update={"arguments": {**step.arguments, field: v}}) for v in values]


def _with(step: PlanStep, **arguments: Any) -> PlanStep:
    return step.model_copy(update={"arguments": {**step.arguments, **arguments}})


def apply_entities(plan: Plan, entities: Entities) -> Plan:
    """Every search_filings step searches the companies, fiscal years and period read from the question (ADR-017:
    question-derived filters are the biggest retrieval gain). Companies are the question's: a step with no company,
    or with one the question does not name, searches each of the question's companies, and every search query covers
    all of them ("risks Apple and Microsoft describe" searches both even when the model wrote one). Years and the
    period fill only what a step leaves empty (a step may need another year, e.g. "compared with the prior year")."""
    if plan.refuse or not plan.steps:
        return plan
    tickers = entities.tickers
    steps: list[PlanStep] = []
    for step in plan.steps:
        if step.tool != SEARCH:
            steps.append(step)
            continue
        ticker = step.arguments.get("ticker")
        targets = [_with(step, ticker=t) for t in tickers] if tickers and ticker not in tickers else [step]
        for s in targets:
            for s2 in _filled(s, "fiscal_year", entities.years):
                period = entities.periods[0] if len(entities.periods) == 1 else None
                if period and s2.arguments.get("fiscal_period") is None:
                    s2 = _with(s2, fiscal_period=period)
                steps.append(s2)
    if tickers:                                         # each query searches every company the question names
        for query in dict.fromkeys(s.arguments["query"] for s in steps if s.tool == SEARCH):
            group = [s for s in steps if s.tool == SEARCH and s.arguments["query"] == query]
            covered = {s.arguments.get("ticker") for s in group}
            steps += [_with(group[0], ticker=t) for t in tickers if t not in covered]
    return plan.model_copy(update={"steps": [PlanStep.model_validate(s.model_dump()) for s in steps]})


def finalize(plan: Plan, entities: Entities, question: str, history: Sequence[str] = ()) -> tuple[Plan, list[str]]:
    """The model's plan after the deterministic rules, in order: the question-level scope checks (periods, then
    companies), a search-only line item's substitutes replaced by a search, the question's
    companies in every search, the arguments the question and the tools decide, a drivers question's searches in the
    MD&A, the step-level scope check, the
    expected calculation; then duplicate steps removed, the step cap applied, and the plan validated again (an
    emptied partial refusal becomes a refusal). `entities` are the planning entities (planning_entities)."""
    plan, note = scope_check(plan, entities, question)
    notes = [note] if note else []
    plan, note = company_scope(plan, entities, question)
    notes += [note] if note else []
    plan, note = search_only_steps(plan, question, history)
    notes += [note] if note else []
    plan = normalize_arguments(apply_entities(plan, entities), cue_text(question, history))
    plan, note = mdna_searches(plan, question)
    notes += [note] if note else []
    plan, note = step_scope(plan, question)
    notes += [note] if note else []
    plan = align_calculation(plan, question)
    steps = list({json.dumps(s.model_dump(), sort_keys=True): s for s in plan.steps}.values())
    if len(steps) > MAX_PLAN_STEPS:
        notes.append(f"planner: {len(steps) - MAX_PLAN_STEPS} step(s) beyond the {MAX_PLAN_STEPS}-step cap were "
                     "dropped")
        steps = steps[:MAX_PLAN_STEPS]
    update: dict[str, Any] = {"steps": steps}
    if plan.partial and not steps:
        update = {"refuse": True, "steps": [], "calculation": None}
    elif not steps:
        update["calculation"] = None
    return Plan.model_validate(plan.model_copy(update=update).model_dump()), notes


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
            plan = typing.cast(Plan, again.parsed)
        result.notes.append("planner: an R3 refusal contradicted the indexed scope; asked again"
                            + ("" if again.parsed is not None else f" ({again.error}); kept the refusal"))
    result.plan, notes = finalize(plan, entities, state["question"], state.get("history") or [])
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
