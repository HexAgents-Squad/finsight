"""The critic's checks of what the draft answers against what the question asks (W3-03, ADR-039): no LLM, no network;
the companies file and the corpus scope are read once (cached).

    semantic_violations(question, history, draft, results)  -> [Violation]

A1's critic checked the contract (agent/critic_checks.py) and units, and caught none of A1's wrong answers on gold:
they were well formed, but about another line item, a company total for a segment, a refusal under the wrong rule.
These checks read the question against the draft and the tools, one class of wrong answer each:

    line_item       the question asks for a line item A1 had to substitute (deposits, noninterest or operating
                    expenses, current assets, liabilities or ratio, the medical care ratio): the answer must state it,
                    and no fact for another metric may stand in for it, unless the answer says the asked item is not
                    available and names what it gives instead. Since W3-04 the facts tool has deposits, noninterest
                    expense and current assets and liabilities (ADR-040): their facts are the item, and a current
                    ratio's are its two parts
    inputs          a calculation's fetched inputs are each named by the question or are the denominator its
                    operation implies (revenue for a margin or a share of revenue); one named input no longer lets
                    any other through
    net_sales       a question about net sales is not answered with the total-revenue fact of a company that reports
                    net sales apart from membership and other income (Costco, Walmart)
    segment         a question about a segment (AWS, Intelligent Cloud, ...) is not answered with company totals, also
                    when a follow-up ("its operating margin") asks about the segment its earlier turn named
    not_found       an answer that the asked item was not found or is not reported, while a retrieved passage gives
                    that item, as its own line, with a figure in the same clause (not a year or a Note number)
    summary         a "what drove X" answer cites a passage that states X's change, and states that change's size
                    when the passage gives one
    refusal_rule    a question that also asks for a fact is not refused whole (a partial refusal, spec §6); an
                    instruction to ignore the rules or change role is R6; tax or legal advice is R7, not R5; a forecast
                    is R2, not R3 (spec §6: R2 before R3)

The period and duration of each value are already held to the question by the contract (a Value's duration fits its
fiscal period) and by invariant 4 (the question's years and periods), so they need no check here.

The vocabularies below are general filing and policy terms (the concept map's gaps, the spec's rules), not phrases of any
gold question. The graph sends a failing draft back once with the messages (spec §9), and the composer rewrites it from
the evidence it already has (the plan is not run again; a refusal is rebuilt from the plan): so each message asks for
what the composer can do, state the asked figure from the passages if they give it, or say it is not available, never
present another figure as the one asked. The checks read a passage as the composer saw it (composer.shown_text). A
refusal under the wrong rule cannot be rewritten that way: it leaves unverified, which is the point (it is never
verified); repairing it needs a replan (A5).
"""
from __future__ import annotations

import functools
import re
from collections.abc import Sequence
from typing import Any

from agent.contract import AnswerContract
from agent.critic_checks import DRIVERS, Violation, expected_operations
from core.figures import figures

# ----------------------------------------------------------------------------------------------- vocabularies
# Line items A1 substituted for, as a question or an answer words them (W3-03); the facts tool's own metrics for them
# (W3-04) count as the item through question_metrics and IMPLIED_METRICS
LINE_ITEMS: dict[str, re.Pattern] = {
    "deposits": re.compile(r"\bdeposits\b", re.IGNORECASE),
    "noninterest expense": re.compile(r"\bnon-?interest expenses?\b", re.IGNORECASE),
    "operating expenses": re.compile(r"\b(?:total )?operating expenses\b", re.IGNORECASE),
    "current ratio": re.compile(r"\bcurrent ratio\b", re.IGNORECASE),
    "current assets": re.compile(r"\b(?:total )?current assets\b", re.IGNORECASE),
    "current liabilities": re.compile(r"\b(?:total )?current liabilities\b", re.IGNORECASE),
    "medical care ratio": re.compile(r"\bmedical (?:care|cost|loss) ratio\b|\bMCR\b", re.IGNORECASE),
}
# The facts tool's metrics that are the item, or that it is made of, however the question words it (named_metrics
# reads only the concept map's labels: "noninterest expenses", "non-interest expense" or bare "operating expenses"
# are the same line; a company without it gets not_reported, never another metric)
IMPLIED_METRICS: dict[str, frozenset[str]] = {
    "noninterest expense": frozenset({"noninterest_expense"}),
    "operating expenses": frozenset({"noninterest_expense"}),
    "current ratio": frozenset({"current_assets", "current_liabilities"}),
}
# "operating expenses" inside a longer name that is another line item
LINE_ITEM_MASKS = (re.compile(r"\bnon-?interest expenses?\b", re.IGNORECASE),)
# Segments of the indexed companies, as their 10-Ks name them (the facts tool returns company totals only)
SEGMENT = re.compile(r"\bAWS\b|\bAmazon Web Services\b|\bIntelligent Cloud\b|\bProductivity and Business "
                     r"Processes\b|\bMore Personal Computing\b|\bGoogle Cloud\b|\bGoogle Services\b|\bFamily of Apps\b|"
                     r"\bReality Labs\b|\bOptum\b|\bUnitedHealthcare\b|\bInnovative Medicine\b|\bMedTech\b|"
                     r"\bCompute (?:&|and) Networking\b|\bGlobal Banking (?:&|and) Markets\b|\bAsset (?:&|and) Wealth "
                     r"Management\b|\bPlatform Solutions\b", re.IGNORECASE)
NET_SALES = re.compile(r"\bnet sales\b", re.IGNORECASE)
TOTAL_REVENUE_CONCEPT = "us-gaap:Revenues"
# Companies whose income statement reports net sales apart from membership and other income, so that their total
# revenues fact is not net sales (Costco: net sales + membership fees; Walmart: net sales + membership and other income)
SEPARATE_NET_SALES = frozenset({"COST", "WMT"})
NOT_AVAILABLE = re.compile(r"\bnot (?:\w+ ){0,2}(?:reported|disclosed|available|provided|found|returned)\b|\bno "
                           r"(?:separate |such )?(?:figure|line|value|fact)\b|\b(?:did|could|does|do) not (?:find|"
                           r"report|disclose|provide|present|include)\b|\bunavailable\b|\bnot (?:separately )?"
                           r"(?:broken out|stated)\b", re.IGNORECASE)
# How a passage names an item as its own line (for not_found): "time deposits" or "deposits with banks" are not a
# bank's total deposits
LINE_ITEM_IN_PASSAGE: dict[str, re.Pattern] = {"deposits": re.compile(r"\btotal deposits\b", re.IGNORECASE)}
# A reference, not a value: "Note 3", "Item 7", "Table 9", "page 42"
REFERENCE = re.compile(r"\b(?:note|notes|item|items|table|page|pages|part|exhibit)\s*$", re.IGNORECASE)
# Operations whose denominator is revenue unless the question names another
REVENUE_SHARE = re.compile(r"\bmargin\b|\b(?:percentage|percent|share|proportion|fraction) of\b|% of\b",
                           re.IGNORECASE)
# A question that asks what drove a change: agent.critic_checks.DRIVERS (imported above); the words that name a
# change in a passage
# The net income line a question may name with its holders ("net income attributable to common shareholders"), which
# named_metrics masks (it is not the consolidated line) but which the facts tool's net_income is
NET_INCOME_NAMED = re.compile(r"\bnet (?:income|earnings)\b", re.IGNORECASE)
CHANGE = re.compile(r"\b(?:increas\w*|decreas\w*|declin\w*|grew|grow\w*|rose|rise|fell|fall\w*|drop\w*|lower|higher|"
                    r"was (?:up|down)|were (?:up|down)|improv\w*|reduc\w*)\b", re.IGNORECASE)
WORD = re.compile(r"[a-z][a-z&'\-]+")
POSSESSIVE = re.compile(r"['’]s?(?=\s|$|[,.?;:])")  # "Costco's", "Sachs'" -> the name, before reading words
QUESTION_WORDS = frozenset(
    "what why how did does do was were is are the a an in of for to and or its their his her drove drive driving "
    "driver drivers explains explain explained caused cause causes contributed contribute reasons reason say says "
    "said attribute attributes fiscal year quarter so sharply strongly much more less behind change changes "
    "changed increase increased decrease decreased decline declined growth grow grew rise rose fall fell drop "
    "dropped higher lower up down during from with that this which company company's expansion expand expanded "
    "improvement improve improved contraction compared comparison versus prior previous year-over-year "
    "year-on-year main primary key major record first second third fourth third-quarter second-quarter "
    "first-quarter fourth-quarter full-year annual quarterly".split())
SENTENCE = re.compile(r"(?<=[.!?])\s+")
YEAR = re.compile(r"^(?:19|20)\d{2}$")
# Refusal-rule cues (docs/spec.md §6)
RULE_BREAKING = re.compile(r"\b(?:ignore|disregard|forget|bypass|override)\b.{0,40}?\b(?:rules?|instructions?|guidelines?|"
                           r"polic(?:y|ies)|system prompt|restrictions?)\b|\byou are now\b|\bact as\b|\bpretend\b|"
                           r"\bno restrictions\b|\bdeveloper mode\b", re.IGNORECASE)
TAX_LEGAL = re.compile(r"\btax(?:es|ed|ation)?\b|\blegal(?:ly)?\b|\blawyer\b|\battorney\b|\binheri\w+\b",
                       re.IGNORECASE)
FORECAST = re.compile(r"\bwill\b|\b(?:do|does|did) (?:you|\w+(?: \w+)?) expect\b|\bexpects? (?:to|that)\b|"
                      r"\bexpected to\b|\bforecast(?:s|ed|ing)?\b(?! (?:accuracy|error))|\bpredict\w*\b|"
                      r"\bnext (?:fiscal )?year\b|\bgoing forward\b|\bin the future\b", re.IGNORECASE)
FACT_ASK = re.compile(r"\b(?:what|how much)\b[^?]*?\b(?:was|were)\b|\bgiven\b", re.IGNORECASE)
FISCAL_YEAR = re.compile(r"\bfiscal (?:year )?(\d{4})\b", re.IGNORECASE)


# ----------------------------------------------------------------------------------------------- helpers
@functools.cache
def reported_years() -> frozenset[int]:
    """The fiscal years the indexed filings report (config/companies.yaml: the 10-K and 10-Q years)."""
    from ingest.edgar import load_scope
    scope = load_scope()
    return frozenset(scope.tenk_years) | frozenset(scope.tenq_years)


def asked_line_items(text: str) -> list[str]:
    """The line items of LINE_ITEMS that `text` names, in vocabulary order (a current ratio's parts count as the
    ratio)."""
    masked = text
    found = []
    for name, pattern in LINE_ITEMS.items():
        if name == "operating expenses":
            for mask in LINE_ITEM_MASKS:
                masked = mask.sub(" ", masked)
            if pattern.search(masked):
                found.append(name)
        elif pattern.search(text):
            found.append(name)
    if "current ratio" in found:
        found = [f for f in found if f not in ("current assets", "current liabilities")]
    return found


def _names_a_company(text: str) -> bool:
    from retrieval.entities import extract
    return bool(extract(text).tickers)


def subject_text(question: str, history: Sequence[str], has: Any) -> str:
    """The question, or, for a follow-up that names no company and nothing `has` finds, its latest earlier turn
    that does (W2-06 memory: "What was its operating margin that year?" asks about the turn before)."""
    if has(question) or _names_a_company(question) or question_metrics(question) or asked_line_items(question):
        return question
    for prior in reversed(list(history)):
        if has(prior):
            return prior
    return question


def used_fact_rows(draft: AnswerContract, results: Sequence[Any]) -> list[dict[str, Any]]:
    from agent.critic import draft_fact_rows
    return draft_fact_rows(draft, list(results))


def _answer_admits(answer: str, item_pattern: re.Pattern) -> bool:
    """The answer names the item and says it is not available (in the same sentence)."""
    return any(item_pattern.search(s) and NOT_AVAILABLE.search(s) for s in SENTENCE.split(answer))


def _hits(results: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """chunk_id -> every passage a search returned."""
    out: dict[str, dict[str, Any]] = {}
    for r in results:
        if r.call.name == "search_filings" and r.error_type is None and isinstance(r.output, list):
            for hit in r.output:
                if isinstance(hit, dict) and isinstance(hit.get("chunk_id"), str):
                    out[hit["chunk_id"]] = hit
    return out


def _seen(hit: dict[str, Any]) -> str:
    """A passage's text as the composer was shown it (its first MAX_PASSAGE_CHARS characters)."""
    from agent.composer import shown_text
    return shown_text({**hit, "text": str(hit.get("text") or "")})


def _label(row: dict[str, Any]) -> str:
    return str(row.get("label") or row["metric"].replace("_", " "))


def _numbers(text: str) -> list[Any]:
    return [f for f in figures(text) if not YEAR.match(f.text.strip())]


def _states(answer_figs: list[Any], fig: Any) -> bool:
    return any(a.percent == fig.percent and abs(a.value - fig.value) <= max(a.tolerance, fig.tolerance) + 1e-9
               for a in answer_figs)


# ----------------------------------------------------------------------------------------------- the checks
def question_metrics(text: str) -> frozenset[str]:
    """The facts tool's metrics the question names (agent.critic.named_metrics), counting net income named with its
    holders ("net income attributable to common shareholders"), which named_metrics masks."""
    from agent.critic import named_metrics
    return named_metrics(text) | ({"net_income"} if NET_INCOME_NAMED.search(text) else set())


def check_line_item(question: str, history: Sequence[str], draft: AnswerContract,
                    results: Sequence[Any]) -> list[Violation]:
    text = subject_text(question, history, asked_line_items)
    items = asked_line_items(text)
    if not items or draft.refused:
        return []
    out = []
    rows = used_fact_rows(draft, results)
    for item in items:
        pattern = LINE_ITEMS[item]
        own = IMPLIED_METRICS.get(item, frozenset())        # the tool's own fact for it, stated under its label
        if not pattern.search(draft.answer) and not (item == "current ratio" and re.search(
                r"\bcurrent\b", draft.answer, re.IGNORECASE)) and not any(r["metric"] in own for r in rows):
            out.append(Violation("line_item", f"the question asks for {item}, but the answer does not state it: state "
                                              f"{item} from the figures or passages if they give it, otherwise say that "
                                              f"{item} is not available"))
    admitted = any(_answer_admits(draft.answer, LINE_ITEMS[i]) for i in items)
    allowed = question_metrics(text).union(*(IMPLIED_METRICS.get(i, frozenset()) for i in items))
    for row in rows:
        if row["metric"] not in allowed and not admitted:
            out.append(Violation("line_item", f"the question asks for {', '.join(items)}, but the answer gives "
                                              f"{_label(row)} instead; never present another figure as {items[0]}: state "
                                              f"{items[0]} from the figures or passages if they give it, or say it is not "
                                              f"available and name what you give instead"))
    return out


def check_operation_inputs(question: str, history: Sequence[str], draft: AnswerContract,
                           results: Sequence[Any]) -> list[Violation]:
    from agent.critic import metric_question_text
    text = metric_question_text(question, list(history))
    operations = expected_operations(text)
    wanted = question_metrics(text)
    if not operations or not wanted or not draft.calculations or asked_line_items(text):
        return []                                    # an outside line item is check_line_item's
    implied = {"revenue"} if REVENUE_SHARE.search(text) else set()
    inputs = {x for c in draft.calculations for x in c.inputs}
    out = []
    for row in used_fact_rows(draft, results):
        used = any(abs(float(row["value"]) - x) <= 1e-9 * max(1.0, abs(x)) for x in inputs)
        if used and row["metric"] not in wanted | implied:
            out.append(Violation("inputs", f"the calculation uses {_label(row)}, which the question does not ask for "
                                           f"(it names {', '.join(sorted(w.replace('_', ' ') for w in wanted))}); do not "
                                           f"present this result as the measure asked: if the evidence lacks the asked "
                                           f"input, say so"))
    return out


def check_net_sales(question: str, history: Sequence[str], draft: AnswerContract,
                    results: Sequence[Any]) -> list[Violation]:
    text = subject_text(question, history, NET_SALES.search)
    if not NET_SALES.search(text) or draft.refused or _answer_admits(draft.answer, NET_SALES):
        return []
    return [Violation("net_sales", f"the question asks for net sales, but the answer gives {row.get('ticker')}'s "
                                   f"{_label(row)} ({row['value']}), which includes membership and other income: state "
                                   f"net sales from the passages if they give it, otherwise say net sales are not "
                                   f"available and name what you give instead")
            for row in used_fact_rows(draft, results)
            if row["metric"] == "revenue" and row.get("ticker") in SEPARATE_NET_SALES
            and row.get("concept") == TOTAL_REVENUE_CONCEPT and "sales" not in str(row.get("label") or "").lower()]


def check_segment(question: str, history: Sequence[str], draft: AnswerContract,
                  results: Sequence[Any]) -> list[Violation]:
    text = subject_text(question, history, SEGMENT.search)
    match = SEGMENT.search(text)
    if not match or draft.refused or DRIVERS.search(question) or _answer_admits(draft.answer, SEGMENT):
        return []
    rows = used_fact_rows(draft, results)
    if not rows:
        return []
    metrics = ", ".join(sorted({_label(r) for r in rows}))
    return [Violation("segment", f"the question is about a segment ({match.group(0)}), but the answer gives company "
                                 f"totals ({metrics}): never present them as the segment's; state the segment's own "
                                 f"figures from the passages if they give them, otherwise say they are not available")]


def check_not_found(question: str, history: Sequence[str], draft: AnswerContract,
                    results: Sequence[Any]) -> list[Violation]:
    text = subject_text(question, history, asked_line_items)
    items = asked_line_items(text)
    if not items or draft.refused or draft.values or not NOT_AVAILABLE.search(draft.answer):
        return []
    for hit in _hits(results).values():
        passage = _seen(hit)
        for item in items:
            for m in LINE_ITEM_IN_PASSAGE.get(item, LINE_ITEMS[item]).finditer(passage):
                clause = re.split(r"[.;]\s", passage[m.end():m.end() + 80], maxsplit=1)[0]
                values = [f for f in re.finditer(r"\(?\$?\d[\d,]*(?:\.\d+)?%?", clause)
                          if not REFERENCE.search(clause[:f.start()]) and not YEAR.match(f.group(0))]
                if values:
                    snippet = " ".join(passage[m.start():m.end() + len(clause)].split())[:100]
                    return [Violation("not_found", f"the answer says {item} is not available, but the passage "
                                                   f"{hit.get('citation') or ''} gives it ({snippet!r}): answer from "
                                                   f"that passage and cite it")]
    return []


def x_words(question: str) -> set[str]:
    """The content words of what a "what drove X" question asks about (X), without the company's name."""
    from retrieval.entities import extract
    text = POSSESSIVE.sub("", question.lower())               # "Costco's", "Sachs'" -> costco, sachs
    words = set(WORD.findall(text)) - QUESTION_WORDS
    names = {w for t in extract(question).tickers for w in _ticker_words(t)}
    return {w for w in words if w not in names and not w.endswith("ly") and len(w) > 2}


@functools.cache
def _ticker_words(ticker: str) -> frozenset[str]:
    """The words of a company's name and aliases (config/companies.yaml), so X never includes the company."""
    import yaml

    from ingest.edgar import COMPANIES_FILE
    for c in yaml.safe_load(COMPANIES_FILE.read_text(encoding="utf-8"))["companies"]:
        if c["ticker"] == ticker:
            names = [c["name"], *c.get("aliases", [])]
            return frozenset({w for n in names for w in WORD.findall(n.lower().replace("'s", ""))} | {ticker.lower()})
    return frozenset({ticker.lower()})


def check_summary(question: str, history: Sequence[str], draft: AnswerContract,
                  results: Sequence[Any]) -> list[Violation]:
    if not DRIVERS.search(question) or draft.refused or draft.calculations:
        return []
    want = x_words(question)
    if not want:
        return []
    hits = _hits(results)
    cited = [_seen(hits[c.chunk_id]) for c in draft.citations if c.source_type == "filing" and c.chunk_id in hits]
    if not cited:
        return []                                   # no passage cited: the contract's citation check says so
    # the subject words a stating sentence must carry: all of one or two, else two in three (rounded up), or the
    # subject's acronym as the filing writes it ("MCR" for the medical care ratio, W3-05)
    need = len(want) if len(want) <= 2 else max(2, -(-len(want) * 2 // 3))
    question_order = WORD.findall(POSSESSIVE.sub("", question.lower()))
    ordered = list(dict.fromkeys(w for w in question_order if w in want))
    subject = " ".join(ordered)                                                   # "data center revenue", as asked
    acronym = re.compile(rf"\b{''.join(w[0] for w in ordered).upper()}s?\b") if len(ordered) >= 2 else None
    stating = [s for p in cited for s in SENTENCE.split(" ".join(p.split()))
               if CHANGE.search(s) and (len(want & set(WORD.findall(POSSESSIVE.sub("", s.lower())))) >= need
                                        or (acronym is not None and acronym.search(s)))]
    if not stating:
        return [Violation("summary", f"no cited passage states the change in {subject} the question asks about: "
                                     f"answer from a passage that states that change and its drivers, and if none does, "
                                     f"say the evidence does not state them")]
    answer_figs = _numbers(draft.answer)
    sized = [s for s in stating if _numbers(s)]
    if sized and not any(_states(answer_figs, f) for s in sized for f in _numbers(s)):
        return [Violation("summary", f"the cited passage gives the size of the change in {subject} "
                                     f"({sized[0][:120]!r}); state it with its drivers")]
    return []


def check_refusal_rule(question: str, draft: AnswerContract) -> list[Violation]:
    if not draft.refused:
        return []
    rule = draft.refusal_rule
    out = []
    if RULE_BREAKING.search(question) and rule != "R6":
        out.append(Violation("refusal_rule", f"the question tells FinSight to ignore its rules or change its role: that "
                                             f"is R6 (spec §6), not {rule}"))
    elif TAX_LEGAL.search(question) and rule == "R5":
        out.append(Violation("refusal_rule", "tax or legal advice is R7 (off-topic, spec §6), not R5 personal data"))
    elif FORECAST.search(question) and rule == "R3":
        out.append(Violation("refusal_rule", "a question asking what will happen is R2 (a forecast), even for a period "
                                             "outside the corpus (spec §6: R2 before R3)"))
    years = {int(y) for y in FISCAL_YEAR.findall(question)}
    names_fact = (bool(question_metrics(question) or asked_line_items(question)) and _names_a_company(question)
                  and bool(years & reported_years()) and bool(FACT_ASK.search(question)))  # the fact's year
    if names_fact and rule in ("R1", "R2", "R4") and not RULE_BREAKING.search(question):
        out.append(Violation("refusal_rule", f"the question also asks for a fact the filings report; answer that part "
                                             f"and decline only the rest in one sentence ({rule}, a partial refusal, "
                                             f"spec §6), with refused false"))
    return out


def semantic_violations(question: str, history: Sequence[str], draft: AnswerContract,
                        results: Sequence[Any]) -> list[Violation]:
    """Every check above on the draft, in order; duplicates removed."""
    found = (check_line_item(question, history, draft, results)
             + check_operation_inputs(question, history, draft, results)
             + check_net_sales(question, history, draft, results)
             + check_segment(question, history, draft, results)
             + check_not_found(question, history, draft, results)
             + check_summary(question, history, draft, results)
             + check_refusal_rule(question, draft))
    return list(dict.fromkeys(found))
