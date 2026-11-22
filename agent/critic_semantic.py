"""The critic's checks of what the draft answers against what the question asks (W3-03, ADR-039): no LLM, no I/O.

    semantic_violations(question, history, draft, results, entities=None)  -> [Violation]

A1's critic checked the contract (agent/critic_checks.py) and units, and caught none of A1's wrong answers on gold:
they were well formed, but about another line item, a company total for a segment, a refusal under the wrong rule.
These checks read the question against the draft and the tools, one class of wrong answer each:

    line_item       the question asks for a line item the facts tool does not have (deposits, noninterest or
                    operating expenses, current assets, liabilities or ratio, the medical care ratio): the answer
                    must state it, and no fact for another metric may stand in for it, unless the answer says the
                    asked item is not available and names what it gives instead
    operation       a calculation's fetched inputs are each named by the question or are the denominator its
                    operation implies (revenue for a margin or a share of revenue); one named input no longer lets
                    any other through
    net_sales       a question about net sales is not answered with a total-revenue fact whose label is not net sales
                    (Costco's and Walmart's total revenues include membership income)
    segment         a question about a segment (AWS, Intelligent Cloud, ...) is not answered with company totals, also
                    when a follow-up ("its operating margin") asks about the segment its earlier turn named
    not_found       an answer that the asked item was not found or is not reported, while a retrieved passage gives
                    that item with a figure
    summary         a "what drove X" answer cites a passage that states X's change, and states that change's size
                    when the passage gives one
    refusal_rule    a question that also asks for a fact is not refused whole (a partial refusal, spec §6); an
                    instruction to ignore the rules or change role is R6; tax or legal advice is R7, not R5; a forecast
                    is R2, not R3 (spec §6: R2 before R3)

The period and duration of each value are already held to the question by the contract (a Value's duration fits its
fiscal period) and by invariant 4 (the question's years and periods), so they need no check here.

The vocabularies below are general filing and policy terms (the concept map's gaps, the spec's rules), not phrases of any
gold question. Each message says what to fix, since the graph sends the draft back once with them (spec §9).
"""
from __future__ import annotations

import functools
import re
from collections.abc import Sequence
from typing import Any

from agent.contract import AnswerContract
from agent.critic_checks import Violation, expected_operations
from core.figures import figures

# ----------------------------------------------------------------------------------------------- vocabularies
# Line items the facts tool's 13 metrics (config/concept_map.yaml) do not cover, as a question or an answer words them
LINE_ITEMS: dict[str, re.Pattern] = {
    "deposits": re.compile(r"\bdeposits\b", re.IGNORECASE),
    "noninterest expense": re.compile(r"\bnon-?interest expenses?\b", re.IGNORECASE),
    "operating expenses": re.compile(r"\b(?:total )?operating expenses\b", re.IGNORECASE),
    "current ratio": re.compile(r"\bcurrent ratio\b", re.IGNORECASE),
    "current assets": re.compile(r"\b(?:total )?current assets\b", re.IGNORECASE),
    "current liabilities": re.compile(r"\b(?:total )?current liabilities\b", re.IGNORECASE),
    "medical care ratio": re.compile(r"\bmedical (?:care|cost|loss) ratio\b|\bMCR\b|\bmedical costs? (?:as a "
                                     r"(?:percentage|share) of|divided by|to) premium", re.IGNORECASE),
}
# "operating expenses" inside a longer name that is another line item
LINE_ITEM_MASKS = (re.compile(r"\bnon-?interest expenses?\b", re.IGNORECASE),)
# Segments of the indexed companies, as their 10-Ks name them (the facts tool returns company totals only)
SEGMENT = re.compile(r"\bsegments?\b|\bAWS\b|\bAmazon Web Services\b|\bIntelligent Cloud\b|\bProductivity and Business "
                     r"Processes\b|\bMore Personal Computing\b|\bGoogle Cloud\b|\bGoogle Services\b|\bFamily of Apps\b|"
                     r"\bReality Labs\b|\bOptum\b|\bUnitedHealthcare\b|\bInnovative Medicine\b|\bMedTech\b|"
                     r"\bCompute (?:&|and) Networking\b|\bGlobal Banking (?:&|and) Markets\b|\bAsset (?:&|and) Wealth "
                     r"Management\b|\bPlatform Solutions\b", re.IGNORECASE)
NET_SALES = re.compile(r"\bnet sales\b", re.IGNORECASE)
TOTAL_REVENUE_CONCEPT = "us-gaap:Revenues"
NOT_AVAILABLE = re.compile(r"\bnot (?:\w+ ){0,2}(?:reported|disclosed|available|provided|found|returned)\b|\bno "
                           r"(?:separate |such )?(?:figure|line|value|fact)\b|\bdid not find\b|\bcould not find\b",
                           re.IGNORECASE)
# Operations whose denominator is revenue unless the question names another
REVENUE_SHARE = re.compile(r"\bmargin\b|\b(?:percentage|percent|share|proportion|fraction) of\b|% of\b",
                           re.IGNORECASE)
# A question that asks what drove a change (a driver word is needed: "What does X say about its pricing?" asks for
# what the filing says, not for a change), and the words that name a change in a passage
DRIVERS = re.compile(r"\bwhat (?:drove|explains|caused|contributed to|were the (?:drivers|reasons))\b|\bwhy did\b|"
                     r"\bdrove\b|\bdrivers? of\b|\battribut\w* (?:the|its|this|that)\b[^?]{0,60}\bto\b",
                     re.IGNORECASE)
# The net income line a question may name with its holders ("net income attributable to common shareholders"), which
# named_metrics masks (it is not the consolidated line) but which the facts tool's net_income is
NET_INCOME_NAMED = re.compile(r"\bnet (?:income|earnings)\b", re.IGNORECASE)
CHANGE = re.compile(r"\b(?:increas\w*|decreas\w*|declin\w*|grew|grow\w*|rose|rise|fell|fall\w*|drop\w*|lower|higher|"
                    r"was (?:up|down)|were (?:up|down)|improv\w*|reduc\w*)\b", re.IGNORECASE)
WORD = re.compile(r"[a-z][a-z&'\-]+")
POSSESSIVE = re.compile(r"['’]s\b")                  # "Costco's" -> "Costco" before reading words
QUESTION_WORDS = frozenset(
    "what why how did does do was were is are the a an in of for to and or its their his her drove drive driving "
    "driver drivers explains explain explained caused cause causes contributed contribute reasons reason say says "
    "said attribute attributes fiscal year quarter so sharply strongly much more less behind change changes "
    "changed increase increased decrease decreased decline declined growth grow grew rise rose fall fell drop "
    "dropped higher lower up down during from with that this which company company's".split())
SENTENCE = re.compile(r"(?<=[.!?])\s+")
YEAR = re.compile(r"^(?:19|20)\d{2}$")
# Refusal-rule cues (docs/spec.md §6)
RULE_BREAKING = re.compile(r"\b(?:ignore|disregard|forget|bypass|override)\b.{0,40}?\b(?:rules?|instructions?|guidelines?|"
                           r"polic(?:y|ies)|system prompt|restrictions?)\b|\byou are now\b|\bact as\b|\bpretend\b|"
                           r"\bno restrictions\b|\bdeveloper mode\b", re.IGNORECASE)
TAX_LEGAL = re.compile(r"\btax(?:es|ed|ation)?\b|\blegal(?:ly)?\b|\blawyer\b|\battorney\b|\binheri\w+\b",
                       re.IGNORECASE)
FORECAST = re.compile(r"\bwill\b|\bexpect\w*\b|\bforecast\w*\b|\bpredict\w*\b|\bproject(?:ed|ion)s?\b|\bnext year\b|"
                      r"\bgoing forward\b|\bin the future\b", re.IGNORECASE)
FACT_ASK = re.compile(r"\b(?:what|how much)\b[^?]*?\b(?:was|were)\b|\bgiven\b", re.IGNORECASE)
# ----------------------------------------------------------------------------------------------- helpers
def asked_line_items(text: str) -> list[str]:
    """The line items outside the facts tool that `text` names, in vocabulary order (a current ratio's parts count
    as the ratio)."""
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
    if has(question) or _names_a_company(question):
        return question
    for prior in reversed(list(history)):
        if has(prior):
            return prior
    return question


def _fact_rows(results: Sequence[Any]) -> list[dict[str, Any]]:
    from agent.critic import _fact_rows as rows
    return rows(list(results))


def used_fact_rows(draft: AnswerContract, results: Sequence[Any]) -> list[dict[str, Any]]:
    from agent.critic import draft_fact_rows
    return draft_fact_rows(draft, list(results))


def _answer_admits(answer: str, item_pattern: re.Pattern) -> bool:
    """The answer names the item and says it is not available (in the same sentence)."""
    return any(item_pattern.search(s) and NOT_AVAILABLE.search(s) for s in SENTENCE.split(answer))


def _passages(results: Sequence[Any]) -> dict[str, str]:
    """chunk_id -> text of every passage a search returned."""
    out: dict[str, str] = {}
    for r in results:
        if r.call.name == "search_filings" and r.error_type is None and isinstance(r.output, list):
            for hit in r.output:
                if isinstance(hit, dict) and isinstance(hit.get("chunk_id"), str):
                    out[hit["chunk_id"]] = str(hit.get("text") or "")
    return out


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
    for item in items:
        pattern = LINE_ITEMS[item]
        if not pattern.search(draft.answer) and not (item == "current ratio" and re.search(
                r"\bcurrent\b", draft.answer, re.IGNORECASE)):
            out.append(Violation("line_item", f"the question asks for {item}, but the answer does not state it; the "
                                              f"facts tool has no such metric, so search the filing for {item} (or "
                                              f"say it is not available)"))
    admitted = any(_answer_admits(draft.answer, LINE_ITEMS[i]) for i in items)
    allowed = question_metrics(text)
    for row in used_fact_rows(draft, results):
        if row["metric"] not in allowed and not admitted:
            out.append(Violation("line_item", f"the question asks for {', '.join(items)}, but the answer uses the "
                                              f"{row['metric'].replace('_', ' ')} fact ({row.get('label')}) instead; a "
                                              f"substitute is never the item asked: search for {items[0]} or say it is "
                                              f"not available"))
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
            out.append(Violation("operation", f"the calculation uses {row['metric'].replace('_', ' ')}, which the "
                                              f"question does not ask for (it names {', '.join(sorted(wanted))}); "
                                              f"every input must be a metric the question names"))
    return out


def check_net_sales(question: str, history: Sequence[str], draft: AnswerContract,
                    results: Sequence[Any]) -> list[Violation]:
    text = subject_text(question, history, NET_SALES.search)
    if not NET_SALES.search(text) or draft.refused:
        return []
    return [Violation("net_sales", f"the question asks for net sales, but the answer uses {row.get('ticker')}'s "
                                   f"{row.get('label')} ({row['value']}), which includes other revenue; search the "
                                   f"income statement for net sales")
            for row in used_fact_rows(draft, results)
            if row["metric"] == "revenue" and row.get("concept") == TOTAL_REVENUE_CONCEPT
            and "sales" not in str(row.get("label") or "").lower()]


def check_segment(question: str, history: Sequence[str], draft: AnswerContract,
                  results: Sequence[Any]) -> list[Violation]:
    text = subject_text(question, history, SEGMENT.search)
    match = SEGMENT.search(text)
    if not match or draft.refused or DRIVERS.search(question):
        return []
    rows = used_fact_rows(draft, results)
    if not rows:
        return []
    metrics = ", ".join(sorted({r["metric"].replace("_", " ") for r in rows}))
    return [Violation("segment", f"the question is about a segment ({match.group(0)}), but the answer uses company "
                                 f"totals from the facts tool ({metrics}); search the segment information for the "
                                 f"segment's own figures")]


def check_not_found(question: str, history: Sequence[str], draft: AnswerContract,
                    results: Sequence[Any]) -> list[Violation]:
    text = subject_text(question, history, asked_line_items)
    items = asked_line_items(text)
    if not items or draft.refused or draft.values or not NOT_AVAILABLE.search(draft.answer):
        return []
    for chunk_id, passage in _passages(results).items():
        for item in items:
            for m in LINE_ITEMS[item].finditer(passage):
                after = passage[m.end():m.end() + 80]
                if _numbers(after):
                    return [Violation("not_found", f"the answer says {item} was not found or is not reported, but "
                                                   f"a retrieved passage ({chunk_id}) gives it: answer from that "
                                                   f"passage and cite it")]
    return []


def x_words(question: str) -> set[str]:
    """The content words of what a "what drove X" question asks about (X), without the company's name."""
    from retrieval.entities import extract
    text = re.sub(r"['’]s\b", "", question.lower())        # "Costco's net sales" -> costco, net, sales
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
    passages = _passages(results)
    cited = [passages[c.chunk_id] for c in draft.citations if c.source_type == "filing" and c.chunk_id in passages]
    if not cited:
        return []                                   # no passage cited: the contract's citation check says so
    stating = [s for p in cited for s in SENTENCE.split(" ".join(p.split()))
               if CHANGE.search(s) and want <= set(WORD.findall(re.sub(r"['’]s\b", "", s.lower())))]
    subject = " ".join(sorted(want))
    if not stating:
        return [Violation("summary", f"no cited passage states the change in {subject} the question asks about; "
                                     f"answer from the passage of the asked year's MD&A that states it and its "
                                     f"drivers")]
    answer_figs = _numbers(draft.answer)
    sized = [s for s in stating if _numbers(s)]
    if sized and not any(_states(answer_figs, f) for s in sized for f in _numbers(s)):
        return [Violation("summary", f"the cited passage gives the size of the change in {subject} "
                                     f"({sized[0][:120]!r}); state it with its drivers")]
    return []


def check_refusal_rule(question: str, draft: AnswerContract) -> list[Violation]:
    if not draft.refused:
        return []
    from agent.critic import named_metrics
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
    names_fact = (bool(named_metrics(question) or asked_line_items(question)) and _names_a_company(question)
                  and bool(re.search(r"\bfiscal (?:20(?:2[3-5]))\b", question)) and bool(FACT_ASK.search(question)))
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
