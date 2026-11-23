"""The critic's checks (docs/spec.md §8 invariants): pure functions over an AnswerContract, no LLM, no I/O.

    check_contract(contract, scope=None, question=None) -> [Violation]      # all checks; [] = verified
    with_verification(contract, violations) -> AnswerContract              # verified and "could not verify" notes

One function per rule, each returning the violations it finds:
    1  check_numbers_in_values   every figure in `answer` is a value, as displayed ("$109.4 billion" for 109433
                                 USD millions, "44.6%" for 44.6 %; signs are stated in words, so magnitudes)
    2  check_values_supported    every value has a citation with its ticker, fiscal year and period, or is the
                                 result of a calculation whose inputs are all values
    3  check_calculations        every calculation equals tools.calculator.compute on its inputs (§7 rounding)
    4  check_scope               cited tickers and periods are the question's; every question company is cited
    5  check_refusal             refused => a rule, no values or calculations, the rule's §6 sentence
    6  check_partial_refusal     a rule without refused => exactly one declining sentence plus the factual part
    7  check_verified            verified false => a "could not verify: ..." note; true => none, and no violation
       check_citations_present   §8 citations[]: at least one for every answer that is not refused
       check_operation           the calculation's operation fits the question (ADR-016: margin -> percent,
                                 % change -> growth ...); only when the question asks for one

The messages are written for the model: the critic sends them back with the draft once (spec §9), and they
become the "could not verify" notes if the retry fails too. The evaluation (W1-09) runs the same checks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.contract import (
    COULD_NOT_VERIFY,
    R3_MARKER,
    REFUSAL_REASONS,
    SCOPE_TEXT,
    AnswerContract,
    refusal_template,
)
from core.figures import Figure, figures
from tools.base import ToolError
from tools.calculator import UNIT_FROM_ARGUMENT, compute

# A value's unit as a multiplier of the figure a reader sees ("$109.4 billion" is 1.094e11 USD)
UNIT_SCALE: dict[str, float] = {"USD": 1.0, "USD thousands": 1e3, "USD millions": 1e6, "USD billions": 1e9,
                                "shares": 1.0, "shares millions": 1e6, "USD per share": 1.0, "count": 1.0,
                                "%": 1.0, "ratio": 1.0}
RELATIVE_EPSILON = 1e-9                 # float noise allowed when two numbers are "the same" (inputs, results)
# Typographic variants a model writes for the §6 sentences
NORMALISE = str.maketrans({"’": "'", "‘": "'", "–": "-", "—": "-", "“": '"', "”": '"'})
PLACEHOLDER = re.compile(r"[{}]")                 # a template placeholder left in, e.g. "{suggestion}"
# A sentence ends at . ! or ? before a capital, except after an abbreviation ("U.S. GAAP", "e.g. Data Center")
SENTENCE_END = re.compile(r"(?<!\bU\.S\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bInc\.)(?<!\bCorp\.)(?<!\bCo\.)(?<!\bvs\.)"
                          r"(?<!\bNo\.)(?<=[.!?])\s+(?=[A-Z\"'(])")
# The §6 sentence's offer ("I can share facts ... instead, for example ...") is not the factual part of an answer
OFFER = re.compile(r"\bI can share facts from the indexed SEC filings\b|\bI can answer, for example\b", re.IGNORECASE)
QUOTED = re.compile(r"\s*'(.*)'([.?]?)\s*", re.DOTALL)     # a suggestion in single quotes
# A declining sentence speaks for FinSight ("I can't help ...", "FinSight does not provide ..."): "the filings do
# not provide a fiscal 2026 figure" states a fact and is no decline
DECLINE = re.compile(r"\b(?:I|FinSight)\s+(?:can't|cannot|can not|won't|will not|am unable to|is unable to|do not|"
                     r"don't|does not|doesn't)\s+(?:\w+\s+){0,2}?(?:help|provide|give|offer|answer|predict|forecast|"
                     r"share|comment|recommend|advise)\b", re.IGNORECASE)
CANNOT = re.compile(r"\bcan ?not\b", re.IGNORECASE)       # "I cannot help" is the §6 sentence's "I can't help"
# Rewrites before reading figures (critic only; the scorer's reading is unchanged). Not figures: a period's length
# ("the 6 months ended", "52 weeks ended": Costco, gold A11/B11), a fiscal-year span ("2024-25", "FY2024/25"), a
# split ratio ("10-for-1"), a page or note reference. Read as meant: "1.03x" is the ratio 1.03; "USD391,035" and
# "22.7-percent" spaced out
FIGURE_REWRITES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b(?:3|6|9|12|13|26|39|52|53)[- ](?:months?|weeks?)\b", re.IGNORECASE), " "),
    (re.compile(r"(?<!\d)((?:19|20)\d{2})[-/]\d{2}\b"), r"\1"),          # "FY2024/25": no \b after "FY"
    (re.compile(r"\b\d+-for-\d+\b", re.IGNORECASE), " "),
    (re.compile(r"\b(?:page|pages|p\.|note|notes)\s+\d+\b", re.IGNORECASE), " "),
    (re.compile(r"(\d)x\b"), r"\1 "),
    (re.compile(r"\b(USD|US\$)(?=\d)"), r"\1 "),
    (re.compile(r"(\d)-(percent|per cent)\b", re.IGNORECASE), r"\1 \2"),
)
# Units get_financial_fact never returns: such a value is computed or read from a passage, never an XBRL fact
DERIVED_UNITS = frozenset({"%", "ratio"})
# An answer that gives the inputs because the measure is undefined (§7 rule 2) or the figure is not reported
# (§7 rule 5) has no calculation to show
NO_RESULT = re.compile(r"\bnot meaningful\b|\bnot (?:\w+ )?(?:reported|disclosed)\b", re.IGNORECASE)

# Question wording -> the operations that answer it (ADR-016: the critic checks the operation against the
# question). Matched on the question text; no match means the question asks for no particular operation.
CHANGE = r"\b(?:grow|grew|grown|growth|chang\w*|increas\w*|decreas\w*|declin\w*|rise|rose|fell|fall|drop\w*)\b"
IN_PERCENT = r"%|\bper ?cent|\bpercentage\b"
OPERATION_RULES: tuple[tuple[re.Pattern, frozenset[str]], ...] = (
    (re.compile(r"\bcagr\b|\bcompound(?:ed)? annual", re.IGNORECASE), frozenset({"cagr"})),
    (re.compile(rf"(?:{CHANGE}.*(?:{IN_PERCENT}))|(?:(?:{IN_PERCENT}).*{CHANGE})", re.IGNORECASE),
     frozenset({"growth", "cagr"})),
    (re.compile(r"\bmargin\b|\bas an? (?:percentage|percent|share|proportion|fraction) of\b|\bshare of\b"
                r"|\bpercent(?:age)? of\b|% of\b", re.IGNORECASE), frozenset({"percent"})),
    # a bank's efficiency or CET1 ratio is stated in %, so a ratio question may be answered with percent
    (re.compile(r"\bratio\b", re.IGNORECASE), frozenset({"ratio", "percent"})),
    (re.compile(r"\bhow much (?:more|less|higher|lower|larger|smaller|bigger)\b|\bdifference (?:between|in)\b"
                r"|\bby how much\b", re.IGNORECASE), frozenset({"difference"})),
    (re.compile(r"\bcombined\b|\bin total\b|\btotal of\b|\btogether\b|\bsum of\b", re.IGNORECASE),
     frozenset({"sum"})),
)

# A question that asks what drove a change (a driver word is needed: "What does X say about its pricing?" asks for
# what the filing says, not for a change). It asks for the drivers, not an operation: "Why did the medical care ratio
# rise?" names a ratio but wants no calculation (W3-05), unless it also asks for a figure ("What was the revenue CAGR
# ... and what drove it?": FIGURE_ASK)
DRIVERS = re.compile(r"\bwhat (?:drove|explains|caused|contributed to|were the (?:drivers|reasons))\b|\bwhy did\b|"
                     r"\bdrove\b|\bdrivers? of\b|\battribut\w* (?:the|its|this|that)\b[^?]{0,60}\bto\b",
                     re.IGNORECASE)

FIGURE_ASK = re.compile(r"\b(?:what|how much)\s+(?:was|were|is|are)\b(?!\s+the\s+(?:drivers|reasons|factors|causes)\b)",
                        re.IGNORECASE)


@dataclass(frozen=True)
class Violation:
    rule: str                   # "1" ... "7" (spec §8 invariants), "citations", "operation"
    message: str

    def __str__(self) -> str:
        label = f"invariant {self.rule}" if self.rule.isdigit() else self.rule
        return f"{label}: {self.message}"


@dataclass(frozen=True)
class QuestionScope:
    """The companies and periods the question asks about (the planner's entities; in the evaluation, the gold
    item's companies and fiscal years). An empty set is not checked."""
    tickers: frozenset[str]
    fiscal_years: frozenset[int] = field(default_factory=frozenset)
    fiscal_periods: frozenset[str] = field(default_factory=frozenset)


def same_number(a: float, b: float) -> bool:
    return a == b or abs(a - b) <= RELATIVE_EPSILON * max(1.0, abs(a), abs(b))


def _norm(text: str) -> str:
    return " ".join(text.translate(NORMALISE).split())


def sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_END.split(_norm(text)) if s]


# ----------------------------------------------------------------------------------------------- invariant 1
def _shown(fig: Figure, contract: AnswerContract) -> bool:
    """`fig` displays one of the values: the value as written in its unit ("391,035") or scaled to dollars,
    shares ... ("$391.0 billion"), at the precision the answer printed it; % only as a percentage, and a ratio
    also as one ("0.45, i.e. 45%"). A CAGR's number of years ("a 3-year CAGR") is backed by its calculation."""
    candidates: list[tuple[float, bool]] = [(abs(c.years), False) for c in contract.calculations if c.years]
    for v in contract.values:
        candidates += [(abs(v.value), v.unit == "%"), (abs(v.value) * UNIT_SCALE[v.unit], v.unit == "%")]
        if v.unit == "ratio":
            candidates.append((abs(v.value) * 100, True))
    return any(fig.percent == percent
               and abs(fig.value - number) <= fig.tolerance + RELATIVE_EPSILON * max(1.0, number)
               for number, percent in candidates)


def unbacked_figures(contract: AnswerContract) -> list[Figure]:
    """The figures the answer states that no value matches (invariant 1), as the critic reads them."""
    text = _norm(contract.answer).replace(_norm(SCOPE_TEXT), " ")       # FinSight's own coverage, not figures
    for pattern, replacement in FIGURE_REWRITES:
        text = pattern.sub(replacement, text)
    return [fig for fig in figures(text) if not _shown(fig, contract)]


def check_numbers_in_values(contract: AnswerContract) -> list[Violation]:
    return [Violation("1", f"the answer states {fig.text!r} but no value in values matches it")
            for fig in unbacked_figures(contract)]


# ----------------------------------------------------------------------------------------------- invariant 2
def check_values_supported(contract: AnswerContract) -> list[Violation]:
    numbers = [v.value for v in contract.values]
    out = []
    for v in contract.values:
        # a % or ratio is never an XBRL fact: it is computed, or stated in a cited passage (spec §7 rule 2)
        kinds = ("filing",) if v.unit in DERIVED_UNITS else ("filing", "xbrl")
        cited = any((c.ticker, c.fiscal_year, c.fiscal_period) == (v.ticker, v.fiscal_year, v.fiscal_period)
                    and c.source_type in kinds for c in contract.citations)
        computed = any(same_number(calc.result, v.value) and calc.unit in (None, v.unit)
                       and all(any(same_number(x, n) for n in numbers) for x in calc.inputs)
                       for calc in contract.calculations)
        if not (cited or computed):
            source = "filing passage" if v.unit in DERIVED_UNITS else "citation"
            out.append(Violation("2", f"value {v.name} = {v.value} {v.unit} ({v.ticker} {v.fiscal_period} "
                                      f"{v.fiscal_year}) has no {source} for that company and period and is not "
                                      f"the result of a calculation on the values"))
    return out


# ----------------------------------------------------------------------------------------------- invariant 3
def check_calculations(contract: AnswerContract) -> list[Violation]:
    out = []
    for calc in contract.calculations:
        unit = calc.unit if calc.operation in UNIT_FROM_ARGUMENT else None
        try:
            fresh = compute(calc.operation, list(calc.inputs), calc.years, unit)
        except (ToolError, ValueError) as err:          # pydantic's ValidationError is a ValueError
            message = err.message if isinstance(err, ToolError) else str(err).splitlines()[0]
            out.append(Violation("3", f"{calc.operation} of {calc.inputs} cannot be recomputed: {message}"))
            continue
        if not same_number(fresh["result"], calc.result):
            out.append(Violation("3", f"{calc.operation} of {calc.inputs} is {fresh['result']}, not {calc.result} "
                                      f"({fresh['expression']})"))
        elif fresh["unit"] != calc.unit:
            out.append(Violation("3", f"{calc.operation} gives unit {fresh['unit']!r}, not {calc.unit!r}"))
    return out


# ----------------------------------------------------------------------------------------------- invariant 4
def check_scope(contract: AnswerContract, scope: QuestionScope | None) -> list[Violation]:
    if scope is None:
        return []
    out = []
    for c in contract.citations:
        where = f"citation {c.ticker} {c.form} {c.fiscal_period} {c.fiscal_year}"
        if scope.tickers and c.ticker not in scope.tickers:
            out.append(Violation("4", f"{where}: {c.ticker} is not a company the question asks about"))
        if scope.fiscal_years and c.fiscal_year not in scope.fiscal_years:
            out.append(Violation("4", f"{where}: fiscal {c.fiscal_year} is not a year the question asks about"))
        if scope.fiscal_periods and c.fiscal_period not in scope.fiscal_periods:
            out.append(Violation("4", f"{where}: {c.fiscal_period} is not a period the question asks about"))
    for v in contract.values:                   # a computed value carries its own company and period too
        where = f"value {v.name} ({v.ticker} {v.fiscal_period} {v.fiscal_year})"
        if scope.tickers and v.ticker not in scope.tickers:
            out.append(Violation("4", f"{where}: {v.ticker} is not a company the question asks about"))
        if scope.fiscal_years and v.fiscal_year not in scope.fiscal_years:
            out.append(Violation("4", f"{where}: fiscal {v.fiscal_year} is not a year the question asks about"))
        if scope.fiscal_periods and v.fiscal_period not in scope.fiscal_periods:
            out.append(Violation("4", f"{where}: {v.fiscal_period} is not a period the question asks about"))
    if not contract.refused:
        cited = {c.ticker for c in contract.citations}
        out += [Violation("4", f"{t} is in the question but has no citation") for t in sorted(scope.tickers - cited)]
    return out


# ----------------------------------------------------------------------------------------------- invariants 5, 6
def is_refusal_sentence(answer: str, rule: str) -> bool:
    """`answer` is the §6 sentence of `rule`: its fixed text up to "for example", then a non-empty suggestion
    ending the sentence (both templates end in "{suggestion}."; a suggestion written as a question may end in
    "?", may be quoted, and may follow a comma or colon). Allowed variants: curly apostrophes and quotes, dashes,
    spacing, "cannot" for "can't". An unfilled placeholder is not a suggestion (measured: Qwen3-8B copied
    "{suggestion}" as written), and nothing may follow the suggestion's sentence."""
    before = _norm(refusal_template(rule)).split("{suggestion}")[0].rstrip(" ,")       # "... for example"
    text = CANNOT.sub("can't", _norm(answer))
    if not text.startswith(before):
        return False
    suggestion = text[len(before):].replace('"', "").rstrip()
    quoted = QUOTED.fullmatch(suggestion)
    if quoted:                                           # one pair of single quotes, apostrophes kept
        suggestion = quoted.group(1) + quoted.group(2)
    return (suggestion.endswith((".", "?")) and bool(suggestion.rstrip(".?").strip())
            and not PLACEHOLDER.search(suggestion) and len(sentences(suggestion)) == 1)


def check_refusal(contract: AnswerContract) -> list[Violation]:
    if not contract.refused:
        return []
    out = []
    if contract.refusal_rule is None:
        out.append(Violation("5", "refused is true but refusal_rule is not set"))
    if contract.values:
        out.append(Violation("5", f"a refusal has no values, found {len(contract.values)}"))
    if contract.calculations:
        out.append(Violation("5", f"a refusal has no calculations, found {len(contract.calculations)}"))
    if contract.refusal_rule is not None and not is_refusal_sentence(contract.answer, contract.refusal_rule):
        out.append(Violation("5", f"the answer is not the {contract.refusal_rule} refusal sentence of spec §6: "
                                  f"{refusal_template(contract.refusal_rule)!r}"))
    return out


def declines(sentence: str, rule: str) -> bool:
    marker = R3_MARKER if rule == "R3" else REFUSAL_REASONS[rule]
    return marker.lower() in sentence.lower() or bool(DECLINE.search(sentence))


def check_partial_refusal(contract: AnswerContract) -> list[Violation]:
    if contract.refused or contract.refusal_rule is None:
        return []
    parts = sentences(contract.answer)
    declining = [s for s in parts if declines(s, contract.refusal_rule)]
    if len(declining) != 1:
        found = "none" if not declining else f"{len(declining)}"
        return [Violation("6", f"a partial refusal ({contract.refusal_rule}) declines the refused part in exactly "
                               f"one sentence; found {found}")]
    factual = [s for s in parts if s not in declining and not OFFER.search(s)]
    if not factual and not contract.values:             # "X was $416.2 billion, but I can't ..." is both
        return [Violation("6", "a partial refusal answers the factual part too; the answer only declines")]
    return []


# ----------------------------------------------------------------------------------------------- invariant 7, §8
def _unverified_notes(contract: AnswerContract) -> list[str]:
    return [n for n in contract.notes if n.strip().lower().startswith(COULD_NOT_VERIFY.strip())]


def check_verified(contract: AnswerContract, other: list[Violation]) -> list[Violation]:
    """`other` are the violations of every other check."""
    notes = _unverified_notes(contract)
    if not contract.verified and not notes:
        return [Violation("7", "verified is false but notes has no 'could not verify: ...' entry")]
    if contract.verified and notes:
        return [Violation("7", "verified is true but notes says it could not verify")]
    if contract.verified and other:
        return [Violation("7", f"verified is true but {len(other)} check(s) fail")]
    return []


def check_citations_present(contract: AnswerContract) -> list[Violation]:
    if contract.refused or contract.citations:
        return []
    return [Violation("citations", "an answer that is not refused needs at least one citation")]


# ----------------------------------------------------------------------------------------------- operation
def expected_operations(question: str) -> frozenset[str]:
    """The operations the question's wording asks for; empty when it asks for none in particular, or asks only what
    drove a change (DRIVERS without FIGURE_ASK)."""
    out: frozenset[str] = frozenset()
    if DRIVERS.search(question) and not FIGURE_ASK.search(question):
        return out
    for pattern, operations in OPERATION_RULES:
        if pattern.search(question):
            out |= operations
    return out


def check_operation(contract: AnswerContract, question: str | None) -> list[Violation]:
    """The question asks for an operation: a calculation must do it (§7 rule 2: every computed number comes
    from the calculator), unless the answer says the measure is not meaningful or the figure not reported, or
    the question is refused."""
    if not question or contract.refused:
        return []
    wanted = expected_operations(question)
    if not wanted:
        return []
    used = {c.operation for c in contract.calculations}
    if not used:
        stated = [v for v in contract.values if v.unit in DERIVED_UNITS and any(
            c.source_type == "filing" and (c.ticker, c.fiscal_year, c.fiscal_period)
            == (v.ticker, v.fiscal_year, v.fiscal_period) for c in contract.citations)]
        # a % or ratio the filing states itself (Apple's gross margin percentage, gold B13) is read, not computed
        if NO_RESULT.search(contract.answer) or stated:
            return []
        return [Violation("operation", f"the question asks for {' or '.join(sorted(wanted))}, but no calculation "
                                       f"computes it (every computed number comes from the calculator)")]
    if not used & wanted:
        return [Violation("operation", f"the question asks for {' or '.join(sorted(wanted))}, but the "
                                       f"calculations use {', '.join(sorted(used))}")]
    return []


# ----------------------------------------------------------------------------------------------- all
def check_contract(contract: AnswerContract, scope: QuestionScope | None = None,
                   question: str | None = None) -> list[Violation]:
    """Every check, in rule order. Invariant 4 needs `scope` and the operation check `question`; without them
    they are skipped (the result says nothing about them)."""
    other = (check_numbers_in_values(contract) + check_values_supported(contract) + check_calculations(contract)
             + check_scope(contract, scope) + check_refusal(contract) + check_partial_refusal(contract)
             + check_citations_present(contract) + check_operation(contract, question))
    return other + check_verified(contract, other)


def with_verification(contract: AnswerContract, violations: list[Violation]) -> AnswerContract:
    """The contract as the critic returns it: verified when nothing but invariant 7 failed, otherwise verified
    false with one "could not verify: ..." note per violation (old ones replaced). check_contract on the result
    reports the same violations except invariant 7."""
    problems = [v for v in violations if v.rule != "7"]
    kept = [n for n in contract.notes if n not in _unverified_notes(contract)]
    notes = kept + [f"{COULD_NOT_VERIFY}{v}" for v in dict.fromkeys(problems)]
    return contract.model_copy(update={"verified": not problems, "notes": notes})
