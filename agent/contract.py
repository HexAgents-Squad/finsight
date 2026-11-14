"""The answer contract (docs/spec.md §8): the one JSON object every FinSight response is, as a pydantic model.

    AnswerContract {answer, values[], citations[], calculations[], verified, refused, refusal_rule, notes[]}

The models check shape: types (strict, as the model must write them: "2024" is not a fiscal year), allowed units,
periods and Items, and what each object says about itself (an xbrl citation has a concept and an accession
number; a 10-K is cited for "FY"; a chunk id names the filing and Item the citation gives; a Q2 value lasts 3 or
6 months). The spec §8 invariants, which relate the parts to each other and to the question, are the critic's
checks (agent/critic_checks.py): a draft that breaks one is still a valid contract, so the critic can say what
is wrong and the evaluation can count it.

Built from the tools' outputs with Value.from_fact, Citation.from_fact, Citation.from_search and
Calculation.from_tool; refusal answers with refusal_answer (the §6 sentences). contract_json_schema() is the
schema for response_format json_schema (ADR-016) and for prompts.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from core.spec import ITEMS_OF_FORM, FiscalPeriod, Form, RefusalRule, Unit
from ingest.sections import doc_id
from tools.calculator import Operation

STRICT = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
Number = int | float                    # strict: true and "12" are not numbers
Duration = Literal["3M", "6M", "9M", "12M"]


def _not_blank(v: str) -> str:
    if not v.strip():
        raise ValueError("must not be blank")
    return v


# Checked after parsing, not as a schema pattern: guided decoding (ADR-016) matches a pattern against the WHOLE
# string, so r"\S" would allow exactly one character (measured: the endpoint answered "4"). Every pattern below
# is therefore written as a full match, ^...$.
NonBlank = Annotated[str, Field(min_length=1), AfterValidator(_not_blank)]

# What a value's duration can be in each fiscal period: a quarter's own 3 months or the year to date; None is a
# point-in-time (balance-sheet) value (spec §8 values[].duration)
DURATIONS_OF_PERIOD: dict[str, frozenset[str | None]] = {
    "FY": frozenset({"12M", None}), "Q1": frozenset({"3M", None}), "Q2": frozenset({"3M", "6M", None}),
    "Q3": frozenset({"3M", "9M", None})}
PERIODS_OF_FORM: dict[str, frozenset[str]] = {"10-K": frozenset({"FY"}), "10-Q": frozenset({"Q1", "Q2", "Q3"})}
TICKER_PATTERN = r"^[A-Z]{1,5}$"                 # config/companies.yaml tickers (AAPL ... C)
ACCESSION_PATTERN = r"^\d{10}-\d{2}-\d{6}$"      # SEC accession number, e.g. 0000950170-24-087843
CONCEPT_PATTERN = r"^[a-z][a-z-]*:[A-Za-z][A-Za-z0-9]*$"   # taxonomy:Concept, e.g. us-gaap:OperatingIncomeLoss
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
SEC_URL_PATTERN = r"^https://www\.sec\.gov/\S*$"
CHUNK_SUFFIX = re.compile(r"_C\d_\d{4}$")        # <doc_id>_<variant>_<n>: AAPL_10-K_2024_Item8_C3_0012
COULD_NOT_VERIFY = "could not verify: "          # the note an unverified answer carries (spec §8 invariant 7)

# §6 refusal sentences. {reason} per policy rule; R3 (out of corpus) has its own sentence. SCOPE_TEXT states
# FinSight's own coverage, so its numbers are not figures the answer must back with values (invariant 1).
REFUSAL_REASONS: dict[str, str] = {
    "R1": "investment advice",
    "R2": "a forecast that the filings do not state",
    "R4": "a valuation opinion",
    "R5": "personal information that FinSight does not provide",
    "R6": "a request to bypass FinSight's rules",
    "R7": "outside what FinSight does (questions about SEC filings)",
}
POLICY_SENTENCE = ("I can't help with that because it is {reason}. I can share facts from the indexed SEC filings "
                   "instead, for example {suggestion}.")
SCOPE_TEXT = "(20 companies, 10-K fiscal 2023–2025 and 10-Q fiscal 2025 Q1–Q3)"
R3_SENTENCE = ("That is outside the filings FinSight has indexed " + SCOPE_TEXT
               + ". I can answer, for example, {suggestion}.")
R3_MARKER = "outside the filings FinSight has indexed"     # how an R3 decline is recognised in a partial answer


def refusal_template(rule: str) -> str:
    """The §6 sentence for `rule`, with {suggestion} left to fill."""
    if rule == "R3":
        return R3_SENTENCE
    return POLICY_SENTENCE.replace("{reason}", REFUSAL_REASONS[rule])


def refusal_answer(rule: str, suggestion: str) -> str:
    """The refusal answer: the §6 sentence for `rule` with one concrete in-scope question as the suggestion."""
    if not suggestion.strip():
        raise ValueError("a refusal needs a suggestion (spec §6)")
    return refusal_template(rule).replace("{suggestion}", suggestion.strip().rstrip(".?"))


class Value(BaseModel):
    """One number the answer states, in filing units (spec §7 rule 3)."""
    model_config = STRICT

    name: NonBlank
    value: Number
    unit: Unit
    ticker: str = Field(pattern=TICKER_PATTERN)
    fiscal_year: int
    fiscal_period: FiscalPeriod
    duration: Duration | None
    period_end: str = Field(pattern=DATE_PATTERN)

    @field_validator("period_end")
    @classmethod
    def _real_date(cls, v: str) -> str:
        date.fromisoformat(v)                   # 2024-02-30 is not a date
        return v

    @model_validator(mode="after")
    def _duration_fits_period(self) -> Value:
        if self.duration not in DURATIONS_OF_PERIOD[self.fiscal_period]:
            raise ValueError(f"a {self.fiscal_period} value cannot last {self.duration}")
        return self

    @classmethod
    def from_fact(cls, fact: dict[str, Any], name: str | None = None) -> Value:
        """A get_financial_fact result as a value (named after its metric unless `name` is given)."""
        return cls(name=name or fact["metric"], value=fact["value"], unit=fact["unit"], ticker=fact["ticker"],
                   fiscal_year=fact["fiscal_year"], fiscal_period=fact["fiscal_period"],
                   duration=fact["duration"], period_end=fact["period_end"])


class Citation(BaseModel):
    """A source: a filing passage (item + chunk_id) or an XBRL fact (concept + accession_no)."""
    model_config = STRICT

    source_type: Literal["filing", "xbrl"]
    ticker: str = Field(pattern=TICKER_PATTERN)
    form: Form
    fiscal_year: int
    fiscal_period: FiscalPeriod
    item: str | None = None
    chunk_id: str | None = None
    concept: str | None = Field(default=None, pattern=CONCEPT_PATTERN)
    accession_no: str | None = Field(default=None, pattern=ACCESSION_PATTERN)
    url: str = Field(pattern=SEC_URL_PATTERN)

    @model_validator(mode="after")
    def _consistent(self) -> Citation:
        if self.fiscal_period not in PERIODS_OF_FORM[self.form]:
            raise ValueError(f"a {self.form} is not cited for {self.fiscal_period}")
        if self.source_type == "xbrl":
            if self.concept is None or self.accession_no is None:
                raise ValueError("an xbrl citation needs concept and accession_no")
            if self.item is not None or self.chunk_id is not None:
                raise ValueError("an xbrl citation has no item or chunk_id")
            return self
        if self.item is None or self.chunk_id is None:
            raise ValueError("a filing citation needs item and chunk_id")
        if self.concept is not None or self.accession_no is not None:
            raise ValueError("a filing citation has no concept or accession_no")
        if self.item not in ITEMS_OF_FORM[self.form]:
            raise ValueError(f"{self.item!r} is not an Item of a {self.form} (spec §3)")
        prefix = doc_id(self.ticker, self.form, self.fiscal_year, self.fiscal_period, self.item) + "_"
        if not (self.chunk_id.startswith(prefix) and CHUNK_SUFFIX.search(self.chunk_id)
                and len(self.chunk_id) == len(prefix) + len("C3_0000")):
            raise ValueError(f"chunk_id {self.chunk_id!r} is not a chunk of {prefix[:-1]}")
        return self

    @classmethod
    def from_fact(cls, fact: dict[str, Any]) -> Citation:
        """The citation for a get_financial_fact result (the filing that carries the value, its period)."""
        return cls(source_type="xbrl", ticker=fact["ticker"], form=fact["form"], fiscal_year=fact["fiscal_year"],
                   fiscal_period=fact["fiscal_period"], concept=fact["concept"], accession_no=fact["accession_no"],
                   url=fact["url"])

    @classmethod
    def from_search(cls, hit: dict[str, Any]) -> Citation:
        """The citation for one search_filings hit."""
        return cls(source_type="filing", ticker=hit["ticker"], form=hit["form"], fiscal_year=hit["fiscal_year"],
                   fiscal_period=hit["fiscal_period"], item=hit["item"], chunk_id=hit["chunk_id"],
                   url=hit["source_url"])


class Calculation(BaseModel):
    """A computed number as the calculate tool returned it (its notes go to the answer's notes)."""
    model_config = STRICT

    operation: Operation
    inputs: list[Number] = Field(min_length=2)
    years: Number | None = None         # cagr only (tools/calculator.py); needed to recompute it
    result: Number
    unit: Unit | None
    expression: NonBlank

    @classmethod
    def from_tool(cls, output: dict[str, Any]) -> Calculation:
        """A calculate result; an error result is refused (an undefined measure is stated, not computed)."""
        if "error" in output:
            raise ValueError(f"calculate failed: {output['error'].get('message')}")
        return cls(**{k: v for k, v in output.items() if k != "notes"})


class AnswerContract(BaseModel):
    """Spec §8. `verified` is set by the critic: true only when every check in agent/critic_checks.py passes."""
    model_config = STRICT

    answer: NonBlank
    values: list[Value] = []
    citations: list[Citation] = []
    calculations: list[Calculation] = []
    verified: bool
    refused: bool
    refusal_rule: RefusalRule | None = None
    notes: list[str] = []


def _drop_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _drop_titles(v) for k, v in node.items() if k != "title" or not isinstance(v, str)}
    if isinstance(node, list):
        return [_drop_titles(v) for v in node]
    return node


def contract_json_schema() -> dict[str, Any]:
    """The contract's JSON schema without pydantic's titles: for response_format json_schema with strict guided
    decoding (ADR-016) and for prompts. Cross-field rules (citation kinds, durations) are checked after parsing."""
    return _drop_titles(AnswerContract.model_json_schema())
