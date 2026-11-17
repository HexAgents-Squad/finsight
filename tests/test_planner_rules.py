"""W2-02: the planner's deterministic rules after the model (agent/planner.py finalize and its parts).

Each rule corrects the model only where the question, the corpus or the tools show it is wrong, never wholesale;
the cases below include the ones three reviews reproduced. Offline: plans are written by hand, no LLM."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from agent import planner as pl
from agent.state import Plan, PlanStep
from retrieval.entities import Entities, from_turns

NULLS = {"ticker": None, "fiscal_year": None, "fiscal_period": None, "form": None, "item": None, "k": 5}


def search(**args) -> PlanStep:
    return PlanStep(tool="search_filings", purpose="find it", arguments={**NULLS, "query": "revenue drivers", **args})


def fact(ticker="META", year=2025, period="FY", metric="revenue", duration=None) -> PlanStep:
    return PlanStep(tool="get_financial_fact", purpose="x", arguments={
        "ticker": ticker, "metric": metric, "fiscal_year": year, "fiscal_period": period, "duration": duration})


def compare(ticker="META", metric="revenue", years=(2023, 2024)) -> PlanStep:
    return PlanStep(tool="compare_periods", purpose="x",
                    arguments={"ticker": ticker, "metric": metric, "years": list(years)})


def plan(*steps, rule=None, calculation=None) -> Plan:
    return Plan(reasoning="x", refuse=False, refusal_rule=rule, steps=list(steps), calculation=calculation)


def run(p: Plan, question: str, history=()) -> tuple[Plan, list[str]]:
    """finalize as the node calls it: planning entities from the whole conversation."""
    text = " ".join([*history, question])
    return pl.finalize(p, pl.planning_entities(from_turns([*history, question]), text), question, history)


def args(p: Plan, i: int = 0) -> dict:
    return p.steps[i].arguments


# ----------------------------------------------------------------------------------------------- periods
@pytest.mark.parametrize(("question", "planned", "expected"), [
    ("What was AbbVie's revenue in fiscal 2025?", "Q4", "FY"),                 # no period cue: the year
    ("What was Merck's operating income in fiscal 2025?", "Q1", "FY"),
    ("What was Meta's revenue in Q2 fiscal 2025?", "Q2", "Q2"),               # the model's quarter stands
    ("What was Meta's second-quarter revenue in fiscal 2025?", "Q2", "Q2"),    # hyphenated
    ("What was Microsoft's revenue in the quarter ended September 30, 2024?", "Q1", "Q1"),
    ("What were P&G's net sales in the fourth quarter of fiscal 2024?", "Q4", "Q4"),   # the tool answers Q4
])
def test_a_fact_keeps_the_models_quarter_unless_the_question_asks_for_a_year(question, planned, expected):
    year = 2024 if "2024" in question and "ended" not in question else 2025   # ended Sep 30, 2024 = MSFT FY2025 Q1
    out, _ = run(plan(fact(year=year, period=planned)), question)
    assert args(out)["fiscal_period"] == expected


def test_a_question_mixing_periods_keeps_each_steps_period():
    question = "How did Apple's total assets change from the end of fiscal 2024 to Q3 of fiscal 2025?"
    out, notes = run(plan(fact("AAPL", 2024, "FY", "total_assets"), fact("AAPL", 2025, "Q3", "total_assets")),
                     question)
    assert [(a["fiscal_year"], a["fiscal_period"]) for a in (args(out, 0), args(out, 1))] == [(2024, "FY"),
                                                                                                (2025, "Q3")]
    assert notes == [] and not out.refuse and out.refusal_rule is None
    both, _ = run(plan(fact("AAPL", 2024, "Q4"), fact("AAPL", 2024, "FY")),
                  "What were Apple's fourth-quarter and full-year revenue in fiscal 2024?")
    assert [args(both, i)["fiscal_period"] for i in (0, 1)] == ["Q4", "FY"]


def test_a_follow_up_takes_its_period_cue_from_the_earlier_turn():
    history = ["What was Apple's revenue in the fourth quarter of fiscal 2024?"]
    out, _ = run(plan(fact("AAPL", 2024, "Q4", "operating_income")), "And its operating income?", history)
    assert args(out)["fiscal_period"] == "Q4"
    year_only, _ = run(plan(fact("AAPL", 2025, "Q4")), "What about fiscal 2025?", history)
    assert args(year_only)["fiscal_period"] == "FY"                    # a turn naming a year asks about the year


@pytest.mark.parametrize(("question", "period", "duration"), [
    ("What was Meta's revenue in Q2 fiscal 2025?", "Q2", None),                     # the tool picks 3M
    ("What was Meta's revenue for the six months ended June 30, 2025?", "Q2", "6M"),
    ("What was Meta's year-to-date revenue in Q3 fiscal 2025?", "Q3", "9M"),
    ("What was Meta's H1 fiscal 2025 revenue?", "Q2", "6M"),
    ("How much cash did Meta hold at the end of fiscal 2025?", "FY", None),          # point in time: the tool's
])
def test_the_tool_chooses_the_duration_unless_year_to_date_is_asked(question, period, duration):
    out, _ = run(plan(fact(period=period, duration="12M")), question)
    assert (args(out)["fiscal_period"], args(out)["duration"]) == (period, duration)


def test_searches_return_the_top_five_and_their_form_follows_the_period():
    out, _ = run(plan(search(ticker="KO", fiscal_year=2024, form="10-Q", k=1)),
                 "What did Coca-Cola say about sugar in fiscal 2024?")
    assert (args(out)["fiscal_period"], args(out)["form"], args(out)["k"]) == ("FY", "10-K", pl.SEARCH_K)
    question = "What did Meta say about AI in the second quarter of fiscal 2025?"
    out, _ = run(plan(search(ticker="META", fiscal_year=2025, fiscal_period="Q2", form="10-K")), question)
    assert (args(out)["fiscal_period"], args(out)["form"]) == ("Q2", "10-Q")


def test_a_search_for_an_unindexed_quarter_searches_the_whole_year():
    question = "What did Apple say about services in fiscal 2024 and in its Q2 fiscal 2025 report?"
    out, _ = run(plan(search(ticker="AAPL", fiscal_year=2024, fiscal_period="Q2", form="10-Q")), question)
    assert (args(out)["fiscal_year"], args(out)["fiscal_period"], args(out)["form"]) == (2024, None, None)
    refused, _ = run(plan(search(ticker="AAPL", fiscal_year=2024, fiscal_period="Q2")),
                     "What did Apple say about services in Q2 of fiscal 2024?")      # only an unindexed quarter
    assert refused.refuse and refused.refusal_rule == "R3"


def test_a_guessed_item_is_dropped_unless_the_question_names_one():
    out, _ = run(plan(search(ticker="MSFT", item="Item 1")), "What drove Microsoft's revenue?")
    assert args(out)["item"] is None
    out, _ = run(plan(search(ticker="AAPL", item="Item 7")), "What does Item 7 of Apple's 10-K say?")
    assert args(out)["item"] == "Item 7"


def test_redundant_compare_periods_steps_are_dropped():
    out, _ = run(plan(fact(year=2025), compare(years=[2023]), compare(years=[2025]), compare(years=[2023, 2024, 2025])),
                 "Meta's revenue trend from fiscal 2023 to fiscal 2025?")
    assert [s.tool for s in out.steps] == ["get_financial_fact", "compare_periods"]
    assert args(out, 1)["years"] == [2023, 2024, 2025]


# ----------------------------------------------------------------------------------------------- scope
def test_steps_for_unindexed_periods_are_dropped_and_declined():
    question = "What was Apple's Q2 revenue in fiscal 2024 and fiscal 2025?"
    out, notes = run(plan(fact("AAPL", 2024, "Q2"), fact("AAPL", 2025, "Q2")), question)
    assert [args(out, i)["fiscal_year"] for i in range(len(out.steps))] == [2025]
    assert out.partial and out.refusal_rule == "R3"
    assert notes == ["planner: steps for fiscal 2024 Q2 dropped (not indexed); the rest answered (R3)"]


def test_a_forecast_part_is_declined_as_r2():
    question = "What were Amazon's net sales in fiscal 2024, and what will they be in fiscal 2026?"
    out, _ = run(plan(fact("AMZN", 2024), fact("AMZN", 2026)), question)
    assert out.partial and out.refusal_rule == "R2" and len(out.steps) == 1


def test_compare_periods_keeps_only_indexed_years():
    out, _ = run(plan(compare(years=[2021, 2023, 2024])), "Meta's revenue in fiscal 2021, 2023 and 2024?")
    assert args(out)["years"] == [2023, 2024] and out.refusal_rule == "R3"


def test_nothing_left_is_a_refusal():
    out, _ = run(plan(fact("AAPL", 2024, "Q1"), fact("AAPL", 2024, "Q2")),
                 "What was Apple's revenue in Q1 and Q2 of fiscal 2024?")
    assert out.refuse and out.refusal_rule == "R3" and out.steps == []


def test_an_emptied_partial_refusal_becomes_a_refusal():
    out, _ = run(plan(compare(years=[2025]), rule="R1"), "What was Apple's revenue trend in fiscal 2025, and should I buy?")
    assert out.refuse and out.refusal_rule == "R1" and out.steps == [] and out.calculation is None


def test_finalize_returns_a_valid_plan_without_duplicates():
    out, _ = run(plan(fact("AAPL", 2025, "Q4"), fact("AAPL", 2025, "Q1")), "What was Apple's revenue in fiscal 2025?")
    assert len(out.steps) == 1 and args(out)["fiscal_period"] == "FY"
    assert Plan.model_validate(out.model_dump()) == out


@pytest.mark.parametrize(("question", "years"), [
    ("What was Apple's revenue in 2020?", (2020,)),                       # a bare year is the fiscal year: R3
    ("What does Microsoft say about its 2030 carbon-negative commitment?", ()),
    ("What does Apple's latest 10-K say about its notes due 2030?", (2025,)),
    ("How did Walmart describe the 2008 financial crisis in fiscal 2024?", (2024,)),
    ("What were Pfizer's revenues in fiscal years 2019 and 2020?", (2019, 2020)),
    ("What will Apple's revenue be in FY2030?", (2030,)),
])
def test_only_dates_in_the_text_are_not_fiscal_years(question, years):
    assert pl.planning_entities(from_turns([question]), question).years == years


def test_a_bare_unindexed_year_is_refused_by_the_scope_check():
    out, notes = run(plan(fact("AAPL", 2020)), "What was Apple's revenue in 2020?")
    assert out.refuse and out.refusal_rule == "R3" and "fiscal 2020 FY" in notes[0]


# ----------------------------------------------------------------------------------------------- entities
def test_searches_get_the_questions_companies_year_and_period():
    out, _ = run(plan(search()), "What drove Meta's revenue in Q2 fiscal 2025?")
    assert (args(out)["ticker"], args(out)["fiscal_year"], args(out)["fiscal_period"]) == ("META", 2025, "Q2")


def test_every_company_the_question_names_is_searched():
    question = "What risks do Apple and Microsoft describe about AI in fiscal 2024?"
    out, _ = run(plan(search(ticker="AAPL", query="AI risks")), question)
    assert sorted(a["ticker"] for a in (s.arguments for s in out.steps)) == ["AAPL", "MSFT"]
    wrong, _ = run(plan(search(ticker="GOOGL", query="AI risks")), question)   # a company it does not name
    assert sorted(s.arguments["ticker"] for s in wrong.steps) == ["AAPL", "MSFT"]


def test_entities_leave_refusals_and_facts_alone_and_the_cap_is_noted():
    refusal = Plan(reasoning="x", refuse=True, refusal_rule="R1", steps=[])
    assert pl.apply_entities(refusal, Entities(("AAPL",), (2024,), ())) is refusal
    facts = plan(fact("MSFT", 2023))
    assert pl.apply_entities(facts, Entities(("MSFT",), (2023,), ())).steps == facts.steps
    many, notes = pl.finalize(plan(*[search(query=f"q{n}") for n in range(5)]),
                              Entities(("AAPL", "MSFT", "KO"), (2024,), ()), "Compare their fiscal 2024 risks.")
    assert len(many.steps) == 10 and notes == ["planner: 5 step(s) beyond the 10-step cap were dropped"]


def test_a_plan_decoded_with_nulls_validates_for_every_tool():
    """What strict guided decoding writes (every argument present, null where optional) is a valid plan."""
    p = plan(search(), fact(), compare(), calculation="growth")
    assert [s.tool for s in p.steps] == ["search_filings", "get_financial_fact", "compare_periods"]


# ----------------------------------------------------------------------------------------------- calculation
@pytest.mark.parametrize(("question", "planned", "expected"), [
    ("What was Merck's operating margin in fiscal 2025?", "ratio", "percent"),      # the critic expects percent
    ("What was NVIDIA's gross margin in fiscal 2023?", None, "percent"),
    ("By what percentage did Citigroup's net income change from fiscal 2023 to 2024?", None, "growth"),
    ("What was Apple's current ratio in fiscal 2024?", "ratio", "ratio"),           # ratio or percent: kept
    ("What was Apple's revenue in fiscal 2024?", None, None),                       # nothing expected
])
def test_the_calculation_follows_what_the_critic_expects(question, planned, expected):
    assert pl.align_calculation(plan(fact(), calculation=planned), question).calculation == expected


# ----------------------------------------------------------------------------------------------- pin
def test_every_message_literal_is_pinned():
    literals = [v for k, v in vars(pl).items() if isinstance(v, str) and k.isupper()
                and k.endswith(("_LINE", "_HEADER", "_TEXT", "_HINT", "NONE_NAMED"))]
    assert set(literals) <= set(pl.MESSAGE_TEMPLATES) and len(literals) == len(pl.MESSAGE_TEMPLATES)


def test_the_pin_follows_the_messages_and_the_schema(monkeypatch):
    pinned = pl.prompt_sha256()
    monkeypatch.setattr(pl, "MESSAGE_TEMPLATES", (*pl.MESSAGE_TEMPLATES, "x"))
    assert pl.prompt_sha256() != pinned
    monkeypatch.undo()
    monkeypatch.setattr(pl, "plan_json_schema", lambda: {"changed": True})
    assert pl.prompt_sha256() != pinned


def _copy_sources(tmp_path, edit):
    for name in pl.RULE_SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(edit(name, (pl.REPO_ROOT / name).read_text(encoding="utf-8")).encode("utf-8"))


def test_the_rules_hash_ignores_layout_docstrings_comments_and_the_pin(tmp_path, monkeypatch):
    original = pl.rules_sha256()

    def cosmetic(name, text):
        text = re.sub(r'(?m)^PROMPT_SHA256 = "', 'PROMPT_SHA256 = "other', text)
        text = text.replace('"""The entities without', '"""Reworded. The entities without')
        return text.replace("\n", "  # a comment\n", 3).replace("\n", "\r\n")
    _copy_sources(tmp_path, cosmetic)
    monkeypatch.setattr(pl, "REPO_ROOT", Path(tmp_path))
    assert pl.rules_sha256() == original


def test_the_rules_hash_changes_with_the_code(tmp_path, monkeypatch):
    original = pl.rules_sha256()
    _copy_sources(tmp_path, lambda name, text: text.replace("SEARCH_K = 5", "SEARCH_K = 4"))
    monkeypatch.setattr(pl, "REPO_ROOT", Path(tmp_path))
    assert pl.rules_sha256() != original


def test_no_control_characters_in_the_planner_sources():
    """A stray backspace (\b written through a shell) silently broke a regex twice while this was written."""
    names = (*pl.RULE_SOURCES, "agent/state.py", "agent/prompts/planner.txt", "eval/run_planner.py")
    for name in names:
        text = (pl.REPO_ROOT / name).read_text(encoding="utf-8")
        bad = sorted({hex(ord(c)) for c in text if ord(c) < 32 and c not in "\n\t"})
        assert not bad, (name, bad)


def test_the_runner_scores_company_coverage_not_just_membership():
    from eval.run_planner import entity_filters_used
    one = plan(search(ticker="AAPL"))
    assert entity_filters_used(one, ("AAPL", "MSFT")) is False           # one of two named companies
    assert entity_filters_used(plan(search(ticker="AAPL"), search(ticker="MSFT")), ("AAPL", "MSFT")) is True
    assert entity_filters_used(plan(search(ticker="KO")), ("AAPL",)) is False
    assert entity_filters_used(plan(fact()), ("META",)) is None


SAMPLE_SOURCE = '''"""Module docstring."""
import re  # a comment

PROMPT_SHA256 = "abc123"


@staticmethod
def f(x, *, y=2):
    """Function docstring."""
    return (x +
            y), f"{x}-{y!r}"


class C:
    """Class docstring."""

    def g(self):
        return "not a docstring"
'''


@pytest.mark.skipif(sys.version_info < (3, 12), reason="f-strings are tokenised from Python 3.12 on")
def test_the_rules_code_is_the_same_on_every_python_from_3_12():
    """Pinned token output: CI (3.12) and development (3.14) must agree, which ast.dump did not (3.13 changed it)."""
    assert pl._code(SAMPLE_SOURCE) == (
        '<nl> import re <nl> PROMPT_SHA256 = "" <nl> @ staticmethod <nl> def f ( x , * , y = 2 ) : <nl> <in> <nl> '
        'return ( x + y ) , f" { x } - { y ! r } " <nl> <de> class C : <nl> <in> <nl> def g ( self ) : <nl> <in> '
        'return "not a docstring" <nl> <de> <de>')
