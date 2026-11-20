"""W2-06: memory across turns. A LangGraph checkpointer thread keeps the conversation, so a follow-up keeps the
company, year and period of the turns before (retrieval.entities.from_turns). Offline: the stub nodes (their
planner reads entities with from_turns over the state's history) and an in-memory checkpointer.

The chains follow the gold set's follow-up structure (a first turn naming the company and year, then follow-ups that
name only the year, or only the company), built from development questions only (dev-A02/dev-F01, dev-A03/dev-F02,
pdev-34)."""
from __future__ import annotations

import asyncio

import pytest

from agent import graph as g
from agent.stubs import stub_nodes

LIMITS = g.Limits(max_steps=10, max_tokens=12_000, timeout_s=10)


def conversation():
    graph = g.build_graph(stub_nodes(), checkpointer=g.in_memory_checkpointer())

    def ask(question, session="s1", **kw):
        return asyncio.run(g.run(question, graph=graph, limits=LIMITS, config=g.thread(session), **kw))

    return graph, ask


def seen(result):
    e = result.state["entities"]
    return e.tickers, e.years, e.periods


def test_a_three_turn_follow_up_keeps_the_right_entities():
    """Company and year, then a pronoun follow-up that names nothing (dev-F01), then year only, then company only."""
    _, ask = conversation()
    first = ask("What was UnitedHealth's total revenue in fiscal 2023?")                           # dev-A02's shape
    second = ask("And what were its net earnings attributable to common shareholders that year?")  # dev-F01
    third = ask("And in fiscal 2024?")
    fourth = ask("What about Pfizer?")
    assert seen(first) == (("UNH",), (2023,), ())
    assert seen(second) == (("UNH",), (2023,), ())             # "its" and "that year": the turn before's
    assert seen(third) == (("UNH",), (2024,), ())              # only the year changes
    assert seen(fourth) == (("PFE",), (2024,), ())             # only the company changes
    assert fourth.state["history"] == [first.state["question"], second.state["question"], third.state["question"]]


def test_a_quarter_follow_up_keeps_the_company_and_year():
    _, ask = conversation()
    ask("What was Meta's revenue in the second quarter of fiscal 2025?")                          # dev-A03's shape
    assert seen(ask("And in the third quarter?")) == (("META",), (2025,), ("Q3",))                 # dev-F02
    assert seen(ask("And Microsoft?")) == (("MSFT",), (2025,), ("Q3",))


def test_a_follow_up_naming_a_year_drops_the_earlier_quarter():
    _, ask = conversation()
    ask("What was Pfizer's revenue in the third quarter of fiscal 2025?")
    assert seen(ask("What about fiscal 2023?")) == (("PFE",), (2023,), ())                         # pdev-34's shape


def test_each_session_is_its_own_conversation():
    _, ask = conversation()
    ask("What was Apple's revenue in fiscal 2024?", session="alice")
    ask("What was Walmart's revenue in fiscal 2025?", session="bob")
    assert seen(ask("And in fiscal 2023?", session="alice")) == (("AAPL",), (2023,), ())
    assert seen(ask("And in fiscal 2023?", session="bob")) == (("WMT",), (2023,), ())


def test_a_new_turn_inherits_no_draft_counter_or_tool_result():
    graph, ask = conversation()
    ask("What was Apple's revenue in fiscal 2024?")
    second = ask("And in fiscal 2023?")
    assert second.state["visits"]["planner"] == 1 and second.state["tokens"] == 0
    assert second.answer.answer.startswith("FinSight's agent skeleton")


def test_an_explicit_history_is_used_as_given():
    """The evaluation replays a gold chain with its own history; the thread's turns are not mixed in."""
    _, ask = conversation()
    ask("What was Apple's revenue in fiscal 2024?")
    result = ask("And in fiscal 2023?", history=["What was Costco's revenue in fiscal 2024?"])
    assert result.state["history"] == ["What was Costco's revenue in fiscal 2024?"]
    assert seen(result) == (("COST",), (2023,), ())


def test_without_a_thread_there_is_no_memory():
    graph = g.build_graph(stub_nodes())                         # no checkpointer
    asyncio.run(g.run("What was Apple's revenue in fiscal 2024?", graph=graph, limits=LIMITS))
    result = asyncio.run(g.run("And in fiscal 2023?", graph=graph, limits=LIMITS))
    assert result.state["history"] == [] and seen(result) == ((), (2023,), ())


def test_the_history_is_capped():
    _, ask = conversation()
    for year in range(2019, 2019 + g.MAX_HISTORY_TURNS + 3):
        result = ask(f"What was Apple's revenue in fiscal {year}?")
    assert len(result.state["history"]) == g.MAX_HISTORY_TURNS
    assert result.state["history"][-1].endswith(f"fiscal {year - 1}?")
    long = ask("And in fiscal 2023?", history=[f"q{i}" for i in range(20)])
    assert long.state["history"] == [f"q{i}" for i in range(20 - g.MAX_HISTORY_TURNS, 20)]


def test_a_thread_needs_a_session_id():
    with pytest.raises(ValueError):
        g.thread(" ")
    assert g.thread("abc") == {"configurable": {"thread_id": "abc"}}


def test_thread_history_without_a_checkpointer_or_thread_is_empty():
    graph = g.build_graph(stub_nodes())
    assert asyncio.run(g.thread_history(graph, g.thread("x"))) == []
    graph = g.build_graph(stub_nodes(), checkpointer=g.in_memory_checkpointer())
    assert asyncio.run(g.thread_history(graph, None)) == []
    assert asyncio.run(g.thread_history(graph, g.thread("never-used"))) == []
