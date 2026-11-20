"""W2-10: the Streamlit UI. Offline: the API client against httpx's MockTransport, the card's rendering as plain
functions, and the app itself through Streamlit's AppTest with a fake API client (no server, no endpoint)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx
import pytest

from ui import client as api
from ui import render

APP = Path(__file__).resolve().parents[1] / "ui" / "app.py"
SEC = "https://www.sec.gov/Archives/edgar/data/886982/000088698224000007/gs-20231231.htm"
BODY = {
    "session_id": "sid-1", "cached": False, "seconds": 3.2, "history": [],
    "answer": {"answer": "Goldman Sachs' total net revenues in fiscal 2023 were $46,254 million.", "verified": True,
               "refused": False, "refusal_rule": None, "notes": [],
               "values": [{"name": "revenue", "value": 46254.0, "unit": "USD millions", "ticker": "GS",
                           "fiscal_year": 2023, "fiscal_period": "FY"}],
               "calculations": [],
               "citations": [{"source_type": "xbrl", "ticker": "GS", "form": "10-K", "fiscal_year": 2023,
                              "fiscal_period": "FY", "concept": "us-gaap:RevenuesNetOfInterestExpense", "url": SEC}]},
    "run": {"verdict": "verified", "steps": 1, "tokens": 2685, "seconds": 3.1, "visits": {"planner": 1, "critic": 1},
            "stop_reason": None,
            "tool_calls": [{"tool": "get_financial_fact", "arguments": {"ticker": "GS"}, "error": None}]},
}


def body(**answer) -> dict:
    b = copy.deepcopy(BODY)
    b["answer"].update(answer)
    return b


# ----------------------------------------------------------------------------------------------- the API client
def transport(handler):
    return httpx.MockTransport(handler)


def test_ask_posts_the_question_and_sends_the_session():
    seen = {}

    def handler(request):
        seen["body"], seen["sid"] = json.loads(request.content), request.headers.get(api.SESSION_HEADER)
        return httpx.Response(200, json=BODY, headers={api.SESSION_HEADER: "sid-1"})

    reply = api.ApiClient("http://api", transport=transport(handler)).ask("Q?", session_id="sid-0")
    assert reply.ok and reply.body == BODY and reply.session_id == "sid-1"
    assert seen == {"body": {"question": "Q?"}, "sid": "sid-0"}


def test_no_session_header_on_a_first_question():
    seen = {}

    def handler(request):
        seen["sid"] = request.headers.get(api.SESSION_HEADER)
        return httpx.Response(200, json=BODY)

    api.ApiClient("http://api", transport=transport(handler)).ask("Q?")
    assert seen["sid"] is None


@pytest.mark.parametrize(("status", "headers", "payload", "says"), [
    (429, {"Retry-After": "42"}, {"detail": "rate limit exceeded: 10 per 1 minute"}, "Try again in 42 seconds"),
    (422, {}, {"detail": [{"msg": "String should have at least 1 character"}]}, "at least 1 character"),
    (503, {"Retry-After": "30"}, {"detail": "busy"}, "busy or still starting"),
    (500, {}, {"detail": "internal error"}, "internal error"),
    (413, {}, {"detail": "request body too large"}, "too long"),
    (418, {}, {"detail": "teapot"}, "418: teapot"),
])
def test_a_refused_request_is_a_message(status, headers, payload, says):
    reply = api.ApiClient("http://api", transport=transport(
        lambda r: httpx.Response(status, json=payload, headers=headers))).ask("Q?")
    assert not reply.ok and reply.status == status and says in reply.error


def test_an_unreachable_or_slow_api_is_a_message():
    def down(request):
        raise httpx.ConnectError("refused")

    def slow(request):
        raise httpx.ReadTimeout("slow")

    assert "not reachable at http://api" in api.ApiClient("http://api", transport=transport(down)).ask("Q?").error
    assert "took too long" in api.ApiClient("http://api", transport=transport(slow)).ask("Q?").error


def test_health_reads_the_body_even_while_starting():
    starting = {"status": "starting", "warm_up": "pending"}
    c = api.ApiClient("http://api", transport=transport(lambda r: httpx.Response(503, json=starting)))
    assert c.health() == starting

    def down(request):
        raise httpx.ConnectError("refused")

    assert api.ApiClient("http://api", transport=transport(down)).health() is None


def test_the_api_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("FINSIGHT_API_URL", "http://api.example:9000/")
    assert api.api_url() == "http://api.example:9000"
    monkeypatch.delenv("FINSIGHT_API_URL")
    assert api.api_url() == api.DEFAULT_API_URL


# ----------------------------------------------------------------------------------------------- the card
def test_escape_keeps_dollar_figures_out_of_latex():
    assert render.escape("$46,254 million and $53,512 million") == r"\$46,254 million and \$53,512 million"
    assert render.escape("a_b *c* [d](e) <x>") == r"a\_b \*c\* \[d\](e) \<x\>"


@pytest.mark.parametrize(("answer", "label", "color"), [
    ({"verified": True}, "Verified", "green"),
    ({"verified": False}, "Not verified", "orange"),
    ({"refused": True, "refusal_rule": "R2"}, "Refused (R2)", "blue"),
    ({"verified": True, "refusal_rule": "R1"}, "Verified, part refused (R1)", "green"),
    ({"verified": False, "refusal_rule": "R1"}, "Not verified, part refused (R1)", "orange"),
])
def test_badge_says_the_verdict_in_words(answer, label, color):
    assert render.badge(answer)[:2] == (label, color)


@pytest.mark.parametrize(("value", "unit", "shown"), [
    (46254.0, "USD millions", "$46,254 million"), (22.1, "%", "22.1%"), (6.08, "USD per share", "$6.08 per share"),
    (-8516.0, "USD millions", "-$8,516 million"), (1.23, "ratio", "1.23"), (42, "USD", "$42"),
    (15000.0, "shares millions", "15,000 million shares"), (3, "count", "3"), (1.5, "USD billions", "$1.5 billion"),
])
def test_number(value, unit, shown):
    assert render.number(value, unit) == shown


def test_figures_calculations_and_citations():
    answer = body(calculations=[{"operation": "growth", "inputs": [8516.0, 14276.0], "result": 67.6, "unit": "%",
                                 "expression": "(14276 - 8516) / 8516 * 100"}],
                  citations=[BODY["answer"]["citations"][0],
                             {"source_type": "filing", "ticker": "GS", "form": "10-Q", "fiscal_year": 2025,
                              "fiscal_period": "Q2", "item": "Part I Item 2", "url": "https://evil.example/x"}])["answer"]
    assert render.figure_rows(answer) == [{"figure": "revenue", "company": "GS", "period": "FY2023",
                                           "value": "$46,254 million"}]
    assert render.calculation_lines(answer) == ["**growth** of 8,516, 14,276: `(14276 - 8516) / 8516 * 100` = "
                                                "**67.6%**"]
    links = render.citation_links(answer)
    assert links[0] == f"[GS 10-K FY2023 · XBRL RevenuesNetOfInterestExpense]({SEC})"
    assert links[1] == "GS 10-Q Q2 FY2025 · Part I Item 2"          # not an SEC URL: shown, not linked


def test_trace_summarises_the_run():
    t = render.trace(body() | {"cached": True, "history": ["Earlier?"]})
    assert t["verdict"] == "verified" and t["cached"] and t["tokens"] == 2685 and t["steps"] == 1
    assert t["nodes"] == "planner ×1, critic ×1" and t["history"] == ["Earlier?"]
    assert t["tool_calls"] == [{"tool": "get_financial_fact", "arguments": '{"ticker": "GS"}', "result": "ok"}]
    failed = copy.deepcopy(BODY)
    failed["run"]["tool_calls"][0]["error"] = "not_reported"
    assert render.trace(failed)["tool_calls"][0]["result"] == "not_reported"      # an error shows as its type


def test_scope_hints_come_from_the_corpus_scope():
    hints = render.scope_hints()
    assert len(hints["companies"]) == 20 and ("Apple Inc.", "AAPL") in hints["companies"]
    assert hints["tenk_years"] == [2023, 2024, 2025] and hints["tenq_quarters"] == ["Q1 FY2025", "Q2 FY2025",
                                                                                    "Q3 FY2025"]


def test_example_questions_are_never_gold():
    gold = Path(__file__).resolve().parents[1] / "eval" / "gold_set.jsonl"
    questions = {json.loads(line)["question"] for line in gold.read_text(encoding="utf-8").splitlines() if line}
    assert not questions & set(render.EXAMPLE_QUESTIONS)


def test_the_trace_link_uses_the_sessions_hash(monkeypatch):
    import hashlib
    url = render.session_trace_url("sid-1", "https://lf.example/project/p/sessions/")
    assert url == "https://lf.example/project/p/sessions/" + hashlib.sha256(b"sid-1").hexdigest()[:16]
    assert "sid-1" not in url
    monkeypatch.delenv("FINSIGHT_TRACE_SESSIONS_URL", raising=False)
    assert render.session_trace_url("sid-1") is None and render.session_trace_url(None, "https://x") is None


def test_the_hash_matches_the_apis():
    from api.sessions import session_digest
    url = render.session_trace_url("abc.def", "https://lf/")
    assert url == "https://lf/" + session_digest("abc.def")


# ----------------------------------------------------------------------------------------------- the app
class FakeApi:
    """ui.client.ApiClient's stand-in: scripted replies, recorded questions and session ids."""
    base_url = "http://fake-api"

    def __init__(self, *replies, health=None):
        self.replies = list(replies)
        self.calls: list[tuple[str, str | None]] = []
        self._health = {"status": "ok", "model": "fake-model"} if health is None else health

    def ask(self, question, session_id=None):
        self.calls.append((question, session_id))
        return self.replies.pop(0)

    def health(self):
        return self._health or None


def ok(b=None, sid="sid-1"):
    return api.Reply(ok=True, status=200, body=b or BODY, session_id=sid)


def app(fake):
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["client"] = fake
    return at.run()


def markdown(at) -> list[str]:
    return [m.value for m in at.markdown]


def test_the_sidebar_has_the_disclaimer_the_status_and_the_scope():
    at = app(FakeApi())
    assert not at.exception
    assert at.sidebar.info[0].value == render.DISCLAIMER and "API ready" in at.sidebar.success[0].value
    labels = [b.label for b in at.sidebar.button]
    assert labels[0] == "New conversation" and labels[1:] == list(render.EXAMPLE_QUESTIONS)
    assert any("Apple Inc. (AAPL)" in c.value for c in at.sidebar.caption)


def test_an_unreachable_api_is_shown():
    at = app(FakeApi(health={}))
    assert "API not reachable at http://fake-api" in at.sidebar.error[0].value


def test_an_answer_is_a_card_with_badge_figures_sources_and_trace():
    fake = FakeApi(ok())
    at = app(fake)
    at.chat_input[0].set_value("What were Goldman Sachs' total net revenues in fiscal 2023?").run()
    md = markdown(at)
    assert not at.exception and fake.calls == [("What were Goldman Sachs' total net revenues in fiscal 2023?", None)]
    assert ":green-badge[:material/verified: Verified]" in md
    assert r"Goldman Sachs' total net revenues in fiscal 2023 were \$46,254 million." in md
    assert any(SEC in m and m.startswith("**Sources**") for m in md)
    assert [e.label for e in at.expander] == ["Show trace"]
    assert [(m.label, m.value) for m in at.metric] == [("Verdict", "verified"), ("Tool calls", "1"),
                                                       ("Tokens", "2,685"), ("Seconds", "3.1")]
    assert len(at.dataframe) == 2                                    # the figures and the tool calls
    assert at.session_state["session_id"] == "sid-1"


def test_a_follow_up_sends_the_latest_session_id():
    fake = FakeApi(ok(sid="sid-1"), ok(body() | {"history": ["Q1?"]}, sid="sid-2"), ok(sid="sid-2"))
    at = app(fake)
    at.chat_input[0].set_value("Q1?").run()
    at.chat_input[0].set_value("And in fiscal 2024?").run()
    at.chat_input[0].set_value("And in fiscal 2025?").run()
    assert [sid for _, sid in fake.calls] == [None, "sid-1", "sid-2"]  # each response's id replaces the last
    assert len([c for c in at.chat_message if c.name == "assistant"]) == 3


def test_an_error_is_shown_and_the_conversation_goes_on():
    fake = FakeApi(api.Reply(ok=False, status=429, error="Too many questions in a short time. Try again in 42 "
                                                          "seconds."), ok())
    at = app(fake)
    at.chat_input[0].set_value("Q1?").run()
    assert "Try again in 42 seconds" in at.error[0].value
    at.chat_input[0].set_value("Q2?").run()
    assert not at.exception and len(at.expander) == 1


@pytest.mark.parametrize(("answer", "badge"), [
    ({"refused": True, "verified": True, "refusal_rule": "R2", "values": [], "citations": []},
     ":blue-badge[:material/block: Refused (R2)]"),
    ({"verified": False, "notes": ["could not verify: invariant 4: WFC has no citation"]},
     ":orange-badge[:material/warning: Not verified]"),
])
def test_refusals_and_unverified_answers_say_so(answer, badge):
    at = app(FakeApi(ok(body(**answer))))
    at.chat_input[0].set_value("Q?").run()
    assert badge in markdown(at)
    if answer.get("notes"):
        assert any("could not verify" in c.value for c in at.caption)


def test_a_cached_answer_is_marked():
    at = app(FakeApi(ok(body() | {"cached": True, "seconds": 0.01})))
    at.chat_input[0].set_value("Q?").run()
    assert ":gray-badge[:material/bolt: From cache]" in markdown(at)
    assert ("Seconds", "0.01") in [(m.label, m.value) for m in at.metric]


def test_an_example_question_is_asked_and_new_conversation_clears():
    fake = FakeApi(ok(), ok())
    at = app(fake)
    at.sidebar.button[1].click().run()
    assert fake.calls == [(render.EXAMPLE_QUESTIONS[0], None)]
    at.sidebar.button[0].click().run()                               # New conversation
    assert at.session_state["messages"] == [] and at.session_state["session_id"] is None
    assert not [c for c in at.chat_message]


def test_the_trace_link_appears_when_configured(monkeypatch):
    monkeypatch.setenv("FINSIGHT_TRACE_SESSIONS_URL", "https://lf.example/project/p/sessions")
    at = app(FakeApi(ok()))
    at.chat_input[0].set_value("Q?").run()
    buttons = at.get("link_button")
    assert len(buttons) == 1 and "lf.example/project/p/sessions/" in buttons[0].proto.url


def test_a_question_with_markdown_is_shown_as_typed():
    at = app(FakeApi(ok()))
    at.chat_input[0].set_value("What was *Apple*'s $ revenue?").run()
    assert r"What was \*Apple\*'s \$ revenue?" in markdown(at)
