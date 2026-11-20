"""W2-10: the Streamlit UI. Offline: the API client against httpx's MockTransport, the card's rendering as plain
functions, and the app itself through Streamlit's AppTest with a fake API client (no server, no endpoint)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from streamlit.runtime.scriptrunner_utils.script_requests import RerunData
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from streamlit.testing.v1 import AppTest

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


def passage(item="Item 7", url=SEC) -> dict:
    return {"source_type": "filing", "ticker": "MSFT", "form": "10-K", "fiscal_year": 2023, "fiscal_period": "FY",
            "item": item, "url": url}


# ----------------------------------------------------------------------------------------------- the API client
def transport(handler):
    return httpx.MockTransport(handler)


def client(handler) -> api.ApiClient:
    return api.ApiClient("http://api", transport=transport(handler))


def test_ask_posts_the_question_and_sends_the_session():
    seen = {}

    def handler(request):
        seen["body"], seen["sid"] = json.loads(request.content), request.headers.get(api.SESSION_HEADER)
        return httpx.Response(200, json=BODY, headers={api.SESSION_HEADER: "sid-1"})

    reply = client(handler).ask("Q?", session_id="sid-0")
    assert reply.ok and reply.body == BODY and reply.session_id == "sid-1"
    assert seen == {"body": {"question": "Q?"}, "sid": "sid-0"}


def test_no_session_header_on_a_first_question():
    seen = {}

    def handler(request):
        seen["sid"] = request.headers.get(api.SESSION_HEADER)
        return httpx.Response(200, json=BODY)

    client(handler).ask("Q?")
    assert seen["sid"] is None


@pytest.mark.parametrize(("status", "headers", "payload", "says"), [
    (429, {"Retry-After": "42"}, {"detail": "rate limit exceeded: 10 per 1 minute"}, "Try again in 42 seconds"),
    (429, {}, {"detail": "rate limit exceeded"}, "Try again in 60 seconds"),
    (422, {}, {"detail": [{"msg": "String should have at least 1 character"}]}, "at least 1 character"),
    (503, {"Retry-After": "30"}, {"detail": "busy: too many questions at once, try again shortly"},
     "busy with other questions. Try again in 30 seconds"),
    (503, {"Retry-After": "7"}, {"detail": "starting up"}, "still starting. Try again in 7 seconds"),
    (503, {"Retry-After": "soon"}, {"detail": "busy"}, "Try again in 30 seconds"),
    (500, {}, {"detail": "internal error"}, "internal error"),
    (413, {}, {"detail": "request body too large"}, "too long"),
    (418, {}, {"detail": "teapot"}, "418: teapot"),
    (418, {}, ["not", "a", "dict"], "418: "),
])
def test_a_refused_request_is_a_message(status, headers, payload, says):
    reply = client(lambda r: httpx.Response(status, json=payload, headers=headers)).ask("Q?")
    assert not reply.ok and reply.status == status and says in reply.error


def test_a_refused_request_keeps_the_session_id():
    reply = client(lambda r: httpx.Response(503, json={"detail": "busy"}, headers={api.SESSION_HEADER: "sid-9"})).ask("Q?")
    assert not reply.ok and reply.session_id == "sid-9"


@pytest.mark.parametrize("response", [
    httpx.Response(200, text="<html>proxy error</html>"),
    httpx.Response(200, json=["a", "list"]),
    httpx.Response(200, json={"session_id": "s", "run": {}}),                      # no answer
    httpx.Response(200, json={"answer": {"answer": "x"}}),                          # no run
    httpx.Response(200, json={"answer": {"verified": True}, "run": {}}),            # no answer text
])
def test_a_response_that_is_not_an_answer_is_a_message(response):
    reply = client(lambda r: response).ask("Q?")
    assert not reply.ok and reply.status == 200 and "not an answer" in reply.error


def test_an_unreachable_or_slow_api_is_a_message():
    def down(request):
        raise httpx.ConnectError("refused")

    def slow(request):
        raise httpx.ReadTimeout("slow")

    assert "not reachable at http://api" in client(down).ask("Q?").error
    assert "took too long" in client(slow).ask("Q?").error


def test_any_other_failure_while_sending_is_a_message():
    def broken(request):
        raise RuntimeError("not an httpx error")

    reply = client(broken).ask("Q?")
    assert not reply.ok and "could not be sent" in reply.error


def test_connecting_is_bounded_separately_from_the_answer():
    timeout = api.ApiClient("http://api")._http.timeout
    assert timeout.connect == api.CONNECT_TIMEOUT_S and timeout.read == api.TIMEOUT_S


def test_health_reads_the_body_even_while_starting():
    starting = {"status": "starting", "warm_up": "pending"}
    assert client(lambda r: httpx.Response(503, json=starting)).health() == starting

    def down(request):
        raise httpx.ConnectError("refused")

    assert client(down).health() is None
    assert client(lambda r: httpx.Response(200, json=["ok"])).health() is None
    assert client(lambda r: httpx.Response(200, text="ok")).health() is None
    deep = "[" * 100_000 + "]" * 100_000                                         # past the recursion limit
    assert client(lambda r: httpx.Response(200, text=deep)).health() is None


@pytest.mark.parametrize("status", [200, 500, 503])
def test_a_body_nested_past_the_recursion_limit_is_a_message(status):
    deep = "[" * 100_000 + "]" * 100_000
    reply = client(lambda r: httpx.Response(status, text=deep, headers={"content-type": "application/json"})).ask("Q?")
    assert not reply.ok and reply.error


@pytest.mark.parametrize("bad", [123, ["x"], "séssion", "a\nb", "x" * 129, "a b", ""])
def test_a_session_id_without_the_apis_shape_is_ignored(bad):
    """Sent back, it would make every later question fail (httpx refuses it, or the API answers 422)."""
    reply = client(lambda r: httpx.Response(200, json=BODY | {"session_id": bad})).ask("Q?")
    assert reply.ok and reply.session_id is None
    if isinstance(bad, str) and bad.isascii() and "\n" not in bad:
        header = client(lambda r: httpx.Response(503, json={"detail": "busy"}, headers={api.SESSION_HEADER: bad}))
        assert header.ask("Q?").session_id is None
    seen = {}

    def handler(request):
        seen["sid"] = request.headers.get(api.SESSION_HEADER)
        return httpx.Response(200, json=BODY)

    assert client(handler).ask("Q?", session_id=bad).ok and seen["sid"] is None      # never sent either


def test_a_valid_session_id_from_the_header_is_used_when_the_body_has_none():
    b = {k: v for k, v in BODY.items() if k != "session_id"}
    reply = client(lambda r: httpx.Response(200, json=b, headers={api.SESSION_HEADER: "abc.DEF_1-2"})).ask("Q?")
    assert reply.session_id == "abc.DEF_1-2"


def test_a_retry_after_that_is_not_ascii_digits_falls_back():
    reply = client(lambda r: httpx.Response(429, json={"detail": "x"}, headers={"Retry-After": b"\xb2"})).ask("Q?")
    assert "Try again in 60 seconds" in reply.error


def test_the_api_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("FINSIGHT_API_URL", "http://api.example:9000/")
    assert api.api_url() == "http://api.example:9000"
    monkeypatch.delenv("FINSIGHT_API_URL")
    assert api.api_url() == api.DEFAULT_API_URL


# ----------------------------------------------------------------------------------------------- the card
def test_a_conversation_records_replies_and_keeps_the_latest_session_id():
    chat = api.Conversation()
    chat.ask("Q1?")
    assert chat.pending == "Q1?" and chat.messages == [{"role": "user", "text": "Q1?"}]
    chat.record(ok(sid="sid-1"))
    assert chat.pending is None and chat.session_id == "sid-1"
    assert chat.messages[-1] == {"role": "assistant", "body": BODY, "session_id": "sid-1"}
    chat.ask("Q2?")
    chat.record(api.Reply(ok=False, status=503, error="busy", session_id="sid-2"))
    assert chat.session_id == "sid-2" and chat.messages[-1] == {"role": "assistant", "error": "busy"}
    chat.ask("Q3?")
    chat.record(api.Reply(ok=False, error="not reachable"))                       # no id: the last one stays
    assert chat.session_id == "sid-2" and len(chat.messages) == 6
    chat.clear()
    assert chat == api.Conversation()


def test_escape_keeps_dollar_figures_out_of_latex():
    assert render.escape("$46,254 million and $53,512 million") == r"\$46\,254 million and \$53\,512 million"
    assert render.escape("a_b *c* [d](e) <x>") == r"a\_b \*c\* \[d\]\(e\) \<x\>"


@pytest.mark.parametrize(("text", "shown"), [
    ("See https://evil.example/x.", r"See `https://evil.example/x`\."),
    ("(www.evil.example)", r"\(`www.evil.example`\)"),
    ("[a](HTTPS://X.example/y)", r"\[a\]\(`HTTPS://X.example/y`\)"),
    ("ftp://f.example/a`b", "`ftp://f.example/ab`"),
    ("mail someone@example.com", "mail `someone@example.com`"),
    ("Item 7, Note 12", r"Item 7\, Note 12"),
])
def test_escape_turns_urls_into_code_spans(text, shown):
    """Streamlit links a bare URL or email even when its punctuation is escaped (checked in Chrome): a code span is
    never linked."""
    assert render.escape(text) == shown


@pytest.mark.parametrize("text", [
    "Revenue\n===", "Revenue\n---", "1. first", "- item", "+ item", "# heading", "![img](x.png)",
    ":material/star: :rocket: :red[x]", "&amp; &#36;", "`code`", "<b>bold</b>", "~strike~ | pipe |",
])
def test_escape_leaves_no_markdown_active(text):
    escaped = render.escape(text)
    unescaped = escaped.replace("\\", "")
    assert unescaped == text.replace("\\", "")
    assert all(escaped[i - 1] == "\\" for i, ch in enumerate(escaped) if ch in "$*_[]()<>#`:.-+=!&;/~|{}" and i)
    assert escaped[0] not in "$*_[]()<>#`:.-+=!&;/~|{}"


@pytest.mark.parametrize(("answer", "label", "color"), [
    ({"verified": True}, "Verified", "green"),
    ({"verified": False}, "Not verified", "orange"),
    ({"refused": True, "refusal_rule": "R2"}, "Refused: forecast (R2)", "blue"),
    ({"refused": True, "refusal_rule": "R6"}, "Refused: bypassing FinSight's rules (R6)", "blue"),
    ({"refused": True, "refusal_rule": "R9"}, "Refused: R9", "blue"),               # an unknown rule as it is
    ({"refused": True}, "Refused: policy", "blue"),
    ({"verified": True, "refusal_rule": "R1"}, "Verified, part refused: investment advice (R1)", "green"),
    ({"verified": False, "refusal_rule": "R1"}, "Not verified, part refused: investment advice (R1)", "orange"),
])
def test_badge_says_the_verdict_in_words(answer, label, color):
    assert render.badge(answer)[:2] == (label, color)


def test_every_refusal_rule_has_words():
    from agent.contract import REFUSAL_REASONS
    assert set(REFUSAL_REASONS) | {"R3"} == set(render.REFUSAL_WORDS)


@pytest.mark.parametrize(("value", "unit", "shown"), [
    (46254.0, "USD millions", "$46,254 million"), (22.1, "%", "22.1%"), (6.08, "USD per share", "$6.08 per share"),
    (-8516.0, "USD millions", "-$8,516 million"), (1.23, "ratio", "1.23"), (42, "USD", "$42"),
    (15000.0, "shares millions", "15,000 million shares"), (3, "count", "3"), (1.5, "USD billions", "$1.5 billion"),
    (0.00001, "ratio", "0.00001"), (0.1 + 0.2, None, "0.3"), (1234567.891, None, "1,234,567.891"),
    (2.0000000001, "USD millions", "$2 million"),
    (float("nan"), "%", "n/a"), (float("inf"), None, "n/a"), (None, "USD", "None"), ("12", None, "12"),
    (True, None, "True"), (100, "shares", "100 shares"),
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
    assert render.calculation_lines(answer) == [rf"**growth** of {render.escape('8,516, 14,276')}: "
                                                r"`(14276 - 8516) / 8516 * 100` = **67\.6\%**"]
    links = render.citation_links(answer)
    assert links[0] == f"[{render.escape('GS 10-K FY2023 · XBRL RevenuesNetOfInterestExpense')}]({SEC})"
    assert links[1] == render.escape("GS 10-Q Q2 FY2025 · Part I Item 2")       # not an SEC URL: shown, not linked


def test_a_backtick_cannot_end_the_expression_span():
    line = render.calculation_lines({"calculations": [
        {"operation": "x", "inputs": [], "result": 1, "expression": "a` [link](https://evil.example) `b"}]})[0]
    assert line.count("`") == 2 and "`a [link](https://evil.example) b`" in line


@pytest.mark.parametrize("url", [
    SEC + " extra", SEC + '"title"', SEC + ")", SEC + "(x)", SEC + "<x>", "https://www.sec.gov.evil.example/x",
    "http://www.sec.gov/x", "https://www.sec.gov/", "javascript:alert(1)", "",
])
def test_only_a_plain_sec_url_is_linked(url):
    assert not render.citation_links({"citations": [passage(url=url)]})[0].startswith("[")


def test_passages_of_one_section_are_numbered():
    links = render.citation_links({"citations": [passage(), passage("Item 8"), passage()]})
    assert links == [f"[{render.escape('MSFT 10-K FY2023 · Item 7, passage 1')}]({SEC})",
                     f"[{render.escape('MSFT 10-K FY2023 · Item 8')}]({SEC})",
                     f"[{render.escape('MSFT 10-K FY2023 · Item 7, passage 2')}]({SEC})"]


def test_the_card_parts_tolerate_odd_fields():
    odd = {"values": [{"value": "?"}, "junk"], "calculations": "junk", "citations": [{"ticker": "X"}, 3]}
    assert render.figure_rows(odd) == [{"figure": "", "company": "", "period": "", "value": "?"}]
    assert render.calculation_lines(odd) == [] and len(render.citation_links(odd)) == 1
    t = render.trace({"run": "junk", "history": None})
    assert t["tokens"] == 0 and t["tool_calls"] == [] and t["nodes"] == ""
    assert render.trace({"run": {"tool_calls": ["junk", {"tool": "t", "arguments": {"d": {1}}}]}}  # not JSON
                        )["tool_calls"] == [{"tool": "t", "arguments": '{"d": "{1}"}', "result": "ok"}]


def test_a_string_where_a_list_belongs_is_not_spelled_out():
    assert render.trace({"history": "abc", "run": {}})["history"] == []
    assert render.figure_rows({"values": "abc"}) == [] and render.as_list("abc") == []
    assert render.calculation_lines({"calculations": [{"operation": "x", "inputs": "12", "result": 1}]})[0].startswith(
        "**x** of : ")


def test_a_blank_line_cannot_end_a_code_span():
    line = render.calculation_lines({"calculations": [
        {"operation": "x", "inputs": [], "result": 1, "expression": "a\n\n# [click](https://evil.example) *x*"}]})[0]
    assert "\n" not in line and "`a # [click](https://evil.example) *x*`" in line
    assert render.code("a`b\n\n c") == "ab c"


def test_the_trace_table_puts_the_result_before_the_long_arguments():
    assert list(render.trace(BODY)["tool_calls"][0]) == ["tool", "result", "arguments"]


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
    assert hints["tenk_years"] == (2023, 2024, 2025)
    assert hints["tenq_quarters"] == ("Q1 FY2025", "Q2 FY2025", "Q3 FY2025")
    assert render.scope_hints() is hints                                          # read once


def test_the_disclaimer_counts_the_companies():
    assert "filings of 20 companies" in render.disclaimer(20) and "no investment advice" in render.disclaimer(20)


def test_example_questions_are_never_gold():
    gold = Path(__file__).resolve().parents[1] / "eval" / "gold_set.jsonl"
    questions = {json.loads(line)["question"] for line in gold.read_text(encoding="utf-8").splitlines() if line}
    assert not questions & set(render.EXAMPLE_QUESTIONS)


def test_the_trace_link_uses_the_sessions_hash(monkeypatch):
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

    def __init__(self, *replies, health=None, during_ask=None):
        self.replies = list(replies)
        self.calls: list[tuple[str, str | None]] = []
        self.health_calls = 0
        self._health = {"status": "ok", "model": "fake-model"} if health is None else health
        self.during_ask = during_ask

    def ask(self, question, session_id=None):
        self.calls.append((question, session_id))
        if self.during_ask:
            self.during_ask()
        return self.replies.pop(0)

    def health(self):
        self.health_calls += 1
        return self._health or None


def ok(b=None, sid="sid-1"):
    return api.Reply(ok=True, status=200, body=b or BODY, session_id=sid)


def app(fake):
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["client"] = fake
    return at.run()


def markdown(at) -> list[str]:
    return [m.value for m in at.markdown]


def assistants(at) -> list:
    return [c for c in at.chat_message if c.name == "assistant"]


def test_the_sidebar_has_the_disclaimer_the_status_and_the_scope():
    at = app(FakeApi())
    assert not at.exception
    assert at.sidebar.info[0].value == render.disclaimer(20) and "API ready" in at.sidebar.success[0].value
    labels = [b.label for b in at.sidebar.button]
    assert labels[0] == "New conversation" and labels[1:] == list(render.EXAMPLE_QUESTIONS)
    assert any("Apple Inc. (AAPL)" in c.value for c in at.sidebar.caption)


@pytest.mark.parametrize(("health", "kind", "says"), [
    ({}, "error", "API not reachable at http://fake-api"),
    ({"status": "starting", "warm_up": "pending"}, "warning", "API starting"),
    ({"status": "degraded", "warm_up": "APIConnectionError"}, "warning", "warm-up failed (APIConnectionError)"),
])
def test_the_api_status_is_shown(health, kind, says):
    at = app(FakeApi(health=health))
    assert says in getattr(at.sidebar, kind)[0].value


def test_a_ready_status_is_not_re_read_on_every_rerun():
    fake = FakeApi(ok(), ok())
    at = app(fake)
    at.chat_input[0].set_value("Q1?").run()
    at.chat_input[0].set_value("Q2?").run()
    assert fake.health_calls == 1                      # five runs within the TTL, one /health


def test_a_status_that_is_not_ready_is_re_read_after_a_short_wait():
    fake = FakeApi(health={"status": "starting", "warm_up": "pending"})
    at = app(fake)
    at.run()
    assert fake.health_calls == 1                      # not on every run: each read can wait up to 3 s
    at.session_state["health_at"] -= 6                 # past HEALTH_RETRY_S, well inside the ready TTL
    at.run()
    assert fake.health_calls == 2


def test_the_question_box_takes_what_the_api_accepts():
    from api.main import MAX_QUESTION_CHARS
    at = app(FakeApi())
    assert at.chat_input[0].max_chars == MAX_QUESTION_CHARS


def test_an_answer_is_a_card_with_badge_figures_sources_and_trace():
    fake = FakeApi(ok())
    at = app(fake)
    at.chat_input[0].set_value("What were Goldman Sachs' total net revenues in fiscal 2023?").run()
    md = markdown(at)
    assert not at.exception and fake.calls == [("What were Goldman Sachs' total net revenues in fiscal 2023?", None)]
    assert ":green-badge[:material/verified: Verified]" in md
    assert render.escape(BODY["answer"]["answer"]) in md
    assert any(SEC in m and m.startswith("**Sources**") for m in md)
    assert [e.label for e in at.expander] == ["Show trace"]
    assert [(m.label, m.value) for m in at.metric] == [("Verdict", "verified"), ("Tool calls", "1"),
                                                       ("Tokens", "2,685"), ("Seconds", "3.1")]
    assert len(at.dataframe) == 2                                    # the figures and the tool calls
    assert at.session_state["chat"].session_id == "sid-1" and at.session_state["chat"].pending is None
    assert not at.chat_input[0].disabled


def test_an_interaction_during_a_request_does_not_lose_the_answer():
    """Streamlit stops a run at its next command once the user interacts (a click while the spinner turns): the
    reply and the session id must already be stored by then, and the question must not be asked twice."""
    def click_meanwhile():
        get_script_run_ctx().script_requests.request_rerun(RerunData())

    fake = FakeApi(ok(sid="sid-7"), during_ask=click_meanwhile)
    at = app(fake)
    at.chat_input[0].set_value("Q1?").run()
    assert not at.exception and fake.calls == [("Q1?", None)]
    assert at.session_state["chat"].session_id == "sid-7" and at.session_state["chat"].pending is None
    assert len(assistants(at)) == 1 and ":green-badge[:material/verified: Verified]" in markdown(at)


def test_a_follow_up_sends_the_latest_session_id():
    fake = FakeApi(ok(sid="sid-1"), ok(body() | {"history": ["Q1?"]}, sid="sid-2"), ok(sid="sid-2"))
    at = app(fake)
    at.chat_input[0].set_value("Q1?").run()
    at.chat_input[0].set_value("And in fiscal 2024?").run()
    at.chat_input[0].set_value("And in fiscal 2025?").run()
    assert [sid for _, sid in fake.calls] == [None, "sid-1", "sid-2"]  # each response's id replaces the last
    assert len(assistants(at)) == 3


def test_an_error_is_shown_and_the_conversation_goes_on():
    fake = FakeApi(api.Reply(ok=False, status=503, error="FinSight is busy with other questions. Try again in 30 "
                                                          "seconds.", session_id="sid-5"), ok())
    at = app(fake)
    at.chat_input[0].set_value("Q1?").run()
    assert "Try again in 30 seconds" in at.error[0].value
    at.chat_input[0].set_value("Q2?").run()
    assert not at.exception and len(at.expander) == 1
    assert fake.calls[1] == ("Q2?", "sid-5")                         # an error's session id is still the latest


def test_a_malformed_reply_in_the_history_does_not_break_the_conversation():
    broken = api.Reply(ok=True, status=200, body={"answer": {"answer": "x", "values": [{"value": 1}]}, "run": {},
                                                   "cached": False}, session_id="sid-1")
    at = app(FakeApi(api.Reply(ok=True, status=200, body={}, session_id="sid-1"), broken, ok()))
    at.chat_input[0].set_value("Q1?").run()
    assert not at.exception and "could not be shown" in at.error[0].value
    at.chat_input[0].set_value("Q2?").run()
    at.chat_input[0].set_value("Q3?").run()
    assert not at.exception and len(assistants(at)) == 3
    assert ":green-badge[:material/verified: Verified]" in markdown(at)


@pytest.mark.parametrize(("answer", "badge"), [
    ({"refused": True, "verified": True, "refusal_rule": "R2", "values": [], "citations": []},
     ":blue-badge[:material/block: Refused: forecast (R2)]"),
    ({"verified": False, "notes": ["could not verify: invariant 4: WFC has no citation"]},
     ":orange-badge[:material/warning: Not verified]"),
])
def test_refusals_and_unverified_answers_say_so(answer, badge):
    at = app(FakeApi(ok(body(**answer))))
    at.chat_input[0].set_value("Q?").run()
    assert badge in markdown(at)
    if answer.get("notes"):
        assert any("could not verify" in c.value for c in at.caption)


def test_a_cached_answer_is_marked_and_not_linked_to_this_conversations_traces(monkeypatch):
    monkeypatch.setenv("FINSIGHT_TRACE_SESSIONS_URL", "https://lf.example/project/p/sessions")
    at = app(FakeApi(ok(body() | {"cached": True, "seconds": 0.01})))
    at.chat_input[0].set_value("Q?").run()
    assert ":gray-badge[:material/bolt: From cache]" in markdown(at)
    assert ("Seconds", "0.01") in [(m.label, m.value) for m in at.metric]
    assert not at.get("link_button") and any("reply cache" in c.value for c in at.caption)


def test_an_example_question_is_asked_and_new_conversation_clears():
    fake = FakeApi(ok(), ok())
    at = app(fake)
    at.sidebar.button[1].click().run()
    assert fake.calls == [(render.EXAMPLE_QUESTIONS[0], None)]
    at.sidebar.button[0].click().run()                               # New conversation
    assert at.session_state["chat"].messages == [] and at.session_state["chat"].session_id is None
    assert not [c for c in at.chat_message]


def test_the_inputs_are_disabled_while_a_question_is_asked():
    import streamlit as st
    fake = FakeApi(ok(), during_ask=st.stop)           # the run ends mid-request: what the user sees while waiting
    at = app(fake)
    at.chat_input[0].set_value("Q?").run()
    assert fake.calls == [("Q?", None)] and at.session_state["chat"].pending == "Q?"
    assert at.chat_input[0].disabled and all(b.disabled for b in at.sidebar.button[1:])     # the examples
    assert [c.name for c in at.chat_message] == ["user", "assistant"]   # the question, and the spinner's message
    assert not at.sidebar.button[0].disabled                           # New conversation is always the way out
    at.sidebar.button[0].click().run()
    assert at.session_state["chat"] == api.Conversation() and not at.chat_input[0].disabled
    assert fake.calls == [("Q?", None)]                                # clearing does not ask again


def test_an_exception_while_asking_is_a_message_and_the_chat_goes_on():
    def broken():
        raise RuntimeError("bug")

    fake = FakeApi(ok(), ok(), during_ask=broken)
    at = app(fake)
    at.chat_input[0].set_value("Q1?").run()
    assert not at.exception and "Something went wrong" in at.error[0].value
    assert at.session_state["chat"].pending is None and not at.chat_input[0].disabled
    assert fake.calls == [("Q1?", None)]
    fake.during_ask = None
    at.chat_input[0].set_value("Q2?").run()
    assert not at.exception and len(assistants(at)) == 2


def test_a_malformed_session_id_never_breaks_the_follow_up():
    """End to end with the real client: an id httpx cannot send is ignored, so the follow-up is still asked."""
    def handler(request):
        return httpx.Response(200, json=BODY | {"session_id": "séssion"})

    at = app(client(handler))
    at.chat_input[0].set_value("Q1?").run()
    at.chat_input[0].set_value("Q2?").run()
    assert not at.exception and len(assistants(at)) == 2 and not at.error


def test_notes_that_are_not_a_list_are_not_spelled_out():
    at = app(FakeApi(ok(body(notes="abc"))))
    at.chat_input[0].set_value("Q?").run()
    assert not at.exception and not [c for c in at.caption if c.value in ("a", "b", "c")]


def test_the_trace_link_appears_when_configured(monkeypatch):
    monkeypatch.setenv("FINSIGHT_TRACE_SESSIONS_URL", "https://lf.example/project/p/sessions")
    at = app(FakeApi(ok()))
    at.chat_input[0].set_value("Q?").run()
    buttons = at.get("link_button")
    assert len(buttons) == 1 and "lf.example/project/p/sessions/" in buttons[0].proto.url


def test_a_question_with_markdown_is_shown_as_typed():
    at = app(FakeApi(ok()))
    at.chat_input[0].set_value("What was *Apple*'s $ revenue?").run()
    assert render.escape("What was *Apple*'s $ revenue?") in markdown(at)
