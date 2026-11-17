"""scripts/probe_tool_calls.py offline: a fake OpenAI client with real response types and a fake tool, so the
measurement, its scoring, the re-scoring and the report run without the endpoint or the developer's data/."""
import asyncio
import json

import pytest
from openai.types.chat import ChatCompletion

import scripts.probe_endpoint as pe
import scripts.probe_tool_calls as ptc
from core.settings import Settings
from tools import registry

SECRET_HOST, SECRET_KEY = "secret-group-host.example", "sk-team-secret-123"


def settings():
    return Settings(_env_file=None, LLM_BASE_URL=f"https://{SECRET_HOST}/v1", LLM_API_KEY=SECRET_KEY,
                    LLM_MODEL="Qwen/Qwen3-8B", SEC_USER_AGENT="a b@c.d", LLM_ENABLE_THINKING=False)


def completion(tool_calls=None, content=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [{"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": a}}
                                 for i, (n, a) in enumerate(tool_calls)]
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop", "message": message}],
        "usage": {"prompt_tokens": 1800, "completion_tokens": 30, "total_tokens": 1830}})


class FakeClient:
    """chat.completions.create answers with reply(request); every request is kept."""

    def __init__(self, reply):
        self.reply, self.requests = reply, []
        self.chat = self
        self.completions = self

    async def create(self, **request):
        self.requests.append(request)
        return self.reply(request)


def fake_tool(name, arguments):
    """The registry's answers without data/: a metric outside the 13 or a unit on growth is invalid_input."""
    args = arguments if isinstance(arguments, dict) else json.loads(arguments)
    if name == "get_financial_fact":
        if args.get("metric") not in registry.TOOLS[name][1].model_fields["metric"].annotation.__args__:
            return {"error": {"type": "invalid_input", "message": f"unknown metric {args.get('metric')!r}"}}
        return {"value": {2023: 100.0, 2024: 121.0}.get(args.get("fiscal_year"), 50.0), "unit": "USD millions",
                "ticker": args["ticker"], "metric": args["metric"], "fiscal_year": args["fiscal_year"],
                "fiscal_period": "FY"}
    if name == "calculate" and args.get("unit") and args["operation"] not in ("difference", "sum"):
        return {"error": {"type": "invalid_input", "message": "unit is only for difference and sum"}}
    return {"result": 1.0}


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    monkeypatch.setattr(pe, "MIN_GAP_S", 0.0)
    monkeypatch.setattr(registry, "call_tool", fake_tool)      # accepts() imports it from the registry


QUESTIONS = [
    {"id": "pdev-x1", "kind": "control", "question": "What was Apple's revenue in fiscal 2024?", "history": []},
    {"id": "pdev-x2", "kind": "control", "question": "What stock-based compensation did Meta report?", "history": []},
    {"id": "pdev-x3", "kind": "control", "question": "And in fiscal 2023?",
     "history": ["What was Apple's revenue in fiscal 2024?"]},
]
FACT_2024 = {"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024, "fiscal_period": "FY", "duration": None}
REFERENCES = {"pdev-x1": ("get_financial_fact", FACT_2024),
              "pdev-x2": ("search_filings", {"query": "stock-based compensation", "ticker": "META", "fiscal_year": None}),
              "pdev-x3": ("get_financial_fact", {**FACT_2024, "fiscal_year": 2023})}
CASE = {"id": "pdev-x4", "question": "By what percentage did Apple's revenue change from fiscal 2023 to 2024?",
        "history": [], "expected": ["cagr", "growth"],
        "facts": [{"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2023},
                  {"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024}]}


def scripted(request):
    """The fake model: a wrong metric name for the first question, a substitute metric for the line item, a text
    answer for the follow-up, and growth with a unit once the figures are in."""
    if request["messages"][-1]["role"] == "tool":
        return completion([("calculate", json.dumps({"operation": "growth", "inputs": [100.0, 121.0],
                                                     "unit": "USD millions"}))])
    question = request["messages"][-1]["content"]
    if "stock-based" in question:
        return completion([("get_financial_fact", json.dumps({"ticker": "META", "metric": "rd_expense",
                                                              "fiscal_year": 2024}))])
    if "Earlier question" in question:
        return completion(content="I need more information.")
    return completion([("get_financial_fact", json.dumps({"ticker": "AAPL", "metric": "net_sales",
                                                          "fiscal_year": 2024}))])


def measured(reply=scripted):
    client = FakeClient(reply)
    run = asyncio.run(ptc.measure(pe.Session(client, settings()), QUESTIONS, [CASE], fake_tool))
    run.update(label="t", model="m", date="d", schemas_sha256="0" * 64, schema_tokens=1)
    run["summary"] = ptc.summarize(run, REFERENCES)
    return run, client


def test_each_question_is_asked_once_with_the_tools_and_tool_choice_auto():
    _, client = measured()
    first = client.requests[0]
    assert first["tool_choice"] == "auto" and first["temperature"] == 0.0
    assert [t["function"]["name"] for t in first["tools"]] == [s["name"] for s in registry.SCHEMAS]
    assert first["messages"][0] == {"role": "system", "content": ptc.EXECUTOR_PROMPT}
    assert client.requests[2]["messages"][1]["content"] == (
        "Earlier question in this conversation: What was Apple's revenue in fiscal 2024?\nQuestion: And in fiscal 2023?")
    assert len(client.requests) == len(QUESTIONS) + 1


def test_first_calls_are_scored_for_acceptance_and_for_the_right_thing():
    run, _ = measured()
    s = run["summary"]
    assert (s["questions"], s["tool_calls"], s["accepted"]) == (3, 2, 1)       # net_sales rejected, text not a call
    assert (s["with_reference"], s["matches_plan"]) == (3, 0)                   # rd_expense is accepted but wrong
    rejected = run["first_call"][0]
    assert not rejected["accepted"] and "unknown metric 'net_sales'" in rejected["why"]


def test_the_calculator_case_sees_the_fetched_figures_and_is_scored():
    run, client = measured()
    calc_request = client.requests[-1]["messages"]
    assert calc_request[2]["tool_calls"][0]["function"]["name"] == "get_financial_fact"
    assert [json.loads(m["content"])["value"] for m in calc_request[3:]] == [100.0, 121.0]
    calc = run["calculator"][0]
    assert calc["operation"] == "growth" and calc["right_operation"] and not calc["accepted"]
    s = run["summary"]
    assert (s["calc_cases"], s["calc_called"], s["calc_right_operation"], s["calc_accepted"]) == (1, 1, 1, 0)


def test_a_case_whose_figure_cannot_be_fetched_is_skipped():
    def no_data(name, arguments):
        return {"error": {"type": "not_reported", "message": "gap"}}
    assert ptc.calc_messages(CASE, no_data) is None
    run = asyncio.run(ptc.measure(pe.Session(FakeClient(scripted), settings()), [], [CASE], no_data))
    assert run["calculator"] == [{"id": "pdev-x4", "skipped": "a planned figure could not be fetched"}]


def test_rescoring_rechecks_the_recorded_calls_without_a_request(monkeypatch):
    run, _ = measured()
    monkeypatch.setattr(registry, "call_tool", lambda name, arguments: {"result": 1.0})   # a more lenient tool
    monkeypatch.setattr(ptc, "reference_steps", lambda: REFERENCES)
    again = ptc.rescore(run, "t2")
    assert again["rescored_from"] == "t" and again["first_call"][2] == run["first_call"][2]      # text stays text
    assert again["summary"]["accepted"] == 2 and again["summary"]["calc_accepted"] == 1


def test_the_report_compares_labels_and_never_shows_the_key_or_host(tmp_path, monkeypatch):
    monkeypatch.setattr(ptc, "reference_steps", lambda: REFERENCES)
    before, _ = measured()
    after, _ = measured(lambda r: completion([("get_financial_fact", json.dumps(
        {"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024}))]))
    ptc.write("before", {**before, "label": "before"}, tmp_path, tmp_path / "s.md")
    ptc.write("after", {**after, "label": "after"}, tmp_path, tmp_path / "s.md")
    text = (tmp_path / "s.md").read_text(encoding="utf-8")
    assert "| Measure | before | after |" in text and "| first call accepted by the tool | 1/3 | 3/3 |" in text
    assert "accepted, but not what the reviewed plan fetches: search_filings" in text
    for path in tmp_path.iterdir():
        content = path.read_text(encoding="utf-8")
        assert SECRET_HOST not in content and SECRET_KEY not in content


def test_the_questions_are_answerable_development_questions_only():
    qs = ptc.questions()
    assert len(qs) == 31 and all(q["id"].startswith(("dev-", "pdev-")) and q["kind"] != "refusal" for q in qs)


def test_the_calculator_cases_and_references_come_from_the_committed_planner_run():
    cases = ptc.calc_cases(ptc.questions())
    assert {c["id"]: c["expected"] for c in cases} == {
        "dev-B01": ["percent"], "dev-B02": ["percent"], "dev-C01": ["cagr", "growth"], "dev-C02": ["cagr", "growth"],
        "pdev-25": ["percent"]}
    references = ptc.reference_steps()
    assert references["pdev-32"][0] == "search_filings"
    assert references["dev-G02"][0] == "get_financial_fact" and references["dev-G02"][1]["metric"] == "capex"


@pytest.mark.parametrize(("arguments", "matches"), [
    ({"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024}, True),                  # FY when left out
    ({"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024, "fiscal_period": "FY"}, True),
    ({"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2023}, False),                 # the wrong year
    ({"ticker": "AAPL", "metric": "net_income", "fiscal_year": 2024}, False),
    ({"ticker": "MSFT", "metric": "revenue", "fiscal_year": 2024}, False),
    ({"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024, "fiscal_period": "Q2"}, False),
])
def test_the_right_call_checks_company_metric_year_and_period(arguments, matches):
    call = {"tool_call": True, "name": "get_financial_fact", "arguments": json.dumps(arguments)}
    assert ptc.matches_plan(call, ("get_financial_fact", FACT_2024)) is matches


def test_a_search_matches_the_plans_company_and_year_where_set():
    plan = ("search_filings", {"query": "risks", "ticker": "GS", "fiscal_year": 2023, "fiscal_period": "FY"})
    def call(**a):
        return {"tool_call": True, "name": "search_filings", "arguments": json.dumps({"query": "x", **a})}
    assert ptc.matches_plan(call(ticker="GS", fiscal_year=2023), plan) is True
    assert ptc.matches_plan(call(ticker="GS", fiscal_year=2024), plan) is False
    assert ptc.matches_plan(call(ticker="GS", fiscal_year=2023), ("get_financial_fact", FACT_2024)) is False
