"""W2-11: the agent's gold run (eval/run_agent.py) and how the scorer and the grading tools read it. Offline: stub
or recording nodes, no endpoint, no data/, no .env; the real gold set, refusal tests and context packs."""
from __future__ import annotations

import asyncio
import csv
import functools
import io
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import eval.score_answers as sa
from agent import graph as g
from agent.contract import AnswerContract
from agent.state import Plan, ToolCall, ToolResult
from agent.stubs import stub_nodes
from eval import grading
from eval import run_agent as ra
from eval.runs_schema import AgentInfo, B0Answer, Evidence, RunMeta, RunRecord, ToolCallRecord

LIMITS = g.Limits(max_steps=10, max_tokens=12_000, timeout_s=10)
UNCOMMITTED = ra.uncommitted                     # the real one, for its own test (the fixture below replaces it)


@pytest.fixture(autouse=True)
def _a_clean_checkout(monkeypatch):
    """A gold run checks that the code is committed (W3-08); the tests must not depend on the checkout's git state."""
    monkeypatch.setattr(ra, "uncommitted", lambda paths=ra.CODE_PATHS: [])
PACKS = [ra.RESULTS_DIR / ra.GOLD_PACK, ra.RESULTS_DIR / ra.REFUSAL_PACK]
B0 = sa.B0_REFERENCE
CHUNK = "AAPL_10-K_2024_Item7_C3_0012"
FACT = {"value": 391035.0, "unit": "USD millions", "ticker": "AAPL", "fiscal_year": 2024, "fiscal_period": "FY",
        "duration": "12M", "concept": "us-gaap:Revenues", "accession_no": "0000320193-24-000123", "filed": "2024-11-01",
        "url": "https://www.sec.gov/x", "metric": "revenue", "label": "total net sales", "form": "10-K",
        "restated": False, "original_value": 391035.0}          # tools/financial_facts.py's shape (USD millions)


def settings(**kw):
    return SimpleNamespace(**({"LLM_MODEL": "Qwen/Qwen3-8B", "LLM_ENABLE_THINKING": False,
                               "LLM_MAX_COMPLETION_TOKENS": 1024, "EMBED_MODEL": "bge-small-en-v1.5",
                               "RETRIEVAL_RERANK": False, "LLM_CACHE_DIR": "cache"} | kw))


# ----------------------------------------------------------------------------------------------- the questions
def test_the_items_are_the_packs_102_questions_with_their_earlier_questions_only():
    items = ra.load_items(PACKS)
    assert len(items) == 102 and len({i.id for i in items}) == 102
    f01 = next(i for i in items if i.id == "F01")
    assert f01.history == ("What were Apple's total net sales in fiscal 2024?",)        # the question, no answer
    assert all(isinstance(h, str) for i in items for h in i.history)


def test_the_agent_sees_only_questions_never_a_gold_answer_or_a_passage():
    """The planner gets the question and its earlier questions; the packs' passages and every earlier answer
    (the gold answer of a follow-up's first turn) stay out of the agent's state."""
    seen = []

    async def planner(state):
        seen.append((state["question"], list(state.get("history") or [])))
        return {"plan": Plan(reasoning="r", refuse=False, refusal_rule=None, steps=[])}

    nodes = g.Nodes(planner=planner, executor=stub_nodes().executor, composer=stub_nodes().composer,
                    critic=stub_nodes().critic)
    items = ra.load_items(PACKS)
    records, stopped = asyncio.run(ra.run_items(items, nodes, LIMITS, ra.RUN_ID))
    assert stopped is None and [r.id for r in records] == [i.id for i in items]
    assert seen == [(i.question, list(i.history)) for i in items]
    gold_answers = {json.loads(x)["gold_answer"] for x in sa.GOLD_FILE.read_text(encoding="utf-8").splitlines() if x}
    passages = {p["text"] for path in PACKS for r in sa.load_pack(path).values() for p in r.get("passages", [])}
    shown = {text for q, h in seen for text in [q, *h]}
    assert not shown & gold_answers and not shown & passages


# ----------------------------------------------------------------------------------------------- evidence
def test_a_search_returns_its_passages_by_chunk_id():
    out = [{"chunk_id": CHUNK, "text": "Net sales rose 2% to $391.0 billion.", "ticker": "AAPL", "form": "10-K",
            "fiscal_year": 2024, "fiscal_period": "FY", "item": "Item 7", "score": 0.03}]
    assert ra.evidence("search_filings", out) == [Evidence(source=CHUNK, text="Net sales rose 2% to $391.0 billion.",
                                                           label="AAPL 10-K 2024 FY Item 7")]


def test_a_fact_is_named_by_concept_and_accession_and_keeps_only_its_value():
    [e] = ra.evidence("get_financial_fact", FACT)
    assert e.source == "us-gaap:Revenues 0000320193-24-000123" and e.text == "391035"
    assert e.label == "AAPL 10-K FY2024 FY · total net sales (USD millions)"           # for a grader


def test_a_restated_fact_keeps_its_original_value_too():
    [e] = ra.evidence("get_financial_fact", FACT | {"restated": True, "original_value": 390000.0})
    assert e.text == "391035 390000"


def test_a_period_comparison_returns_each_year_with_its_growth_as_a_percentage():
    rows = [FACT | {"value": 383285.0, "original_value": 383285.0, "fiscal_year": 2023, "yoy_growth_pct": None,
                    "yoy_calculation": None},
            FACT | {"yoy_growth_pct": 2.0, "yoy_calculation": {"operation": "growth", "inputs": [383285.0, 391035.0],
                                                               "result": 2.0, "unit": "%"}}]
    assert [e.text for e in ra.evidence("compare_periods", rows)] == ["383285", "391035 2%"]


def test_a_calculation_gives_its_result_never_its_inputs():
    out = {"operation": "growth", "inputs": [383285.0, 391035.0], "years": None, "result": 2.02, "unit": "%"}
    assert ra.evidence("calculate", out) == [Evidence(source="calculate", text="2.02%",
                                                      label="growth of 383285, 391035")]
    assert ra.evidence("calculate", out | {"unit": None, "result": 1.69})[0].text == "1.69"


def test_an_error_returned_nothing():
    assert ra.evidence("get_financial_fact", {"error": {"type": "data_missing", "message": "x"}}) == []


@pytest.mark.parametrize(("value", "text"), [(60458000000.0, "60458000000"), (0.00001, "0.00001"), (22.1, "22.1"),
                                             (7, "7"), (0.0, "0"), (1e-7, "0.0000001"),
                                             (1234.56789012, "1234.56789012"), (-42012.0, "-42012")])
def test_figure_text_has_no_exponent_or_trailing_zeros(value, text):
    assert ra.figure_text(value) == text


def test_citations_are_named_as_evidence_names_its_sources():
    assert ra.citation_id({"source_type": "filing", "chunk_id": CHUNK, "item": "Item 7"}) == CHUNK
    assert ra.citation_id({"source_type": "xbrl", "concept": FACT["concept"], "accession_no": FACT["accession_no"]}) \
        == "us-gaap:Revenues 0000320193-24-000123"


# ----------------------------------------------------------------------------------------------- the B0 fields
def value(v, unit="USD millions", ticker="AAPL", year=2024):
    return {"name": "revenue", "value": v, "unit": unit, "ticker": ticker, "fiscal_year": year, "fiscal_period": "FY"}


def record(**kw):
    return {"answer": "x", "refused": False, "refusal_rule": None, "values": [], "calculations": [],
            "citations": []} | kw


def test_one_value_is_the_answers_value():
    a = ra.b0_answer(record(answer="Apple's total net sales were $391,035 million.", values=[value(391035.0)]))
    assert (a.value, a.unit, a.company, a.fiscal_year) == (391035.0, "USD millions", "AAPL", 2024)


def test_a_calculation_wins_over_its_inputs():
    a = ra.b0_answer(record(answer="Net sales grew 2.02%.", values=[value(383285.0, year=2023), value(391035.0)],
                            calculations=[{"operation": "growth", "result": 2.02, "unit": "%"}]))
    assert (a.value, a.unit, a.company, a.fiscal_year) == (2.02, "%", "AAPL", None)       # two years: no one year


def test_a_value_the_answer_does_not_state_is_not_its_value():
    """B08: the answer says the ratio is not reported; the one fact it fetched (revenue) is no answer to it."""
    a = ra.b0_answer(record(answer="The ratio is not reported separately.", values=[value(400278.0)]))
    assert (a.value, a.unit) == (None, None)
    a = ra.b0_answer(record(answer="It grew.", calculations=[{"operation": "growth", "result": 2.02, "unit": "%"}]))
    assert a.value is None


def test_a_not_reported_answer_has_no_value_even_when_it_offers_a_related_figure():
    a = ra.b0_answer(record(answer="Amazon does not report total liabilities; it reports total current liabilities "
                                   "of $179,431 million.", values=[value(179431.0, ticker="AMZN")]))
    assert a.value is None and a.company == "AMZN"
    s = sa.check(QUESTIONS["G08"], agent_record("G08", a, []), {}, False, MATCHERS)
    assert s["correct"] is True                                       # the trap is answered, not given a value


def test_an_unstated_calculation_falls_back_to_the_one_stated_value():
    a = ra.b0_answer(record(answer="Costco's net sales were $275,235 million in fiscal 2025.",
                            values=[value(275235.0, ticker="COST", year=2025)],
                            calculations=[{"operation": "growth", "result": 8.2, "unit": "%"}]))
    assert (a.value, a.unit) == (275235.0, "USD millions")


def test_a_not_reported_aside_after_the_answer_keeps_its_value():
    a = ra.b0_answer(record(answer="Apple's total net sales were $391,035 million in fiscal 2024; the Services split "
                                   "is not reported separately.", values=[value(391035.0)]))
    assert a.value == 391035.0


def test_a_rounded_statement_of_the_value_counts_as_stating_it():
    a = ra.b0_answer(record(answer="Apple's total net sales were $391.0 billion.", values=[value(391035.0)]))
    assert a.value == 391035.0


def test_several_values_without_a_calculation_give_no_value_and_two_companies_no_company():
    """Even when the answer states both figures: which one is "the" value is not the mapping's to decide."""
    a = ra.b0_answer(record(answer="Pfizer's net income was $2,119 million and Johnson & Johnson's $35,153 million.",
                            values=[value(2119.0, ticker="PFE"), value(35153.0, ticker="JNJ")]))
    assert (a.value, a.unit, a.company, a.fiscal_year) == (None, None, None, 2024)


def test_without_values_the_citations_name_the_company_and_year():
    cite = {"source_type": "filing", "ticker": "MSFT", "form": "10-K", "fiscal_year": 2023, "fiscal_period": "FY",
            "item": "Item 7", "chunk_id": "MSFT_10-K_2023_Item7_C3_0001"}
    a = ra.b0_answer(record(citations=[cite, cite | {"chunk_id": "MSFT_10-K_2023_Item7_C3_0002"}]))
    assert (a.company, a.fiscal_year, a.citations) == ("MSFT", 2023, ["MSFT_10-K_2023_Item7_C3_0001",
                                                                       "MSFT_10-K_2023_Item7_C3_0002"])


def test_a_refusal_keeps_its_decision_and_rule():
    a = ra.b0_answer(record(answer="I can't help with that.", refused=True, refusal_rule="R2"))
    assert (a.refused, a.refusal_rule, a.value) == (True, "R2", None)


# ----------------------------------------------------------------------------------------------- one record
def agent_run(answer, results=(), **state):
    return g.AgentRun(answer=answer, state={"history": [], "tool_results": list(results), "tokens": 2700,
                                            "visits": {"planner": 1, "executor": 1, "composer": 1, "critic": 1},
                                            **state}, seconds=3.25)


def test_a_record_keeps_the_tool_calls_their_evidence_and_the_cost():
    results = [ToolResult(call=ToolCall(name="get_financial_fact", arguments={"ticker": "AAPL"}), output=FACT),
               ToolResult(call=ToolCall(name="search_filings", arguments={"query": "q"}),
                          output={"error": {"type": "no_results", "message": "none"}})]
    answer = AnswerContract(answer="Apple's revenue was $391,035 million in fiscal 2024.", verified=False,
                            refused=False, notes=["could not verify: y"])
    spent = ra.Counts(calls=3, http_retries=1, cache_hits=1, prompt_tokens=2500, completion_tokens=200,
                      json_retries=1)
    rec = ra.run_record(ra.Item("A01", "Q?", ()), ra.RUN_ID, agent_run(answer, results), spent)
    assert rec.parsed.answer == answer.answer and rec.json_valid and rec.retries == 1 and not rec.cached
    assert (rec.prompt_tokens, rec.completion_tokens, rec.latency_s) == (2500, 200, 3.25)
    assert [(c.tool, c.error, len(c.evidence)) for c in rec.tool_calls] == [("get_financial_fact", None, 1),
                                                                            ("search_filings", "no_results", 0)]
    assert (rec.agent.verdict, rec.agent.tool_steps, rec.agent.llm_calls, rec.agent.live_requests) == \
        ("not verified", 2, 3, 3)                                     # 3 sent - 1 HTTP retry + 1 cache hit
    assert json.loads(rec.raw_reply)["notes"] == ["could not verify: y"]
    assert RunRecord.model_validate_json(rec.to_line()) == rec


def test_a_question_answered_from_the_cache_is_marked_cached():
    rec = ra.run_record(ra.Item("A01", "Q?", ()), ra.RUN_ID,
                        agent_run(AnswerContract(answer="x", verified=True, refused=False)), ra.Counts(cache_hits=2))
    assert rec.cached and rec.agent.live_requests == 0 and rec.agent.llm_calls == 2 and rec.agent.cache_hits == 2


def test_counts_charge_each_question_its_own_share():
    usage = SimpleNamespace(calls=5, http_retries=1, cache_hits=2, prompt_tokens=100, completion_tokens=10,
                            json_retries=0)
    before = ra.Counts.of(usage)
    usage.calls, usage.cache_hits, usage.prompt_tokens = 8, 3, 160
    assert ra.Counts.of(usage) - before == ra.Counts(3, 0, 1, 60, 0, 0) and ra.Counts.of(None) == ra.Counts()


# ----------------------------------------------------------------------------------------------- the CLI
def run_cli(tmp_path, *extra, nodes=None):
    return ra.main(["--out-dir", str(tmp_path), *extra], nodes=nodes or stub_nodes(), settings=settings(),
                   limits=LIMITS)


def test_the_cli_writes_a_runs_file_the_scorer_reads(tmp_path):
    assert run_cli(tmp_path) == 0
    lines = (tmp_path / f"{ra.RUN_ID}.jsonl").read_text(encoding="utf-8").splitlines()
    meta = RunMeta.model_validate_json((tmp_path / f"{ra.RUN_ID}.meta.json").read_text(encoding="utf-8"))
    assert len(lines) == 102 and meta.context == "agent" and meta.questions == "gold"
    assert meta.context_pack_sha256 == {p.name: sa.sha256(p) for p in PACKS}
    from agent import composer, planner
    assert (meta.planner_prompt_sha256, meta.composer_prompt_sha256) == (planner.PROMPT_SHA256, composer.PROMPT_SHA256)
    assert meta.prompt_version.startswith(planner.PROMPT_VERSION)
    assert meta.limits == {"max_steps": 10, "max_tokens": 12_000, "timeout_s": 10}
    assert meta.retrieval == {"embed_model": "bge-small-en-v1.5", "rerank": False, "chunks": "C3",
                              "fusion": "hybrid RRF", "statements_first": True}        # what the A4 ablation switches
    rec = RunRecord.model_validate_json(lines[0])
    assert rec.tool_calls == [] and rec.agent is not None
    assert not {"steps", "agreement"} & set(json.loads(lines[0]))                           # no S-only fields


def test_the_dev_questions_go_to_their_own_directory(tmp_path):
    assert run_cli(tmp_path, "--questions", "dev") == 0
    meta = RunMeta.model_validate_json((tmp_path / f"{ra.RUN_ID}.meta.json").read_text(encoding="utf-8"))
    assert list(meta.context_pack_sha256) == [ra.DEV_PACK] and meta.questions == "dev"
    assert ra.DEV_RUNS_DIR != ra.RUNS_DIR and ra.DEV_RUNS_DIR.parent.parent == ra.RUNS_DIR.parent


@pytest.mark.parametrize("run_id", ["M9_qwen3-8b_rag_think-off_b0-v1", "A1_qwen3-8b_rag_think-off_a-v1", "agent"])
def test_a_run_id_that_is_not_an_agent_run_is_refused_before_any_request(tmp_path, run_id):
    assert run_cli(tmp_path, "--run-id", run_id) == 2 and not list(tmp_path.iterdir())


def test_a_question_that_raises_is_recorded_and_the_run_goes_on(tmp_path, capsys):
    calls = []

    async def planner(state):
        calls.append(state["question"])
        if len(calls) == 2:
            raise RuntimeError("a node bug")
        return await stub_nodes().planner(state)

    nodes = g.Nodes(planner=planner, executor=stub_nodes().executor, composer=stub_nodes().composer,
                    critic=stub_nodes().critic)
    assert run_cli(tmp_path, nodes=nodes) == 1
    recs = [RunRecord.model_validate_json(x) for x in
            (tmp_path / f"{ra.RUN_ID}.jsonl").read_text(encoding="utf-8").splitlines()]
    bad = [r for r in recs if r.error]
    assert len(recs) == 102 and len(bad) == 1 and bad[0].error == "RuntimeError: a node bug" and bad[0].parsed is None
    assert "errors: " + bad[0].id in capsys.readouterr().out


@functools.cache
def _first_run_files() -> tuple[bytes, bytes]:
    """One stub run's runs file and meta, made once for the tests that replay it (W3-07: each made its own, the same
    102 questions through the graph)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        assert run_cli(out) == 0
        return (out / f"{ra.RUN_ID}.jsonl").read_bytes(), (out / f"{ra.RUN_ID}.meta.json").read_bytes()


def earlier_run(tmp_path, change=None):
    """A first run's files, with its own timings and cost (as a live run records them)."""
    first = tmp_path / "first"
    first.mkdir()
    runs, meta = _first_run_files()
    (first / f"{ra.RUN_ID}.jsonl").write_bytes(runs)
    (first / f"{ra.RUN_ID}.meta.json").write_bytes(meta)
    path = first / f"{ra.RUN_ID}.jsonl"
    lines = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        r = json.loads(line)
        r["latency_s"], r["cached"] = 4.5 + n, False
        r["agent"].update(llm_calls=2, live_requests=2, tokens=900 + n)
        r["agent"].pop("cache_hits")                                   # an older runs file has no such field
        if change and n == 0:
            change(r)
        lines.append(json.dumps(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = json.loads((first / f"{ra.RUN_ID}.meta.json").read_text(encoding="utf-8"))
    meta.update(started="2026-11-20T20:51:02+00:00", finished="2026-11-20T21:00:21+00:00", git_commit="abc",
                requests={"live": 204, "cache_hits": 0, "http_retries": 0, "embed_calls": 0})
    (first / f"{ra.RUN_ID}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_a_replay_keeps_the_earlier_runs_timings_cost_and_dates(tmp_path):
    previous = earlier_run(tmp_path)
    out = tmp_path / "replay"
    assert run_cli(out, "--replay-of", str(previous)) == 0
    recs = [RunRecord.model_validate_json(x) for x in (out / f"{ra.RUN_ID}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r.latency_s for r in recs[:2]] == [4.5, 5.5] and not any(r.cached for r in recs)
    assert (recs[0].agent.llm_calls, recs[0].agent.live_requests, recs[0].agent.cache_hits) == (2, 2, 0)
    assert [r.agent.tokens for r in recs[:2]] == [900, 901]                  # the whole cost is the earlier run's
    meta = json.loads((out / f"{ra.RUN_ID}.meta.json").read_text(encoding="utf-8"))
    assert (meta["started"], meta["git_commit"]) == ("2026-11-20T20:51:02+00:00", "abc")
    assert meta["requests"] == {"live": 204, "cache_hits": 0, "http_retries": 0}       # embed_calls dropped
    assert meta["replay"]["requests"] == {"live": 0, "cache_hits": 0, "http_retries": 0}


@pytest.mark.parametrize(("meta_change", "argv"), [
    ({}, ["--ablation", "no-critic"]),                                 # an ablation must not borrow A1's timings
    ({"planner_prompt_sha256": "other"}, []),
    ({"limits": {"max_steps": 5, "max_tokens": 12_000, "timeout_s": 10}}, []),
    ({"context_pack_sha256": {"context_pack_gold.jsonl": "other"}}, []),
])
def test_a_replay_of_another_run_is_refused(tmp_path, capsys, meta_change, argv):
    previous = earlier_run(tmp_path)
    meta_path = previous.with_name(previous.stem + ".meta.json")
    meta_path.write_text(json.dumps(json.loads(meta_path.read_text(encoding="utf-8")) | meta_change), encoding="utf-8")
    out = tmp_path / "replay"
    assert run_cli(out, "--replay-of", str(previous), *argv) == 2 and not out.exists()
    assert "another run" in capsys.readouterr().err


def test_a_replay_goes_live_at_most_once_and_then_writes_nothing(tmp_path, monkeypatch, capsys):
    """A cache miss (a prompt changed) must not spend the shared key's budget: the budget is 1, and a replay that
    sent anything is refused."""
    previous = earlier_run(tmp_path)
    budgets = []

    async def fake_run_real(items, settings_, max_requests, limits, run_id, ablation=None):
        budgets.append(max_requests)
        records, _ = await ra.run_items(items, stub_nodes(), limits, run_id)
        return records, None, SimpleNamespace(calls=1, cache_hits=186, http_retries=0)

    monkeypatch.setattr(ra, "run_real", fake_run_real)
    out = tmp_path / "replay"
    code = ra.main(["--out-dir", str(out), "--replay-of", str(previous), "--max-requests", "400"],
                   settings=settings(), limits=LIMITS)
    assert code == 2 and budgets == [1] and not out.exists()
    assert "not in the cache (1 live request(s))" in capsys.readouterr().err


@pytest.mark.parametrize("change", [
    lambda r: r.update(raw_reply=json.dumps(json.loads(r["raw_reply"]) | {"answer": "another answer"})),
    lambda r: r.update(tool_calls=[{"tool": "calculate", "arguments": {}, "evidence": []}]),
])
def test_a_replay_whose_answers_or_tool_calls_differ_is_refused(tmp_path, capsys, change):
    previous = earlier_run(tmp_path, change)
    out = tmp_path / "replay"
    assert run_cli(out, "--replay-of", str(previous)) == 2 and not out.exists()
    assert "not a replay" in capsys.readouterr().err


def test_the_budget_stops_the_run_and_writes_nothing(tmp_path, monkeypatch, capsys):
    from eval.run_demo import BudgetExceeded

    async def planner(state):
        raise BudgetExceeded("the budget of 0 live requests is used up")

    nodes = g.Nodes(planner=planner, executor=stub_nodes().executor, composer=stub_nodes().composer,
                    critic=stub_nodes().critic)
    assert run_cli(tmp_path, nodes=nodes) == 2 and not list(tmp_path.iterdir())
    assert "nothing written" in capsys.readouterr().err


# ----------------------------------------------------------------------------------------------- scoring an agent run
def agent_record(qid, parsed, calls, **kw):
    info = AgentInfo(verdict="verified", verified=True, values=[], calculations=kw.pop("calculations", []),
                     citations=[], notes=[], history=[], plan=None, visits={"planner": 1}, stop_reason=None,
                     node_notes=[], tool_steps=len(calls), tokens=2700, llm_calls=2, live_requests=1, cache_hits=1)
    fields = {"id": qid, "run_id": ra.RUN_ID, "raw_reply": "{}", "parsed": parsed, "json_valid": True, "retries": 0,
              "prompt_tokens": 2500, "completion_tokens": 200, "latency_s": 3.0, "cached": False,
              "tool_calls": calls, "agent": info} | kw
    return RunRecord(**fields)


QUESTIONS = sa.load_questions(sa.GOLD_FILE, sa.REFUSALS_FILE)
GOLD_PACKS = sa.load_pack(sa.RESULTS_DIR / sa.GOLD_PACK) | sa.load_pack(sa.RESULTS_DIR / sa.REFUSAL_PACK)
MATCHERS = sa.company_matchers()
FACT_CALL = ToolCallRecord(tool="get_financial_fact", arguments={}, evidence=[
    Evidence(source="us-gaap:Revenues 0000320193-24-000123", text="391035")])


def score(qid, parsed, calls, **kw):
    return sa.check(QUESTIONS[qid], agent_record(qid, parsed, calls, **kw), GOLD_PACKS.get(qid, {}), False, MATCHERS)


def fact_call(*values):
    return ToolCallRecord(tool="get_financial_fact", arguments={},
                          evidence=[Evidence(source=f"us-gaap:X {n}", text=ra.figure_text(v)) for n, v in enumerate(values)])


def calc_call(inputs, result, unit="%"):
    out = {"operation": "growth", "inputs": inputs, "result": result, "unit": unit}
    return ToolCallRecord(tool="calculate", arguments={"operation": "growth", "inputs": inputs},
                          evidence=ra.evidence("calculate", out))


def test_a_calculation_is_evidence_only_when_its_inputs_were_fetched():
    """The planner chooses a calculation's inputs: an invented input passed through the calculator must not make
    the result "supported" (B0's derived numbers never had that benefit)."""
    fetched = sa.agent_texts(agent_record("C01", None, [fact_call(8516.0, 14276.0), calc_call([8516.0, 14276.0], 67.6)]))
    assert fetched == ["8516", "14276", "67.6%"]
    invented = sa.agent_texts(agent_record("C01", None, [fact_call(8516.0), calc_call([8516.0, 14000.0], 64.4)]))
    assert invented == ["8516"]
    chained = sa.agent_texts(agent_record("C01", None, [fact_call(10.0, 20.0), calc_call([10.0, 20.0], 30.0, None),
                                                        calc_call([30.0, 20.0], 50.0)]))
    assert chained == ["10", "20", "30", "50%"]                       # an earlier result may feed a later one
    assert sa.agent_texts(agent_record("C01", None, [calc_call([], 5.0)])) == []         # no inputs: nothing


def test_a_round_invented_input_is_not_supported_by_a_nearby_fetched_figure():
    assert sa.agent_texts(agent_record("C01", None, [fact_call(8516.0, 14276.0), calc_call([8516.0, 14000.0], 64.4)])) \
        == ["8516", "14276"]
    assert sa.input_supported(14276.0, [f for f in sa.figures("14276")]) and not sa.input_supported(7, sa.figures("14276"))


def test_under_b0s_rule_no_calculation_result_is_evidence():
    q = QUESTIONS["C01"]
    calls = [fact_call(8516.0, 14276.0), calc_call([8516.0, 14276.0], 67.6)]
    assert sa.agent_texts(agent_record("C01", None, calls), calculations=False) == ["8516", "14276"]
    a = B0Answer(answer="It grew 66.0%.", value=66.0, unit="%", refused=False)
    s = score("C01", a, calls + [calc_call([8516.0, 14276.0], 66.0)])
    assert s["checks"]["unsupported_numbers"] == [] and "66.0%" in s["checks"]["unsupported_numbers_b0_rule"]
    assert "unsupported_numbers_b0_rule" not in sa.check(q, agent_record("C01", a, [], tool_calls=None, agent=None),
                                                         {}, False, MATCHERS)["checks"]


def test_a_sign_flipped_calculation_keeps_its_own_label():
    calc = [{"operation": "growth", "inputs": [14276.0, 8516.0], "result": -40.3, "unit": "%"}]
    q = QUESTIONS["C01"]
    a = B0Answer(answer=f"It fell {abs(q.gold_value)}%.", value=-q.gold_value, unit="%", refused=False)
    s = score("C01", a, [fact_call(8516.0, 14276.0)], calculations=calc)
    assert s["suggested_label"] == "arithmetic_slip" and "sign flipped" in s["reason"]


def test_an_errored_agent_record_is_left_out_of_table_8():
    meta = RunMeta(run_id=ra.RUN_ID, model="m", endpoint="hosted", context="agent", thinking=False, temperature=0.0,
                   max_tokens=1, prompt_version="a", context_pack="p", context_pack_sha256={})
    err = RunRecord(id="A01", run_id=ra.RUN_ID, raw_reply=None, parsed=None, json_valid=False, retries=0,
                    prompt_tokens=0, completion_tokens=0, latency_s=0.0, cached=False, error="RuntimeError: x")
    [s] = sa.score_run(meta, [err], {"A01": QUESTIONS["A01"]}, GOLD_PACKS, MATCHERS)
    assert s["gold_figures_in_pack"] is None and s["suggested_label"] == "malformed_output"


def test_a_derived_number_from_an_invented_input_is_unsupported():
    q = QUESTIONS["C01"]
    a = B0Answer(answer=f"It grew 64.4%. {q.gold_answer}", value=64.4, unit="%", refused=False)
    s = score("C01", a, [fact_call(8516.0), calc_call([8516.0, 14000.0], 64.4)])
    assert "64.4%" in s["checks"]["unsupported_numbers"]


def test_a_wrong_calculated_value_is_a_wrong_line_item_or_an_invented_number_never_a_slip():
    calc = [{"operation": "growth", "inputs": [1.0, 2.0], "result": 100.0, "unit": "%"}]
    a = B0Answer(answer="It grew 100%.", value=100.0, unit="%", refused=False)
    fetched = score("C01", a, [fact_call(1.0, 2.0), calc_call([1.0, 2.0], 100.0)], calculations=calc)
    assert fetched["correct"] is False and fetched["suggested_label"] == "wrong_line_item"
    invented = score("C01", a, [fact_call(1.0), calc_call([1.0, 2.0], 100.0)], calculations=calc)
    assert invented["suggested_label"] == "invented_number"
    pack_run = sa.check(QUESTIONS["C01"], agent_record("C01", a, [], tool_calls=None, agent=None), {}, False, MATCHERS)
    assert pack_run["suggested_label"] != "wrong_line_item"           # B0's labelling is unchanged


def test_an_agent_answer_is_checked_against_what_its_tools_returned():
    good = B0Answer(answer="Apple's total net sales were $391,035 million in fiscal 2024.", value=391035.0,
                    unit="USD millions", company="AAPL", fiscal_year=2024,
                    citations=["us-gaap:Revenues 0000320193-24-000123"], refused=False)
    s = score("A01", good, [FACT_CALL])
    assert s["correct"] and s["checks"]["unsupported_numbers"] == [] and s["checks"]["bad_citations"] == []
    invented = good.model_copy(update={"answer": good.answer + " Services grew to $96.2 billion."})
    assert score("A01", invented, [FACT_CALL])["checks"]["unsupported_numbers"] == ["96.2 billion"]


def test_an_agent_citation_must_name_a_source_its_tools_returned_and_pack_ids_mean_nothing():
    a = B0Answer(answer="x", value=391035.0, unit="USD millions", citations=["P1", "us-gaap:Other 1"], refused=False)
    assert score("A01", a, [FACT_CALL])["checks"]["bad_citations"] == ["P1", "US-GAAP:OTHER 1"]


def test_an_agent_never_saw_the_earlier_answer_so_its_figures_are_not_evidence():
    """F01's pack history holds A01's gold answer ($391,035 million): B0 saw it, the agent did not."""
    a = B0Answer(answer="Up from $391,035 million: $383,285 million in fiscal 2023.", value=383285.0,
                 unit="USD millions", refused=False)
    assert score("F01", a, [])["checks"]["unsupported_numbers"] == ["391,035 million"]
    rag = sa.check(QUESTIONS["F01"], agent_record("F01", a, [], tool_calls=None, agent=None), GOLD_PACKS["F01"],
                   True, MATCHERS)
    assert rag["checks"]["unsupported_numbers"] == []                  # a pack run saw the history's answer


def test_gold_figures_count_as_retrieved_digit_for_digit_in_a_facts_millions():
    q = QUESTIONS["A01"]                                               # gold $391,035 million
    assert sa.gold_figures_in_pack(q, {}, ["391035"]) is True and sa.gold_figures_in_pack(q, {}, ["391,035"]) is True
    assert sa.gold_figures_in_pack(q, {}, ["391035000000", "1391035", "391035.5"]) is False
    assert sa.gold_figures_in_pack(q, {}, []) is False


def test_the_agent_table_counts_cache_hits_as_recorded_even_with_retries():
    meta = RunMeta(run_id=ra.RUN_ID, model="m", endpoint="hosted", context="agent", thinking=False, temperature=0.0,
                   max_tokens=1, prompt_version="a", context_pack="p", context_pack_sha256={})
    scored = [{"id": "A01", "answer_type": "short_numeric", "correct": True, "verified": True, "tool_steps": 1,
               "tool_errors": 0, "stop_reason": None, "live_requests": 2, "cache_hits": 0, "calls": 1,
               "several_values": False},                         # 2 requests (1 HTTP retry) for 1 reply
              {"id": "A02", "answer_type": "short_numeric", "correct": False, "verified": False, "tool_steps": 2,
               "tool_errors": 1, "stop_reason": "timeout", "live_requests": 0, "cache_hits": 2, "calls": 2,
               "several_values": False}]
    text = sa.agent_table([(meta, scored)])
    assert "| live requests / cache hits (chat replies) | 2 / 2 |" in text
    assert "| correct among the not verified (decided) | 0/1 |" in text and "| tool calls that returned an error | 1/3 |" in text
    assert "| stopped by a limit (time, steps, tokens) | 1 (timeout 1) |" in text
    assert sa.agent_table([(RunMeta(**(meta.model_dump() | {"context": "rag"})), scored)]) == ""


# ----------------------------------------------------------------------------------------------- the report
def agent_runs_dir(tmp_path):
    """B0 (the committed M1 runs file) and an A1 made by the stub nodes, side by side, with the committed sheet."""
    runs = tmp_path / "runs"
    runs.mkdir()
    for suffix in (".jsonl", ".meta.json"):
        shutil.copy(sa.RUNS_DIR / f"{B0}{suffix}", runs / f"{B0}{suffix}")
    assert run_cli(runs) == 0
    out = tmp_path / "out"
    shutil.copytree(sa.RESULTS_DIR / "grading", out / "grading")
    return runs, out


def test_the_agent_report_pairs_the_agent_with_b0_and_adds_its_own_checks(tmp_path):
    runs, out = agent_runs_dir(tmp_path)
    assert sa.main(["--runs-dir", str(runs), "--out-dir", str(out)]) == 0
    report = (out / sa.AGENT_REPORT).read_text(encoding="utf-8")
    assert report.startswith(f"# {sa.REPORT_TITLES[sa.AGENT_REPORT]}") and "(written for W2-11)" in report
    assert f"| Answer type | {B0} | {ra.RUN_ID} |" in report
    assert f"### 10. Paired with {B0} (B0)" in report and "### 11. The agent's own checks and cost" in report
    assert "| verified by its critic | 0/102 |" in report                    # the stubs never verify
    assert "no calculation result counted as evidence (B0's rule" in report  # the like-for-like grounding row
    assert "B0's rule" not in (out / sa.REPORT_NAME).read_text(encoding="utf-8")      # not in the M-run report
    assert "| tool calls per question (mean); questions with none | 0.0; 102 |" in report
    keyed = {line.split(",")[0]: line.split(",")[1]
             for line in (out / "grading" / sa.KEY_NAME).read_text(encoding="utf-8").splitlines()[1:]}
    rows = sa.read_sheet_rows(out / "grading" / sa.SHEET_NAME)
    agent_rows = [r for r in rows if keyed.get(r["key"]) == ra.RUN_ID]
    assert agent_rows and {r["task"] for r in agent_rows} == {"grade", "label"}    # summaries to grade, wrong to label
    assert all(not r[c] for r in agent_rows for c in sa.HUMAN_COLUMNS)              # new rows: for a --pending round


def test_an_agent_run_without_b0_beside_it_is_refused(tmp_path):
    assert run_cli(tmp_path / "runs") == 0
    assert sa.main(["--runs-dir", str(tmp_path / "runs"), "--out-dir", str(tmp_path / "out")]) == 1


def test_adding_the_agent_run_changes_no_b0_score(tmp_path):
    runs, out = agent_runs_dir(tmp_path)
    assert sa.main(["--runs-dir", str(runs), "--out-dir", str(out)]) == 0
    assert (out / "scores" / f"{B0}.jsonl").read_bytes() == (sa.SCORES_DIR / f"{B0}.jsonl").read_bytes()


# ----------------------------------------------------------------------------------------------- the handout
def test_the_handout_shows_what_the_agent_retrieved_without_naming_it(tmp_path):
    runs = tmp_path / "runs"
    passage = Evidence(source="MSFT_10-K_2023_Item7_C3_0004", text="Revenue increased $13.6 billion or 7%.")
    fact = ra.evidence("get_financial_fact", FACT)[0]
    summary = next(q for q in QUESTIONS.values() if q.answer_type == "summary")
    recs = [agent_record(qid, B0Answer(answer=f"answer {qid}", refused=False),
                         [ToolCallRecord(tool="search_filings", arguments={}, evidence=[passage]),
                          ToolCallRecord(tool="get_financial_fact", arguments={}, evidence=[fact])]
                         if qid == summary.id else []) for qid in QUESTIONS]
    from eval.run_models import write_run
    write_run(recs, RunMeta(run_id=ra.RUN_ID, model="m", endpoint="hosted", context="agent", thinking=False,
                            temperature=0.0, max_tokens=1, prompt_version="a",
                            context_pack="p", context_pack_sha256={}), runs)
    sheet, key = tmp_path / "sheet.csv", tmp_path / "key.csv"
    k = sa.sheet_key(ra.RUN_ID, summary.id, f"answer {summary.id}")
    sheet.write_text(",".join(sa.SHEET_COLUMNS) + f"\n{k},grade,{summary.question},g,a,,,,,,\n", encoding="utf-8")
    key.write_text(f"key,run_id,id\n{k},{ra.RUN_ID},{summary.id}\n", encoding="utf-8")
    text = grading.handout(sheet, key, sa.RESULTS_DIR / sa.GOLD_PACK, sa.GOLD_FILE, runs_dir=runs)
    assert "### What a system that searched for itself retrieved" in text
    assert "**MSFT 10-K 2023 Item7 C3 0004** (search_filings)" in text and passage.text in text   # no label: id
    assert "**AAPL 10-K FY2024 FY · total net sales (USD millions)** (get_financial_fact)" in text  # whose figure
    assert ra.RUN_ID not in text and "agent" not in text.lower()


# ----------------------------------------------------------------------------------------------- the ablations (W3-01)
@pytest.mark.parametrize(("name", "run_id", "statements_first"), [
    ("no-calculator", "A2_qwen3-8b_agent_think-off_a-v1-no-calculator", True),
    ("no-critic", "A3_qwen3-8b_agent_think-off_a-v1-no-critic", True),
    ("no-statements-first", "A4_qwen3-8b_agent_think-off_a-v1-no-statements-first", False),
])
def test_an_ablation_writes_its_own_run_marked_as_an_ablation_of_a1(tmp_path, name, run_id, statements_first):
    assert ra.ABLATIONS[name].run_id(name) == run_id
    assert run_cli(tmp_path, "--ablation", name) == 0
    meta = json.loads((tmp_path / f"{run_id}.meta.json").read_text(encoding="utf-8"))
    assert (meta["ablation"], meta["ablation_of"]) == (name, ra.RUN_ID)
    assert meta["retrieval"]["statements_first"] is statements_first
    # the same prompts as A1, so its unchanged calls replay from the reply cache
    from agent import composer, planner
    assert (meta["planner_prompt_sha256"], meta["composer_prompt_sha256"]) == (planner.PROMPT_SHA256,
                                                                               composer.PROMPT_SHA256)
    assert len((tmp_path / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()) == 102
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{run_id}.jsonl", f"{run_id}.meta.json"]   # any OS order


def test_a1s_meta_keeps_its_exact_keys(tmp_path):
    assert run_cli(tmp_path) == 0
    meta = json.loads((tmp_path / f"{ra.RUN_ID}.meta.json").read_text(encoding="utf-8"))
    committed = json.loads((ra.RUNS_DIR / f"{ra.RUN_ID}.meta.json").read_text(encoding="utf-8"))
    assert set(meta) == set(committed) - {"replay"} and "ablation" not in meta


@pytest.mark.parametrize(("argv", "message"), [
    (["--ablation", "no-critic", "--run-id", "A2_qwen3-8b_agent_think-off_a-v1-x"], "starts with A3"),
    (["--ablation", "no-critic", "--run-id", ra.RUN_ID], "starts with A3"),
    (["--run-id", "A3_qwen3-8b_agent_think-off_a3-v1"], "pass --ablation no-critic"),
    (["--run-id", "A4_qwen3-8b_agent_think-off_x"], "pass --ablation no-statements-first"),
])
def test_a_run_id_must_say_which_component_is_off(tmp_path, capsys, argv, message):
    assert run_cli(tmp_path, *argv) == 2 and not list(tmp_path.iterdir())
    assert message in capsys.readouterr().err


def test_an_ablation_with_its_own_numbered_run_id_is_accepted(tmp_path):
    assert run_cli(tmp_path, "--ablation", "no-critic", "--run-id", "A3_qwen3-8b_agent_think-off_trial") == 0
    assert (tmp_path / "A3_qwen3-8b_agent_think-off_trial.jsonl").exists()


def test_a_live_ablation_run_builds_the_real_nodes_with_its_switch(tmp_path, monkeypatch):
    seen = []

    async def fake_run_real(items, settings_, max_requests, limits, run_id, ablation=None):
        seen.append((run_id, ablation, max_requests))
        records, _ = await ra.run_items(items, stub_nodes(), limits, run_id)
        return records, None, SimpleNamespace(calls=0, cache_hits=0, http_retries=0)

    monkeypatch.setattr(ra, "run_real", fake_run_real)
    assert ra.main(["--out-dir", str(tmp_path), "--ablation", "no-calculator", "--max-requests", "150"],
                   settings=settings(), limits=LIMITS) == 0
    assert seen == [("A2_qwen3-8b_agent_think-off_a-v1-no-calculator", "no-calculator", 150)]


class FakeLLM:
    """No call is made: building the nodes must not touch the endpoint."""


def test_ablation_nodes_turn_off_exactly_one_component():
    a1 = ra.ablation_nodes(FakeLLM(), None)
    assert all(getattr(a1, n) is not None for n in ("planner", "executor", "composer", "critic"))
    assert ra.ablation_nodes(FakeLLM(), "no-critic").critic is None
    assert ra.ablation_nodes(FakeLLM(), "no-calculator").critic is not None
    assert ra.ablation_nodes(FakeLLM(), "no-statements-first").critic is not None


def run_executor(nodes, plan_):
    state = {"plan": plan_, "tool_results": [], "tokens": 0, "limits": {"max_steps": 10, "max_tokens": 12_000}}
    return asyncio.run(nodes.executor(state))


def test_the_no_calculator_nodes_never_run_the_calculation():
    from agent.state import PlanStep
    calls = []

    async def tools(name, arguments):
        calls.append(name)
        return {**FACT, "fiscal_year": arguments["fiscal_year"]}

    steps = [PlanStep(tool="get_financial_fact", purpose="x", arguments={"ticker": "AAPL", "metric": "revenue",
                                                                         "fiscal_year": y, "fiscal_period": "FY"})
             for y in (2023, 2024)]
    p = Plan(reasoning="x", refuse=False, refusal_rule=None, steps=steps, calculation="growth")
    run_executor(ra.ablation_nodes(FakeLLM(), None, tools), p)
    assert calls == ["get_financial_fact", "get_financial_fact", "calculate"]
    calls.clear()
    out = run_executor(ra.ablation_nodes(FakeLLM(), "no-calculator", tools), p)
    assert calls == ["get_financial_fact", "get_financial_fact"] and "calculator is switched off" in out["errors"][0]


def test_the_no_statements_first_nodes_search_through_the_registry_without_them(monkeypatch):
    from agent.state import PlanStep
    from tools import registry
    seen = []

    async def call_tool_async(name, arguments, *, statements_first=True):
        seen.append((name, statements_first))
        return [{"chunk_id": "c", "text": "t"}]

    monkeypatch.setattr(registry, "call_tool_async", call_tool_async)
    p = Plan(reasoning="x", refuse=False, refusal_rule=None, steps=[PlanStep(tool="search_filings", purpose="x",
                                                          arguments={"query": "risks", "ticker": "AAPL"})])
    run_executor(ra.ablation_nodes(FakeLLM(), "no-statements-first"), p)
    run_executor(ra.ablation_nodes(FakeLLM(), None), p)
    assert seen == [("search_filings", False), ("search_filings", True)]


# ----------------------------------------------------------------------------------------------- scoring the ablations
A3 = ra.ABLATIONS["no-critic"].run_id("no-critic")
A2 = ra.ABLATIONS["no-calculator"].run_id("no-calculator")


def keyed_rows(out):
    keyed = {line.split(",")[0]: line.split(",")[1]
             for line in (out / "grading" / sa.KEY_NAME).read_text(encoding="utf-8").splitlines()[1:]}
    return [(keyed[r["key"]], r) for r in sa.read_sheet_rows(out / "grading" / sa.SHEET_NAME) if r["key"] in keyed]


def test_the_ablations_get_their_own_report_paired_with_a1_and_leave_the_agent_report_alone(tmp_path):
    runs, out = agent_runs_dir(tmp_path)
    assert sa.main(["--runs-dir", str(runs), "--out-dir", str(out)]) == 0
    before = (out / sa.AGENT_REPORT).read_text(encoding="utf-8")
    assert not (out / sa.ABLATION_REPORT).exists()                       # no ablation run, no report
    assert run_cli(runs, "--ablation", "no-critic") == 0
    assert sa.main(["--runs-dir", str(runs), "--out-dir", str(out)]) == 0
    assert (out / sa.AGENT_REPORT).read_text(encoding="utf-8") == before      # A3 is not an agent baseline
    report = (out / sa.ABLATION_REPORT).read_text(encoding="utf-8")
    assert report.startswith(f"# {sa.REPORT_TITLES[sa.ABLATION_REPORT]}") and "(written for W3-01)" in report
    assert f"| Answer type | {ra.RUN_ID} | {A3} |" in report and B0 not in report
    assert f"### 10. Paired with {ra.RUN_ID} (A1): questions right only with the ablation / only with A1" in report
    assert "| All decided answers | 0 / 0 (p = 1.00, n = " in report      # the stubs' answers are the same
    assert f"### 12. What each ablation changed from {ra.RUN_ID}" in report
    assert "| component switched off | no-critic |" in report
    assert "| answers that differ from the reference's (text) | 0/102 |" in report
    a1 = [json.loads(x) for x in (out / "scores" / f"{ra.RUN_ID}.jsonl").read_text(encoding="utf-8").splitlines()]
    a3 = [json.loads(x) for x in (out / "scores" / f"{A3}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(s["id"], s["correct"], s["sheet_key"]) for s in a1] == [(s["id"], s["correct"], s["sheet_key"])
                                                                     for s in a3]       # graded once, shared
    assert [run for run, _ in keyed_rows(out) if run == A3] == []           # nothing new for the graders
    assert all(s["shared_grade"] for s in a3)                              # counted once, under A1, in table 7


def test_a_changed_ablation_answer_is_graded_on_its_own_and_a_grade_on_a1_carries_to_the_unchanged_ones(tmp_path):
    runs, out = agent_runs_dir(tmp_path)
    assert run_cli(runs, "--ablation", "no-calculator") == 0
    summary = next(q.id for q in QUESTIONS.values() if q.answer_type == "summary")
    other = [q.id for q in QUESTIONS.values() if q.answer_type == "summary"][1]
    path = runs / f"{A2}.jsonl"
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    for r in recs:
        if r["id"] == summary:
            r["parsed"]["answer"] += " A changed sentence."
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    assert sa.main(["--runs-dir", str(runs), "--out-dir", str(out)]) == 0
    rows = keyed_rows(out)
    ((_, changed),) = [(run, r) for run, r in rows if run == A2]           # one row: the changed answer
    assert changed["task"] == "grade" and changed["model_answer"].endswith("A changed sentence.")
    # the graders grade A1's row of the other summary: the unchanged A2 answer gets the same grade
    a1_key = next(json.loads(x)["sheet_key"] for x in (out / "scores" / f"{ra.RUN_ID}.jsonl")
                  .read_text(encoding="utf-8").splitlines() if json.loads(x)["id"] == other)
    sheet = out / "grading" / sa.SHEET_NAME
    lines = sheet.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    table = list(csv.DictReader(io.StringIO("\n".join(lines))))
    for row in table:
        if row["key"] == a1_key:
            row.update(grader_1="2", grader_2="2", final="2")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(table)
    sheet.write_text(buf.getvalue(), encoding="utf-8")
    assert sa.main(["--runs-dir", str(runs), "--out-dir", str(out)]) == 0
    a2 = {json.loads(x)["id"]: json.loads(x) for x in (out / "scores" / f"{A2}.jsonl").read_text(encoding="utf-8")
          .splitlines()}
    assert a2[other]["human_grade"] == 2 and a2[other]["correct"] is True
    assert a2[summary]["human_grade"] is None and a2[summary]["correct"] is None      # pending its own grade


def test_an_ablation_without_its_reference_run_is_refused(tmp_path, capsys):
    runs = tmp_path / "runs"
    for suffix in (".jsonl", ".meta.json"):
        shutil.copy(sa.RUNS_DIR / f"{B0}{suffix}", tmp_path / f"{B0}{suffix}")
    assert run_cli(runs, "--ablation", "no-critic") == 0
    for suffix in (".jsonl", ".meta.json"):
        shutil.copy(tmp_path / f"{B0}{suffix}", runs / f"{B0}{suffix}")
    assert sa.main(["--runs-dir", str(runs), "--out-dir", str(tmp_path / "out")]) == 1
    assert f"an ablation of {ra.RUN_ID}, which is not in the runs directory" in capsys.readouterr().err


# ----------------------------------------------------------------------------------------------- review follow-ups (W3-01)
def test_an_ablation_runs_on_the_gold_questions_only(tmp_path, capsys):
    assert run_cli(tmp_path, "--questions", "dev", "--ablation", "no-critic") == 2 and not list(tmp_path.iterdir())
    assert "paired with A1's gold run" in capsys.readouterr().err


def test_a_replay_is_refused_when_the_earlier_file_switched_another_component(tmp_path, capsys):
    first = tmp_path / "first"
    assert run_cli(first, "--ablation", "no-critic") == 0
    meta_path = first / f"{A3}.meta.json"
    meta_path.write_text(json.dumps(json.loads(meta_path.read_text(encoding="utf-8")) | {"ablation": "no-calculator"}),
                         encoding="utf-8")                      # same run id, another switch: not the same run
    out = tmp_path / "replay"
    assert run_cli(out, "--ablation", "no-critic", "--replay-of", str(first / f"{A3}.jsonl")) == 2
    assert not out.exists() and "(ablation differ)" in capsys.readouterr().err


def scored_row(qid, answer, correct, label, key):
    return {"id": qid, "model_answer": answer, "answer_type": "short_numeric", "correct": correct,
            "suggested_label": label, "sheet_key": key}


def test_a_shared_row_needs_the_same_text_verdict_and_suggested_label():
    a1 = SimpleNamespace(run_id=ra.RUN_ID, model_extra={})
    a3 = SimpleNamespace(run_id=A3, model_extra={"ablation": "no-critic", "ablation_of": ra.RUN_ID})
    ref = [scored_row("B01", "x", False, "wrong_line_item", "k1"), scored_row("B02", "y", False, "wrong_line_item", "k2"),
           scored_row("B03", "z", None, None, "k3"), scored_row("B04", "w", False, None, "k4")]
    abl = [scored_row("B01", "x", False, "wrong_line_item", "n1"),        # identical: shares A1's row
           scored_row("B02", "y", False, "invented_number", "n2"),        # its own evidence reads otherwise: own row
           scored_row("B03", "z changed", None, None, "n3"),              # changed text: own row
           scored_row("B04", "w", None, None, "n4")]                      # another automatic verdict: own row
    sa.share_grades([(a1, [], ref), (a3, [], abl)])
    assert [s["sheet_key"] for s in abl] == ["k1", "n2", "n3", "n4"]
    assert [bool(s.get("shared_grade")) for s in abl] == [True, False, False, False]
    assert not any(s.get("shared_grade") for s in ref)                  # the reference owns its rows


# ----------------------------------------------------------------------------------------------- dev set v2 (W3-02)
def test_the_dev_v2_questions_go_to_their_own_directory_with_their_pack(tmp_path):
    assert run_cli(tmp_path, "--questions", "dev2") == 0
    meta = RunMeta.model_validate_json((tmp_path / f"{ra.RUN_ID}.meta.json").read_text(encoding="utf-8"))
    assert list(meta.context_pack_sha256) == [ra.DEV2_PACK] and meta.questions == "dev2"
    lines = (tmp_path / f"{ra.RUN_ID}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 30 and json.loads(lines[0])["id"] == "dev-A21"
    assert ra.QUESTION_SETS["dev2"][1] == ra.DEV2_RUNS_DIR != ra.DEV_RUNS_DIR


def test_a_dev_v2_follow_up_gets_its_earlier_question_only():
    items = {i.id: i for i in ra.load_items([ra.RESULTS_DIR / ra.DEV2_PACK])}
    assert items["dev-F21"].history == (items["dev-A24"].question,)


def test_an_ablation_is_refused_on_dev_v2_too(tmp_path, capsys):
    assert run_cli(tmp_path, "--questions", "dev2", "--ablation", "no-critic") == 2 and not list(tmp_path.iterdir())


# ----------------------------------------------------------------------------------------------- W3-08 pinning
def test_a_gold_run_starts_only_when_its_configuration_is_pinned(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ra, "pin_problems", lambda: ["the planner prompt is not its pinned PROMPT_SHA256"])
    assert ra.main(["--out-dir", str(tmp_path)], settings=settings(), limits=LIMITS) == 2
    assert not list(tmp_path.iterdir()) and "not pinned, nothing run" in capsys.readouterr().err


def test_pin_problems_names_a_changed_prompt_and_uncommitted_code(monkeypatch):
    from agent import composer, planner
    monkeypatch.setattr(ra, "uncommitted", lambda paths=ra.CODE_PATHS: [])
    assert ra.pin_problems() == []
    monkeypatch.setattr(planner, "PROMPT_SHA256", "0" * 64)
    monkeypatch.setattr(ra, "uncommitted", lambda paths=ra.CODE_PATHS: ["agent/composer.py"])
    problems = ra.pin_problems()
    assert len(problems) == 2 and "planner prompt" in problems[0] and "agent/composer.py" in problems[1]
    assert composer.prompt_sha256() == composer.PROMPT_SHA256


def test_uncommitted_lists_changed_and_new_files(monkeypatch):
    out = ' M agent/composer.py\n?? agent/new_rule.py\nR  agent/old.py -> agent/renamed.py\n?? "eval/a b.py"\n'
    monkeypatch.setattr(ra.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=out))
    assert UNCOMMITTED() == ["agent/composer.py", "agent/new_rule.py", "agent/renamed.py", "eval/a b.py"]

    def fail(*a, **k):
        raise OSError("no git")
    monkeypatch.setattr(ra.subprocess, "run", fail)
    assert UNCOMMITTED() == ["(git unavailable)"]


def test_a_run_other_than_a1_records_the_planners_rules(tmp_path):
    from agent import planner
    assert run_cli(tmp_path, "--run-id", "A5_qwen3-8b_agent_think-off_a5-v1") == 0
    meta = json.loads((tmp_path / "A5_qwen3-8b_agent_think-off_a5-v1.meta.json").read_text(encoding="utf-8"))
    assert meta["rules_sha256"] == planner.rules_sha256()
