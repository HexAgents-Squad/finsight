"""W2-11: the agent's gold run (eval/run_agent.py) and how the scorer and the grading tools read it. Offline: stub
or recording nodes, no endpoint, no data/, no .env; the real gold set, refusal tests and context packs."""
from __future__ import annotations

import asyncio
import json
import shutil
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
                                             (7, "7"), (0.0, "0")])
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


def test_a_rounded_statement_of_the_value_counts_as_stating_it():
    a = ra.b0_answer(record(answer="Apple's total net sales were $391.0 billion.", values=[value(391035.0)]))
    assert a.value == 391035.0


def test_several_values_without_a_calculation_give_no_value_and_two_companies_no_company():
    a = ra.b0_answer(record(values=[value(1.0, ticker="PFE"), value(2.0, ticker="JNJ")]))
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
    assert rec.cached and rec.agent.live_requests == 0 and rec.agent.llm_calls == 2


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


def earlier_run(tmp_path, change=None):
    """A first run's files, with its own timings and cost (as a live run records them)."""
    first = tmp_path / "first"
    assert run_cli(first) == 0
    path = first / f"{ra.RUN_ID}.jsonl"
    lines = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        r = json.loads(line)
        r["latency_s"], r["cached"] = 4.5 + n, False
        r["agent"].update(llm_calls=2, live_requests=2)
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
    meta = json.loads((out / f"{ra.RUN_ID}.meta.json").read_text(encoding="utf-8"))
    assert (meta["started"], meta["git_commit"]) == ("2026-11-20T20:51:02+00:00", "abc")
    assert meta["requests"] == {"live": 204, "cache_hits": 0, "http_retries": 0}       # embed_calls dropped
    assert meta["replay"]["requests"] == {"live": 0, "cache_hits": 0, "http_retries": 0}


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
