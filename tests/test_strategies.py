"""eval/strategies.py (W1-09) offline: S1-S3 with a fake client on the real context packs and prompts."""
import json

import pytest

import eval.run_models as rm
import eval.strategies as st
from core.exceptions import FinSightError
from core.llm import Reply
from eval.runs_schema import B0Answer, RunRecord

RECORDS = {r["id"]: r for r in rm.load_records(rm.RESULTS_DIR)}
DEV = {r["id"]: r for r in rm.load_records(rm.RESULTS_DIR, "dev")}


def answer(value=391035, unit="USD millions", company="Apple", refused=False, rule=None, **extra):
    return {"answer": "x", "value": value, "unit": unit, "company": company, "fiscal_year": 2024,
            "citations": ["P1"], "refused": refused, "refusal_rule": rule, **extra}


def cot(**kw):
    return json.dumps({"reasoning": "P1 shows it.", **answer(**kw)})


class FakeLLM:
    """Replies with the queued texts (or raises the queued errors) and records every request."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    async def complete(self, messages, temperature=0.0, sample=0, seed=None, response_format=None):
        self.calls.append({"messages": messages, "temperature": temperature, "sample": sample, "seed": seed,
                           "response_format": response_format})
        out = self.replies.pop(0)
        if isinstance(out, Exception):
            raise out
        return Reply(out, out, 100, 20, False, 0, 0.5)


def run(coro):
    import asyncio
    return asyncio.run(coro)


# --- which strategy, which prompts ---------------------------------------------------------------------------------

def test_a_run_name_picks_its_strategy():
    assert [st.strategy_for(r) for r in ("M1", "M3", "S1", "S2", "S3")] == ["b0", "b0", "s1", "s2", "s3"]
    with pytest.raises(ValueError, match="unknown run 'S4'"):
        st.strategy_for("S4")


@pytest.mark.parametrize("strategy", ["s1", "s2", "s3"])
def test_every_strategy_keeps_the_b0_rules_word_for_word(strategy):
    assert st.system_prompt(strategy).startswith(st.b0_rules())


def test_the_examples_are_valid_b0_answers_from_development_questions_only():
    examples = st.load_examples()
    assert st.example_ids() == ["dev-A03", "dev-C01", "dev-G02", "dev-G01"]
    assert all(e["dev_id"] in DEV and e["question"] == DEV[e["dev_id"]]["question"] for e in examples)
    for e in examples:
        B0Answer.model_validate(e["answer"])
        assert e["answer"]["citations"] == [p["pid"] for p in e["passages"]] or e["answer"]["refused"]


def test_the_example_passages_are_verbatim_filing_lines():
    """Every example line is a line of the chunk it names (checked where the chunks are built; CI has none)."""
    from retrieval.common import chunks_path, read_chunks
    path = chunks_path("C3")
    if not path.exists():
        pytest.skip("data/chunks not built")
    wanted = {p["chunk_id"]: p["lines"] for e in st.load_examples() for p in e["passages"]}
    found = {c.chunk_id: c.text.splitlines() for c in read_chunks(path) if c.chunk_id in wanted}
    assert set(found) == set(wanted)
    for cid, lines in wanted.items():
        assert all(line in found[cid] for line in lines), cid


def test_s1_puts_the_examples_between_the_system_prompt_and_the_question():
    messages = st.s1_messages(RECORDS["A01"])
    plain = rm.build_messages(RECORDS["A01"], "rag", st.system_prompt("s1"))
    examples = st.example_messages()
    assert messages == [plain[0], *examples, *plain[1:]]
    assert [m["role"] for m in examples] == ["user", "assistant"] * 4
    assert examples[0]["content"].startswith("Passages:\n[P1] [META 10-Q FY2025 Q2 · Part I Item 1]\n")


def test_guided_decoding_asks_for_the_reasoning_first():
    fmt = st.guided(st.CoTAnswer)
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert list(fmt["json_schema"]["schema"]["properties"])[0] == "reasoning"
    assert st.CoTAnswer.model_validate_json(cot()).b0() == B0Answer.model_validate(answer())


def test_prompt_hashes_cover_every_file_of_the_strategy():
    assert len({st.prompt_sha256(s) for s in ("s1", "s2", "s3")}) == 3
    assert st.EXAMPLES_FILE.read_text(encoding="utf-8") in st.prompt_texts("s1")
    assert st.EXTRACT_FILE.read_text(encoding="utf-8") in st.prompt_texts("s3")


# --- S1 ------------------------------------------------------------------------------------------------------------

def test_s1_is_one_guided_call_recorded_like_b0():
    llm = FakeLLM(json.dumps(answer()))
    r = run(st.run_s1(llm, "S1_qwen3-8b_rag_think-off_s1-v1", RECORDS["A01"]))
    assert r.parsed == B0Answer.model_validate(answer()) and r.json_valid and r.steps is None
    assert llm.calls[0]["response_format"]["json_schema"]["name"] == "B0Answer"
    assert (llm.calls[0]["temperature"], llm.calls[0]["seed"]) == (0.0, None)


def test_an_invalid_reply_is_asked_again_once_with_the_error():
    llm = FakeLLM("not json", json.dumps(answer()))
    r = run(st.run_s1(llm, "S1_qwen3-8b_rag_think-off_s1-v1", RECORDS["A01"]))
    assert r.json_valid and r.retries == 1 and (r.prompt_tokens, r.completion_tokens) == (200, 40)
    assert "not valid JSON" in llm.calls[1]["messages"][-1]["content"]


def test_a_fatal_error_stops_the_run():
    with pytest.raises(rm.FatalRunError):
        run(st.run_s1(FakeLLM(FinSightError("no", error_type="AuthenticationError")), "S1_x_rag_think-off_s1-v1",
                      RECORDS["A01"]))


# --- S2 ------------------------------------------------------------------------------------------------------------

def test_s2_samples_with_seeds_and_keeps_the_first_answer_of_the_majority():
    replies = [cot(value=1.0), cot(value=391035, company="Apple Inc."), cot(value=391.035, unit="USD billions"),
               "garbage", "garbage", cot(value=391035)]                       # sample 3 invalid twice
    llm = FakeLLM(*replies)
    r = run(st.run_s2(llm, "S2_qwen3-8b_rag_think-off_s2-v1", RECORDS["A01"]))
    assert [c["seed"] for c in llm.calls] == [0, 1, 2, 3, 3, 4]
    assert {c["temperature"] for c in llm.calls} == {st.SAMPLE_TEMPERATURE}
    assert r.agreement == "3/4" and r.parsed.company == "Apple Inc."          # the first of the 3 that agree
    assert [s.name for s in r.steps] == [f"sample {i}" for i in range(5)] and not r.steps[3].json_valid
    assert r.retries == 1 and r.latency_s == pytest.approx(3.0)


def test_s2_without_a_valid_sample_has_no_answer():
    llm = FakeLLM(*([FinSightError("busy", error_type="RateLimitError")] * 5))
    r = run(st.run_s2(llm, "S2_qwen3-8b_rag_think-off_s2-v1", RECORDS["A01"]))
    assert r.parsed is None and not r.json_valid and r.agreement == "0/5" and r.error.startswith("RateLimitError")


@pytest.mark.parametrize("a, b, same", [
    (answer(value=47516), answer(value=47.516, unit="USD billions"), True),
    (answer(value=47516), answer(value=47600), False),
    (answer(value=None, refused=True, rule="R2"), answer(value=None, refused=True, rule="R2", company="Citi"), False),
    (answer(value=None, company="Johnson & Johnson"), answer(value=None, company="J&J"), True),
    (answer(value=67.6, unit="%"), answer(value=67.64, unit="%"), True),
])
def test_votes_compare_the_decision_not_the_wording(a, b, same):
    assert (st.vote_key(B0Answer.model_validate(a)) == st.vote_key(B0Answer.model_validate(b))) is same


def test_a_tie_goes_to_the_decision_seen_first():
    answers = [B0Answer.model_validate(answer(value=v)) for v in (1, 2, 2, 1)]
    assert st.vote(answers) == (0, 2)


# --- S3 ------------------------------------------------------------------------------------------------------------

EXTRACTION = json.dumps({"sub_questions": ["Apple net sales FY2024"], "figures": [
    {"label": "net sales", "value": 391035, "unit": "USD millions", "company": "Apple", "fiscal_year": 2024,
     "period": "FY", "pid": "P1", "quote": "Total net sales | 391,035"}], "missing": [], "out_of_scope": False})


def test_s3_researches_then_answers_with_the_research_after_the_question():
    llm = FakeLLM(EXTRACTION, json.dumps(answer()))
    r = run(st.run_s3(llm, "S3_qwen3-8b_rag_think-off_s3-v1", RECORDS["A01"]))
    assert [c["response_format"]["json_schema"]["name"] for c in llm.calls] == ["Extraction", "B0Answer"]
    last = llm.calls[1]["messages"][-1]["content"]
    assert last.endswith(st.RESEARCH_HEADING + "\n" + st.Extraction.model_validate_json(EXTRACTION).model_dump_json())
    assert f"Question: {RECORDS['A01']['question']}" in last
    assert llm.calls[1]["messages"][0]["content"] == st.system_prompt("s3")
    assert [s.name for s in r.steps] == ["extract", "answer"] and r.json_valid and r.prompt_tokens == 200


def test_s3_answers_even_when_the_research_step_fails():
    llm = FakeLLM("no", "still no", json.dumps(answer()))
    r = run(st.run_s3(llm, "S3_qwen3-8b_rag_think-off_s3-v1", RECORDS["A01"]))
    note = llm.calls[2]["messages"][-1]["content"].split(st.RESEARCH_HEADING + "\n")[1]
    assert json.loads(note) == {"failed": "the research step gave no valid JSON"}
    assert r.json_valid and not r.steps[0].json_valid


# --- the runner with a strategy --------------------------------------------------------------------------------------

def test_a_strategy_record_keeps_its_steps_and_a_b0_record_its_old_bytes(tmp_path):
    from tests.test_run_models import settings
    s2 = run(st.run_s2(FakeLLM(*[cot()] * 5), "S2_qwen3-8b_rag_think-off_s2-v1", RECORDS["A01"]))
    meta = rm.run_meta("S2_qwen3-8b_rag_think-off_s2-v1", settings(), "rag", rm.RESULTS_DIR,
                       st.system_prompt("s2"), strategy="s2")
    path = rm.write_run([s2], meta, tmp_path)
    assert RunRecord.model_validate_json(path.read_text(encoding="utf-8")) == s2
    assert (meta.temperature, meta.prompt_version, meta.model_extra["samples"]) == (st.SAMPLE_TEMPERATURE, "s2-v1", 5)
    b0 = rm.RUNS_DIR / "M1_qwen3-8b_rag_think-off_b0-v1.jsonl"
    records = [RunRecord.model_validate_json(line) for line in b0.read_text(encoding="utf-8").splitlines()]
    m1 = rm.write_run(records, meta, tmp_path / "b0")
    assert m1.read_bytes() == b0.read_bytes()


def test_a_dev_run_reads_only_the_dev_pack_and_is_kept_apart(monkeypatch, tmp_path, capsys):
    from tests.test_run_models import FakeClientFactory, cli_env
    cli_env(monkeypatch, tmp_path)
    monkeypatch.setattr(rm, "LLMClient", FakeClientFactory(FakeLLM(*[json.dumps(answer())] * len(DEV))))
    out = tmp_path / "dev"
    assert rm.main(["--run", "S1", "--label", "qwen3-8b", "--context", "rag", "--questions", "dev",
                    "--out-dir", str(out)]) == 0
    meta = json.loads((out / "S1_qwen3-8b_rag_think-off_s1-v1.meta.json").read_text(encoding="utf-8"))
    assert list(meta["context_pack_sha256"]) == ["context_pack_dev.jsonl"] and meta["examples"] == st.example_ids()
    assert rm.main(["--run", "S1", "--label", "qwen3-8b", "--context", "rag", "--questions", "dev",
                    "--out-dir", str(rm.RUNS_DIR)]) == 1
    assert rm.main(["--run", "S2", "--label", "qwen3-8b", "--context", "closed"]) == 1
    assert "must not be written next to the gold runs" in capsys.readouterr().err


# --- frozen before the gold runs -----------------------------------------------------------------------------------

FROZEN = {"s1": "53954d70fd69cbff948dcf00ae932ea3ca1a5a01fd0a4653ff75dbdd8a1d3f11",
          "s2": "df7382e8f715d278c74212b43a4999ba53510096398f21666ef33ffd391d9c64",
          "s3": "ea129236ce2a404534cbd9520d9cc26004f262f1beb62e85c500476097257959"}


@pytest.mark.parametrize("strategy", sorted(FROZEN))
def test_the_strategy_prompts_are_frozen(strategy):
    """Tuned on the development set only (three passes, ADR-018), then frozen before the gold runs: a change needs
    a new version in PROMPT_VERSIONS and new pins here, never a quiet edit between runs."""
    assert st.prompt_sha256(strategy) == FROZEN[strategy]
    assert st.PROMPT_VERSIONS[strategy] == f"{strategy}-v1"


def test_no_strategy_prompt_quotes_a_development_answer_outside_the_s1_examples():
    """The S1 examples are development questions by design (left out when S1 is judged on that set); no other
    prompt text may carry a development figure, or the tuning would grade itself."""
    import re

    from eval.validate_dev_set import DEV_FILE
    figures = {f for line in DEV_FILE.read_text(encoding="utf-8").splitlines()
               for f in re.findall(r"\d{1,3}(?:,\d{3})+|\d+\.\d+", json.loads(line)["gold_answer"])}
    for strategy in FROZEN:
        text = st.system_prompt(strategy) + (st.EXTRACT_FILE.read_text(encoding="utf-8") if strategy == "s3" else "")
        text = text.replace(st.b0_rules(), "")
        assert not {f for f in figures if f in text or f.replace(",", "") in text}, strategy
