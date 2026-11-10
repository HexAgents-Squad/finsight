"""eval/run_models.py offline: a fake client, the real context packs and prompts, and the W1-08 scorer reading the
runner's output (the runner and scorer share eval/runs_schema.py)."""
import hashlib
import json

import pytest

import eval.run_models as rm
import eval.score_answers as sa
from core.exceptions import FinSightError
from core.llm import Reply
from core.settings import Settings

PROMPT_SHA256 = {"rag": "ab65e68f902e00d44be3e3844a46834756aa2dada0782b8feb0d959927a80faf",
                 "closed": "4c53a30ec2ba6d7e30bd9533fd482468ab6f83add72b454250d7113f7b9da485"}
RECORDS = {r["id"]: r for r in rm.load_records(rm.RESULTS_DIR)}
RUN = "M1_qwen3-8b_rag_think-off_b0-v1"
VALID = json.dumps({"answer": "Apple's total net sales were $391,035 million.", "value": 391035,
                    "unit": "USD millions", "company": "Apple", "fiscal_year": 2024, "citations": ["P1"],
                    "refused": False, "refusal_rule": None})


def run(coro):
    import asyncio
    return asyncio.run(coro)


class FakeLLM:
    """Replies with the queued texts (or raises the queued errors) and records what it was sent."""

    def __init__(self, *replies, prompt_tokens=100, cached=False):
        self.replies, self.sent = list(replies), []
        self.prompt_tokens, self.cached = prompt_tokens, cached

    async def complete(self, messages, temperature=0.0):
        self.sent.append(messages)
        out = self.replies.pop(0)
        if isinstance(out, Exception):
            raise out
        return Reply(out, out, self.prompt_tokens, 20, self.cached, 0, 0.5)


# --- the frozen prompt ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("context", rm.CONTEXTS)
def test_the_b0_prompt_is_frozen(context):
    """B0 was checked for format on non-gold questions only, then frozen: a change needs a new PROMPT_VERSION
    (and new pins here), never a quiet edit between runs."""
    assert hashlib.sha256(rm.prompt_file(context).read_bytes()).hexdigest() == PROMPT_SHA256[context]
    assert rm.PROMPT_VERSION == "b0-v1"


# --- messages ---------------------------------------------------------------------------------------------------

def test_rag_messages_carry_the_rendered_passages_and_the_question():
    record = RECORDS["A01"]
    system = rm.prompt_file("rag").read_text(encoding="utf-8")
    messages = rm.build_messages(record, "rag", system)
    assert [m["role"] for m in messages] == ["system", "user"] and messages[0]["content"] == system
    rendered = rm.render_passages(rm.Passage(**p) for p in record["passages"])
    assert messages[1]["content"] == f"Passages:\n{rendered}\n\nQuestion: {record['question']}"


def test_a_follow_up_shows_the_earlier_turn_with_its_gold_answer():
    record = RECORDS["F04"]
    messages = rm.build_messages(record, "closed", "S")
    turn = record["history"][0]
    assert messages[1:] == [{"role": "user", "content": f"Question: {turn['question']}"},
                            {"role": "assistant", "content": turn["answer"]},
                            {"role": "user", "content": f"Question: {record['question']}"}]


def test_a_question_without_passages_says_so():
    assert "Passages:\n(none)\n\nQuestion:" in rm.build_messages(RECORDS["G06"], "rag", "S")[-1]["content"]


# --- one answer ---------------------------------------------------------------------------------------------------

def test_a_valid_reply_is_recorded_with_its_cost():
    r = run(rm.answer(FakeLLM(VALID), RUN, RECORDS["A01"], [{"role": "user", "content": "q"}]))
    assert (r.json_valid, r.retries, r.prompt_tokens, r.completion_tokens, r.latency_s) == (True, 0, 100, 20, 0.5)
    assert r.parsed.value == 391035 and r.raw_reply == VALID and r.error is None and r.cached is False


def test_an_invalid_reply_is_asked_again_once_with_the_error():
    llm = FakeLLM("not json at all", VALID)
    r = run(rm.answer(llm, RUN, RECORDS["A01"], [{"role": "user", "content": "q"}]))
    assert r.json_valid and r.retries == 1 and r.prompt_tokens == 200 and r.latency_s == 1.0
    retry = llm.sent[1]
    assert retry[-2] == {"role": "assistant", "content": "not json at all"}
    assert retry[-1]["content"].startswith("Your reply was not valid JSON in the required format:")


def test_two_invalid_replies_leave_no_parsed_answer():
    r = run(rm.answer(FakeLLM("nope", '{"answer": "x"}'), RUN, RECORDS["A01"], [{"role": "user", "content": "q"}]))
    assert (r.json_valid, r.parsed, r.retries, r.raw_reply) == (False, None, 1, '{"answer": "x"}')


def test_thousands_separators_are_repaired_but_not_called_valid():
    bad = VALID.replace("391035", "391,035")
    r = run(rm.answer(FakeLLM(bad, bad), RUN, RECORDS["A01"], [{"role": "user", "content": "q"}]))
    assert r.json_valid is False and r.parsed.value == 391035 and r.retries == 1
    assert "Apple's total net sales were $391,035 million." == r.parsed.answer          # strings untouched


def test_repair_changes_only_separated_numbers_in_value_and_fiscal_year():
    text = '{"answer": "1,234 and 5,678", "value": -1,234.5, "fiscal_year": 2,024, "citations": ["P1,2"]}'
    assert rm.repair_separators(text) == ('{"answer": "1,234 and 5,678", "value": -1234.5, "fiscal_year": 2024, '
                                          '"citations": ["P1,2"]}')


def test_an_endpoint_failure_is_recorded_not_raised():
    error = FinSightError("LLM request failed: 429", error_type="RateLimitError")
    r = run(rm.answer(FakeLLM(error), RUN, RECORDS["A01"], [{"role": "user", "content": "q"}]))
    assert r.error == "RateLimitError: LLM request failed: 429" and r.parsed is None and r.json_valid is False


def test_a_truncated_prompt_is_an_error_not_an_answer():
    llm = FakeLLM(VALID, prompt_tokens=2050)
    r = run(rm.answer(llm, RUN, RECORDS["F05"], [{"role": "user", "content": "q"}], expected_tokens=4703))
    assert r.error.startswith("ContextTruncated: the server saw 2050 prompt tokens of ~4703") and r.parsed is None
    fine = run(rm.answer(FakeLLM(VALID, prompt_tokens=4119), RUN, RECORDS["F05"], [{"role": "user", "content": "q"}],
                         expected_tokens=4703))                                            # Llama's own count
    assert fine.error is None and fine.json_valid


def test_cached_only_when_every_attempt_was_cached():
    assert run(rm.answer(FakeLLM(VALID, cached=True), RUN, RECORDS["A01"], [{"role": "user", "content": "q"}])).cached

    class Mixed(FakeLLM):
        async def complete(self, messages, temperature=0.0):
            reply = await super().complete(messages, temperature)
            return Reply(reply.text, reply.raw, 100, 20, len(self.sent) == 1, 0, 0.5)   # first cached only

    r = run(rm.answer(Mixed("nope", VALID), RUN, RECORDS["A01"], [{"role": "user", "content": "q"}]))
    assert r.cached is False


# --- run ids, settings and meta -------------------------------------------------------------------------------------

def test_run_ids_follow_the_schema_pattern():
    assert rm.run_id_for("M2", "llama3.2-3b", "rag", False) == "M2_llama3.2-3b_rag_think-off_b0-v1"
    with pytest.raises(ValueError, match="does not match"):
        rm.run_id_for("M1", "Qwen3 8B", "rag", False)


def args(**kw):
    import argparse
    return argparse.Namespace(**({"thinking": "off", "base_url": None, "model": None, "api_key": None} | kw))


def test_a_local_server_needs_no_ca_certificate(monkeypatch):
    for k, v in {"LLM_BASE_URL": "https://endpoint.test/v1", "LLM_API_KEY": "team-key", "LLM_MODEL": "Qwen/Qwen3-8B",
                 "SEC_USER_AGENT": "a b@c.d"}.items():
        monkeypatch.setenv(k, v)
    local = rm.settings_for(args(base_url="http://localhost:11434/v1", model="llama3.2-3b-8k", api_key="ollama"))
    assert (local.LLM_BASE_URL, local.LLM_MODEL, local.LLM_CA_CERT, local.LLM_ENABLE_THINKING) == \
        ("http://localhost:11434/v1", "llama3.2-3b-8k", None, False)
    assert local.LLM_API_KEY.get_secret_value() == "ollama"
    hosted = rm.settings_for(args(thinking="on"))
    assert hosted.LLM_BASE_URL == "https://endpoint.test/v1" and hosted.LLM_ENABLE_THINKING is True


def settings(url="https://endpoint.test/v1"):
    return Settings(_env_file=None, LLM_BASE_URL=url, LLM_API_KEY="k", LLM_MODEL="Qwen/Qwen3-8B",
                    SEC_USER_AGENT="a b@c.d")


def test_meta_records_model_server_kind_packs_and_prompt():
    system = rm.prompt_file("rag").read_text(encoding="utf-8")
    meta = rm.run_meta(RUN, settings(), "rag", rm.RESULTS_DIR, system)
    assert (meta.endpoint, meta.context, meta.temperature, meta.max_tokens, meta.prompt_version) == \
        ("hosted", "rag", 0.0, 1024, "b0-v1")
    assert meta.context_pack_sha256 == {n: sa.sha256(rm.RESULTS_DIR / n) for n in (sa.GOLD_PACK, sa.REFUSAL_PACK)}
    assert meta.model_extra["prompt_sha256"] == PROMPT_SHA256["rag"]
    assert rm.run_meta(RUN, settings("http://localhost:11434/v1"), "rag", rm.RESULTS_DIR, system).endpoint == "local"


# --- end to end: runner output scored by the W1-08 scorer ------------------------------------------------------------

def test_a_full_run_is_written_atomically_and_scored(tmp_path):
    """Every question answered "Apple ... 391035": A01 right, most others wrong; the point is the format contract."""
    records = list(RECORDS.values())
    llm = FakeLLM(*[VALID] * len(records))
    system = rm.prompt_file("rag").read_text(encoding="utf-8")
    results = run(rm.run(llm, RUN, records, "rag", system, progress=False))
    assert [r.id for r in results] == [r["id"] for r in records] and len(results) == 102
    out = tmp_path / "runs"
    rm.write_run(results, rm.run_meta(RUN, settings(), "rag", rm.RESULTS_DIR, system), out)
    assert sorted(p.name for p in out.iterdir()) == [f"{RUN}.jsonl", f"{RUN}.meta.json"]
    assert sa.main(["--runs-dir", str(out), "--out-dir", str(tmp_path / "scored")]) == 0
    scored = {json.loads(line)["id"]: json.loads(line)
              for line in (tmp_path / "scored" / "scores" / f"{RUN}.jsonl").read_text(encoding="utf-8").splitlines()}
    assert scored["A01"]["correct"] is True and scored["A02"]["correct"] is False


def test_cli_guards(tmp_path, capsys):
    assert rm.main(["--run", "M1", "--label", "qwen3-8b", "--context", "rag", "--ids", "ZZ9"]) == 1
    assert "unknown question ids" in capsys.readouterr().err
    assert rm.main(["--run", "M1", "--label", "qwen3-8b", "--context", "rag", "--ids", "A01"]) == 1
    assert "must not overwrite the full runs file" in capsys.readouterr().err
    assert rm.main(["--run", "m1", "--label", "qwen3-8b", "--context", "rag"]) == 1     # not the run id pattern
