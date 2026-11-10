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
    _, problem = rm.parse("not json at all")
    assert retry[-1] == {"role": "user", "content": rm.RETRY_INSTRUCTION.format(error=problem)}


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
    monkeypatch.setattr(rm, "ENV_FILE", None)                           # never the developer's .env
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



# --- added after the independent review of W1-13 ---------------------------------------------------------------

def test_a_failed_retry_keeps_a_repairable_first_reply():
    first = VALID.replace("391035", "391,035")
    err = FinSightError("cut", error_type="TruncatedResponseError")
    r = run(rm.answer(FakeLLM(first, err), RUN, RECORDS["A01"], [{"role": "user", "content": "q"}]))
    assert r.error is None and r.parsed.value == 391035 and r.json_valid is False and r.retries == 1
    prose = run(rm.answer(FakeLLM(first, "Sorry, here is prose."), RUN, RECORDS["A01"],
                          [{"role": "user", "content": "q"}]))
    assert prose.parsed.value == 391035 and prose.json_valid is False


def test_an_error_without_any_usable_reply_stays_an_error():
    err = FinSightError("cut", error_type="TruncatedResponseError")
    r = run(rm.answer(FakeLLM("prose", err), RUN, RECORDS["A01"], [{"role": "user", "content": "q"}]))
    assert r.parsed is None and r.error == "TruncatedResponseError: cut"


@pytest.mark.parametrize("seen, expected, cut", [
    (2050, 4703, True),          # Ollama's 2,048 window (measured)
    (4100, 4703, True),          # a 4,096 window: 87% would pass the ratio, the window size gives it away
    (4119, 4703, False),         # the 8k model, Llama's own count (measured)
    (0, 4703, False),            # no token count sent: unknown, not truncated
    (2050, 2000, False),         # a short prompt that happens to be near 2,048 is complete
    (4100, 4100, False),         # the hosted endpoint's exact count: a complete prompt that is 4,100 tokens long
    (4105, 4110, False),         # within the window slack of its own count: no shortfall, no cut
    (100, None, False),          # no expectation: no check
])
def test_truncation_detection(seen, expected, cut):
    assert rm.truncated(seen, expected) is cut


def test_answer_checks_the_cut_at_the_stated_window():
    msgs = [{"role": "user", "content": "q"}]
    flagged = run(rm.answer(FakeLLM(VALID, prompt_tokens=4105), RUN, RECORDS["A01"], msgs, 4599))
    kept = run(rm.answer(FakeLLM(VALID, prompt_tokens=4105), RUN, RECORDS["A01"], msgs, 4599, (8192,)))
    assert flagged.error.startswith("ContextTruncated") and kept.error is None and kept.json_valid


def test_a_failed_attempt_adds_no_seconds():
    err = FinSightError("cut", error_type="TruncatedResponseError")
    r = run(rm.answer(FakeLLM("prose", err), RUN, RECORDS["A01"], [{"role": "user", "content": "q"}]))
    assert r.error and r.latency_s == 0.5                    # the one reply received; the error adds nothing


def test_a_stated_window_replaces_the_common_ones():
    # M2 C09 (measured): a complete prompt that landed near 4,096 by chance on the 8k model
    assert rm.truncated(4105, 4599) is True
    assert rm.truncated(4105, 4599, (8192,)) is False
    assert rm.truncated(8195, 9800, (8192,)) is True         # cut at the stated window
    assert rm.truncated(3000, 9800, (8192,)) is True         # the ratio still applies


def test_a_window_must_be_positive(monkeypatch, tmp_path, capsys):
    cli_env(monkeypatch, tmp_path)
    assert rm.main(["--run", "M2", "--label", "llama3.2-3b", "--context", "rag", "--context-window", "0",
                    "--out-dir", str(tmp_path / "o")]) == 1
    assert "--context-window must be a positive" in capsys.readouterr().err


def test_errors_that_every_question_would_hit_stop_the_run():
    for kind in ("AuthenticationError", "NotFoundError", "TLSError"):
        with pytest.raises(rm.FatalRunError, match=kind):
            run(rm.answer(FakeLLM(FinSightError("no", error_type=kind)), RUN, RECORDS["A01"],
                          [{"role": "user", "content": "q"}]))


def test_run_passes_the_prompts_qwen_count_to_the_truncation_check():
    seen = []
    original = rm.answer

    async def spy(llm, run_id, record, messages, expected_tokens=None, windows=rm.COMMON_WINDOWS):
        seen.append((record["id"], expected_tokens, rm.count_prompt_tokens(messages, len), windows))
        return await original(llm, run_id, record, messages, expected_tokens, windows)

    import unittest.mock as mock
    with mock.patch.object(rm, "answer", spy):
        run(rm.run(FakeLLM(VALID), RUN, [RECORDS["A01"]], "rag", "S", count=len, progress=False))
        run(rm.run(FakeLLM(VALID), RUN, [RECORDS["A01"]], "rag", "S", count=len, progress=False, windows=(8192,)))
    assert seen[0][1] == seen[0][2] > 0
    assert (seen[0][3], seen[1][3]) == (rm.COMMON_WINDOWS, (8192,))


def cli_env(monkeypatch, tmp_path):
    monkeypatch.setattr(rm, "ENV_FILE", None)
    for k, v in {"LLM_BASE_URL": "https://endpoint.test/v1", "LLM_API_KEY": "k", "LLM_MODEL": "Qwen/Qwen3-8B",
                 "SEC_USER_AGENT": "a b@c.d"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(rm, "qwen_counter", lambda: None)


class FakeClientFactory:
    """Stands in for LLMClient(settings) as an async context manager around a FakeLLM."""

    def __init__(self, llm):
        self.llm = llm

    def __call__(self, settings):
        llm = self.llm
        llm.usage = type("U", (), {"calls": 0, "cache_hits": 0, "http_retries": 0, "total_tokens": 0})()

        class Ctx:
            async def __aenter__(self):
                return llm

            async def __aexit__(self, *exc):
                return None

        return Ctx()


def test_a_fatal_error_writes_nothing(monkeypatch, tmp_path, capsys):
    cli_env(monkeypatch, tmp_path)
    monkeypatch.setattr(rm, "LLMClient", FakeClientFactory(FakeLLM(FinSightError("bad key", error_type="AuthenticationError"))))
    assert rm.main(["--run", "M1", "--label", "qwen3-8b", "--context", "rag", "--out-dir", str(tmp_path / "o")]) == 2
    assert "stopped, nothing written: AuthenticationError" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_a_run_where_everything_failed_is_not_written(monkeypatch, tmp_path, capsys):
    cli_env(monkeypatch, tmp_path)
    errors = [FinSightError("busy", error_type="RateLimitError")] * len(RECORDS)
    monkeypatch.setattr(rm, "LLMClient", FakeClientFactory(FakeLLM(*errors)))
    monkeypatch.setattr(rm, "load_records", lambda pack_dir: list(RECORDS.values()))
    assert rm.main(["--run", "M1", "--label", "qwen3-8b", "--context", "rag", "--out-dir", str(tmp_path / "o")]) == 2
    assert "every answer failed; nothing written" in capsys.readouterr().err


def test_a_local_server_needs_a_model_name(monkeypatch, tmp_path, capsys):
    cli_env(monkeypatch, tmp_path)
    assert rm.main(["--run", "M2", "--label", "llama3.2-3b", "--context", "rag",
                    "--base-url", "http://localhost:11434/v1"]) == 1
    assert "--base-url needs --model" in capsys.readouterr().err


def test_a_limited_run_cannot_overwrite_the_runs_file_through_a_relative_path(monkeypatch, capsys):
    monkeypatch.chdir(rm.REPO_ROOT)
    assert rm.main(["--run", "M1", "--label", "qwen3-8b", "--context", "rag", "--ids", "A01",
                    "--out-dir", "eval/results/model_runs"]) == 1
    assert "must not overwrite the full runs file" in capsys.readouterr().err


def test_meta_records_how_the_run_was_made():
    system = rm.prompt_file("rag").read_text(encoding="utf-8")
    meta = rm.run_meta(RUN, settings(), "rag", rm.RESULTS_DIR, system, truncation_check=True,
                       started="2026-11-10T20:51:00+00:00")
    extra = meta.model_extra
    assert (extra["repair"], extra["truncation_check"], extra["json_retries"]) == ("separators-v1", True, 1)
    assert extra["context_window"] is None
    assert rm.run_meta(RUN, settings(), "rag", rm.RESULTS_DIR, system, context_window=8192).model_extra[
        "context_window"] == 8192
    assert extra["framing_sha256"] == rm.framing_sha256() and len(extra["framing_sha256"]) == 64
    assert extra["started"] == "2026-11-10T20:51:00+00:00" and extra["finished"]
    assert extra["git_commit"] is None or len(extra["git_commit"]) == 40
    assert meta.context_pack == "results/context_pack_gold.jsonl + context_pack_refusal.jsonl (cp-v1)"


def test_a_failed_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    from pathlib import Path

    def disk_full(self, other):
        raise OSError("disk full")

    system = rm.prompt_file("rag").read_text(encoding="utf-8")
    meta = rm.run_meta(RUN, settings(), "rag", rm.RESULTS_DIR, system)
    monkeypatch.setattr(Path, "replace", disk_full)
    with pytest.raises(OSError):
        rm.write_run([], meta, tmp_path)
    assert list(tmp_path.iterdir()) == []


# --- added after the review of PR #21 ----------------------------------------------------------------------------

@pytest.mark.parametrize("qid", ["A01", "F01", "RF01"])
def test_rag_and_closed_book_differ_only_in_the_passages(qid):
    rag, closed = (rm.build_messages(RECORDS[qid], c, "S") for c in ("rag", "closed"))
    assert rag[:-1] == closed[:-1]                                       # system prompt and earlier turns
    assert rag[-1]["role"] == closed[-1]["role"] == "user"
    assert rag[-1]["content"].startswith("Passages:\n")
    assert rag[-1]["content"].endswith("\n\n" + closed[-1]["content"])
    assert closed[-1]["content"] == f"Question: {RECORDS[qid]['question']}"
