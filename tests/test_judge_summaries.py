"""eval/judge_summaries.py offline: a fake judge, real gold items, and the W1-08 scorer reading the output."""
import asyncio
import hashlib
import json

import eval.judge_summaries as js
import eval.score_answers as sa
from core.exceptions import FinSightError
from eval.runs_schema import B0Answer, RunRecord

RUN = "M1_qwen3-8b_rag_think-off_b0-v1"
GOLD = {i.id: (i.question, i.gold_answer) for i in sa.validate_file(sa.GOLD_FILE).items
        if i.category == js.SUMMARY_CATEGORY}


def record(qid, answer="Cloud revenue grew on Google Cloud Platform.", parsed=True):
    b0 = B0Answer(answer=answer, refused=False) if parsed else None
    return RunRecord(id=qid, run_id=RUN, raw_reply=answer, parsed=b0, json_valid=parsed, retries=0, prompt_tokens=1,
                     completion_tokens=1, latency_s=0.1, cached=False)


class FakeJudge:
    def __init__(self, *outcomes):
        self.outcomes, self.sent = list(outcomes), []

    async def chat_json(self, messages, schema):
        self.sent.append(messages)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return schema(score=out, reason="matches the key points")


def test_the_rubric_is_frozen():
    assert hashlib.sha256(js.RUBRIC_FILE.read_bytes()).hexdigest() == \
        "56d65aca17eaa929250c67b3ffc448396f03d64332855791bd1af3b6338d5c11"
    assert js.JUDGE_VERSION == "j-v2"


def test_only_summary_answers_with_a_reply_are_judged_blind_to_the_model():
    judge = FakeJudge(2, 1)
    records = [record("E01"), record("A01"), record("E02", parsed=False), record("E03")]
    grades = asyncio.run(js.judge_run(judge, records, GOLD, "RUBRIC"))
    stamp = {"judge_version": "j-v2", "rubric_sha256": hashlib.sha256(b"RUBRIC").hexdigest()}
    assert grades == [{"id": "E01", "score": 2, "reason": "matches the key points", **stamp},
                      {"id": "E03", "score": 1, "reason": "matches the key points", **stamp}]
    system, user = judge.sent[0]
    assert system == {"role": "system", "content": "RUBRIC"}
    question, reference = GOLD["E01"]
    assert user["content"] == (f"Question: {question}\n\nReference answer: {reference}\n\n"
                               f"Answer to grade: Cloud revenue grew on Google Cloud Platform.")
    assert RUN not in user["content"] and "qwen" not in user["content"].lower()


def test_a_failed_judge_call_is_left_out(capsys):
    judge = FakeJudge(FinSightError("bad JSON", error_type="JSONValidationError"), 2)
    grades = asyncio.run(js.judge_run(judge, [record("E01"), record("E02")], GOLD, "R"))
    assert [g["id"] for g in grades] == ["E02"] and "E01: no judge grade (JSONValidationError)" in capsys.readouterr().err


def test_judge_scores_are_strictly_0_1_or_2():
    import pytest
    from pydantic import ValidationError
    for bad in (3, "2", True):
        with pytest.raises(ValidationError):
            js.JudgeScore(score=bad, reason="x")


def test_the_scorer_reads_the_judge_file(tmp_path):
    grades = asyncio.run(js.judge_run(FakeJudge(2, 0), [record("E01"), record("E02")], GOLD, "R"))
    path = tmp_path / f"judge_{RUN}.jsonl"
    path.write_text("".join(json.dumps(g) + "\n" for g in grades), encoding="utf-8")
    assert sa.read_judge(path, set(GOLD)) == {"E01": 2, "E02": 0}


def test_no_runs_exits_2(tmp_path):
    assert js.main(["--runs-dir", str(tmp_path / "none"), "--out-dir", str(tmp_path)]) == 2



def test_passage_markers_are_hidden_from_the_judge():
    judge = FakeJudge(2)
    asyncio.run(js.judge_run(judge, [record("E01", answer="Cloud grew [P1] strongly [P12].")], GOLD, "R"))
    assert judge.sent[0][1]["content"].endswith("Answer to grade: Cloud grew strongly.")


def test_main_writes_one_file_per_run_and_rejects_bad_runs_files(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / f"{RUN}.jsonl").write_text(record("E01").model_dump_json() + "\n", encoding="utf-8")
    for k, v in {"LLM_BASE_URL": "https://endpoint.test/v1", "LLM_API_KEY": "k", "LLM_MODEL": "m",
                 "SEC_USER_AGENT": "a b@c.d"}.items():
        monkeypatch.setenv(k, v)
    from core.settings import Settings
    monkeypatch.setattr(js, "get_settings", lambda: Settings(_env_file=None))      # never the developer's .env

    class Ctx:
        def __init__(self, settings):
            self.judge = FakeJudge(2)

        async def __aenter__(self):
            return self.judge

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(js, "LLMClient", Ctx)
    assert js.main(["--runs-dir", str(runs), "--out-dir", str(tmp_path / "grading")]) == 0
    lines = (tmp_path / "grading" / f"judge_{RUN}.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["score"] for x in lines] == [2]
    (runs / f"{RUN}.jsonl").write_text("{not a record}\n", encoding="utf-8")
    assert js.main(["--runs-dir", str(runs), "--out-dir", str(tmp_path / "grading")]) == 1
