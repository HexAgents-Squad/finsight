"""W1-13/W1-08: LLM-as-judge grades for the summary answers (gold category E) of the model-comparison runs.

Usage (after the runs; one process on the team key at a time):
    python eval/judge_summaries.py                       # every eval/results/model_runs/*.jsonl
    python eval/judge_summaries.py --runs-dir DIR --out-dir DIR

For each run, every summary answer with a parsed reply is graded 0/1/2 against the gold answer by the endpoint's
model under the frozen rubric agent/prompts/judge_summary_rubric.txt (JUDGE_VERSION; a test pins its sha256),
at temperature 0 through core.llm (queue, retries, cache: a re-run is identical and free). The judge sees the
question, the gold answer and the model's answer, never which model wrote it. Output:
eval/results/grading/judge_<run_id>.jsonl ({"id", "score", "reason", ...}), read by eval/score_answers.py, which reports
the judge's agreement with the human grades (W1-14). The human grades decide; the judge is a second opinion, and
judging Qwen with Qwen can favour it, which the report must say.
Every grade also records the judge model, thinking setting and temperature. An error every call would hit (bad
key, unknown model, TLS) stops the run with nothing written, and a run none of whose answers could be judged keeps
its previous file. Exit codes: 0 = written, 1 = invalid input, 2 = no runs, settings missing, or judging failed.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import BaseModel, ConfigDict, Field, ValidationError  # noqa: E402

from core.exceptions import FinSightError  # noqa: E402
from core.llm import LLMClient  # noqa: E402
from core.settings import REPO_ROOT, get_settings  # noqa: E402
from eval.run_retrieval import RESULTS_DIR  # noqa: E402
from eval.runs_schema import RunRecord  # noqa: E402
from eval.score_answers import CITATION_MARK  # noqa: E402
from eval.validate_gold import DEFAULT_PATH as GOLD_FILE  # noqa: E402
from eval.validate_gold import validate_file  # noqa: E402

JUDGE_VERSION = "j-v2"                               # j-v2: the rubric's reply example is valid JSON
JUDGE_TEMPERATURE = 0.0                              # what LLMClient.chat_json always asks for
# Errors every call would hit: stop instead of writing an empty grades file over good ones
FATAL_ERRORS = frozenset({"AuthenticationError", "PermissionDeniedError", "NotFoundError", "TLSError",
                          "TLSConfigError"})


class FatalJudgeError(Exception):
    """The judge cannot work at all (bad key, unknown model, TLS): nothing is written."""
RUBRIC_FILE = REPO_ROOT / "agent" / "prompts" / "judge_summary_rubric.txt"
RUNS_DIR, GRADING_DIR = RESULTS_DIR / "model_runs", RESULTS_DIR / "grading"
SUMMARY_CATEGORY = "qualitative"


class JudgeScore(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    # A strict int, not Literal[0, 1, 2]: the literal accepts true as 1 (True == 1 in Python)
    score: int = Field(ge=0, le=2)
    reason: str = Field(min_length=1)


def judge_messages(rubric: str, question: str, gold: str, answer: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": rubric},
            {"role": "user", "content": f"Question: {question}\n\nReference answer: {gold}\n\nAnswer to grade: {answer}"}]


async def judge_run(llm: LLMClient, records: list[RunRecord], gold: dict[str, tuple[str, str]],
                    rubric: str, about: dict | None = None) -> list[dict]:
    """Grades for the run's summary answers that have a parsed reply; a failed judge call is left out (the
    scorer then shows no judge score for that answer) and reported. [Pn] markers are removed first, as for the
    human graders, so the judge cannot tell a RAG answer from a closed-book one. Each grade records the judge
    version and rubric hash, plus `about` (the judge model and settings). An error in FATAL_ERRORS raises
    FatalJudgeError."""
    stamp = {"judge_version": JUDGE_VERSION, "rubric_sha256": hashlib.sha256(rubric.encode("utf-8")).hexdigest(),
             **(about or {})}
    out = []
    for rec in records:
        if rec.id not in gold or rec.parsed is None:
            continue
        question, reference = gold[rec.id]
        try:
            shown = CITATION_MARK.sub("", rec.parsed.answer).strip()
            grade = await llm.chat_json(judge_messages(rubric, question, reference, shown), JudgeScore)
        except FinSightError as err:
            if err.error_type in FATAL_ERRORS:
                raise FatalJudgeError(f"{err.error_type}: {err.message}") from err
            print(f"  {rec.id}: no judge grade ({err.error_type})", file=sys.stderr)
            continue
        out.append({"id": rec.id, "score": grade.score, "reason": grade.reason, **stamp})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out-dir", type=Path, default=GRADING_DIR)
    ap.add_argument("--gold", type=Path, default=GOLD_FILE)
    args = ap.parse_args(argv)
    run_files = sorted(args.runs_dir.glob("*.jsonl")) if args.runs_dir.exists() else []
    if not run_files:
        print(f"ERROR: no runs files in {args.runs_dir}", file=sys.stderr)
        return 2
    rep = validate_file(args.gold)
    if not rep.ok:
        print(f"ERROR: gold set {args.gold} is invalid", file=sys.stderr)
        return 1
    gold = {i.id: (i.question, i.gold_answer) for i in rep.items if i.category == SUMMARY_CATEGORY}
    try:
        rubric = RUBRIC_FILE.read_text(encoding="utf-8")
        runs = {p.stem: [RunRecord.model_validate_json(line) for line in p.read_text(encoding="utf-8").splitlines()
                         if line.strip()] for p in run_files}
    except (OSError, ValidationError) as err:
        print(f"ERROR: unreadable rubric or runs file: {err}", file=sys.stderr)
        return 1
    try:
        settings = get_settings()
    except ValidationError as err:
        print(f"ERROR: settings are incomplete: {err}", file=sys.stderr)
        return 2

    about = {"judge_model": settings.LLM_MODEL, "judge_thinking": settings.LLM_ENABLE_THINKING,
             "judge_temperature": JUDGE_TEMPERATURE}

    async def go() -> dict[str, list[dict]]:
        async with LLMClient(settings) as llm:
            return {run_id: await judge_run(llm, records, gold, rubric, about) for run_id, records in runs.items()}

    try:
        results = asyncio.run(go())
    except FatalJudgeError as err:
        print(f"ERROR: stopped, nothing written: {err}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    status = 0
    for run_id, grades in results.items():
        path = args.out_dir / f"judge_{run_id}.jsonl"
        eligible = sum(r.id in gold and r.parsed is not None for r in runs[run_id])
        if eligible and not grades:
            print(f"ERROR: {run_id}: no answer could be judged; {path.name} left as it was", file=sys.stderr)
            status = 2
            continue
        tmp = path.with_name(path.name + ".part")
        try:
            tmp.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in grades), encoding="utf-8",
                           newline="\n")
            tmp.replace(path)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        print(f"{run_id}: {len(grades)} summary answers graded ({JUDGE_VERSION}) -> {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
