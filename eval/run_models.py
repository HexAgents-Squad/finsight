"""W1-13: run one model over the context pack with the frozen zero-shot prompt B0 (docs/week1_plan.txt §2.4-2.5, §3);
W1-09: the same with a prompting strategy (S1 few-shot, S2 chain of thought + self-consistency, S3 decomposition;
eval/strategies.py).

Usage (the endpoint from .env; one run at a time on the team key):
    python eval/run_models.py --run M1 --label qwen3-8b --context rag
    python eval/run_models.py --run M3 --label qwen3-8b --context closed
    python eval/run_models.py --run M2 --label llama3.2-3b --context rag \\
        --base-url http://localhost:11434/v1 --model llama3.2-3b-8k --api-key ollama \\
        --timeout 300 --context-window 8192                                  # local, in parallel
    python eval/run_models.py --run M1 --label qwen3-8b --context rag --ids A01 RF01 --out-dir /tmp/check
    python eval/run_models.py --run S2 --label qwen3-8b --context rag                # a W1-09 strategy (rag only)
    python eval/run_models.py --run S1 --label qwen3-8b --context rag --questions dev # tuning: development set

Every model sees exactly the same inputs: the question, the earlier turns of a follow-up with their gold answers,
and (rag) the context pack's passages rendered by eval.context_pack.render_passages, under the frozen prompt
agent/prompts/b0_zero_shot_<rag|closed>.txt (PROMPT_VERSION; a test pins the files' sha256, so a changed prompt
needs a new version). Temperature 0, LLM_MAX_COMPLETION_TOKENS (1,024), no tools. A reply that is not valid B0
JSON (eval.runs_schema.B0Answer) is sent back once with the error. Calls go through core.llm (one at a time per
endpoint, 2/4/8 s retries, reply cache), so a re-run gives the same answers for free (latency is replayed from the
cache; the meta's timestamps and commit are the re-run's own). An error that retrying cannot fix (bad key, unknown
model, TLS) stops the run, and a run whose every answer failed is not written, so a mistyped command never
replaces a good runs file.

Output: eval/results/model_runs/<run_id>.jsonl (one RunRecord per question, gold order then refusal tests) and
<run_id>.meta.json (RunMeta), written when the run completes. run_id = <run>_<label>_<context>_think-<on|off>_
<prompt version>: M<n> runs use B0 (b0-v1), S1-S3 runs their strategy's prompts (s1-v1, ...). With --questions
dev the run answers the 15 development questions (eval/results/context_pack_dev.jsonl, built by
`python eval/context_pack.py --dev`) and is written to eval/results/dev/model_runs/, never next to the gold runs.
Exit codes: 0 = written, 1 = invalid input, 2 = settings or packs missing.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError  # noqa: E402

from core.exceptions import FinSightError  # noqa: E402
from core.llm import LLMClient, TokenCounter, count_prompt_tokens, extract_json  # noqa: E402
from core.settings import ENV_FILE, HTTPS, REPO_ROOT, Settings  # noqa: E402
from eval.context_pack import (  # noqa: E402
    DEV_PACK,
    GOLD_PACK,
    PACK_VERSION,
    REFUSAL_PACK,
    Passage,
    render_passages,
    sha256,
)
from eval.run_retrieval import RESULTS_DIR  # noqa: E402
from eval.runs_schema import RUN_ID_PATTERN, B0Answer, RunMeta, RunRecord  # noqa: E402

PROMPT_VERSION = "b0-v1"
PROMPTS_DIR = REPO_ROOT / "agent" / "prompts"
CONTEXTS = ("rag", "closed")
RUNS_DIR = RESULTS_DIR / "model_runs"
DEV_RUNS_DIR = RESULTS_DIR / "dev" / "model_runs"
QUESTION_SETS = {"gold": (GOLD_PACK, REFUSAL_PACK), "dev": (DEV_PACK,)}     # the packs each set reads
JSON_RETRIES = 1                                    # the plan's "one retry on invalid JSON"
TEMPERATURE = 0.0
NO_PASSAGES = "(none)"
RETRY_INSTRUCTION = ("Your reply was not valid JSON in the required format: {error}\n"
                     "Reply again with ONE JSON object only, with exactly the fields described.")
MAX_ERROR_CHARS = 800
# A server that reports far fewer prompt tokens than the prompt has dropped part of it (Ollama's default window
# silently keeps the last 2,048 tokens: measured 2,050 of a 4,703-token prompt, with the system prompt lost).
# Llama's tokenizer counts ~12% fewer tokens than Qwen's (4,119 vs 4,703 on the same prompt), so 0.7 separates.
TRUNCATION_RATIO = 0.7
# A server that reports exactly its window size (Ollama kept 2,050 of a longer prompt with a 2,048 window) cut the
# prompt even when the ratio above cannot tell (a 4,096 window keeps 87% of a 4,703-token prompt in Llama tokens).
# These are only a guess at the server's window: a run that states its window (--context-window, e.g. 8192 for the
# Ollama 8k model) is checked at that size instead, since a complete prompt can land near 4,096 by chance (M2 C09:
# 4,105 Llama tokens of ~4,599 Qwen, ratio 0.89, flagged before the window was stated).
COMMON_WINDOWS = (2048, 4096)
WINDOW_SLACK = 16
# The one repair applied after the JSON retry also failed: a number written with thousands separators in "value"
# or "fiscal_year" (Llama-3.2-3B writes "value": 10,714 even when told not to, measured on a format check).
# json_valid stays false (the reply was invalid as written); `parsed` keeps the content scoreable.
REPAIR_VERSION = "separators-v1"                  # recorded in the meta: the repair is part of how a run is scored
SEPARATED_NUMBER = re.compile(r'("(?:value|fiscal_year)"\s*:\s*)(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?)(?=\s*[,}])')


def prompt_file(context: str) -> Path:
    return PROMPTS_DIR / f"b0_zero_shot_{context}.txt"


def build_messages(record: dict, context: str, system: str) -> list[dict[str, str]]:
    """System prompt, the earlier turns (question, gold answer), then the question with (rag) its passages."""
    messages = [{"role": "system", "content": system}]
    for turn in record["history"]:
        messages += [{"role": "user", "content": f"Question: {turn['question']}"},
                     {"role": "assistant", "content": turn["answer"]}]
    question = f"Question: {record['question']}"
    if context == "rag":
        passages = render_passages(Passage(**p) for p in record["passages"]) or NO_PASSAGES
        question = f"Passages:\n{passages}\n\n{question}"
    messages.append({"role": "user", "content": question})
    return messages


def repair_separators(text: str) -> str:
    """'"value": 10,714' -> '"value": 10714'; nothing else in the reply changes."""
    return SEPARATED_NUMBER.sub(lambda m: m.group(1) + m.group(2).replace(",", ""), text)


def parse(text: str) -> tuple[B0Answer | None, str | None]:
    """The B0 answer in a reply, or (None, the validation error)."""
    try:
        return B0Answer.model_validate_json(extract_json(text)), None
    except ValidationError as err:
        return None, str(err)[:MAX_ERROR_CHARS]


# Errors that the same request would hit again: the run stops instead of writing a file of them
FATAL_ERRORS = frozenset({"AuthenticationError", "PermissionDeniedError", "NotFoundError", "TLSError",
                          "TLSConfigError"})


class FatalRunError(Exception):
    """An endpoint error that every remaining question would hit (bad key, unknown model, TLS)."""


def truncated(prompt_tokens: int, expected_tokens: int | None,
              windows: tuple[int, ...] = COMMON_WINDOWS) -> bool:
    """The server saw less than the prompt: too few tokens, or exactly a window size in `windows` (the server's
    stated window, else COMMON_WINDOWS). 0 means the server sent no token count, which is unknown, not truncated."""
    if not expected_tokens or prompt_tokens <= 0:
        return False
    # A cut needs a shortfall too: on the hosted endpoint the count is exact, so a complete 4,100-token prompt
    # reports 4,100 and must not be taken for a 4,096 window
    at_window = any(expected_tokens > w and abs(prompt_tokens - w) <= WINDOW_SLACK
                    and prompt_tokens < expected_tokens - WINDOW_SLACK for w in windows)
    return prompt_tokens < TRUNCATION_RATIO * expected_tokens or at_window


async def answer(llm: LLMClient, run_id: str, record: dict, messages: list[dict[str, str]],
                 expected_tokens: int | None = None, windows: tuple[int, ...] = COMMON_WINDOWS) -> RunRecord:
    """One question: the reply, re-asked once if it is not valid B0 JSON. Tokens and seconds add up over the
    replies received (a failed attempt adds none: its time would differ between a live run and a replay);
    `cached` is true only when every attempt came from the cache. With `expected_tokens` (the prompt's Qwen
    count), a reply whose server saw less than the prompt is recorded as ContextTruncated, not scored. If no
    reply is valid, repair_separators is tried on each reply, latest first (parsed, json_valid false); a usable
    first reply is kept even if the retry fails. An error in FATAL_ERRORS raises FatalRunError."""
    tokens_in = tokens_out = retries = 0
    seconds, cached, raw, parsed, error = 0.0, True, None, None, None
    texts: list[str] = []
    working = list(messages)
    for attempt in range(JSON_RETRIES + 1):
        try:
            reply = await llm.complete(working, temperature=TEMPERATURE)
        except FinSightError as err:
            if err.error_type in FATAL_ERRORS:
                raise FatalRunError(f"{err.error_type}: {err.message}") from err
            error, cached = f"{err.error_type}: {err.message}", False
            break
        seconds += reply.seconds
        tokens_in, tokens_out = tokens_in + reply.prompt_tokens, tokens_out + reply.completion_tokens
        cached = cached and reply.cached
        raw = reply.raw
        texts.append(reply.text)
        if attempt == 0 and truncated(reply.prompt_tokens, expected_tokens, windows):
            error = (f"ContextTruncated: the server saw {reply.prompt_tokens} prompt tokens of ~{expected_tokens} "
                     f"(raise its context window, e.g. Ollama num_ctx)")
            texts.clear()                           # a reply to a cut prompt is never scored
            break
        parsed, problem = parse(reply.text)
        if parsed is not None or attempt == JSON_RETRIES:
            break
        retries += 1
        working = [*working, {"role": "assistant", "content": reply.text},
                   {"role": "user", "content": RETRY_INSTRUCTION.format(error=problem)}]
    json_valid = parsed is not None
    for text in reversed(texts if parsed is None else []):
        parsed, _ = parse(repair_separators(text))
        if parsed is not None:
            error = None                            # an answer was obtained; a failed retry does not erase it
            break
    return RunRecord(id=record["id"], run_id=run_id, raw_reply=raw, parsed=parsed, json_valid=json_valid,
                     retries=retries, prompt_tokens=tokens_in, completion_tokens=tokens_out,
                     latency_s=round(seconds, 3), cached=cached and error is None, error=error)


def load_records(pack_dir: Path, questions: str = "gold") -> list[dict]:
    """Gold questions then refusal tests (or the development questions), in pack order."""
    records = []
    for name in QUESTION_SETS[questions]:
        records += [json.loads(line) for line in (pack_dir / name).read_text(encoding="utf-8").splitlines()]
    return records


def run_id_for(run: str, label: str, context: str, thinking: bool, version: str = PROMPT_VERSION) -> str:
    run_id = f"{run}_{label}_{context}_think-{'on' if thinking else 'off'}_{version}"
    if not re.fullmatch(RUN_ID_PATTERN, run_id):
        raise ValueError(f"run id {run_id!r} does not match {RUN_ID_PATTERN} (lower-case label, run like M1)")
    return run_id


def git_commit() -> str | None:
    """The code's commit, for the record; None outside a git checkout."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def framing_sha256(extra: tuple[str, ...] = ()) -> str:
    """The fixed text around the prompt (question and passage framing, retry instruction, and a strategy's own,
    e.g. S3's research heading): part of what every model is sent, so recorded like the prompt itself."""
    framing = "\n".join(["Question: ", "Passages:\n", NO_PASSAGES, RETRY_INSTRUCTION, *extra])
    return hashlib.sha256(framing.encode("utf-8")).hexdigest()


def run_meta(run_id: str, settings: Settings, context: str, pack_dir: Path, system: str,
             truncation_check: bool = False, started: str | None = None,
             context_window: int | None = None, strategy: str = "b0", questions: str = "gold") -> RunMeta:
    packs = QUESTION_SETS[questions]
    common = dict(run_id=run_id, model=settings.LLM_MODEL,
                  endpoint="hosted" if settings.LLM_BASE_URL.lower().startswith(HTTPS) else "local",
                  context=context, thinking=bool(settings.LLM_ENABLE_THINKING),
                  max_tokens=settings.LLM_MAX_COMPLETION_TOKENS,
                  context_pack=f"{pack_dir.name}/{' + '.join(packs)} ({PACK_VERSION})",
                  context_pack_sha256={n: sha256(pack_dir / n) for n in packs},
                  json_retries=JSON_RETRIES, truncation_check=truncation_check, context_window=context_window,
                  git_commit=git_commit(), started=started, finished=datetime.now(UTC).isoformat(timespec="seconds"))
    if strategy == "b0":
        return RunMeta(**common, temperature=TEMPERATURE, prompt_version=PROMPT_VERSION,
                       prompt_sha256=hashlib.sha256(system.encode("utf-8")).hexdigest(),
                       framing_sha256=framing_sha256(), repair=REPAIR_VERSION)
    from eval import strategies as st
    extra: dict[str, Any] = {"strategy": strategy, "response_format": "json_schema (strict guided decoding)",
                             "repair": None}
    if strategy == "s1":
        extra["examples"] = st.example_ids()
    if strategy == "s2":
        extra |= {"samples": st.SAMPLES, "seeds": list(range(st.SAMPLES)), "vote": "majority of valid samples on "
                  "refused, rule, company and value (3 significant digits); ties to the first decision seen"}
    return RunMeta(**common, temperature=st.SAMPLE_TEMPERATURE if strategy == "s2" else TEMPERATURE,
                   prompt_version=st.PROMPT_VERSIONS[strategy], prompt_sha256=st.prompt_sha256(strategy),
                   framing_sha256=framing_sha256((st.RESEARCH_HEADING,) if strategy == "s3" else ()), **extra)


def write_run(records: list[RunRecord], meta: RunMeta, out_dir: Path) -> Path:
    """Meta first, then the records, each through a temporary file: a crash leaves the old pair, or a new meta
    with the old records (the scorer then refuses a mismatch), never half a file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{meta.run_id}.jsonl"
    # a B0 record leaves out the strategy fields it never has, so B0 runs files keep their exact bytes
    lines = "".join(r.model_dump_json(exclude={k for k in ("steps", "agreement") if getattr(r, k) is None}) + "\n"
                    for r in records)
    for target, text in ((out_dir / f"{meta.run_id}.meta.json", meta.model_dump_json(indent=2) + "\n"),
                         (path, lines)):
        tmp = target.with_name(target.name + ".part")
        try:
            tmp.write_text(text, encoding="utf-8", newline="\n")
            tmp.replace(target)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
    return path


def settings_for(args: argparse.Namespace) -> Settings:
    """.env settings, with a different server for a local model (no CA certificate for plain HTTP)."""
    overrides: dict[str, Any] = {"LLM_ENABLE_THINKING": args.thinking == "on"}
    if args.base_url:
        if not args.model:
            raise ValueError("--base-url needs --model (the .env model name belongs to the .env server)")
        overrides["LLM_BASE_URL"] = args.base_url
        if not args.base_url.lower().startswith(HTTPS):
            overrides["LLM_CA_CERT"] = ""           # the group CA vouches for the hosted endpoint only
    if args.model:
        overrides["LLM_MODEL"] = args.model
    if args.api_key:
        overrides["LLM_API_KEY"] = args.api_key
    if getattr(args, "timeout", None):
        overrides["LLM_TIMEOUT"] = args.timeout       # a timed-out request is retried and generated again
    return Settings(_env_file=ENV_FILE, **overrides)


def qwen_counter() -> TokenCounter | None:
    """The Qwen tokenizer for the truncation check, or None (then the check is off, with a note)."""
    try:
        from core.tokens import count_tokens
        count_tokens("warm-up")
        return count_tokens
    except (ImportError, FinSightError) as err:
        print(f"note: truncation check off (Qwen tokenizer unavailable: {err})", file=sys.stderr)
        return None


async def run(llm: LLMClient, run_id: str, records: list[dict], context: str, system: str,
              count: TokenCounter | None = None, progress: bool = True,
              windows: tuple[int, ...] = COMMON_WINDOWS, strategy: str = "b0") -> list[RunRecord]:
    """Every record with B0 (the truncation check on) or with a W1-09 strategy (eval/strategies.py; hosted
    endpoint only, so no truncation check)."""
    runner = None
    if strategy != "b0":
        from eval.strategies import RUNNERS
        runner = RUNNERS[strategy]
    out = []
    for n, record in enumerate(records, start=1):
        if runner is None:
            messages = build_messages(record, context, system)
            expected = count_prompt_tokens(messages, count) if count else None
            result = await answer(llm, run_id, record, messages, expected, windows)
        else:
            result = await runner(llm, run_id, record)
        out.append(result)
        if progress:
            status = result.error or ("ok" if result.json_valid else "invalid JSON")
            votes = f", agreement {result.agreement}" if result.agreement else ""
            print(f"[{n}/{len(records)}] {record['id']}: {status} ({result.latency_s:.1f} s"
                  f"{', cached' if result.cached else ''}{votes})", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="M1, M2, ... (B0), or S1, S2, S3 (a W1-09 strategy)")
    ap.add_argument("--label", required=True, help="short lower-case model name for the run id, e.g. qwen3-8b")
    ap.add_argument("--context", required=True, choices=CONTEXTS)
    ap.add_argument("--thinking", choices=("on", "off"), default="off")
    ap.add_argument("--base-url", help="another OpenAI-compatible server, e.g. http://localhost:11434/v1")
    ap.add_argument("--model", help="model name on that server")
    ap.add_argument("--api-key", help="key for that server (Ollama accepts any, e.g. 'ollama')")
    ap.add_argument("--timeout", type=float, help="seconds per request (a CPU model needs ~300; .env default 30)")
    ap.add_argument("--context-window", type=int,
                    help="the server's context window in its own tokens: must equal the model's num_ctx "
                         "(8192 only for the llama3.2-3b-8k Modelfile). The truncation check then looks for a "
                         "cut at that size instead of 2,048 / 4,096, so a value above the real window hides cuts")
    ap.add_argument("--ids", nargs="+", help="only these question ids (a check run; use --out-dir)")
    ap.add_argument("--questions", choices=tuple(QUESTION_SETS), default="gold",
                    help="gold = gold set + refusal tests; dev = the 15 development questions (prompt tuning)")
    ap.add_argument("--pack-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out-dir", type=Path, help=f"default {RUNS_DIR} (dev: {DEV_RUNS_DIR})")
    args = ap.parse_args(argv)
    try:
        from eval.strategies import PROMPT_VERSIONS, strategy_for, system_prompt
        strategy = strategy_for(args.run)
        if strategy != "b0" and args.context != "rag":
            raise ValueError(f"{args.run} is a RAG strategy: use --context rag")
        out_dir = args.out_dir or (DEV_RUNS_DIR if args.questions == "dev" else RUNS_DIR)
        if args.questions == "dev" and out_dir.resolve() == RUNS_DIR.resolve():
            raise ValueError("a development-set run must not be written next to the gold runs")
        version = PROMPT_VERSION if strategy == "b0" else PROMPT_VERSIONS[strategy]
        run_id = run_id_for(args.run, args.label, args.context, args.thinking == "on", version)
        system = prompt_file(args.context).read_text(encoding="utf-8") if strategy == "b0" else \
            system_prompt(strategy)
        records = load_records(args.pack_dir, args.questions)
        if args.ids:
            unknown = set(args.ids) - {r["id"] for r in records}
            if unknown:
                raise ValueError(f"unknown question ids {sorted(unknown)}")
            records = [r for r in records if r["id"] in set(args.ids)]
            if out_dir.resolve() in (RUNS_DIR.resolve(), DEV_RUNS_DIR.resolve()):
                raise ValueError("a run limited by --ids must not overwrite the full runs file: pass --out-dir")
        if args.context_window is not None and args.context_window <= 0:
            raise ValueError("--context-window must be a positive number of tokens")
        settings = settings_for(args)
        meta = run_meta(run_id, settings, args.context, args.pack_dir, system, strategy=strategy,
                        questions=args.questions)
    except ValidationError as err:
        print(f"ERROR: settings: {err}", file=sys.stderr)
        return 2
    except FileNotFoundError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    started = datetime.now(UTC).isoformat(timespec="seconds")
    count = qwen_counter() if strategy == "b0" else None
    windows = (args.context_window,) if args.context_window else COMMON_WINDOWS

    async def go() -> list[RunRecord]:
        async with LLMClient(settings) as llm:
            results = await run(llm, run_id, records, args.context, system, count, windows=windows,
                                strategy=strategy)
            u = llm.usage
            print(f"{run_id}: {len(results)} answers, {sum(r.json_valid for r in results)} valid, "
                  f"{sum(r.error is not None for r in results)} errors; {u.calls} requests, {u.cache_hits} cached, "
                  f"{u.http_retries} retries, {u.total_tokens} tokens")
            return results

    try:
        results = asyncio.run(go())
    except FatalRunError as err:
        print(f"ERROR: stopped, nothing written: {err}", file=sys.stderr)
        return 2
    if all(r.error is not None for r in results):
        print("ERROR: every answer failed; nothing written (the previous runs file, if any, is kept)", file=sys.stderr)
        return 2
    meta = run_meta(run_id, settings, args.context, args.pack_dir, system, count is not None, started,
                    args.context_window, strategy, args.questions)
    print(f"written: {write_run(results, meta, out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
