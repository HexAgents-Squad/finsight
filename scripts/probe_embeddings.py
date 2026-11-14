"""W1-16: probe the embedding endpoint in .env and write the evidence for ADR-021 (which model indexes the corpus).

Run (about 40 requests and a minute; run outside the key freeze, the endpoint is shared by the course):
    python scripts/probe_embeddings.py                         # -> eval/results/embedding_probe.{json,md}
    python scripts/probe_embeddings.py --skip parity --out-dir /tmp/p   # a subset never overwrites the report

The settings must select the hosted endpoint (EMBED_BACKEND=openai, EMBED_BASE_URL, EMBED_API_KEY, EMBED_MODEL,
EMBED_CA_CERT); an environment variable overrides .env for one run.

Probes: basic (dimensions, vector length, tokens reported), batch limit (16 vs 17 texts), token limit (512 vs 513
tokens with [CLS] and [SEP], built with the pinned tokenizer), determinism (the same text twice), throughput on
real C3 chunks cut to 512 tokens (texts/s by batch size, and the time to embed every C3 chunk), concurrency (two
batches at once), and parity with BAAI/bge-small-en-v1.5 run locally (cosine of the two vectors per text; the
local model downloads once).

Requests use core.llm's configured embeddings client (EMBED_CA_CERT, no retries), one at a time at least
MIN_GAP_S apart except the deliberate concurrent pair, so a refusal or a capacity error is measured as it
happens. The key never appears and a remote host is shown as <endpoint>. A probe that crashes is recorded and
the others still run.
Exit codes: 0 = written, 1 = settings missing, a subset aimed at the committed report, or no request answered.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import openai  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from core.llm import LLMClient  # noqa: E402
from core.settings import HTTPS, Settings, get_settings  # noqa: E402
from scripts.smoke_llm import RedactHost, redact  # noqa: E402

PROBE_VERSION = "e-v1"
OUT_DIR = ROOT / "eval" / "results"
OUT_NAME = "embedding_probe"
C3_CHUNKS = ROOT / "data" / "chunks" / "chunks_C3.jsonl"
MIN_GAP_S = 0.5                 # between request starts: polite on a server the whole course shares
MAX_BATCH = 16                  # the endpoint's documented batch limit (probed: 16 and 17)
MAX_TOKENS = 512                # the endpoint's documented token limit (probed: 512 and 513)
THROUGHPUT_BATCHES = (1, 4, 16)
THROUGHPUT_TEXTS = 64           # chunks per batch size (4 x 16, 16 x 4, 64 x 1)
PARITY_TEXTS = ("Apple's total net sales were $391,035 million in fiscal 2024.",
                "Risk factors: supply-chain concentration in China and Taiwan.",
                "What was Coca-Cola's operating margin in fiscal 2024?",
                "Net interest income decreased due to higher deposit costs.")
LOCAL_PARITY_MODEL = "BAAI/bge-small-en-v1.5"
SAME_VECTOR_COSINE = 0.999      # parity: the hosted and the local model give the same vector


class Session:
    """Sends embedding requests through the configured client, MIN_GAP_S apart, and records each one."""

    def __init__(self, client: Any, settings: Settings) -> None:
        self.client, self.settings, self.last_start = client, settings, 0.0

    async def send(self, label: str, texts: list[str], spaced: bool = True) -> tuple[dict[str, Any], Any]:
        if spaced:
            wait = self.last_start + MIN_GAP_S - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
        self.last_start = time.monotonic()
        start = time.perf_counter()
        row: dict[str, Any] = {"label": label, "texts": len(texts)}
        try:
            response = await self.client.embeddings.create(model=self.settings.EMBED_MODEL, input=texts)
        except openai.APIStatusError as err:
            row.update(ok=False, status=err.status_code, error_type=type(err).__name__, error=self.clean(str(err)),
                       seconds=round(time.perf_counter() - start, 3))
            return row, None
        except openai.APIError as err:
            row.update(ok=False, status=None, error_type=type(err).__name__, error=self.clean(str(err)),
                       seconds=round(time.perf_counter() - start, 3))
            return row, None
        vectors = [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        usage = getattr(response, "usage", None)
        row.update(ok=True, status=200, seconds=round(time.perf_counter() - start, 3), vectors=len(vectors),
                   dimensions=len(vectors[0]) if vectors else None,
                   prompt_tokens=getattr(usage, "prompt_tokens", None))
        return row, vectors

    def clean(self, text: str) -> str:
        key = self.settings.EMBED_API_KEY or self.settings.LLM_API_KEY
        url = self.settings.EMBED_BASE_URL or self.settings.LLM_BASE_URL
        text = redact(text, url)
        secret = key.get_secret_value()
        return (text.replace(secret, "<key>") if secret else text)[:300]


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm(a) * norm(b))


def text_of_tokens(n: int) -> str:
    """A text that takes exactly `n` embedding tokens with [CLS] and [SEP] (the pinned tokenizer)."""
    from core.tokens import embed_tokenizer
    tok, text = embed_tokenizer(), ""
    base = "revenue increased "                                 # 2 tokens per repeat
    text = base * ((n - 2) // 2)
    if (n - 2) % 2:
        text += "growth"
    count = len(tok.encode(text).ids)
    if count != n:
        raise ValueError(f"built a text of {count} tokens, wanted {n}")
    return text


def c3_sample(n: int, path: Path = C3_CHUNKS) -> list[str]:
    """`n` real C3 chunk texts spread over the file, cut to MAX_TOKENS as the client cuts them."""
    from core.tokens import cut_to_embedding_tokens
    lines = path.read_text(encoding="utf-8").splitlines()
    step = max(1, len(lines) // n)
    return [cut_to_embedding_tokens(json.loads(line)["text"], MAX_TOKENS)[0] for line in lines[::step][:n]]


# --------------------------------------------------------------------------------------------- probes
async def probe_basic(s: Session) -> dict[str, Any]:
    row, vectors = await s.send("2 texts", ["Example document chunk", "Example search query"])
    findings = {"ok": row["ok"]}
    if vectors:
        findings.update(dimensions=len(vectors[0]), norms=[round(norm(v), 6) for v in vectors],
                        prompt_tokens=row.get("prompt_tokens"))
    return {"requests": [row], "findings": findings}


async def probe_batch_limit(s: Session) -> dict[str, Any]:
    rows = [(await s.send(f"{n} texts", [f"text {i}" for i in range(n)]))[0] for n in (MAX_BATCH, MAX_BATCH + 1)]
    return {"requests": rows, "findings": {f"{r['texts']}_texts": {"accepted": r["ok"], "status": r["status"],
                                                                     "error": r.get("error")} for r in rows}}


async def probe_token_limit(s: Session) -> dict[str, Any]:
    rows = [(await s.send(f"{n} tokens", [text_of_tokens(n)]))[0] for n in (MAX_TOKENS, MAX_TOKENS + 1)]
    return {"requests": rows, "findings": {f"{n}_tokens": {"accepted": r["ok"], "status": r["status"],
                                                            "error": r.get("error")}
                                           for n, r in zip((MAX_TOKENS, MAX_TOKENS + 1), rows, strict=True)}}


async def probe_determinism(s: Session) -> dict[str, Any]:
    text = ["Apple total net sales fiscal 2024"]
    (a, va), (b, vb) = await s.send("same text #1", text), await s.send("same text #2", text)
    same = va is not None and vb is not None and va[0] == vb[0]
    return {"requests": [a, b], "findings": {"identical": same,
                                             "cosine": round(cosine(va[0], vb[0]), 9) if va and vb else None}}


async def probe_throughput(s: Session) -> dict[str, Any]:
    texts = c3_sample(THROUGHPUT_TEXTS)
    rows, findings = [], {}
    for size in THROUGHPUT_BATCHES:
        got = [(await s.send(f"batch {size} #{i // size + 1}", texts[i:i + size]))[0]
               for i in range(0, len(texts), size)]
        rows += got
        ok = [r for r in got if r["ok"]]
        seconds = sum(r["seconds"] for r in ok)
        findings[f"batch_{size}"] = {"texts": sum(r["texts"] for r in ok), "seconds": round(seconds, 2),
                                     "texts_per_s": round(sum(r["texts"] for r in ok) / seconds, 1) if seconds else None,
                                     "failed": len(got) - len(ok)}
    best = findings.get(f"batch_{MAX_BATCH}", {})
    chunks = sum(1 for _ in C3_CHUNKS.open(encoding="utf-8")) if C3_CHUNKS.exists() else None
    if best.get("texts_per_s") and chunks:
        # request time only: the spacing between requests (MIN_GAP_S) is this probe's courtesy, not the server's
        findings["c3_chunks"] = chunks
        findings["c3_minutes_at_batch_16"] = round(chunks / best["texts_per_s"] / 60, 1)
    findings["sample_tokens_mean"] = round(sum(r.get("prompt_tokens") or 0 for r in rows if r["ok"])
                                           / max(1, sum(r["texts"] for r in rows if r["ok"])), 1)
    return {"requests": rows, "findings": findings}


async def probe_concurrency(s: Session) -> dict[str, Any]:
    texts = [f"concurrent text {i}" for i in range(MAX_BATCH)]
    await asyncio.sleep(MIN_GAP_S)
    pair = await asyncio.gather(s.send("concurrent A", texts, spaced=False), s.send("concurrent B", texts, spaced=False))
    rows = [r for r, _ in pair]
    return {"requests": rows, "findings": {"statuses": [r["status"] for r in rows], "both_ok": all(r["ok"] for r in rows)}}


async def probe_parity(s: Session) -> dict[str, Any]:
    row, remote = await s.send("parity texts", list(PARITY_TEXTS))
    if remote is None:
        return {"requests": [row], "findings": {"skipped": "the hosted request failed"}}
    try:
        from sentence_transformers import SentenceTransformer
        model = await asyncio.to_thread(SentenceTransformer, LOCAL_PARITY_MODEL)
        local = (await asyncio.to_thread(model.encode, list(PARITY_TEXTS), normalize_embeddings=True)).tolist()
    except Exception as err:                    # noqa: BLE001 - no local model: say so, keep the other probes
        return {"requests": [row], "findings": {"skipped": f"local model unavailable ({type(err).__name__})"}}
    cosines = [round(cosine(a, b), 6) for a, b in zip(remote, local, strict=True)]
    return {"requests": [row], "findings": {"local_model": LOCAL_PARITY_MODEL, "cosines": cosines,
                                            "same_model": min(cosines) >= SAME_VECTOR_COSINE}}


PROBES = {"basic": probe_basic, "batch_limit": probe_batch_limit, "token_limit": probe_token_limit,
          "determinism": probe_determinism, "throughput": probe_throughput, "concurrency": probe_concurrency,
          "parity": probe_parity}


# --------------------------------------------------------------------------------------------- report
def git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
                                timeout=10).stdout.strip() or None
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True, check=True, timeout=10).stdout
        return commit, any(line.strip() and OUT_NAME not in line for line in status.splitlines())
    except (OSError, subprocess.SubprocessError):
        return None, None


def markdown(report: dict[str, Any]) -> str:
    meta = report["meta"]
    out = [f"# Embedding endpoint probe ({meta['probe_version']}, {meta['date']})", "",
           f"Model {meta['model']} on the {meta['endpoint']} embedding endpoint; generated by "
           f"`scripts/probe_embeddings.py` (code {meta['git_commit'] or 'unknown'}). Decision: ADR-021.", ""]
    for name, probe in report["probes"].items():
        out += [f"## {name}", "", "| Request | Texts | OK | Status | Seconds | Tokens | Error |", "|---|---|---|---|---|---|---|"]
        for r in probe["requests"]:
            out.append(f"| {r['label']} | {r['texts']} | {'yes' if r['ok'] else 'no'} | {r.get('status') or '-'} | "
                       f"{r['seconds']} | {r.get('prompt_tokens') or '-'} | "
                       f"{(r.get('error') or '').replace('|', '/')[:120]} |")
        out += ["", "```json", json.dumps(probe["findings"], indent=1), "```", ""]
    return "\n".join(out)


def write_report(report: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, text in ((".json", json.dumps(report, indent=1) + "\n"), (".md", markdown(report))):
        path, tmp = out_dir / f"{OUT_NAME}{suffix}", out_dir / f"{OUT_NAME}{suffix}.part"
        try:
            tmp.write_text(text, encoding="utf-8", newline="\n")
            tmp.replace(path)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        paths.append(path)
    return paths


async def run(client: Any, settings: Settings, names: list[str]) -> dict[str, Any]:
    session, probes = Session(client, settings), {}
    for name in names:
        print(f"[{name}] ...", flush=True)
        try:
            result = await PROBES[name](session)
        except Exception as err:                # noqa: BLE001 - keep the other probes' evidence
            probes[name] = {"requests": [], "findings": {"crashed": session.clean(f"{type(err).__name__}: {err}")}}
            print(f"[{name}] crashed: {probes[name]['findings']['crashed']}", flush=True)
            continue
        probes[name] = result
        print(f"[{name}] {len(result['requests'])} requests, {sum(not r['ok'] for r in result['requests'])} refused "
              f"or failed", flush=True)
    commit, changed = git_state()
    url = settings.EMBED_BASE_URL or settings.LLM_BASE_URL
    meta = {"probe_version": PROBE_VERSION, "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": settings.EMBED_MODEL, "endpoint": "hosted" if url.lower().startswith(HTTPS) else "local",
            "ca": settings.EMBED_CA_CERT.name if settings.EMBED_CA_CERT else None, "min_gap_s": MIN_GAP_S,
            "git_commit": commit, "code_changed_since_commit": changed}
    return {"meta": meta, "probes": probes}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(PROBES))
    ap.add_argument("--skip", nargs="+", choices=list(PROBES), default=[])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        settings = get_settings()
    except ValidationError as err:
        fields = ", ".join(".".join(str(x) for x in e["loc"]) for e in err.errors())
        print(f"ERROR: settings are incomplete or invalid ({fields}); check .env against .env.example", file=sys.stderr)
        return 1
    if settings.EMBED_BACKEND != "openai":
        print("ERROR: EMBED_BACKEND is not 'openai': the probe needs the hosted endpoint (an environment variable "
              "overrides .env for one run)", file=sys.stderr)
        return 1
    names = [n for n in (args.only or list(PROBES)) if n not in args.skip]
    if names != list(PROBES) and args.out_dir.resolve() == OUT_DIR.resolve():
        print("ERROR: a subset of the probes must not overwrite the full report: pass --out-dir", file=sys.stderr)
        return 1
    for handler in logging.getLogger().handlers:
        handler.addFilter(RedactHost(settings.EMBED_BASE_URL or settings.LLM_BASE_URL))

    async def go() -> dict[str, Any]:
        async with LLMClient(settings) as llm:            # its embeddings client: EMBED_CA_CERT, no retries
            return await run(llm.embed_client, settings, names)

    report = asyncio.run(go())
    if not any(r["ok"] for p in report["probes"].values() for r in p["requests"]):
        print("ERROR: the endpoint answered no request (network, TLS or key); nothing written", file=sys.stderr)
        return 1
    for path in write_report(report, args.out_dir):
        print(f"written: {path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sys.exit(main())
