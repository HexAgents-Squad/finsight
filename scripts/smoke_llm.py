"""Smoke test for core/llm.py against the endpoint in .env: one chat call, one embedding batch and one
JSON call, with tokens and seconds for each. Run: python scripts/smoke_llm.py   (exit 0 = all passed)

Works for the local Ollama stand-in and the hosted endpoint (HTTPS verified against LLM_CA_CERT). The output
never shows the API key, and shows a remote endpoint's host as <endpoint>, so it can be pasted into the team
chat as the smoke-test evidence (W1-10) without sharing the group's address.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field, ValidationError  # noqa: E402

from core.exceptions import FinSightError  # noqa: E402
from core.llm import LLMClient  # noqa: E402
from core.settings import HTTPS, Settings, get_settings  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class FilingExtractTest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    year: int = Field(..., description="Reporting fiscal year")
    summary: str = Field(..., description="One-sentence financial summary")


def shown_url(url: str) -> str:
    """The URL with a remote host replaced by <endpoint> (local hosts are shown as they are)."""
    parts = urlsplit(url)
    if (parts.hostname or "") in LOCAL_HOSTS:
        return url
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://<endpoint>{port}{parts.path}"


def tls_line(settings: Settings) -> str:
    if settings.LLM_CA_CERT is not None:
        return f"verified against {settings.LLM_CA_CERT.name} (LLM_CA_CERT)"
    if settings.LLM_BASE_URL.lower().startswith(HTTPS):
        return "verified against the system CAs"
    return "none (plain HTTP; fine for a local server only)"


def step(llm: LLMClient, label: str, started: float, tokens_before: int) -> None:
    print(f"   {label}: {time.perf_counter() - started:.1f} s, {llm.usage.total_tokens - tokens_before} tokens")


async def main() -> int:
    # Windows consoles default to cp1252; never crash on a character the console cannot show
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    print("FinSight LLM smoke test")
    try:
        settings = get_settings()
    except ValidationError as err:
        print(f"FAILED: settings are incomplete. Copy .env.example to .env and fill it in.\n{err}")
        return 1
    print(f"   LLM endpoint: {shown_url(settings.LLM_BASE_URL)} ({settings.LLM_MODEL})")
    print(f"   TLS:          {tls_line(settings)}")
    thinking = {None: "switch not sent", True: "on", False: "off"}[settings.LLM_ENABLE_THINKING]
    print(f"   Thinking:     {thinking}")
    embed_model = settings.EMBED_MODEL if settings.EMBED_BACKEND == "openai" else settings.EMBED_LOCAL_MODEL_NAME
    print(f"   Embeddings:   {settings.EMBED_BACKEND} ({embed_model})")

    try:
        llm = LLMClient(settings)
    except FinSightError as err:                    # e.g. LLM_CA_CERT is not a PEM certificate
        print(f"FAILED [{err.error_type}]: {err.message}")
        return 1
    async with llm:
        return await run_checks(llm)


async def run_checks(llm: LLMClient) -> int:
    try:
        print("\n[1/3] chat")
        t, tok = time.perf_counter(), llm.usage.total_tokens
        reply = await llm.chat([{"role": "user", "content": "Reply with the single word OPERATIONAL."}])
        print(f"   reply: {reply}")
        step(llm, "chat", t, tok)

        print("\n[2/3] embed")
        vectors = await llm.embed(["Apple Inc. fiscal 2024 10-K net sales.", "Microsoft fiscal 2024 revenue."])
        print(f"   {len(vectors)} vectors, {len(vectors[0])} dimensions, {llm.usage.embed_seconds:.1f} s")

        print("\n[3/3] chat_json")
        t, tok = time.perf_counter(), llm.usage.total_tokens
        result = await llm.chat_json(
            [{"role": "user", "content": "Extract details: Microsoft Corporation filed its FY2023 annual report."}],
            FilingExtractTest,
        )
        print(f"   {result.model_dump_json()}")
        step(llm, "chat_json", t, tok)
        print(f"   JSON retries: {llm.usage.json_retries}")
    except FinSightError as err:
        print(f"\nFAILED [{err.error_type}]: {err.message}")
        return 1

    u = llm.usage
    print(f"\nOK: {u.calls} chat calls, {u.total_tokens} tokens "
          f"({u.prompt_tokens} prompt + {u.completion_tokens} completion), {u.seconds:.1f} s in chat calls; "
          f"{u.embed_calls} embedding request(s), {u.embed_seconds:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
