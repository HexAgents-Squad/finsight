"""Smoke test for core/llm.py against the endpoint in .env: one chat call, one embedding batch and one
JSON call, with tokens and seconds for each. Run: python scripts/smoke_llm.py   (exit 0 = all passed)
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field, ValidationError  # noqa: E402

from core.exceptions import FinSightError  # noqa: E402
from core.llm import LLMClient  # noqa: E402
from core.settings import get_settings  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class FilingExtractTest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    year: int = Field(..., description="Reporting fiscal year")
    summary: str = Field(..., description="One-sentence financial summary")


def step(llm: LLMClient, label: str, started: float, tokens_before: int) -> None:
    print(f"   {label}: {time.perf_counter() - started:.1f} s, {llm.usage.total_tokens - tokens_before} tokens")


async def main() -> int:
    # Windows consoles default to cp1252; never crash on a character the console cannot show
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    print("FinSight WP7 smoke test")
    try:
        settings = get_settings()
    except ValidationError as err:
        print(f"FAILED: settings are incomplete. Copy .env.example to .env and fill it in.\n{err}")
        return 1
    print(f"   LLM endpoint: {settings.LLM_BASE_URL} ({settings.LLM_MODEL})")
    embed_model = settings.EMBED_MODEL if settings.EMBED_BACKEND == "openai" else settings.EMBED_LOCAL_MODEL_NAME
    print(f"   Embeddings:   {settings.EMBED_BACKEND} ({embed_model})")

    async with LLMClient(settings) as llm:
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
