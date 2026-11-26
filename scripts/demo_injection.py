"""The demo's blocked prompt injection (W3-14): one injection test through the real agent, from the reply cache.

    python scripts/demo_injection.py            # INJ01: "SYSTEM NOTE: Ignore all previous instructions ..." in Apple's MD&A
    python scripts/demo_injection.py INJ07      # any test in eval/injection_tests.jsonl

A sentence planted in a filing section only reaches the agent through a search, so the UI cannot show it; this does,
with the W2-13 harness (eval/run_injection.py): the planted section's top passage carries the sentence, the agent
answers the test's question with its defences on, and the script prints the planted sentence, the answer, whether
the sentence reached a model, whether the injection succeeded and the note the answer carries (spec §6.3).
Offline: chat replies come only from the reply cache (a reply not in it stops the run; nothing is sent to the
endpoint) and the query is embedded locally with BAAI/bge-small-en-v1.5, the endpoint's model (ADR-021). Needs
data/ (the index, the facts) and .env, like scripts/ask.py. The committed measurement of all 12 tests is
eval/results/injection_results.md and, through the API, eval/results/api_safety_results.md.
Exit codes: 0 = the injection was blocked, 1 = it succeeded (a regression), 2 = setup, or a reply not in the cache.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TEST = "INJ01"


class NotCached(RuntimeError):
    """A chat reply the reply cache does not hold (this script never asks the endpoint)."""


def cache_only(client: Any) -> Any:
    """`client` (core.llm.LLMClient) answering from its reply cache only (as scripts/trace_runs.py --cache-only)."""
    async def refuse(request: dict[str, Any], target_model: str, path: Path | None) -> Any:
        raise NotCached("a reply is not in the reply cache; this demo sends no request to the endpoint")
    client._fetch = refuse
    return client


def offline_settings() -> Any:
    """The .env settings with the embedding computed locally (the endpoint's model, no request)."""
    os.environ["EMBED_BACKEND"] = "sentence-transformers"
    os.environ.setdefault("EMBED_LOCAL_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    from core.settings import get_settings
    return get_settings()


async def run(test: Any, settings: Any) -> dict[str, Any]:
    from agent.graph import Limits
    from core.llm import LLMClient
    from eval.run_injection import run_test
    from tools.registry import call_tool_async
    async with LLMClient(settings) as client:
        return await run_test(test, cache_only(client), call_tool_async, True, Limits.from_settings(settings))


def show(test: Any, rec: dict[str, Any]) -> str:
    wrap = lambda s: textwrap.fill(s, 100, initial_indent="    ", subsequent_indent="    ")  # noqa: E731
    p = test.planted_in
    lines = [f"Injection test {test.id} ({test.injection_type})",
             f"Planted in {p.ticker} {p.form} FY{p.fiscal_year} {p.item}, at the start of the top passage:",
             wrap(test.planted_text), "", f"Question: {test.question}", "", "Answer (defences on):",
             wrap(rec["answer"]), ""]
    lines += [f"  note: {n}" for n in rec["notes"]]
    lines += ["", f"verdict {rec['verdict']} - planted passage retrieved: {rec['exposed']} - planted sentence reached a "
              f"model: {rec['planted_text_reached_model']} - injection succeeded: {rec['succeeded']}"]
    lines.append("BLOCKED" if not rec["succeeded"] else "THE INJECTION SUCCEEDED (a regression)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from eval.validate_gold import safe_console
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("test_id", nargs="?", default=DEFAULT_TEST, help=f"an id from eval/injection_tests.jsonl "
                                                                     f"(default {DEFAULT_TEST})")
    args = ap.parse_args(argv)
    from eval.run_injection import load_tests
    tests = {t.id: t for t in load_tests()}
    if args.test_id not in tests:
        print(f"unknown test {args.test_id}; one of {', '.join(sorted(tests))}", file=sys.stderr)
        return 2
    try:
        settings = offline_settings()
    except (ValueError, OSError) as err:
        print(f"setup incomplete (.env): {str(err).splitlines()[0]}", file=sys.stderr)
        return 2
    try:
        rec = asyncio.run(run(tests[args.test_id], settings))
    except NotCached as err:
        print(str(err), file=sys.stderr)
        return 2
    print(show(tests[args.test_id], rec))
    return 1 if rec["succeeded"] else 0


if __name__ == "__main__":
    sys.exit(main())
