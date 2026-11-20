"""Ask FinSight one question from the command line, through the whole agent graph (W2-07).

    python scripts/ask.py "What was Apple's total revenue in fiscal 2024?"
    python scripts/ask.py "And in fiscal 2023?" --history "What was Apple's total revenue in fiscal 2024?"
    python scripts/ask.py "..." --json          # the full record: contract, tool calls, tokens, seconds
    python scripts/ask.py "..." --trace         # also send the run to Langfuse (LANGFUSE_ENABLED=true)

The real nodes (agent.nodes.real_nodes): the planner and the composer call the shared endpoint (one or two
requests each, through the reply cache), the executor's tools read data/ (the index and the XBRL facts), and the
critic checks the draft. Needs a complete .env. Prints the answer, its verdict, citations and what the run cost.
Exit codes: 0 = an answer (verified, refused or not), 2 = the setup is incomplete (.env). Without data/ the tools
return typed errors and the answer says it could not be verified.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def ask(question: str, history: list[str], nodes: Any, limits: Any = None, trace: Any = None,
              settings: Any = None) -> Any:
    """The run. Without `nodes`, the real ones on an LLM client opened and closed here (core/llm.py: a client left
    open fails later with "Event loop is closed" on Windows)."""
    from agent.graph import build_graph, run
    if nodes is not None:
        return await run(question, history, graph=build_graph(nodes), limits=limits, trace=trace)
    from agent.nodes import real_nodes
    from core.llm import LLMClient
    async with LLMClient(settings) as llm:
        return await run(question, history, graph=build_graph(real_nodes(llm)), limits=limits, trace=trace)


def main(argv: list[str] | None = None, *, nodes: Any = None, limits: Any = None) -> int:
    from eval.validate_gold import safe_console
    safe_console()                                  # a Windows console cannot print "·" or "’": replace, not crash
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("question")
    parser.add_argument("--history", action="append", default=[], metavar="QUESTION",
                        help="an earlier question of the conversation, oldest first (repeat for more)")
    parser.add_argument("--json", action="store_true", help="print the full record as JSON")
    parser.add_argument("--trace", action="store_true", help="send the run to Langfuse")
    args = parser.parse_args(argv)
    if not args.question.strip():
        parser.error("the question is empty")
    history = [q for q in args.history if q.strip()]
    from agent.graph import Limits
    from agent.transcript import markdown, run_record
    info, settings = None, None
    if nodes is None or args.trace:
        try:                                        # setup only: an error inside the run is a bug and propagates
            from core.settings import get_settings
            settings = get_settings()               # an incomplete .env raises pydantic's ValidationError here
        except (ValueError, OSError) as err:        # ValidationError is a ValueError
            print(f"setup incomplete (.env): {type(err).__name__}: {str(err).splitlines()[0]}", file=sys.stderr)
            return 2
    if limits is None:                              # the .env limits, or the defaults with given nodes (a test)
        limits = Limits.from_settings(settings) if settings is not None else Limits()
    if args.trace:
        from core import tracing
        if not settings.LANGFUSE_ENABLED:
            print("--trace needs LANGFUSE_ENABLED=true and the Langfuse keys in .env.", file=sys.stderr)
            return 2
        info = tracing.TraceInfo(question_id="ASK", run_id="ask-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    try:
        result = asyncio.run(ask(args.question, history, nodes, limits, info, settings))
    finally:
        if info is not None:
            tracing.flush()
    record = run_record(result, args.question)      # the history the run saw (its state), not the arguments
    if args.json:
        print(json.dumps(record, indent=1, ensure_ascii=False))
    else:
        print(markdown(record))
    if info is not None:
        print(f"Langfuse: filter tags = run:{info.run_id} (visible after a few minutes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
