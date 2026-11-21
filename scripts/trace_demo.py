"""Send one question through the agent with tracing on, so its full trace shows in Langfuse (W2-08 acceptance).

    python scripts/trace_demo.py "What was Apple's total revenue in fiscal 2024?" [--question-id DEMO-1] [--run-id ID]

The graph uses the real nodes: the planner and the composer (one or two requests each to the shared endpoint), the
executor and its tools (needs data/: the index and the XBRL facts; a search embeds the query through the embedding
endpoint) and the critic. Needs
LANGFUSE_ENABLED=true and the Langfuse keys in .env, and the `langfuse` package. Prints the run id: in Langfuse open
Tracing and filter `tags = run:<id>` (a few minutes after the run), or run scripts/trace_dashboard.py --run <id>.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def demo_nodes(llm: Any, call_tool: Any = None) -> Any:
    """The real nodes (agent.nodes.real_nodes): planner, executor, composer and critic."""
    from agent.nodes import real_nodes
    return real_nodes(llm, call_tool)


async def traced_run(question: str, info: Any, nodes: Any, tracer: Any = None, limits: Any = None) -> Any:
    """The traced run; `limits` default to the .env settings (agent.graph.run), tests pass their own."""
    from agent.graph import build_graph, run
    return await run(question, graph=build_graph(nodes), trace=info, tracer=tracer, limits=limits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("question")
    parser.add_argument("--question-id", default="DEMO-1", help="becomes the tag q:<id>")
    parser.add_argument("--run-id", default=None, help="becomes the tag run:<id> (default: demo-<UTC time>)")
    args = parser.parse_args(argv)
    from core.pii import install_log_masking
    install_log_masking()                           # W2-13: a question's e-mail or phone number never reaches a log
    from core import tracing
    from core.llm import LLMClient
    from core.settings import get_settings
    settings = get_settings()
    if not settings.LANGFUSE_ENABLED:
        print("LANGFUSE_ENABLED is false in .env: the run would not be traced (see .env.example).", file=sys.stderr)
        return 2
    run_id = args.run_id or "demo-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    info = tracing.TraceInfo(question_id=args.question_id, run_id=run_id)
    async def traced_with_client() -> Any:          # the client closed in the same loop (core/llm.py)
        async with LLMClient(settings) as llm:
            return await traced_run(args.question, info, demo_nodes(llm))

    result = asyncio.run(traced_with_client())
    tracing.flush()
    state = result.state
    print(f"run {run_id}: verified={result.answer.verified} refused={result.answer.refused} "
          f"seconds={result.seconds:.1f} tokens={state.get('tokens') or 0} "
          f"tool calls={len(state.get('tool_results') or [])} visits={dict(state.get('visits') or {})}")
    print(f"Langfuse: Tracing, filter tags = run:{run_id} (visible after a few minutes); "
          f"then: python scripts/trace_dashboard.py --run {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
