"""The real nodes (W2-07): the graph end to end.

    real_nodes(llm=None, call_tool=None, injection_guard=True, calculator=True, critic=True, statements_first=True)
        -> Nodes(planner, executor, composer, critic)

- planner: agent.planner.make_planner(llm) (W2-02, one guided call; refusals decided here)
- executor: agent.executor.make_executor(call_tool) (W2-04; default tools.registry.call_tool_async, which needs
  data/: the chunk index and the XBRL facts; a search embeds the query through the embedding endpoint)
- composer: agent.composer.make_composer(llm) (W2-06, one guided call, two at most; W2-13: tampered passages left out)
- critic: agent.critic.make_critic() (W2-05, no LLM call; W2-13: injected instructions checked)
- Nodes.injection_guard: finish's own payload check on a draft the critic did not pass (W2-13)

`llm` defaults to core.llm.LLMClient(get_settings()), the shared endpoint with the reply cache, so it needs a
complete .env; tests pass a fake client and fake tools. agent.graph.build_graph() uses these nodes when none are
given; agent/stubs.py keeps the stubs the offline graph tests use. `injection_guard=False` turns the W2-13
defences off, only to measure them (eval/run_injection.py); everything else runs with them.

The ablations (W3-01, ADR-036; eval/run_agent.py --ablation) each turn one component off and leave the rest as A1:
`calculator=False` (A2: the executor never runs the plan's calculation), `critic=False` (A3: no critic node, every
draft leaves unverified), `statements_first=False` (A4: the default tools, with search_filings on plain hybrid
retrieval; ignored when a `call_tool` is given, which then decides for itself).
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.graph import Nodes


def warm_up() -> None:
    """The agent's one-time loads, so a server's first question does not pay for them (W3-07, measured on the
    development set v2: the first question's planner took ~0.2 s longer): the planner's and composer's prompts and
    reply schemas, the two config files (parsed once per version), the company names the entity reader, the
    composer and the critic use. No data/ and no request; the index and the facts are tools.registry.warm_up's."""
    from agent import composer, critic_semantic, planner
    from ingest.edgar import load_scope
    from ingest.xbrl import load_concept_map
    from retrieval.entities import extract
    planner.system_prompt()
    planner.plan_json_schema()
    composer.system_prompt()
    composer.reply_schema()
    load_concept_map()
    for company in load_scope().companies:
        composer.company(company.ticker)
        critic_semantic._ticker_words(company.ticker)
    extract("What was Apple's revenue in fiscal 2024?")


def real_nodes(llm: Any = None, call_tool: Any = None, *, injection_guard: bool = True, calculator: bool = True,
               critic: bool = True, statements_first: bool = True) -> Nodes:
    """The planner, executor, composer and critic, sharing one LLM client (the switches: the module docstring)."""
    from agent.composer import make_composer
    from agent.critic import make_critic
    from agent.executor import make_executor
    from agent.graph import Nodes
    from agent.planner import make_planner
    if llm is None:
        from core.llm import LLMClient
        from core.settings import get_settings
        llm = LLMClient(get_settings())
    if call_tool is None and not statements_first:
        call_tool = functools.partial(call_tool_async, statements_first=False)
    return Nodes(planner=make_planner(llm), executor=make_executor(call_tool, calculator=calculator),
                 composer=make_composer(llm, injection_guard=injection_guard),
                 critic=make_critic(injection_guard=injection_guard) if critic else None,
                 injection_guard=injection_guard)


async def call_tool_async(name: str, arguments: dict[str, Any], *, statements_first: bool = True) -> Any:
    """tools.registry.call_tool_async, imported on first use (the registry loads the tools)."""
    from tools.registry import call_tool_async as call
    return await call(name, arguments, statements_first=statements_first)
