"""The real nodes (W2-07): the graph end to end.

    real_nodes(llm=None, call_tool=None)  -> Nodes(planner, executor, composer, critic)

- planner: agent.planner.make_planner(llm) (W2-02, one guided call; refusals decided here)
- executor: agent.executor.make_executor(call_tool) (W2-04; default tools.registry.call_tool_async, which needs
  data/: the chunk index and the XBRL facts; a search embeds the query through the embedding endpoint)
- composer: agent.composer.make_composer(llm) (W2-06, one guided call, two at most)
- critic: agent.critic.make_critic() (W2-05, no LLM call)

`llm` defaults to core.llm.LLMClient(get_settings()), the shared endpoint with the reply cache, so it needs a
complete .env; tests pass a fake client and fake tools. agent.graph.build_graph() uses these nodes when none are
given; agent/stubs.py keeps the stubs the offline graph tests use.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.graph import Nodes


def real_nodes(llm: Any = None, call_tool: Any = None) -> Nodes:
    """The planner, executor, composer and critic, sharing one LLM client."""
    from agent.composer import make_composer
    from agent.critic import make_critic
    from agent.executor import make_executor
    from agent.graph import Nodes
    from agent.planner import make_planner
    if llm is None:
        from core.llm import LLMClient
        from core.settings import get_settings
        llm = LLMClient(get_settings())
    return Nodes(planner=make_planner(llm), executor=make_executor(call_tool), composer=make_composer(llm),
                 critic=make_critic())
