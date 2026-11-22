"""The four tools the agent can call (week0 §2.3), their JSON schemas, and one safe dispatcher.

    SCHEMAS                       [{name, description, parameters}] to give the model (OpenAI function format)
    call_tool(name, arguments)    run a tool from the model's arguments; always returns a result, never raises
    call_tool_async(...)          the same from async code (runs in a worker thread, see below)
    call_tool(..., statements_first=False)   search_filings without the statements first (the A4 ablation, W3-01);
                                  a run setting, never one of the model's arguments
    warm_up()                     load the XBRL store, the search index and the embedding model at start-up

call_tool validates the arguments against the tool's own model before calling it, so an unknown or misspelt
argument is an invalid_input result (not a Python TypeError), and test-only parameters such as `store` or
`retriever` can never be set by the model. Arguments may be a mapping or the JSON string an OpenAI-style tool call
carries (`tool_calls[].function.arguments`); broken JSON is invalid_input, so a small model can retry.

The tools are synchronous and the search embedder runs its own event loop, which cannot run inside another one:
an async agent (core/llm.py is async) must use call_tool_async, which runs the call in a worker thread.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError

from tools import calculator, compare_periods, financial_facts, search_filings
from tools.base import INVALID_INPUT, ToolArgs, error_result, tool_boundary, validation_message

ToolFn = Callable[..., Any]
WARM_UP_QUERY = "total revenue"         # any query: the point is to load the embedding model once

TOOLS: dict[str, tuple[ToolFn, type[ToolArgs], dict[str, Any]]] = {
    "calculate": (calculator.calculate, calculator.CalculateArgs, calculator.SCHEMA),
    "get_financial_fact": (financial_facts.get_financial_fact, financial_facts.FactArgs, financial_facts.SCHEMA),
    "compare_periods": (compare_periods.compare_periods, compare_periods.CompareArgs, compare_periods.SCHEMA),
    "search_filings": (search_filings.search_filings, search_filings.SearchArgs, search_filings.SCHEMA),
}
SCHEMAS: list[dict[str, Any]] = [schema for _, _, schema in TOOLS.values()]


def _parse_arguments(arguments: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(arguments as a dict, None), or (None, an error result): a mapping or a JSON object string is accepted."""
    if arguments is None:
        return {}, None
    if isinstance(arguments, str | bytes):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            return None, error_result(INVALID_INPUT, f"arguments are not valid JSON: {err}")
    if not isinstance(arguments, Mapping):
        return None, error_result(INVALID_INPUT, f"arguments must be a JSON object, not {type(arguments).__name__}")
    return dict(arguments), None


def call_tool(name: str, arguments: Mapping[str, Any] | str | None, *, statements_first: bool = True) -> Any:
    """The tool's result for the model's arguments, or {"error": {...}}; never raises. `statements_first` reaches
    search_filings only (the A4 ablation)."""
    if not isinstance(name, str) or name not in TOOLS:
        return error_result(INVALID_INPUT, f"unknown tool {name!r}; use one of {sorted(TOOLS)}")
    parsed, error = _parse_arguments(arguments)
    if error is not None:
        return error
    fn, args_model, _ = TOOLS[name]
    try:
        args = args_model.model_validate(parsed)
    except ValidationError as err:
        return error_result(INVALID_INPUT, validation_message(err))
    options = {"statements_first": False} if name == "search_filings" and not statements_first else {}
    return fn(**args.model_dump(), **options)


async def call_tool_async(name: str, arguments: Mapping[str, Any] | str | None, *,
                          statements_first: bool = True) -> Any:
    """call_tool for async callers: runs in a worker thread, so the search embedder's own event loop never runs
    inside the caller's."""
    return await asyncio.to_thread(call_tool, name, arguments, statements_first=statements_first)


@tool_boundary
def warm_up() -> dict[str, Any]:
    """Build the fact store and the search index and run one search, so the embedding model is loaded now and
    the first question does not pay for it (~45 s of a ~60 s cold start on a CPU, close to the 60 s query limit).
    Returns {"ok": true} or the error result of whatever is missing."""
    financial_facts.default_store()
    result = search_filings.search_filings(WARM_UP_QUERY, k=1)
    if isinstance(result, dict) and "error" in result:
        return result
    return {"ok": True}
