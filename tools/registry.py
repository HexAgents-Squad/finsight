"""The four tools the agent can call (week0 §2.3), their JSON schemas, and one safe dispatcher.

    SCHEMAS                     [{name, description, parameters}] to give the model (OpenAI function format)
    call_tool(name, arguments)  run a tool from the model's JSON arguments; always returns a result, never raises
    warm_up()                   load the XBRL concept map and the search index before the first question

call_tool validates the arguments against the tool's own model before calling it, so an unknown or misspelt
argument is an invalid_input result (not a Python TypeError), and test-only parameters such as `store` or
`retriever` can never be set by the model.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError

from tools import calculator, compare_periods, financial_facts, search_filings
from tools.base import INVALID_INPUT, ToolArgs, error_result, tool_boundary, validation_message

ToolFn = Callable[..., Any]

TOOLS: dict[str, tuple[ToolFn, type[ToolArgs], dict[str, Any]]] = {
    "calculate": (calculator.calculate, calculator.CalculateArgs, calculator.SCHEMA),
    "get_financial_fact": (financial_facts.get_financial_fact, financial_facts.FactArgs, financial_facts.SCHEMA),
    "compare_periods": (compare_periods.compare_periods, compare_periods.CompareArgs, compare_periods.SCHEMA),
    "search_filings": (search_filings.search_filings, search_filings.SearchArgs, search_filings.SCHEMA),
}
SCHEMAS: list[dict[str, Any]] = [schema for _, _, schema in TOOLS.values()]


def call_tool(name: str, arguments: Mapping[str, Any] | None) -> Any:
    """The tool's result for the model's arguments, or {"error": {...}}; never raises."""
    if not isinstance(name, str) or name not in TOOLS:
        return error_result(INVALID_INPUT, f"unknown tool {name!r}; use one of {sorted(TOOLS)}")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        return error_result(INVALID_INPUT, f"arguments must be a JSON object, not {type(arguments).__name__}")
    fn, args_model, _ = TOOLS[name]
    try:
        args = args_model.model_validate(dict(arguments))
    except ValidationError as err:
        return error_result(INVALID_INPUT, validation_message(err))
    return fn(**args.model_dump())


@tool_boundary
def warm_up() -> dict[str, Any]:
    """Build the default fact store and search index now, so the first question does not pay for loading the
    embedding model (~30-50 s on a CPU). Returns {"ok": true} or the error result of whatever is missing."""
    financial_facts.default_store()
    search_filings.default_retriever()
    return {"ok": True}
