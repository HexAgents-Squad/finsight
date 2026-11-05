"""What every tool shares: typed errors, the never-raise boundary and JSON schemas built from the argument models.

A tool is called by the agent with JSON arguments. Each tool validates them with a strict pydantic model (no
type coercion: "2024" is not a fiscal year, true is not a number), and the same model produces the tool's JSON
schema, so the schema the model sees and the validation that runs can never disagree.

Errors never escape a tool (week0 §2.3): the agent receives {"error": {"type": ..., "message": ...}} and can
replan. Inside the tools, failures are typed exceptions (ToolError subclasses), so code such as the critic can
call the underlying functions and handle them directly.
"""
from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from core.exceptions import FinSightError

log = logging.getLogger(__name__)

# Error types the agent can branch on (the "type" field of an error result)
INVALID_INPUT = "invalid_input"          # arguments fail validation: fix the call
UNDEFINED = "undefined"                  # the calculation is not meaningful for these inputs (spec §7.2)
NOT_REPORTED = "not_reported"            # the company does not report this metric for the period (spec §7.5)
OUT_OF_CORPUS = "out_of_corpus"          # company, form or period outside config/companies.yaml (spec §6 R3)
DATA_MISSING = "data_missing"            # a local cache or index is not built, or does not match the settings
NO_RESULTS = "no_results"                # a search matched nothing
SERVICE_UNAVAILABLE = "service_unavailable"   # the embedding/LLM endpoint failed (core/llm.py): retry later
INTERNAL_ERROR = "internal_error"        # a bug: logged with its traceback


class ToolError(FinSightError):
    """A failure the agent should see as a typed error result. `details` are extra fields for the agent."""

    def __init__(self, error_type: str, message: str, **details: Any) -> None:
        super().__init__(message, error_type)
        self.details = details

    def result(self) -> dict[str, Any]:
        return error_result(self.error_type, self.message, **self.details)


class CalculationError(ToolError):
    """The calculator cannot produce a meaningful result (division by zero, growth from 0, CAGR domain)."""


class FactNotFound(ToolError):
    """No reported value for the requested ticker, metric and period."""


def error_result(error_type: str, message: str, **details: Any) -> dict[str, Any]:
    return {"error": {"type": error_type, "message": message, **details}}


class ToolArgs(BaseModel):
    """Base for tool argument models: strict types, unknown arguments rejected, no NaN or infinity."""
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


Args = TypeVar("Args", bound=ToolArgs)


def validation_message(err: ValidationError) -> str:
    """One readable line per problem ("fiscal_year: Input should be a valid integer"), for the model to fix."""
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in err.errors())


def tool_boundary(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn every failure of a tool call into an error result, so no exception reaches the agent loop.
    ToolError keeps its type; a FinSightError from core/llm.py (network, endpoint) is service_unavailable;
    anything else is a bug, logged with a traceback and reported as internal_error."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ToolError as err:
            return err.result()
        except ValidationError as err:
            return error_result(INVALID_INPUT, validation_message(err))
        except FinSightError as err:                   # core/llm.py: endpoint down or refusing, not a code bug
            log.warning("tool %s: %s (%s)", fn.__name__, err.message, err.error_type)
            return error_result(SERVICE_UNAVAILABLE, err.message, cause=err.error_type)
        except Exception as err:                       # the agent must get a result, never a traceback
            log.exception("tool %s failed", fn.__name__)
            return error_result(INTERNAL_ERROR, f"{type(err).__name__}: {err}")

    return wrapper


def tool_schema(name: str, description: str, args_model: type[ToolArgs]) -> dict[str, Any]:
    """OpenAI-style function schema: {"name", "description", "parameters"} from the argument model."""
    parameters = args_model.model_json_schema()
    parameters.pop("title", None)
    for prop in parameters.get("properties", {}).values():
        prop.pop("title", None)
    return {"name": name, "description": description, "parameters": parameters}
