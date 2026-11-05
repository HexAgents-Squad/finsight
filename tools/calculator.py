"""calculate: the only place a number in an answer is computed (spec §7.2). Whitelisted operations, no eval().

    calculate(operation, inputs, years?, unit?) -> {operation, inputs, result, unit, expression, notes}

Operations and input order (chronological where order matters: old first, new second):
    ratio       [a, b]          a / b                                       unit "ratio", 2 decimals
    percent     [a, b]          a / b * 100                                 unit "%", 1 decimal
    growth      [old, new]      (new - old) / abs(old) * 100                unit "%", 1 decimal
    difference  [old, new]      new - old                                   unit = `unit` argument, if given
    sum         [a, b, ...]     a + b + ...                                 unit = `unit` argument, if given
    cagr        [start, end]    ((end / start) ** (1 / years) - 1) * 100    unit "%", 1 decimal; `years` required

Undefined cases are errors of type "undefined", never a number (spec §7.2): ratio/percent with b = 0, growth from
old = 0, CAGR with start <= 0, end < 0 or years <= 0, and results too large to represent. A negative growth base
is allowed and noted (the answer must say the base was negative).

Arithmetic is exact decimal arithmetic on the inputs as written (Decimal(repr(x))), then rounded half-up to the
spec §7.4 decimals, as the gold set is: 23 / 80 * 100 is exactly 28.75 -> 28.8, where binary floats give
28.749999... -> 28.7. The critic recomputes with compute() and gets the identical value (spec §8 invariant 3).
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal, DecimalException, localcontext
from typing import Annotated, Any, Literal

from pydantic import Field

from tools.base import (
    INVALID_INPUT,
    UNDEFINED,
    CalculationError,
    ToolArgs,
    ToolError,
    tool_boundary,
    tool_schema,
)

Operation = Literal["ratio", "percent", "growth", "difference", "sum", "cagr"]
# Units an answer may carry (spec §7, "Allowed unit values")
Unit = Literal["USD", "USD thousands", "USD millions", "USD billions", "USD per share", "shares", "shares millions",
               "%", "ratio", "count"]

PAIR_OPERATIONS = frozenset({"ratio", "percent", "growth", "difference", "cagr"})   # exactly two inputs
MIN_INPUTS = 2                          # every operation combines at least two numbers
MAX_INPUTS = 50                         # far more than any answer needs; bounds the expression length
# Largest input magnitude: 1e15 is ~10,000x Walmart's revenue in USD millions, and every number up to 2**53
# (~9e15) is an exact float, so inputs keep the digits they were given and print as plain integers
MAX_ABS_INPUT = 1e15
RESULT_UNIT: dict[str, str] = {"ratio": "ratio", "percent": "%", "growth": "%", "cagr": "%"}
UNIT_FROM_ARGUMENT = frozenset({"difference", "sum"})       # the result is in the inputs' own unit
# Spec §7.4: percentages 1 decimal, per-share values 2; ratios 2 (gold B15 asks for two, and one decimal would
# collapse e.g. a 1.03 current ratio to 1.0). Other units keep their value up to OTHER_DECIMALS, which only
# trims the tail of non-terminating results.
DECIMALS = {"%": 1, "ratio": 2, "USD per share": 2}
OTHER_DECIMALS = 6
# Decimal digits of working precision: enough to hold any float's integer part (up to ~1e308) plus the rounding
# digits exactly, so quantize never fails on a representable result
PRECISION = 400
# A fractional power is irrational in general (never exactly halfway), so 50 digits are plenty and keep CAGR fast;
# Decimal's power is correctly rounded, so exact cases (1.21 ** 0.5 = 1.1) stay exact
POWER_PRECISION = 50
HUNDRED = Decimal(100)
NEGATIVE_BASE_NOTE = "the base (old value) is negative, so growth is measured against its absolute value"
NO_UNIT_NOTE = "no unit given: the result is in the inputs' unit"


class CalculateArgs(ToolArgs):
    operation: Operation = Field(description="ratio | percent | growth | difference | sum | cagr")
    inputs: list[Annotated[float, Field(ge=-MAX_ABS_INPUT, le=MAX_ABS_INPUT)]] = Field(
        min_length=MIN_INPUTS, max_length=MAX_INPUTS, description=(
            "Numbers exactly as returned by get_financial_fact. ratio/percent: [a, b] for a / b. "
            "growth/difference: [old, new]. cagr: [start, end]. sum: two or more values."))
    years: float | None = Field(default=None, description="cagr only: number of years between start and end")
    unit: Unit | None = Field(default=None, description=(
        "difference and sum only (optional): the unit the inputs share, e.g. 'USD millions', returned as the "
        "result's unit; other operations set their unit themselves"))


def _dec(x: float) -> Decimal:
    """The number as written (0.1 -> Decimal('0.1')), not its binary approximation."""
    return Decimal(repr(x))


def round_half_up(value: float | Decimal, decimals: int) -> float:
    """Half-up rounding of the value as written (2.25 -> 2.3), unlike round() (banker's rounding on binary floats)."""
    with localcontext() as ctx:
        ctx.prec = PRECISION
        exact = value if isinstance(value, Decimal) else _dec(value)
        return float(exact.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


def _num(x: float, wrap: bool = True) -> str:
    """A number as it appears in an expression: integers without ".0", negatives in parentheses (unless the
    number is already an argument, as in abs(-5))."""
    text = str(int(x)) if x.is_integer() and abs(x) < 1e15 else repr(x)
    return f"({text})" if x < 0 and wrap else text


def _check_arity(operation: str, inputs: list[float]) -> None:
    if operation in PAIR_OPERATIONS and len(inputs) != 2:
        raise ToolError(INVALID_INPUT, f"{operation} takes exactly 2 inputs, got {len(inputs)}")


def _evaluate(op: str, xs: list[float], years: float | None) -> tuple[Decimal, str, list[str]]:
    """(exact value, expression, notes) for validated arguments; undefined cases raise CalculationError."""
    notes: list[str] = []
    ds = [_dec(x) for x in xs]
    if op in ("ratio", "percent"):
        (a, b), (da, db) = xs, ds
        if b == 0:
            raise CalculationError(UNDEFINED, f"{op} is undefined: the denominator is 0", inputs=xs)
        value = da / db * (HUNDRED if op == "percent" else 1)
        expression = f"{_num(a)} / {_num(b)}" + (" * 100" if op == "percent" else "")
    elif op == "growth":
        (old, new), (d_old, d_new) = xs, ds
        if old == 0:
            raise CalculationError(UNDEFINED, "growth is undefined: the old value is 0", inputs=xs)
        if old < 0:
            notes.append(NEGATIVE_BASE_NOTE)
        value = (d_new - d_old) / abs(d_old) * HUNDRED
        expression = f"({_num(new)} - {_num(old)}) / abs({_num(old, wrap=False)}) * 100"
    elif op == "difference":
        (old, new), (d_old, d_new) = xs, ds
        value, expression = d_new - d_old, f"{_num(new)} - {_num(old)}"
    elif op == "sum":
        value, expression = sum(ds, Decimal(0)), " + ".join(_num(x) for x in xs)
    else:                                                                       # cagr
        (start, end), (d_start, d_end) = xs, ds
        n = years
        if start <= 0 or end < 0 or n <= 0:
            raise CalculationError(UNDEFINED, "cagr is undefined when start <= 0, end < 0 or years <= 0",
                                   inputs=xs, years=n)
        with localcontext() as ctx:
            ctx.prec = POWER_PRECISION
            factor = (d_end / d_start) ** (Decimal(1) / _dec(n))
        value = (factor - 1) * HUNDRED
        expression = f"(({_num(end)} / {_num(start)}) ** (1 / {_num(n)}) - 1) * 100"
    return value, expression, notes


def compute(operation: str, inputs: list[float], years: float | None = None, unit: str | None = None) -> dict:
    """The calculation result. Raises pydantic ValidationError for wrong argument types, ToolError (invalid_input)
    for a wrong number of inputs or a misplaced years/unit, CalculationError (undefined) for spec §7.2's undefined
    cases. The critic calls this directly to recompute a calculation."""
    args = CalculateArgs(operation=operation, inputs=inputs, years=years, unit=unit)
    op, xs = args.operation, args.inputs
    _check_arity(op, xs)
    if (args.years is not None) != (op == "cagr"):
        raise ToolError(INVALID_INPUT, "years is required for cagr and not allowed for other operations")
    if args.unit is not None and op not in UNIT_FROM_ARGUMENT:
        raise ToolError(INVALID_INPUT, f"unit is only for difference and sum; {op} returns "
                                       f"{RESULT_UNIT[op]!r}")
    result_unit = args.unit if op in UNIT_FROM_ARGUMENT else RESULT_UNIT[op]
    try:
        with localcontext() as ctx:
            ctx.prec = PRECISION
            value, expression, notes = _evaluate(op, xs, args.years)
            result = round_half_up(value, DECIMALS.get(result_unit, OTHER_DECIMALS))
    except (OverflowError, DecimalException) as err:      # beyond the decimal or float range
        raise CalculationError(UNDEFINED, f"{op} result is not a finite number", inputs=xs) from err
    if not math.isfinite(result):                             # representable as a decimal, not as a float
        raise CalculationError(UNDEFINED, f"{op} result is not a finite number", inputs=xs)
    if op in UNIT_FROM_ARGUMENT and result_unit is None:
        notes.append(NO_UNIT_NOTE)
    return {"operation": op, "inputs": xs, "result": result, "unit": result_unit, "expression": expression,
            "notes": notes}


@tool_boundary
def calculate(operation: str, inputs: list[float], years: float | None = None,
              unit: str | None = None) -> dict[str, Any]:
    """Tool entry point: the result, or {"error": {...}} (never raises)."""
    return compute(operation, inputs, years, unit)


SCHEMA = tool_schema(
    "calculate",
    "Compute a ratio, percent, growth, difference, sum or CAGR from numbers returned by get_financial_fact. "
    "Every computed number in an answer must come from this tool. Returns result, unit and the expression used; "
    "undefined cases (division by zero, growth from 0, CAGR with non-positive start) return an error.",
    CalculateArgs)
