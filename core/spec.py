"""Names docs/spec.md fixes, defined once for the tools, the agent and the evaluation.

Units (§7 "Allowed unit values"), fiscal periods (§4), refusal rules (§6), forms, the exact Item names (§3) and
the document id every section and chunk is named by.
tools/calculator.py, agent/contract.py and eval/schema.py import these; a change here is a spec change (spec
header, "Change rule": team agreement and an entry in docs/decisions.md).
"""
from __future__ import annotations

from typing import Literal, get_args

Unit = Literal["USD", "USD thousands", "USD millions", "USD billions", "USD per share", "shares", "shares millions",
               "%", "ratio", "count"]
ALLOWED_UNITS: frozenset[str] = frozenset(get_args(Unit))

FiscalPeriod = Literal["FY", "Q1", "Q2", "Q3"]
Form = Literal["10-K", "10-Q"]
RefusalRule = Literal["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
REFUSAL_RULES: tuple[str, ...] = get_args(RefusalRule)
NOT_REPORTED = "not reported"

# §3 "Item naming": a 10-K has Items 1-16, a 10-Q has Parts I and II
TENK_ITEMS: frozenset[str] = frozenset(f"Item {n}" for n in (
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "9C",
    "10", "11", "12", "13", "14", "15", "16"))
TENQ_ITEMS: frozenset[str] = frozenset(
    [f"Part I Item {n}" for n in ("1", "2", "3", "4")]
    + [f"Part II Item {n}" for n in ("1", "1A", "2", "3", "4", "5", "6")])
ITEMS_OF_FORM: dict[str, frozenset[str]] = {"10-K": TENK_ITEMS, "10-Q": TENQ_ITEMS}
ItemName = Literal[tuple(sorted(TENK_ITEMS | TENQ_ITEMS))]      # every Item of either form, for schemas


def doc_id(ticker: str, form: str, fiscal_year: int, fiscal_period: str, key: str) -> str:
    """'AAPL_10-K_2024_Item7', 'AAPL_10-Q_2025Q2_PartIItem2' (week0_plan §2.1); a chunk id adds _<variant>_<n>."""
    period = "" if fiscal_period == "FY" else fiscal_period
    return f"{ticker}_{form}_{fiscal_year}{period}_{key.replace(' ', '')}"
