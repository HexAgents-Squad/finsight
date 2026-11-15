"""The financial statements first, for a question about a reported figure (the Item 8 preference, W1-05, ADR-017).

A reported figure is answered and cited from the financial statements (spec §8), but the MD&A repeats the same
figures in prose and often ranks above them: before W1-05 most retrieval misses were the right filing and the
wrong Item (ADR-013, ADR-023). StatementFirstIndex runs a second search restricted to the filing's statement Item
(10-K: Item 8, 10-Q: Part I Item 1) and interleaves its results first. It applies only when the search names a
fiscal period (so the form, and the Item, is known), names no Item itself, and the query asks for a figure.
"""
from __future__ import annotations

import re
from dataclasses import replace

from retrieval.common import NO_FILTERS, Filters, SearchResult
from retrieval.hybrid import Retriever, interleave

FIGURE_WORDS = re.compile(
    r"\b(?:revenues?|net\s+sales|sales|income|earnings|profit|loss(?:es)?|margin|eps|earnings\s+per\s+share|assets|"
    r"liabilities|equity|cash|debt|expenditures?|capex|research\s+and\s+development|r&d|expenses?|dividends?|"
    r"deposits|loans|goodwill|inventor(?:y|ies)|repurchases?|buybacks?|shares\s+outstanding)\b",
    re.IGNORECASE)
QUALITATIVE_WORDS = re.compile(
    r"\b(?:drove|drive|driven|drivers?|explains?|explained|why|describes?|discuss(?:es)?|risks?(?!-)|factors?|"
    r"strategy|outlook|mention|attribute[sd]?)\b", re.IGNORECASE)
ANNUAL_STATEMENTS = "Item 8"
QUARTERLY_STATEMENTS = "Part I Item 1"


def asks_for_figure(text: str) -> bool:
    """Whether a question asks for a reported figure: a metric word, and no word asking for an explanation."""
    return bool(FIGURE_WORDS.search(text)) and not QUALITATIVE_WORDS.search(text)


def statement_item(fiscal_period: str | None) -> str | None:
    """The Item that holds the financial statements of a filing with this period (None: period unknown)."""
    if fiscal_period is None:
        return None
    return ANNUAL_STATEMENTS if fiscal_period == "FY" else QUARTERLY_STATEMENTS


class StatementFirstIndex:
    """`base`, with a figure question's statement-Item results interleaved first (1st statement hit, 1st hit,
    2nd statement hit, ...), without repeats."""

    def __init__(self, base: Retriever):
        self.base = base

    def search(self, query: str, k: int = 5, filters: Filters = NO_FILTERS) -> list[SearchResult]:
        hits = self.base.search(query, k, filters)
        item = statement_item(filters.fiscal_period) if filters.item is None and asks_for_figure(query) else None
        if item is None:
            return hits
        return interleave([self.base.search(query, k, replace(filters, item=item)), hits], k)
