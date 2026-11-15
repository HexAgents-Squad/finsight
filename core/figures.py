"""Figures in free text, for checking that every number an answer states is backed by a value (W1-07 critic,
spec §8 invariant 1) or by the evidence (W1-08 scorer, eval/score_answers.py).

figures(text) finds the numbers a reader sees as figures, with the scale word applied ("$10.1 billion" ->
1.01e10) and the precision they were printed at, and skips what is not a figure: dates, bare years, ordinals,
footnote markers, Item/form/quarter names and passage ids. A percentage range counts as two percentages.
Moved here unchanged from eval/score_answers.py so the agent does not depend on the evaluation code.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

SCALE_WORDS = {"thousand": 1e3, "thousands": 1e3, "k": 1e3, "million": 1e6, "millions": 1e6, "mn": 1e6, "mm": 1e6,
               "m": 1e6, "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12, "trillions": 1e12,
               "tn": 1e12}
MONTHS = (r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
          r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b\.?")
DAY = r"\d{1,2}(?:st|nd|rd|th)?(?!\d|[.,]\d|%)"    # "June 30," yes; never "44" of "44.6%" or "12" of "12,345"
NOT_A_FIGURE = re.compile(
    rf"\b{MONTHS}\s+{DAY}(?:,?\s+\d{{4}})?"          # September 28, 2024 / June 30
    rf"|\b{DAY}\s+{MONTHS}(?:,?\s+\d{{4}})?"         # 28 September 2024
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b10-[KQ]s?\b|\b(?:Part\s+I{1,2}\s+)?Item\s+\d{1,2}[A-C]?\b|\bQ[1-4]\b|\b\d+Q\d{2,4}\b|\bFY\s?\d{2,4}\b"
    r"|\b\d+(?:st|nd|rd|th)\b|\bS&P\s*500\b|\bCOVID-19\b|(?<=[A-Za-z)])\(\d{1,2}\)|\[P\d+\]|\bP\d+\b",
    re.IGNORECASE)
# A dollar figure as the gold set writes it ("391,035"): what the verbatim-figure checks look for (ADR-012
# check 2, W1-08 evidence, W1-05 answer figure in the retrieved chunks)
GROUPED_FIGURE = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+)(?![\d,])")
PERCENT_RANGE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%")       # "10-12%" -> "10% 12%"
FIGURE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:\s*(%|percent|per cent|pp|percentage points?)"
                    r"|\s*(thousands?|millions?|billions?|trillions?|bn|mn|mm|tn|k|m|b)\b)?(?![\w%])", re.IGNORECASE)
YEAR_RANGE = range(1990, 2040)


@dataclass(frozen=True)
class Figure:
    value: float           # absolute value with its scale word applied ("$10.1 billion" -> 1.01e10)
    tolerance: float       # half a unit of its last printed digit, scaled ("10.1 billion" -> 0.05e9)
    percent: bool
    text: str


def figures(text: str) -> list[Figure]:
    """The figures in `text`, without dates, bare years, ordinals, footnote markers, Item/form/quarter names and
    passage ids; a percentage range counts as two percentages."""
    clean = PERCENT_RANGE.sub(r"\1% \2%", NOT_A_FIGURE.sub(" ", text))
    out = []
    for m in FIGURE.finditer(clean):
        whole, frac, pct, word = m.group(1), m.group(2) or "", m.group(3), m.group(4)
        number = float(whole.replace(",", "") + frac)
        if not math.isfinite(number):                 # hundreds of digits: no figure, and int() would overflow
            continue
        if not frac and not pct and not word and "," not in whole and int(number) in YEAR_RANGE:
            continue                                  # a bare year such as 2024
        scale = SCALE_WORDS[word.lower()] if word else 1.0
        decimals = len(frac) - 1 if frac else 0
        out.append(Figure(number * scale, 0.5 * 10 ** -decimals * scale, bool(pct), m.group(0).strip()))
    return out


def contains_figure(text: str, figure: str) -> bool:
    """Whether `figure` ("391,035") appears in `text` as a whole number, not inside a longer one ("1,391,035")."""
    return re.search(rf"(?<![\d,]){re.escape(figure)}(?![\d,])", text) is not None
