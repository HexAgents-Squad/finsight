"""What the answer card shows (W2-10), as plain data and markdown: no Streamlit here, so it is tested directly.

    escape(text)                     model-written text made inert for st.markdown ("$" would start LaTeX)
    badge(answer)                    (label, color, icon): verified, not verified, refused, partial refusal, in words
    number(value, unit)              "$46,254 million", "22.1%", "$6.08 per share"
    figure_rows / calculation_lines / citation_links / trace(...)   the card's parts
    scope_hints(path)                the companies, fiscal years and quarters FinSight covers (config/companies.yaml)
    disclaimer(companies)            the research-assistant disclaimer
    session_trace_url(session_id)    the Langfuse page of a conversation, when FINSIGHT_TRACE_SESSIONS_URL is set

The API returns the answer contract (agent/contract.py) as JSON; this module reads it as dicts and never imports the
agent, so the UI runs on its own (streamlit, httpx, pyyaml). Every function tolerates a missing or odd field: a card
shows what it can rather than failing the conversation.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import string
from pathlib import Path
from typing import Any

import yaml

COMPANIES_FILE = Path(__file__).resolve().parents[1] / "config" / "companies.yaml"
SEC_URL = re.compile(r"^https://www\.sec\.gov/[A-Za-z0-9/._-]+$")      # nothing that could end or break a link
MARKDOWN_SPECIAL = re.compile("([" + re.escape(string.punctuation) + "])")
BARE_URL = re.compile(r"((?:[A-Za-z][A-Za-z0-9+.-]*://|www\.)\S+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+)",  # what a
                      re.IGNORECASE)                                                       # linkifier would link
SCALE_WORDS = {"USD thousands": " thousand", "USD millions": " million", "USD billions": " billion"}
REFUSAL_WORDS = {                              # agent/contract.py REFUSAL_REASONS, short (the UI does not import it)
    "R1": "investment advice", "R2": "forecast", "R3": "outside the indexed filings", "R4": "valuation opinion",
    "R5": "personal information", "R6": "bypassing FinSight's rules", "R7": "off-topic",
}
EXAMPLE_QUESTIONS = (                          # from the W2-07 demo set (development and new questions, never gold)
    "What was Procter & Gamble's operating margin in fiscal 2023?",
    "Which had the higher net income in fiscal 2023, Pfizer or Johnson & Johnson?",
    "What drove the change in Microsoft's revenue in fiscal 2023?",
    "How much did Wells Fargo spend on capital expenditures in fiscal 2023?",
)


def disclaimer(companies: int) -> str:
    return (f"FinSight is a research assistant. It answers from the indexed SEC 10-K and 10-Q filings of {companies} "
            "companies, cites them, and can be wrong: check the cited filing before relying on a figure. It gives no "
            "investment advice, forecasts or opinions on valuation.")


def escape(text: Any) -> str:
    """Model text made inert for st.markdown, shown as written. Every ASCII punctuation character is backslash-escaped
    (CommonMark allows it for all of them): "$46,254 million ... $53,512 million" is not LaTeX, "*" and "_" not
    emphasis, "1." or "-" not a list, "===" not a heading, ":rocket:" not an emoji. A URL ("https://...", "www....")
    or an email address becomes a code span, because Streamlit links one even when its punctuation is escaped; a
    backtick inside it would end the span, so it is dropped. (Checked in Chrome on Streamlit 1.67, ADR-033.)"""
    parts = []
    for i, piece in enumerate(BARE_URL.split(str(text))):
        if i % 2:
            url = piece.rstrip(".,;:!?)]}'\"")             # a sentence's punctuation after a URL is not part of it
            parts.append(f"`{code(url)}`" + MARKDOWN_SPECIAL.sub(r"\\\1", piece[len(url):]))
        elif piece:
            parts.append(MARKDOWN_SPECIAL.sub(r"\\\1", piece))
    return "".join(parts)


def code(text: Any) -> str:
    """Text for inside a code span: no backtick (it would end the span) and no line break (a blank line would too)."""
    return " ".join(str(text).replace("`", "").split())


def refusal_words(rule: Any) -> str:
    rule = str(rule or "")
    return f"{REFUSAL_WORDS[rule]} ({rule})" if rule in REFUSAL_WORDS else (rule or "policy")


def badge(answer: dict[str, Any]) -> tuple[str, str, str]:
    """The verdict as (label, color, icon); the label says it in words, never by color alone."""
    if answer.get("refused"):
        return f"Refused: {refusal_words(answer.get('refusal_rule'))}", "blue", ":material/block:"
    if answer.get("refusal_rule"):
        part = f"part refused: {refusal_words(answer['refusal_rule'])}"
        if answer.get("verified"):
            return f"Verified, {part}", "green", ":material/verified:"
        return f"Not verified, {part}", "orange", ":material/warning:"
    if answer.get("verified"):
        return "Verified", "green", ":material/verified:"
    return "Not verified", "orange", ":material/warning:"


def number(value: Any, unit: str | None) -> str:
    """A figure as the answer writes it (a magnitude with its sign: the card lists values, not prose). Up to six
    decimals, never scientific notation or float noise (0.1 + 0.2 shows as 0.3)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    magnitude = round(abs(value), 6)
    text = f"{int(magnitude):,}" if magnitude == int(magnitude) else f"{magnitude:,.6f}".rstrip("0")
    if unit in ("USD", *SCALE_WORDS):
        return f"{sign}${text}{SCALE_WORDS.get(unit, '')}"
    if unit == "USD per share":
        return f"{sign}${text} per share"
    if unit == "%":
        return f"{sign}{text}%"
    if unit in ("shares", "shares millions"):
        return f"{sign}{text}{' million' if unit == 'shares millions' else ''} shares"
    return f"{sign}{text}"


def period(fiscal_year: Any, fiscal_period: Any) -> str:
    if fiscal_year is None:
        return ""
    return f"FY{fiscal_year}" if fiscal_period in ("FY", None) else f"{fiscal_period} FY{fiscal_year}"


def as_list(items: Any) -> list[Any]:
    """A list field as a list: anything else (a string would be iterated letter by letter) as none."""
    return items if isinstance(items, list) else []


def _dicts(items: Any) -> list[dict[str, Any]]:
    return [x for x in as_list(items) if isinstance(x, dict)]


def figure_rows(answer: dict[str, Any]) -> list[dict[str, str]]:
    """The values behind the answer: what, whose, when, how much."""
    return [{"figure": str(v.get("name", "")).replace("_", " "), "company": str(v.get("ticker", "")),
             "period": period(v.get("fiscal_year"), v.get("fiscal_period")),
             "value": number(v.get("value"), v.get("unit"))}
            for v in _dicts(answer.get("values"))]


def calculation_lines(answer: dict[str, Any]) -> list[str]:
    """Each calculation as markdown: the operation, its inputs, the expression (a code span, so shown verbatim; see
    code()) and the result."""
    lines = []
    for c in _dicts(answer.get("calculations")):
        inputs = ", ".join(number(x, None) for x in as_list(c.get("inputs")))
        expression = code(c.get("expression", "")) or "-"
        lines.append(f"**{escape(c.get('operation', ''))}** of {escape(inputs)}: `{expression}` = "
                     f"**{escape(number(c.get('result'), c.get('unit')))}**")
    return lines


def citation_links(answer: dict[str, Any]) -> list[str]:
    """Each citation as a markdown line, linked when its URL is a plain SEC one (anything else is shown, not linked).
    Passages cited from the same section get "passage 1", "passage 2", in the order the answer cites them."""
    citations = _dicts(answer.get("citations"))
    labels = []
    for c in citations:
        where = c.get("item") or str(c.get("concept") or "").replace("us-gaap:", "XBRL ")
        labels.append(f"{c.get('ticker', '')} {c.get('form', '')} "
                      f"{period(c.get('fiscal_year'), c.get('fiscal_period'))} · {where}")
    seen: dict[str, int] = {}
    lines = []
    for c, label in zip(citations, labels, strict=True):
        if labels.count(label) > 1:
            seen[label] = seen.get(label, 0) + 1
            label = f"{label}, passage {seen[label]}"
        url = str(c.get("url") or "")
        lines.append(f"[{escape(label)}]({url})" if SEC_URL.match(url) else escape(label))
    return lines


def trace(body: dict[str, Any]) -> dict[str, Any]:
    """The run behind the answer, for the "Show trace" expander."""
    run = body.get("run") if isinstance(body.get("run"), dict) else {}
    visits = run.get("visits") if isinstance(run.get("visits"), dict) else {}
    return {
        "verdict": str(run.get("verdict") or ""),
        "cached": bool(body.get("cached")),
        "request_seconds": body.get("seconds"),
        "run_seconds": run.get("seconds"),
        "tokens": run.get("tokens") or 0,
        "steps": run.get("steps") or 0,
        "nodes": ", ".join(f"{k} ×{v}" for k, v in visits.items()),
        "stop_reason": run.get("stop_reason"),
        "tool_calls": [{"tool": str(t.get("tool", "")), "result": str(t.get("error") or "ok"),   # result before the
                        "arguments": json.dumps(t.get("arguments") or {}, sort_keys=True, default=str)}   # long arguments
                       for t in _dicts(run.get("tool_calls"))],
        "history": [str(h) for h in as_list(body.get("history"))],
    }


@functools.cache
def scope_hints(path: Path = COMPANIES_FILE) -> dict[str, Any]:
    """Companies (name, ticker), 10-K fiscal years and 10-Q quarters, from the corpus scope file (read once)."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    forms = cfg.get("forms") or {}
    tenq = forms.get("10-Q") or {}
    return {"companies": tuple((c.get("name", c["ticker"]), c["ticker"]) for c in cfg.get("companies") or []),
            "tenk_years": tuple((forms.get("10-K") or {}).get("fiscal_years") or []),
            "tenq_quarters": tuple(f"{q} FY{y}" for y in tenq.get("fiscal_years") or []
                                   for q in tenq.get("fiscal_periods") or [])}


def session_trace_url(session_id: str | None, base: str | None = None) -> str | None:
    """The conversation's Langfuse sessions page: the API traces a session under the first 16 hex characters of the
    id's SHA-256 (api/sessions.py session_digest), never the id itself."""
    base = base if base is not None else os.environ.get("FINSIGHT_TRACE_SESSIONS_URL", "")
    if not session_id or not base:
        return None
    return base.rstrip("/") + "/" + hashlib.sha256(session_id.encode()).hexdigest()[:16]
