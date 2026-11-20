"""What the answer card shows (W2-10), as plain data and markdown: no Streamlit here, so it is tested directly.

    escape(text)                     model-written text made safe for st.markdown ("$" would start LaTeX)
    badge(answer)                    (label, color, icon): verified, not verified, refused, partial refusal
    number(value, unit)              "$46,254 million", "22.1%", "$6.08 per share"
    figure_rows / calculation_lines / citation_links / trace(...)   the card's parts
    scope_hints(path)                the companies, fiscal years and quarters FinSight covers (config/companies.yaml)
    session_trace_url(session_id)    the Langfuse page of a conversation, when FINSIGHT_TRACE_SESSIONS_URL is set

The API returns the answer contract (agent/contract.py) as JSON; this module reads it as dicts and never imports the
agent, so the UI runs on its own (streamlit, httpx, pyyaml).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

COMPANIES_FILE = Path(__file__).resolve().parents[1] / "config" / "companies.yaml"
SEC_URL = re.compile(r"^https://www\.sec\.gov/")
MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|~$])")
SCALE_WORDS = {"USD thousands": " thousand", "USD millions": " million", "USD billions": " billion"}
EXAMPLE_QUESTIONS = (                          # from the W2-07 demo set (development and new questions, never gold)
    "What was Procter & Gamble's operating margin in fiscal 2023?",
    "Which had the higher net income in fiscal 2023, Pfizer or Johnson & Johnson?",
    "What drove the change in Microsoft's revenue in fiscal 2023?",
    "How much did Wells Fargo spend on capital expenditures in fiscal 2023?",
)
DISCLAIMER = ("FinSight is a research assistant. It answers from the indexed SEC 10-K and 10-Q filings of 20 "
              "companies, cites them, and can be wrong: check the cited filing before relying on a figure. It gives "
              "no investment advice, forecasts or opinions on valuation.")


def escape(text: str) -> str:
    """Markdown-active characters escaped: an answer's "$46,254 million ... $53,512 million" would otherwise render
    as LaTeX, and a passage's "*" or "_" as emphasis."""
    return MARKDOWN_SPECIAL.sub(r"\\\1", text)


def badge(answer: dict[str, Any]) -> tuple[str, str, str]:
    """The verdict as (label, color, icon); the label says it in words, never by color alone."""
    if answer.get("refused"):
        return f"Refused ({answer.get('refusal_rule') or 'policy'})", "blue", ":material/block:"
    if answer.get("refusal_rule"):
        if answer.get("verified"):
            return f"Verified, part refused ({answer['refusal_rule']})", "green", ":material/verified:"
        return f"Not verified, part refused ({answer['refusal_rule']})", "orange", ":material/warning:"
    if answer.get("verified"):
        return "Verified", "green", ":material/verified:"
    return "Not verified", "orange", ":material/warning:"


def number(value: float, unit: str | None) -> str:
    """A figure as the answer writes it (a magnitude with its sign: the card lists values, not prose)."""
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    text = f"{int(magnitude):,}" if magnitude == int(magnitude) else f"{magnitude:,}"
    if unit in ("USD", *SCALE_WORDS):
        return f"{sign}${text}{SCALE_WORDS.get(unit, '')}"
    if unit == "USD per share":
        return f"{sign}${text} per share"
    if unit == "%":
        return f"{sign}{text}%"
    if unit in ("shares", "shares millions"):
        return f"{sign}{text}{' million' if unit == 'shares millions' else ''} shares"
    return f"{sign}{text}"


def period(fiscal_year: int, fiscal_period: str) -> str:
    return f"FY{fiscal_year}" if fiscal_period == "FY" else f"{fiscal_period} FY{fiscal_year}"


def figure_rows(answer: dict[str, Any]) -> list[dict[str, str]]:
    """The values behind the answer: what, whose, when, how much."""
    return [{"figure": str(v.get("name", "")).replace("_", " "), "company": str(v.get("ticker", "")),
             "period": period(v["fiscal_year"], v["fiscal_period"]), "value": number(v["value"], v.get("unit"))}
            for v in answer.get("values") or []]


def calculation_lines(answer: dict[str, Any]) -> list[str]:
    """Each calculation as markdown: the operation, its inputs, the expression and the result."""
    lines = []
    for c in answer.get("calculations") or []:
        inputs = ", ".join(number(x, None) for x in c.get("inputs") or [])
        lines.append(f"**{escape(str(c.get('operation', '')))}** of {inputs}: `{c.get('expression', '')}` = "
                     f"**{escape(number(c['result'], c.get('unit')))}**")
    return lines


def citation_links(answer: dict[str, Any]) -> list[str]:
    """Each citation as a markdown line, linked when its URL is an SEC one (anything else is shown, not linked)."""
    lines = []
    for c in answer.get("citations") or []:
        where = c.get("item") or (c.get("concept") or "").replace("us-gaap:", "XBRL ")
        label = escape(f"{c.get('ticker', '')} {c.get('form', '')} {period(c['fiscal_year'], c['fiscal_period'])} · "
                       f"{where}")
        url = str(c.get("url", ""))
        lines.append(f"[{label}]({url})" if SEC_URL.match(url) and ")" not in url else label)
    return lines


def trace(body: dict[str, Any]) -> dict[str, Any]:
    """The run behind the answer, for the "Show trace" expander."""
    run = body.get("run") or {}
    return {
        "verdict": run.get("verdict", ""),
        "cached": bool(body.get("cached")),
        "request_seconds": body.get("seconds"),
        "run_seconds": run.get("seconds"),
        "tokens": run.get("tokens", 0),
        "steps": run.get("steps", 0),
        "nodes": ", ".join(f"{k} ×{v}" for k, v in (run.get("visits") or {}).items()),
        "stop_reason": run.get("stop_reason"),
        "tool_calls": [{"tool": t.get("tool", ""), "arguments": json.dumps(t.get("arguments") or {}, sort_keys=True),
                        "result": t.get("error") or "ok"} for t in run.get("tool_calls") or []],
        "history": list(body.get("history") or []),
    }


def scope_hints(path: Path = COMPANIES_FILE) -> dict[str, Any]:
    """Companies (name, ticker), 10-K fiscal years and 10-Q quarters, from the corpus scope file."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    forms = cfg.get("forms") or {}
    tenq = forms.get("10-Q") or {}
    return {"companies": [(c.get("name", c["ticker"]), c["ticker"]) for c in cfg.get("companies") or []],
            "tenk_years": list((forms.get("10-K") or {}).get("fiscal_years") or []),
            "tenq_quarters": [f"{q} FY{y}" for y in tenq.get("fiscal_years") or [] for q in tenq.get("fiscal_periods") or []]}


def session_trace_url(session_id: str | None, base: str | None = None) -> str | None:
    """The conversation's Langfuse sessions page: the API traces a session under the first 16 hex characters of the
    id's SHA-256 (api/sessions.py session_digest), never the id itself."""
    base = base if base is not None else os.environ.get("FINSIGHT_TRACE_SESSIONS_URL", "")
    if not session_id or not base:
        return None
    return base.rstrip("/") + "/" + hashlib.sha256(session_id.encode()).hexdigest()[:16]
