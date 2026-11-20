"""A run as a transcript (W2-07): what scripts/ask.py prints and eval/run_demo.py commits.

    run_record(run, question, history=())  -> dict    the answer contract plus what the run cost
    markdown(record)                       -> str     one run, readable

The record keeps the answer, the verdict, the values, citations (with their SEC links) and calculations, and the
run's numbers: tool calls (name, arguments, typed error), tokens, seconds, node visits, the stop reason and the
nodes' notes. No key, endpoint address or local path is in it; the question is the user's own text.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent.graph import AgentRun


def verdict(run: AgentRun) -> str:
    """"refused", "partial refusal, verified", "partial refusal, not verified", "verified" or "not verified"."""
    a = run.answer
    if a.refused:
        return "refused"
    if a.refusal_rule:
        return "partial refusal, verified" if a.verified else "partial refusal, not verified"
    return "verified" if a.verified else "not verified"


def run_record(run: AgentRun, question: str, history: Sequence[str] | None = None) -> dict[str, Any]:
    """`history` defaults to the earlier questions the run saw (the state's: given, or from the thread)."""
    a, state = run.answer, run.state
    plan = state.get("plan")
    return {
        "question": question,
        "history": list(state.get("history") or []) if history is None else list(history),
        "verdict": verdict(run),
        "answer": a.answer,
        "verified": a.verified,
        "refused": a.refused,
        "refusal_rule": a.refusal_rule,
        "values": [v.model_dump(mode="json") for v in a.values],
        "citations": [c.model_dump(mode="json", exclude_none=True) for c in a.citations],
        "calculations": [c.model_dump(mode="json") for c in a.calculations],
        "notes": list(a.notes),
        "plan": None if plan is None else {"refuse": plan.refuse, "refusal_rule": plan.refusal_rule,
                                           "calculation": plan.calculation,
                                           "steps": [{"tool": s.tool, "arguments": s.arguments} for s in plan.steps]},
        "tool_calls": [{"tool": r.call.name, "arguments": r.call.arguments, "error": r.error_type}
                       for r in state.get("tool_results") or []],
        "steps": len(state.get("tool_results") or []),
        "tokens": state.get("tokens") or 0,
        "seconds": round(run.seconds, 2),
        "visits": dict(state.get("visits") or {}),
        "stop_reason": state.get("stop_reason"),
        "node_notes": list(state.get("errors") or []),
    }


def _number(x: float) -> str:
    """A figure as a reader writes it: thousands separators, no exponent, no trailing ".0" (13,600,000,000)."""
    return f"{int(x):,}" if x == int(x) else f"{x:,}"


def _citation_line(c: dict[str, Any]) -> str:
    where = c.get("item") or c.get("concept") or ""
    return f"[{c['ticker']} {c['form']} FY{c['fiscal_year']} {c['fiscal_period']} · {where}] {c['url']}"


def markdown(record: dict[str, Any], title: str | None = None) -> str:
    lines = [f"### {title or record['question']}", ""]
    if title:
        lines += [f"**Question:** {record['question']}", ""]
    if record["history"]:
        lines += ["**Earlier questions:** " + " / ".join(record["history"]), ""]
    lines += [f"**Answer ({record['verdict']}):** {record['answer']}", ""]
    if record["values"]:
        lines.append("**Values:** " + "; ".join(
            f"{v['name']} = {_number(v['value'])} {v['unit']} ({v['ticker']} {v['fiscal_period']} {v['fiscal_year']})"
            for v in record["values"]))
    for c in record["calculations"]:
        lines.append(f"**Calculation:** {c['operation']} of {', '.join(_number(x) for x in c['inputs'])} = "
                     f"{_number(c['result'])}")
    if record["citations"]:
        lines += ["**Citations:**", *[f"- {_citation_line(c)}" for c in record["citations"]]]
    if record["tool_calls"]:
        lines.append("**Tool calls:** " + ", ".join(
            r["tool"] + (f" ({r['error']})" if r["error"] else "") for r in record["tool_calls"]))
    lines.append(f"**Run:** {record['steps']} tool calls, {record['tokens']} tokens, {record['seconds']} s, "
                 f"visits {record['visits']}" + (f", stopped: {record['stop_reason']}" if record["stop_reason"] else ""))
    for note in [*record["notes"], *record["node_notes"]]:
        lines.append(f"- note: {note}")
    return "\n".join(lines) + "\n"
