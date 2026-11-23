"""W3-09: the evaluation report, generated from committed results (docs/week3_plan.txt §5, the plan §9).

    python eval/report.py            # rewrite the generated tables, the charts and the PDF
    python eval/report.py --check    # exit 1 if any of them is stale (tests do the same, like the weekly notes)

Writes eval/results/evaluation_report.md (only the tables between the GENERATED markers; the text around them is
written by hand and kept), its charts eval/results/report/*.svg, each next to the table it draws, and
eval/results/report/evaluation_report.pdf, the whole report rendered from that Markdown (text, tables and charts).
Every number is read from committed files, so a fresh clone (CI) regenerates every byte, without a key or data/:
    eval/results/scores/<run>.jsonl        the W1-08 scorer's verdicts, grades, labels, checks and cost per question
    eval/results/model_runs/<run>.meta.json   each run's requests (live and from the reply cache)
    eval/results/traces/<run>.jsonl        the traced runs (scripts/trace_runs.py, W3-09)
The versions are the plan's: B0, the best prompting strategy (the most correct answers of S1-S3), A1 (the agent),
the ablations A2-A4 (A1 with one component off) and A5 (W3-08's final version). The failure taxonomy (the plan
§9.3) maps each wrong answer's failure label (confirmed by the graders, else the scorer's suggestion; TAXONOMY):
    refusal error     over_refusal, under_refusal, wrong_refusal_rule
    hallucination     invented_number, ignored_not_reported (a figure the filings do not report)
    tool-call error   malformed_output (no usable answer: invalid JSON, or no single value)
    retrieval miss    wrong_line_item and incomplete_summary when the gold figures were not in what the run
                      retrieved (its passages, or for the agent its tools' output); the scorer checks this for
                      numeric answers, so a summary counts as a reasoning error
    reasoning error   the rest: wrong_fiscal_year, unit_confusion, arithmetic_slip, wrong_company, and
                      wrong_line_item / incomplete_summary with the figures in hand
Exit codes: 0 = written (or current with --check), 1 = stale (--check) or a source is unreadable.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "eval" / "results"
SCORES = RESULTS / "scores"
RUNS = RESULTS / "model_runs"
TRACES = RESULTS / "traces"
REPORT = RESULTS / "evaluation_report.md"
CHART_DIR = RESULTS / "report"
PDF = CHART_DIR / "evaluation_report.pdf"
START, END = "<!-- GENERATED:START -->", "<!-- GENERATED:END -->"

B0 = "M1_qwen3-8b_rag_think-off_b0-v1"
STRATEGIES = ("S1_qwen3-8b_rag_think-off_s1-v1", "S2_qwen3-8b_rag_think-off_s2-v1", "S3_qwen3-8b_rag_think-off_s3-v1")
AGENT_RUNS = (("A1", "A1_qwen3-8b_agent_think-off_a-v1", "the agent"),
              ("A2", "A2_qwen3-8b_agent_think-off_a-v1-no-calculator", "A1, no calculator"),
              ("A3", "A3_qwen3-8b_agent_think-off_a-v1-no-critic", "A1, no critic"),
              ("A4", "A4_qwen3-8b_agent_think-off_a-v1-no-statements-first", "A1, no statements-first retrieval"),
              ("A5", "A5_qwen3-8b_agent_think-off_a-v1", "the final version (W3-08)"))
TRACED = (("W3-09-demo-a5", "the 10 demo queries, A5", "live"),
          ("W3-09-a1-failures-a1", "A1's 16 wrong answers, A1's code", "from the cache"),
          ("W3-09-a1-failures-a5", "the same 16 questions, A5", "from the cache"))
ANSWER_TYPES = (("Short: numeric", ("short_numeric",)), ("Short: entity", ("short_entity",)),
                ("Summary", ("summary",)), ("Refusal & not reported", ("refusal", "trap", "partial", "control")))
CLASSES = ("retrieval miss", "reasoning error", "tool-call error", "hallucination", "refusal error")
TAXONOMY = {"over_refusal": "refusal error", "under_refusal": "refusal error", "wrong_refusal_rule": "refusal error",
            "invented_number": "hallucination", "ignored_not_reported": "hallucination",
            "malformed_output": "tool-call error"}
RETRIEVAL_LABELS = ("wrong_line_item", "incomplete_summary")
COLORS = ("#2f6db5", "#e07b39", "#4b9a5a", "#c44e52", "#8172b2")      # one per series, in legend order


class SourceError(ValueError):
    """A committed source is missing or has changed shape: named, never an empty cell."""


# ----------------------------------------------------------------------------------------------- the data
@dataclass(frozen=True)
class Version:
    label: str            # B0, S1, A1 ...
    run_id: str
    what: str
    scored: tuple[dict[str, Any], ...]
    meta: dict[str, Any]

    @property
    def agent(self) -> bool:
        return self.meta.get("context") == "agent"


def _scores(run_id: str) -> tuple[dict[str, Any], ...]:
    path = SCORES / f"{run_id}.jsonl"
    try:
        rows = tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, ValueError) as err:
        raise SourceError(f"{path.relative_to(REPO)}: {err}") from err
    if not rows or any(not {"id", "answer_type", "correct"} <= set(r) for r in rows):
        raise SourceError(f"{path.relative_to(REPO)}: not a scores file")
    return rows


def _meta(run_id: str) -> dict[str, Any]:
    path = RUNS / f"{run_id}.meta.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise SourceError(f"{path.relative_to(REPO)}: {err}") from err


def correct(scored: tuple[dict[str, Any], ...]) -> int:
    return sum(s["correct"] is True for s in scored)


def versions() -> list[Version]:
    """B0, the best strategy, A1-A5 (the order of the charts). Every answer must be decided: a pending grade in a
    committed run is an error (the report states final numbers only)."""
    strategies = [Version(r.split("_", 1)[0], r, "prompting strategy", _scores(r), _meta(r)) for r in STRATEGIES]
    best = max(strategies, key=lambda v: (correct(v.scored), -STRATEGIES.index(v.run_id)))
    out = [Version("B0", B0, "single-shot RAG baseline", _scores(B0), _meta(B0)),
           Version(best.label, best.run_id, "the best prompting strategy", best.scored, best.meta)]
    out += [Version(label, run_id, what, _scores(run_id), _meta(run_id)) for label, run_id, what in AGENT_RUNS]
    for v in out:
        if pending := sorted(s["id"] for s in v.scored if s["correct"] is None):
            raise SourceError(f"{v.run_id} has answers waiting for a grade: {', '.join(pending[:5])}")
    return out


def taxonomy_class(s: dict[str, Any]) -> str:
    """The plan's failure class of one wrong answer (module docstring)."""
    label = s.get("failure_label") or s.get("suggested_label")
    if label in TAXONOMY:
        return TAXONOMY[label]
    if label in RETRIEVAL_LABELS and s.get("gold_figures_in_pack") is False:
        return "retrieval miss"
    return "reasoning error"


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def live_latency(v: Version) -> bool:
    """Whether the run's latencies are live ones: a prompting run's record keeps each reply's own seconds even from
    the cache, but an agent run's latency is the graph's wall time, in which a cached reply takes none (A2-A4 were
    replayed from A1's replies, so their seconds say nothing about speed)."""
    return not v.agent or not any(s.get("cache_hits") for s in v.scored)


def cache_rate(v: Version) -> str:
    """Chat replies that came from the reply cache when the run was made (a later replay does not count): the
    agent's per-question counts; a prompting run's per-question flag."""
    if v.agent:
        hits = sum(s.get("cache_hits") or 0 for s in v.scored)
        total = hits + sum(s.get("live_requests") or 0 for s in v.scored)
        return f"{hits}/{total} = {hits / total:.0%}" if total else "–"
    cached = sum(bool(s.get("cached")) for s in v.scored)
    return f"{cached}/{len(v.scored)} = {cached / len(v.scored):.0%}"


# ----------------------------------------------------------------------------------------------- the tables
def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def accuracy_rows(vs: list[Version]) -> list[list[str]]:
    rows = []
    for label, types in ANSWER_TYPES:
        cells = []
        for v in vs:
            pool = [s for s in v.scored if s["answer_type"] in types]
            cells.append(f"{correct(tuple(pool))}/{len(pool)}")
        rows.append([label, *cells])
    rows.append(["**All**", *[f"**{correct(v.scored)}/{len(v.scored)}**" for v in vs]])
    return rows


def taxonomy_counts(v: Version) -> dict[str, int]:
    counts = dict.fromkeys(CLASSES, 0)
    for s in v.scored:
        if s["correct"] is False:
            counts[taxonomy_class(s)] += 1
    return counts


def agent_rows(vs: list[Version]) -> list[list[str]]:
    """Task completion, tool calls, retries and the step cap per task (the plan §9.2, Agent)."""
    rows: dict[str, list[str]] = {k: [] for k in (
        "task completed (an answer contract, no error, within the limits)", "tool calls per task (mean)",
        "tool calls that returned an error", "JSON retries per task (mean)", "stopped by the step cap",
        "verified by the critic", "answers whose citations all name a source the run had")}
    for v in vs:
        n = len(v.scored)
        done = sum(bool(s.get("json_valid")) and not s.get("error") and not s.get("stop_reason") for s in v.scored)
        rows["task completed (an answer contract, no error, within the limits)"].append(f"{done}/{n}")
        if v.agent:
            steps = [s.get("tool_steps") or 0 for s in v.scored]
            rows["tool calls per task (mean)"].append(f"{statistics.fmean(steps):.2f}")
            rows["tool calls that returned an error"].append(f"{sum(s.get('tool_errors') or 0 for s in v.scored)}"
                                                              f"/{sum(steps)}")
            capped = sum("step" in str(s.get("stop_reason") or "") for s in v.scored)
            rows["stopped by the step cap"].append(f"{capped}/{n}")
            rows["verified by the critic"].append(f"{sum(bool(s.get('verified')) for s in v.scored)}/{n}")
        else:
            for key in ("tool calls per task (mean)", "tool calls that returned an error", "stopped by the step cap",
                        "verified by the critic"):
                rows[key].append("–")
        rows["JSON retries per task (mean)"].append(f"{statistics.fmean(s.get('retries') or 0 for s in v.scored):.2f}")
        answered = [s for s in v.scored if not s.get("refused") and s.get("json_valid")]
        good = sum(not (s.get("checks") or {}).get("bad_citations") for s in answered)
        rows["answers whose citations all name a source the run had"].append(f"{good}/{len(answered)}")
    return [[k, *v] for k, v in rows.items()]


def cost_rows(vs: list[Version]) -> list[list[str]]:
    rows: dict[str, list[str]] = {k: [] for k in (
        "tokens per question, mean", "tokens per question, p50", "latency p50 (s)", "latency p95 (s)",
        "chat replies from the reply cache (when run)")}
    for v in vs:
        tokens = [(s.get("prompt_tokens") or 0) + (s.get("completion_tokens") or 0) for s in v.scored]
        latency = [float(s.get("latency_s") or 0.0) for s in v.scored if not s.get("error")]
        rows["tokens per question, mean"].append(f"{statistics.fmean(tokens):,.0f}")
        rows["tokens per question, p50"].append(f"{statistics.median(tokens):,.0f}")
        live = live_latency(v)
        rows["latency p50 (s)"].append(f"{statistics.median(latency):.2f}" if live else "– (replayed)")
        rows["latency p95 (s)"].append(f"{pct(latency, 0.95):.2f}" if live else "– (replayed)")
        rows["chat replies from the reply cache (when run)"].append(cache_rate(v))
    return [[k, *v] for k, v in rows.items()]


def traced_rows() -> list[list[str]]:
    """The traced runs (scripts/trace_runs.py): what each answered; the dashboard tables are in eval/results/traces."""
    rows = []
    for run_id, what, mode in TRACED:
        path = TRACES / f"{run_id}.jsonl"
        try:
            recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError) as err:
            raise SourceError(f"{path.relative_to(REPO)}: {err}") from err
        if not recs:
            raise SourceError(f"{path.relative_to(REPO)}: no question")
        seconds = [r["seconds"] for r in recs]
        rows.append([f"`{run_id}`", what, mode, str(len(recs)), str(sum(r["verified"] for r in recs)),
                     str(sum(r["refused"] for r in recs)),
                     f"{statistics.median(seconds):.1f}" if mode == "live" else "– (cached)",
                     f"{statistics.fmean(r['tokens'] for r in recs):,.0f}",
                     f"{statistics.fmean(r['tool_calls'] for r in recs):.2f}"])
    return rows


def generated(vs: list[Version]) -> str:
    """The Markdown between the markers: the versions, then one section per measure, each chart with its table."""
    names = [v.label for v in vs]
    out = ["### Versions", ""] + table(["Version", "Run", "What it is"],
                                       [[v.label, f"`{v.run_id}`", v.what] for v in vs])
    out += ["", "### 1. Accuracy per version (correct answers / questions)", "",
            "![Accuracy per version by answer type](report/accuracy.svg)", ""]
    out += table(["Answer type", *names], accuracy_rows(vs))
    out += ["", "### 2. Failure taxonomy of the wrong answers", "",
            "![Wrong answers per version by failure class](report/failure_taxonomy.svg)", ""]
    out += table(["Failure class", *names], [[c, *[str(taxonomy_counts(v)[c]) for v in vs]] for c in CLASSES] +
                 [["**wrong answers**", *[f"**{sum(taxonomy_counts(v).values())}**" for v in vs]]])
    out += ["", "### 3. Task completion, tool calls, retries and citations", ""]
    out += table(["Measure", *names], agent_rows(vs))
    out += ["", "### 4. Cost: tokens, latency and the reply cache", "",
            "![Latency p50 and p95 per version](report/latency.svg)", ""]
    out += table(["Measure", *names], cost_rows(vs))
    out += ["", "### 5. Traced runs (Langfuse)", ""]
    out += table(["Run (tag run:<id>)", "Questions", "Replies", "n", "Verified", "Refused", "Seconds p50",
                  "Tokens (mean)",
                  "Tool calls (mean)"], traced_rows())
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------------- the charts
@dataclass(frozen=True)
class Chart:
    title: str
    categories: list[str]            # the x axis (versions)
    series: list[tuple[str, list[float]]]
    stacked: bool
    unit: str                        # "" or "%" or "s"


def charts(vs: list[Version]) -> dict[str, Chart]:
    names = [v.label for v in vs]
    acc = [(label, [100 * correct(tuple(s for s in v.scored if s["answer_type"] in types)) /
                    max(1, sum(s["answer_type"] in types for s in v.scored)) for v in vs])
           for label, types in ANSWER_TYPES]
    tax = [(c, [float(taxonomy_counts(v)[c]) for v in vs]) for c in CLASSES]
    live = [v for v in vs if live_latency(v)]           # replayed agent runs have no comparable latency
    lat = [(name, [q(sorted(float(s.get("latency_s") or 0.0) for s in v.scored if not s.get("error")))
                   for v in live]) for name, q in (("p50", statistics.median), ("p95", lambda x: pct(x, 0.95)))]
    return {"accuracy.svg": Chart("Accuracy by answer type (% correct)", names, acc, False, "%"),
            "failure_taxonomy.svg": Chart("Wrong answers by failure class", names, tax, True, ""),
            "latency.svg": Chart("Latency per question, seconds (runs made live)", [v.label for v in live], lat,
                                 False, "s")}


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    step = 10 ** math.floor(math.log10(value))
    return next(m * step for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if m * step >= value)


def layout(chart: Chart, width: float, height: float) -> dict[str, Any]:
    """Every shape of a chart in a width x height box, y downwards (shared by the SVG and the PDF)."""
    left, right, top, bottom = 44.0, 12.0, 30.0, 58.0
    plot_w, plot_h = width - left - right, height - top - bottom
    totals = [sum(vals[i] for _, vals in chart.series) for i in range(len(chart.categories))] if chart.stacked else \
        [max(vals[i] for _, vals in chart.series) for i in range(len(chart.categories))]
    top_value = 100.0 if chart.unit == "%" else _nice_max(max(totals or [0.0]))
    group = plot_w / len(chart.categories)
    bars, ticks = [], []
    for i, _ in enumerate(chart.categories):
        base = 0.0
        n = 1 if chart.stacked else len(chart.series)
        bar_w = group * 0.7 / n
        for j, (_, vals) in enumerate(chart.series):
            h = plot_h * vals[i] / top_value
            x = left + group * i + group * 0.15 + (0 if chart.stacked else j * bar_w)
            y = top + plot_h - (base if chart.stacked else 0.0) - h
            if h > 0:
                bars.append((x, y, bar_w, h, j))
            if chart.stacked:
                base += h
    for k in range(5):
        value = top_value * k / 4
        ticks.append((top + plot_h - plot_h * k / 4, f"{value:g}{chart.unit if chart.unit != 's' else ''}"))
    labels = [(left + group * (i + 0.5), top + plot_h + 14, c) for i, c in enumerate(chart.categories)]
    legend = [(left + i * (plot_w / len(chart.series)), height - 14, name, i) for i, (name, _) in enumerate(chart.series)]
    return {"plot": (left, top, plot_w, plot_h), "bars": bars, "ticks": ticks, "labels": labels, "legend": legend}


def _f(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(chart: Chart, width: float = 640, height: float = 320) -> str:
    """The chart as SVG: plain shapes and text, fixed precision, no timestamp (the same bytes on every machine)."""
    g = layout(chart, width, height)
    left, top, plot_w, plot_h = g["plot"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{_f(width)}" height="{_f(height)}" '
           f'viewBox="0 0 {_f(width)} {_f(height)}" font-family="Helvetica, Arial, sans-serif" font-size="11">',
           f'<rect width="{_f(width)}" height="{_f(height)}" fill="#ffffff"/>',
           f'<text x="{_f(width / 2)}" y="18" text-anchor="middle" font-size="13" font-weight="bold">'
           f'{_xml(chart.title)}</text>']
    for y, text in g["ticks"]:
        out.append(f'<line x1="{_f(left)}" y1="{_f(y)}" x2="{_f(left + plot_w)}" y2="{_f(y)}" stroke="#dddddd"/>')
        out.append(f'<text x="{_f(left - 6)}" y="{_f(y + 4)}" text-anchor="end">{_xml(text)}</text>')
    for x, y, w, h, j in g["bars"]:
        out.append(f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}" fill="{COLORS[j]}"/>')
    out.append(f'<line x1="{_f(left)}" y1="{_f(top + plot_h)}" x2="{_f(left + plot_w)}" y2="{_f(top + plot_h)}" '
               f'stroke="#333333"/>')
    for x, y, text in g["labels"]:
        out.append(f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="middle">{_xml(text)}</text>')
    for x, y, name, j in g["legend"]:
        out.append(f'<rect x="{_f(x)}" y="{_f(y - 9)}" width="10" height="10" fill="{COLORS[j]}"/>')
        out.append(f'<text x="{_f(x + 14)}" y="{_f(y)}">{_xml(name)}</text>')
    return "\n".join(out) + "\n</svg>\n"


# ----------------------------------------------------------------------------------------------- the PDF
WIN_ANSI = {"–": "-", "—": "-", "→": "->", "≥": ">=", "≤": "<=", "’": "'", "‘": "'", "“": '"', "”": '"',
            "…": "...", "×": "x", "§": "\xa7", "·": "\xb7", "²": "2", "≈": "~"}


def _pdf_text(text: str) -> str:
    plain = "".join(WIN_ANSI.get(ch, ch) for ch in text)
    plain = plain.encode("latin-1", "replace").decode("latin-1")
    return plain.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Pdf:
    """A small PDF writer: A4 pages of Helvetica / Courier text, lines and filled rectangles (the standard fonts,
    nothing embedded), no date and no compression: the same bytes for the same input on every platform."""

    W, H, MARGIN = 595.0, 842.0, 50.0

    def __init__(self) -> None:
        self.pages: list[list[str]] = []
        self.y = 0.0
        self.new_page()

    def new_page(self) -> None:
        self.pages.append([])
        self.y = self.H - self.MARGIN

    def need(self, height: float) -> None:
        if self.y - height < self.MARGIN:
            self.new_page()

    def text(self, x: float, y: float, s: str, size: float = 10, font: str = "F1", rgb: str = "0 0 0") -> None:
        self.pages[-1].append(f"BT {rgb} rg /{font} {_f(size)} Tf {_f(x)} {_f(y)} Td ({_pdf_text(s)}) Tj ET")

    def rect(self, x: float, y: float, w: float, h: float, color: str) -> None:
        r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
        self.pages[-1].append(f"{r:.3f} {g:.3f} {b:.3f} rg {_f(x)} {_f(y)} {_f(w)} {_f(h)} re f")

    def line(self, x1: float, y1: float, x2: float, y2: float, gray: float = 0.8) -> None:
        self.pages[-1].append(f"{gray:.2f} G 0.5 w {_f(x1)} {_f(y1)} m {_f(x2)} {_f(y2)} l S")

    def paragraph(self, s: str, size: float = 10, font: str = "F1", indent: float = 0.0, gap: float = 4) -> None:
        width = (self.W - 2 * self.MARGIN - indent) / (size * (0.6 if font == "F3" else 0.5))
        lines, line = [], ""
        for word in s.split():
            if line and len(line) + 1 + len(word) > width:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)
        for ln in lines:
            self.need(size + 3)
            self.y -= size + 3
            self.text(self.MARGIN + indent, self.y, ln, size, font)
        self.y -= gap

    def chart(self, chart: Chart, height: float = 230) -> None:
        width = self.W - 2 * self.MARGIN
        self.need(height + 8)
        g, top = layout(chart, width, height), self.y

        def y_of(v: float) -> float:
            return top - v

        x0 = self.MARGIN
        self.text(x0 + width / 2 - len(chart.title) * 2.6, y_of(18), chart.title, 11, "F2")
        left, ptop, plot_w, plot_h = g["plot"]
        for y, label in g["ticks"]:
            self.line(x0 + left, y_of(y), x0 + left + plot_w, y_of(y))
            self.text(x0 + left - 6 - len(label) * 4.4, y_of(y + 3), label, 8)
        for x, y, w, h, j in g["bars"]:
            self.rect(x0 + x, y_of(y + h), w, h, COLORS[j])
        self.line(x0 + left, y_of(ptop + plot_h), x0 + left + plot_w, y_of(ptop + plot_h), 0.2)
        for x, y, label in g["labels"]:
            self.text(x0 + x - len(label) * 2.5, y_of(y), label, 9)
        for x, y, name, j in g["legend"]:
            self.rect(x0 + x, y_of(y + 1), 8, 8, COLORS[j])
            self.text(x0 + x + 12, y_of(y), name, 8)
        self.y = top - height - 8

    def table(self, rows: list[list[str]]) -> None:
        clean = [[re.sub(r"[*`]", "", c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in clean if i < len(r)) for i in range(len(clean[0]))]
        size = max(5.5, min(8.0, (self.W - 2 * self.MARGIN) / (0.6 * (sum(widths) + 2 * len(widths)))))
        for n, r in enumerate(clean):
            text = "  ".join(c.ljust(widths[i]) for i, c in enumerate(r))
            self.need(size + 3)
            self.y -= size + 3
            self.text(self.MARGIN, self.y, text, size, "F3")
            if n == 0:
                self.line(self.MARGIN, self.y - 2, self.W - self.MARGIN, self.y - 2, 0.5)
        self.y -= 6

    def bytes(self) -> bytes:
        objects = ["<< /Type /Catalog /Pages 2 0 R >>", "",
                   "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
                   "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
                   "<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>"]
        kids = []
        streams: dict[int, bytes] = {}
        for n, ops in enumerate(self.pages, start=1):
            # uncompressed: zlib builds differ between platforms (zlib-ng on Windows), and --check compares bytes
            content = ("\n".join(ops) + f"\nBT 0.5 0.5 0.5 rg /F1 8 Tf {_f(self.W / 2 - 10)} 25 Td "
                       f"({n} / {len(self.pages)}) Tj ET").encode("latin-1")
            objects.append("")
            streams[len(objects)] = content
            objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_f(self.W)} {_f(self.H)}] "
                           f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> /Contents {len(objects)} 0 R >>")
            kids.append(f"{len(objects)} 0 R")
        objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
        out, offsets = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"), []
        for i, body in enumerate(objects, start=1):
            offsets.append(len(out))
            if i in streams:
                data = streams[i]
                out += f"{i} 0 obj\n<< /Length {len(data)} >>\nstream\n".encode() + data + \
                    b"\nendstream\nendobj\n"
            else:
                out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
        xref = len(out)
        out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
        out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
        out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        return bytes(out)


def pdf(markdown: str, chart_specs: dict[str, Chart]) -> bytes:
    """The report's Markdown rendered as a PDF: headings, paragraphs, bullets, tables and the charts it links."""
    doc = Pdf()
    rows: list[list[str]] = []

    def flush_table() -> None:
        if rows:
            doc.table(rows)
            rows.clear()

    lines = [raw.rstrip() for raw in markdown.splitlines()]

    def following(n: int) -> str:
        """The next line with content after line n (a heading keeps its chart or table on its page)."""
        return next((x for x in lines[n + 1:] if x and not x.startswith("<!--")), "")

    for n, line in enumerate(lines):
        if line.startswith("|"):
            if not re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", line):
                rows.append([c.strip() for c in line.strip("|").split("|")])
            continue
        flush_table()
        if not line or line in (START, END) or line.startswith("<!--"):
            continue
        if m := re.fullmatch(r"!\[[^\]]*\]\(report/([^)]+)\)", line):
            doc.chart(chart_specs[m.group(1)])
        elif m := re.fullmatch(r"(#{1,4})\s+(.*)", line):
            size = {1: 18.0, 2: 14.0, 3: 12.0, 4: 11.0}[len(m.group(1))]
            nxt = following(n)
            doc.need(size + 24 + (238 if nxt.startswith("![") else 60 if nxt.startswith("|") else 0))
            doc.y -= 8
            doc.paragraph(re.sub(r"[*`]", "", m.group(2)), size, "F2", gap=4)
        elif m := re.fullmatch(r"(\s*)[-*]\s+(.*)", line):
            doc.paragraph("- " + re.sub(r"[*`]", "", m.group(2)), 10, indent=10 + 2 * len(m.group(1)), gap=1)
        else:
            doc.paragraph(re.sub(r"[*`]", "", line), 10)
    flush_table()
    return doc.bytes()


# ----------------------------------------------------------------------------------------------- the CLI
def build(report_text: str) -> tuple[str, dict[Path, bytes]]:
    """The report with its generated block replaced, and every file it consists of (Markdown, SVGs, PDF)."""
    if START not in report_text or END not in report_text:
        raise SourceError(f"{REPORT.relative_to(REPO)} lost its GENERATED markers")
    vs = versions()
    head, rest = report_text.split(START, 1)
    text = head + START + "\n" + generated(vs) + END + rest.split(END, 1)[1]
    specs = charts(vs)
    files = {REPORT: text.encode("utf-8")}
    files |= {CHART_DIR / name: svg(chart).encode("utf-8") for name, chart in specs.items()}
    files[PDF] = pdf(text, specs)
    return text, files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if a committed file is stale")
    args = ap.parse_args(argv)
    try:
        _, files = build(REPORT.read_text(encoding="utf-8"))
    except (OSError, SourceError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    stale = [p for p, data in files.items() if not p.exists() or p.read_bytes() != data]
    if args.check:
        for p in stale:
            print(f"stale: {p.relative_to(REPO)} (run python eval/report.py)", file=sys.stderr)
        return 1 if stale else 0
    for p, data in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    print("wrote " + ", ".join(str(p.relative_to(REPO)) for p in files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
