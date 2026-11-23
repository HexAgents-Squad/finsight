"""W3-09: eval/report.py, the evaluation report generated from committed results (offline; no key, no data/)."""
from __future__ import annotations

import re

import pytest

from eval import report as rp


def test_the_committed_report_its_charts_and_pdf_are_current():
    """Like the weekly notes: a change to a committed result without `python eval/report.py` fails here."""
    assert rp.main(["--check"]) == 0


@pytest.mark.parametrize(("row", "expected"), [
    ({"failure_label": "over_refusal"}, "refusal error"),
    ({"suggested_label": "wrong_refusal_rule"}, "refusal error"),
    ({"failure_label": "invented_number"}, "hallucination"),
    ({"failure_label": "ignored_not_reported"}, "hallucination"),
    ({"failure_label": "malformed_output"}, "tool-call error"),
    ({"failure_label": "wrong_line_item", "gold_figures_in_pack": False}, "retrieval miss"),
    ({"failure_label": "wrong_line_item", "gold_figures_in_pack": True}, "reasoning error"),
    ({"failure_label": "incomplete_summary", "gold_figures_in_pack": None}, "reasoning error"),
    ({"failure_label": "arithmetic_slip", "gold_figures_in_pack": False}, "reasoning error"),
    ({"failure_label": "", "suggested_label": "wrong_company"}, "reasoning error"),
])
def test_each_failure_label_maps_to_the_plans_class(row, expected):
    """The graders' label first, else the scorer's suggestion; a retrieval miss needs the figures to be absent."""
    assert rp.taxonomy_class(row) == expected


def test_the_versions_are_the_plans_path_and_every_answer_is_decided():
    vs = rp.versions()
    assert [v.label for v in vs] == ["B0", "S1", "A1", "A2", "A3", "A4", "A5"]
    assert all(len(v.scored) == 102 and all(s["correct"] is not None for s in v.scored) for v in vs)


def test_a_pending_grade_or_a_missing_file_is_named(monkeypatch):
    real = rp._scores

    def pending(run_id):
        rows = real(run_id)
        return (dict(rows[0], correct=None), *rows[1:]) if run_id.startswith("A5_") else rows
    monkeypatch.setattr(rp, "_scores", pending)
    with pytest.raises(rp.SourceError, match="waiting for a grade"):
        rp.versions()
    monkeypatch.setattr(rp, "TRACED", (("no-such-run", "x", "live"),))
    with pytest.raises(rp.SourceError, match="no-such-run"):
        rp.traced_rows()


def test_a_replayed_agent_run_has_no_latency():
    """An agent run's latency is graph wall time, in which a cached reply takes none (A2-A4 were replays)."""
    agent = {"context": "agent"}
    assert rp.live_latency(rp.Version("A1", "a", "x", ({"cache_hits": 0},), agent))
    assert not rp.live_latency(rp.Version("A3", "a", "x", ({"cache_hits": 2},), agent))
    assert rp.live_latency(rp.Version("B0", "m", "x", ({"cached": True},), {"context": "rag"}))


CHART = rp.Chart("Wrong answers", ["B0", "A5"], [("retrieval miss", [7.0, 2.0]), ("hallucination", [9.0, 0.0])],
                 True, "")


def test_a_stacked_chart_stacks_and_scales_to_a_round_maximum():
    g = rp.layout(CHART, 640, 320)
    assert [t for _, t in g["ticks"]] == ["0", "4", "8", "12", "16"]          # four round steps reaching 16
    heights = {}
    for x, _, _, h, _ in g["bars"]:
        heights[round(x)] = heights.get(round(x), 0) + h
    assert len(g["bars"]) == 3 and max(heights.values()) == pytest.approx(g["plot"][3])    # a zero draws nothing
    assert [rp._nice_max(v) for v in (52, 12.5, 7, 0)] == [60, 16, 8, 4]       # every tick a round number


def test_the_svg_is_the_same_bytes_every_time_and_escapes_text():
    chart = rp.Chart("A & B <x>", ["B0"], [("p50", [2.5])], False, "s")
    one, two = rp.svg(chart), rp.svg(chart)
    assert one == two and "A &amp; B &lt;x&gt;" in one and one.startswith("<svg") and one.endswith("</svg>\n")


def test_the_pdf_renders_the_markdown_and_is_reproducible():
    md = ("# Title (draft)\n\nA paragraph with \\ a backslash and – a dash.\n\n- a bullet\n\n"
          "| a | b |\n|---|---|\n| 1 | 2 |\n\n![chart](report/c.svg)\n")
    one, two = rp.pdf(md, {"c.svg": CHART}), rp.pdf(md, {"c.svg": CHART})
    assert one == two and one.startswith(b"%PDF-1.4") and one.rstrip().endswith(b"%%EOF")
    streams = re.findall(rb"stream\n(.*?)\nendstream", one, re.S)            # uncompressed: the same on every OS
    text = b"".join(streams).decode("latin-1")
    assert "Title \\(draft\\)" in text and "\\\\ a backslash" in text and "- a dash" in text
    assert "a  b" in text and "1  2" in text and "Wrong answers" in text and " re f" in text


def test_a_report_without_its_markers_is_refused():
    with pytest.raises(rp.SourceError, match="GENERATED markers"):
        rp.build("# Evaluation report\n")


def test_hard_wrapped_lines_are_one_paragraph_and_a_bullets_continuation_joins_it():
    """Review: each Markdown source line was its own PDF paragraph, so the report's wrapped sentences broke."""
    text = rp.pdf("First line of a sentence\nand its second line.\n\n- a bullet\n  continued here\n- another\n", {})
    assert b"(First line of a sentence and its second line.) Tj" in text
    assert b"(- a bullet continued here) Tj" in text and b"(- another) Tj" in text


def test_odd_sources_are_named_errors_not_crashes(monkeypatch, tmp_path):
    """Review: an unknown chart link, a traced record without a field and no live run each raise SourceError."""
    with pytest.raises(rp.SourceError, match="not a generated chart"):
        rp.pdf("![x](report/missing.svg)\n", {})
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "t.jsonl").write_text('{"id": "q1", "verified": true}\n', encoding="utf-8")
    monkeypatch.setattr(rp, "TRACES", traces)
    monkeypatch.setattr(rp, "TRACED", (("t", "x", "live"),))
    with pytest.raises(rp.SourceError, match="lacks refused, seconds, tokens, tool_calls"):
        rp.traced_rows()
    replayed = [rp.Version("A3", "a", "x", ({"cache_hits": 1, "answer_type": "summary", "correct": True},),
                           {"context": "agent"})]
    with pytest.raises(rp.SourceError, match="no version was run live"):
        rp.charts(replayed)


def test_the_p95_is_the_scorers():
    """Review: the report said 12.50 s for A1 where agent_baselines.md says 12.4 (nearest rank vs interpolated)."""
    from eval.score_answers import percentile
    values = [float(v) for v in range(1, 102)]
    assert rp.p95(values) == percentile(values, 95)
