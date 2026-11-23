"""scripts/week2_notes.py (W2-14): the Week 2 notes are generated from committed results and kept current. Offline:
every source is a committed file (no key, no data/)."""
import csv

import pytest

import scripts.week2_notes as w2


def test_the_committed_notes_are_current():
    """A change to the agent report, the demo, the injection results, the smoke tests or the ADR log without
    re-running the script fails here."""
    assert w2.main(["--check"]) == 0


def test_the_note_compares_every_run_by_short_name():
    header, rows = w2.agent_rows()
    assert header == ["Answer type", "Qwen3-8B, passages (B0)", "S1 few-shot", "S2 reasoning + 5-sample vote",
                      "S3 decomposition", "A1 agent"]
    by = {r[0]: r for r in rows}
    assert by["Short: numeric"][-1] == "42/48 (88%)" and by["Right only with the run / only with B0 (sign test)"][1] == "–"
    assert "A1_qwen3" not in w2.note_block() and "M1_qwen3" not in w2.note_block()


def test_the_demo_summary_counts_the_ten_queries():
    rows = dict(w2.demo_summary()[1])
    assert rows["queries that met their expectation"] == "10/10" and rows["under the 60 s limit"] == "10/10"
    assert rows["verdicts"].count(";") == 2                    # a verdict holding a comma stays one item
    assert len(w2.demo_rows()[1]) == 10


def test_a_later_run_in_the_agent_report_does_not_change_the_week2_note(monkeypatch, tmp_path):
    """Week 3 may score A5 into the same report: the Week 2 note keeps its five runs."""
    import re

    def widen(line):                                           # one more column in every table: run A5_x
        if not line.startswith("|"):
            return line
        if re.fullmatch(r"\|[-:| ]+\|", line.strip()):
            return line + "---|"
        return line + (" A5_x |" if w2.A1 in line else " 9 |")
    report = tmp_path / "agent_baselines.md"
    report.write_text("\n".join(widen(x) for x in w2.AGENT_REPORT.read_text(encoding="utf-8").splitlines()),
                      encoding="utf-8")
    monkeypatch.setattr(w2, "AGENT_REPORT", report)
    assert "A5_x" in w2.table(report, "1. Accuracy by answer type").header           # the report has A5 ...
    header, rows = w2.agent_rows()
    assert header[-1] == "A1 agent" and len(header) == 6 and all(len(r) == 6 for r in rows)   # ... the note not


def test_only_week2_adrs_are_listed_and_every_one_is_accepted():
    ids = [r[0] for r in w2.week2_adrs()]
    assert ids == [f"ADR-0{n}" for n in range(24, 36)]                 # ADR-036 onwards are Week 3's
    assert all(r[3] == "Accepted" for r in w2.week2_adrs())            # every Week 2 PR is merged


def test_a_missing_or_changed_source_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(w2, "DEMO", tmp_path / "demo.jsonl")
    with pytest.raises(w2.SourceError, match="is missing"):
        w2.demo_summary()
    (tmp_path / "demo.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(w2.SourceError, match="is empty"):
        w2.demo_summary()
    (tmp_path / "demo.jsonl").write_text('{"id": "demo-01"}\n', encoding="utf-8")
    with pytest.raises(w2.SourceError, match="line 1: no category"):
        w2.demo_summary()
    (tmp_path / "demo.jsonl").write_text("not json\n", encoding="utf-8")
    with pytest.raises(w2.SourceError, match="demo.jsonl"):
        w2.demo_summary()
    monkeypatch.setattr(w2, "INJECTION", tmp_path / "inj.md")
    (tmp_path / "inj.md").write_text("# no table\n", encoding="utf-8")
    with pytest.raises(w2.SourceError, match="0 tables headed 'Measure'"):
        w2.injection_rows()
    monkeypatch.setattr(w2, "AGENT_REPORT", tmp_path / "agent.md")
    (tmp_path / "agent.md").write_text("### 1. Accuracy by answer type\n\nno table\n", encoding="utf-8")
    with pytest.raises(w2.SourceError, match="0 tables under"):
        w2.agent_rows()


def test_the_hand_written_text_is_kept_and_the_markers_are_required(tmp_path):
    path = tmp_path / "note.md"
    path.write_text(f"# Note\nintro\n{w2.START}\nold\n{w2.END}\nafter\n", encoding="utf-8")
    assert w2.fill(path, "new\n", "T") == f"# Note\nintro\n{w2.START}\nnew\n{w2.END}\nafter\n"
    for broken in ("no markers", f"{w2.END}\n{w2.START}\n"):
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(w2.SourceError, match="GENERATED markers"):
            w2.fill(path, "new\n", "T")


def test_the_week2_markers_differ_from_week1s():
    """The two scripts must never rewrite each other's notes."""
    from scripts import week1_notes as w1
    assert w2.START != w1.START and w2.NOTE != w1.NOTE


# The figures the hand-written text cites, each with the committed line it must still match: a re-run or a regraded
# answer that changes a source fails here until the prose is updated (the tables are covered by --check).
R = "eval/results/"
AGENT = R + "agent_baselines.md"
CITED = [
    ("42 of 48 numeric questions right against 16", AGENT, "| Short: numeric | 16/48 = 33% [22%–47%]"),
    ("42 of 48 numeric questions right against 16", AGENT, "42/48 = 88% [75%–94%] |"),
    ("11 of 12 period comparisons against 0", AGENT, "| period_comparison | 0/12 = 0% [0%–24%]"),
    ("11 of 12 period comparisons against 0", AGENT, "11/12 = 92% [65%–99%] |"),
    ("44 questions where B0 is wrong against\n  8", AGENT, "44 / 8 (p < 0.01, n = 102) |"),
    ("0 of 81 answers with an\n  unsupported number; 4 of 81", AGENT, "| 19/88 | 0/81 |"),
    ("0 of 81 answers with an\n  unsupported number; 4 of 81", AGENT, "| 19/88 | 4/81 |"),
    ("5 of 12 against B0's 8", AGENT, "| Summary | 8/12 = 67% [39%–86%]"),
    ("5 of 12 against B0's 8", AGENT, "5/12 = 42% [19%–68%] |"),
    ("(agreement kappa 0.29)", AGENT, "| final grade vs judge: Cohen's kappa | 1.00 | 0.29 | 0.12 | 0.10 | 0.29 |"),
    ("higher than the graders on 4 of\n  12 and lower on 2", AGENT, "higher on 4, lower on 2"),
    ("median 5.0 s and 12.4 s", AGENT, "| 8.2 / 16.1 | 5.0 / 12.4 |"),
    ("3.0k tokens and 1.8\n  model calls", AGENT, "| 7397 | 3000 |"),
    ("3.0k tokens and 1.8\n  model calls", AGENT, "| 5.0 | 2.0 | 1.8 |"),
    ("no question stopped by a limit", AGENT, "| stopped by a limit (time, steps, tokens) | 0 |"),
    ("the slowest in 20.3 s", R + "demo/W2-07_demo.md", "| 20.28 |"),
    ("(10 of 12 did before)", R + "injection_results.md", "| planted sentence reached the composer as written | 10/12 | 0/12 |"),
    ("from 24 to 31 of 31", "docs/decisions.md", "first call accepted by the tool 24/31 -> 31/31"),
    ("verified all 102 answers and caught none of the 16 wrong ones", AGENT, "| verified by its critic | 102/102 |"),
    ("verified all 102 answers and caught none of the 16 wrong ones", AGENT,
     "| correct among the verified (decided) | 86/102 |"),
]
DETAILED_CITED = [
    ("refusal decisions 48 of 49", R + "planner_runs/P8.md",              # P8's own summary (W3-04 ran P9-P12)
     "| refusal decision correct (refused or not, and the rule) | 48/49"),
    ("21 of 22, kappa 0.91", AGENT, "| 14/14 | 10/12 | 21/22 |"),
    ("21 of 22, kappa 0.91", AGENT, "| 1.00 | 0.67 | 0.91 |"),
    ("10 of 10 met their expectation", R + "demo/W2-07_demo.md", "# W2-07 demo: 10/10 met their expectation"),
    ("it caught 27/60 (first\n  set, frozen)", R + "injection_results.md", "With the detector frozen it caught **27/60**"),
    ("then 28/60 and 29/60 on a second set", R + "injection_results.md", "caught **28/60** and flagged **1/60**"),
    ("then 28/60 and 29/60 on a second set", R + "injection_results.md", "moved it to 29/60 caught, 1/60"),
]


@pytest.mark.parametrize("phrase, source, line", CITED)
def test_every_figure_the_note_cites_is_in_its_source(phrase, source, line):
    assert phrase in w2.NOTE.read_text(encoding="utf-8"), f"the note no longer says {phrase!r}"
    assert line in (w2.REPO / source).read_text(encoding="utf-8"), f"{source} no longer has {line!r}"


@pytest.mark.parametrize("phrase, source, line", DETAILED_CITED)
def test_every_figure_the_detailed_text_cites_is_in_its_source(phrase, source, line):
    assert phrase in w2.DETAILED.read_text(encoding="utf-8"), f"the detailed note no longer says {phrase!r}"
    assert line in (w2.REPO / source).read_text(encoding="utf-8"), f"{source} no longer has {line!r}"


def test_the_second_rounds_agreement_the_note_cites_is_the_sheets():
    """'60 grades, kappa 0.89, 133 labels': computed from the committed grading sheet, rows of S1-S3 and A1."""
    from eval import grading as gr
    from eval import score_answers as sa
    assert "(60 grades, kappa\n0.89, 133 labels)" in w2.NOTE.read_text(encoding="utf-8")
    run_of = {r["key"]: r["run_id"] for r in csv.DictReader((sa.GRADING_DIR / sa.KEY_NAME).open(encoding="utf-8"))}
    second = [r for r in gr.load(gr.SHEET) if run_of[r["key"]][0] == "S" or run_of[r["key"]].startswith("A1_")]
    pairs = [(int(r["grader_1"]), int(r["grader_2"])) for r in second if r["task"] == sa.GRADE]
    labels = [r for r in second if r["task"] == sa.LABEL and r["failure_label"]]
    assert len(pairs) == 60 and round(gr.kappa(pairs), 2) == 0.89 and len(labels) == 133
