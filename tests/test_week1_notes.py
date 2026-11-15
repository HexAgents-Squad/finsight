"""scripts/week1_notes.py (W1-11): the Week 1 notes are generated from committed results and kept current."""
import pytest

import scripts.week1_notes as wn


def test_the_committed_notes_are_current():
    """A change to a results report, a question set or the ADR log without re-running the script fails here."""
    assert wn.main(["--check"]) == 0


def test_tables_are_found_under_their_heading():
    text = "# Title\n\n### 1. Accuracy\n\n| A | B |\n|---|---|\n| x | 1 |\n| y | 2 |\n\nOther text\n| C |\n|---|\n| z |\n"
    first, second = wn.tables(text)
    assert (first.heading, first.header, first.rows) == ("1. Accuracy", ["A", "B"], [["x", "1"], ["y", "2"]])
    assert (second.heading, second.row("z")) == ("Other text", ["z"])
    with pytest.raises(wn.SourceError, match="no row 'q'"):
        first.row("q")
    with pytest.raises(wn.SourceError, match="no column 'C'"):
        first.column("C")


@pytest.mark.parametrize("cell, out", [("16/48 = 33% [22%–47%]", "16/48 (33%)"),
                                       ("6/8 = 75% [41%–93%], 2 pending", "6/8 (75%), 2 pending"),
                                       ("pending (12)", "pending (12)"), ("20/74", "20/74")])
def test_cells_are_shortened_without_losing_pending_notes(cell, out):
    assert wn.short(cell) == out


def test_a_missing_or_changed_source_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(wn, "RESULTS", tmp_path)
    with pytest.raises(wn.SourceError, match="is missing"):
        wn.note_block()
    (tmp_path / "model_comparison.md").write_text("### 1. Accuracy by answer type\n\nno table here\n", encoding="utf-8")
    with pytest.raises(wn.SourceError, match="0 tables under"):
        wn.model_rows(tmp_path / "model_comparison.md")


def test_the_hand_written_text_is_kept_and_the_markers_are_required(tmp_path):
    path = tmp_path / "note.md"
    path.write_text(f"# Note\nintro\n{wn.START}\nold\n{wn.END}\nafter\n", encoding="utf-8")
    assert wn.fill(path, "new\n", "T") == f"# Note\nintro\n{wn.START}\nnew\n{wn.END}\nafter\n"
    path.write_text("no markers", encoding="utf-8")
    with pytest.raises(wn.SourceError, match="lost its GENERATED markers"):
        wn.fill(path, "new\n", "T")


def test_only_week1_adrs_are_listed_in_order():
    rows = wn.week1_adrs()
    assert [r[0] for r in rows] == sorted(r[0] for r in rows) and "ADR-015" not in [r[0] for r in rows]
    assert {"ADR-016", "ADR-017", "ADR-018", "ADR-023"} <= {r[0] for r in rows}
    assert all(len(r[2]) <= 160 for r in rows)


def test_the_note_names_the_runs_without_their_ids():
    block = wn.note_block()
    assert "Qwen3-8B, passages (B0)" in block and "M1_qwen3-8b" not in block
    assert "| + financial statements first (default, ADR-017) | 0.986 |" in block


# The figures the hand-written text cites, each with the committed line it must still match: a re-run that changes
# a source fails here until the prose is updated (the generated tables are covered by test_the_committed_notes...).
RESULTS = "eval/results/"
CITED = [
    ("17 of 18", RESULTS + "prompting_baselines.md", "| all in the passages | 12/18 = 67%"),
    ("17 of 18", RESULTS + "prompting_baselines.md", "| 17/18 = 94%"),
    ("6 better / 0 worse, p = 0.03", RESULTS + "prompting_baselines.md", "| Refusal & not reported | 6 / 0 (p = 0.03, n = 32)"),
    ("20 of 74", RESULTS + "prompting_baselines.md", "| unsupported numbers | 20/74 | 15/72 | 4/85 | 19/88 |"),
    ("~23 s per question", RESULTS + "prompting_baselines.md", "| 22.9 / 31.7 |"),
    ("45 of 70", RESULTS + "model_comparison.md", "| unsupported numbers | 20/74 | 15/66 | 45/70 |"),
    ("from 0.774 to 0.986", RESULTS + "retrieval_summary.md", "| hybrid | question+period+statement | 0.808 | 0.986 |"),
    ("0.630 to 0.788 (p = 0.011", RESULTS + "embedding_comparison.md", "| 0.315 / 0.630 / 0.822 / 0.487 | 0.336 / 0.788 / 0.877 / 0.576 | 18 / 5 / 50 | 0.0106 |"),
    ("16,817", "docs/decisions.md", "16,817 lines removed"),
    ("0.747 → 0.774", "docs/decisions.md", "0.747 -> 0.774"),
    ("7 of 12", RESULTS + "model_comparison.md", "| graders' exact agreement (independent grades) | 7/12 |"),
    ("kappa 0.17", RESULTS + "model_comparison.md", "| graders' Cohen's kappa | 0.17 |"),
]


@pytest.mark.parametrize("phrase, source, line", CITED)
def test_every_figure_the_text_cites_is_in_its_source(phrase, source, line):
    note = wn.NOTE.read_text(encoding="utf-8")
    assert phrase in note, f"the note no longer says {phrase!r}"
    assert line in (wn.REPO / source).read_text(encoding="utf-8"), f"{source} no longer has {line!r}"


DETAILED_CITED = [
    ("arithmetic slips (10), over-refusals (10), invented numbers (8)\n  and wrong line items (7)", RESULTS + "model_comparison.md",
     "| arithmetic_slip | 10 |"),
    ("closed book, invented numbers lead (28", RESULTS + "model_comparison.md", "| invented_number | 8 | 14 | 28 |"),
    ("valid JSON for 72 of 102", RESULTS + "model_comparison.md", "| valid B0 JSON | 102/102 | 72/102 | 102/102 |"),
    ("under-refusal** (S2 7, S3 8)", RESULTS + "prompting_baselines.md", "| under-refusals | 3 | 0 | 7 | 8 |"),
    ("B0 4 of its 6 misses", RESULTS + "prompting_baselines.md", "(4 of its 6 misses)"),
]


@pytest.mark.parametrize("phrase, source, line", DETAILED_CITED)
def test_every_figure_the_detailed_text_cites_is_in_its_source(phrase, source, line):
    assert phrase in wn.DETAILED.read_text(encoding="utf-8"), f"the detailed note no longer says {phrase!r}"
    assert line in (wn.REPO / source).read_text(encoding="utf-8"), f"{source} no longer has {line!r}"


def test_markers_in_the_wrong_order_are_an_error(tmp_path):
    path = tmp_path / "note.md"
    path.write_text(f"{wn.END}\n{wn.START}\n", encoding="utf-8")
    with pytest.raises(wn.SourceError, match="GENERATED markers"):
        wn.fill(path, "x\n", "T")
