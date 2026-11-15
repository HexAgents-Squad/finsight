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
