"""eval/grading.py offline, on the committed grading sheet and key (copied to a temporary folder; the repo's files
are never written)."""
import csv
import shutil

import pytest

import eval.grading as gr
import eval.score_answers as sa


@pytest.fixture
def sheet(tmp_path):
    for name in (sa.SHEET_NAME, sa.KEY_NAME):
        shutil.copy(sa.GRADING_DIR / name, tmp_path / name)
    return tmp_path / sa.SHEET_NAME


def rows_of(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(path, rows, excel=False):
    """Write rows as a grader's spreadsheet would: Excel adds a byte-order mark and CRLF line ends."""
    with path.open("w", encoding="utf-8-sig" if excel else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sa.SHEET_COLUMNS, lineterminator="\r\n" if excel else "\n")
        w.writeheader()
        w.writerows(rows)


def fill(sheet, grader, grade_for, label_for=lambda r: r["suggested_label"] or "incomplete_summary", notes=None,
         excel=False):
    """A grader's finished copy: their grade column on every grade row, a label on their own label rows."""
    rows, split = rows_of(sheet), gr.split_row(rows_of(sheet))
    for n, r in enumerate(rows, start=2):
        if r["task"] == sa.GRADE:
            r[f"grader_{grader}"] = grade_for(n)
        elif r["task"] == sa.LABEL and gr.owner(n, split) == grader:
            r["failure_label"] = label_for(r)
        r["note"] = (notes or {}).get(n, "")
    path = sheet.with_name(f"g{grader}.csv")
    save(path, rows, excel)
    return path


def grade_numbers(sheet):
    return [n for n, r in enumerate(rows_of(sheet), start=2) if r["task"] == sa.GRADE]


# --- the split -------------------------------------------------------------------------------------------------

def test_the_label_rows_are_split_in_half_in_sheet_order():
    rows = [{"task": t} for t in ("label", "grade", "label", "label", "grade", "label", "label")]
    assert gr.split_row(rows) == 7                      # labels on rows 2, 4, 5 | 7, 8: grader 1 takes three
    assert [gr.owner(n, 7) for n in (2, 4, 5, 7, 8)] == [1, 1, 1, 2, 2]
    assert gr.split_row([{"task": "grade"}, {"task": "label"}]) == 4      # a single label row: grader 1's


def test_the_committed_sheet_splits_95_and_94(sheet):
    rows = rows_of(sheet)
    split = gr.split_row(rows)
    labels = [n for n, r in enumerate(rows, start=2) if r["task"] == sa.LABEL]
    assert (sum(n < split for n in labels), sum(n >= split for n in labels)) == (95, 94)


# --- copies ----------------------------------------------------------------------------------------------------

def test_copies_are_the_untouched_sheet_with_assignments(sheet, tmp_path):
    out = tmp_path / "copies"
    lines = gr.copies(sheet, ["Chandra", "Amresh"], out)
    for name in ("Chandra", "Amresh"):
        assert (out / f"grading_sheet_{name}.csv").read_bytes() == sheet.read_bytes()
    text = (out / "assignments.txt").read_text(encoding="utf-8")
    assert text == "\n".join(lines) and "Grader 1 = Chandra" in text and "column grader_2 on all 32" in text
    assert "spreadsheet rows 2-114" in text and "spreadsheet rows 115-220" in text


@pytest.mark.parametrize("names, message", [(["Chandra"], "exactly two"), (["A", "a"], "exactly two"),
                                            (["Chandra", "../x"], "file names")])
def test_copies_need_two_safe_names(sheet, tmp_path, names, message):
    with pytest.raises(gr.GradingError, match=message):
        gr.copies(sheet, names, tmp_path / "c")


def test_copies_of_a_graded_sheet_are_refused(sheet, tmp_path):
    rows = rows_of(sheet)
    rows[grade_numbers(sheet)[0] - 2]["grader_1"] = "2"
    save(sheet, rows)
    with pytest.raises(gr.GradingError, match="would show them"):
        gr.copies(sheet, ["Chandra", "Amresh"], tmp_path / "c")


# --- handout ---------------------------------------------------------------------------------------------------

def test_the_handout_has_every_grade_question_with_passages_and_names_no_model(sheet):
    text = gr.handout(sheet, sheet.with_name(sa.KEY_NAME), sa.RESULTS_DIR / sa.GOLD_PACK, sa.GOLD_FILE)
    questions = {r["question"] for r in rows_of(sheet) if r["task"] == sa.GRADE}
    assert all(f"## {q}" in text for q in questions) and text.count("\n## ") == len(questions)
    lowered = text.lower()
    assert not any(word in lowered for word in ("qwen", "llama", "m1_", "m2_", "m3_", "closed_think", "rag_think"))
    first = text.split("\n## ")[1]                       # gold-source passages come first
    assert first.split("**Gold answer:**")[1].split("\n**", 1)[1].startswith("★")


# --- merge -----------------------------------------------------------------------------------------------------

def test_merge_sets_final_where_grades_agree_and_lists_the_rest(sheet):
    numbers = grade_numbers(sheet)
    g1 = fill(sheet, 1, lambda n: "2", notes={numbers[0]: "overturn? the figure is right"}, excel=True)
    g2 = fill(sheet, 2, lambda n: "1" if n == numbers[1] else "2", notes={numbers[0]: "fine"})
    merged, rep = gr.merge(sheet, g1, g2)
    by_row = dict(enumerate(merged, start=2))
    assert (rep["pairs"], rep["agree"], rep["missing"], rep["warnings"]) == (32, 31, [], [])
    assert by_row[numbers[0]]["final"] == "2" and by_row[numbers[1]]["final"] == ""
    assert rep["disagree"][0].startswith(f"row {numbers[1]}: grader 1 = 2, grader 2 = 1")
    assert by_row[numbers[0]]["note"] == "G1: overturn? the figure is right | G2: fine"
    assert rep["overturn"] == [f"row {numbers[0]}: G1: overturn? the figure is right | G2: fine"]
    split = gr.split_row(rows_of(sheet))
    labels = {n: r for n, r in by_row.items() if r["task"] == sa.LABEL}
    assert all(r["failure_label"] for r in labels.values()) and all(r["final"] == "" for r in labels.values())
    path = sheet.with_name("merged.csv")
    path.write_text(gr.csv_text(merged), encoding="utf-8", newline="")
    assert len(sa.read_sheet(path)) == len(merged)                         # the scorer accepts the result
    assert split in labels


def test_the_merged_sheet_differs_from_the_repo_sheet_only_in_grader_cells(sheet):
    g1, g2 = fill(sheet, 1, lambda n: "0"), fill(sheet, 2, lambda n: "0")
    merged, _ = gr.merge(sheet, g1, g2)
    original = rows_of(sheet)
    for a, b in zip(merged, original, strict=True):
        assert {c: a[c] for c in sa.SHEET_COLUMNS if c not in gr.GRADER_COLUMNS + ("note",)} == \
               {c: b[c] for c in sa.SHEET_COLUMNS if c not in gr.GRADER_COLUMNS + ("note",)}


def test_a_label_comes_from_the_rows_owner_only(sheet):
    split = gr.split_row(rows_of(sheet))
    g1 = fill(sheet, 1, lambda n: "2")
    rows = rows_of(sheet.with_name("g1.csv"))
    theirs = next(n for n, r in enumerate(rows, start=2) if r["task"] == sa.LABEL and n >= split)
    rows[theirs - 2]["failure_label"] = "invented_number"                 # grader 1 labels grader 2's row
    save(g1, rows)
    g2 = fill(sheet, 2, lambda n: "2", label_for=lambda r: "wrong_company")
    merged, rep = gr.merge(sheet, g1, g2)
    assert merged[theirs - 2]["failure_label"] == "wrong_company"
    assert rep["warnings"] == [f"row {theirs}: a label from the grader who does not own the row was ignored"]


def test_missing_grades_and_labels_are_listed(sheet):
    numbers = grade_numbers(sheet)
    g1 = fill(sheet, 1, lambda n: "" if n == numbers[2] else "1", label_for=lambda r: "")
    g2 = fill(sheet, 2, lambda n: "1")
    merged, rep = gr.merge(sheet, g1, g2)
    assert f"row {numbers[2]}: grade missing (grader 1: -, grader 2: 1)" in rep["missing"]
    assert sum("no failure_label" in m for m in rep["missing"]) == 95 and merged[numbers[2] - 2]["final"] == ""


def test_a_re_formatted_answer_cell_is_a_warning_and_the_repo_text_is_kept(sheet):
    g1, g2 = fill(sheet, 1, lambda n: "2"), fill(sheet, 2, lambda n: "2")
    rows = rows_of(g2)
    rows[0]["model_answer"] = rows[0]["model_answer"].replace(",", "") + " "  # what a spreadsheet may do
    save(g2, rows)
    merged, rep = gr.merge(sheet, g1, g2)
    assert rep["warnings"] == ["grader 2: text re-formatted or edited on rows [2] (the repo's text is kept)"]
    assert merged[0]["model_answer"] == rows_of(sheet)[0]["model_answer"]


@pytest.mark.parametrize("damage, message", [
    (lambda rows: rows.sort(key=lambda r: r["question"]), "re-ordered"),
    (lambda rows: rows.pop(5), "removed"),
    (lambda rows: rows[0].update(final="2"), "grader 2's row 2: grader_1 or final filled"),
    (lambda rows: rows[0].update(grader_1="2"), "grader 2's row 2: grader_1 or final filled"),
])
def test_a_damaged_copy_is_refused(sheet, damage, message):
    g1, g2 = fill(sheet, 1, lambda n: "2"), fill(sheet, 2, lambda n: "2")
    rows = rows_of(g2)
    damage(rows)
    save(g2, rows)
    with pytest.raises(gr.GradingError, match=message):
        gr.merge(sheet, g1, g2)


def test_merge_never_overwrites_a_graded_sheet(sheet):
    g1, g2 = fill(sheet, 1, lambda n: "2"), fill(sheet, 2, lambda n: "2")
    merged, _ = gr.merge(sheet, g1, g2)
    sheet.write_text(gr.csv_text(merged), encoding="utf-8", newline="")
    with pytest.raises(gr.GradingError, match="already holds grades"):
        gr.merge(sheet, g1, g2)


def test_an_invalid_grade_is_refused_with_its_line(sheet):
    g1, g2 = fill(sheet, 1, lambda n: "3"), fill(sheet, 2, lambda n: "2")
    with pytest.raises(ValueError, match="grades must be 0/1/2"):
        gr.merge(sheet, g1, g2)


# --- examples and the CLI ----------------------------------------------------------------------------------------

def test_examples_list_the_noted_rows_with_their_model(sheet):
    numbers = grade_numbers(sheet)
    g1 = fill(sheet, 1, lambda n: "0", notes={numbers[0]: "Example: invents a driver"})
    g2 = fill(sheet, 2, lambda n: "0")
    merged, _ = gr.merge(sheet, g1, g2)
    sheet.write_text(gr.csv_text(merged), encoding="utf-8", newline="")
    text = gr.examples(sheet, sheet.with_name(sa.KEY_NAME))
    key = {r["key"]: r for r in csv.DictReader(sheet.with_name(sa.KEY_NAME).open(encoding="utf-8"))}
    row = merged[numbers[0] - 2]
    assert text.count("\n## ") == 1 and f"- **Run:** {key[row['key']]['run_id']}" in text
    assert "- **Why:** G1: Example: invents a driver" in text and "graded 0" in text


def test_no_examples_yet_is_said_plainly(sheet):
    assert "(no row is marked `example` yet)" in gr.examples(sheet, sheet.with_name(sa.KEY_NAME))


def test_cli_merge_checks_first_and_writes_only_when_asked(sheet, capsys):
    g1, g2 = fill(sheet, 1, lambda n: "2"), fill(sheet, 2, lambda n: "2")
    before = sheet.read_bytes()
    assert gr.main(["merge", "--sheet", str(sheet), "--g1", str(g1), "--g2", str(g2)]) == 0
    assert sheet.read_bytes() == before and "check only" in capsys.readouterr().out
    assert gr.main(["merge", "--sheet", str(sheet), "--g1", str(g1), "--g2", str(g2), "--write"]) == 0
    assert "32 graded by both, 32 agree" in capsys.readouterr().out
    assert all(r["final"] == "2" for r in rows_of(sheet) if r["task"] == sa.GRADE)
    assert not list(sheet.parent.glob("*.part"))
    assert gr.main(["merge", "--sheet", str(sheet), "--g1", str(g1), "--g2", str(g2)]) == 1
    assert "already holds grades" in capsys.readouterr().err
