"""eval/grading.py offline, on the committed grading sheet and key (copied to a temporary folder; the repo's files
are never written)."""
import csv
import shutil

import pytest

import eval.grading as gr
import eval.score_answers as sa


@pytest.fixture
def sheet(tmp_path):
    """The committed sheet and key, with every grader cell emptied: the tests keep passing after W1-14's graded
    sheet is committed."""
    shutil.copy(sa.GRADING_DIR / sa.KEY_NAME, tmp_path / sa.KEY_NAME)
    rows = rows_of(sa.GRADING_DIR / sa.SHEET_NAME)
    for r in rows:
        r.update({c: "" for c in gr.GRADER_COLUMNS + ("note",)})
    save(tmp_path / sa.SHEET_NAME, rows)
    return tmp_path / sa.SHEET_NAME


def label_rows(sheet):
    return [n for n, r in enumerate(rows_of(sheet), start=2) if r["task"] == sa.LABEL]


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


def test_the_committed_sheet_splits_in_half(sheet):
    split, labels = gr.split_row(rows_of(sheet)), label_rows(sheet)
    first, second = sum(n < split for n in labels), sum(n >= split for n in labels)
    assert first + second == len(labels) and first - second in (0, 1)


# --- copies ----------------------------------------------------------------------------------------------------

def test_copies_are_the_untouched_sheet_with_assignments(sheet, tmp_path):
    out = tmp_path / "copies"
    lines = gr.copies(sheet, ["Chandra", "Amresh"], out)
    for name in ("Chandra", "Amresh"):
        assert (out / f"grading_sheet_{name}.csv").read_bytes() == sheet.read_bytes()
    text = (out / "assignments.txt").read_text(encoding="utf-8")
    n = len(grade_numbers(sheet))                        # every grade row (32 for M1-M3, more with W1-09's runs)
    assert text == "\n".join(lines) and "Grader 1 = Chandra" in text and f"column grader_2 on all {n}" in text
    split, labels = gr.split_row(rows_of(sheet)), label_rows(sheet)
    mine = [n for n in labels if n < split], [n for n in labels if n >= split]
    assert f"spreadsheet rows {mine[0][0]}-{mine[0][-1]}" in text
    assert f"spreadsheet rows {mine[1][0]}-{mine[1][-1]}" in text


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
    g1 = fill(sheet, 1, lambda n: " 2", notes={numbers[0]: "overturn? the figure is right"}, excel=True)
    g2 = fill(sheet, 2, lambda n: "1" if n == numbers[1] else "2", notes={numbers[0]: "fine"})
    merged, rep = gr.merge(sheet, g1, g2)
    by_row = dict(enumerate(merged, start=2))
    n = len(numbers)
    assert (rep["pairs"], rep["agree"], rep["missing"], rep["warnings"]) == (n, n - 1, [], [])
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
    assert rep["warnings"] == ["grader 1 also labelled 1 of the other grader's rows: the owner's label decides; "
                               "these fill a label the owner left empty and give the label agreement"]
    assert rep["label_pairs"] == 1 and rep["label_agree"] == 0
    assert rep["label_differs"] == [f"row {theirs}: grader 1 = invented_number, grader 2 = wrong_company "
                                    f"(kept: wrong_company)"]


def test_missing_grades_and_labels_are_listed(sheet):
    numbers = grade_numbers(sheet)
    g1 = fill(sheet, 1, lambda n: "" if n == numbers[2] else "1", label_for=lambda r: "")
    g2 = fill(sheet, 2, lambda n: "1")
    merged, rep = gr.merge(sheet, g1, g2)
    assert f"row {numbers[2]}: grade missing (grader 1: -, grader 2: 1)" in rep["missing"]
    split = gr.split_row(rows_of(sheet))
    assert sum("no failure_label" in m for m in rep["missing"]) == sum(n < split for n in label_rows(sheet))
    assert merged[numbers[2] - 2]["final"] == ""


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
    with pytest.raises(gr.GradingError, match="already holds grades.*git checkout -- "
                                              "eval/results/grading/grading_sheet.csv"):
        gr.merge(sheet, g1, g2)


def test_an_invalid_grade_is_refused_with_its_line(sheet):
    g1, g2 = fill(sheet, 1, lambda n: "3"), fill(sheet, 2, lambda n: "2")
    with pytest.raises(ValueError, match=r"line \d+, column grader_1: '3' is not 0, 1 or 2"):
        gr.merge(sheet, g1, g2)


# --- examples and the CLI ----------------------------------------------------------------------------------------

def test_examples_list_the_noted_rows_with_their_model(sheet):
    numbers = grade_numbers(sheet)
    g1 = fill(sheet, 1, lambda n: "0", notes={numbers[0]: "Example: invents\na driver",
                                              numbers[1]: "for example, vague"})
    g2 = fill(sheet, 2, lambda n: "0")
    merged, _ = gr.merge(sheet, g1, g2)
    sheet.write_text(gr.csv_text(merged), encoding="utf-8", newline="")
    text = gr.examples(sheet, sheet.with_name(sa.KEY_NAME))
    key = {r["key"]: r for r in csv.DictReader(sheet.with_name(sa.KEY_NAME).open(encoding="utf-8"))}
    row = merged[numbers[0] - 2]
    assert text.count("\n## ") == 1 and f"- **Run:** {key[row['key']]['run_id']}" in text
    assert "- **Why:** G1: Example: invents a driver" in text and "graded 0" in text      # one line, one row


def test_no_examples_yet_is_said_plainly(sheet):
    assert "(no row is marked `example` yet)" in gr.examples(sheet, sheet.with_name(sa.KEY_NAME))


def test_cli_merge_checks_first_and_writes_only_when_asked(sheet, capsys):
    g1, g2 = fill(sheet, 1, lambda n: "2"), fill(sheet, 2, lambda n: "2")
    before = sheet.read_bytes()
    assert gr.main(["merge", "--sheet", str(sheet), "--g1", str(g1), "--g2", str(g2)]) == 0
    assert sheet.read_bytes() == before and "check only" in capsys.readouterr().out
    assert gr.main(["merge", "--sheet", str(sheet), "--g1", str(g1), "--g2", str(g2), "--write"]) == 0
    n = len(grade_numbers(sheet))
    assert f"{n} graded by both, {n} agree" in capsys.readouterr().out
    assert all(r["final"] == "2" for r in rows_of(sheet) if r["task"] == sa.GRADE)
    assert not list(sheet.parent.glob("*.part"))
    assert gr.main(["merge", "--sheet", str(sheet), "--g1", str(g1), "--g2", str(g2)]) == 1
    assert "already holds grades" in capsys.readouterr().err


# --- added after the review of the W1-14 branch -------------------------------------------------------------------

def test_check_finds_a_copys_mistakes_before_the_other_copy_is_back(sheet, capsys):
    numbers, split = grade_numbers(sheet), gr.split_row(rows_of(sheet))
    g1 = fill(sheet, 1, lambda n: "" if n == numbers[0] else "2")
    out = gr.check(sheet, g1, 1)
    assert out[0] == "grader 1: 1 missing" and out[1] == f"row {numbers[0]}: no grade in grader_1"
    assert gr.main(["check", "--grader", "2", "--sheet", str(sheet), str(g1)]) == 1      # the wrong grader's copy
    assert "grader 2's row" in capsys.readouterr().err
    g2 = fill(sheet, 2, lambda n: "1", label_for=lambda r: "")
    assert gr.check(sheet, g2, 2)[0] == f"grader 2: {sum(n >= split for n in label_rows(sheet))} missing"


def test_a_label_on_a_grade_row_is_kept_or_flagged(sheet):
    numbers = grade_numbers(sheet)
    g1 = fill(sheet, 1, lambda n: "0")
    g2 = fill(sheet, 2, lambda n: "0")
    for path, labels in ((g1, {numbers[0]: "over_refusal", numbers[1]: "wrong_company"}),
                         (g2, {numbers[1]: "invented_number"})):
        rows = rows_of(path)
        for n, label in labels.items():
            rows[n - 2]["failure_label"] = label
        save(path, rows)
    merged, rep = gr.merge(sheet, g1, g2)
    assert merged[numbers[0] - 2]["failure_label"] == "over_refusal" and merged[numbers[1] - 2]["failure_label"] == ""
    assert any(d.startswith(f"row {numbers[1]}: the graders gave different labels") for d in rep["disagree"])


def test_marks_need_the_guides_spelling(sheet):
    numbers = grade_numbers(sheet)
    g1 = fill(sheet, 1, lambda n: "2", notes={numbers[0]: "no overturn needed", numbers[1]: "overturn ? right"})
    _, rep = gr.merge(sheet, g1, fill(sheet, 2, lambda n: "2"))
    assert [o.split(":")[0] for o in rep["overturn"]] == [f"row {numbers[1]}"]


def test_a_long_list_of_problems_is_counted(sheet):
    g1 = fill(sheet, 1, lambda n: "2")
    g2 = fill(sheet, 2, lambda n: "2")
    rows = rows_of(g2)
    for r in rows:
        r["final"] = "2"
    save(g2, rows)
    with pytest.raises(gr.GradingError, match=r"\.\.\. and \d+ more"):
        gr.merge(sheet, g1, g2)


def test_when_both_graders_label_every_row_the_second_label_fills_gaps(sheet):
    # What the W1-14 graders did (measured): each labelled all label rows, the owner missed a few
    split, labels = gr.split_row(rows_of(sheet)), label_rows(sheet)
    mine1, mine2 = [n for n in labels if n < split], [n for n in labels if n >= split]
    g1 = fill(sheet, 1, lambda n: "2", label_for=lambda r: "invented_number")
    g2 = fill(sheet, 2, lambda n: "2", label_for=lambda r: "wrong_line_item")
    for path, label in ((g1, "invented_number"), (g2, "wrong_line_item")):
        rows = rows_of(path)
        for n in labels:
            rows[n - 2]["failure_label"] = label
        save(path, rows)
    rows = rows_of(g1)
    rows[mine1[0] - 2]["failure_label"] = ""                                 # grader 1 missed one of theirs
    save(g1, rows)
    merged, rep = gr.merge(sheet, g1, g2)
    assert merged[mine1[0] - 2]["failure_label"] == "wrong_line_item"         # filled from grader 2
    assert merged[mine1[1] - 2]["failure_label"] == "invented_number"        # the owner decides
    assert merged[mine2[0] - 2]["failure_label"] == "wrong_line_item"
    assert rep["missing"] == [] and rep["label_pairs"] == len(labels) - 1 and rep["label_agree"] == 0
    assert f"row {mine1[0]}: the owner left the label empty; the other grader's wrong_line_item is used" \
        in rep["warnings"]


# --- a later round on the ungraded rows only (--pending, W1-09) ---------------------------------------------------

@pytest.fixture
def graded_sheet(tmp_path):
    """A sheet whose first grade row is already graded and some label rows labelled, the rest not: a later round
    (like the W1-09 strategy rows next to the graded M1-M3 rows)."""
    shutil.copy(sa.GRADING_DIR / sa.KEY_NAME, tmp_path / sa.KEY_NAME)
    rows = rows_of(sa.GRADING_DIR / sa.SHEET_NAME)
    for r in rows:
        r.update({c: "" for c in gr.GRADER_COLUMNS + ("note",)})
    first_grade = next(r for r in rows if r["task"] == sa.GRADE)
    first_grade.update(grader_1="2", grader_2="2", final="2")
    for r in [r for r in rows if r["task"] == sa.LABEL][:3]:
        r["failure_label"] = "arithmetic_slip"
    save(tmp_path / sa.SHEET_NAME, rows)
    return tmp_path / sa.SHEET_NAME


def test_pending_copies_hold_only_the_ungraded_rows(graded_sheet, tmp_path):
    with pytest.raises(gr.GradingError, match="add --pending"):
        gr.copies(graded_sheet, ["Chandra", "Amresh"], tmp_path / "plain")
    lines = gr.copies(graded_sheet, ["Chandra", "Amresh"], tmp_path / "c", pending=True)
    todo = gr.pending_rows(gr.load(graded_sheet))
    copy = rows_of(tmp_path / "c" / "grading_sheet_Chandra.csv")
    assert [r["key"] for r in copy] == [r["key"] for r in todo] and len(copy) == len(rows_of(graded_sheet)) - 4
    assert not gr.has_grades(copy) and lines[0].startswith(f"ungraded rows of the sheet: {len(todo)} rows")


def test_a_pending_merge_fills_those_rows_and_leaves_the_graded_ones(graded_sheet):
    copy_dir = graded_sheet.parent / "c"
    gr.copies(graded_sheet, ["Chandra", "Amresh"], copy_dir, pending=True)
    base = copy_dir / "grading_sheet_Chandra.csv"
    g1, g2 = fill(base, 1, lambda n: "2"), fill(base, 2, lambda n: "2")
    assert gr.check(graded_sheet, g1, 1, pending=True)[0] == "grader 1: 0 missing"
    merged, rep = gr.merge(graded_sheet, g1, g2, pending=True)
    before = rows_of(graded_sheet)
    assert [r["key"] for r in merged] == [r["key"] for r in before]
    graded = [n for n, r in enumerate(before) if gr.has_grades([r])]
    assert all(merged[n] == before[n] for n in graded)                          # untouched, not shown
    assert all(r["final"] == "2" for n, r in enumerate(merged) if r["task"] == sa.GRADE and n not in graded)
    assert rep["pairs"] == sum(r["task"] == sa.GRADE for r in gr.pending_rows(gr.load(graded_sheet)))
    with pytest.raises(gr.GradingError, match="already holds grades"):
        gr.merge(graded_sheet, g1, g2)                                       # without --pending: refused


def test_the_committed_sheet_has_the_second_round_merged():
    """W2-12: the --pending round for the W1-09 strategy runs (S1-S3) and the W2-11 agent run (A1) is merged, so no
    row waits for a grader; every S and A grade row carries both graders' grades, and every label row a label."""
    sheet = gr.load(gr.SHEET)
    assert gr.pending_rows(sheet) == []
    key = {r["key"]: r["run_id"] for r in csv.DictReader((sa.GRADING_DIR / sa.KEY_NAME).open(encoding="utf-8"))}
    second = [r for r in sheet if key[r["key"]][0] in "SA"]
    assert len(second) == 193
    assert all(r["grader_1"] and r["grader_2"] for r in second if r["task"] == sa.GRADE)
    assert all(r["failure_label"] for r in second if r["task"] == sa.LABEL)
