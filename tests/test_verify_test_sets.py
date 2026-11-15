"""eval/verify_test_sets.py (W1-10): the verifier's sheet and the apply step, on copies of the real test sets."""
import csv
import json

import pytest

import eval.verify_test_sets as vt
from eval.validate_test_sets import SETS


@pytest.fixture
def sets(tmp_path, monkeypatch):
    """Copies of the committed refusal and injection sets as they were before verification (verified_by null);
    the script reads and writes only these."""
    copies = {}
    for name, (model, item_rules, set_rules, path) in SETS.items():
        copy = tmp_path / path.name
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        copy.write_text("".join(json.dumps({**r, "verified_by": None}, ensure_ascii=False) + "\n" for r in records),
                        encoding="utf-8", newline="\n")
        copies[name] = (model, item_rules, set_rules, copy)
    monkeypatch.setattr(vt, "SETS", copies)
    return {name: spec[3] for name, spec in copies.items()}


def make_sheet(tmp_path, sets):
    sheet = tmp_path / "verification.csv"
    assert vt.main(["sheet", str(sheet), "--force"]) == 0
    with sheet.open(encoding="utf-8-sig", newline="") as f:
        return sheet, list(csv.DictReader(f))


def save(sheet, rows):
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=vt.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def fill(rows, verifier="Amresh", verdict="agree", only=None):
    for row in rows:
        if only is None or row["id"] in only:
            row.update(verifier=verifier, verdict=verdict)
    return rows


def test_the_sheet_has_one_row_per_test_with_what_the_verifier_checks(tmp_path, sets):
    _, rows = make_sheet(tmp_path, sets)
    assert len(rows) == 34 and [r["set"] for r in rows].count("injection") == 12
    by_id = {r["id"]: r for r in rows}
    assert by_id["RF01"]["expected"] == "refuse (R1, investment advice)"
    assert by_id["RF05"]["expected"].startswith("refuse (R3, outside the indexed filings")
    assert "must NOT be refused" in by_id["RF18"]["expected"]
    assert by_id["RF14"]["expected"].startswith("answer the factual part")
    inj = by_id["INJ01"]
    assert "391,035" in inj["must_contain_any"] and "1 trillion" in inj["must_not_contain"]
    assert inj["filing"].startswith("AAPL 10-K FY2024 FY Item 7") and inj["planted_text"]
    assert by_id["INJ03"]["expected"].startswith("refuse the user's question (R1")
    assert all(r["fingerprint"].startswith("fp-") and not r["verdict"] for r in rows)
    assert inj["forbidden_values"] == "1000000; 1000000000000"                  # every field the hash covers
    assert by_id["RF05"]["notes"] == "Microsoft has filed its FY2026 10-K, but it is not indexed"


def test_agreeing_on_every_test_sets_verified_by_and_passes_the_final_rules(tmp_path, sets, capsys):
    before = {name: path.read_text(encoding="utf-8") for name, path in sets.items()}
    sheet, rows = make_sheet(tmp_path, sets)
    save(sheet, fill(rows))
    assert vt.main(["apply", str(sheet)]) == 0
    assert "OK: validate_test_sets.py --final passes" in capsys.readouterr().out
    for name, path in sets.items():
        after = path.read_text(encoding="utf-8")
        assert after.count('"verified_by": "Amresh"') == after.count("\n")
        # nothing but verified_by changed, byte for byte
        assert after.replace('"verified_by": "Amresh"', '"verified_by": null') == before[name]


def test_blank_rows_stay_unverified_and_dry_run_writes_nothing(tmp_path, sets, capsys):
    sheet, rows = make_sheet(tmp_path, sets)
    save(sheet, fill(rows, only={"RF01", "INJ02"}))
    before = {name: path.read_bytes() for name, path in sets.items()}
    assert vt.main(["apply", str(sheet), "--dry-run"]) == 0
    assert {name: path.read_bytes() for name, path in sets.items()} == before
    assert vt.main(["apply", str(sheet)]) == 0
    out = capsys.readouterr().out
    assert "2 of 34 tests verified" in out and "--final not yet" in out
    assert '"id": "RF01"' in sets["refusal"].read_text(encoding="utf-8").split("\n")[0]
    assert '"verified_by": "Amresh"' in sets["refusal"].read_text(encoding="utf-8").split("\n")[0]


@pytest.mark.parametrize("change, problem", [
    (lambda r: r.update(verdict="disagree", comment="RF02 is R1 but reads as R4"), "disagrees: RF02 is R1"),
    (lambda r: r.update(verifier="alok"), "wrote this test"),               # the author, any case
    (lambda r: r.update(verdict="yes"), "verdict must be 'agree' or 'disagree'"),
    (lambda r: r.update(verifier=""), "verdict without a verifier name"),
    (lambda r: r.update(fingerprint="000000000000"), "changed after the sheet was made"),
    (lambda r: r.update(id="RF99"), "no refusal test with this id"),
])
def test_any_problem_row_means_nothing_is_written(tmp_path, sets, capsys, change, problem):
    sheet, rows = make_sheet(tmp_path, sets)
    fill(rows)
    change(next(r for r in rows if r["id"] == "RF02"))
    save(sheet, rows)
    before = {name: path.read_bytes() for name, path in sets.items()}
    assert vt.main(["apply", str(sheet)]) == 1
    out = capsys.readouterr().out
    assert problem in out and "nothing written" in out
    assert {name: path.read_bytes() for name, path in sets.items()} == before


def test_a_test_edited_after_the_sheet_needs_a_new_sheet(tmp_path, sets):
    sheet, rows = make_sheet(tmp_path, sets)
    save(sheet, fill(rows))
    text = sets["refusal"].read_text(encoding="utf-8")
    sets["refusal"].write_text(text.replace("buy/sell recommendation", "a buy recommendation"), encoding="utf-8")
    assert vt.main(["apply", str(sheet)]) == 1
    _, fresh = make_sheet(tmp_path, sets)
    save(sheet, fill(fresh))
    assert vt.main(["apply", str(sheet)]) == 0


def test_a_duplicate_row_and_a_sheet_without_the_columns_are_refused(tmp_path, sets, capsys):
    sheet, rows = make_sheet(tmp_path, sets)
    save(sheet, fill(rows) + [dict(rows[0])])
    assert vt.main(["apply", str(sheet)]) == 1
    assert "the same test appears twice" in capsys.readouterr().out
    sheet.write_text("id,question\nRF01,x\n", encoding="utf-8")
    assert vt.main(["apply", str(sheet)]) == 2


def test_verified_tests_show_as_agreed_in_a_new_sheet(tmp_path, sets):
    sheet, rows = make_sheet(tmp_path, sets)
    save(sheet, fill(rows, only={"INJ05"}))
    assert vt.main(["apply", str(sheet)]) == 0
    _, again = make_sheet(tmp_path, sets)
    row = next(r for r in again if r["id"] == "INJ05")
    assert (row["verifier"], row["verdict"]) == ("Amresh", "agree")
    assert vt.main(["apply", str(sheet)]) == 0                              # applying twice changes nothing more


def test_the_fingerprint_ignores_only_verified_by():
    rec = {"id": "RF01", "question": "q", "written_by": "Alok", "verified_by": None}
    assert vt.fingerprint(rec) == vt.fingerprint({**rec, "verified_by": "Amresh"})
    assert vt.fingerprint(rec) != vt.fingerprint({**rec, "question": "q2"})


@pytest.mark.parametrize("verifier", ["Alok Srivastava", "ALOK", "srivastava, alok", " alok "])
def test_the_author_cannot_verify_under_a_fuller_name(tmp_path, sets, capsys, verifier):
    # the tests are signed "Alok": a sheet naming "Alok Srivastava" once passed as a second person (W1-10 review)
    sheet, rows = make_sheet(tmp_path, sets)
    save(sheet, fill(rows, verifier=verifier, only={"RF01"}))
    assert vt.main(["apply", str(sheet)]) == 1
    assert "wrote this test" in capsys.readouterr().out


@pytest.mark.parametrize("a, b, same", [
    ("Alok", "Alok Srivastava", True), ("Amresh Kumar", "Amresh", True), ("Chandra Shekhar", "chandra shekhar", True),
    ("Amresh Kumar", "Alok", False), ("Chandra Shekhar", "Chandra Mohan", False), ("Alok", "", False),
])
def test_same_person_compares_the_words_of_two_names(a, b, same):
    assert vt.same_person(a, b) is same


def test_sheet_never_overwrites_a_filled_sheet_without_force(tmp_path, sets):
    sheet, rows = make_sheet(tmp_path, sets)
    save(sheet, fill(rows))
    assert vt.main(["sheet", str(sheet)]) == 1                               # review: verdicts were lost
    assert sum(r["verdict"] == "agree" for r in csv.DictReader(sheet.open(encoding="utf-8-sig"))) == 34


def test_a_fingerprint_never_looks_like_a_number():
    # Excel turns an all-digit cell into 6.62345E+11 on save, and the row could never be applied (review)
    rec = {"id": "RF01", "question": "q", "written_by": "Alok", "notes": "n446", "verified_by": None}
    for notes in ("n446", "a", "b", "c", "d"):
        fp = vt.fingerprint({**rec, "notes": notes})
        assert fp.startswith("fp-") and len(fp) == 15
        with pytest.raises(ValueError):
            float(fp)                                                        # no spreadsheet reads it as a number


def test_an_older_sheet_does_not_replace_a_newer_verifier(tmp_path, sets, capsys):
    sheet, rows = make_sheet(tmp_path, sets)
    save(sheet, fill(rows, verifier="Amresh Kumar"))
    assert vt.main(["apply", str(sheet)]) == 0
    stale = fill([dict(r) for r in rows], verifier="Giridharan", only={"RF01"})
    save(sheet, stale)
    before = sets["refusal"].read_bytes()
    assert vt.main(["apply", str(sheet)]) == 1
    assert "already verified by Amresh Kumar" in capsys.readouterr().out
    assert sets["refusal"].read_bytes() == before
    save(sheet, fill([dict(r) for r in rows], verifier="Amresh"))           # the same person, shorter name
    assert vt.main(["apply", str(sheet)]) == 0
    assert sets["refusal"].read_bytes() == before                            # the first recorded name stays


def test_a_name_in_another_unicode_form_is_the_same_person():
    composed, decomposed = "Alók", "Alók"                         # NFC and NFD (a Mac keyboard)
    assert composed != decomposed and vt.same_person(composed, decomposed)


def test_both_files_or_neither_when_a_write_fails(tmp_path, sets, monkeypatch, capsys):
    sheet, rows = make_sheet(tmp_path, sets)
    save(sheet, fill(rows))
    before = {name: path.read_bytes() for name, path in sets.items()}
    real_replace, calls = vt.os.replace, []

    def flaky_replace(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise PermissionError("the file is open in another program")
        real_replace(src, dst)

    monkeypatch.setattr(vt.os, "replace", flaky_replace)
    assert vt.main(["apply", str(sheet)]) == 2
    assert "nothing written" in capsys.readouterr().err
    assert {name: path.read_bytes() for name, path in sets.items()} == before
    assert not list(tmp_path.glob("*.tmp"))


def test_excel_quirks_are_handled_or_named(tmp_path, sets, capsys):
    sheet, rows = make_sheet(tmp_path, sets)
    save(sheet, fill(rows, only={"RF01"}))
    with sheet.open("a", encoding="utf-8", newline="") as f:
        f.write("," * (len(vt.COLUMNS) - 1) + "\r\n" + "," * (len(vt.COLUMNS) - 1) + "\r\n")   # empty rows at the end
    assert vt.main(["apply", str(sheet), "--dry-run"]) == 0
    sheet.write_text(";".join(vt.COLUMNS) + "\nrefusal;RF01\n", encoding="utf-8")
    assert vt.main(["apply", str(sheet)]) == 2
    assert "semicolons" in capsys.readouterr().err
    sheet.write_bytes(",".join(vt.COLUMNS).encode() + b"\nrefusal,RF01,,,,,,,,,,,,,Jos\xe9,agree,\n")   # cp1252
    assert vt.main(["apply", str(sheet)]) == 2
    assert "CSV UTF-8" in capsys.readouterr().err


def test_a_comment_without_a_verdict_is_reported(tmp_path, sets, capsys):
    sheet, rows = make_sheet(tmp_path, sets)
    rows[0]["comment"] = "looks fine"
    save(sheet, rows)
    assert vt.main(["apply", str(sheet)]) == 1
    assert "a comment without a verdict" in capsys.readouterr().out


def test_applying_the_committed_verification_again_changes_no_byte(tmp_path, monkeypatch):
    # not circular: the committed files as they are in git, not rewritten by the script's own serializer
    copies = {}
    for name, (model, item_rules, set_rules, path) in SETS.items():
        copy = tmp_path / path.name
        copy.write_bytes(path.read_bytes())
        copies[name] = (model, item_rules, set_rules, copy)
    monkeypatch.setattr(vt, "SETS", copies)
    sheet = tmp_path / "again.csv"
    assert vt.main(["sheet", str(sheet)]) == 0
    assert vt.main(["apply", str(sheet)]) == 0
    assert all(spec[3].read_bytes() == SETS[name][3].read_bytes() for name, spec in copies.items())
