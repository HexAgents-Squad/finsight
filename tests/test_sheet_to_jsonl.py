"""Google Sheet (CSV) → eval/gold_set.jsonl converter."""
import csv
import json

import pytest

from eval.sheet_to_jsonl import (
    CellError,
    main,
    parse_bool,
    parse_category,
    parse_number,
    parse_sources,
    parse_year,
)

HEADER = ["ID", "Category", "Question", "Companies", "Fiscal_Years", "Gold_Answer", "Gold_Value", "Unit",
          "Tolerance_Pct", "Gold_Sources", "Must_Refuse", "Refusal_Rule", "Previous_Turn_Id", "Written_By",
          "Verified_By", "Notes"]

ROWS = [
    # out of order on purpose: the output must be sorted by id
    ["b01", "B", "What was Microsoft's operating margin in fiscal 2024?", "msft", "FY2024", "44.6%", "44.6%", "%",
     "", "MSFT | 10-K | 2024 | Item 8", "FALSE", "", "", "Anirudh", "Chandra Shekhar", ""],
    ["A01", "fact", "What were Apple's net sales in fiscal 2025 Q2?", "AAPL", "2025", "95,359", "$95,359",
     "USD millions", "0.5", "AAPL | 10-Q | 2025 | Q2 | Part I  Item 1", "no", "", "", "Amresh", "Giridharan", ""],
    ["G01", "refusal", "Should I buy NVIDIA stock right now?", "", "", "Refuse", "", "", "", "", "TRUE", "r1", "",
     "Alok", "Anirudh", "R1 example"],
    ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],                 # empty sheet row: skipped
    ["D01", "multi-company", "Which had the higher operating margin in fiscal 2024, Microsoft or Alphabet?",
     "MSFT, GOOGL", "2024", "Microsoft", "44.6", "%", "",
     "MSFT | 10-K | 2024 | Item 8\nGOOGL | 10-K | 2024 | FY | Item 8", "", "", "", "Giridharan", "Alok", ""],
]


def write_csv(path, rows, header=HEADER, bom=True):
    with path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def test_round_trip_writes_sorted_valid_jsonl(tmp_path, capsys):
    out = tmp_path / "gold.jsonl"
    assert main([str(write_csv(tmp_path / "s.csv", ROWS)), "-o", str(out)]) == 0
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in records] == ["A01", "B01", "D01", "G01"]
    a01, b01, d01, g01 = records
    assert a01["gold_value"] == 95359 and a01["tolerance_pct"] == 0.5
    assert a01["gold_sources"][0] == {"ticker": "AAPL", "form": "10-Q", "fiscal_year": 2025, "fiscal_period": "Q2",
                                      "item": "Part I Item 1"}
    assert b01["companies"] == ["MSFT"] and b01["fiscal_years"] == [2024] and b01["tolerance_pct"] == 1.0
    assert b01["gold_sources"][0]["fiscal_period"] == "FY"          # inferred for a 10-K
    assert len(d01["gold_sources"]) == 2
    assert g01["category"] == "refuse" and g01["must_refuse"] is True and g01["refusal_rule"] == "R1"
    assert list(a01) == ["id", "category", "question", "companies", "fiscal_years", "gold_answer", "gold_value",
                         "unit", "tolerance_pct", "gold_sources", "must_refuse", "refusal_rule",
                         "previous_turn_id", "written_by", "verified_by", "notes"]
    assert "OK: wrote 4 items" in capsys.readouterr().out


def test_invalid_sheet_is_not_written_and_errors_name_sheet_rows(tmp_path, capsys):
    rows = [r[:] for r in ROWS]
    rows[0][6] = ""                                                  # B01 ratio without gold_value (sheet row 2)
    out = tmp_path / "gold.jsonl"
    assert main([str(write_csv(tmp_path / "s.csv", rows)), "-o", str(out)]) == 1
    text = capsys.readouterr().out
    assert "ERROR: row 2 (B01): category 'ratio' needs a numeric gold_value" in text
    assert "NOT WRITTEN" in text
    assert not out.exists()


def test_unparseable_cell_reports_row_and_continues(tmp_path, capsys):
    rows = [r[:] for r in ROWS]
    rows[1][10] = "maybe"                                            # A01 must_refuse (sheet row 3)
    assert main([str(write_csv(tmp_path / "s.csv", rows)), "--dry-run"]) == 1
    assert "row 3 (A01): 'maybe' is not TRUE/FALSE" in capsys.readouterr().out


def test_missing_required_column(tmp_path, capsys):
    header = [h for h in HEADER if h != "Question"]
    rows = [r[:2] + r[3:] for r in ROWS[:1]]
    assert main([str(write_csv(tmp_path / "s.csv", rows, header=header)), "--dry-run"]) == 1
    assert "missing required column(s): ['question']" in capsys.readouterr().out


def test_unknown_columns_warn_but_do_not_block(tmp_path, capsys):
    header = HEADER + ["Reviewer comment"]
    rows = [r + (["looks fine"] if r[0] else [""]) for r in ROWS]
    assert main([str(write_csv(tmp_path / "s.csv", rows, header=header, bom=False)), "--dry-run"]) == 0
    assert "WARNING: unknown column(s) ignored: ['reviewer comment']" in capsys.readouterr().out


def test_row_with_content_but_no_id_gets_a_clear_message(tmp_path, capsys):
    rows = [*ROWS, ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "remember to add a bank question"]]
    assert main([str(write_csv(tmp_path / "s.csv", rows)), "--dry-run"]) == 1
    assert "row 7: id is empty (fill it in, or delete the row" in capsys.readouterr().out


@pytest.mark.parametrize("value, expected", [("A", "fact"), ("g", "refuse"), ("Period comparison", "period_comparison"),
                                             ("multi-turn", "multi_turn"), ("qualitative", "qualitative")])
def test_parse_category(value, expected):
    assert parse_category(value) == expected


@pytest.mark.parametrize("value, expected", [("391,035", 391035.0), ("$1,234.5", 1234.5), ("44.6%", 44.6),
                                             ("(1,200)", -1200.0), ("", None), ("  ", None)])
def test_parse_number(value, expected):
    assert parse_number(value) == expected


def test_parse_errors():
    with pytest.raises(CellError):
        parse_number("about 5")
    with pytest.raises(CellError):
        parse_year("last year")
    with pytest.raises(CellError):
        parse_bool("perhaps")
    with pytest.raises(CellError, match="PERIOD may be omitted only for a 10-K"):
        parse_sources("AAPL | 10-Q | 2025 | Part I Item 2")
    assert parse_year(" fy2024 ") == 2024
    assert parse_bool("Yes") is True and parse_bool("") is False


@pytest.mark.parametrize("value", ["nan", "inf", "-Infinity", "3.91E+05"])
def test_parse_number_rejects_non_finite_and_scientific(value):
    with pytest.raises(CellError):
        parse_number(value)


def test_parse_number_accepts_unicode_minus_and_years_need_4_digits():
    assert parse_number("−5.2") == -5.2
    for bad in ("FY24", "24", "02024"):
        with pytest.raises(CellError):
            parse_year(bad)


def test_missing_or_non_utf8_csv_exits_2(tmp_path, capsys):
    assert main([str(tmp_path / "missing.csv"), "--dry-run"]) == 2
    cp1252 = tmp_path / "excel.csv"
    cp1252.write_bytes(",".join(HEADER).encode() + b"\nA01,fact,What were Apple\x92s net sales?\n")
    assert main([str(cp1252), "--dry-run"]) == 2
    assert "is not UTF-8" in capsys.readouterr().err
