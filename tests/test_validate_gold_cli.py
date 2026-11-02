"""Command-line behaviour of eval/validate_gold.py: exit codes, messages, file handling."""
import json

from eval.validate_gold import main
from tests.factories import full_gold_set, make


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_valid_file_exits_0(tmp_path, capsys):
    path = write_jsonl(tmp_path / "g.jsonl", [make("fact"), make("refuse")])
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "OK: gold set valid (draft rules)" in out
    assert "2 items read" in out


def test_problems_exit_1_with_line_numbers(tmp_path, capsys):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(make("fact")) + "\n{not json\n"
                    + json.dumps(make("ratio", gold_value=None, unit=None)) + "\n", encoding="utf-8")
    assert main([str(path)]) == 1
    out = capsys.readouterr().out
    assert "ERROR: line 2: invalid JSON" in out
    assert "ERROR: line 3 (B01): category 'ratio' needs a numeric gold_value" in out
    assert "FAILED: 2 problem(s)" in out


def test_final_mode_applies_freeze_rules(tmp_path, capsys):
    draft = write_jsonl(tmp_path / "draft.jsonl", [make("fact")])
    assert main([str(draft)]) == 0
    assert main([str(draft), "--final"]) == 1
    full = write_jsonl(tmp_path / "full.jsonl", full_gold_set())
    assert main([str(full), "--final"]) == 0
    assert "OK: gold set valid (final rules)" in capsys.readouterr().out


def test_missing_file(tmp_path, capsys):
    missing = tmp_path / "nope.jsonl"
    assert main([str(missing)]) == 2
    assert main([str(missing), "--allow-missing"]) == 0
    assert "SKIP" in capsys.readouterr().out


def test_blank_lines_and_utf8_bom_are_tolerated(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text("﻿" + json.dumps(make("fact")) + "\n\n   \n", encoding="utf-8")
    assert main([str(path)]) == 0


def test_non_object_line_is_an_error(tmp_path, capsys):
    path = tmp_path / "g.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    assert main([str(path)]) == 1
    assert "must be a JSON object" in capsys.readouterr().out
