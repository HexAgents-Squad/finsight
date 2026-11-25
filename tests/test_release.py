"""scripts/release_check.py (W3-13, ADR-048): the v1.0 release checks. Offline; the gold-v1 comparison is skipped in a
clone without the tag (CI's shallow checkout), and the generated-report checks are covered by their own tests."""
from __future__ import annotations

import json
import re
import subprocess

import pytest

import scripts.release_check as rc


def test_every_place_states_the_release_version():
    found = rc.versions()
    assert found["pyproject.toml"] == found["api/main.py"] == rc.VERSION
    assert found["docs/release_v1.0.md"] == rc.TAG.lstrip("v")          # the notes are titled "FinSight v1.0"
    assert rc.check_version() == []


def test_the_release_notes_are_traceable():
    assert rc.check_notes() == []


def test_every_generated_report_with_a_check_is_in_the_release_check():
    """A new script with --check must be added to CHECKS, or the release could ship a stale report."""
    have = {cmd[0] for cmd in rc.CHECKS if "--check" in cmd}
    scripts = sorted(p.relative_to(rc.REPO).as_posix() for d in ("scripts", "eval") for p in (rc.REPO / d).glob("*.py")
                     if re.search(r"""add_argument\(\s*["']--check["']""", p.read_text(encoding="utf-8")))
    assert scripts and set(scripts) <= have, set(scripts) - have


def test_the_ci_validators_are_in_the_release_check():
    ci = (rc.REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    validators = re.findall(r"run: python (eval/validate_\S+(?: \S+)*)", ci)
    assert validators and all(v.split() in rc.CHECKS for v in validators)


def _jsonl(*items: dict) -> str:
    return "".join(json.dumps(i) + "\n" for i in items)


def test_frozen_fields_may_not_change_but_review_fields_may():
    old = _jsonl({"id": "RF01", "question": "Q?", "expected_rule": "R1", "verified_by": None, "rationale": "a"})
    new = _jsonl({"id": "RF01", "question": "Q?", "expected_rule": "R1", "verified_by": "Amresh", "rationale": "b"})
    fields = rc.FROZEN_FIELDS["eval/refusal_tests.jsonl"]
    assert rc.frozen_differences(old, new, fields) == []
    changed = new.replace('"R1"', '"R2"')
    assert rc.frozen_differences(old, changed, fields) == ["RF01: expected_rule changed"]
    added = new + _jsonl({"id": "RF99", "question": "new"})
    assert rc.frozen_differences(old, added, fields) == ["RF99 was added"]
    assert rc.frozen_differences(added, new, fields) == ["RF99 was removed"]


def _has_tag(tag: str) -> bool:
    return subprocess.run(["git", "-C", str(rc.REPO), "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"],
                          capture_output=True).returncode == 0


@pytest.mark.skipif(not _has_tag(rc.GOLD_TAG), reason="the gold-v1 tag is not in this clone (shallow checkout)")
def test_the_gold_set_and_the_tests_are_as_frozen_at_gold_v1():
    assert rc.check_frozen() == []


def test_the_tag_lands_only_on_the_merged_pushed_main_and_never_moves():
    assert rc.tag_target_problems("main", "abc", "abc", tag_exists=False) == []
    assert rc.tag_target_problems("w3/UPGRAD-88-release", "abc", "abc", False)[0].startswith("HEAD is w3/")
    assert rc.tag_target_problems("main", "abc", "def", False) == [
        "main differs from origin/main: git pull --ff-only (and push nothing unreviewed)"]
    assert rc.tag_target_problems("main", "abc", None, False) == ["origin/main is unknown: git fetch origin"]
    assert rc.tag_target_problems("main", "abc", "abc", True) == ["the tag v1.0 exists already: a released tag is never moved"]


def test_a_frozen_file_missing_at_the_tag_is_named(monkeypatch):
    real = rc.git

    def fake(*args):
        if args[0] == "show" and args[1].endswith("injection_tests.jsonl"):
            return real("show", "no-such-ref:nothing")
        return real(*args)
    if not _has_tag(rc.GOLD_TAG):
        pytest.skip("the gold-v1 tag is not in this clone")
    monkeypatch.setattr(rc, "git", fake)
    assert rc.check_frozen() == ["eval/injection_tests.jsonl is not in the tag gold-v1"]


def test_the_makefile_runs_the_release_check():
    makefile = (rc.REPO / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^release-check:.*\n\t\$\(PYTHON\) scripts/release_check\.py", makefile, re.M)
