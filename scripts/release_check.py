"""W3-13: everything that must hold before the v1.0 tag, in one command (docs/release_v1.0.md).

Usage:
    python scripts/release_check.py               # on main, merged and pushed: all checks, before `git tag v1.0`
    python scripts/release_check.py --allow-dirty # on a branch, before the merge (what `make release-check` runs)

Checks, each printed as OK or with its problems:
  1. version     pyproject.toml, the API's version and the release notes all say VERSION
  2. frozen      eval/gold_set.jsonl is byte for byte the tag gold-v1's; the refusal and injection tests keep every
                 question, expected decision and planted passage they had at the tag (only the review fields may
                 differ: who verified them, a rationale, the injection tests' match lists fixed in the Week 0 review)
  3. generated   every committed report is current: the Week 1 and Week 2 notes, the evaluation report and the
                 technical report (each script's --check), and the gold, test and development sets (--final)
  4. notes       the release notes pass the report's traceability check (scripts/final_report.py): every number is
                 in a file the notes cite
  5. clean       no uncommitted change (skipped with --allow-dirty)
  6. tag target  HEAD is main and equals origin/main (as last fetched), and the tag v1.0 does not exist yet, so the tag
                 lands on the merged, pushed commit and never moves (skipped with --allow-dirty)
Offline; needs git and the gold-v1 tag (`git fetch --tags`). Exit codes: 0 = ready to tag, 1 = a check failed.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

VERSION = "1.0.0"
TAG = "v1.0"
GOLD_TAG = "gold-v1"
NOTES = REPO / "docs" / "release_v1.0.md"
GOLD = "eval/gold_set.jsonl"
# The fields of each test that define it; the others are review metadata (verified_by, rationale, notes) or, for
# the injection tests, the scorer's match lists that the Week 0 review corrected before any measurement (e1922a0)
FROZEN_FIELDS = {"eval/refusal_tests.jsonl": ("id", "kind", "question", "expected_refused", "expected_rule"),
                 "eval/injection_tests.jsonl": ("id", "question", "planted_in", "planted_text", "injection_type")}
# Every committed generated report and its check; tests/test_release.py checks that no script with --check is missing
CHECKS = [["scripts/week1_notes.py", "--check"], ["scripts/week2_notes.py", "--check"],
          ["eval/report.py", "--check"], ["scripts/final_report.py", "--check"],
          ["eval/validate_gold.py", "--final"], ["eval/validate_test_sets.py", "--final"],
          ["eval/validate_dev_set.py", "--final"], ["eval/validate_dev_set.py", "eval/dev_set_v2.jsonl", "--final"]]


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, encoding="utf-8")


def versions() -> dict[str, str | None]:
    """The version each place states."""
    pyproject = re.search(r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    api = re.search(r'FastAPI\(title="FinSight", version="([^"]+)"', (REPO / "api" / "main.py").read_text(encoding="utf-8"))
    notes = NOTES.read_text(encoding="utf-8") if NOTES.exists() else ""
    stated = re.search(r"^# FinSight v(\d+\.\d+(?:\.\d+)?)", notes, re.M)
    return {"pyproject.toml": pyproject and pyproject.group(1), "api/main.py": api and api.group(1),
            "docs/release_v1.0.md": stated and stated.group(1)}


def check_version() -> list[str]:
    found = versions()
    out = [f"{where} says {v}, not {VERSION}" for where, v in found.items() if where != "docs/release_v1.0.md"
           and v != VERSION]
    if found["docs/release_v1.0.md"] not in (VERSION, TAG.lstrip("v")):
        out.append(f"docs/release_v1.0.md has no '# FinSight v{TAG.lstrip('v')}' title")
    return out


def frozen_differences(old_text: str, new_text: str, fields: tuple[str, ...]) -> list[str]:
    """What changed in a test set's defining fields between two versions of the file (by test id)."""
    def rows(text: str) -> dict[str, dict]:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
        return {item["id"]: item for item in items}
    old, new = rows(old_text), rows(new_text)
    out = [f"{i} was removed" for i in sorted(old.keys() - new.keys())]
    out += [f"{i} was added" for i in sorted(new.keys() - old.keys())]
    for i in sorted(old.keys() & new.keys()):
        out += [f"{i}: {f} changed" for f in fields if old[i].get(f) != new[i].get(f)]
    return out


def check_frozen() -> list[str]:
    if git("rev-parse", "--verify", "--quiet", f"{GOLD_TAG}^{{commit}}").returncode != 0:
        return [f"the tag {GOLD_TAG} is not in this clone: git fetch --tags"]
    out = []
    tagged = git("show", f"{GOLD_TAG}:{GOLD}")
    if tagged.returncode != 0 or tagged.stdout != (REPO / GOLD).read_text(encoding="utf-8"):
        out.append(f"{GOLD} differs from {GOLD_TAG}")
    for path, fields in FROZEN_FIELDS.items():
        old = git("show", f"{GOLD_TAG}:{path}")
        if old.returncode != 0:
            out.append(f"{path} is not in the tag {GOLD_TAG}")
            continue
        out += [f"{path}: {d}" for d in frozen_differences(old.stdout, (REPO / path).read_text(encoding="utf-8"), fields)]
    return out


def check_generated() -> list[str]:
    out = []
    for cmd in CHECKS:
        r = subprocess.run([sys.executable, *cmd], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if r.returncode != 0:
            last = (r.stderr.strip() or r.stdout.strip()).splitlines()[-1:] or ["(no output)"]
            out.append(f"python {' '.join(cmd)} failed: {last[0][:200]}")
    return out


def check_notes() -> list[str]:
    from scripts.final_report import untraced
    if not NOTES.exists():
        return [f"{NOTES.relative_to(REPO).as_posix()} is missing"]
    return [f"release notes, {p}" for p in untraced(NOTES.read_text(encoding="utf-8"))]


def check_clean() -> list[str]:
    status = git("status", "--porcelain").stdout.strip()
    return [f"uncommitted: {line}" for line in status.splitlines()[:10]]


def tag_target_problems(branch: str, head: str, upstream: str | None, tag_exists: bool) -> list[str]:
    """Why HEAD is not the commit to tag: it must be main, equal to origin/main, and the tag must not exist yet."""
    out = []
    if branch != "main":
        out.append(f"HEAD is {branch or 'detached'}, not main: git checkout main && git pull --ff-only")
    if upstream is None:
        out.append("origin/main is unknown: git fetch origin")
    elif head != upstream:
        out.append("main differs from origin/main: git pull --ff-only (and push nothing unreviewed)")
    if tag_exists:
        out.append(f"the tag {TAG} exists already: a released tag is never moved")
    return out


def check_tag_target() -> list[str]:
    upstream = git("rev-parse", "--verify", "--quiet", "origin/main")
    return tag_target_problems(git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
                               git("rev-parse", "HEAD").stdout.strip(),
                               upstream.stdout.strip() if upstream.returncode == 0 else None,
                               git("rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}").returncode == 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--allow-dirty", action="store_true",
                    help="before the merge: skip the clean-tree and tag-target checks")
    args = ap.parse_args(argv)
    steps: list[tuple[str, Callable[[], list[str]]]] = [
        ("version", check_version), ("frozen", check_frozen), ("generated", check_generated),
        ("notes", check_notes)] + ([] if args.allow_dirty else [("clean", check_clean),
                                                                 ("tag target", check_tag_target)])
    failed = False
    for name, step in steps:
        problems = step()
        failed |= bool(problems)
        print(f"{'OK  ' if not problems else 'FAIL'} {name}" + "".join(f"\n       {p}" for p in problems))
    print(f"\n{'NOT ready' if failed else 'Ready'} to tag {TAG} ({VERSION}).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
