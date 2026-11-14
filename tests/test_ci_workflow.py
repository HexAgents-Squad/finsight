"""The CI workflow must be valid YAML with the expected steps: an invalid file makes GitHub skip every job
and show no checks at all, which looks like "nothing ran" rather than a failure (happened after PR #15)."""
from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).resolve().parents[1] / ".github" / "workflows").glob("*.yml"))
REQUIRED_RUNS = ["ruff check .", "pytest -q", "python eval/validate_gold.py --final", "python eval/validate_test_sets.py",
                 "python eval/validate_dev_set.py --final"]


def test_there_is_a_workflow():
    assert WORKFLOWS


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_is_valid_yaml_with_jobs(path):
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict) and workflow.get("jobs"), f"{path.name} has no jobs"
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            assert isinstance(step.get("name", ""), str)


def test_ci_runs_every_check():
    ci = yaml.safe_load((WORKFLOWS[0].parent / "ci.yml").read_text(encoding="utf-8"))
    runs = [step.get("run") for job in ci["jobs"].values() for step in job["steps"]]
    assert all(cmd in runs for cmd in REQUIRED_RUNS), runs
