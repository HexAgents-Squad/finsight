"""The demo (W3-14, ADR-049): every scripted question is one the system answered and committed, the offline demo mode
cannot reach the endpoint, and the demo script is traceable. Offline: no browser, no endpoint, no Docker."""
from __future__ import annotations

import re

import pytest
import yaml

import scripts.demo_autopilot as auto
import scripts.demo_injection as inj
from scripts.final_report import untraced

ROOT = auto.ROOT
SCRIPT = (ROOT / "docs" / "demo" / "demo_script.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("step", auto.STEPS, ids=lambda s: s.id)
def test_each_step_asks_a_committed_question_and_expects_its_committed_answer(step):
    question, answer = auto.committed_answer(step)
    assert step.question == question
    assert step.expect in answer, (step.expect, answer)


def test_the_steps_follow_the_plan_and_only_the_follow_up_keeps_the_conversation():
    assert [s.new_conversation for s in auto.STEPS].count(False) == 1
    follow = next(i for i, s in enumerate(auto.STEPS) if not s.new_conversation)
    assert auto.STEPS[follow - 1].id == "demo-06" and auto.STEPS[follow].id == "demo-07"
    assert [s.open_trace for s in auto.STEPS].count(True) == 1
    for step in auto.STEPS:                       # the script's table names every step's question
        assert step.question in SCRIPT, step.id


def test_the_demo_runs_on_verified_answers_and_refusals_only():
    """A step must show what the system does well: a verified answer or a refusal (demo-03 was not verified)."""
    import json
    rows = {}
    for path in {s.source for s in auto.STEPS}:
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    for step in auto.STEPS:
        row = rows[step.id]
        api = row.get("api") or {}
        assert row.get("verified", api.get("verified")) or row.get("refused", api.get("refused")), step.id


def test_the_demo_script_is_traceable_and_names_every_member():
    assert untraced(SCRIPT) == []
    for name in ("Alok", "Giridharan", "Chandra Shekhar", "Amresh", "Anirudh"):
        assert name in SCRIPT


def test_the_injection_demo_uses_a_committed_test_and_never_asks_the_endpoint():
    from eval.run_injection import load_tests
    assert inj.DEFAULT_TEST in {t.id for t in load_tests()}

    class Client:
        async def _fetch(self, *a):
            raise AssertionError("the endpoint was asked")
    client = inj.cache_only(Client())
    with pytest.raises(inj.NotCached):
        import asyncio
        asyncio.run(client._fetch({}, "m", None))


def test_the_offline_demo_mode_blocks_the_endpoint_and_tracing():
    offline = yaml.safe_load((ROOT / "docker-compose.offline.yml").read_text(encoding="utf-8"))
    assert set(offline["services"]) == {"api"}                    # the UI is unchanged
    env = offline["services"]["api"]["environment"]
    assert env["HTTPS_PROXY"] == env["HTTP_PROXY"] == "http://127.0.0.1:9"   # a proxy that does not exist
    no_proxy = env["NO_PROXY"].split(",")
    assert "api" in no_proxy and "huggingface.co" in no_proxy and not any(re.match(r"\d", h) for h in no_proxy
                                                                           if h not in ("127.0.0.1",))
    assert env["EMBED_BACKEND"] == "sentence-transformers" and env["EMBED_LOCAL_MODEL_NAME"] == "BAAI/bge-small-en-v1.5"
    assert env["LANGFUSE_ENABLED"] == "false"
    assert "LLM_BASE_URL" not in env                             # the reply cache's key names the endpoint


def test_the_makefile_starts_the_offline_demo_mode():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^demo-offline: check-data\b.*\n(\t.*\n)*\t\$\(COMPOSE\) -f docker-compose\.yml "
                     r"-f docker-compose\.offline\.yml up .*api ui", makefile, re.M)
