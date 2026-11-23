"""W3-10: eval/run_api_safety.py. Offline: stub nodes behind the real FastAPI app (no endpoint, no data/, no
Langfuse); the committed results are read as files."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace

import pytest

from agent import graph as g
from agent.stubs import stub_nodes
from api.main import ApiConfig
from eval import run_api_safety as ras

CONFIG = replace(ApiConfig(limits=g.Limits(max_steps=10, max_tokens=12_000, timeout_s=10)), trace=False)


def test_a_question_goes_through_post_ask_of_the_real_app():
    async def work(client):
        return await ras.ask(client, "What were Apple's total net sales in fiscal 2024?")
    status, body, headers = asyncio.run(ras.serve(g.build_graph(stub_nodes()), CONFIG, work))
    assert status == 200 and body["session_id"] and "answer" in body and "x-session-id" in headers


def test_the_session_limit_answers_429_with_retry_after_and_the_repeats_come_from_the_cache(monkeypatch):
    import agent.nodes
    monkeypatch.setattr(agent.nodes, "real_nodes", lambda llm: stub_nodes())     # the limit, not the agent
    config = replace(CONFIG, rate_limit="3/minute")
    out = asyncio.run(ras.limit_and_cache(None, config))
    statuses = [r["status"] for r in out["responses"]]
    assert statuses == [200, 200, 200, 200, 429]                 # the first in its own bucket, then 3 + 1 refused
    assert out["responses"][-1]["retry_after"] == "60"


def test_log_evidence_counts_records_and_keeps_only_masked_demo_lines():
    lines = [("httpx", "DEBUG httpx: HTTP Request: POST https://10.0.0.1/v1/chat [phone]"),
             ("openai", "DEBUG openai: jane.doe@example.com leaked"),
             (ras.DEMO_LOGGER, "INFO demo: question received: e-mail [email] at https://x.example/a")]
    ev = ras.pii_log_evidence(lines)
    assert ev["records_by_logger"] == {"httpx": 1, "openai": 1} and ev["api_records"] == 2
    assert (ev["api_records_with_raw_pii"], ev["api_records_with_a_mask"]) == (1, 1)
    assert ev["demo_lines"] == ["INFO demo: question received: e-mail [email] at <url>"]     # no address kept
    assert "10.0.0.1" not in json.dumps(ev)                                       # an HTTP line is never stored


def test_the_masking_create_app_installs_masks_a_log_call():
    """What the demonstration shows: after create_app, a log call carrying an address is masked for any handler."""
    from api.main import create_app
    create_app(graph=g.build_graph(stub_nodes()), config=CONFIG)
    capture = ras.Capture()
    log = logging.getLogger(ras.DEMO_LOGGER)
    log.addHandler(capture)
    log.setLevel(logging.INFO)
    try:
        log.info("question received: %s", ras.PII_PROMPTS[0][1])
        log.info("question received: %s", ras.PII_PROMPTS[1][1])
    finally:
        log.removeHandler(capture)
    text = " ".join(line for _, line in capture.lines)
    assert "[email]" in text and "[phone]" in text and not any(p in text for p in ras.RAW_PII)


def test_the_committed_results_hold_every_test_and_the_report_is_current():
    rows = [json.loads(x) for x in (ras.OUT_DIR / "api_tests.jsonl").read_text(encoding="utf-8").splitlines() if x]
    sets = {s: [r["id"] for r in rows if r["set"] == s] for s in ("refusal", "injection", "pii")}
    assert (len(sets["refusal"]), len(sets["injection"]), len(sets["pii"])) == (22, 12, 3)
    meta = json.loads((ras.OUT_DIR / "meta.json").read_text(encoding="utf-8"))
    assert meta["runs"] and all("requests" in r for r in meta["runs"])
    assert ras.report(ras.OUT_DIR) == ras.REPORT.read_text(encoding="utf-8")       # regenerated from the files


def test_a_merge_replaces_only_the_parts_run(tmp_path):
    first = {"refusal": [{"set": "refusal", "id": "RF01"}], "injection": [{"set": "injection", "id": "INJ01"}],
             "pii": [{"set": "pii", "id": "PII-1"}], "log_lines": [], "limit": {"responses": []},
             "requests": {"live": 5, "cache_hits": 1, "http_retries": 0}}
    ras.write_run(first, tmp_path, "t1", ras.PARTS, True)
    again = {"pii": [{"set": "pii", "id": "PII-1", "new": True}], "log_lines": [],
             "requests": {"live": 0, "cache_hits": 2, "http_retries": 0}}
    ras.write_run(again, tmp_path, "t2", ("pii",), False)
    rows = [json.loads(x) for x in (tmp_path / "api_tests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["RF01", "INJ01", "PII-1"] and rows[-1].get("new")
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert [(r["started"], r["pii_traced"]) for r in meta["runs"]] == [("t1", True), ("t2", False)]


@pytest.mark.parametrize("raw", ["jane.doe@example.com", "+1 415 555 2671"])          # the prompts' own forms
def test_the_made_up_contact_details_are_masked_by_the_shared_patterns(raw):
    """A bare "415 555 2671" is left alone on purpose (a row of figures, ADR-035); RAW_PII lists it to catch a leak."""
    from core.pii import mask_pii
    assert raw not in mask_pii(f"reach me at {raw} please")
