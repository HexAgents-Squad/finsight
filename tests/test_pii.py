"""W2-13: personal data masked before logging and tracing (core/pii.py, core/tracing.py). Offline."""
from __future__ import annotations

import logging

import pytest

from core import pii
from core.tracing import TraceInfo, scrub

QUESTION = "I'm jane.doe@example.com, call me on (415) 555-2671 or +44 20 7946 0958: what was Apple's revenue?"


@pytest.fixture
def masking():
    """Log masking installed for one test, and the record factory restored after it (it is process-wide)."""
    before = logging.getLogRecordFactory()
    pii.install_log_masking()
    yield
    logging.setLogRecordFactory(before)


def test_mask_pii_masks_emails_and_phones_in_a_question():
    assert pii.mask_pii(QUESTION) == "I'm [email], call me on [phone] or [phone]: what was Apple's revenue?"


@pytest.mark.parametrize("text", ["Net sales were $391,035 million in fiscal 2024 (2023: 383,285).",
                                  "Accession 0000320193-24-000123, filed 2024-11-01, FY2023-2025.",
                                  "(123) 456 789 | +1.25 | 34.6%"])
def test_financial_text_is_never_masked(text):
    assert pii.mask_pii(text) == text


def test_tracing_masks_with_the_same_function():
    assert scrub({"q": QUESTION}) == {"q": pii.mask_pii(QUESTION)}


def test_trace_tags_are_masked():
    info = TraceInfo(question_id="Q1", run_id="r1", tags=("user:jane.doe@example.com",))
    assert info.all_tags("from 415-555-2671") == ["q:Q1", "run:r1", "user:[email]", "from [phone]"]


def test_a_span_status_message_is_masked():
    from core.tracing import _LangfuseSpan

    class Raw:
        def update(self, **kw):
            self.kw = kw

    raw = Raw()
    _LangfuseSpan(raw).update(level="ERROR", status_message=f"failed on {QUESTION}")
    assert "jane.doe" not in raw.kw["status_message"] and "[email]" in raw.kw["status_message"]


def test_every_log_record_is_masked_message_arguments_and_traceback(masking, caplog):
    log = logging.getLogger("finsight.test")
    with caplog.at_level(logging.INFO):
        log.info("question: %s", QUESTION)
        log.warning("plain %s message", "jane.doe@example.com")
        try:
            raise ValueError(f"bad question {QUESTION}")
        except ValueError:
            log.exception("tool failed")
    text = caplog.text
    assert "jane.doe" not in text and "555-2671" not in text and "7946" not in text
    assert "question: I'm [email], call me on [phone]" in text and "plain [email] message" in text
    assert "ValueError: bad question I'm [email]" in text                       # the traceback too


def test_records_from_any_logger_are_masked_even_propagated_ones(masking, caplog):
    with caplog.at_level(logging.INFO):
        logging.getLogger("uvicorn.error").info("from %s", "jane.doe@example.com")
        logging.getLogger("agent.composer").info("from %s", "jane.doe@example.com")
    assert caplog.text.count("[email]") == 2 and "jane.doe" not in caplog.text


def test_installing_twice_changes_nothing(masking):
    factory = logging.getLogRecordFactory()
    pii.install_log_masking()
    assert logging.getLogRecordFactory() is factory


def test_a_bad_format_string_is_left_to_logging(masking):
    record = logging.getLogRecordFactory()("x", logging.INFO, __file__, 1, "%d", ("not a number",), None)
    assert record.msg == "%d" and record.args == ("not a number",)          # unchanged: logging reports it


def test_the_api_and_the_clis_install_it():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for rel in ("api/main.py", "scripts/ask.py", "scripts/trace_demo.py"):
        assert "install_log_masking()" in (root / rel).read_text(encoding="utf-8"), rel


def test_creating_the_app_installs_masking():
    before = logging.getLogRecordFactory()
    try:
        logging.setLogRecordFactory(logging.LogRecord)
        from api.main import create_app
        create_app(graph=object())
        assert getattr(logging.getLogRecordFactory(), "_finsight_masks_pii", False)
    finally:
        logging.setLogRecordFactory(before)
