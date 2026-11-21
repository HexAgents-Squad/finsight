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


def make_record(msg, args, exc_info=None):
    return logging.getLogRecordFactory()("x", logging.INFO, __file__, 1, msg, args, exc_info)


def test_a_bad_format_string_is_left_to_logging_with_its_text_masked(masking):
    record = make_record("%d for jane@corp.example", ("call 415-555-2671",))
    assert record.msg == "%d for [email]" and record.args == ("call [phone]",)     # logging reports the bad format


def test_arguments_keep_their_shape_and_types(masking):
    """Handlers may unpack the arguments (uvicorn's access formatter does), so a tuple stays a tuple of the same
    length, a mapping stays a mapping, numbers stay numbers and only text with personal data changes."""
    record = make_record("%s %d %.1f %r", ("jane@corp.example", 7, 2.5, ["clean"]))
    assert record.args == ("[email]", 7, 2.5, ["clean"]) and record.getMessage() == "[email] 7 2.5 ['clean']"
    mapping = make_record("%(who)s asked %(n)d", {"who": "jane@corp.example", "n": 3})
    assert mapping.args == {"who": "[email]", "n": 3} and mapping.getMessage() == "[email] asked 3"

    class Person:
        def __str__(self):
            return "Person(jane@corp.example)"
    record = make_record("%s", (Person(),))                          # an object's text: the message, flattened
    assert record.getMessage() == "Person([email])" and record.args is None


class Unprintable:
    def __str__(self):
        raise RuntimeError("no text")

    __repr__ = __str__


def test_the_factory_never_raises_into_the_callers_log_call(masking):
    """An argument or message whose text fails is the handler's to report (logging's handleError), as without the
    factory: making the record never raises."""
    record = make_record("obj %r for jane@corp.example", (Unprintable(),))
    assert record.msg == "obj %r for [email]"                         # the template's own text still masked
    assert make_record(Unprintable(), None).msg.__class__ is Unprintable


@pytest.mark.parametrize("msg", [ValueError("no filing for jane@corp.example"), {"q": "call 415-555-2671"}])
def test_a_message_that_is_not_a_string_is_masked(masking, caplog, msg):
    with caplog.at_level(logging.INFO):
        logging.getLogger("finsight.test").error(msg)
    assert "jane@" not in caplog.text and "555-2671" not in caplog.text
    assert "[email]" in caplog.text or "[phone]" in caplog.text


def test_masking_a_long_run_without_an_at_sign_takes_linear_time():
    import time
    start = time.perf_counter()
    for text in ("a" * 50_000, "1-" * 25_000, "x.y" * 20_000):
        pii.mask_pii(text)
    assert time.perf_counter() - start < 1.0                         # was 17 s for one 50k run (quadratic)


@pytest.mark.parametrize("phone", ["415 555-2671", "+44 (0)20 7946 0958", "+44 (0) 20 7946 0958"])
def test_common_phone_shapes_are_masked(phone):
    assert pii.mask_pii(f"call {phone} today") == "call [phone] today"


@pytest.mark.parametrize("text", ["jane%40corp.example", "q=jane.doe%40corp.co.uk&x=1"])
def test_a_url_encoded_email_is_masked(text):
    assert "corp" not in pii.mask_pii(text) and "[email]" in pii.mask_pii(text)


def test_personal_data_in_the_format_string_itself_is_masked(masking):
    record = make_record("user jane@corp.example asked %s", ("about revenue",))
    assert record.getMessage() == "user [email] asked about revenue"


def test_uvicorns_formatters_still_work_and_print_masked_lines(masking):
    """The access log unpacks five arguments (client, method, path, version, status); with args=None it raised and
    the line was lost (the first W2-13 version)."""
    from uvicorn.logging import AccessFormatter, DefaultFormatter
    access = make_record('%s - "%s %s HTTP/%s" %d',
                         ("127.0.0.1:5000", "GET", "/ask?email=jane@corp.example", "1.1", 200))
    line = AccessFormatter(fmt="%(client_addr)s %(request_line)s %(status_code)s", use_colors=False).format(access)
    assert line == "127.0.0.1:5000 GET /ask?email=[email] HTTP/1.1 200 OK"
    default = make_record("question from %s", ("+1 415-555-2671",))
    assert DefaultFormatter(fmt="%(levelprefix)s %(message)s", use_colors=False).format(default).endswith(
        "question from [phone]")


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
