"""Masking of personal data in what FinSight writes down (W2-08 for traces, W2-13 for logs; plan §10).

    mask_pii(text)          e-mail addresses -> [email], phone numbers -> [phone]; financial text untouched
    install_log_masking()   every log record from then on is masked (idempotent; the API and the CLIs call it)

A user can type an e-mail address or a phone number into a question. FinSight answers it (the planner refuses a
request for someone's personal information, R5), but the question must not reach a log line or a trace as typed:
core/tracing.scrub masks everything sent to Langfuse, and install_log_masking masks every log record, its message,
its arguments and an exception's text, whichever logger or handler writes it. The run's own records (the reply
cache, eval runs files) keep the text: they are local, git-ignored or evaluation data without users.

The patterns are chosen so financial text is never masked (tests/test_tracing.py and tests/test_pii.py): a run of
years ("2023 2024 2025"), a bare figure (383285000000 or 4155552671), a date (2024-09-28), a range (2023-2025), an
accession number (0000320193-24-000123), a negative column "(123) 456 789" or a signed change "+1.25 34.6" all stay.
  North American: 415-555-2671, 415.555.2671, (415) 555-2671, an optional leading 1; plain spaces only with a
    leading 1 ("1 800 555 1234"), since "391 035 2024" in a table is figures, not a phone number.
  International: +country code, then 2 to 4 groups of at least 2 digits joined by a space or a hyphen, at least
    10 digits in all (checked in _mask_phone); "+383285000000" or "+3.5 12.25" is a signed figure.
Not recognised, on purpose: "415 555 2671" (bare spaces), local numbers without a country code outside North
America ("020 7946 0958"), and anything without separators: all too close to columns of figures. Separators are
never line breaks. This module imports nothing from FinSight, so any module can use it.
"""
from __future__ import annotations

import logging
import re
from typing import Any

MASK_EMAIL = "[email]"
MASK_PHONE = "[phone]"
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?<![\w.+-])(?:\+\d{1,3}(?:[ -]\d{2,5}){2,4}"
                   r"|(?:1[ .-]?)?(?:\(\d{3}\) ?\d{3}[ .-]\d{4}|\d{3}-\d{3}-\d{4}|\d{3}\.\d{3}\.\d{4})"
                   r"|1 \d{3} \d{3} \d{4})(?![\w-]|[.,]\d)")
MIN_INTERNATIONAL_DIGITS = 10


def _mask_phone(match: re.Match[str]) -> str:
    text = match.group()
    if text.startswith("+") and sum(c.isdigit() for c in text) < MIN_INTERNATIONAL_DIGITS:
        return text                                      # "+12 345 678": a signed figure, not a phone number
    return MASK_PHONE


def mask_pii(text: str) -> str:
    """`text` with e-mail addresses and phone numbers masked."""
    return PHONE.sub(_mask_phone, EMAIL.sub(MASK_EMAIL, text))


# ----------------------------------------------------------------------------------------------- logging
_FACTORY_MARK = "_finsight_masks_pii"


def _masked_record_factory(previous: Any) -> Any:
    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        try:
            message = record.getMessage()                # the message with its arguments filled in
        except Exception:                                # noqa: BLE001 - a bad format string is logging's to report
            return record
        record.msg, record.args = mask_pii(message), None
        if record.exc_info and not record.exc_text:      # the traceback's text, formatted once and masked
            record.exc_text = mask_pii(logging.Formatter().formatException(record.exc_info))
        if record.stack_info:
            record.stack_info = mask_pii(record.stack_info)
        return record

    setattr(factory, _FACTORY_MARK, True)
    return factory


def install_log_masking() -> None:
    """From now on every log record (any logger, any handler: uvicorn's, a file, the console) has its message, its
    arguments and its exception text masked. Installed once per process; calling it again changes nothing."""
    current = logging.getLogRecordFactory()
    if not getattr(current, _FACTORY_MARK, False):
        logging.setLogRecordFactory(_masked_record_factory(current))
