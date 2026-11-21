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
  North American: 415-555-2671, 415.555.2671, (415) 555-2671, an optional leading 1 or +1; plain spaces only with
    a leading 1 ("1 800 555 1234"), since "391 035 2024" in a table is figures, not a phone number.
  International: +country code, then 2 to 5 groups of digits joined by a space or a hyphen ("+33 1 23 45 67 89"),
    at least 10 digits in all (checked in _mask_phone), or the compact E.164 form, "+" and 11 to 15 digits
    ("+14155552671"). The compact form also covers a signed figure of 11 or more digits without separators
    ("+93736000000"): filings and FinSight's tools write figures with commas or without a sign, and a typed
    phone number is the likelier reading, so it is masked. "+3.5 12.25" or "+12 345 678" stays a figure.
Not recognised, on purpose: "415 555 2671" (bare spaces), local numbers without a country code outside North
America ("020 7946 0958"), and anything without separators or a "+": all too close to columns of figures.
Separators are never line breaks. This module imports nothing from FinSight, so any module can use it.

Log records: the message and string arguments are masked in place, keeping the arguments' shape (a handler may
unpack them, as uvicorn's access log does), and so are the exception text and stack. Not masked: `exc_info`
itself (a handler that re-formats the live exception sees its raw text) and attributes passed with `extra=`.
FinSight's own handlers use the default formatter, which reads the masked exc_text.
"""
from __future__ import annotations

import logging
import re
from typing import Any

MASK_EMAIL = "[email]"
MASK_PHONE = "[phone]"
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?<![\w.+-])(?:\+\d{1,3}(?:[ -]\d{1,6}){2,5}|\+[1-9]\d{10,14}"
                   r"|(?:\+?1[ .-]?)?(?:\(\d{3}\) ?\d{3}[ .-]\d{4}|\d{3}-\d{3}-\d{4}|\d{3}\.\d{3}\.\d{4})"
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


def _mask_arg(value: Any) -> Any:
    """A log argument with personal data masked: a string masked, another object replaced by its masked text only if
    that text holds personal data (numbers and clean objects stay as they are, so "%d" and "%r" still work)."""
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, (int, float)) or value is None:
        return value
    text = str(value)
    masked = mask_pii(text)
    return value if masked == text else masked


def _mask_args(args: Any) -> Any:
    """The record's arguments masked in the shape they came in (a tuple, or one mapping for "%(name)s"): handlers
    such as uvicorn's access formatter unpack them."""
    if isinstance(args, dict):
        return {key: _mask_arg(value) for key, value in args.items()}
    if isinstance(args, tuple):
        return tuple(_mask_arg(value) for value in args)
    return args


def _masked_record_factory(previous: Any) -> Any:
    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        if record.args:
            record.args = _mask_args(record.args)
        try:
            message = record.getMessage()                # the message with its (masked) arguments filled in
        except Exception:                                # noqa: BLE001 - a bad format string is logging's to report
            message = None
        if isinstance(record.msg, str) and (message is None or mask_pii(message) != message):
            if message is None:                          # logging reports the bad format; its text is masked
                record.msg = mask_pii(record.msg)
            else:                                        # personal data in the format string itself: send it filled in
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
