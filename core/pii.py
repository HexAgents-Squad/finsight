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
  North American: 415-555-2671, 415.555.2671, (415) 555-2671, 415 555-2671, an optional leading 1 or +1; plain
    spaces only with a leading 1 ("1 800 555 1234"), since "391 035 2024" in a table is figures, not a phone number.
  International: +country code, then 2 to 5 groups of digits joined by a space or a hyphen ("+33 1 23 45 67 89"),
    at least 10 digits in all (checked in _mask_phone); a trunk "(0)" ("+44 (0)20 7946 0958"); or the compact
    E.164 form, "+" and 11 to 15 digits ("+14155552671"). The compact form also covers a signed figure of 11 or
    more digits without separators ("+93736000000"), and the grouped form a signed row of figures with 10 or more
    digits ("+383 285 000 000"): filings and FinSight's tools write figures with commas or without a sign, and a
    typed phone number is the likelier reading, so they are masked (logs and traces only, never an answer).
    "+3.5 12.25" or "+12 345 678" stays a figure.
Not recognised, on purpose: "415 555 2671" (bare spaces), local numbers without a country code outside North
America ("020 7946 0958"), and anything without separators or a "+": all too close to columns of figures.
Separators are never line breaks. This module imports nothing from FinSight, so any module can use it.

Log records: string arguments are masked in place, keeping the arguments' shape (a handler may unpack them, as
uvicorn's access log does); if the filled-in message still holds personal data (in the template, in an object's
text, a non-str message, or a number formed across arguments), the record carries the masked message with no
arguments. The exception text and stack are masked too. The factory never raises into the caller's log call.
Not masked: `exc_info` itself (a handler that re-formats the live exception sees its raw text) and attributes
passed with `extra=`. FinSight's own handlers use the default formatter, which reads the masked exc_text.
"""
from __future__ import annotations

import logging
import re
from typing import Any

MASK_EMAIL = "[email]"
MASK_PHONE = "[phone]"
# Anchored and bounded (a local part of at most 64 characters, RFC 5321), so a long run of word characters without
# an "@" costs linear time, not quadratic; "%40" is an "@" in a URL-encoded query
EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]{1,64}(?:@|%40)[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?<![\w.+-])(?:\+\d{1,3}(?:[ -]\d{1,6}){2,5}|\+[1-9]\d{10,14}|\+\d{1,3} ?\(0\) ?\d{1,4}(?:[ -]\d{2,5}){1,3}"
                   r"|(?:\+?1[ .-]?)?(?:\(\d{3}\) ?\d{3}[ .-]\d{4}|\d{3}-\d{3}-\d{4}|\d{3}\.\d{3}\.\d{4}"
                   r"|\d{3} \d{3}-\d{4})"
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
    """A string argument masked; any other object is left as it is (no str() call of ours on it: the filled-in
    message is checked below, and only a message that still holds personal data is flattened)."""
    return mask_pii(value) if isinstance(value, str) else value


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
        except Exception:                                # noqa: BLE001 - a bad format string or a failing __str__:
            message = None                               # logging reports it when the record is handled
        if message is None:                              # the template's own text masked, whatever its type
            try:
                template = record.msg if isinstance(record.msg, str) else str(record.msg)
                if mask_pii(template) != template:
                    record.msg = mask_pii(template)
            except Exception:                            # noqa: BLE001 - never raise into the caller's log call
                pass
        elif mask_pii(message) != message:               # personal data in the template itself (or a non-str message,
            record.msg, record.args = mask_pii(message), None    # an object's text): sent filled in and masked
        if record.exc_info and not record.exc_text:      # the traceback's text, formatted once and masked
            try:
                record.exc_text = mask_pii(logging.Formatter().formatException(record.exc_info))
            except Exception:                            # noqa: BLE001 - the handler formats it then, unmasked
                pass
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
