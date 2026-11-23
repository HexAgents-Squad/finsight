"""Shared test setup."""
import logging

import pytest

from core.settings import Settings

CORPUS_OPTION = "--corpus"


def pytest_addoption(parser):
    parser.addoption(CORPUS_OPTION, action="store_true", default=False,
                     help="also run the tests marked corpus, which scan the whole downloaded corpus (minutes, W3-07)")


def pytest_collection_modifyitems(config, items):
    """The corpus scans (pytest.mark.corpus) are opt-in, so the default suite runs in under two minutes on a
    checkout with data (W3-07); CI has no corpus, so they would skip there anyway."""
    if config.getoption(CORPUS_OPTION):
        return
    skip = pytest.mark.skip(reason=f"scans the whole corpus: run with {CORPUS_OPTION}")
    for item in items:
        if item.get_closest_marker("corpus"):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_settings_from_the_shell(monkeypatch):
    """Tests build Settings with `_env_file=None`, but pydantic-settings still reads the process environment: a
    developer who exported LLM_CA_CERT or LLM_ENABLE_THINKING to try the hosted endpoint would get failures CI
    never sees. Every setting is removed from the environment for each test; a test that needs one sets it."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    # Never the repo's reply cache, a tokenizer download or real back-off delays, whatever a test builds
    for name, value in {"LLM_CACHE_DIR": "", "LLM_CONTEXT_TOKENS": "", "LLM_RETRY_BASE_S": "0",
                        "LLM_MIN_INTERVAL_S": "0"}.items():
        monkeypatch.setenv(name, value)


@pytest.fixture(autouse=True)
def _log_record_factory_restored():
    """api.main.create_app installs the PII log masking process-wide (W2-13, core/pii.py): put the record factory
    back after each test, so no test sees another test's logging setup."""
    before = logging.getLogRecordFactory()
    yield
    logging.setLogRecordFactory(before)
