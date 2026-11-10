"""Shared test setup."""
import pytest

from core.settings import Settings


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
