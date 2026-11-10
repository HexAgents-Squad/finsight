"""FinSight settings, read from environment variables or the repo's `.env` (template: `.env.example`).

Switching between the local Ollama stand-in and the hosted endpoint (Qwen3-8B over HTTPS with the group's CA
certificate, docs/week1_plan.txt §2.1) must only mean editing `.env`.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always the repo root's .env, whichever folder a script is started from; relative paths in it (LLM_CA_CERT)
# are resolved against the repo root for the same reason.
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"
HTTPS = "https://"


class SecSettings(BaseSettings):
    """SEC access only: the download steps (ingest/01, ingest/05) need nothing else, so an incomplete LLM
    section in .env does not stop them."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore", frozen=True,
                                      str_strip_whitespace=True)

    # SEC EDGAR fair-access policy: automated clients must identify themselves
    SEC_USER_AGENT: str = Field(..., min_length=1, description="User-Agent header required by SEC EDGAR")


class Settings(SecSettings):
    """Configuration for the LLM client, embeddings, agent limits and SEC access."""

    # Chat model on an OpenAI-compatible endpoint (min_length: an empty value in .env is an error here,
    # not a confusing failure inside the openai client)
    LLM_BASE_URL: str = Field(..., min_length=1, description="Target OpenAI-compatible endpoint URL")
    # SecretStr: the team key must never appear in a repr, a log line or a validation error message
    LLM_API_KEY: SecretStr = Field(..., min_length=1, description="API key for the endpoint ('ollama' locally)")
    LLM_MODEL: str = Field(..., min_length=1, description="Chat model name")
    # Worst case for one call is LLM_TIMEOUT x (LLM_HTTP_RETRIES + 1) plus the waits (2 + 4 + 8 s by default, a
    # server's Retry-After up to 30 s each); the agent enforces its per-query time limit on top of this.
    LLM_TIMEOUT: float = Field(30.0, gt=0, allow_inf_nan=False, description="HTTP request timeout in seconds")
    # Two separate retry budgets. core.llm retries 429 (one active request per team key, 60/min), 5xx, timeouts
    # and network errors after LLM_RETRY_BASE_S x 2^n seconds (2, 4, 8: the learner guide's advice), the openai
    # SDK's own retries being off so the two never multiply; chat_json re-prompts after invalid JSON (spec: 2).
    LLM_HTTP_RETRIES: int = Field(3, ge=0, le=10, description="retries for 429/5xx/timeout/network errors")
    LLM_RETRY_BASE_S: float = Field(2.0, ge=0, le=30, allow_inf_nan=False,
                                    description="first retry delay; doubles each retry (each wait at most 30 s)")
    # One request at a time per endpoint in a process, and at most one start per interval (60 requests/min)
    LLM_MIN_INTERVAL_S: float = Field(1.0, ge=0, le=60, allow_inf_nan=False,
                                      description="minimum seconds between request starts")
    # Disk cache of replies (spec §9): an identical request (server, model, messages, temperature, max_tokens,
    # thinking, sample) is answered from here, so an evaluation re-run is identical and costs no calls. Empty = no cache.
    LLM_CACHE_DIR: Path | None = Field(Path("data/cache/llm"), description="reply cache; relative to the repo")
    LLM_JSON_RETRIES: int = Field(2, ge=0, description="chat_json re-prompts after invalid output")
    # Caps one reply, so a runaway generation from a small model cannot use the whole per-query token budget
    LLM_MAX_COMPLETION_TOKENS: int = Field(1024, gt=0, description="max_tokens sent with every chat request")
    # The endpoint's limit for prompt + reply (8,192). A request that cannot fit is refused before it is sent,
    # counted with the Qwen3 tokenizer (core/tokens.py) and its chat template; empty = no check.
    LLM_CONTEXT_TOKENS: int | None = Field(8192, gt=0, description="prompt + max_tokens limit per call")
    # The hosted endpoint's certificate is issued by the group's own CA (a Caddy local root), which no default
    # trust store has. HTTPS is verified against this file instead; it is never switched off. Trust the CA
    # file, never the server's own certificate: the server certificate is renewed every few hours.
    LLM_CA_CERT: Path | None = Field(None, description="PEM CA certificate for LLM_BASE_URL; empty = the HTTP "
                                                       "library's default trust store")
    # Qwen3 reasoning mode, sent as chat_template_kwargs.enable_thinking (Ollama ignores the field). Off: the
    # answer comes directly and output tokens go to the answer, not to reasoning. None (empty in .env) = do
    # not send the field, for an endpoint that rejects unknown request fields.
    LLM_ENABLE_THINKING: bool | None = Field(False, description="Qwen3 thinking switch; empty = not sent")

    # Embeddings. The default is the local model the retrieval index was built with (ADR-013): a .env without
    # the key must still match the index, or every search is refused as built with another model.
    EMBED_BACKEND: Literal["openai", "sentence-transformers"] = "sentence-transformers"
    EMBED_BASE_URL: str | None = Field(None, description="Embeddings endpoint; empty = LLM_BASE_URL")
    EMBED_API_KEY: SecretStr | None = Field(None, description="Key for EMBED_BASE_URL; empty = LLM_API_KEY")
    EMBED_MODEL: str = Field("nomic-embed-text", description="Model name on the embeddings endpoint")
    EMBED_LOCAL_MODEL_NAME: str = Field("all-MiniLM-L6-v2", description="HuggingFace sentence-transformers model")
    EMBED_BATCH_SIZE: int = Field(64, gt=0, description="Texts per embeddings request")

    # Agent limits (CLAUDE.md "Agent robustness")
    MAX_AGENT_STEPS: int = Field(10, gt=0)
    MAX_TOKENS_PER_QUERY: int = Field(12_000, gt=0)
    # spec §9: the agent wraps a whole query in this limit (one LLM call alone can take LLM_TIMEOUT x retries)
    QUERY_TIMEOUT_S: float = Field(60.0, gt=0, description="seconds allowed for one question, end to end")

    @field_validator("LLM_CA_CERT", "LLM_ENABLE_THINKING", "EMBED_API_KEY", "LLM_CACHE_DIR", "LLM_CONTEXT_TOKENS",
                     mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        """`LLM_CA_CERT=` in .env means "no certificate", not a path to the current folder."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("LLM_CA_CERT")
    @classmethod
    def _ca_file_for_https(cls, value: Path | None, info: ValidationInfo) -> Path | None:
        """A wrong path must fail at start-up with the path in the message, not as a TLS error on the first
        call. Relative paths are taken from the repo root, so every script finds the same file. A CA only means
        something for an https:// URL; with http:// it would be silently unused while the user believes the
        connection is verified. A field validator, not a model validator: a model-level error message would
        quote every input value, the API key included."""
        if value is None:
            return None
        base_url = info.data.get("LLM_BASE_URL")        # declared above, so already validated (absent if invalid)
        if base_url is not None and not base_url.lower().startswith(HTTPS):
            raise ValueError("LLM_CA_CERT is set but LLM_BASE_URL is not an https:// URL")
        path = value.expanduser()
        path = path if path.is_absolute() else REPO_ROOT / path
        if not path.is_file():
            raise ValueError(f"no certificate file at {path}")
        return path

    @field_validator("LLM_CACHE_DIR")
    @classmethod
    def _cache_dir_from_repo_root(cls, value: Path | None) -> Path | None:
        """The same cache whichever folder a script is started from."""
        if value is None:
            return None
        path = value.expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    @field_validator("LLM_CONTEXT_TOKENS")
    @classmethod
    def _context_leaves_room_for_the_prompt(cls, value: int | None, info: ValidationInfo) -> int | None:
        """A context no larger than the reply cap would refuse every request."""
        cap = info.data.get("LLM_MAX_COMPLETION_TOKENS")
        if value is not None and cap is not None and value <= cap:
            raise ValueError(f"LLM_CONTEXT_TOKENS ({value}) must exceed LLM_MAX_COMPLETION_TOKENS ({cap})")
        return value

    @field_validator("EMBED_BACKEND", mode="before")
    @classmethod
    def _backend_name(cls, value: object) -> object:
        """'Sentence-Transformers' or ' openai' typed in .env still select the backend."""
        return value.strip().lower() if isinstance(value, str) else value


@cache
def get_settings() -> Settings:
    """The settings, read from the environment and `.env` once per process."""
    return Settings()


@cache
def get_sec_settings() -> SecSettings:
    """The SEC settings alone, read from the environment and `.env` once per process."""
    return SecSettings()
