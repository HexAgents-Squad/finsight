"""FinSight settings, read from environment variables or the repo's `.env` (template: `.env.example`).

Switching from the local Ollama stand-in to the official endpoint must only mean editing `.env`.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always the repo root's .env, whichever folder a script is started from
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    """Configuration for the LLM client, embeddings, agent limits and SEC access."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore", frozen=True,
                                      str_strip_whitespace=True)

    # Chat model on an OpenAI-compatible endpoint (min_length: an empty value in .env is an error here,
    # not a confusing failure inside the openai client)
    LLM_BASE_URL: str = Field(..., min_length=1, description="Target OpenAI-compatible endpoint URL")
    LLM_API_KEY: str = Field(..., min_length=1, description="API key for the endpoint ('ollama' locally)")
    LLM_MODEL: str = Field(..., min_length=1, description="Chat model name")
    # Worst case for one call is LLM_TIMEOUT x (LLM_HTTP_RETRIES + 1) plus back-off; the agent enforces its
    # overall per-query time limit on top of this.
    LLM_TIMEOUT: float = Field(30.0, gt=0, description="HTTP request timeout in seconds")
    # Two separate retry budgets: the openai client retries network errors, 429 and 5xx with back-off;
    # chat_json re-prompts the model after invalid JSON (spec: at most 2).
    LLM_HTTP_RETRIES: int = Field(2, ge=0, description="openai-client retries for network/429/5xx errors")
    LLM_JSON_RETRIES: int = Field(2, ge=0, description="chat_json re-prompts after invalid output")

    # Embeddings
    EMBED_BACKEND: Literal["openai", "sentence-transformers"] = "openai"
    EMBED_BASE_URL: str | None = Field(None, description="Embeddings endpoint; empty = LLM_BASE_URL")
    EMBED_API_KEY: str | None = Field(None, description="Key for EMBED_BASE_URL; empty = LLM_API_KEY")
    EMBED_MODEL: str = Field("nomic-embed-text", description="Model name on the embeddings endpoint")
    EMBED_LOCAL_MODEL_NAME: str = Field("all-MiniLM-L6-v2", description="HuggingFace sentence-transformers model")
    EMBED_BATCH_SIZE: int = Field(64, gt=0, description="Texts per embeddings request")

    # Agent limits (CLAUDE.md "Agent robustness")
    MAX_AGENT_STEPS: int = Field(10, gt=0)
    MAX_TOKENS_PER_QUERY: int = Field(12_000, gt=0)

    # SEC EDGAR fair-access policy: automated clients must identify themselves
    SEC_USER_AGENT: str = Field(..., min_length=1, description="User-Agent header required by SEC EDGAR")

    @field_validator("EMBED_BACKEND", mode="before")
    @classmethod
    def _backend_name(cls, value: object) -> object:
        """'Sentence-Transformers' or ' openai' typed in .env still select the backend."""
        return value.strip().lower() if isinstance(value, str) else value


@cache
def get_settings() -> Settings:
    """The settings, read from the environment and `.env` once per process."""
    return Settings()
