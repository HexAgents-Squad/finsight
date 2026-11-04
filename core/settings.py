"""FinSight settings, read from environment variables or `.env` (template: `.env.example`).

Switching from the local Ollama stand-in to the official endpoint must only mean editing `.env`.
"""
from __future__ import annotations

from functools import cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the LLM client, embeddings, agent limits and SEC access."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True)

    # Chat model on an OpenAI-compatible endpoint
    LLM_BASE_URL: str = Field(..., description="Target OpenAI-compatible endpoint URL")
    LLM_API_KEY: str = Field(..., description="API key for the endpoint ('ollama' locally)")
    LLM_MODEL: str = Field(..., description="Chat model name")
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
    SEC_USER_AGENT: str = Field(..., description="User-Agent header required by SEC EDGAR")


@cache
def get_settings() -> Settings:
    """The settings, read from the environment and `.env` once per process."""
    return Settings()
