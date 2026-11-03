from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Strict configuration loader with LLM, embedding, and SEC compliance fields."""
    
    # LLM Settings
    LLM_BASE_URL: str = Field(..., description="Target OpenAI-compatible endpoint URL")
    LLM_API_KEY: str = Field(..., description="API Key for target provider")
    LLM_MODEL: str = Field(..., description="Default text generation model")
    LLM_TIMEOUT: float = Field(30.0, description="HTTP request timeout in seconds")
    LLM_MAX_RETRIES: int = Field(2, description="Max retry attempts for LLM operations")

    # Embedding Settings
    EMBED_BACKEND: str = Field("openai", description="Backend type: 'openai' or 'sentence-transformers'")
    EMBED_MODEL: str = Field("nomic-embed-text", description="Model name for OpenAI-compatible embedding endpoint")
    EMBED_LOCAL_MODEL_NAME: str = Field("all-MiniLM-L6-v2", description="HuggingFace model name for local sentence-transformers")

    # SEC EDGAR Compliance
    SEC_USER_AGENT: str = Field(..., description="User-Agent header required by SEC EDGAR API")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True
    )

def get_settings() -> Settings:
    """Dependency factory for settings."""
    return Settings()