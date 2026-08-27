"""Application settings loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Portkey
    PORTKEY_API_KEY: str = "dummy-portkey-key"
    PORTKEY_FRONTIER_VIRTUAL_KEY: str = "frontier-virtual-key"
    PORTKEY_SLM_VIRTUAL_KEY: str = "slm-virtual-key"

    # Policy file path (optional — built-in profiles always loaded)
    POLICY_FILE_PATH: str | None = None

    # Telemetry sink: "stdout", "file", or a URL
    TELEMETRY_SINK: str = "stdout"
    TELEMETRY_LOG_FILE: str = "telemetry.jsonl"

    # Embedding model used by the Groundedness Auditor
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_API_KEY: str = "dummy-openai-key"

    # Vector store top-K retrieval
    VECTOR_STORE_TOP_K: int = 5


settings = Settings()
