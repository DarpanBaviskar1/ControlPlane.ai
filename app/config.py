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

    # Telemetry sink: "stdout", "file", or a URL (kept for local fallback)
    TELEMETRY_SINK: str = "stdout"
    TELEMETRY_LOG_FILE: str = "telemetry.jsonl"

    # Embedding model used by the Groundedness Auditor
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_API_KEY: str = "dummy-openai-key"

    # Vector store top-K retrieval
    VECTOR_STORE_TOP_K: int = 5

    # Langfuse observability (Req. 6)
    # Set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY to enable cloud tracing.
    # If empty the gateway falls back to local stdout logging.
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    # Maximum buffered events when Langfuse is unreachable (Req. 6.10)
    LANGFUSE_BUFFER_SIZE: int = 1000
    # Retry interval in seconds when Langfuse backend is unreachable
    LANGFUSE_RETRY_INTERVAL_S: int = 30

    # Guardrails AI output validation (Req. 2.11-13)
    # Comma-separated list of validator IDs to load from Guardrails AI Hub.
    GUARDRAILS_VALIDATORS: str = "toxic-language,competitor-check"

    # Obot agent governance (Req. 11)
    OBOT_ENABLED: bool = True
    # Maximum tool calls per request before ESCALATE_TO_HUMAN (Req. 11.6)
    OBOT_MAX_TOOL_CALLS_DEFAULT: int = 10
    # Latency budget for a single Obot authorisation check in ms (Req. 11.7)
    OBOT_LATENCY_BUDGET_MS: int = 20

    # Worldsense multi-turn agentic oversight (Req. 12)
    WORLDSENSE_ENABLED: bool = True
    # Maximum evaluation time before treating verdict as RISK_DETECTED (Req. 12.6)
    WORLDSENSE_TIMEOUT_MS: int = 300

    # Red Team Runner (Req. 10)
    REDTEAM_ENABLED: bool = True
    REDTEAM_MIN_PROMPTS: int = 50
    # Scheduled interval for automated red-team runs (cron expression, empty = disabled)
    REDTEAM_SCHEDULE: str = ""


settings = Settings()
