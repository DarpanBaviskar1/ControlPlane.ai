"""Application settings loaded from environment variables.

All configuration is driven by environment variables or a .env file in the
project root.  Copy .env.example to .env and fill in your real values:

    cp .env.example .env

See .env.example for documentation on every variable and how to obtain keys.
Run  GET /v1/config/health  at any time to see which integrations are active.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _is_real_key(value: str) -> bool:
    """Return True when *value* is a non-empty, non-whitespace, non-dummy API key.

    Used by every integration to decide between live and mock/degraded paths.
    Examples:
        _is_real_key("")                -> False
        _is_real_key("   ")             -> False
        _is_real_key("dummy-portkey")   -> False
        _is_real_key("sk-real-abc123")  -> True
    """
    return bool(value) and value.strip() != "" and not value.startswith("dummy")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # silently ignore unknown env vars (e.g. old OPENAI_API_KEY)
    )

    # -------------------------------------------------------------------------
    # 1. LLM Provider — dispatched via LiteLLM (BSD-3, no gateway service)
    # -------------------------------------------------------------------------
    # Model strings may be bare ("gemini-2.5-flash") or fully qualified
    # ("gemini/gemini-2.5-flash").  Bare names are prefixed with the LiteLLM
    # provider derived from LLM_PROVIDER at call time.
    # If LLM_API_KEY is absent/dummy, the gateway serves safe contextual mock
    # responses — useful for local dev, CI and the test suite.
    LLM_PROVIDER: Literal["openai", "anthropic", "google", "grok", "generic"] = "openai"
    LLM_API_KEY: str = ""

    # Two-tier routing.  SLM handles ROUTINE prompts cheaply; FRONTIER handles
    # COMPLEX ones.  Each is the other's fallback on dispatch failure.
    # These string defaults ("gpt-4o-mini" / "gpt-4o") are deliberate
    # back-compat sentinels, not vendor lock-in: blanking them yields a
    # malformed model string ("openai/") once prefixed in providers.py.
    SLM_MODEL: str = "gpt-4o-mini"
    FRONTIER_MODEL: str = "gpt-4o"

    # Retained as the alias older configs use for the SLM tier.  When
    # SLM_MODEL is left at its default and this is set, this wins.
    LLM_FALLBACK_MODEL: str = "gpt-4o-mini"

    # Per-call egress budget.  LiteLLM performs the retries internally.
    LLM_TIMEOUT_S: float = 30.0
    LLM_MAX_RETRIES: int = 2

    # Optional explicit base URL — set this for self-hosted or
    # OpenAI-compatible endpoints (Ollama, vLLM, LM Studio, LiteLLM proxy).
    # A backend reached only through this URL may have no API key at all
    # (e.g. a local Ollama server) — is_live() treats this as live too.
    LLM_API_BASE: str = ""

    # -------------------------------------------------------------------------
    # 2. Portkey Gateway (recommended for production)
    # -------------------------------------------------------------------------
    PORTKEY_API_KEY: str = "dummy-portkey-key"
    PORTKEY_SLM_VIRTUAL_KEY: str = "slm-virtual-key"
    PORTKEY_FRONTIER_VIRTUAL_KEY: str = "frontier-virtual-key"

    # -------------------------------------------------------------------------
    # 3. Langfuse Observability
    # -------------------------------------------------------------------------
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_BUFFER_SIZE: int = 1000
    LANGFUSE_RETRY_INTERVAL_S: int = 30

    # -------------------------------------------------------------------------
    # 4. Guardrails AI Output Validation
    # -------------------------------------------------------------------------
    GUARDRAILS_VALIDATORS: str = "toxic-language,competitor-check"
    GUARDRAILS_HUB_TOKEN: str = ""

    # -------------------------------------------------------------------------
    # 5. Worldsense MCP Server (multi-turn agentic oversight)
    # -------------------------------------------------------------------------
    WORLDSENSE_ENABLED: bool = True
    WORLDSENSE_MCP_URL: str = "http://localhost:9100/evaluate"
    WORLDSENSE_TIMEOUT_MS: int = 300

    # -------------------------------------------------------------------------
    # 6. Redteam MCP Server (automated adversarial testing)
    # -------------------------------------------------------------------------
    REDTEAM_ENABLED: bool = True
    REDTEAM_MIN_PROMPTS: int = 50
    REDTEAM_SCHEDULE: str = ""

    # -------------------------------------------------------------------------
    # 7. Semantic Cache (Phase 3 — requires gptcache)
    # -------------------------------------------------------------------------
    CACHE_SIMILARITY_THRESHOLD: float = 0.92

    # -------------------------------------------------------------------------
    # 8. Policy & Pipeline
    # -------------------------------------------------------------------------
    POLICY_FILE_PATH: str | None = None
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    VECTOR_STORE_TOP_K: int = 5

    # -------------------------------------------------------------------------
    # 9. Telemetry
    # -------------------------------------------------------------------------
    TELEMETRY_SINK: str = "stdout"
    TELEMETRY_LOG_FILE: str = "telemetry.jsonl"

    # -------------------------------------------------------------------------
    # 10. Obot Agent Governance
    # -------------------------------------------------------------------------
    OBOT_ENABLED: bool = True
    OBOT_MAX_TOOL_CALLS_DEFAULT: int = 10
    OBOT_LATENCY_BUDGET_MS: int = 20


settings = Settings()
