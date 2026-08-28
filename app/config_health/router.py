"""GET /v1/config/health — integration status endpoint.

Returns the live status of every external integration by reading cached
in-memory state only. No outbound network calls are made at request time.

Requirements: 6.1–6.6
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import settings, _is_real_key
from app.models import ConfigHealthResponse, IntegrationStatus

router = APIRouter()


@router.get("/v1/config/health", response_model=ConfigHealthResponse)
async def config_health(request: Request) -> ConfigHealthResponse:
    """Return live status of all external integrations (read-only, no I/O)."""

    # --- Portkey ---
    portkey_real = _is_real_key(settings.PORTKEY_API_KEY)
    portkey = IntegrationStatus(
        status="active" if portkey_real else "degraded",
        detail=(
            f"Portkey API key configured; provider={settings.LLM_PROVIDER}"
            if portkey_real
            else "PORTKEY_API_KEY not set — mock responses active"
        ),
    )

    # --- Langfuse ---
    tracer = getattr(request.app.state, "langfuse_tracer", None)
    langfuse_active = tracer is not None and getattr(tracer, "_enabled", False)
    langfuse = IntegrationStatus(
        status="active" if langfuse_active else "degraded",
        detail=(
            f"Langfuse active; host={settings.LANGFUSE_HOST}"
            if langfuse_active
            else "LANGFUSE_PUBLIC_KEY/SECRET_KEY not set — stdout fallback active"
        ),
    )

    # --- Guardrails ---
    from app.judges.output_validator import _LOADED_VALIDATORS
    n_validators = len(_LOADED_VALIDATORS)
    guardrails = IntegrationStatus(
        status="active" if n_validators > 0 else "degraded",
        detail=(
            f"{n_validators} validator(s) loaded: " + ", ".join(v[0] for v in _LOADED_VALIDATORS)
            if n_validators > 0
            else "No validators loaded — output validation pass-through active"
        ),
    )

    # --- Worldsense ---
    ws_healthy = getattr(request.app.state, "worldsense_mcp_healthy", False)
    worldsense = IntegrationStatus(
        status="active" if ws_healthy else "degraded",
        detail=(
            f"Worldsense MCP reachable at {settings.WORLDSENSE_MCP_URL}"
            if ws_healthy
            else "Worldsense MCP unreachable — heuristic fallback active"
        ),
    )

    # --- LLM direct key ---
    llm_direct_ok = _is_real_key(settings.LLM_API_KEY)
    llm_direct = IntegrationStatus(
        status="active" if llm_direct_ok else "degraded",
        detail=(
            f"LLM_API_KEY configured; fallback model={settings.LLM_FALLBACK_MODEL}"
            if llm_direct_ok
            else "LLM_API_KEY not set — direct-call fallback disabled"
        ),
    )

    return ConfigHealthResponse(
        portkey=portkey,
        langfuse=langfuse,
        guardrails=guardrails,
        worldsense=worldsense,
        llm_direct=llm_direct,
    )
