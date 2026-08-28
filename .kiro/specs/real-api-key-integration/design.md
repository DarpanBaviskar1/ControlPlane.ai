# Design: Real API Key Integration

## Overview

This design covers five targeted changes across the gateway codebase:

1. Rename `OPENAI_API_KEY` → `LLM_API_KEY` and add `LLM_PROVIDER` + `LLM_FALLBACK_MODEL`
2. Wire Portkey dispatch with the `x-portkey-provider` header
3. Tighten Langfuse key validation at startup
4. Add Guardrails Hub pre-flight and `GUARDRAILS_HUB_TOKEN`
5. Move Worldsense MCP URL into `config.py` and add a startup health probe
6. Add a `GET /v1/config/health` endpoint and a startup summary log

All changes are additive or rename-only. The five-stage pipeline logic is untouched. Every
new integration point degrades gracefully when credentials are absent, matching the existing
pattern used by Langfuse, GLiNER, and the NLI scorer.

---

## 1. Config Layer (`app/config.py`)

### New and renamed settings

```python
# Renamed from OPENAI_API_KEY
LLM_API_KEY: str = ""                  # empty = direct-call fallback disabled

# New: selects the upstream provider forwarded to Portkey
LLM_PROVIDER: Literal[
    "openai", "anthropic", "google", "grok", "generic"
] = "openai"

# New: model name used by the in-process direct-call fallback
LLM_FALLBACK_MODEL: str = "gpt-4o-mini"

# New: Guardrails Hub auth token (optional, public validators work without it)
GUARDRAILS_HUB_TOKEN: str = ""

# Worldsense MCP URL moved here from module-level constant in oversight.py
# The port-based default mirrors the existing WORLDSENSE_MCP_PORT env var logic.
WORLDSENSE_MCP_URL: str = "http://localhost:9100/evaluate"
```

`OPENAI_API_KEY` is removed. `PORTKEY_API_KEY`, `PORTKEY_FRONTIER_VIRTUAL_KEY`,
`PORTKEY_SLM_VIRTUAL_KEY`, `LANGFUSE_*`, and `GUARDRAILS_VALIDATORS` are unchanged.

### Validation helper

A module-level utility `_is_real_key(value: str) -> bool` returns `True` when the value is
non-empty, non-whitespace, and does not start with `"dummy"`. Used by every integration to
decide between real and mock/degraded paths without duplicating the check.

---

## 2. Model Router (`app/router/model_router.py`)

### Direct-call fallback (no RouteLLM / no Portkey)

```
if _controller is None:
    portkey_is_real = _is_real_key(settings.PORTKEY_API_KEY)
    if portkey_is_real:
        → dispatch via Portkey (see §3 below)
    elif _is_real_key(settings.LLM_API_KEY):
        → existing openai.AsyncOpenAI path, but:
             model=settings.LLM_FALLBACK_MODEL  (not hardcoded "gpt-4o-mini")
             api_key=settings.LLM_API_KEY       (not os.environ.get("OPENAI_API_KEY"))
    else:
        → return _generate_contextual_response(prompt)  (mock, as today)
```

This replaces the `os.environ.get("OPENAI_API_KEY")` import-time lookup with a
settings-driven read, and generalises the model name.

### Portkey dispatch (RouteLLM path — `selected_tier` known)

When `_controller` is not `None` (RouteLLM available) and `PORTKEY_API_KEY` is real, the
Portkey call gains the `x-portkey-provider` header:

```python
headers = {
    "x-portkey-api-key": settings.PORTKEY_API_KEY,
    "x-portkey-virtual-key": (
        settings.PORTKEY_FRONTIER_VIRTUAL_KEY
        if selected_tier == "FRONTIER"
        else settings.PORTKEY_SLM_VIRTUAL_KEY
    ),
    "x-portkey-provider": settings.LLM_PROVIDER,
    "Content-Type": "application/json",
}
```

No other changes to the routing logic.

---

## 3. Streaming Router (`app/ingress/streaming_router.py`)

The `_stream_tokens_from_llm()` function already reads `settings.PORTKEY_API_KEY` and checks
for the `"dummy"` prefix. The only change needed is to add `"x-portkey-provider"` to the
headers dict when the key is real:

```python
headers = {
    "x-portkey-api-key": portkey_key,
    "x-portkey-virtual-key": settings.PORTKEY_SLM_VIRTUAL_KEY,
    "x-portkey-provider": settings.LLM_PROVIDER,   # NEW
    "Content-Type": "application/json",
}
```

---

## 4. Langfuse Tracer (`app/observability/langfuse_tracer.py`)

### Key validation in `start()`

Before creating the `Langfuse` client, `start()` applies `_is_real_key()` to both
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`:

```python
async def start(self) -> None:
    if not _LANGFUSE_AVAILABLE:
        logger.info("Langfuse SDK unavailable — tracing disabled")
        return
    if not (_is_real_key(settings.LANGFUSE_PUBLIC_KEY)
            and _is_real_key(settings.LANGFUSE_SECRET_KEY)):
        logger.info("LANGFUSE_DEGRADED — stdout fallback active")
        return
    try:
        self._client = Langfuse(...)
        self._enabled = True
        logger.info("LANGFUSE_ACTIVE host=%s", settings.LANGFUSE_HOST)
        self._retry_task = asyncio.create_task(self._retry_loop(), ...)
    except Exception as exc:
        logger.warning("Langfuse init failed: %s — LANGFUSE_DEGRADED stdout fallback", exc)
```

The `_is_real_key` import comes from `app.config`. No other changes to the tracer.

---

## 5. Guardrails Output Validator (`app/judges/output_validator.py`)

### `load_validators()` with hub pre-flight

```python
def load_validators() -> None:
    if not _GUARDRAILS_AVAILABLE:
        logger.warning("GUARDRAILS_DEGRADED — no validators active")
        return

    loaded, skipped = [], []
    for vid in validator_ids:
        try:
            validator = gd.hub.load(vid)
            _LOADED_VALIDATORS.append((vid, validator))
            loaded.append(vid)
        except Exception:
            # Try hub install if load failed
            try:
                _hub_install(vid)
                validator = gd.hub.load(vid)
                _LOADED_VALIDATORS.append((vid, validator))
                loaded.append(vid)
            except Exception as exc:
                logger.warning("GUARDRAILS_SKIPPED validator=%s reason=%s", vid, exc)
                skipped.append(vid)

    if loaded:
        logger.info("GUARDRAILS_LOADED validators=%s", ",".join(loaded))
    if skipped:
        logger.warning("GUARDRAILS_SKIPPED validators=%s", ",".join(skipped))
    if not loaded:
        logger.warning("GUARDRAILS_DEGRADED — no validators active")
```

### `_hub_install(validator_id)` helper

```python
def _hub_install(validator_id: str) -> None:
    """Attempt `guardrails hub install <id>` in a subprocess."""
    import subprocess, sys
    env = os.environ.copy()
    if settings.GUARDRAILS_HUB_TOKEN:
        env["GUARDRAILS_TOKEN"] = settings.GUARDRAILS_HUB_TOKEN
    subprocess.check_call(
        [sys.executable, "-m", "guardrails", "hub", "install", validator_id],
        env=env,
        timeout=60,
    )
```

The subprocess call is synchronous and runs during `asyncio.to_thread(load_validators)` at
startup (step 5b in `main.py` lifespan), so it does not block the event loop.

---

## 6. Worldsense Oversight (`app/oversight/worldsense_oversight.py`)

### Move URL constant to config

Remove the module-level `_MCP_URL` computation from `worldsense_oversight.py`:

```python
# Before (remove):
_MCP_URL: str = os.getenv(
    "WORLDSENSE_MCP_URL",
    f"http://localhost:{os.getenv('WORLDSENSE_MCP_PORT', '9100')}/evaluate",
)

# After (read from settings):
from app.config import settings
_MCP_URL: str = settings.WORLDSENSE_MCP_URL
```

No other logic changes.

### Startup health probe in `main.py` lifespan

A new Step 11 is added after the existing Step 10 (redteam probe):

```python
# 11. Worldsense MCP health probe (Req. 5.2)
try:
    from urllib.parse import urlparse, urlunparse
    _ws_parsed = urlparse(settings.WORLDSENSE_MCP_URL)
    _ws_health = urlunparse(_ws_parsed._replace(path="/health", query="", fragment=""))
    async with _httpx.AsyncClient(timeout=2.0) as _probe:
        _resp = await _probe.get(_ws_health)
        app.state.worldsense_mcp_healthy = _resp.status_code == 200
        if app.state.worldsense_mcp_healthy:
            logger.info("WORLDSENSE_MCP_ACTIVE url=%s", settings.WORLDSENSE_MCP_URL)
        else:
            logger.warning("WORLDSENSE_MCP_UNAVAILABLE — heuristic fallback active")
except Exception as _probe_exc:
    app.state.worldsense_mcp_healthy = False
    logger.warning("WORLDSENSE_MCP_UNAVAILABLE — %s", _probe_exc)
```

---

## 7. Integration Health Endpoint (`app/config_health/router.py`)

A new router module is created. It is mounted in `main.py` alongside the existing routers.

### Response model (`app/models.py` addition)

```python
class IntegrationStatus(BaseModel):
    status: Literal["active", "degraded"]
    detail: str

class ConfigHealthResponse(BaseModel):
    portkey:    IntegrationStatus
    langfuse:   IntegrationStatus
    guardrails: IntegrationStatus
    worldsense: IntegrationStatus
    llm_direct: IntegrationStatus
```

### Endpoint logic

```python
@router.get("/v1/config/health", response_model=ConfigHealthResponse)
async def config_health(request: Request) -> ConfigHealthResponse:
    from app.config import settings
    from app.judges.output_validator import _LOADED_VALIDATORS
    from app.oversight.worldsense_oversight import _MCP_HEALTHY

    # Portkey
    portkey_real = _is_real_key(settings.PORTKEY_API_KEY)
    portkey = IntegrationStatus(
        status="active" if portkey_real else "degraded",
        detail=(
            f"Portkey API key configured; provider={settings.LLM_PROVIDER}"
            if portkey_real
            else "PORTKEY_API_KEY not set — mock responses active"
        ),
    )

    # Langfuse
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

    # Guardrails
    n_validators = len(_LOADED_VALIDATORS)
    guardrails = IntegrationStatus(
        status="active" if n_validators > 0 else "degraded",
        detail=(
            f"{n_validators} validator(s) loaded: "
            + ", ".join(v[0] for v in _LOADED_VALIDATORS)
            if n_validators > 0
            else "No validators loaded — output validation pass-through active"
        ),
    )

    # Worldsense
    ws_healthy = getattr(request.app.state, "worldsense_mcp_healthy", False)
    worldsense = IntegrationStatus(
        status="active" if ws_healthy else "degraded",
        detail=(
            f"Worldsense MCP reachable at {settings.WORLDSENSE_MCP_URL}"
            if ws_healthy
            else "Worldsense MCP unreachable — heuristic fallback active"
        ),
    )

    # LLM direct key
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
```

All data is read from in-memory state — no outbound calls.

---

## 8. Startup Summary Log (`app/main.py`)

At the end of the lifespan startup sequence (after Step 11), a single formatted `INFO` log
is emitted:

```python
logger.info(
    "ControlPlane.ai configuration summary:\n"
    "  LLM direct key : %s\n"
    "  Portkey        : %s\n"
    "  Langfuse       : %s\n"
    "  Guardrails     : %s\n"
    "  Worldsense MCP : %s",
    "CONFIGURED" if _is_real_key(settings.LLM_API_KEY) else "NOT CONFIGURED",
    f"ACTIVE (provider={settings.LLM_PROVIDER})" if _is_real_key(settings.PORTKEY_API_KEY) else "DEGRADED (mock)",
    f"ACTIVE (host={settings.LANGFUSE_HOST})" if langfuse_tracer._enabled else "DEGRADED (stdout)",
    f"ACTIVE ({len(_LOADED_VALIDATORS)} validators)" if _LOADED_VALIDATORS else "DEGRADED (none loaded)",
    f"ACTIVE ({settings.WORLDSENSE_MCP_URL})" if app.state.worldsense_mcp_healthy else "DEGRADED (heuristic)",
)
```

No secret values appear in this output (only presence flag and non-secret attributes).

---

## 9. `_is_real_key` Placement

`_is_real_key` is defined in `app/config.py` as a module-level function (not a method on
`Settings`) so it can be imported anywhere without creating a circular dependency:

```python
def _is_real_key(value: str) -> bool:
    """Return True when value is a non-empty, non-dummy API key."""
    return bool(value) and value.strip() != "" and not value.startswith("dummy")
```

---

## 10. File Change Summary

| File | Change |
|---|---|
| `app/config.py` | Rename `OPENAI_API_KEY→LLM_API_KEY`; add `LLM_PROVIDER`, `LLM_FALLBACK_MODEL`, `GUARDRAILS_HUB_TOKEN`, `WORLDSENSE_MCP_URL`; add `_is_real_key()` |
| `app/router/model_router.py` | Use `settings.LLM_API_KEY`, `settings.LLM_FALLBACK_MODEL`; add `x-portkey-provider` header; import `_is_real_key` |
| `app/ingress/streaming_router.py` | Add `x-portkey-provider` header to Portkey streaming call |
| `app/observability/langfuse_tracer.py` | Use `_is_real_key` for key validation; standardise startup log messages |
| `app/judges/output_validator.py` | Add `_hub_install()` helper; update `load_validators()` with install-on-miss logic and summary logs |
| `app/oversight/worldsense_oversight.py` | Replace module-level `_MCP_URL` constant with `settings.WORLDSENSE_MCP_URL` |
| `app/main.py` | Add Step 11 (Worldsense health probe); add startup summary log; mount config_health router; import `_LOADED_VALIDATORS` for summary |
| `app/models.py` | Add `IntegrationStatus` and `ConfigHealthResponse` Pydantic models |
| `app/config_health/router.py` | New file — `GET /v1/config/health` endpoint |
| `app/config_health/__init__.py` | New file — empty package init |
| `tests/unit/test_config_health.py` | New test file — health endpoint and `_is_real_key` tests |

---

## 11. Backward Compatibility Notes

- `OPENAI_API_KEY` is removed from `Settings`. Operators with an existing `.env` file that
  sets `OPENAI_API_KEY` MUST rename it to `LLM_API_KEY` for the direct-call fallback to work.
  Portkey virtual keys and all other settings are unaffected.
- All existing tests that mock `OPENAI_API_KEY` (currently none in the test suite — the key
  is read via `os.environ.get()` at call time) will continue to pass. Tests that set
  `settings.OPENAI_API_KEY` directly MUST be updated to `settings.LLM_API_KEY`.
- `pydantic-settings` will silently ignore unknown environment variables (`extra="ignore"`)
  so environments that still export `OPENAI_API_KEY` will not cause startup failures.

---

## Correctness Properties

### P-CFG-1 — `_is_real_key` never treats a dummy or blank key as real
For any string that is empty, whitespace-only, or starts with `"dummy"`,
`_is_real_key(value)` MUST return `False`.

### P-CFG-2 — Health endpoint returns no outbound calls
For any application state, `GET /v1/config/health` MUST complete without making any network
call (reads only `app.state`, `settings`, and module-level globals).

### P-CFG-3 — Portkey mock path unchanged when key is dummy
For any prompt, when `PORTKEY_API_KEY.startswith("dummy")` is `True`, the routing result
MUST have `response == _generate_contextual_response(prompt)` (i.e. the mock path).

### P-CFG-4 — `LLM_PROVIDER` header forwarded on every real Portkey call
For any real `PORTKEY_API_KEY` value, every outbound call to `api.portkey.ai` MUST include
an `x-portkey-provider` header whose value equals `settings.LLM_PROVIDER`.

### P-CFG-5 — Langfuse startup: empty key never triggers connection
For any `LANGFUSE_PUBLIC_KEY` or `LANGFUSE_SECRET_KEY` that is empty or whitespace,
`LangfuseTracer._enabled` MUST be `False` after `start()` completes.
