# Tasks: Real API Key Integration

## Task List

- [ ] 1. Add `_is_real_key` utility and new settings to `app/config.py`
  - Rename `OPENAI_API_KEY` to `LLM_API_KEY` (default `""`)
  - Add `LLM_PROVIDER: Literal["openai","anthropic","google","grok","generic"] = "openai"`
  - Add `LLM_FALLBACK_MODEL: str = "gpt-4o-mini"`
  - Add `GUARDRAILS_HUB_TOKEN: str = ""`
  - Add `WORLDSENSE_MCP_URL: str = "http://localhost:9100/evaluate"`
  - Add module-level `_is_real_key(value: str) -> bool` function
  - **Acceptance:** `settings.LLM_API_KEY` exists; `settings.OPENAI_API_KEY` no longer exists; `_is_real_key("")` is `False`; `_is_real_key("dummy-key")` is `False`; `_is_real_key("sk-real")` is `True`

- [ ] 2. Update `app/router/model_router.py` for generic key + Portkey provider header
  - Replace `os.environ.get("OPENAI_API_KEY")` with `settings.LLM_API_KEY`
  - Replace hardcoded `"gpt-4o-mini"` with `settings.LLM_FALLBACK_MODEL`
  - Add `"x-portkey-provider": settings.LLM_PROVIDER` header to every real Portkey call
  - Use `_is_real_key` for the key-presence check
  - **Acceptance:** Mock path still returns contextual response when both keys are absent; `x-portkey-provider` header present in outbound Portkey call when `PORTKEY_API_KEY` is real

- [ ] 3. Add `x-portkey-provider` header to streaming router
  - In `_stream_tokens_from_llm()` in `app/ingress/streaming_router.py`, add `"x-portkey-provider": settings.LLM_PROVIDER` to the headers dict
  - **Acceptance:** Header present in streaming call headers when key is real; mock simulated-stream path unaffected

- [ ] 4. Tighten Langfuse key validation in `app/observability/langfuse_tracer.py`
  - Import `_is_real_key` from `app.config`
  - Replace the `not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY` check with `_is_real_key` checks on both keys
  - Standardise startup log messages to `LANGFUSE_ACTIVE` / `LANGFUSE_DEGRADED — stdout fallback active`
  - **Acceptance:** Whitespace-only key does not trigger a connection attempt; log emits correct message for both paths

- [ ] 5. Add Guardrails Hub pre-flight and `GUARDRAILS_HUB_TOKEN` support
  - Add `_hub_install(validator_id: str) -> None` helper in `app/judges/output_validator.py`
  - Update `load_validators()` to try `_hub_install` on load failure, then retry load
  - Log `GUARDRAILS_LOADED`, `GUARDRAILS_SKIPPED`, `GUARDRAILS_DEGRADED` messages
  - Use `settings.GUARDRAILS_HUB_TOKEN` in `_hub_install` env when non-empty
  - **Acceptance:** Failed validator logs `GUARDRAILS_SKIPPED` and does not raise; successful load logs `GUARDRAILS_LOADED`; empty validator list logs `GUARDRAILS_DEGRADED`

- [ ] 6. Move Worldsense MCP URL to `app/config.py` and add startup health probe
  - Remove module-level `_MCP_URL` construction from `app/oversight/worldsense_oversight.py`; replace with `settings.WORLDSENSE_MCP_URL`
  - Add Step 11 to the lifespan in `app/main.py`: HTTP GET to `<base>/health` with 2 s timeout; store result at `app.state.worldsense_mcp_healthy`
  - Log `WORLDSENSE_MCP_ACTIVE` or `WORLDSENSE_MCP_UNAVAILABLE` accordingly
  - **Acceptance:** `app.state.worldsense_mcp_healthy` is a boolean after startup; module reads URL from settings; three-tier fallback chain in `evaluate_oversight` is unchanged

- [ ] 7. Add `IntegrationStatus` and `ConfigHealthResponse` models to `app/models.py`
  - Add `IntegrationStatus(status: Literal["active","degraded"], detail: str)` Pydantic model
  - Add `ConfigHealthResponse` with five `IntegrationStatus` fields: `portkey`, `langfuse`, `guardrails`, `worldsense`, `llm_direct`
  - **Acceptance:** Models import cleanly; Pydantic validates `status` as a literal; existing model tests unaffected

- [ ] 8. Implement `GET /v1/config/health` endpoint
  - Create `app/config_health/__init__.py` (empty)
  - Create `app/config_health/router.py` with the health endpoint reading in-memory state only
  - Mount the router in `app/main.py` (`app.include_router(config_health_router)`)
  - **Acceptance:** `GET /v1/config/health` returns HTTP 200 with valid JSON matching `ConfigHealthResponse` schema; no outbound network calls; responds within 50 ms

- [ ] 9. Add startup configuration summary log to `app/main.py`
  - After Step 11 (Worldsense probe) in the lifespan, emit the structured `INFO` summary log
  - Import `_LOADED_VALIDATORS` from `app.judges.output_validator` for the Guardrails count
  - No secret values in log output (presence flag and non-secret attributes only)
  - **Acceptance:** Summary log appears at startup with correct values for each integration; running with all dummy keys shows all `DEGRADED`/`NOT CONFIGURED`

- [ ] 10. Write tests for the new integration points
  - `tests/unit/test_config_health.py`:
    - Test `_is_real_key` for empty string, whitespace, dummy prefix, real value
    - Test `GET /v1/config/health` returns 200 with correct schema
    - Test each integration shows `"degraded"` when keys are absent (mock `app.state`)
    - Test `portkey` shows `"active"` when `PORTKEY_API_KEY` is non-dummy
  - `tests/unit/test_model_router_provider.py`:
    - Test `LLM_API_KEY` setting read (not `OPENAI_API_KEY`)
    - Test `LLM_FALLBACK_MODEL` used in the direct-call fallback
    - Test mock path returns contextual response when both `LLM_API_KEY` and `PORTKEY_API_KEY` are absent/dummy
  - **Acceptance:** All new tests pass; all existing 228 tests continue to pass
