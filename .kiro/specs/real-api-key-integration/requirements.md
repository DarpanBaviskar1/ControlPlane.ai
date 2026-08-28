# Requirements: Real API Key Integration

## Overview

The ControlPlane.ai Enterprise AI Proxy Gateway currently runs on dummy/hardcoded keys for
every external integration. This spec covers wiring all five external integrations to real
credentials, replacing the OpenAI-specific direct-call fallback with a generic key, adding
startup-time configuration reporting, and exposing a health endpoint that shows operators
which integrations are active versus degraded.

---

## Requirement 1 — Generic LLM Provider Key

### 1.1
The system MUST rename `OPENAI_API_KEY` to `LLM_API_KEY` in `app/config.py` and in every
place it is read at runtime (`app/router/model_router.py`).

### 1.2
The existing direct-call fallback in `model_router.route_and_call()` that imports `openai`
and hard-codes `gpt-4o-mini` MUST be generalised to read the model name from a new
`LLM_FALLBACK_MODEL` setting (default: `"gpt-4o-mini"`) rather than a string literal.

### 1.3
`LLM_API_KEY` MUST remain optional (default `""`). When the value is empty or absent, the
mock-response path already present in `model_router.py` MUST continue to function without
raising an exception.

### 1.4
The system MUST log an `INFO` message at startup stating whether `LLM_API_KEY` is configured
(`LLM_API_KEY_CONFIGURED=true`) or absent (`LLM_API_KEY_CONFIGURED=false`).

---

## Requirement 2 — Portkey Gateway Integration

### 2.1
The system MUST read a real `PORTKEY_API_KEY` from the environment. The default dummy value
(`"dummy-portkey-key"`) MUST be preserved as the fallback for local development and tests.

### 2.2
When `PORTKEY_API_KEY` starts with `"dummy"`, all Portkey dispatch paths (non-streaming in
`model_router.py` and streaming in `streaming_router.py`) MUST fall through to mock/simulated
responses — no outbound HTTP calls to `api.portkey.ai`.

### 2.3
When `PORTKEY_API_KEY` is a real key, the router MUST use it together with
`PORTKEY_FRONTIER_VIRTUAL_KEY` (for `selected_tier="FRONTIER"`) and
`PORTKEY_SLM_VIRTUAL_KEY` (for `selected_tier="SLM"`) when dispatching to
`https://api.portkey.ai/v1/chat/completions`.

### 2.4
A new setting `LLM_PROVIDER` (default: `"openai"`, allowed values: `"openai"`,
`"anthropic"`, `"google"`, `"grok"`, `"generic"`) MUST be added to `app/config.py`. The
value MUST be forwarded as the `x-portkey-provider` header on every outbound Portkey call,
enabling Portkey to route to the correct upstream LLM without changes to the gateway's
dispatch logic.

### 2.5
The system MUST log an `INFO` message at startup: either
`PORTKEY_ACTIVE provider=<LLM_PROVIDER>` when `PORTKEY_API_KEY` is a real key, or
`PORTKEY_DEGRADED — mock responses active` when the key is a dummy value.

---

## Requirement 3 — Langfuse Observability Keys

### 3.1
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` MUST remain optional (default `""`).
The existing stdout-fallback behaviour MUST be preserved when either key is absent.

### 3.2
The Langfuse client initialisation in `LangfuseTracer.start()` MUST validate that the
supplied keys are non-empty and non-whitespace before attempting to connect. An empty or
whitespace-only key MUST be treated as absent and MUST NOT trigger a connection attempt.

### 3.3
When a Langfuse connection attempt fails (network error, invalid key), the tracer MUST emit
a single `WARNING` log with the failure reason (without retrying during startup) and MUST
fall back to stdout logging for the lifetime of that process. The retry loop (Req. 6.10 of
the existing spec) handles subsequent reconnects and is unaffected.

### 3.4
The system MUST log an `INFO` message at startup: either
`LANGFUSE_ACTIVE host=<LANGFUSE_HOST>` when credentials are present and the initial
connection succeeds, or `LANGFUSE_DEGRADED — stdout fallback active` otherwise.

---

## Requirement 4 — Guardrails AI Output Validators

### 4.1
The existing graceful-degradation behaviour (pass-through when `guardrails-ai` is not
installed) MUST be preserved unchanged.

### 4.2
`load_validators()` in `app/judges/output_validator.py` MUST, for each validator ID listed
in `GUARDRAILS_VALIDATORS`, attempt to install it from the Guardrails Hub if it is not
already present. A failed install MUST log a `WARNING` and skip that validator — it MUST NOT
raise an exception or abort startup.

### 4.3
A new setting `GUARDRAILS_HUB_TOKEN` (default: `""`) MUST be added to `app/config.py` for
authenticated access to the Guardrails Hub. When the value is empty the hub call proceeds
without an auth token (public validators only).

### 4.4
After `load_validators()` completes, the system MUST log an `INFO` summary:
`GUARDRAILS_LOADED validators=<comma-separated-ids>` for each successfully loaded validator,
and `GUARDRAILS_SKIPPED validators=<comma-separated-ids>` for each that could not be loaded.

### 4.5
When `GUARDRAILS_VALIDATORS` is empty or all validators fail to load, the output validation
stage MUST pass every output through unchanged (existing behaviour) and MUST log
`GUARDRAILS_DEGRADED — no validators active`.

---

## Requirement 5 — Worldsense MCP Server

### 5.1
The Worldsense MCP server URL MUST be configurable via a `WORLDSENSE_MCP_URL` environment
variable. The current default construction logic
(`http://localhost:<WORLDSENSE_MCP_PORT>/evaluate`) MUST be moved into `app/config.py` as a
computed default so it is visible alongside other settings.

### 5.2
At application startup (inside the FastAPI lifespan), the system MUST perform a health probe
against `<WORLDSENSE_MCP_URL_BASE>/health` (i.e. the root of `WORLDSENSE_MCP_URL` with the
path replaced by `/health`) using an `httpx.AsyncClient` with a 2-second timeout.

### 5.3
The health probe result MUST be stored at `app.state.worldsense_mcp_healthy` (boolean).

### 5.4
The system MUST log an `INFO` message: either
`WORLDSENSE_MCP_ACTIVE url=<WORLDSENSE_MCP_URL>` when the probe returns HTTP 200, or
`WORLDSENSE_MCP_UNAVAILABLE — heuristic fallback active` on any error or non-200 response.

### 5.5
The three-tier fallback chain in `worldsense_oversight.evaluate_oversight()` (MCP → SDK →
heuristic) MUST remain unchanged in logic. The health probe at startup is informational only
and does not alter the runtime fallback behaviour.

---

## Requirement 6 — Integration Health Endpoint

### 6.1
The system MUST expose a `GET /v1/config/health` endpoint that returns HTTP 200 with a JSON
body describing the status of every external integration.

### 6.2
The response schema MUST be:

```json
{
  "portkey":    { "status": "active" | "degraded", "detail": "<string>" },
  "langfuse":   { "status": "active" | "degraded", "detail": "<string>" },
  "guardrails": { "status": "active" | "degraded", "detail": "<string>" },
  "worldsense": { "status": "active" | "degraded", "detail": "<string>" },
  "llm_direct": { "status": "active" | "degraded", "detail": "<string>" }
}
```

### 6.3
Status values MUST reflect the runtime state, not configuration intent:
- `"active"` — the integration is reachable and operating normally.
- `"degraded"` — the integration is unavailable and a fallback is in effect.

### 6.4
The endpoint MUST NOT make any new outbound network calls at request time. It MUST read
cached state set during startup (e.g. `app.state.worldsense_mcp_healthy`,
`app.state.langfuse_tracer._enabled`) and the configuration values already in memory.

### 6.5
The endpoint MUST be accessible without authentication (same as `/v1/metrics`).

### 6.6
The `detail` field MUST contain a human-readable one-sentence explanation, e.g.
`"Portkey API key configured; provider=openai"` or
`"PORTKEY_API_KEY not set — mock responses active"`.

---

## Requirement 7 — Startup Configuration Summary Log

### 7.1
At the end of the lifespan startup sequence (after all integrations are probed), the system
MUST emit a single structured `INFO` log block summarising configuration state:

```
ControlPlane.ai configuration summary:
  LLM direct key : CONFIGURED / NOT CONFIGURED
  Portkey        : ACTIVE (provider=<X>) / DEGRADED (mock)
  Langfuse       : ACTIVE (host=<H>) / DEGRADED (stdout)
  Guardrails     : ACTIVE (<N> validators) / DEGRADED (none loaded)
  Worldsense MCP : ACTIVE (<URL>) / DEGRADED (heuristic)
```

### 7.2
Sensitive key values MUST NOT appear in any log output. Only the presence/absence of a key
(`CONFIGURED` / `NOT CONFIGURED`) or a non-secret attribute (provider name, host URL) MUST
be logged.

---

## Non-Functional Requirements

### NFR-1
All changes to `app/config.py` MUST be backward-compatible: any existing `.env` file that
sets `OPENAI_API_KEY` will no longer be read (by design); operators MUST update it to
`LLM_API_KEY`. The change MUST be documented in the spec.

### NFR-2
The `/v1/config/health` endpoint MUST respond within 50 ms (no outbound calls, reads only
in-memory state).

### NFR-3
No new mandatory dependencies MUST be added. All new provider-specific SDK imports MUST
follow the same optional-import pattern used elsewhere in the codebase
(`try/except ImportError` with `_AVAILABLE` boolean flag and graceful degradation).

### NFR-4
All existing tests MUST continue to pass. New tests MUST be added for the health endpoint
and for the renamed `LLM_API_KEY` config path.
