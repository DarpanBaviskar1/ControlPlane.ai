# ControlPlane.ai — Current State and Architecture

## Overview

**ControlPlane.ai** is an Enterprise AI Proxy Gateway built on FastAPI. It mediates every
interaction between enterprise applications or agents and underlying Large Language Models
through a deterministic five-stage safety pipeline that enforces safety, cost governance,
groundedness assurance, and output triage before a response is delivered to callers.

This document reflects the current implementation state after four rounds of development:
the original five-stage pipeline, five targeted hardening improvements, the **Phase 3
upgrades** (Semantic Cache, NLI Groundedness, GLiNER Masking, SSE Streaming, Redteam MCP),
and the **Real API Key Integration** round that generalised all external integrations, added
a provider-agnostic LLM dispatch layer, and introduced the `GET /v1/config/health` endpoint.

---

## Repository Layout (top-level)

```
app/
  config.py                   — Pydantic-Settings config; _is_real_key() utility
  main.py                     — FastAPI app, lifespan (Steps 1–11), run_pipeline()
  models.py                   — Shared Pydantic/dataclass data models
  config_health/
    __init__.py
    router.py                 — GET /v1/config/health (integration status, no I/O)
  groundedness/
    auditor.py                — Two-stage embedding + NLI groundedness auditor
    nli_scorer.py             — NLI cross-encoder wrapper (Phase 3)
    vector_store.py           — FAISS vector store + pgvector stub
  ingress/
    router.py                 — POST /v1/chat handler
    sliding_window.py         — Sentence-chunk token buffer for SSE (Phase 3)
    streaming_router.py       — POST /v1/chat/stream SSE endpoint (Phase 3)
  judges/
    gliner_masker.py          — GLiNER custom entity masker Tier 1.5 (Phase 3)
    orchestrator.py           — Micro-judge stage coordinator
    output_validator.py       — Guardrails AI output validation chain
    p1_judge.py               — Toxicity + prompt-injection judge
    p2_judge.py               — PII detection + masking judge
    p3_judge.py               — Query-clarity judge
    pii_masking.py            — Three-tier PII masking engine (NLP + GLiNER + regex)
  observability/
    langfuse_tracer.py        — Langfuse span tracing with stdout fallback
  oversight/
    worldsense_oversight.py   — Multi-turn agentic consequence-chain oversight
  policy/
    loader.py                 — Hot-reloading policy file loader
  redteam/
    router.py                 — POST /v1/redteam/run + GET /v1/redteam/report
    runner.py                 — MCP-first red team orchestrator (Phase 3)
  router/
    model_router.py           — Multi-provider LLM dispatch (Portkey + direct + mock)
    semantic_cache.py         — GPTCache-backed vector similarity cache (Phase 3)
  telemetry/
    logger.py                 — Async telemetry queue + RetentionManager
    router.py                 — GET /v1/metrics + /v1/metrics/accuracy
  triage/
    compressor.py             — Token-budget summarisation for COMPRESS_AND_EDIT
    gateway.py                — Four-state priority triage matrix (Priority 0–4)
  feedback/
    router.py                 — GET /v1/feedback/export + POST /v1/feedback/override

mcp_servers/
  worldsense/
    server.py                 — Isolated Worldsense oversight MCP server
    requirements.txt
    mcp.json
  redteam/
    server.py                 — Isolated Redteam MCP server (Phase 3)
    requirements.txt
    mcp.json

tests/
  unit/                       — 245 unit + property tests (4 skipped)

.env.example                  — Canonical config reference (committed, documented)
.env                          — Local dev config (git-ignored, never committed)

.kiro/
  hooks/
    redteam-on-policy-save.json
    redteam_trigger.py        — PostFileSave hook: MCP health check + redteam trigger
  specs/
    gateway-phase3-upgrades/  — Phase 3 spec (requirements, design, tasks)
    real-api-key-integration/ — API key integration spec (requirements, design, tasks)
  steering/
    performance-budget.md     — Per-component latency ceilings (auto-included)
```

---

## Environment Configuration

**Files:** `.env.example` (committed), `.env` (git-ignored)

All configuration is driven by environment variables. The `.env` file is the single place
operators put credentials. Copy `.env.example` to `.env` to get started:

```bash
cp .env.example .env
```

`.env.example` documents every variable with explanations, where to obtain keys, and what
happens when each key is absent (graceful degradation path). It is kept in sync with
`app/config.py` and is the canonical operator reference.

Run `GET /v1/config/health` at any time to see which integrations are live vs degraded.

### `_is_real_key(value)` utility (`app/config.py`)

Module-level helper used by every integration:
```python
_is_real_key("")                  # → False
_is_real_key("   ")               # → False
_is_real_key("dummy-portkey-key") # → False  (any "dummy" prefix)
_is_real_key("sk-real-abc123")    # → True
```

---

## Settings Reference (`app/config.py`)

Settings are grouped into 10 numbered sections matching `.env.example`:

| Section | Key settings |
|---|---|
| 1. LLM Provider | `LLM_API_KEY`, `LLM_PROVIDER`, `LLM_FALLBACK_MODEL` |
| 2. Portkey Gateway | `PORTKEY_API_KEY`, `PORTKEY_SLM_VIRTUAL_KEY`, `PORTKEY_FRONTIER_VIRTUAL_KEY` |
| 3. Langfuse | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` |
| 4. Guardrails AI | `GUARDRAILS_VALIDATORS`, `GUARDRAILS_HUB_TOKEN` |
| 5. Worldsense MCP | `WORLDSENSE_ENABLED`, `WORLDSENSE_MCP_URL`, `WORLDSENSE_TIMEOUT_MS` |
| 6. Redteam MCP | `REDTEAM_ENABLED`, `REDTEAM_MIN_PROMPTS`, `REDTEAM_SCHEDULE` |
| 7. Semantic Cache | `CACHE_SIMILARITY_THRESHOLD` |
| 8. Policy & Pipeline | `POLICY_FILE_PATH`, `EMBEDDING_MODEL`, `VECTOR_STORE_TOP_K` |
| 9. Telemetry | `TELEMETRY_SINK`, `TELEMETRY_LOG_FILE` |
| 10. Obot Governance | `OBOT_ENABLED`, `OBOT_MAX_TOOL_CALLS_DEFAULT`, `OBOT_LATENCY_BUDGET_MS` |

`extra="ignore"` is set on the model so unknown env vars (e.g. old `OPENAI_API_KEY`) are
silently ignored and never cause startup errors.

---

## The 5-Stage Safety Pipeline

The pipeline is defined in `app/main.py` as `run_pipeline()` and stored at
`app.state.pipeline_fn`. Each request travels through five sequential checkpoints.
A `HARD_BLOCK` verdict at any stage short-circuits all downstream stages.

### Stage 0 — Semantic Cache (Phase 3)

**File:** `app/router/semantic_cache.py`

Before the Orchestrator runs, `run_pipeline()` checks the Semantic Cache when
`profile.cache_enabled=True`. A hit short-circuits all five stages: the LLM is never called
and `ctx.routing_decision` stays `None`.

- **Lookup**: cosine-similarity scan via FAISS; hit threshold = `CACHE_SIMILARITY_THRESHOLD` (default 0.92).
- **Store**: after `PASS_AND_DELIVER` / `COMPRESS_AND_EDIT`, response stored with per-profile TTL.
- **TTL eviction**: expired entries pruned on every lookup; `invalidate_expired()` for background use.
- **Graceful degradation**: no-op stub when `gptcache` absent; `INFO` at startup.
- **Exception isolation**: any index error → `SEMANTIC_CACHE_ERROR` logged, `hit=False` returned.

---

### Stage 1 — Orchestrator (Micro-Judges & Policy Layer)

**Files:** `app/judges/orchestrator.py`, `app/judges/p1_judge.py`, `app/judges/p2_judge.py`,
`app/judges/p3_judge.py`, `app/judges/pii_masking.py`, `app/policy/loader.py`

P1, P2, and P3 judges run concurrently via `asyncio.gather` inside `asyncio.wait_for(inspection_timeout_ms)`.

#### PII Masking Engine — three-tier design

- **Tier 1 — NLPMasker**: LLM Guard `Anonymize` (Presidio-backed); used when `llm-guard` installed.
- **Tier 1.5 — GLiNERMasker** *(Phase 3)*: zero-shot NER for `UseCaseProfile.custom_entity_terms`.
  Produces `[CUSTOM_ENTITY_REDACTED_N]` placeholders. Skipped when terms empty, model None, or `_gliner_degraded`.
  Startup round-trip validation; failure → `GLINER_DEGRADED` alert, `is_healthy` unchanged.
- **Tier 2 — RegexOnlyMasker**: compiled-regex scanner (SSN, email, phone, CC). Always available.
- Startup validation: 5 synthetic prompts; NLP failure → downgrade to regex + `MASKING_DEGRADED_TO_REGEX`.
  Both tiers fail → `is_healthy=False` → HTTP 503.

#### Policy Loader

YAML/JSON hot-reload via watchdog (≤5 s). Atomic swap on validation success. Two built-in
profiles always available: `customer_chatbot`, `internal_copilot`. `get_profile()` is `async`.

#### P1, P3 Judges

P1: LLM Guard Toxicity + PromptInjection (stub fallback when absent).
P3: tiktoken token count + spaCy dependency parse (stub fallback when absent).

**Failure isolation**: P1 exception → `BLOCK/BLOCK`; P2 exception → `pii_count=sys.maxsize`; P3 → `AMBIGUOUS`.

---

### Stage 2 — Model Router

**File:** `app/router/model_router.py`

Three-tier dispatch in priority order:

#### Tier A — Portkey Gateway (recommended for production)

When `_is_real_key(PORTKEY_API_KEY)`, all calls go through Portkey with:
```
x-portkey-api-key:      PORTKEY_API_KEY
x-portkey-virtual-key:  PORTKEY_SLM_VIRTUAL_KEY or PORTKEY_FRONTIER_VIRTUAL_KEY
x-portkey-provider:     LLM_PROVIDER   ← routes to correct upstream (openai/anthropic/google/grok)
```
Portkey handles retries, SLM↔Frontier fallback, and cost tracking.

#### Tier B — Direct provider call (when Portkey absent, `LLM_API_KEY` set)

Uses `openai.AsyncOpenAI` with a provider-specific base URL:

| `LLM_PROVIDER` | Base URL |
|---|---|
| `openai` | *(default OpenAI)* |
| `google` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `anthropic` | `https://api.anthropic.com/v1/` |
| `grok` | `https://api.x.ai/v1/` |
| `generic` | *(default OpenAI base)* |

Model name comes from `LLM_FALLBACK_MODEL` (default `gpt-4o-mini`).
Confirmed working with **Google Gemini** (`LLM_PROVIDER=google`, `LLM_FALLBACK_MODEL=gemini-2.5-flash`).

#### Tier C — Mock contextual responses (always available)

When both Portkey and LLM_API_KEY are absent/dummy, returns realistic canned responses for
local dev, CI, and tests. Never raises an exception.

**Startup log**: `PORTKEY_ACTIVE provider=<X>` or `PORTKEY_DEGRADED — mock responses active`.

#### Guardrails AI Output Validation (`app/judges/output_validator.py`)

- `load_validators()`: tries `gd.hub.load(vid)`; on failure attempts `_hub_install(vid)` subprocess
  then retries. Uses `GUARDRAILS_HUB_TOKEN` env for private validators.
- Structured logs: `GUARDRAILS_LOADED`, `GUARDRAILS_SKIPPED`, `GUARDRAILS_DEGRADED`.
- `action="fix"` → pipeline continues with fixed output; `"filter"` or `"exception"` → `HARD_BLOCK`.

---

### Stage 3 — Groundedness Auditor (two-stage, Phase 3)

**Files:** `app/groundedness/auditor.py`, `app/groundedness/nli_scorer.py`, `app/groundedness/vector_store.py`

**Stage 3a — Embedding similarity** (always): FAISS cosine similarity → `groundedness_score ∈ [0.0, 1.0]`.

**Stage 3b — NLI cross-encoder** (when `nli_scorer` available): scores each `(doc_text, response)` pair;
aggregates with `CONTRADICTION > ENTAILMENT > NEUTRAL`; sets `AuditResult.nli_label` and
`technique="nli_embedding_similarity"`. Any exception → `NLI_SCORER_ERROR` logged, `nli_label=None`.

**NLIScorer**: wraps `cross-encoder/nli-deberta-v3-small`; all inference via `asyncio.to_thread`.
Graceful no-op when `sentence-transformers` absent.

---

### Stage 3c — Worldsense Multi-Turn Agentic Oversight

**File:** `app/oversight/worldsense_oversight.py`

URL sourced from `settings.WORLDSENSE_MCP_URL` (previously a hard-coded `os.getenv` call).
Three-tier evaluation: MCP server → local SDK → heuristic. Returns `SAFE`, `RISK_DETECTED`
(→ `ESCALATE_TO_HUMAN`), or `CONSEQUENCE_ALERT` (→ `HARD_BLOCK`). Bounded by
`WORLDSENSE_TIMEOUT_MS` (default 300 ms); timeout → `RISK_DETECTED`.

---

### Stage 4 — Triage Gateway

**File:** `app/triage/gateway.py`

| Priority | State | Condition |
|---|---|---|
| **0** | **`HARD_BLOCK`** | `nli_label == "CONTRADICTION"` → `blocking_reason="NLI_CONTRADICTION"` |
| 1 | `HARD_BLOCK` | Upstream HARD_BLOCK OR `groundedness_score < 0.5` |
| 2 | `ESCALATE_TO_HUMAN` | `0.5 ≤ score ≤ pass_threshold` OR `p3_clarity=AMBIGUOUS` |
| 3 | `COMPRESS_AND_EDIT` | `score > threshold` AND `token_count > compression_threshold` |
| 4 | `PASS_AND_DELIVER` | All other cases |

Priority 0 was added in Phase 3. All other priorities unchanged.

---

## SSE Streaming Endpoint (Phase 3)

**File:** `app/ingress/streaming_router.py`

`POST /v1/chat/stream` — streams via `StreamingResponse(media_type="text/event-stream")`.

Full pre-flight pipeline → HARD_BLOCK gate → cache lookup → token streaming → per-chunk
validation (including `flush_remaining()` chunks) → post-stream NLI audit → cache store → `[DONE]`.

| Frame | Meaning |
|---|---|
| `data: <chunk>` | Clean content chunk |
| `data: [DONE]` | End of clean stream (always final) |
| `data: [REDACTED DUE TO POLICY]` | Policy violation — stream severed |
| `data: [STREAM_ERROR]` | Unhandled LLM exception |

---

## Integration Health Endpoint

**File:** `app/config_health/router.py`

`GET /v1/config/health` — returns HTTP 200 with live status of all five external integrations.
Reads only in-memory state (`app.state`, `settings`, `_LOADED_VALIDATORS`). Zero outbound calls.
Responds in <50 ms. No auth required.

```json
{
  "portkey":    { "status": "active|degraded", "detail": "..." },
  "langfuse":   { "status": "active|degraded", "detail": "..." },
  "guardrails": { "status": "active|degraded", "detail": "..." },
  "worldsense": { "status": "active|degraded", "detail": "..." },
  "llm_direct": { "status": "active|degraded", "detail": "..." }
}
```

Example with Gemini key configured:
```json
{ "llm_direct": { "status": "active", "detail": "LLM_API_KEY configured; fallback model=gemini-2.5-flash" } }
```

---

## Lifespan — Steps 1–11 (`app/main.py`)

| Step | What | Stored at |
|---|---|---|
| 1 | PolicyLoader start + watchdog | `app.state.policy_loader` |
| 2 | P1 scanner models | module-level in `p1_judge.py` |
| 3 | P3 spaCy model | module-level in `p3_judge.py` |
| 4 | PIIMaskingEngine + startup validation | `app.state.pii_engine` |
| 5 | TelemetryLogger + LangfuseTracer + Guardrails validators | `app.state.telemetry_logger`, `app.state.langfuse_tracer` |
| 6 | RouteLLM Controller | `app.state.routellm_controller` |
| 7 | SemanticCache | `app.state.semantic_cache` |
| 8 | GLiNER model | `app.state.gliner_model`, `pii_engine._app_gliner_model` |
| 9 | NLIScorer | `app.state.nli_scorer` |
| 10 | Redteam MCP health probe (`localhost:9200/health`, 2 s) | `app.state.redteam_mcp_healthy` |
| 11 | Worldsense MCP health probe (`WORLDSENSE_MCP_URL base /health`, 2 s) | `app.state.worldsense_mcp_healthy` |

After Step 11, a structured `INFO` startup summary is logged (no secret values):
```
ControlPlane.ai configuration summary:
  LLM direct key : CONFIGURED / NOT CONFIGURED
  Portkey        : ACTIVE (provider=X) / DEGRADED (mock)
  Langfuse       : ACTIVE (host=H) / DEGRADED (stdout)
  Guardrails     : ACTIVE (N validators) / DEGRADED (none loaded)
  Worldsense MCP : ACTIVE (URL) / DEGRADED (heuristic)
```

---

## Ancillary Endpoints & Routers

| Router | Endpoint | Purpose |
|---|---|---|
| `app.ingress.router` | `POST /v1/chat` | Primary (non-streaming) LLM interaction |
| `app.ingress.streaming_router` | `POST /v1/chat/stream` | SSE streaming |
| `app.config_health.router` | `GET /v1/config/health` | Live integration status |
| `app.telemetry.router` | `GET /v1/metrics`, `GET /v1/metrics/accuracy` | Aggregate metrics |
| `app.feedback.router` | `GET /v1/feedback/export`, `POST /v1/feedback/override` | Operator overrides |
| `app.redteam.router` | `POST /v1/redteam/run`, `GET /v1/redteam/report` | Adversarial testing |

---

## Observability

**Langfuse** (`app/observability/langfuse_tracer.py`): key validation uses `_is_real_key()` —
whitespace-only keys never trigger a connection attempt. Logs `LANGFUSE_ACTIVE host=...` or
`LANGFUSE_DEGRADED — stdout fallback active`. Buffered retry loop for offline events.

**Telemetry records** include Phase 3 fields: `cache_hit: bool` and `nli_label`.

---

## Redteam MCP Server (Phase 3)

**File:** `mcp_servers/redteam/server.py` — isolated FastAPI process, zero `app.*` imports.

`POST /run` priority: PyRIT → Garak → built-in 5-category library (jailbreaks, injection,
toxicity, PII extraction, competitor injection). `GET /health`, `GET /report`.

`RedTeamRunner.run()` tries MCP first (`POST {MCP_URL}/run`, 120 s timeout); falls back to
in-process on any error. `_record_breakthrough()` logs `RED_TEAM_BREAKTHROUGH` at ERROR level
via Langfuse for both MCP and in-process paths.

---

## Developer Automation — Kiro Hooks

**PostFileSave** hook fires when a policy YAML/JSON is saved:
1. `GET localhost:9200/health` (2 s timeout) — logs `REDTEAM_MCP_UNAVAILABLE` on failure.
2. Always proceeds to `POST /v1/redteam/run` on the Gateway.
3. Exits 0 — never blocks the save.

---

## Data Models (`app/models.py`)

**Phase 3 additions:**

```python
# UseCaseProfile
cache_enabled: bool = False
cache_ttl_seconds: int = Field(ge=1, default=300)
cache_similarity_threshold: float = Field(ge=0.0, le=1.0, default=0.92)
custom_entity_terms: list[str] = Field(default_factory=list)

# TelemetryRecord
cache_hit: bool = False
nli_label: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"] | None = None

# AuditResult
nli_label: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"] | None = None
```

**API key integration additions:**

```python
# IntegrationStatus — used by GET /v1/config/health
class IntegrationStatus(BaseModel):
    status: Literal["active", "degraded"]
    detail: str

class ConfigHealthResponse(BaseModel):
    portkey: IntegrationStatus
    langfuse: IntegrationStatus
    guardrails: IntegrationStatus
    worldsense: IntegrationStatus
    llm_direct: IntegrationStatus
```

---

## Test Suite

All tests live in `tests/unit/`. **245 tests pass, 4 skipped** in the current environment.

### Pre-Phase-3 tests (154 tests)

| Test file | What it covers |
|---|---|
| `test_models_properties.py` | Pydantic field-range validation |
| `test_policy_loader.py` | Hot-reload, atomic swap, built-in profiles |
| `test_ingress.py` | 422/503/504 responses, UUID v4, state isolation |
| `test_ingress_properties.py` | Invalid rejection, latency budget, concurrent isolation |
| `test_pii_masking.py` | Mask/unmask round-trip, startup validation |
| `test_pii_properties.py` | Token replacement, round-trip fidelity |
| `test_pii_graceful_degradation.py` | NLP→regex downgrade, `MASKING_DEGRADED_TO_REGEX` alert |
| `test_judges.py` | P1/P2/P3 verdicts, error defaults, edge cases |
| `test_orchestrator.py` | Concurrent execution, P1 short-circuit, timeout, failure isolation |
| `test_telemetry_queue.py` | Async queue, retry back-off, 90-day retention floor |
| `test_output_validator_properties.py` | P-OV-1 through P-OV-7 |

### Phase 3 property tests (74 tests)

| Test file | Properties |
|---|---|
| `test_phase3_model_properties.py` | SC-4, NLI-1, `cache_ttl_seconds` and `cache_similarity_threshold` validation |
| `test_semantic_cache_properties.py` | SC-2 (TTL eviction), SC-3 (exception isolation) |
| `test_nli_scorer_properties.py` | NLI-3 (aggregation priority) |
| `test_groundedness_auditor_properties.py` | NLI-1 (score range), NLI-4 (scorer exception isolation) |
| `test_triage_gateway_properties.py` | NLI-2 (CONTRADICTION → HARD_BLOCK) |
| `test_gliner_masker_properties.py` | GL-1 through GL-4 |
| `test_streaming_endpoint_properties.py` | SSE-1 through SSE-4 |
| `test_redteam_properties.py` | RT-1 (MCP fallback), RT-2 (no `app.*` imports), RT-3 (breakthrough logging) |

### API key integration tests (17 tests)

| Test file | What it covers |
|---|---|
| `test_config_health.py` | `_is_real_key()` edge cases; `GET /v1/config/health` schema, status, latency |
| `test_model_router_provider.py` | `LLM_API_KEY` rename; `LLM_FALLBACK_MODEL`; mock path when keys absent |

---

## Infrastructural State

### Python Environment

- Python 3.13.x; `pyproject.toml` with `>=` lower bounds.
- Primary venv: `.venv` in project root.
- Worldsense MCP: `mcp_servers/worldsense/requirements.txt` (isolated venv).
- Redteam MCP: `mcp_servers/redteam/requirements.txt` (isolated venv).

### Key Dependencies

| Package | Role | Required? |
|---|---|---|
| **FastAPI** + **uvicorn** | HTTP framework and ASGI server | Yes |
| **Pydantic v2** + **pydantic-settings** | Validation and env-var config | Yes |
| **watchdog** | Policy Layer hot-reload | Yes |
| **httpx** | Async HTTP (MCP probes, router, streaming) | Yes |
| **openai** | Direct LLM calls (OpenAI-compatible) + Gemini via base_url | Yes |
| **FAISS** (`faiss-cpu`) | Groundedness Auditor + SemanticCache | Yes |
| **tiktoken** | P3 Judge token counting | Yes |
| **hypothesis** + **pytest-asyncio** | Property-based + async testing | Dev |
| **spaCy** (`en_core_web_sm`) | P3 Judge dependency parse | Optional |
| **llm-guard** | NLP PII scanning Tier 1 | Optional |
| **routellm** + **portkey-ai** | RouteLLM complexity router + Portkey dispatch | Optional |
| **langfuse** | Distributed tracing | Optional |
| **guardrails-ai** | Output validation chain | Optional |
| **gptcache** | SemanticCache embedding + FAISS index | Optional |
| **sentence-transformers** | NLI cross-encoder for groundedness | Optional |
| **gliner** | Custom entity NER Tier 1.5 | Optional |
| **worldsense** | Multi-turn oversight (also via MCP) | Optional |

### LLM Provider Compatibility

The gateway's direct-call path uses the OpenAI Python SDK with provider-specific base URLs:

| Provider | `LLM_PROVIDER` | Base URL |
|---|---|---|
| OpenAI | `openai` | *(default)* |
| Google Gemini | `google` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Anthropic | `anthropic` | `https://api.anthropic.com/v1/` |
| Grok / xAI | `grok` | `https://api.x.ai/v1/` |
| Any compatible | `generic` | *(default OpenAI base)* |

Tested live with `gemini-2.5-flash` (free tier: ~1,500 req/day). Note: `gemini-1.5-flash` and
`gemini-2.0-flash` were retired June 2026 — use `gemini-2.5-flash` or newer.

### Notable Fixed Issues

- **`OPENAI_API_KEY` → `LLM_API_KEY`**: renamed for provider generality. Old `OPENAI_API_KEY`
  in `.env` is silently ignored (`extra="ignore"`); operators must rename it.
- **Gemini model deprecation**: `gemini-1.5-flash` / `gemini-2.0-flash` shut down June 1, 2026.
  Current free-tier model is `gemini-2.5-flash`.
- **`run_orchestrator` alias**: `app/judges/orchestrator.py` exports
  `run_orchestrator = run_micro_judges` to satisfy local imports in streaming_router.
- **`get_profile()` is async**: streaming router calls `await policy_loader.get_profile()`.
  Caught during Phase 3 property test development.
- **`flush_remaining()` validation gap**: initial streaming router did not validate
  `flush_remaining()` chunks. Fixed — all chunks now pass through `validate_output()`.
- **`_is_real_key` whitespace guard**: plain `not settings.LANGFUSE_PUBLIC_KEY` check
  allowed whitespace-only strings to pass as valid keys. Replaced with `_is_real_key()`.
- **Worldsense URL in config**: `_MCP_URL` was constructed from `os.getenv()` inside
  `worldsense_oversight.py`. Moved to `settings.WORLDSENSE_MCP_URL` for consistency.
- **Hypothesis fixture scoping** and **`_FallbackScanner` closure bug**: see Phase 3 notes.
