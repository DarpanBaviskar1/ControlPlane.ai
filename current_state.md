# ControlPlane.ai — Current State and Architecture

## Overview

**ControlPlane.ai** is an Enterprise AI Proxy Gateway built on FastAPI. It mediates every
interaction between enterprise applications or agents and underlying Large Language Models
through a deterministic five-stage safety pipeline that enforces safety, cost governance,
groundedness assurance, and output triage before a response is delivered to callers.

This document reflects the current implementation state after three rounds of development:
the original five-stage pipeline, five targeted hardening improvements, and the **Phase 3
upgrades** that added Semantic Cache, NLI-Based Groundedness Auditing, GLiNER Custom Entity
Masking, SSE Streaming, and an isolated Redteam MCP Server.

---

## Repository Layout (top-level)

```
app/
  config.py                   — Pydantic-Settings config (all env-var driven)
  main.py                     — FastAPI app, lifespan, run_pipeline()
  models.py                   — Shared Pydantic/dataclass data models
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
    pii_masking.py            — Two-tier PII masking engine (NLP + regex + GLiNER)
  oversight/
    worldsense_oversight.py   — Multi-turn agentic consequence-chain oversight
  policy/
    loader.py                 — Hot-reloading policy file loader
  redteam/
    router.py                 — POST /v1/redteam/run + GET /v1/redteam/report
    runner.py                 — MCP-first red team orchestrator (Phase 3)
  router/
    model_router.py           — RouteLLM complexity router + Portkey LLM dispatch
    semantic_cache.py         — GPTCache-backed vector similarity cache (Phase 3)
  telemetry/
    logger.py                 — Async telemetry queue + RetentionManager
    router.py                 — GET /v1/metrics + /v1/metrics/accuracy
  triage/
    compressor.py             — Token-budget summarisation for COMPRESS_AND_EDIT
    gateway.py                — Four-state priority triage matrix
  feedback/
    router.py                 — GET /v1/feedback/export + POST /v1/feedback/override
  observability/
    langfuse_tracer.py        — Langfuse span tracing with stdout fallback

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
  unit/                       — 228 unit + property tests (4 skipped)

.kiro/
  hooks/
    redteam-on-policy-save.json
    redteam_trigger.py        — PostFileSave hook: MCP health check + redteam trigger
  specs/
    gateway-phase3-upgrades/  — Requirements, design, and tasks for Phase 3
  steering/
    performance-budget.md     — Per-component latency ceilings (auto-included)
```

---

## The 5-Stage Safety Pipeline

The pipeline is defined in `app/main.py` as `run_pipeline()` and stored at
`app.state.pipeline_fn`. Each request travels through five sequential checkpoints.
A `HARD_BLOCK` verdict at any stage short-circuits all downstream stages.

### Stage 0 — Semantic Cache (Phase 3)

**File:** `app/router/semantic_cache.py`

Before the Orchestrator runs, `run_pipeline()` checks the Semantic Cache when
`profile.cache_enabled=True`.

- **Lookup**: `await app.state.semantic_cache.lookup(ctx.working_prompt, profile.cache_ttl_seconds)`
  — embeds the masked prompt and runs a cosine-similarity scan against stored entries via FAISS.
  A hit (similarity ≥ `cache_similarity_threshold`, default 0.92) short-circuits all five stages:
  `ctx.routing_decision` is never set (remains `None`), the LLM is never called, and the cached
  response is delivered directly.
- **Store**: After a clean `PASS_AND_DELIVER` or `COMPRESS_AND_EDIT` outcome, the response is
  written to the cache with the profile's TTL.
- **TTL eviction**: Expired entries are pruned on every lookup (`expiry_ts < time.monotonic()`).
  `invalidate_expired()` is also available for background maintenance.
- **Graceful degradation**: When `gptcache` is not installed, `_GPTICACHE_AVAILABLE=False`;
  `lookup()` always returns `hit=False` and `store()` is a no-op. An `INFO` log is emitted once
  at startup.
- **Exception isolation**: Any exception from the embedding or FAISS index is caught, logged as
  `SEMANTIC_CACHE_ERROR`, and returns `hit=False` — never propagates.
- **Config**: `CACHE_SIMILARITY_THRESHOLD: float = 0.92` in `app/config.py`.
- **Telemetry**: `TelemetryRecord.cache_hit` is stamped after every request.

---

### Stage 1 — Orchestrator (Micro-Judges & Policy Layer)

**Files:** `app/judges/orchestrator.py`, `app/judges/p1_judge.py`, `app/judges/p2_judge.py`,
`app/judges/p3_judge.py`, `app/judges/pii_masking.py`, `app/policy/loader.py`

The orchestrator runs P1, P2, and P3 judges concurrently via `asyncio.gather` inside
`asyncio.wait_for(inspection_timeout_ms)`.

#### PII Masking Engine (`app/judges/pii_masking.py`) — three-tier design

- **Tier 1 — NLPMasker**: Wraps LLM Guard `Anonymize` (Presidio-backed). Highest accuracy;
  used when `llm-guard` is installed.
- **Tier 1.5 — GLiNERMasker** *(Phase 3)*: Zero-shot NER for custom corporate entity terms
  supplied per-profile via `UseCaseProfile.custom_entity_terms`. Called after Tier 1, before
  Tier 2. Produces `[CUSTOM_ENTITY_REDACTED_N]` placeholders (N starting at 1, in document
  order). Skipped entirely when `custom_entity_terms=[]`, `gliner_model=None`, or
  `_gliner_degraded=True`. Exception caught → `GLINER_SCAN_ERROR` logged → Tier 1/2 result
  returned, no propagation. Startup validation runs a round-trip; on failure sets
  `_gliner_degraded=True` and emits `GLINER_DEGRADED` alert — `is_healthy` stays `True`.
- **Tier 2 — RegexOnlyMasker**: Pure compiled-regex scanner covering SSNs, email addresses,
  US phone numbers, and credit card numbers. Zero extra dependencies, ~1 ms per call.
  Always available.
- **Startup validation**: On startup, 5 synthetic PII prompts run through `mask → unmask`
  round-trip. If the NLP tier fails, it downgrades to the regex tier and emits
  `MASKING_DEGRADED_TO_REGEX` HIGH-priority alert. Only if the regex tier also fails does
  `is_healthy=False` cause the ingress to return HTTP 503.
- `mask()` signature: `mask(prompt, request_id, custom_entity_terms=None, gliner_model=None)`
- `unmask()` restores placeholders from all three tiers using the per-request `placeholder_map`.

#### GLiNER Masker (`app/judges/gliner_masker.py`) *(Phase 3)*

- `scan_sync(prompt, entity_terms, gliner_model)`: calls `gliner_model.predict_entities()`,
  iterates spans in document order, builds `{[CUSTOM_ENTITY_REDACTED_N]: original}` map.
- `scan(prompt, entity_terms, gliner_model)`: async wrapper via `asyncio.to_thread`.
- `_CUSTOM_PLACEHOLDER_RE = re.compile(r"\[CUSTOM_ENTITY_REDACTED_\d+\]")` at module level.
- Graceful no-op when `gliner` is not installed (`_HAS_GLINER=False`).

#### P2 Judge (`app/judges/p2_judge.py`)

Calls `PIIMaskingEngine.mask(prompt, request_id, profile.custom_entity_terms, gliner_model)`.
The `gliner_model` is threaded from `pii_engine._app_gliner_model` (set in lifespan Step 8)
through the orchestrator's `_safe_p2` → `p2_judge` call chain.

#### Policy Loader (`app/policy/loader.py`)

Loads `UseCaseProfile` configurations from YAML or JSON. Uses `watchdog` for hot-reload within
5 seconds of file modification. Atomically swaps the live config only if all profiles pass
Pydantic validation. Two built-in profiles (`customer_chatbot`, `internal_copilot`) are always
available. `get_profile()` is an `async` method (uses an `asyncio.Lock` for thread-safe reads).

#### P1, P3 Judges

Unchanged from Phase 2. P1 uses LLM Guard Toxicity + PromptInjection scanners; P3 uses
tiktoken + spaCy dependency parse.

**Timeout handling**: `asyncio.wait_for` timeout → `upstream_triage_state=ESCALATE_TO_HUMAN`.

**Failure isolation**: P1 exception → `BLOCK/BLOCK`; P2 exception → `pii_count=sys.maxsize`;
P3 exception → `AMBIGUOUS`.

---

### Stage 2 — Model Router (RouteLLM + Portkey)

**Files:** `app/router/model_router.py`

Unchanged from Phase 2. RouteLLM `Controller` classifies complexity; Portkey provides unified
LLM access, retries, and fallback between SLM and Frontier tiers.

**Guardrails AI Output Validation** (`app/judges/output_validator.py`): `action="fix"` →
pipeline continues with fixed output; `action="filter"` or `"exception"` → `HARD_BLOCK`.
Internal exceptions also produce `HARD_BLOCK` and never propagate.

---

### Stage 3 — Groundedness Auditor (two-stage, Phase 3)

**Files:** `app/groundedness/auditor.py`, `app/groundedness/nli_scorer.py`,
`app/groundedness/vector_store.py`

#### Stage 3a — Embedding Similarity (always runs)

Embeds the LLM response and computes mean cosine similarity against the top-K most relevant
documents retrieved from the FAISS vector store. Returns `groundedness_score` in `[0.0, 1.0]`.
Vector store exceptions produce `score=0.0`, `is_unverified=True`, `VECTOR_STORE_UNAVAILABLE`.

#### Stage 3b — NLI Cross-Encoder (Phase 3, runs when nli_scorer is available)

After FAISS retrieval, `audit()` accepts an optional `nli_scorer: NLIScorer | None` parameter.
When supplied and documents were retrieved:

1. Builds `(document_text, response)` pairs.
2. Calls `await nli_scorer.score_pairs(pairs)` — offloaded to `asyncio.to_thread`.
3. Aggregates per-pair labels using `NLIScorer.aggregate()` priority rule:
   **CONTRADICTION > ENTAILMENT > NEUTRAL**.
4. Sets `AuditResult.nli_label` and `technique="nli_embedding_similarity"`.

**Exception isolation**: any cross-encoder exception is caught, `NLI_SCORER_ERROR` is logged,
`nli_label=None` is returned, and `technique` falls back to `"embedding_similarity"`.

#### NLI Scorer (`app/groundedness/nli_scorer.py`) *(Phase 3)*

- Wraps `sentence_transformers.CrossEncoder("cross-encoder/nli-deberta-v3-small")`.
- `score_pairs_sync(pairs)` → `list[tuple[NLILabel, float]]`; returns `[]` when
  `_HAS_SENTENCE_TRANSFORMERS=False`.
- `score_pairs(pairs)` → async wrapper via `asyncio.to_thread`.
- `aggregate(labels)` static method implementing CONTRADICTION > ENTAILMENT > NEUTRAL;
  returns `"NEUTRAL"` for empty input.
- Graceful no-op when `sentence-transformers` is not installed; `INFO` logged at startup.

---

### Stage 3c — Worldsense Multi-Turn Agentic Oversight

**Files:** `app/oversight/worldsense_oversight.py`, `mcp_servers/worldsense/server.py`

Unchanged from Phase 2. Three-tier evaluation: MCP server → local SDK → heuristic evaluator.
Returns `SAFE`, `RISK_DETECTED` (→ `ESCALATE_TO_HUMAN`), or `CONSEQUENCE_ALERT` (→ `HARD_BLOCK`).
Bounded by `WORLDSENSE_TIMEOUT_MS` (default 300 ms). Timeout → `RISK_DETECTED` (fail-safe).

---

### Stage 4 — Triage Gateway

**Files:** `app/triage/gateway.py`, `app/triage/compressor.py`

**Phase 3 addition — Priority 0 (NLI CONTRADICTION):**

A new Priority 0 check precedes all existing rules:

| Priority | State | Condition |
|---|---|---|
| **0** | **`HARD_BLOCK`** | **`nli_label == "CONTRADICTION"`** → `blocking_reason="NLI_CONTRADICTION"` |
| 1 | `HARD_BLOCK` | Upstream `triage_state=HARD_BLOCK` OR `groundedness_score < 0.5` |
| 2 | `ESCALATE_TO_HUMAN` | `0.5 ≤ score ≤ profile.groundedness_pass_threshold` OR `p3_clarity=AMBIGUOUS` |
| 3 | `COMPRESS_AND_EDIT` | `score > threshold` AND `response_token_count > token_compression_threshold` |
| 4 | `PASS_AND_DELIVER` | All other cases |

`evaluate()` gains an optional `nli_label` parameter (default `None`). Existing Priority 1–4
logic is completely unchanged. `nli_label=None` or `"ENTAILMENT"` or `"NEUTRAL"` never
triggers Priority 0.

---

## SSE Streaming Endpoint (Phase 3)

**File:** `app/ingress/streaming_router.py`

`POST /v1/chat/stream` — streams LLM output as Server-Sent Events.

**Request flow:**

1. **Pre-flight pipeline**: full Orchestrator (P1 + P2 + P3 + PII masking) runs on the complete
   prompt before any token is forwarded.
2. **HARD_BLOCK gate**: if `upstream_triage_state=HARD_BLOCK`, the generator returns immediately
   with zero `data:` frames emitted.
3. **Semantic Cache lookup**: on `profile.cache_enabled=True`, checks the cache first. A hit
   yields a single `data: <cached_response>` frame followed immediately by `data: [DONE]`.
4. **Token streaming**: LLM tokens streamed via Portkey (`_stream_tokens_from_llm()`).
5. **SlidingWindow**: tokens are fed into `SlidingWindow.push()` which emits complete sentence-
   chunks at `[.!?]\s+` boundaries. Each chunk (including `flush_remaining()` chunks) is
   validated by `validate_output()`.
   - Validator `action="fix"` → emit `fixed_output`.
   - Validator `action="filter"` or `"exception"` → emit `data: [REDACTED DUE TO POLICY]`
     and close immediately. No further frames are emitted after this.
6. **Post-stream groundedness audit**: full two-stage `audit()` (embedding + NLI) on the
   assembled response. `nli_label="CONTRADICTION"` → emit `data: [REDACTED DUE TO POLICY]`
   and close.
7. **Cache store**: on clean stream completion, stores response in SemanticCache.
8. **Terminal frame**: clean streams always end with `data: [DONE]`.
9. **LLM error handling**: streaming exception emits `data: [STREAM_ERROR]` and closes.
10. **Telemetry**: `TelemetryRecord` written after stream closes with `cache_hit`, `nli_label`,
    total token count, and final triage state.

#### SSE frame format

```
data: <content>\n\n        — token chunk or cached response
data: [DONE]\n\n           — clean end-of-stream
data: [REDACTED DUE TO POLICY]\n\n  — mid-stream or post-stream policy violation
data: [STREAM_ERROR]\n\n   — unhandled LLM streaming exception
```

#### SlidingWindow (`app/ingress/sliding_window.py`) *(Phase 3)*

- `_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")` at module level.
- `push(token)` → `list[str]` of complete sentence-chunks ready to validate and emit.
- `flush_remaining()` → final partial chunk (if any) after the token stream ends.
- `_buffer: str` accumulates tokens between boundaries.

---

## Observability & Telemetry

**Files:** `app/telemetry/logger.py`, `app/telemetry/router.py`,
`app/observability/langfuse_tracer.py`

Unchanged from Phase 2, with two new `TelemetryRecord` fields:
- `cache_hit: bool = False` — set to `True` when a Semantic Cache hit occurs.
- `nli_label: Literal["ENTAILMENT","NEUTRAL","CONTRADICTION"] | None = None` — NLI verdict
  from the groundedness auditor.

---

## Redteam MCP Server (Phase 3)

**File:** `mcp_servers/redteam/server.py`

Isolated FastAPI application — **zero imports from any `app.*` module**.

| Endpoint | Method | Description |
|---|---|---|
| `/run` | POST | Execute adversarial red-team suite |
| `/health` | GET | Liveness probe → `{"status": "ok"}` |
| `/report` | GET | Most recent `RunResponse` or `{}` |

**Execution priority for `POST /run`:**
1. PyRIT multi-turn orchestrator (if `pyrit` installed).
2. Garak probe sweep (if `garak` installed).
3. Built-in five-category adversarial prompt library (always available):
   - Multi-turn jailbreaks
   - Direct prompt injection
   - Toxicity escalation
   - PII extraction attempts
   - Competitor-mention injection

**Infrastructure:**
- `mcp_servers/redteam/requirements.txt`: `fastapi>=0.111.0`, `uvicorn[standard]>=0.29.0`,
  `pydantic>=2.7.1`, `httpx>=0.27.0`, `pyrit>=0.4.0`, `garak>=0.9.0`
- `mcp_servers/redteam/mcp.json`: `command=python`, `args=["mcp_servers/redteam/server.py"]`,
  `env.REDTEAM_MCP_PORT=9200`, `port=9200`

---

## RedTeamRunner — MCP-First Delegation (Phase 3)

**File:** `app/redteam/runner.py`

`run()` now tries the MCP server before falling back to in-process execution:

1. `_try_mcp_run(tracer)`: `POST {MCP_URL}/run` with `httpx.AsyncClient(timeout=120.0)`.
   - On success: `_mcp_healthy=True`; parses response via `_parse_mcp_response()`.
   - On any exception: logs `REDTEAM_MCP_UNAVAILABLE`; sets `_mcp_healthy=False`; returns `None`.
2. If `_try_mcp_run()` returns `None`, existing in-process PyRIT + Garak + built-in library
   logic runs as fallback.
3. `_record_breakthrough(tracer, session_id, result)` — static helper used by **both** MCP and
   in-process paths. Records `RED_TEAM_BREAKTHROUGH` Langfuse span at `ERROR` severity for any
   `AttackResult.breakthrough=True`.

Class attributes: `MCP_URL: str = "http://localhost:9200"`, `_mcp_healthy: bool | None = None`.

---

## Lifespan — Steps 7–10 (Phase 3)

Added to the FastAPI lifespan context manager in `app/main.py`:

| Step | What | Stored at |
|---|---|---|
| 7 | `SemanticCache(similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD)` | `app.state.semantic_cache` |
| 8 | `GLiNER.from_pretrained("urchade/gliner_medium-v2.1")` via `asyncio.to_thread` | `app.state.gliner_model`; also `pii_engine._app_gliner_model` |
| 9 | `NLIScorer()` via `asyncio.to_thread` | `app.state.nli_scorer` |
| 10 | `GET http://localhost:9200/health` (non-blocking probe, 2 s timeout) | `app.state.redteam_mcp_healthy` |

All three optional-dependency steps degrade gracefully: store `None` and log `INFO` when the
package is absent. `streaming_router` is mounted via `app.include_router(streaming_router)`.

---

## Ancillary Endpoints & Routers

| Router | Prefix | Purpose |
|---|---|---|
| `app.ingress.router` | `POST /v1/chat` | Primary (non-streaming) LLM interaction endpoint |
| `app.ingress.streaming_router` *(Phase 3)* | `POST /v1/chat/stream` | SSE streaming endpoint |
| `app.telemetry.router` | `GET /v1/metrics`, `GET /v1/metrics/accuracy` | Aggregate metrics |
| `app.feedback.router` | `GET /v1/feedback/export`, `POST /v1/feedback/override` | Operator overrides |
| `app.redteam.router` | `POST /v1/redteam/run`, `GET /v1/redteam/report` | Adversarial testing |

---

## Developer Automation — Kiro Hooks

**Files:** `.kiro/hooks/redteam-on-policy-save.json`, `.kiro/hooks/redteam_trigger.py`

A **PostFileSave** hook (matcher: `\.(yaml|yml|json)$`) fires when a policy file is saved.
**Phase 3 addition**: before calling `POST /v1/redteam/run`, the hook script now:

1. Sends `GET http://localhost:9200/health` with a 2-second timeout.
2. On failure or non-200: logs `REDTEAM_MCP_UNAVAILABLE` to stderr.
3. **Always** proceeds to call `POST /v1/redteam/run` on the Gateway regardless of MCP health.

The hook still exits 0 — it never blocks the save operation.

---

## Developer Rules — Kiro Steering

**File:** `.kiro/steering/performance-budget.md` (`inclusion: auto`)

Unchanged from Phase 2. Per-component latency ceilings enforced for all CPU-bound scanner
changes. `asyncio.to_thread` rule applies to all three new Phase 3 CPU-bound components
(SemanticCache, NLIScorer, GLiNERMasker).

---

## Data Models (`app/models.py`)

### Phase 3 additions to existing models

**`UseCaseProfile`** (4 new fields):
```python
cache_enabled: bool = False
cache_ttl_seconds: int = Field(ge=1, default=300)
cache_similarity_threshold: float = Field(ge=0.0, le=1.0, default=0.92)
custom_entity_terms: list[str] = Field(default_factory=list)
```

**`TelemetryRecord`** (2 new fields):
```python
cache_hit: bool = False
nli_label: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"] | None = None
```

**`AuditResult`** (1 new field):
```python
nli_label: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"] | None = None
```

All additions are backward-compatible with default values.

---

## Test Suite

All tests live in `tests/unit/`. **228 tests pass, 4 skipped** in the current environment.

### Pre-Phase-3 tests (154 tests)

| Test file | What it covers |
|---|---|
| `test_models_properties.py` | Pydantic field-range validation (Property 22) |
| `test_policy_loader.py` | Hot-reload, atomic swap, built-in profiles (Property 1) |
| `test_ingress.py` | 422/503/504 responses, UUID v4 request IDs, state isolation |
| `test_ingress_properties.py` | Properties 2, 3, 4: invalid rejection, latency budget, concurrent isolation |
| `test_pii_masking.py` | Mask/unmask/discard round-trip, startup validation |
| `test_pii_properties.py` | Properties 6, 16: token replacement, round-trip fidelity |
| `test_pii_graceful_degradation.py` | Two-tier NLP→regex downgrade, `MASKING_DEGRADED_TO_REGEX` alert |
| `test_judges.py` | P1/P2/P3 verdicts, error defaults, edge cases |
| `test_orchestrator.py` | Concurrent execution, P1 short-circuit, timeout, failure isolation |
| `test_telemetry_queue.py` | Async queue, retry back-off, RetentionManager 90-day floor |
| `test_output_validator_properties.py` | P-OV-1 through P-OV-7: never raises, exception/filter→HARD_BLOCK, fix action, event-loop safety |

### Phase 3 property tests (74 tests)

| Test file | Properties covered |
|---|---|
| `test_phase3_model_properties.py` | SC-4 (`cache_hit` default), NLI-1 (`nli_label` default), `cache_ttl_seconds` validation, `cache_similarity_threshold` validation |
| `test_semantic_cache_properties.py` | SC-2 (TTL eviction: expired entries always miss), SC-3 (exception isolation: any error → `hit=False`) |
| `test_nli_scorer_properties.py` | NLI-3 (aggregation priority: CONTRADICTION wins for any input containing it) |
| `test_groundedness_auditor_properties.py` | NLI-1 (score range always `[0.0, 1.0]`), NLI-4 (scorer exception → `nli_label=None`, no propagation) |
| `test_triage_gateway_properties.py` | NLI-2 (CONTRADICTION → HARD_BLOCK/NLI_CONTRADICTION for any groundedness score) |
| `test_gliner_masker_properties.py` | GL-1 (placeholder format `[CUSTOM_ENTITY_REDACTED_\\d+]`), GL-2 (round-trip fidelity), GL-3 (empty terms skips tier), GL-4 (exception → Tier-1/2 result, no propagation) |
| `test_streaming_endpoint_properties.py` | SSE-1 (`/v1/chat` unchanged), SSE-2 (HARD_BLOCK → zero frames), SSE-3 (violation → `[REDACTED DUE TO POLICY]` as final frame), SSE-4 (clean stream → `[DONE]` as final frame) |
| `test_redteam_properties.py` | RT-2 (zero `app.*` imports in MCP server via AST scan), RT-1 (MCP unavailable → fallback, no raise), RT-3 (`breakthrough=True` → `RED_TEAM_BREAKTHROUGH`/ERROR logged) |

---

## Infrastructural State

### Python Environment

- Python 3.13.x; dependencies managed via `pyproject.toml` with `>=` lower bounds.
- Primary venv: `.venv` in the project root.
- Worldsense MCP server: `mcp_servers/worldsense/requirements.txt` for isolated venv.
- Redteam MCP server: `mcp_servers/redteam/requirements.txt` for isolated venv.

### Key Dependencies

| Package | Role |
|---|---|
| **FastAPI** + **uvicorn** | HTTP framework and ASGI server |
| **Pydantic v2** + **pydantic-settings** | Data validation and config |
| **watchdog** | Policy Layer hot-reload |
| **spaCy** (`en_core_web_sm`) | P3 Judge dependency parse |
| **tiktoken** | P3 Judge token counting |
| **FAISS** (`faiss-cpu`) | Groundedness Auditor + SemanticCache vector store |
| **httpx** | Async HTTP client (Worldsense/Redteam MCP calls, runner) |
| **hypothesis** + **pytest-asyncio** | Property-based and async testing |
| **llm-guard** *(optional)* | NLP-based PII scanning Tier 1; regex fallback when absent |
| **routellm** + **portkey-ai** | Model Router complexity classification and LLM dispatch |
| **langfuse** *(optional)* | Distributed tracing; stdout fallback |
| **guardrails-ai** *(optional)* | Output validation chain; no-op pass-through when absent |
| **gptcache** *(optional, Phase 3)* | SemanticCache embedding + FAISS index; no-op when absent |
| **sentence-transformers** *(optional, Phase 3)* | NLI cross-encoder for groundedness; no-op when absent |
| **gliner** *(optional, Phase 3)* | Custom entity NER Tier 1.5; skipped when absent |
| **worldsense** *(not in pyproject.toml)* | Available via MCP server isolation or local install |

### Notable Fixed Issues

- **`run_orchestrator` alias**: `app/judges/orchestrator.py` exports
  `run_orchestrator = run_micro_judges` to satisfy imports in `app/main.py` and
  `app/ingress/streaming_router.py`.
- **`get_profile()` is async**: `app/ingress/streaming_router.py` calls
  `await policy_loader.get_profile(...)` — not a plain call. Caught and fixed during
  Phase 3 property test development.
- **`flush_remaining()` validation gap**: The initial streaming router implementation did
  not run the output validator on `flush_remaining()` chunks. Fixed during Phase 3 property
  test development — all chunks (mid-stream and remainder) now pass through `validate_output()`.
- **Hypothesis fixture scoping**: All `@given` test bodies that mutate `_LOADED_VALIDATORS`
  call `_LOADED_VALIDATORS.clear()` at the start, because Hypothesis does not re-run
  `autouse` fixtures between generated examples.
- **`_FallbackScanner` closure bug**: Nested `@dataclass` definitions inside factory functions
  do not capture closure variables as field defaults; replaced with plain classes using
  `__init__` in the stub validators.
