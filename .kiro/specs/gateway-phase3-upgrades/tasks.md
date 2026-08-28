# Implementation Plan: Gateway Phase 3 Upgrades

## Overview

This plan implements five Phase 3 upgrades to the ControlPlane.ai Enterprise AI Proxy Gateway: Semantic Cache, NLI-Based Groundedness Auditor, GLiNER Custom Entity Masking, SSE Streaming with Sliding-Window Triage, and an isolated Redteam MCP Server. All changes are additive and backward-compatible. Data models are extended first, then independent components are implemented in parallel, and finally pipeline wiring connects everything together.

---

## Tasks

- [x] 1. Extend data models with Phase 3 fields
  - [x] 1.1 Add Phase 3 fields to `UseCaseProfile`, `TelemetryRecord`, and `AuditResult` in `app/models.py`
    - Add `cache_enabled: bool = False`, `cache_ttl_seconds: int = Field(ge=1, default=300)`, `cache_similarity_threshold: float = Field(ge=0.0, le=1.0, default=0.92)` to `UseCaseProfile`
    - Add `custom_entity_terms: list[str] = Field(default_factory=list)` to `UseCaseProfile`
    - Add `cache_hit: bool = False` and `nli_label: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"] | None = None` to `TelemetryRecord`
    - Add `nli_label: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"] | None = None` to the `AuditResult` dataclass
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 3.1, 6.6_

  - [ ]* 1.2 Write property tests for Phase 3 model fields
    - **Property SC-4: `cache_hit` telemetry default** — `TelemetryRecord` constructed without `cache_hit` must have `cache_hit=False`
    - **Property NLI-1 (partial): `nli_label` default** — `TelemetryRecord` and `AuditResult` constructed without `nli_label` must have `nli_label=None`
    - Validate that `cache_ttl_seconds < 1` raises a `ValidationError`
    - Validate that `cache_similarity_threshold` outside `[0.0, 1.0]` raises a `ValidationError`
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 6.6_

- [ ] 2. Implement Semantic Cache component
  - [-] 2.1 Create `app/router/semantic_cache.py` with `CacheLookupResult` dataclass and `SemanticCache` class
    - Implement top-of-file optional import guard for `gptcache`; set `_GPTICACHE_AVAILABLE` flag and log `INFO` when absent
    - Implement `SemanticCache.__init__` accepting `similarity_threshold` and `embedding_model`
    - Implement `lookup(masked_prompt, cache_ttl_seconds)` using `asyncio.to_thread` for embedding and FAISS query; return `CacheLookupResult(hit=False)` when `_GPTICACHE_AVAILABLE=False`
    - Implement `store(masked_prompt, response, ttl_seconds)` as a no-op when `_GPTICACHE_AVAILABLE=False`
    - Implement `invalidate_expired()` to prune TTL-expired entries and return count removed
    - Wrap all GPTCache index operations in `try/except`; log `SEMANTIC_CACHE_ERROR` and return `hit=False` on any exception
    - _Requirements: 1.3, 1.4, 1.7, 1.9, 1.11, 1.12, 6.3, 6.5_

  - [ ]* 2.2 Write property test for SemanticCache — Property SC-2: TTL eviction
    - **Property SC-2: TTL eviction** — For any `CacheEntry` with `expiry_ts < time.monotonic()`, `lookup()` must return `hit=False`
    - _Requirements: 1.7_

  - [ ]* 2.3 Write property test for SemanticCache — Property SC-3: Exception isolation
    - **Property SC-3: Exception isolation** — If the GPTCache index raises any exception, `lookup()` must return `CacheLookupResult(hit=False)` and must not propagate the exception
    - _Requirements: 1.9_

- [-] 3. Add `CACHE_SIMILARITY_THRESHOLD` to `app/config.py`
  - Add `CACHE_SIMILARITY_THRESHOLD: float = 0.92` setting to the application `Settings` / `config.py`
  - Ensure the value is passed to `SemanticCache(similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD)` in the lifespan step
  - _Requirements: 1.3, 1.4_

- [ ] 4. Implement NLI Scorer component
  - [-] 4.1 Create `app/groundedness/nli_scorer.py` with `NLIScorer` class and `NLILabel` type
    - Implement top-of-file optional import guard for `sentence_transformers.CrossEncoder`; set `_HAS_SENTENCE_TRANSFORMERS` flag and log `INFO` when absent
    - Implement `NLIScorer.__init__` loading `cross-encoder/nli-deberta-v3-small` synchronously (for use with `asyncio.to_thread` at startup)
    - Implement `score_pairs_sync(pairs)` that calls the cross-encoder and returns `list[tuple[NLILabel, float]]`; return empty list when `_HAS_SENTENCE_TRANSFORMERS=False`
    - Implement `score_pairs(pairs)` as `asyncio.to_thread(self.score_pairs_sync, pairs)`
    - Implement static `aggregate(labels)` with `CONTRADICTION > ENTAILMENT > NEUTRAL` priority rule
    - _Requirements: 2.3, 2.5, 2.7, 2.8, 6.3, 6.4, 6.5_

  - [ ]* 4.2 Write property test for NLIScorer — Property NLI-3: Aggregation priority
    - **Property NLI-3: Aggregation priority** — For any label list containing at least one `"CONTRADICTION"`, `NLIScorer.aggregate()` must return `"CONTRADICTION"` regardless of other labels present
    - _Requirements: 2.9_

- [ ] 5. Update GroundednessAuditor with NLI pipeline
  - [~] 5.1 Modify `app/groundedness/auditor.py` to accept an optional `nli_scorer` parameter and run the two-stage pipeline
    - Update `audit()` signature to accept `nli_scorer: NLIScorer | None = None`
    - After FAISS retrieval, build `(doc.text, response)` pairs and call `await nli_scorer.score_pairs(pairs)` if `nli_scorer` is not `None`
    - Wrap `score_pairs` call in `try/except`; log `NLI_SCORER_ERROR` and set `nli_label=None` on exception (do not propagate)
    - Set `technique="nli_embedding_similarity"` when NLI runs, `"embedding_similarity"` otherwise
    - Set aggregate `nli_label` via `NLIScorer.aggregate()` using the priority rule
    - Return `AuditResult` with `nli_label` populated accordingly
    - _Requirements: 2.3, 2.6, 2.8, 2.9, 2.10, 6.5_

  - [ ]* 5.2 Write property test for GroundednessAuditor — Property NLI-1: Score range
    - **Property NLI-1: Score range** — `AuditResult.groundedness_score` must always be in `[0.0, 1.0]` for any valid input
    - _Requirements: 2.3_

  - [ ]* 5.3 Write property test for GroundednessAuditor — Property NLI-4: Scorer exception isolation
    - **Property NLI-4: Scorer exception isolation** — If the cross-encoder raises any exception, `audit()` must return a valid `AuditResult` with `nli_label=None` and must not propagate the exception
    - _Requirements: 2.6_

- [ ] 6. Update TriageGateway with NLI CONTRADICTION hard-block
  - [~] 6.1 Modify `app/triage/gateway.py` to add `nli_label` parameter and Priority 0 CONTRADICTION check
    - Extend `evaluate()` signature with `nli_label: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"] | None = None`
    - Insert Priority 0 check: if `nli_label == "CONTRADICTION"` return `TriageResult(triage_state="HARD_BLOCK", blocking_reason="NLI_CONTRADICTION", response_content=None)` before all existing priority checks
    - Ensure the existing Priority 1–4 logic is unchanged
    - _Requirements: 2.4, 6.1_

  - [ ]* 6.2 Write property test for TriageGateway — Property NLI-2: CONTRADICTION → HARD_BLOCK
    - **Property NLI-2: CONTRADICTION → HARD_BLOCK** — For any call to `evaluate()` where `nli_label="CONTRADICTION"`, the result must have `triage_state="HARD_BLOCK"` and `blocking_reason="NLI_CONTRADICTION"` regardless of the numeric `groundedness_score` value
    - _Requirements: 2.4_

- [ ] 7. Implement GLiNER Masker component
  - [-] 7.1 Create `app/judges/gliner_masker.py` with `GLiNERMasker` class
    - Implement top-of-file optional import guard for `gliner`; set `_HAS_GLINER` flag
    - Implement `scan_sync(prompt, entity_terms, gliner_model)` that calls `gliner_model.predict_entities(text, labels=entity_terms)`, iterates spans in document order, assigns `[CUSTOM_ENTITY_REDACTED_N]` placeholders (N starting at 1), and builds `{placeholder: original}` map
    - Implement `scan(prompt, entity_terms, gliner_model)` as `asyncio.to_thread(self.scan_sync, ...)`
    - Compile `_CUSTOM_PLACEHOLDER_RE = re.compile(r"\[CUSTOM_ENTITY_REDACTED_\d+\]")` at module level
    - _Requirements: 3.5, 3.10, 6.5_

  - [ ]* 7.2 Write property test for GLiNERMasker — Property GL-1: Placeholder format
    - **Property GL-1: Placeholder format** — Every entity span detected by `scan_sync()` must produce a placeholder matching `\[CUSTOM_ENTITY_REDACTED_\d+\]`
    - _Requirements: 3.5_

- [ ] 8. Integrate GLiNER Tier 1.5 into PIIMaskingEngine
  - [~] 8.1 Modify `app/judges/pii_masking.py` to wire in `GLiNERMasker` as Tier 1.5
    - Import `GLiNERMasker` and `_HAS_GLINER` from `app.judges.gliner_masker`
    - Instantiate `self._gliner_masker = GLiNERMasker()` in `PIIMaskingEngine.__init__`
    - Extend `mask()` signature with `custom_entity_terms: list[str] | None = None` and `gliner_model: object | None = None`
    - After Tier 1 (`NLPMasker`), conditionally invoke `self._gliner_masker.scan_sync()` when `_HAS_GLINER and custom_entity_terms and gliner_model is not None`; wrap in `try/except`, log `GLINER_SCAN_ERROR` on exception, and skip GLiNER result without propagating
    - Update `unmask()` to restore `[CUSTOM_ENTITY_REDACTED_N]` placeholders from the per-request `placeholder_map`
    - Add GLiNER round-trip to `run_startup_validation()` using `_GLINER_VALIDATION_PROMPT` and `_GLINER_VALIDATION_TERMS`; on failure emit `GLINER_DEGRADED` alert, set a `_gliner_degraded` flag, and keep `is_healthy=True`
    - Respect `_gliner_degraded` flag to skip GLiNER tier for all subsequent requests when degraded
    - _Requirements: 3.2, 3.3, 3.4, 3.6, 3.7, 3.8, 3.11_

  - [ ]* 8.2 Write property test for PIIMaskingEngine — Property GL-2: Round-trip fidelity
    - **Property GL-2: Round-trip fidelity** — `mask()` followed by `unmask()` on any prompt containing a custom entity term must be byte-for-byte identical to the original prompt after whitespace normalisation
    - _Requirements: 3.6_

  - [ ]* 8.3 Write property test for PIIMaskingEngine — Property GL-3: Empty terms skips tier
    - **Property GL-3: Empty terms → tier skipped** — `mask()` called with `custom_entity_terms=[]` must produce output identical to Tiers 1 + 2 alone, with no `[CUSTOM_ENTITY_REDACTED_N]` placeholders
    - _Requirements: 3.4_

  - [ ]* 8.4 Write property test for PIIMaskingEngine — Property GL-4: Exception isolation
    - **Property GL-4: Exception isolation** — If `scan_sync()` raises any exception, `mask()` must return the Tier-1/Tier-2 result without propagating the exception, and the returned prompt must contain no `[CUSTOM_ENTITY_REDACTED_N]` placeholders
    - _Requirements: 3.8_

- [ ] 9. Implement SlidingWindow token buffer
  - [-] 9.1 Create `app/ingress/sliding_window.py` with `SlidingWindow` class
    - Compile `_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")` at module level
    - Implement `SlidingWindow.__init__` initialising `self._buffer: str = ""`
    - Implement `push(token)` that appends to buffer, calls `_flush()`, and returns `list[str]` of complete sentence-chunks
    - Implement `flush_remaining()` that returns any buffered content as a final chunk and clears the buffer
    - Implement `_flush()` that repeatedly searches for sentence boundaries and splits off complete chunks
    - _Requirements: 4.5, 4.12_

  - [ ]* 9.2 Write property test for SlidingWindow — Property SSE-5: Sentence boundary count
    - **Property SSE-5: SlidingWindow sentence boundary** — For any string containing exactly N sentence-ending punctuation marks (`[.!?]`) each followed by whitespace, the total chunks emitted by `push()` across all tokens plus `flush_remaining()` must equal N (plus 1 if trailing non-sentence content remains)
    - _Requirements: 4.5_

- [ ] 10. Implement SSE Streaming endpoint
  - [~] 10.1 Create `app/ingress/streaming_router.py` with `POST /v1/chat/stream` endpoint
    - Import `APIRouter`, `StreamingResponse`, and `Request` from FastAPI
    - Define `router = APIRouter()` and `@router.post("/v1/chat/stream")` handler returning `StreamingResponse(content=_sse_generator(...), media_type="text/event-stream")`
    - Implement `_sse_generator()` async generator that: runs the full Orchestrator on the complete prompt first; on `HARD_BLOCK` closes without emitting any frames; checks SemanticCache when `cache_enabled=True` and returns cached response as single SSEEvent + `[DONE]` on hit; on cache miss streams tokens from LLM via Portkey
    - Wire `SlidingWindow` into token loop: each assembled chunk runs output validator and `GroundednessAuditor`; on violation emit `data: [REDACTED DUE TO POLICY]\n\n` and close; on clean chunk emit `data: <chunk>\n\n`
    - On end-of-stream emit `data: [DONE]\n\n` and store response in SemanticCache if `cache_enabled=True`
    - Wrap LLM streaming in `try/except`; on exception emit `data: [STREAM_ERROR]\n\n` and close without propagating to ASGI layer
    - Record a `TelemetryRecord` after stream closes with final triage state, total token count, and `mid_stream_violation` flag
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.6, 4.7, 4.8, 4.9, 4.10, 4.11, 6.1_

  - [ ]* 10.2 Write property test for StreamingEndpoint — Property SSE-1: Non-streaming endpoint unchanged
    - **Property SSE-1: Non-streaming endpoint unchanged** — `POST /v1/chat` must return the same `ChatResponse` structure and status codes after the streaming router is added
    - _Requirements: 4.2_

  - [ ]* 10.3 Write property test for StreamingEndpoint — Property SSE-2: HARD_BLOCK before streaming
    - **Property SSE-2: HARD_BLOCK before streaming** — When the Orchestrator returns `upstream_triage_state=HARD_BLOCK`, the SSE response body must contain zero `data:` frames
    - _Requirements: 4.4_

  - [ ]* 10.4 Write property test for StreamingEndpoint — Property SSE-3: Violation severs stream
    - **Property SSE-3: Violation severs stream** — When a chunk fails output validation or groundedness audit mid-stream, the immediately following and only subsequent frame must be `data: [REDACTED DUE TO POLICY]`, with no further frames emitted
    - _Requirements: 4.7_

  - [ ]* 10.5 Write property test for StreamingEndpoint — Property SSE-4: Clean stream ends with DONE
    - **Property SSE-4: Clean stream terminates with DONE** — When all chunks pass policy checks, the final emitted frame must be `data: [DONE]` with no frames emitted after it
    - _Requirements: 4.8_

- [~] 11. Checkpoint — Core components complete
  - Ensure all tests pass for tasks 1–10, ask the user if questions arise.

- [ ] 12. Implement Redteam MCP Server
  - [-] 12.1 Create `mcp_servers/redteam/server.py` as an isolated FastAPI application
    - No imports from any `app.*` module
    - Define `RunRequest` and `RunResponse` Pydantic models
    - Implement `POST /run` endpoint: attempt `pyrit` multi-turn orchestrator then `garak` probe sweep if available; fall back to built-in five-category adversarial prompt library (multi-turn jailbreaks, direct prompt injection, toxicity escalation, PII extraction, competitor-mention injection) if either is absent
    - Implement `GET /health` endpoint returning `{"status": "ok"}` with HTTP 200
    - Implement `GET /report` endpoint returning the most recent `RunResponse` (or empty dict if no run yet)
    - Store the built-in attack library as a module-level constant with no dependency on `app.redteam.runner`
    - _Requirements: 5.1, 5.2, 5.4, 5.5, 5.6, 5.12_

  - [-] 12.2 Create `mcp_servers/redteam/requirements.txt` and `mcp_servers/redteam/mcp.json`
    - `requirements.txt`: `fastapi>=0.111.0`, `uvicorn[standard]>=0.29.0`, `pydantic>=2.7.1`, `httpx>=0.27.0`, `pyrit>=0.4.0`, `garak>=0.9.0`
    - `mcp.json`: specify `command=python`, `args=["mcp_servers/redteam/server.py"]`, `env={"REDTEAM_MCP_PORT": "9200"}`, `port=9200`
    - _Requirements: 5.2, 5.3_

  - [ ]* 12.3 Write property test for RedteamMCPServer — Property RT-2: No `app.*` imports
    - **Property RT-2: No `app.*` imports** — A static AST import scan of `mcp_servers/redteam/server.py` must find zero references to modules whose fully-qualified name starts with `"app."`
    - _Requirements: 5.1_

- [ ] 13. Update RedTeamRunner with MCP-first delegation
  - [~] 13.1 Modify `app/redteam/runner.py` to add MCP-first delegation with in-process fallback
    - Add `MCP_URL: str = "http://localhost:9200"` and `_mcp_healthy: bool | None = None` class attributes
    - Implement `_try_mcp_run(tracer)` that POSTs to `{MCP_URL}/run` with `httpx.AsyncClient(timeout=120.0)`, parses the response via `_parse_mcp_response()`, sets `_mcp_healthy=True` on success, logs `REDTEAM_MCP_UNAVAILABLE` and sets `_mcp_healthy=False` on any exception, and returns `None` on failure
    - Update `run()` to call `await self._try_mcp_run(tracer)` first; use existing in-process logic as fallback when result is `None`
    - In both MCP and in-process paths, record a `RED_TEAM_BREAKTHROUGH` Langfuse span at `ERROR` severity for any `AttackResult` with `breakthrough=True`
    - _Requirements: 5.7, 5.8, 5.10_

  - [ ]* 13.2 Write property test for RedTeamRunner — Property RT-1: MCP fallback
    - **Property RT-1: MCP fallback** — When the MCP server returns a non-2xx response or raises a connection error, `run()` must complete without raising an exception and must return a valid `RedTeamReport`
    - _Requirements: 5.8_

  - [ ]* 13.3 Write property test for RedTeamRunner — Property RT-3: Breakthrough logging
    - **Property RT-3: Breakthrough logging** — For any `AttackResult` with `breakthrough=True`, the Langfuse tracer mock must have been called with `name="RED_TEAM_BREAKTHROUGH"` and `level="ERROR"`, whether the run originated from the MCP server or in-process execution
    - _Requirements: 5.10_

- [ ] 14. Update Kiro redteam hook with MCP health check
  - [~] 14.1 Modify `.kiro/hooks/redteam_trigger.py` to health-check MCP before triggering
    - Add a `GET http://localhost:9200/health` check (timeout 2s) before calling `POST /v1/redteam/run`
    - On health-check failure or exception, log `REDTEAM_MCP_UNAVAILABLE` to stderr
    - Always proceed to call `POST /v1/redteam/run` on the Gateway regardless of MCP health result
    - _Requirements: 5.11_

- [ ] 15. Wire Phase 3 components into `app/main.py` lifespan and pipeline
  - [~] 15.1 Add lifespan Steps 7–10 to `app/main.py`
    - Step 7: Import `SemanticCache` and `_GPTICACHE_AVAILABLE`; instantiate `SemanticCache(similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD)` (or no-op stub) and store in `app.state.semantic_cache`; log `INFO` for degraded path
    - Step 8: Import `_HAS_GLINER`; if available, load `urchade/gliner_medium-v2.1` via `asyncio.to_thread` and store in `app.state.gliner_model`; else store `None` and log `INFO`
    - Step 9: Import `NLIScorer` and `_HAS_SENTENCE_TRANSFORMERS`; if available, instantiate via `asyncio.to_thread` and store in `app.state.nli_scorer`; else store `None` and log `INFO`
    - Step 10: Perform non-blocking health probe to `http://localhost:9200/health`; store result in `app.state.redteam_mcp_healthy`; log `REDTEAM_MCP_UNAVAILABLE` once on failure
    - Mount `streaming_router` by adding `app.include_router(streaming_router)` after existing router registration
    - _Requirements: 1.10, 2.7, 3.9, 5.9, 6.3, 6.4_

  - [~] 15.2 Wire SemanticCache lookup and store into `run_pipeline()` in `app/main.py`
    - Before `route_and_call()`: if `profile.cache_enabled`, call `await app.state.semantic_cache.lookup(ctx.working_prompt, profile.cache_ttl_seconds)`; on hit set `cache_hit=True`, `ctx.llm_response=cache_result.response`, skip `route_and_call()`
    - After triage: if `not cache_hit and profile.cache_enabled` and triage state is `PASS_AND_DELIVER` or `COMPRESS_AND_EDIT`, call `await app.state.semantic_cache.store(...)`
    - Pass `nli_scorer=app.state.nli_scorer` to `GroundednessAuditor.audit()` in the pipeline
    - Pass `nli_label=audit_result.nli_label` to `TriageGateway.evaluate()`
    - Pass `custom_entity_terms=profile.custom_entity_terms, gliner_model=app.state.gliner_model` to `PIIMaskingEngine.mask()`
    - Set `telemetry_record.cache_hit = cache_hit` and `telemetry_record.nli_label = audit_result.nli_label` before writing telemetry
    - _Requirements: 1.3, 1.4, 1.5, 1.6, 1.8, 2.3, 2.4, 3.2, 6.1, 6.5_

  - [ ]* 15.3 Write property test for pipeline wiring — Property SC-1: Cache hit bypasses RouteLLM
    - **Property SC-1: Cache hit bypasses RouteLLM** — For any request that produces a cache hit, `RequestContext.routing_decision` must be `None` (RouteLLM was not called)
    - _Requirements: 1.3, 1.4_

- [~] 16. Final checkpoint — All tests pass
  - Ensure all tests pass across all five feature areas, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Each task references specific requirements by number for full traceability
- Tasks 2, 4, 7, 9, and 12 are independent new components and can be implemented in parallel (Wave 1)
- Tasks 5, 6, 8, and 10 depend on the components from Wave 1 and can be parallelised within Wave 2
- Task 15 (pipeline wiring) must be last, after all components are in place
- The `CACHE_SIMILARITY_THRESHOLD` setting (Task 3) must be added before lifespan wiring (Task 15)
- All CPU-bound operations must use `asyncio.to_thread` — this is enforced in every component
- Property tests use `hypothesis` for generative testing; each property is named and linked to its requirement
- The Redteam MCP server runs in an isolated process with its own venv; do not import `app.*` modules there
- GLiNER startup validation degradation (`GLINER_DEGRADED`) keeps `is_healthy=True` — do not change this behaviour

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "3.1", "4.1", "7.1", "9.1", "12.1", "12.2"] },
    { "id": 2, "tasks": ["2.2", "2.3", "4.2", "5.1", "6.1", "7.2", "8.1", "9.2", "10.1", "12.3", "13.1", "14.1"] },
    { "id": 3, "tasks": ["5.2", "5.3", "6.2", "8.2", "8.3", "8.4", "10.2", "10.3", "10.4", "10.5", "13.2", "13.3"] },
    { "id": 4, "tasks": ["15.1", "15.2"] },
    { "id": 5, "tasks": ["15.3"] }
  ]
}
```
