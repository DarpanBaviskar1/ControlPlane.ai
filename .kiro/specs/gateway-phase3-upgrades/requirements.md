# Requirements Document

## Introduction

This document specifies the Phase 3 upgrades to the ControlPlane.ai Enterprise AI Proxy Gateway. Five features are introduced: a semantic cache injected before LLM inference, an NLI-based groundedness auditor, GLiNER custom-entity PII masking, SSE streaming with per-chunk policy enforcement, and an isolated Redteam MCP server. All upgrades extend the existing five-stage safety pipeline without altering the four-state triage matrix or the non-streaming `/v1/chat` endpoint.

---

## Glossary

- **Gateway**: The ControlPlane.ai FastAPI application that mediates all LLM interactions through the five-stage safety pipeline.
- **SemanticCache**: The GPTCache-backed vector-similarity cache injected into `app/router/model_router.py` before the RouteLLM Controller.
- **CacheEntry**: A stored (masked prompt embedding, cached response, TTL timestamp) triple managed by the SemanticCache.
- **GroundednessAuditor**: The component in `app/groundedness/auditor.py` that scores LLM responses against retrieved documents.
- **NLIScorer**: The cross-encoder/nli-deberta-v3-small cross-encoder used in the two-stage groundedness pipeline.
- **NLILabel**: One of three textual verdicts produced by the NLIScorer: `ENTAILMENT`, `NEUTRAL`, or `CONTRADICTION`.
- **PIIMaskingEngine**: The two-tier (NLPMasker → RegexOnlyMasker) PII masking component in `app/judges/pii_masking.py`.
- **GLiNERMasker**: The optional Tier 1.5 zero-shot named-entity scanner inserted between NLPMasker and RegexOnlyMasker.
- **CustomEntityTerms**: The per-profile list of corporate terms supplied in `UseCaseProfile.custom_entity_terms` used to prime GLiNER.
- **StreamingEndpoint**: The new `POST /v1/chat/stream` endpoint that returns Server-Sent Events.
- **SlidingWindow**: The token-buffering mechanism that accumulates LLM output tokens into sentence-chunks for mid-stream policy evaluation.
- **SSEEvent**: A Server-Sent Event frame emitted to the client by the StreamingEndpoint.
- **RedteamMCPServer**: The isolated FastAPI process at `mcp_servers/redteam/server.py` that runs PyRIT and Garak attack sequences.
- **RedTeamRunner**: The in-process orchestrator in `app/redteam/runner.py` that drives adversarial testing.
- **Orchestrator**: The concurrent P1/P2/P3 judge stage in `app/judges/orchestrator.py`.
- **TriageGateway**: The four-state priority-matrix component in `app/triage/gateway.py`.
- **UseCaseProfile**: The per-profile Pydantic model that configures all pipeline stages.
- **TelemetryRecord**: The structured log record written by the Telemetry Logger for every request.
- **AuditResult**: The dataclass returned by the GroundednessAuditor holding `groundedness_score`, `technique`, `is_unverified`, and `nli_label`.
- **lifespan**: The FastAPI application lifespan context manager responsible for loading heavy models once at startup.
- **asyncio.to_thread**: The standard-library bridge for running CPU-bound synchronous calls off the event loop.

---

## Requirements

### Requirement 1 — Semantic Cache Integration

**User Story:** As a platform operator, I want semantically identical prompts to be served from a cache so that repeated queries incur zero LLM inference cost and sub-2 ms response latency.

#### Acceptance Criteria

1. THE `UseCaseProfile` SHALL include a `cache_enabled` boolean field and a `cache_ttl_seconds` integer field (minimum value 1).

2. THE `TelemetryRecord` SHALL include a `cache_hit` boolean field that is `True` when the response was served from the SemanticCache and `False` otherwise.

3. WHEN `cache_enabled` is `True` and a ChatRequest is received, THE `SemanticCache` SHALL compute a vector embedding of the masked prompt and query the cache index for the nearest stored embedding before the RouteLLM Controller is invoked.

4. WHEN the nearest stored embedding has a cosine similarity at or above the profile-configured similarity threshold, THE `SemanticCache` SHALL return the stored CacheEntry response without invoking the RouteLLM Controller.

5. WHEN a cache hit occurs, THE Gateway SHALL still execute the GroundednessAuditor and TriageGateway stages on the cached response before delivering it to the caller.

6. WHEN a cache miss occurs and the RouteLLM Controller produces a `PASS_AND_DELIVER` or `COMPRESS_AND_EDIT` triage result, THE `SemanticCache` SHALL store the masked prompt embedding and the final response as a new CacheEntry with a TTL equal to `cache_ttl_seconds`.

7. WHEN a CacheEntry's TTL has elapsed, THE `SemanticCache` SHALL not return that entry on subsequent queries.

8. WHEN `cache_enabled` is `False`, THE Gateway SHALL bypass the SemanticCache entirely and invoke the RouteLLM Controller directly.

9. WHEN the SemanticCache index raises any exception, THE Gateway SHALL log a `SEMANTIC_CACHE_ERROR` event, treat the result as a cache miss, and continue to the RouteLLM Controller without propagating the exception.

10. THE `SemanticCache` SHALL be initialised once in the `lifespan` context manager and stored in `app.state`.

11. WHEN `gpicache` is not installed, THE Gateway SHALL start without the SemanticCache and treat all requests as cache misses with `cache_hit=False`.

12. THE `SemanticCache` SHALL perform all embedding and index operations via `asyncio.to_thread` so that the event loop is not blocked.

13. WHEN a cache hit is served, THE Gateway SHALL record a latency of no more than 2 ms for the cache lookup step alone (excluding groundedness and triage stages) when measured under nominal load.

---

### Requirement 2 — NLI-Based Groundedness Auditor

**User Story:** As a safety engineer, I want the groundedness auditor to detect contradictory LLM responses using natural language inference so that factually inconsistent outputs are hard-blocked before delivery.

#### Acceptance Criteria

1. THE `AuditResult` dataclass SHALL include an `nli_label` field of type `Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"] | None`.

2. THE `TelemetryRecord` SHALL include an `nli_label` field of type `Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"] | None`.

3. WHEN `sentence-transformers` is installed, THE `GroundednessAuditor` SHALL execute a two-stage pipeline: first FAISS retrieval of the top-K most relevant documents, then NLIScorer cross-encoder inference over each retrieved (document, response) pair, producing an aggregate `nli_label`.

4. WHEN the NLIScorer produces a `CONTRADICTION` label for an audit result, THE `TriageGateway` SHALL return `triage_state=HARD_BLOCK` with `blocking_reason="NLI_CONTRADICTION"` independently of the numeric `groundedness_score`.

5. WHEN `sentence-transformers` is not installed, THE `GroundednessAuditor` SHALL execute the cosine-similarity-only pipeline and set `nli_label=None` on the returned `AuditResult`.

6. WHEN the NLIScorer raises any exception during scoring, THE `GroundednessAuditor` SHALL set `nli_label=None`, log a `NLI_SCORER_ERROR` event, and return the cosine-only score without propagating the exception.

7. THE `NLIScorer` model (`cross-encoder/nli-deberta-v3-small`) SHALL be loaded once in the `lifespan` context manager and stored in `app.state`; it SHALL NOT be loaded per-request.

8. THE `GroundednessAuditor` SHALL execute NLIScorer inference via `asyncio.to_thread` so that the event loop is not blocked.

9. WHEN `sentence-transformers` is installed and the NLI pipeline produces multiple (document, response) pair scores, THE `GroundednessAuditor` SHALL derive the aggregate `nli_label` as `CONTRADICTION` if any pair scores `CONTRADICTION`, otherwise `ENTAILMENT` if any pair scores `ENTAILMENT`, otherwise `NEUTRAL`.

10. THE `AuditResult` `technique` field SHALL be set to `"nli_embedding_similarity"` when the NLI pipeline runs and `"embedding_similarity"` when falling back to cosine-only.

---

### Requirement 3 — GLiNER Custom Entity Masking

**User Story:** As a data-protection officer, I want the PII masking engine to redact domain-specific corporate entities in addition to standard PII so that sensitive business terms are never transmitted to external LLM APIs.

#### Acceptance Criteria

1. THE `UseCaseProfile` SHALL include a `custom_entity_terms` field of type `list[str]` with a default value of an empty list.

2. WHEN `gliner` is installed and `custom_entity_terms` is non-empty, THE `PIIMaskingEngine` SHALL invoke the `GLiNERMasker` as Tier 1.5 after the NLPMasker and before the RegexOnlyMasker in the scanning sequence.

3. WHEN `gliner` is not installed, THE `PIIMaskingEngine` SHALL skip the GLiNERMasker tier entirely and proceed directly from NLPMasker to RegexOnlyMasker without raising an exception.

4. WHEN `custom_entity_terms` is empty, THE `PIIMaskingEngine` SHALL skip the GLiNERMasker tier regardless of whether `gliner` is installed.

5. THE `GLiNERMasker` SHALL produce placeholders of the form `[CUSTOM_ENTITY_REDACTED_N]` where N is a sequential integer starting at 1 for each detected custom entity in document order.

6. THE `PIIMaskingEngine` `unmask` method SHALL restore all `[CUSTOM_ENTITY_REDACTED_N]` placeholders to their original values using the per-request `placeholder_map`.

7. THE `PIIMaskingEngine` `run_startup_validation` method SHALL include at least one round-trip validation prompt that exercises the GLiNERMasker tier when `gliner` is installed.

8. WHEN the GLiNERMasker raises any exception during scanning, THE `PIIMaskingEngine` SHALL log a `GLINER_SCAN_ERROR` event, skip the GLiNERMasker result for that request, and continue to the RegexOnlyMasker without propagating the exception.

9. THE GLiNER model SHALL be loaded once in the `lifespan` context manager and stored in `app.state`; it SHALL NOT be loaded per-request.

10. THE `GLiNERMasker` SHALL execute all GLiNER inference calls via `asyncio.to_thread` so that the event loop is not blocked.

11. WHEN `gliner` is installed and the GLiNER tier fails startup validation, THE `PIIMaskingEngine` SHALL emit a `GLINER_DEGRADED` alert, skip the GLiNERMasker tier for all subsequent requests, and retain `is_healthy=True`.

---

### Requirement 4 — SSE Streaming Endpoint with Sliding-Window Triage

**User Story:** As an application developer, I want a streaming chat endpoint that enforces output policy on each sentence chunk so that policy violations are caught and severed mid-stream rather than delivered in full.

#### Acceptance Criteria

1. THE Gateway SHALL expose a `POST /v1/chat/stream` endpoint that returns a `StreamingResponse` with `Content-Type: text/event-stream`.

2. THE existing `POST /v1/chat` endpoint SHALL remain unchanged and fully functional when the StreamingEndpoint is added.

3. WHEN a streaming request is received, THE Gateway SHALL execute the full Orchestrator (P1, P2, P3 judges) on the complete prompt before any LLM tokens are streamed to the caller.

4. WHEN the Orchestrator returns `upstream_triage_state=HARD_BLOCK` on a streaming request, THE Gateway SHALL close the SSE connection immediately without streaming any LLM tokens.

5. WHEN LLM tokens are received by the StreamingEndpoint, THE `SlidingWindow` SHALL buffer tokens until a sentence boundary is detected (period, exclamation mark, or question mark followed by whitespace or end-of-stream), then emit the accumulated sentence-chunk as an SSEEvent.

6. WHEN a sentence-chunk is assembled, THE StreamingEndpoint SHALL run the output validator and GroundednessAuditor asynchronously on that chunk before emitting the corresponding SSEEvent to the client.

7. WHEN the output validator or GroundednessAuditor detects a policy violation on a sentence-chunk mid-stream, THE StreamingEndpoint SHALL immediately stop emitting further SSEEvents, emit a final SSEEvent with data `[REDACTED DUE TO POLICY]`, and close the SSE connection.

8. WHEN all sentence-chunks have been emitted without a policy violation, THE StreamingEndpoint SHALL emit a final SSEEvent with data `[DONE]` to signal stream completion.

9. WHEN the LLM inference call raises an exception mid-stream, THE StreamingEndpoint SHALL emit a final SSEEvent with data `[STREAM_ERROR]` and close the connection without propagating the exception to the ASGI layer.

10. THE StreamingEndpoint SHALL record a `TelemetryRecord` after the stream closes, reflecting the final triage state, total token count streamed, and whether a mid-stream policy violation occurred.

11. WHEN a streaming request arrives and `cache_enabled` is `True`, THE Gateway SHALL check the SemanticCache before beginning LLM streaming; a cache hit SHALL return the cached response as a single SSEEvent followed by `[DONE]`.

12. THE `SlidingWindow` chunk assembly and policy-check scheduling SHALL execute via `asyncio.to_thread` for any CPU-bound operations so that the event loop is not blocked during token buffering.

---

### Requirement 5 — Redteam MCP Server

**User Story:** As a security engineer, I want an isolated Redteam MCP server that runs PyRIT and Garak in their own dependency environment so that adversarial testing tools never conflict with the production gateway's dependency tree.

#### Acceptance Criteria

1. THE `RedteamMCPServer` SHALL be implemented in `mcp_servers/redteam/server.py` with no imports from any `app.*` module.

2. THE `RedteamMCPServer` SHALL have its own `mcp_servers/redteam/requirements.txt` that lists PyRIT and Garak as dependencies.

3. THE `RedteamMCPServer` SHALL have its own `mcp_servers/redteam/mcp.json` configuration file that specifies the server command, arguments, and port (default `9200`).

4. THE `RedteamMCPServer` SHALL expose a `POST /run` endpoint that accepts a `session_id` and returns a red-team report JSON object.

5. THE `RedteamMCPServer` SHALL expose a `GET /health` endpoint that returns HTTP 200 when the server is reachable.

6. THE `RedteamMCPServer` SHALL expose a `GET /report` endpoint that returns the most recent red-team report.

7. WHEN the `RedTeamRunner` executes a red-team run, THE `RedTeamRunner` SHALL first attempt to delegate to the `RedteamMCPServer` via HTTP POST to `http://localhost:9200/run`.

8. WHEN the `RedteamMCPServer` is unreachable or returns a non-2xx response, THE `RedTeamRunner` SHALL fall back to in-process PyRIT and built-in adversarial prompt execution without raising an exception to the caller.

9. WHEN the `RedteamMCPServer` health check fails at startup, THE `RedTeamRunner` SHALL log a `REDTEAM_MCP_UNAVAILABLE` event once and proceed with in-process execution.

10. WHEN a red-team attack sequence produces a breakthrough, THE `RedTeamRunner` SHALL record a `RED_TEAM_BREAKTHROUGH` span at `ERROR` severity via the Langfuse tracer, whether the run originated from the MCP server or in-process execution.

11. THE `.kiro/hooks/redteam_trigger.py` hook SHALL check the `RedteamMCPServer` health endpoint at `http://localhost:9200/health` before calling `POST /v1/redteam/run`; IF the health check fails, THEN THE hook SHALL log a `REDTEAM_MCP_UNAVAILABLE` warning and proceed to call `POST /v1/redteam/run` on the Gateway directly.

12. THE `RedteamMCPServer` SHALL contain its own copy of the built-in adversarial prompt library covering all five attack categories (multi-turn jailbreaks, direct prompt injection, toxicity escalation, PII extraction, competitor-mention injection) with no dependency on `app.redteam.runner`.

---

### Requirement 6 — Cross-Cutting Constraints

**User Story:** As a platform engineer, I want all Phase 3 components to respect the existing performance budget, degradation patterns, and pipeline invariants so that the gateway remains reliable under partial dependency availability.

#### Acceptance Criteria

1. THE Gateway SHALL preserve the four-state triage matrix (`HARD_BLOCK`, `ESCALATE_TO_HUMAN`, `COMPRESS_AND_EDIT`, `PASS_AND_DELIVER`) for all requests, including those served from the SemanticCache and those processed by the StreamingEndpoint.

2. THE P1 Judge wall-clock time SHALL remain at or below 150 ms, THE P3 Judge wall-clock time SHALL remain at or below 50 ms, and THE full Orchestrator wall-clock time SHALL remain at or below 200 ms after all Phase 3 changes are applied.

3. WHEN any Phase 3 optional dependency (`gpicache`, `sentence-transformers`, `gliner`) is not installed, THE Gateway SHALL start successfully and operate with the degraded code path for that component, logging a single `INFO`-level message identifying which tier is inactive.

4. THE `lifespan` context manager SHALL load all heavy Phase 3 models (SemanticCache index, NLIScorer, GLiNER model) once at startup and store each in `app.state`; no Phase 3 model SHALL be loaded on the first request or per-request.

5. ALL CPU-bound synchronous operations introduced by Phase 3 (cache embedding, NLI inference, GLiNER scanning, Garak probe execution) SHALL be executed via `asyncio.to_thread` so that the FastAPI event loop is not blocked.

6. THE `TelemetryRecord` SHALL be backward-compatible: the new `cache_hit` and `nli_label` fields SHALL default to `False` and `None` respectively so that existing telemetry consumers require no schema migration.
