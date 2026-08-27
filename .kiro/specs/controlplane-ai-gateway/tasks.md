# Implementation Plan: ControlPlane.ai Enterprise AI Proxy Gateway

## Overview

Build a Python/FastAPI enterprise AI proxy gateway that mediates every interaction between
enterprise applications and the underlying LLMs through a deterministic five-stage pipeline:
Enterprise Ingress → Streaming Micro-Judges → Intelligent Model Router → Groundedness Auditor →
Action Triage Gateway. Cross-cutting concerns (Telemetry Logger, Policy Layer, PII Masking Engine,
Feedback Loop) are wired across all stages. The implementation follows the directory structure
defined in the design document (`app/{ingress,policy,judges,router,groundedness,triage,telemetry,feedback}`).

---

## Tasks

- [ ] 1. Project scaffold and shared data models
  - Create the full `app/` directory structure as defined in the design:
    `app/main.py`, `app/dependencies.py`, and subdirectories `ingress/`, `policy/`, `judges/`,
    `router/`, `groundedness/`, `triage/`, `telemetry/`, `feedback/` each with an `__init__.py`.
  - Create `app/models.py` (or per-module `models.py` files) implementing all Pydantic data models:
    `ChatRequest`, `ChatResponse`, `UseCaseProfile`, `RequestContext`, `TelemetryRecord`,
    `OverrideRecord`, and the error envelope schema.
  - Add `requirements.txt` (or `pyproject.toml`) pinning FastAPI, uvicorn, pydantic, llm-guard,
    routellm, portkey-ai, faiss-cpu, spacy, tiktoken, hypothesis, pytest, pytest-asyncio,
    watchdog, and httpx.
  - Create `pytest.ini` (or `pyproject.toml` `[tool.pytest]` section) configuring `asyncio_mode = auto`
    and marking `integration` tests.
  - _Requirements: 1.1, 7.2 (model field constraints must be encoded in Pydantic validators)_

  - [-] 1.1 Implement `ChatRequest`, `ChatResponse`, `UseCaseProfile`, and `RequestContext` models
    - Enforce all field-level constraints with Pydantic `Field` validators (lengths, ranges) as
      specified in the design Data Models section.
    - Implement the two compiled default profiles (`customer_chatbot`, `internal_copilot`) as
      `UseCaseProfile` instances in `app/policy/defaults.py`.
    - _Requirements: 1.1, 1.5, 7.2_

  - [x] 1.2 Implement `TelemetryRecord` and `OverrideRecord` models with all required fields
    - Ensure every field listed in the design `TelemetryRecord` schema is present, typed, and
      nullable where appropriate.
    - _Requirements: 6.1, 6.3, 6.4, 6.5, 8.1, 8.3_

  - [~] 1.3 Write property test for `UseCaseProfile` field validation
    - **Property 22: Policy Validation Rejects Invalid Fields**
    - Generate `UseCaseProfile` dicts with one field set out-of-range (e.g., `latency_budget_ms=-1`,
      `complexity_threshold=2.5`) and assert Pydantic raises `ValidationError` identifying the
      field name and provided value.
    - **Validates: Requirement 7.5**
    - _# Feature: controlplane-ai-gateway, Property 22: Policy Validation Rejects Invalid Fields_

- [ ] 2. Policy Layer — hot-reload config loader
  - Implement `app/policy/loader.py` with the `PolicyLoader` class.
  - On startup: merge the two compiled-in default profiles with any user-supplied YAML/JSON file.
  - Use `watchdog` to detect file modifications; on change, parse into a candidate `dict[str, UseCaseProfile]`,
    validate every profile, and only atomically swap if all pass — otherwise keep previous config and log error.
  - Protect the live config reference with an `asyncio.Lock`.
  - Expose `get_profile(name)`, `reload()`, and `list_profiles()`.
  - _Requirements: 1.2, 1.4, 1.5, 7.1, 7.2, 7.5, 7.6_

  - [~] 2.1 Implement `PolicyLoader` with watchdog hot-reload and atomic swap
    - Include startup bootstrap of the two built-in profiles.
    - Validate that `reload()` completes and updates the live config within 5 seconds of file modification.
    - _Requirements: 7.1, 7.5_

  - [~] 2.2 Implement `get_profile()` returning a `UseCaseProfile` or raising for unknown names
    - Return 422-compatible error information when profile name is absent from the loaded config.
    - _Requirements: 1.2, 1.4_

  - [~] 2.3 Write property test for profile load correctness
    - **Property 1: Profile Load Correctness**
    - For any valid profile name present in the loader, assert `get_profile(name).name == name` and
      all fields satisfy type/range constraints.
    - **Validates: Requirements 1.2, 7.2**
    - _# Feature: controlplane-ai-gateway, Property 1: Profile Load Correctness_

- [ ] 3. Enterprise Ingress — `/v1/chat` endpoint and request lifecycle
  - Implement `app/ingress/router.py` with the `POST /v1/chat` handler.
  - Assign a UUID v4 `request_id` via `app/dependencies.py`.
  - Inject `UseCaseProfile` via FastAPI dependency (`PolicyLoader.get_profile`); return 422 on
    unknown profile or invalid prompt.
  - Wrap the full downstream pipeline coroutine in `asyncio.wait_for(pipeline_coro, timeout=profile.latency_budget_ms / 1000)`;
    catch `asyncio.TimeoutError` and return HTTP 504 with elapsed duration.
  - Create a fresh `RequestContext` per request (no shared mutable state across requests).
  - In the `finally` block, call `PIIMaskingEngine.discard_mapping(request_id)`.
  - Mount the ingress router in `app/main.py`.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 9.1_

  - [~] 3.1 Implement request validation and 422 error responses
    - Return HTTP 422 with the standard error envelope for missing/empty `prompt` and for
      absent/unrecognised `use_case_profile`.
    - _Requirements: 1.3, 1.4_

  - [~] 3.2 Implement UUID v4 `request_id` assignment and latency-budget timeout enforcement
    - Wrap the pipeline in `asyncio.wait_for`; return HTTP 504 on timeout.
    - _Requirements: 1.6, 6.3_

  - [~] 3.3 Implement `RequestContext` creation and per-request state isolation
    - Confirm no app-level mutable state is shared; each request gets its own `RequestContext` instance.
    - _Requirements: 1.7_

  - [~] 3.4 Write property test for invalid request rejection
    - **Property 2: Invalid Request Rejection**
    - For any prompt that is absent, empty, or whitespace-only, assert HTTP 422 response.
    - For any `use_case_profile` not in the Policy Layer, assert HTTP 422 response.
    - **Validates: Requirements 1.3, 1.4**
    - _# Feature: controlplane-ai-gateway, Property 2: Invalid Request Rejection_

  - [~] 3.5 Write property test for latency budget enforcement
    - **Property 3: Latency Budget Enforcement**
    - Mock the downstream pipeline to sleep longer than `latency_budget_ms`; assert HTTP 504 and
      that elapsed time is within 200 ms of the budget.
    - **Validates: Requirement 1.6**
    - _# Feature: controlplane-ai-gateway, Property 3: Latency Budget Enforcement_

  - [~] 3.6 Write property test for concurrent request isolation
    - **Property 4: Concurrent Request Isolation**
    - Send two concurrent requests with distinct profiles; assert each receives its own profile
      config and no state leaks between `RequestContext` instances.
    - **Validates: Requirement 1.7**
    - _# Feature: controlplane-ai-gateway, Property 4: Concurrent Request Isolation_

- [~] 4. Checkpoint — ingress, policy, and data models
  - Ensure all unit tests for Tasks 1–3 pass, project scaffold installs cleanly, and the
    `/v1/chat` endpoint returns correct 422 and 504 responses. Ask the user if questions arise.

- [ ] 5. PII Masking Engine with startup validation suite
  - Implement `app/judges/pii_masking.py` with `PIIMaskingEngine`.
  - `mask(prompt, request_id)`: calls the `llm_guard.input_scanners.Anonymize` scanner (offloaded
    via `asyncio.to_thread`), stores the `placeholder_map` in an internal per-request dict keyed
    by `request_id`, returns `(masked_prompt, placeholder_map)`.
  - `unmask(masked_prompt, request_id)`: restores all placeholders using the stored mapping.
  - `discard_mapping(request_id)`: removes the per-request mapping after response delivery.
  - `run_startup_validation()`: runs the five synthetic PII prompts through mask → unmask;
    returns `True` if all pass byte-for-byte identity after whitespace normalisation
    (`re.sub(r'\s+', ' ', s).strip()`); sets `MASKING_INTEGRITY_FAILURE` flag and returns `False`
    otherwise. Log a `MASKING_INTEGRITY_FAILURE` event on failure.
  - Wire `run_startup_validation()` into `app/main.py` FastAPI `lifespan`; if it returns `False`,
    the `/v1/chat` handler must return HTTP 503 until a subsequent validation pass succeeds.
  - _Requirements: 2.4, 2.5, 2.6, 7.3, 9.1, 9.2, 9.4, 9.5_

  - [~] 5.1 Implement `PIIMaskingEngine.mask()` and `discard_mapping()`
    - Use `Anonymize` scanner; store per-request `placeholder_map`; confirm map is discarded in
      the ingress `finally` block.
    - _Requirements: 2.5, 9.1_

  - [~] 5.2 Implement `PIIMaskingEngine.unmask()` with byte-for-byte fidelity check
    - After whitespace normalisation, result must equal original prompt.
    - _Requirements: 9.2_

  - [~] 5.3 Implement `run_startup_validation()` and lifespan integration
    - Five synthetic prompts; set 503 gate flag on failure; clear flag when subsequent pass succeeds.
    - _Requirements: 9.4, 9.5_

  - [~] 5.4 Write property test for PII masking round-trip fidelity
    - **Property 16: PII Masking Round-Trip Fidelity**
    - For any prompt containing detectable PII tokens, assert `unmask(mask(prompt, rid), rid)` is
      byte-for-byte identical to the original after whitespace normalisation.
    - **Validates: Requirement 9.2**
    - _# Feature: controlplane-ai-gateway, Property 16: PII Masking Round-Trip Fidelity_

  - [~] 5.5 Write property test for P2 PII token replacement
    - **Property 6: P2 PII Masking Replaces Tokens**
    - For prompts containing SSNs, emails, phone numbers, or full names with `pii_masking_enabled=true`,
      assert the masked prompt contains no original PII substring and each token is replaced with a
      `[TYPE_REDACTED]` placeholder.
    - **Validates: Requirement 2.5**
    - _# Feature: controlplane-ai-gateway, Property 6: P2 PII Masking Replaces Tokens_

- [ ] 6. Micro-Judge stage — P1, P2, P3 judges and orchestrator
  - Implement `app/judges/p1_judge.py`, `app/judges/p2_judge.py`, `app/judges/p3_judge.py`, and
    `app/judges/orchestrator.py`.
  - Load all LLM Guard scanner instances (`Toxicity`, `PromptInjection`, `Anonymize`) once at
    startup in the FastAPI `lifespan`; offload per-request scanner calls to `asyncio.to_thread`.
  - Load the spaCy `en_core_web_sm` model once at startup for P3.
  - Orchestrator: run P1, P2, P3 concurrently via `asyncio.gather` inside
    `asyncio.wait_for(..., timeout=profile.inspection_timeout_ms / 1000)`.
  - On `asyncio.TimeoutError`, set `triage_state=ESCALATE_TO_HUMAN` and log a timeout event.
  - Judge failure isolation: P1 exception → `BLOCK`; P2 exception → `pii_count=sys.maxsize`
    (triggers masking); P3 exception → `AMBIGUOUS`. Record error event in Telemetry Logger.
  - After P1 BLOCK: immediately set `triage_state=HARD_BLOCK`, cancel in-flight siblings,
    skip all downstream stages.
  - After P2 with `pii_count > 0` and `pii_masking_enabled=true`: call `PIIMaskingEngine.mask()`;
    if masking fails, set `triage_state=HARD_BLOCK`.
  - After P2 with `pii_masking_enabled=false`: log `PII_MASKING_BYPASSED` event to Telemetry Logger.
  - Record P1 toxicity verdict, P1 injection verdict, P2 PII count, P3 clarity verdict in Telemetry
    Logger for every request regardless of outcome.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 7.3_

  - [~] 6.1 Implement `p1_judge.py` — toxicity and injection verdicts
    - Use `Toxicity` and `PromptInjection` scanners; map `is_valid=False` to `BLOCK`.
    - _Requirements: 2.2_

  - [~] 6.2 Implement `p2_judge.py` — PII detection and masking integration
    - Use `Anonymize` scanner; return `P2Verdict` with `pii_count`, `masked_prompt`, `placeholder_map`.
    - _Requirements: 2.4, 2.5_

  - [~] 6.3 Implement `p3_judge.py` — clarity classifier using tiktoken and spaCy
    - `AMBIGUOUS` when token count ≤ 10 OR no ROOT-tagged VERB found in the dependency parse.
    - _Requirements: 2.7_

  - [~] 6.4 Implement `orchestrator.py` — concurrent judge execution with timeout and failure isolation
    - `asyncio.gather` with `asyncio.wait_for`; implement per-judge exception handling returning
      the most restrictive safe defaults; handle inspection timeout → `ESCALATE_TO_HUMAN`.
    - _Requirements: 2.1, 2.3, 2.8, 2.9_

  - [~] 6.5 Wire P1 BLOCK → pipeline short-circuit and P2 masking → `HARD_BLOCK` on masking failure
    - After orchestrator returns, check verdicts in order: P1 BLOCK first, then P2 masking failure.
    - _Requirements: 2.3, 2.6_

  - [~] 6.6 Write property test for P3 clarity classification rule
    - **Property 7: P3 Clarity Classification Rule**
    - For any prompt where token count ≤ 10 OR spaCy parse produces no ROOT VERB, assert
      `p3_judge(prompt) == "AMBIGUOUS"`; otherwise assert `"CLEAR"`.
    - **Validates: Requirement 2.7**
    - _# Feature: controlplane-ai-gateway, Property 7: P3 Clarity Classification Rule_

  - [~] 6.7 Write property test for P1 BLOCK halting the pipeline
    - **Property 5: P1 Block Halts Pipeline**
    - For any prompt where P1 returns BLOCK, assert `ChatResponse.triage_state == "HARD_BLOCK"` and
      no calls were made to the Model Router, LLM, Groundedness Auditor, or external services.
    - **Validates: Requirement 2.3**
    - _# Feature: controlplane-ai-gateway, Property 5: P1 Block Halts Pipeline_

  - [~] 6.8 Write property test for Micro-Judge stage telemetry completeness
    - **Property 8: Micro-Judge Stage Telemetry Completeness**
    - For any request completing the Micro-Judge stage, assert `TelemetryRecord` has non-null
      `p1_toxicity_verdict`, `p1_injection_verdict`, `p2_pii_count`, and `p3_clarity_verdict`.
    - **Validates: Requirement 2.10**
    - _# Feature: controlplane-ai-gateway, Property 8: Micro-Judge Stage Telemetry Completeness_

- [~] 7. Checkpoint — Micro-Judge stage
  - Ensure all judge unit tests pass (edge cases: exactly 10 tokens, 11 tokens, no verb, multiple verbs;
    P1 BLOCK path; P2 masking-disabled path; orchestrator timeout path). Ask the user if questions arise.

- [x] 8. Intelligent Model Router — RouteLLM + Portkey dispatch
  - Implement `app/router/model_router.py`.
  - Instantiate the RouteLLM `Controller` once at startup with `routers=["mf"]` and Portkey virtual
    keys for both tiers.
  - Per request: embed per-profile `complexity_threshold` into the model string
    `router-mf-{threshold}`; if `p3_clarity == AMBIGUOUS`, prepend a system-message bias prefix
    to nudge the router toward the Frontier Model.
  - Classify `ROUTINE` when RouteLLM score < threshold; `COMPLEX` when ≥ threshold (default 0.7
    when not set in profile).
  - Build Portkey `FRONTIER_CONFIG` and `SLM_CONFIG` objects with fallback strategy.
  - Fallback logic: if primary tier fails, try alternative tier once via Portkey; if both fail,
    set `triage_state=HARD_BLOCK` and log `MODEL_TIER_FAILURE` event.
  - Return `RoutingDecision` dataclass.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 8.1 Implement `route_and_call()` with RouteLLM complexity classification
    - Threshold embedding in model string; default 0.7 fallback; return `RoutingDecision`.
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 8.2 Implement Portkey dispatch with fallback config for both model tiers
    - FRONTIER_CONFIG and SLM_CONFIG with `strategy.mode = "fallback"`; dual-tier retry logic.
    - _Requirements: 3.7_

  - [x] 8.3 Implement fallback on both-tiers-failure → `HARD_BLOCK` + `MODEL_TIER_FAILURE` log
    - Catch Portkey exhausted-retry exception; set `triage_state = HARD_BLOCK`.
    - _Requirements: 3.7_

  - [x] 8.4 Write property test for routing classification determinism
    - **Property 9: Routing Classification is Deterministic and Binary**
    - For any `(score, threshold)` pair, assert `ROUTINE` when `score < threshold` and `COMPLEX`
      when `score >= threshold`; assert output is always exactly one of the two values.
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    - _# Feature: controlplane-ai-gateway, Property 9: Routing Classification is Deterministic and Binary_

  - [x] 8.5 Write property test for router telemetry completeness
    - **Property 10: Router Telemetry Completeness**
    - For any request processed by the Model Router, assert `TelemetryRecord` has non-null
      `routing_decision`, `selected_model_tier`, and `routellm_score`.
    - **Validates: Requirement 3.6**
    - _# Feature: controlplane-ai-gateway, Property 10: Router Telemetry Completeness_

- [x] 9. Groundedness Auditor — embedding similarity vs FAISS vector store
  - Implement `app/groundedness/auditor.py` and `app/groundedness/vector_store.py`.
  - Define the `VectorStore` Protocol with `async similarity_search(embedding, top_k) -> list[Document]`.
  - Implement the FAISS adapter (`FAISSVectorStore`) and the pgvector stub adapter.
  - `audit(response, request_id)`: embed the response, retrieve top-K docs from the vector store,
    compute mean cosine similarity, normalise to [0.0, 1.0]; return `AuditResult`.
  - Streaming support: start embedding computation as a background `asyncio.Task` after the first
    token arrives; emit initial score within 500 ms.
  - Unavailability handling: if the vector store raises any exception, set `groundedness_score=0.0`,
    mark `is_unverified=True`, log `VECTOR_STORE_UNAVAILABLE` event.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 9.1 Implement `VectorStore` Protocol and FAISS adapter
    - `FAISSVectorStore.similarity_search()` returns a list of `Document` objects.
    - _Requirements: 4.1, 4.3_

  - [x] 9.2 Implement `audit()` with embedding similarity scorer and [0.0, 1.0] normalisation
    - Return `AuditResult`; include initial-score-within-500ms logic for streaming.
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 9.3 Implement vector store unavailability fallback and `VECTOR_STORE_UNAVAILABLE` logging
    - Catch all exceptions from the store; return `groundedness_score=0.0`, `is_unverified=True`.
    - _Requirements: 4.5_

  - [x] 9.4 Write property test for groundedness score range
    - **Property 11: Groundedness Score is In-Range**
    - For any LLM response string (including empty strings, very long strings, special characters),
      assert `AuditResult.groundedness_score` is in the closed interval `[0.0, 1.0]`.
    - **Validates: Requirement 4.1**
    - _# Feature: controlplane-ai-gateway, Property 11: Groundedness Score is In-Range_

  - [x] 9.5 Write property test for low-groundedness signal emission
    - **Property 12: Low-Groundedness Signal Emission**
    - For any `AuditResult` where `groundedness_score < 0.5`, assert the low-groundedness signal
      (score + technique) is emitted to the Triage Gateway before response content is delivered.
    - **Validates: Requirement 4.6**
    - _# Feature: controlplane-ai-gateway, Property 12: Low-Groundedness Signal Emission_

- [x] 10. Action Triage Gateway — four-state decision matrix
  - Implement `app/triage/gateway.py` with the `evaluate()` function.
  - Apply decision rules in strict priority order: HARD_BLOCK → ESCALATE_TO_HUMAN →
    COMPRESS_AND_EDIT → PASS_AND_DELIVER.
  - HARD_BLOCK conditions: upstream `triage_state == HARD_BLOCK` OR `groundedness_score < 0.5`.
  - ESCALATE_TO_HUMAN conditions: `0.5 <= groundedness_score <= profile.groundedness_pass_threshold`
    OR `p3_clarity == AMBIGUOUS` (only when `human_escalation_enabled=true`).
  - When `human_escalation_enabled=false`, promote ESCALATE_TO_HUMAN → HARD_BLOCK before returning.
  - COMPRESS_AND_EDIT: `groundedness_score > profile.groundedness_pass_threshold` AND
    `response_token_count > profile.token_compression_threshold`; send summarisation prompt to SLM
    tier via Portkey; validate no new named entities appear in the compressed output.
  - PASS_AND_DELIVER: all other cases (score above threshold, token count within budget, no upstream block).
  - Use `profile.groundedness_pass_threshold` in place of the system default of 0.9 when set.
  - HARD_BLOCK response body: `{triage_state, blocking_reason, response: null}` with HTTP 200.
  - Log full prompt + response to Telemetry Logger for ESCALATE_TO_HUMAN.
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 7.4, 7.7_

  - [x] 10.1 Implement `evaluate()` with the four-state priority matrix
    - All four state branches; priority ordering; upstream triage state preservation.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

  - [x] 10.2 Implement `COMPRESS_AND_EDIT` compressor with named-entity containment check
    - Summarisation prompt to SLM via Portkey; spaCy NER check that no new entities appear.
    - _Requirements: 5.3, 9.3_

  - [x] 10.3 Implement `human_escalation_enabled=false` promotion and `groundedness_pass_threshold` override
    - When flag is false, promote ESCALATE_TO_HUMAN to HARD_BLOCK.
    - Use `profile.groundedness_pass_threshold` for the PASS_AND_DELIVER threshold.
    - _Requirements: 7.4, 7.7_

  - [x] 10.4 Write property test for triage decision matrix completeness and priority
    - **Property 13: Triage Decision Matrix Completeness and Priority**
    - For any combination of `groundedness_score`, `response_token_count`, `upstream_triage_state`,
      `p3_clarity`, and `UseCaseProfile`, assert exactly one `TriageState` is returned; assert
      HARD_BLOCK whenever upstream is HARD_BLOCK; assert HARD_BLOCK whenever score < 0.5; assert
      PASS_AND_DELIVER is never returned when any higher-priority condition holds.
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8**
    - _# Feature: controlplane-ai-gateway, Property 13: Triage Decision Matrix Completeness and Priority_

  - [x] 10.5 Write property test for `human_escalation_enabled=false` promotion
    - **Property 14: `human_escalation_enabled=false` Promotion**
    - For any request/profile where `human_escalation_enabled=false` and conditions would produce
      ESCALATE_TO_HUMAN, assert the returned `triage_state` is `HARD_BLOCK`.
    - **Validates: Requirement 7.4**
    - _# Feature: controlplane-ai-gateway, Property 14: human_escalation_enabled=false Promotion_

  - [x] 10.6 Write property test for custom groundedness threshold
    - **Property 15: Custom Groundedness Threshold Respected**
    - For any profile with `groundedness_pass_threshold=T`, a response with score `S` where
      `T < S <= 0.9` and no other blocking condition should be assigned `PASS_AND_DELIVER`.
    - **Validates: Requirement 7.7**
    - _# Feature: controlplane-ai-gateway, Property 15: Custom Groundedness Threshold Respected_

- [x] 11. Checkpoint — Router, Auditor, and Triage Gateway
  - Ensure all triage matrix unit tests pass (all boundary combinations for score, token count,
    profile flags, upstream state). Ask the user if questions arise.

- [x] 12. Telemetry Logger — async queue writer and metrics aggregator
  - Implement `app/telemetry/logger.py`, `app/telemetry/models.py`, and `app/telemetry/aggregator.py`.
  - `TelemetryLogger` as a singleton with an `asyncio.Queue`; a background consumer task drains
    the queue and writes structured JSON records to the configured sink.
  - Write each log record within 50 ms of the request completing; add no more than 5 ms latency
    to response delivery (fire-and-forget enqueue).
  - Consumer retry: up to 3 attempts with exponential back-off completing within 5 seconds;
    drop record after exhausting retries; increment internal error counter; do not propagate to caller.
  - `record()`: enqueues a `TelemetryRecord`; assigns `request_id` from the active `RequestContext`.
  - `record_override()`: persists an `OverrideRecord` alongside the original telemetry record.
  - Rolling aggregator (`collections.deque` with max-age): lazy on-demand aggregation for `/v1/metrics`.
  - `RetentionManager`: enforce 90-day minimum retention for raw records and override records.
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.9, 8.5_

  - [x] 12.1 Implement async queue writer with fire-and-forget enqueue and retry consumer
    - Background consumer with exponential back-off up to 3 retries within 5 seconds; drop on exhaustion.
    - _Requirements: 6.2, 6.9_

  - [x] 12.2 Implement `record()` writing all required `TelemetryRecord` fields
    - Populate `blocking_trigger` for HARD_BLOCK requests; populate `pii_masking_bypassed` flag.
    - _Requirements: 6.1, 6.4, 6.5_

  - [x] 12.3 Implement rolling metrics aggregator and `get_metrics()` for `/v1/metrics`
    - Lazy O(N) aggregation over `collections.deque`; return `MetricsSummary` with all required fields.
    - _Requirements: 6.7_

  - [x] 12.4 Write property test for telemetry record structural completeness
    - **Property 17: Telemetry Record Structural Completeness**
    - For any completed request reaching the Triage Gateway, assert the `TelemetryRecord` is a
      valid instance with all required fields non-null and `request_id` is a valid UUID v4 string.
    - **Validates: Requirements 6.1, 6.3, 6.4**
    - _# Feature: controlplane-ai-gateway, Property 17: Telemetry Record Structural Completeness_

  - [x] 12.5 Write property test for unique request IDs
    - **Property 18: Unique Request IDs**
    - For any set of N ≥ 2 requests (including concurrent requests), assert all `request_id` values
      are distinct and each matches the UUID v4 regex pattern.
    - **Validates: Requirement 6.3**
    - _# Feature: controlplane-ai-gateway, Property 18: Unique Request IDs_

  - [x] 12.6 Write property test for HARD_BLOCK telemetry blocking trigger
    - **Property 19: HARD_BLOCK Telemetry Includes Trigger**
    - For any request resulting in HARD_BLOCK, assert `TelemetryRecord.blocking_trigger` is
      non-null and is one of the valid trigger identifiers.
    - **Validates: Requirement 6.5**
    - _# Feature: controlplane-ai-gateway, Property 19: HARD_BLOCK Telemetry Includes Trigger_

- [x] 13. Observability endpoints — `/v1/metrics` and `/v1/metrics/accuracy`
  - Implement the `GET /v1/metrics` endpoint in `app/telemetry/` (or a dedicated `metrics/` router):
    - Accept `window_minutes` (int, 1–1440, default 60); return 422 for out-of-range values.
    - Return `MetricsSummary` JSON (triage state counts, average groundedness score, routing distribution).
  - Implement `GET /v1/metrics/accuracy`:
    - Accept `window_days` (int, 1–30); return 422 for out-of-range values.
    - Return FPR, FNR, F1 for P1 toxicity, P1 injection, P2 PII computed from human-reviewed records.
    - Respond within 2 seconds for valid window values.
  - Mount both routers in `app/main.py`.
  - _Requirements: 6.7, 8.2, 8.4_

  - [x] 13.1 Implement `GET /v1/metrics` with window validation and lazy aggregation
    - 422 for out-of-range `window_minutes`; 200 with full `MetricsSummary` otherwise.
    - _Requirements: 6.7_

  - [x] 13.2 Implement `GET /v1/metrics/accuracy` with F1/FPR/FNR computation
    - Compute metrics over `OverrideRecord` set within the requested `window_days` window.
    - _Requirements: 8.2, 8.4_

  - [x] 13.3 Write property test for metrics window validation
    - **Property 20: Metrics Window Validation**
    - For any integer outside [1, 1440] passed as `window_minutes`, assert HTTP 422.
    - For any integer inside [1, 1440], assert HTTP 200 with all required aggregate fields.
    - **Validates: Requirement 6.7**
    - _# Feature: controlplane-ai-gateway, Property 20: Metrics Window Validation_

  - [x] 13.4 Write property test for accuracy metrics window validation
    - **Property 21: Accuracy Metrics Window Validation**
    - For any integer outside [1, 30] as `window_days`, assert HTTP 422.
    - For any integer inside [1, 30], assert HTTP 200 with FPR, FNR, F1 for all three judges
      returned within 2 seconds.
    - **Validates: Requirement 8.4**
    - _# Feature: controlplane-ai-gateway, Property 21: Accuracy Metrics Window Validation_

- [x] 14. Feedback Loop endpoints — `/v1/feedback/export` and `/v1/feedback/override`
  - Implement `app/feedback/router.py` mounting at `/v1/feedback`.
  - `POST /v1/feedback/override`:
    - Accept JSON body with `request_id`, `operator_id`, `original_verdict`, `human_label`,
      `stated_reason`; return 422 for missing/invalid fields.
    - Call `TelemetryLogger.record_override()`; persist `OverrideRecord` with operator ID,
      timestamp, and stated reason.
    - Return 200 with `override_id` and `timestamp`.
  - `GET /v1/feedback/export`:
    - Return JSON array of `FeedbackRecord` objects (original telemetry + override record if present
      + human label) for all escalated and human-overridden cases.
  - Implement `RetentionManager` enforcing 90-day minimum retention on all raw telemetry and
    override records.
  - Mount the feedback router in `app/main.py`.
  - _Requirements: 6.6, 6.8, 8.1, 8.3, 8.5_

  - [x] 14.1 Implement `POST /v1/feedback/override` with `OverrideRecord` persistence
    - Record `operator_id`, `timestamp`, `original_verdict`, `human_label`, `stated_reason`.
    - _Requirements: 8.3_

  - [x] 14.2 Implement `GET /v1/feedback/export` returning escalated and overridden cases
    - Include original telemetry record, override record (if any), and human label.
    - _Requirements: 6.8_

  - [x] 14.3 Implement `RetentionManager` enforcing 90-day minimum retention
    - Records older than 90 days must not be purged; implement enforcement in the log writer or a
      scheduled cleanup job.
    - _Requirements: 8.5_

  - [x] 14.4 Write property test for override records required metadata
    - **Property 23: Override Records Contain Required Metadata**
    - For any override submitted via `POST /v1/feedback/override` with a valid body, assert the
      persisted `OverrideRecord` has non-null `operator_id`, `timestamp`, `original_verdict`,
      `human_label`, `stated_reason`, and `human_label` is one of `PASS`, `SOFT_BLOCK`, `HARD_BLOCK`.
    - **Validates: Requirement 8.3**
    - _# Feature: controlplane-ai-gateway, Property 23: Override Records Contain Required Metadata_

- [x] 15. Wire the full pipeline in `app/main.py` and `app/ingress/router.py`
  - Connect all five stages inside the pipeline coroutine: Ingress → Orchestrator →
    (PII masking if needed) → ModelRouter → GroundednessAuditor → TriageGateway.
  - Implement the pipeline short-circuit: after each stage, check `RequestContext.upstream_triage_state`;
    if `HARD_BLOCK`, skip all remaining stages and call `TriageGateway.evaluate()` directly.
  - Ensure `TelemetryLogger.record()` is called at the end of every request path (happy path,
    early exit, and error path) within the ingress `finally` block.
  - Wire the FastAPI `lifespan` context to:
    - Load scanner models (LLM Guard, RouteLLM Controller, spaCy).
    - Run `PIIMaskingEngine.run_startup_validation()`; gate traffic on failure.
    - Start the `watchdog` observer for the Policy Layer.
    - Start the Telemetry Logger background consumer task.
  - _Requirements: 1.6, 1.7, 2.1, 2.3, 3.1, 5.8, 6.1, 6.2, 9.4, 9.5_

  - [x] 15.1 Implement the main pipeline coroutine and short-circuit logic
    - All five stages in sequence with `upstream_triage_state` guard between each stage.
    - _Requirements: 2.3, 5.7, 5.8_

  - [x] 15.2 Implement FastAPI `lifespan` startup sequence
    - Model loading, masking validation gate, watchdog start, telemetry consumer start.
    - _Requirements: 9.4, 9.5_

  - [x] 15.3 Ensure Telemetry Logger is called on every pipeline exit path
    - Confirm the `finally` block in the ingress handler always calls `record()` and
      `discard_mapping()` regardless of success, early exit, or exception.
    - _Requirements: 6.1, 6.2, 9.1_

- [x] 16. Checkpoint — full pipeline integration
  - Run all unit tests and verify the end-to-end happy path returns `PASS_AND_DELIVER` for both
    built-in profiles (mocked LLM and vector store). Ask the user if questions arise.

- [ ] 17. Integration and performance tests
  - [~] 17.1 Write end-to-end integration test for `customer_chatbot` and `internal_copilot` happy paths
    - Real LLM Guard scanners (ONNX), mocked Portkey responses, FAISS with a synthetic 10-doc corpus.
    - Verify correct `triage_state`, `request_id` UUID v4 format, and telemetry record written.
    - _Requirements: 1.5, 6.1_

  - [~] 17.2 Write integration test for policy hot-reload within 5 seconds
    - Modify the config file, assert updated profile is applied to new requests within 6 seconds.
    - _Requirements: 7.1_

  - [~] 17.3 Write integration test for routing distribution over 200 representative prompts
    - Verify approximately 80% ROUTINE / 20% COMPLEX within ±15 percentage points.
    - _Requirements: 3.5_

  - [~] 17.4 Write concurrent isolation test — 50 concurrent requests across two profiles
    - Assert no cross-contamination of `RequestContext`, `placeholder_map`, or triage state.
    - _Requirements: 1.7_

  - [~] 17.5 Write latency overhead test — 10 active profiles vs 1 profile
    - Assert processing latency does not increase by more than 2 ms vs single-profile baseline.
    - _Requirements: 7.6_

- [~] 18. Final checkpoint — all tests pass
  - Run the full test suite (`pytest --ignore=tests/integration -q` for unit + property tests;
    `pytest -m integration` for integration tests). Ensure all non-optional tests pass and no
    regressions exist. Ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP iteration.
- Every non-optional task must be completed before executing a downstream task in its wave.
- Property tests use [Hypothesis](https://hypothesis.readthedocs.io/) with a minimum of 100 examples each,
  tagged with `# Feature: controlplane-ai-gateway, Property {N}: {title}` for traceability.
- Integration tests are gated behind `@pytest.mark.integration` and require live LLM Guard model
  files; they are excluded from the default test run.
- All LLM Guard scanner models are loaded once at startup (inside the FastAPI `lifespan` context)
  to avoid cold-load latency on the first request.
- The RouteLLM `Controller` is instantiated once at startup; the per-profile threshold is injected
  at request time via the `router-mf-{threshold}` model string.
- Portkey fallback configs (`FRONTIER_CONFIG`, `SLM_CONFIG`) are constructed once at startup and
  reused across requests; they encode the "attempt alternative tier exactly once" requirement.
- The `placeholder_map` in `RequestContext` is always an empty dict even when masking is disabled;
  `discard_mapping()` in the `finally` block is therefore always safe to call.

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "2.1", "2.2"] },
    { "id": 2, "tasks": ["2.3", "3.1", "3.2", "3.3"] },
    { "id": 3, "tasks": ["3.4", "3.5", "3.6", "5.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "6.1", "6.2", "6.3"] },
    { "id": 5, "tasks": ["5.4", "5.5", "6.4", "12.1"] },
    { "id": 6, "tasks": ["6.5", "12.2", "12.3", "9.1"] },
    { "id": 7, "tasks": ["6.6", "6.7", "6.8", "8.1", "9.2", "9.3"] },
    { "id": 8, "tasks": ["8.2", "8.3", "9.4", "9.5", "10.1", "12.4", "12.5", "12.6"] },
    { "id": 9, "tasks": ["8.4", "8.5", "10.2", "10.3", "13.1", "13.2"] },
    { "id": 10, "tasks": ["10.4", "10.5", "10.6", "13.3", "13.4", "14.1", "14.2", "14.3"] },
    { "id": 11, "tasks": ["14.4", "15.1", "15.2"] },
    { "id": 12, "tasks": ["15.3"] },
    { "id": 13, "tasks": ["17.1"] },
    { "id": 14, "tasks": ["17.2", "17.3", "17.4", "17.5"] }
  ]
}
```
