# ControlPlane.ai — Current State and Architecture

## Overview

**ControlPlane.ai** is an Enterprise AI Proxy Gateway built on FastAPI. It mediates every interaction between enterprise applications or agents and underlying Large Language Models through a deterministic five-stage safety pipeline that enforces safety, cost governance, groundedness assurance, and output triage before a response is delivered to callers.

This document reflects the current implementation state after two rounds of development, including the original five-stage pipeline build and five targeted hardening improvements.

---

## The 5-Stage Safety Pipeline

The pipeline is defined in `app/main.py` and wired as `app.state.pipeline_fn`. Each request travels through five sequential checkpoints. A `HARD_BLOCK` verdict at any stage short-circuits all downstream stages.

### Stage 1 — Orchestrator (Micro-Judges & Policy Layer)

**Files:** `app/judges/orchestrator.py`, `app/judges/p1_judge.py`, `app/judges/p2_judge.py`, `app/judges/p3_judge.py`, `app/judges/pii_masking.py`, `app/policy/loader.py`

The orchestrator runs P1, P2, and P3 judges concurrently via `asyncio.gather` inside `asyncio.wait_for(inspection_timeout_ms)`.

**PII Masking Engine** (`app/judges/pii_masking.py`) — two-tier design:
- **Tier 1 — NLPMasker**: Wraps LLM Guard `Anonymize` (Presidio-backed). Highest accuracy; used when `llm-guard` is installed.
- **Tier 2 — RegexOnlyMasker**: Pure compiled-regex scanner covering SSNs, email addresses, US phone numbers, and credit card numbers. Zero extra dependencies, ~1 ms per call. Always available.
- **Startup validation**: On startup, the engine runs 5 synthetic PII prompts through a full `mask → unmask` round-trip. If the NLP tier fails (exception or fidelity mismatch), it **downgrades to the regex tier** and emits a `MASKING_DEGRADED_TO_REGEX` HIGH-priority alert via the Telemetry Logger (or stderr if no logger is yet wired). The gateway **stays online** (`is_healthy=True`). Only if the regex tier also fails does the engine set `is_healthy=False`, causing the ingress to return HTTP 503.
- `active_tier` property reports the currently active scanner (`"nlp"` or `"regex"`).

**Policy Loader** (`app/policy/loader.py`): Loads `UseCaseProfile` configurations from a YAML or JSON file. Uses `watchdog` for hot-reload within 5 seconds of file modification. Atomically swaps the live config only if all profiles pass Pydantic validation; otherwise retains the previous valid config. Two built-in profiles (`customer_chatbot`, `internal_copilot`) are always available even without a config file.

**P1 Judge** (`app/judges/p1_judge.py`): LLM Guard `Toxicity` + `PromptInjection` scanners (with stub fallbacks when `llm-guard` is not installed). Returns `P1Verdict` with `toxicity_verdict` and `injection_verdict`, each `"BLOCK"` or `"PASS"`. Any BLOCK immediately sets `upstream_triage_state=HARD_BLOCK` and skips all downstream stages.

**P2 Judge** (`app/judges/p2_judge.py`): Calls the PII Masking Engine; returns `P2Verdict` with `pii_count`, `masked_prompt`, and `placeholder_map`. When `pii_masking_enabled=False` on the profile, the judge returns `pii_count=0` and the Telemetry Logger records `PII_MASKING_BYPASSED`. An engine error returns `pii_count=sys.maxsize`, triggering `HARD_BLOCK`.

**P3 Judge** (`app/judges/p3_judge.py`): Lightweight local classifier using `tiktoken` (token count) and spaCy `en_core_web_sm` (dependency parse). Returns `"AMBIGUOUS"` when token count ≤ 10 or no ROOT-tagged VERB/AUX token is found; `"CLEAR"` otherwise. The spaCy model is loaded eagerly at import time.

**Timeout handling**: If `asyncio.wait_for` fires, the orchestrator sets `upstream_triage_state=ESCALATE_TO_HUMAN`.

**Failure isolation**: P1 exception → `BLOCK/BLOCK`; P2 exception → `pii_count=sys.maxsize`; P3 exception → `AMBIGUOUS`.

---

### Stage 2 — Model Router (RouteLLM + Portkey)

**Files:** `app/router/model_router.py`

The RouteLLM `Controller` is instantiated once at startup. The per-profile `complexity_threshold` is embedded in the model string as `router-mf-{threshold}`. When `p3_clarity=AMBIGUOUS`, a system-message bias prefix nudges the router toward the Frontier Model.

- **ROUTINE** (score < threshold) → SLM tier (e.g. GPT-4o-mini via Portkey)
- **COMPLEX** (score ≥ threshold) → Frontier Model tier (e.g. GPT-4o via Portkey)

Portkey provides unified access, automatic retries, and fallback: if the primary tier exhausts its retry budget, the alternative tier is attempted once. If both fail, `triage_state=HARD_BLOCK` and `MODEL_TIER_FAILURE` is logged.

**Guardrails AI Output Validation** (`app/judges/output_validator.py`): After the LLM responds, output is run through the Guardrails AI validator chain:
- `action="fix"` → pipeline continues with the fixed output.
- `action="filter"` or `action="exception"` → `upstream_triage_state=HARD_BLOCK`.
- Internal validator exceptions are caught and also produce `HARD_BLOCK` — they never propagate to the event loop.

---

### Stage 3 — Groundedness Auditor

**Files:** `app/groundedness/auditor.py`, `app/groundedness/vector_store.py`

Embeds the LLM response and computes mean cosine similarity against the top-K most relevant documents retrieved from the Enterprise Vector Store. Returns a `groundedness_score` in `[0.0, 1.0]`. The default implementation uses FAISS; a pgvector adapter stub is also provided.

If the vector store raises any exception, `groundedness_score=0.0` is returned with `is_unverified=True` and `VECTOR_STORE_UNAVAILABLE` is logged. This keeps the Triage Gateway able to make a deterministic decision.

---

### Stage 3b — Worldsense Multi-Turn Agentic Oversight

**Files:** `app/oversight/worldsense_oversight.py`, `mcp_servers/worldsense/server.py`

Evaluates the full conversation history and proposed LLM response for hidden consequence chains and privilege escalation patterns. Returns one of three verdicts: `SAFE`, `RISK_DETECTED` (→ `ESCALATE_TO_HUMAN`), or `CONSEQUENCE_ALERT` (→ `HARD_BLOCK`).

**Three-tier evaluation priority** (highest to lowest):

1. **MCP Server** — HTTP POST to the isolated Worldsense MCP server at `http://localhost:9100/evaluate` (configurable via `WORLDSENSE_MCP_URL`). This server runs in its own Python process with its own dependency tree, completely decoupling the `worldsense` SDK from the main gateway's `pyproject.toml`. The `_MCP_HEALTHY` flag is cached per-process; first-time failures are logged once to avoid log spam.

2. **Local SDK** — direct `worldsense.evaluate()` call if the package happens to be installed in the main venv.

3. **Heuristic Evaluator** — always available rule-based fallback: `_RISK_KEYWORDS` detects single-turn risks; `_CONSEQUENCE_PATTERNS` across 2+ turns triggers `CONSEQUENCE_ALERT`.

All paths are bounded by `WORLDSENSE_TIMEOUT_MS` (default 300 ms, `app/config.py`). Timeout → `RISK_DETECTED` (fail-safe).

**Worldsense MCP Server** (`mcp_servers/worldsense/server.py`): Standalone FastAPI app exposing `POST /evaluate`, `GET /health`, `GET /info`. Configured via `mcp_servers/worldsense/mcp.json`. Has its own `requirements.txt` for isolated venv installation. Contains an identical copy of the heuristic evaluator with no imports from `app.*`.

---

### Stage 4 — Triage Gateway

**Files:** `app/triage/gateway.py`, `app/triage/compressor.py`

Applies a four-state priority matrix (highest to lowest):

| Priority | State | Condition |
|---|---|---|
| 1 | `HARD_BLOCK` | Upstream `triage_state=HARD_BLOCK` OR `groundedness_score < 0.5` |
| 2 | `ESCALATE_TO_HUMAN` | `0.5 ≤ score ≤ profile.groundedness_pass_threshold` OR `p3_clarity=AMBIGUOUS` (when `human_escalation_enabled=True`) |
| 3 | `COMPRESS_AND_EDIT` | `score > threshold` AND `response_token_count > profile.token_compression_threshold` |
| 4 | `PASS_AND_DELIVER` | All other cases |

When `human_escalation_enabled=False`, any `ESCALATE_TO_HUMAN` outcome is promoted to `HARD_BLOCK`. When `COMPRESS_AND_EDIT` fires, `app.triage.compressor` sends a token-budget summarisation prompt to the SLM tier via Portkey. The PII Masking Engine's `unmask()` is called on the final response before delivery, restoring all `[TYPE_REDACTED]` placeholders to their original values.

---

## Observability & Telemetry

**Files:** `app/telemetry/logger.py`, `app/telemetry/router.py`, `app/observability/langfuse_tracer.py`

**Telemetry Logger** (`app/telemetry/logger.py`): Singleton with an `asyncio.Queue`. `record()` is fire-and-forget (≤5 ms caller latency). Background consumer retries failed writes up to 3 times with exponential back-off, completing within 5 seconds; drops the record and increments an error counter after exhaustion. A rolling `collections.deque` aggregator supports lazy O(N) metrics computation for `/v1/metrics`. `RetentionManager` enforces a 90-day minimum retention floor on all records.

**Langfuse Tracer** (`app/observability/langfuse_tracer.py`): Tracks execution traces, prompt chains, and LLM call metadata. Falls back to stdout logging when `LANGFUSE_PUBLIC_KEY` is not set. Red team breakthroughs are recorded as `RED_TEAM_BREAKTHROUGH` span events at `ERROR` severity.

---

## Ancillary Endpoints & Routers

| Router | Prefix | Purpose |
|---|---|---|
| `app.ingress.router` | `POST /v1/chat` | Primary LLM interaction endpoint |
| `app.telemetry.router` | `GET /v1/metrics`, `GET /v1/metrics/accuracy` | Aggregate telemetry and judge accuracy metrics |
| `app.feedback.router` | `GET /v1/feedback/export`, `POST /v1/feedback/override` | Human-operator override recording and feedback export |
| `app.redteam.router` | `POST /v1/redteam/run`, `GET /v1/redteam/report` | Adversarial red team testing |

---

## Developer Automation — Kiro Hooks

**File:** `.kiro/hooks/redteam-on-policy-save.json`, `.kiro/hooks/redteam_trigger.py`

A **PostFileSave** Kiro hook (matcher: `\.(yaml|yml|json)$`) fires whenever a YAML or JSON file is saved. The hook script (`redteam_trigger.py`) reads the Kiro session context from stdin, determines whether the saved file is a policy file (path contains `"policy"`, `"profiles"`, or `"use_case"`), then `POST /v1/redteam/run` against the running gateway and prints a summary including block rate and any breakthrough categories. Breakthroughs are emitted as prominent stderr warnings. The hook always exits 0 — it never blocks the save operation.

---

## Developer Rules — Kiro Steering

**File:** `.kiro/steering/performance-budget.md` (`inclusion: auto`)

Automatically injected into every Kiro session. Defines per-component latency ceilings:

| Component | Ceiling |
|---|---|
| P1 Judge | 150 ms |
| P3 Judge | 50 ms |
| Heuristic Evaluator | 20 ms |
| PII Masking Engine (regex) | 30 ms |
| Full Orchestrator (wall-clock) | 200 ms |

Mandates profiling (p99 via `timeit`) before committing any change that adds or modifies a regex pattern, introduces a new loop over tokens or turns, adds a synchronous library import, modifies the spaCy pipeline, or extends `_RISK_KEYWORDS`/`_CONSEQUENCE_PATTERNS` beyond 30 entries. Also enforces the `asyncio.to_thread` async-safety rule for all CPU-bound scanner calls.

---

## Test Suite

All tests live in `tests/unit/`. 133 unit tests pass in the current environment.

| Test file | What it covers |
|---|---|
| `test_models_properties.py` | Pydantic field-range validation (Property 22) |
| `test_policy_loader.py` | Hot-reload, atomic swap, built-in profiles (Property 1) |
| `test_ingress.py` | 422/503/504 responses, UUID v4 request IDs, state isolation |
| `test_ingress_properties.py` | Properties 2, 3, 4 (invalid rejection, latency budget, concurrent isolation) |
| `test_pii_masking.py` | Mask/unmask/discard round-trip, startup validation |
| `test_pii_properties.py` | Properties 6, 16 (token replacement, round-trip fidelity) |
| `test_pii_graceful_degradation.py` | Two-tier NLP→regex downgrade, `MASKING_DEGRADED_TO_REGEX` alert, both-tiers-fail→503 |
| `test_judges.py` | P1/P2/P3 verdicts, error defaults, edge cases |
| `test_orchestrator.py` | Concurrent execution, P1 short-circuit, timeout→ESCALATE_TO_HUMAN, failure isolation |
| `test_telemetry_queue.py` | Async queue, retry back-off, RetentionManager 90-day floor |
| `test_output_validator_properties.py` | Properties P-OV-1 through P-OV-7: never raises, exception→HARD_BLOCK, unfixable→passed=False, fix action, event-loop safety |

---

## Infrastructural State

### Python Environment
- Python 3.13.x; dependencies managed via `pyproject.toml` with `>=` lower bounds (not `==`) to avoid `pydantic-core` Rust build failures on Python 3.13.
- Primary venv: `.venv` in the project root.
- Worldsense MCP server has its own `mcp_servers/worldsense/requirements.txt` for isolated venv installation.

### Key Dependencies
- **FastAPI** + **uvicorn** — HTTP framework and ASGI server
- **Pydantic v2** + **pydantic-settings** — data validation and config
- **watchdog** — filesystem observer for Policy Layer hot-reload
- **spaCy** (`en_core_web_sm`) — P3 Judge dependency parse
- **tiktoken** — P3 Judge token counting
- **FAISS** (`faiss-cpu`) — Groundedness Auditor vector store
- **httpx** — async HTTP client (Worldsense MCP calls, Redteam runner)
- **hypothesis** + **pytest-asyncio** — property-based and async testing
- **llm-guard** *(optional)* — NLP-based PII scanning (Tier 1); regex fallback active when absent
- **routellm**, **portkey-ai** — Model Router complexity classification and LLM dispatch
- **langfuse** *(optional)* — distributed tracing; falls back to stdout
- **guardrails-ai** *(optional)* — output validation chain; no-op pass-through when absent
- **worldsense** *(not in pyproject.toml)* — available via MCP server isolation or local install

### Worldsense Dependency Resolution
The `worldsense` package is **not listed in `pyproject.toml`** to avoid dependency conflicts (particularly the `pydantic-core` Rust compiler requirement on Python 3.13). It is instead accessed through the isolated MCP server (`mcp_servers/worldsense/`). The oversight module uses the three-tier MCP → local SDK → heuristic priority, so the gateway functions correctly under all three installation states.

### Notable Fixed Issues
- **`run_orchestrator` alias**: `app/judges/orchestrator.py` exports `run_orchestrator = run_micro_judges` to satisfy the import in `app/main.py`.
- **Hypothesis fixture scoping**: All `@given` test bodies that mutate `_LOADED_VALIDATORS` call `_LOADED_VALIDATORS.clear()` at the start, because Hypothesis does not re-run `autouse` fixtures between generated examples.
- **`_FallbackScanner` closure bug**: Nested `@dataclass` definitions inside factory functions do not capture closure variables as field defaults; replaced with plain classes using `__init__` in the stub validators.
