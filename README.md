# ControlPlane.ai

**An enterprise AI proxy gateway that sits between your applications and any LLM, enforcing
safety, cost governance, groundedness, and output triage on every request.**

ControlPlane.ai is a FastAPI service that mediates every interaction between enterprise
applications or autonomous agents and the underlying Large Language Model. Rather than letting
callers reach a model directly, each request travels through a deterministic five-stage
pipeline that inspects the prompt, routes it to an appropriate model tier, audits the response
for groundedness, and decides — on policy, not on vibes — whether the result is delivered,
compressed, escalated to a human, or blocked outright.

Every external integration is optional. The gateway starts and serves traffic with no API keys
at all, degrading each unavailable component to a documented fallback instead of failing.

---

## Table of Contents

- [Why a gateway](#why-a-gateway)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Provider Compatibility](#provider-compatibility)
- [Graceful Degradation](#graceful-degradation)
- [Testing](#testing)
- [Project Layout](#project-layout)
- [Current Status and Known Limitations](#current-status-and-known-limitations)

---

## Why a gateway

Putting a model behind a policy layer solves four problems that are awkward to solve inside
application code:

| Concern | How the gateway addresses it |
|---|---|
| **Safety** | Toxicity, prompt-injection, and PII inspection run concurrently before the model is ever called. A hard block short-circuits the remaining stages. |
| **Cost governance** | A semantic cache answers repeat questions without an LLM call. Prompt complexity scoring routes cheap work to a small model and hard work to a frontier model. |
| **Groundedness** | Responses are scored against retrieved context with embedding similarity and an NLI cross-encoder. A detected contradiction is the highest-priority block in the system. |
| **Operational control** | Policy profiles hot-reload from disk without a restart. Every decision is telemetered, and a single endpoint reports which integrations are live versus degraded. |

---

## Architecture

The pipeline lives in `app/main.py` as `run_pipeline()`, stored at `app.state.pipeline_fn`.
Requests pass through five sequential checkpoints; a `HARD_BLOCK` at any stage skips all
downstream work.

```
POST /v1/chat
     │
     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 0 — Semantic Cache                      app/router/semantic_cache.py│
│   Cosine-similarity lookup (threshold 0.92). A hit short-circuits         │
│   all five stages: the LLM is never called.                               │
└───────────────────────────────────────────────────────────────────────────┘
     │ miss
     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Orchestrator                          app/judges/orchestrator.py│
│   P1  toxicity + prompt injection  ┐                                      │
│   P2  PII detection and masking    ├─ concurrent, bounded by              │
│   P3  query clarity                ┘  inspection_timeout_ms               │
└───────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 2 — Model Router                          app/router/model_router.py│
│   Tiered dispatch to the configured provider, with output                 │
│   validation via Guardrails AI.                                           │
└───────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 3 — Groundedness Auditor                 app/groundedness/auditor.py│
│   3a  embedding cosine similarity  → score ∈ [0.0, 1.0]                   │
│   3b  NLI cross-encoder            → ENTAILMENT / NEUTRAL / CONTRADICTION │
│   3c  Worldsense multi-turn consequence-chain oversight                   │
└───────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Stage 4 — Triage Gateway                             app/triage/gateway.py│
│   HARD_BLOCK  │  ESCALATE_TO_HUMAN  │  COMPRESS_AND_EDIT                  │
│   PASS_AND_DELIVER                                                        │
└───────────────────────────────────────────────────────────────────────────┘
```

### Stage 1 — PII masking is three-tiered

| Tier | Component | Behaviour |
|---|---|---|
| 1 | `NLPMasker` | LLM Guard `Anonymize`, Presidio-backed. Used when `llm-guard` is installed. |
| 1.5 | `GLiNERMasker` | Zero-shot NER over `UseCaseProfile.custom_entity_terms`, producing `[CUSTOM_ENTITY_REDACTED_N]`. Skipped when no terms are configured. |
| 2 | `RegexOnlyMasker` | Compiled regex for SSN, email, phone, credit card. Always available. |

Startup validates masking against five synthetic prompts. If the NLP tier fails, the engine
downgrades to regex and emits `MASKING_DEGRADED_TO_REGEX`. If **both** tiers fail, the service
reports unhealthy and returns HTTP 503 rather than silently passing PII through.

Judge failures are isolated rather than fatal: a P1 exception yields `BLOCK/BLOCK`, a P2
exception yields `pii_count=sys.maxsize`, and a P3 exception yields `AMBIGUOUS` — every default
fails closed.

### Stage 4 — The triage matrix

Priorities are evaluated in order; the first match wins.

| Priority | State | Condition |
|---|---|---|
| **0** | `HARD_BLOCK` | `nli_label == "CONTRADICTION"` → `blocking_reason="NLI_CONTRADICTION"` |
| 1 | `HARD_BLOCK` | Upstream hard block, or `groundedness_score < 0.5` |
| 2 | `ESCALATE_TO_HUMAN` | `0.5 ≤ score ≤ pass_threshold`, or `p3_clarity == AMBIGUOUS` |
| 3 | `COMPRESS_AND_EDIT` | `score > threshold` **and** `token_count > compression_threshold` |
| 4 | `PASS_AND_DELIVER` | All other cases |

### Policy profiles

Profiles are loaded from YAML or JSON and hot-reload via watchdog within five seconds, swapped
atomically only after validation succeeds. Two profiles are always available:
`customer_chatbot` and `internal_copilot`. Each profile carries its own latency budget,
groundedness threshold, cache settings, and custom entity terms.

---

## Quick Start

### Requirements

- Python **3.11 or newer** (`pyproject.toml` declares `requires-python = ">=3.11"`; developed
  and verified against 3.14)

### Install

The required dependency floor is deliberately small — six packages, roughly 57 MB:
`fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `httpx`, `watchdog`.

```bash
python -m venv .venv
source .venv/bin/activate

# Minimal gateway. Serves traffic; optional stages degrade to fallbacks.
pip install -e .
```

Everything beyond the floor lives behind an extra, so you install only the pipeline stages you
actually intend to run:

```bash
pip install -e '.[llm]'        # real LLM dispatch via LiteLLM + tiktoken
pip install -e '.[safety]'     # llm-guard, spaCy, GLiNER, Guardrails AI
pip install -e '.[cache]'      # semantic cache (Stage 0)
pip install -e '.[grounded]'   # NLI cross-encoder (Stage 3b)
pip install -e '.[observe]'    # Langfuse tracing
pip install -e '.[dev]'        # pytest, pytest-asyncio, hypothesis
pip install -e '.[all]'        # every runtime extra
```

Installing everything pulls in the full ML stack — on the order of 6.3 GB, against 57 MB for
the floor. Choose extras accordingly.

### Configure

```bash
cp .env.example .env
```

`.env.example` is the canonical operator reference: it documents every variable, where to
obtain each key, and what happens when the key is absent. It is kept in sync with
`app/config.py` — and that sync is enforced by a test, not by review convention. `.env` itself
is git-ignored and must never be committed.

### Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then confirm what is actually live:

```bash
curl -s http://127.0.0.1:8000/v1/config/health
```

Startup logs a structured configuration summary with no secret values:

```
ControlPlane.ai configuration summary:
  LLM direct key : CONFIGURED / NOT CONFIGURED
  Portkey        : ACTIVE (provider=X) / DEGRADED (mock)
  Langfuse       : ACTIVE (host=H) / DEGRADED (stdout)
  Guardrails     : ACTIVE (N validators) / DEGRADED (none loaded)
  Worldsense MCP : ACTIVE (URL) / DEGRADED (heuristic)
```

---

## Configuration

All configuration is environment-driven through pydantic-settings, grouped into ten sections
that mirror `.env.example`:

| Section | Key settings |
|---|---|
| 1. LLM Provider | `LLM_API_KEY`, `LLM_PROVIDER`, `LLM_FALLBACK_MODEL`, `LLM_API_BASE`, `SLM_MODEL`, `FRONTIER_MODEL`, `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES` |
| 2. Portkey Gateway *(being replaced)* | `PORTKEY_API_KEY`, `PORTKEY_SLM_VIRTUAL_KEY`, `PORTKEY_FRONTIER_VIRTUAL_KEY` |
| 3. Langfuse | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` |
| 4. Guardrails AI | `GUARDRAILS_VALIDATORS`, `GUARDRAILS_HUB_TOKEN` |
| 5. Worldsense MCP | `WORLDSENSE_ENABLED`, `WORLDSENSE_MCP_URL`, `WORLDSENSE_TIMEOUT_MS` |
| 6. Redteam MCP | `REDTEAM_ENABLED`, `REDTEAM_MIN_PROMPTS`, `REDTEAM_SCHEDULE` |
| 7. Semantic Cache | `CACHE_SIMILARITY_THRESHOLD` |
| 8. Policy & Pipeline | `POLICY_FILE_PATH`, `EMBEDDING_MODEL`, `VECTOR_STORE_TOP_K` |
| 9. Telemetry | `TELEMETRY_SINK`, `TELEMETRY_LOG_FILE` |
| 10. Obot Governance | `OBOT_ENABLED`, `OBOT_MAX_TOOL_CALLS_DEFAULT`, `OBOT_LATENCY_BUDGET_MS` |

`extra="ignore"` is set on the settings model, so unknown environment variables never cause a
startup error. This has one consequence worth knowing: a legacy `OPENAI_API_KEY` in `.env` is
silently ignored rather than flagged — it was renamed to `LLM_API_KEY` for provider generality,
and operators must rename it themselves.

### Running against a local, keyless model

`LLM_API_BASE` points the gateway at any OpenAI-compatible endpoint — Ollama, vLLM, LM Studio,
or a self-hosted proxy:

```bash
LLM_API_BASE=http://localhost:11434
```

Such a backend needs no API key at all, so a configured `LLM_API_BASE` is treated as
sufficient on its own to consider the egress path live.

### Key validation

A single helper decides whether a credential is real, and every integration uses it. Blank and
whitespace-only values are rejected, as is any `dummy` prefix — which is what makes the
committed placeholder configuration safe:

```python
_is_real_key("")                   # → False
_is_real_key("   ")                # → False
_is_real_key("dummy-portkey-key")  # → False
_is_real_key("sk-real-abc123")     # → True
```

---

## API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/v1/chat` | Primary non-streaming interaction |
| `POST` | `/v1/chat/stream` | Server-sent-events streaming |
| `GET` | `/v1/config/health` | Live integration status |
| `GET` | `/v1/profiles` | Available policy profiles |
| `POST` | `/v1/policy` | Policy management |
| `GET` | `/v1/metrics` | Aggregate metrics |
| `GET` | `/v1/metrics/accuracy` | Accuracy metrics |
| `GET` | `/v1/feedback/export` | Export operator feedback |
| `POST` | `/v1/feedback/override` | Record an operator override |
| `POST` | `/v1/redteam/run` | Trigger an adversarial run |
| `GET` | `/v1/redteam/report` | Fetch the latest red-team report |

### Non-streaming request

Both `prompt` and `use_case_profile` are required.

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{
        "prompt": "What is your refund policy for damaged goods?",
        "use_case_profile": "customer_chatbot"
      }'
```

The response carries the triage decision alongside the content, so a caller can distinguish a
delivered answer from an escalation without parsing prose:

```json
{
  "request_id": "…",
  "response": "…",
  "triage_state": "PASS_AND_DELIVER",
  "blocking_reason": null,
  "latency_ms": 312
}
```

### Streaming

`POST /v1/chat/stream` returns `text/event-stream`. The full pre-flight pipeline runs before
the first token, and every chunk — including those emitted by `flush_remaining()` — passes
through output validation, so a violation mid-stream severs the connection rather than
completing it.

| Frame | Meaning |
|---|---|
| `data: <chunk>` | Clean content chunk |
| `data: [DONE]` | End of a clean stream, always final |
| `data: [REDACTED DUE TO POLICY]` | Policy violation; stream severed |
| `data: [STREAM_ERROR]` | Unhandled upstream exception |

### Integration health

`GET /v1/config/health` reads only in-memory state — no outbound calls, responds in under
50 ms, no authentication required:

```json
{
  "portkey":    { "status": "active|degraded", "detail": "…" },
  "langfuse":   { "status": "active|degraded", "detail": "…" },
  "guardrails": { "status": "active|degraded", "detail": "…" },
  "worldsense": { "status": "active|degraded", "detail": "…" },
  "llm_direct": { "status": "active|degraded", "detail": "…" }
}
```

> **Read this endpoint carefully.** `llm_direct: active` means a key is *configured* — not that
> calls are succeeding. A rate-limited or quota-exhausted key still reports `active` while
> `/v1/chat` quietly serves mock text, because the upstream error is swallowed in the router.

---

## Provider Compatibility

The direct-call path uses the OpenAI SDK against a provider-specific base URL:

| Provider | `LLM_PROVIDER` | Base URL |
|---|---|---|
| OpenAI | `openai` | *(SDK default)* |
| Google Gemini | `google` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Anthropic | `anthropic` | `https://api.anthropic.com/v1/` |
| Grok / xAI | `grok` | `https://api.x.ai/v1/` |
| Any compatible | `generic` | *(SDK default, override with `LLM_API_BASE`)* |

### Notes for Google Gemini free-tier keys

Probed against the live API on 2026-08-30:

- `gemini-3.5-flash` works. `gemini-1.5-flash` and `gemini-2.0-flash` were retired in June
  2026, and `gemini-2.5-flash` now returns 404 "no longer available to new users".
- `gemini-3.6-flash` and `gemini-flash-latest` time out; `gemini-pro-latest` returns 429 and is
  not on the free tier — so a free-tier key has no usable second (frontier) tier.
- The free tier allows **20 requests per day per model**
  (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). Once exhausted, responses fall back to
  mock text while health still reports `active`.
- Gemini 3.x are *thinking* models: internal reasoning tokens are billed against `max_tokens`
  before any visible text is emitted. The dispatch budget is set to 2048 for this reason — at
  512, answers were truncated mid-sentence with `finish_reason=length`.

---

## Graceful Degradation

Every optional dependency is guarded, and the fallback is documented rather than incidental:

| Missing | Consequence |
|---|---|
| `gptcache` | Semantic cache becomes a no-op; every request is a cache miss |
| `llm-guard` | P1 uses stub scanners; PII masking falls back to the regex tier |
| `spaCy` model | P3 falls back to a token-count-only clarity rule |
| `gliner` | Custom-entity masking is skipped; standard PII masking is unaffected |
| `sentence-transformers` | Stage 3b NLI scoring is skipped; embedding similarity still runs |
| `guardrails-ai` | Output validation passes through |
| `langfuse` | Tracing falls back to stdout with a buffered retry loop |
| No LLM credentials | Deterministic contextual mock responses; never raises |

Guards catch `Exception`, not just `ImportError` — deliberately. An optional package can begin
importing and then fail *partway through its own import*, raising something else entirely. Two
real instances: `guardrails` touches `openai.error`, removed in `openai>=1`, raising
`AttributeError`; and `routellm` constructs an `OpenAI()` client at module import time, raising
`OpenAIError` when no `OPENAI_API_KEY` is set. Because `app/main.py` imports both modules at
startup, a narrow guard let each escape and take down the whole application.

---

## Testing

```bash
pip install -e '.[dev]'
pytest tests -q
```

**299 tests collected: 295 pass, 4 skipped, 0 failures** (verified 2026-08-30 on Python 3.14).

| Group | Count | Focus |
|---|---|---|
| Pre-Phase-3 | 154 | Models, policy loader, ingress, PII masking, judges, orchestrator, telemetry |
| Phase 3 properties | 74 | Semantic cache, NLI scoring, GLiNER masking, SSE streaming, red team |
| API key integration | 17 | `_is_real_key()` edge cases, `/v1/config/health` schema and latency |
| Vendor independence | 45 | Complexity scoring, provider tier resolution, packaging floor, `.env.example` sync |
| Frontend integration | 5 | Dashboard serving and static assets |

Two testing conventions in this repository are worth calling out, because both exist to catch
a specific class of silent failure:

**Tests are hermetic by construction.** `tests/conftest.py` holds an `autouse` fixture that
resets every `Settings` field to its declared default, so a developer's `.env` cannot change
what the suite exercises. Without it, real credentials on disk caused two request-validation
tests to dispatch live provider calls — exhausting a daily quota and then failing on the
resulting rate-limit error.

**Anti-regression tests are mutation-checked.** For each such test the original defect is
deliberately reintroduced, the test is confirmed to go red, and the mutated suite is confirmed
to still *collect* — a mutant that fails to import proves nothing. This is not ceremony: it
caught three of the four complexity-scorer signal weights being entirely unconstrained by tests
that looked convincing in the diff and passed.

`.env.example` drift is also a test failure, not a review convention:
`tests/unit/test_env_example_sync.py` fails both when a `Settings` field is documented nowhere
and when `.env.example` names a key no field reads.

---

## Project Layout

```
app/
  config.py                  Pydantic-Settings config; _is_real_key()
  main.py                    FastAPI app, 11-step lifespan, run_pipeline()
  models.py                  Shared Pydantic and dataclass models
  config_health/router.py    GET /v1/config/health
  groundedness/              Two-stage auditor, NLI scorer, vector store protocol
  ingress/                   POST /v1/chat, SSE streaming, sliding-window buffer
  judges/                    P1/P2/P3 judges, orchestrator, PII + GLiNER masking,
                             Guardrails output validation
  observability/             Langfuse tracing with stdout fallback
  oversight/                 Worldsense multi-turn consequence-chain oversight
  policy/loader.py           Hot-reloading policy loader
  redteam/                   MCP-first red-team orchestrator and routes
  router/                    Model dispatch, local complexity scoring,
                             LiteLLM egress, semantic cache
  telemetry/                 Async telemetry queue, retention, metrics routes
  triage/                    Four-state triage matrix and token compressor
  feedback/router.py         Operator export and override

mcp_servers/
  worldsense/                Isolated oversight MCP server (own venv)
  redteam/                   Isolated red-team MCP server (own venv, no app.* imports)

tests/unit/                  299 unit and property tests
.env.example                 Canonical config reference (committed)
.env                         Local config (git-ignored, never committed)
.kiro/                       Specs, steering docs, and PostFileSave automation hooks
```

The lifespan in `app/main.py` initialises components in eleven ordered steps — policy loader,
judge models, PII engine and its startup validation, telemetry and tracing, router controller,
semantic cache, GLiNER, NLI scorer, and health probes for both MCP servers — each stored on
`app.state` for the request path to reach.

### Red teaming

`mcp_servers/redteam/server.py` runs as an isolated FastAPI process with zero `app.*` imports,
enforced by a test. It attempts PyRIT, then Garak, then a built-in five-category prompt library
covering jailbreaks, injection, toxicity, PII extraction, and competitor injection. The
in-process runner tries the MCP server first and falls back on any error; a breakthrough is
logged as `RED_TEAM_BREAKTHROUGH` at `ERROR` level along either path.

A `PostFileSave` hook fires whenever a policy file is saved: it probes MCP health, proceeds to
trigger a red-team run regardless, and always exits 0 so it never blocks a save.

---

## Current Status and Known Limitations

The project is mid-way through a **vendor independence** effort whose goal is to depend on no
commercial software apart from whichever LLM the operator chooses to call. Being explicit about
what has and has not landed matters more here than a clean narrative.

### Landed

| Change | Effect |
|---|---|
| Required dependencies cut to six | Everything else moved behind an extra and guarded at import |
| `faiss-cpu` and `portkey-ai` removed outright | Neither was imported anywhere in `app/`; deleted rather than made optional |
| Import guards widened to `Exception` | A package that imports and *then* fails now degrades instead of crashing startup |
| `app/router/complexity.py` | Local, dependency-free prompt complexity scoring |
| `app/router/providers.py` | LiteLLM egress layer with SLM→FRONTIER fallback and mock degradation. LiteLLM is BSD-3 and a library, not a service: no account, no proxy process, no per-call vendor hop |
| `.env.example` sync enforced by test | Undocumented settings and orphaned keys both fail the suite |

Required install footprint: **~57 MB, down from ~6.3 GB** — roughly 110× — for an operator who
wants the gateway and nothing optional.

### Not yet landed

- **`model_router.py` is still the original Portkey-first implementation.** `complexity.py` and
  `providers.py` are built and tested but **not yet wired into the pipeline**.
- **Tiering is not yet real.** The live router sets a fixed complexity score, so every prompt
  lands on the same side of any threshold and the per-tier configuration is never read.
- **SSE streaming still requires a Portkey key.** An operator with a valid provider key calling
  `/v1/chat/stream` still receives canned mock text. This is the most user-visible remaining
  defect.
- **`PORTKEY_*` settings still exist** in `app/config.py` and `.env.example`, and
  `ConfigHealthResponse` still carries a `portkey` field. Roughly 51 `portkey` references
  remain across 11 files in `app/`, including dashboard copy that still advertises the vendor.
- **Vector-backed retrieval is unimplemented.** `app/groundedness/vector_store.py` defines
  `FAISSVectorStore`, but the class imports no `faiss`, has an empty `__init__`, and its
  `similarity_search` returns a single hardcoded document. It is a named stub — which is why
  `faiss-cpu` was removed outright rather than moved to an extra.

### An honest note on the complexity scorer

`complexity.py` is primarily a **length-and-paste detector**, not a well-calibrated difficulty
model. Measured: a verbose but trivial 161-word log dump with no reasoning terms scores 0.6000
and routes to FRONTIER, while "Prove Gödel's second incompleteness theorem." scores 0.1100 and
routes to SLM. It is strictly better than the fixed constant it replaces — which routed
*everything* identically — but "real tiering" should not be read as "good tiering".
Calibration is deliberately deferred.

---

For the full architectural reference, including per-stage implementation detail, the settings
table, the lifespan breakdown, and the complete record of fixed issues, see
[`current_state.md`](current_state.md).
