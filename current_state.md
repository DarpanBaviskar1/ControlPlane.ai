# Control Plane AI - Current State and Architecture

## Overview
**ControlPlane.ai** is an Enterprise AI Proxy Gateway designed with a robust five-stage safety pipeline. It mediates requests between clients and Large Language Models (LLMs) to ensure safety, policy compliance, groundedness, and security, utilizing an intricate system of micro-judges, vector stores, and output validators.

This document describes the current detailed state of the project, documenting every major feature, how they interact, and recent infrastructural changes.

---

## The 5-Stage Safety Pipeline

At the core of the ControlPlane Gateway is the pipeline engine defined in `app/main.py`. This pipeline processes each request sequentially through five safety checkpoints:

### 1. Orchestrator (Micro-Judges & Policy Layer)
The pipeline begins by validating requests against static policies and scanning for immediate risks.
*   **PII Masking Engine**: Scans incoming prompts and masks Personally Identifiable Information (PII) before the LLM sees it. The engine performs a startup validation suite; if it fails, the gateway enters a `503 Unavailable` state to prevent data leakage.
*   **Policy Loader**: Runs continuously with a watchdog. It loads security policies dynamically from the filesystem without requiring gateway restarts.
*   **P1 & P3 Judges**: CPU-bound scanners executed concurrently.
    *   **P1 Judges**: Includes LLM Guard Toxicity and Prompt Injection detection models to immediately flag hostile user prompts.
    *   **P3 Judges**: Uses `spaCy` NLP models to evaluate the structural clarity and semantic ambiguity (`p3_verdict`) of user prompts.
*   **Verdict**: If the Orchestrator assigns a `HARD_BLOCK` state (due to policy violation or malicious intent), the request bypasses downstream execution and goes straight to Triage.

### 2. Model Router (RouteLLM Controller)
Once a prompt passes initial safety checks, the gateway decides which LLM is most appropriate.
*   **RouteLLM Controller**: Currently integrated and initialized at startup. It evaluates the complexity of the `working_prompt`, the user's `profile`, and the `p3_clarity` score to make a routing decision.
*   **LLM Execution**: The router forwards the request to the chosen LLM backend (e.g., via OpenAI API, Portkey, etc.) and awaits the response.
*   **Output Validation (Guardrails AI)**: Once the LLM generates a response, it is intercepted by Guardrails AI validators (`app.judges.output_validator`).
    *   If validation fails but is fixable, Guardrails applies a patch (`fix` action).
    *   If validation fails and is unfixable, it results in a `filter` or `exception`, triggering an immediate `HARD_BLOCK` upstream triage state.

### 3. Groundedness Auditor
For enterprise reliability, the LLM's response is audited against known facts.
*   **FAISS Vector Store**: Utilized for similarity search and contextual grounding.
*   **Auditor**: Evaluates the LLM's response against the `FAISSVectorStore` and generates a `groundedness_score`. A low score indicates hallucination.

### 3b. Worldsense Multi-Turn Agentic Oversight
For continuous, multi-turn conversations, evaluating single prompts is insufficient. This stage oversees consequence chains across the conversation history.
*   **Integration**: Integrated via `app.oversight.worldsense_oversight`.
*   **SDK vs Heuristic**: It attempts to use the `worldsense` SDK. However, due to recent dependency streamlining (removing `worldsense` from `pyproject.toml` dependencies), the system currently falls back to a **Heuristic Evaluator**.
*   **Heuristic Fallback**: Evaluates the conversation history for single-turn risk keywords (e.g., "override admin", "drop table") resulting in `RISK_DETECTED`, and multi-turn consequence patterns (e.g., repetitive escalation requests) resulting in `CONSEQUENCE_ALERT`.
*   **Budget & Timeouts**: This stage must complete within `WORLDSENSE_TIMEOUT_MS` (default 300ms). If it exceeds the budget, it fails open to a `RISK_DETECTED` state.

### 4. Triage Gateway
The final stage collects verdicts from all previous stages and formulates the final action.
*   **Triage Gateway Evaluator**: Processes `groundedness_score`, `upstream_triage_state`, `p3_clarity`, and user `profile` to determine the final `triage_state`.
*   **Response Compression**: If the gateway decides the response is too verbose and yields a `COMPRESS_AND_EDIT` state, the `app.triage.compressor` compresses the text based on the user profile's token threshold.
*   **Unmasking**: Finally, the PII Masking Engine reverses the initial masking, replacing placeholders with the original sensitive data before the response is returned to the user.

---

## Observability & Telemetry

To monitor safety actions, latencies, and user behaviors, observability tools run continuously:
*   **Langfuse Tracer**: Tracks execution traces, prompt chains, and LLM call metadata for debugging and audit logs.
*   **Telemetry Logger**: Consumes metrics asynchronously, recording decisions made by the triage gateway, routing assignments, and latency budgets.

---

## Ancillary Endpoints & Routers
Beyond the core AI pipeline, the application registers several routers to manage operational capabilities:
*   **Chat / Ingress Router (`app.ingress.router`)**: Primary endpoint for client LLM interaction.
*   **Metrics Router (`app.telemetry.router`)**: Exposes telemetry and Prometheus metrics.
*   **Feedback Router (`app.feedback.router`)**: Collects user feedback (e.g., thumbs up/down, corrections) for continuous model alignment.
*   **Redteam Router (`app.redteam.router`)**: Dedicated endpoint for simulated adversarial attacks, allowing the security team to continuously validate the pipeline's robustness.

---

## Recent Infrastructural Updates & Dependency States

During recent maintenance, several issues were addressed regarding the Python environment and dependencies:

1. **Virtual Environment Isolation (FastAPI)**:
   The project strictly isolates its dependencies in a local `.venv`. A previous issue where `fastapi` was incorrectly installed in the global `AppData` namespace—causing `ModuleNotFoundError` during IDE type-checking—was resolved by explicitly installing it inside the `.venv`.
2. **Dependency Constraint Relaxation (`pyproject.toml`)**:
   Strict version pinning (`==`) in `pyproject.toml` caused build failures. Notably, strict pinning of `pydantic==2.7.1` forced the installation of `pydantic-core==2.18.2`, which lacks pre-compiled wheels for Python 3.13 and failed to build without a Rust compiler.
   *   **Resolution**: Version pinning was relaxed (using `>=`) for core dependencies like `fastapi`, `uvicorn`, `pydantic`, `pytest`, etc.
3. **Worldsense Dependency**:
   The `worldsense` package was explicitly removed from `pyproject.toml`. The system now relies gracefully on its heuristic rule-based fallback for agentic oversight.
4. **Egg-Info Rebuild**:
   All `.egg-info` generated files were updated and committed atomically to reflect the changes in dependencies.

---

## Summary
ControlPlane.ai is currently a highly modular, multi-layered proxy that prioritizes security and context-awareness. It robustly handles everything from basic token generation to complex PII masking, open-source AI guardrails, and deterministic heuristic oversight, backed by modern tracing and telemetry tools.
