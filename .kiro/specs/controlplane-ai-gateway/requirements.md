# Requirements Document

## Introduction

ControlPlane.ai is a Python-based enterprise AI proxy gateway that sits between enterprise applications/agents and underlying large language models. It implements a five-stage request pipeline — Enterprise Ingress, Streaming Micro-Judges, Intelligent Model Router, Groundedness Audit, and Action Triage Gateway — to enforce safety, cost optimisation, and governance policies before delivering AI-generated responses to callers.

The system is designed to handle tens of thousands of interactions per week across multiple business use cases (e.g., Customer Chatbot, Internal Copilot), with configurable risk tolerance and latency budgets per use case profile. It is built on FastAPI, Portkey, RouteLLM, and LLM Guard by Protect AI.

---

## Glossary

- **Gateway**: The ControlPlane.ai FastAPI service that orchestrates the full five-stage pipeline.
- **Request**: An inbound HTTP call containing a user prompt and a use-case profile, received by the Gateway.
- **Use_Case_Profile**: A named configuration object that encodes risk tolerance, latency budget, and routing thresholds for a specific business context (e.g., `customer_chatbot`, `internal_copilot`).
- **Micro_Judge**: A lightweight parallel Safety Language Model inspector that evaluates a single safety dimension of an inbound prompt.
- **P1_Judge**: The Micro_Judge responsible for toxicity detection and prompt-injection detection.
- **P2_Judge**: The Micro_Judge responsible for PII (Personally Identifiable Information) detection and masking.
- **P3_Judge**: The Micro_Judge that signals prompt clarity for downstream routing.
- **PII_Masking_Engine**: The component (backed by LLM Guard) that replaces detected PII tokens with anonymised placeholders before the prompt reaches any LLM.
- **Model_Router**: The component (backed by RouteLLM) that classifies prompt complexity and selects a target model tier.
- **SLM**: Small Language Model — the low-cost model tier used for routine tasks.
- **Frontier_Model**: High-capability model tier (e.g., GPT-4, Claude Sonnet) used for complex reasoning tasks.
- **Groundedness_Auditor**: The streaming RAG evaluator that scores LLM responses against the Enterprise Vector Store.
- **Enterprise_Vector_Store**: The organisation's authoritative knowledge base used as the grounding reference corpus.
- **Action_Triage_Gateway**: The final decision component that maps audit scores and response metadata to one of four deterministic output states.
- **Triage_State**: One of four deterministic outcomes produced by the Action_Triage_Gateway: `PASS_AND_DELIVER`, `COMPRESS_AND_EDIT`, `ESCALATE_TO_HUMAN`, or `HARD_BLOCK`.
- **Hard_Block**: A terminal triage state that prevents any LLM response from being returned to the caller and returns a structured error payload instead.
- **Telemetry_Logger**: The component that records per-request observability data including routing decision, latency measurements, and safety trigger results.
- **Policy_Layer**: The configurable rules engine that maps Use_Case_Profile values to pipeline behaviour thresholds.
- **Groundedness_Score**: A floating-point value in [0.0, 1.0] produced by the Groundedness_Auditor representing how well the LLM response is supported by the Enterprise_Vector_Store.
- **Latency_Budget**: The maximum allowable end-to-end processing time in milliseconds defined per Use_Case_Profile.
- **Feedback_Loop**: The mechanism by which flagged or human-overridden cases are recorded and made available for model/policy improvement.

---

## Requirements

---

### Requirement 1: Enterprise Ingress and Use-Case Profile Routing

**User Story:** As an enterprise application developer, I want to submit a prompt together with a use-case profile so that the Gateway applies the correct risk tolerance, latency budget, and routing thresholds for my business context.

#### Acceptance Criteria

1. THE Gateway SHALL expose an HTTP POST endpoint at `/v1/chat` that accepts a JSON request body containing at minimum a `prompt` string field of 1 to 32,768 characters and a `use_case_profile` string field of 1 to 256 characters.
2. WHEN a Request is received with a non-empty `use_case_profile` value that matches an entry in the Policy_Layer, THE Gateway SHALL load the corresponding Use_Case_Profile configuration from the Policy_Layer before passing the Request to the pipeline.
3. IF a Request arrives with a `prompt` field that is absent or empty, THEN THE Gateway SHALL return an HTTP 422 response with an error body indicating which required field is missing or empty.
4. IF a Request arrives with a `use_case_profile` field that is absent, empty, or does not match any entry in the Policy_Layer, THEN THE Gateway SHALL return an HTTP 422 response with an error body indicating the unrecognised or missing profile value.
5. THE Policy_Layer SHALL support at minimum two built-in Use_Case_Profile configurations: `customer_chatbot` and `internal_copilot`.
6. WHERE a Use_Case_Profile defines a Latency_Budget expressed as a positive integer number of milliseconds between 1 and 300,000, THE Gateway SHALL enforce that budget as the total timeout for the end-to-end pipeline, returning an HTTP 504 response with an error body indicating the elapsed duration exceeded the configured budget if the budget is exceeded.
7. THE Gateway SHALL process concurrent Requests independently, ensuring the Use_Case_Profile, pipeline configuration, and intermediate pipeline state loaded for one Request are not readable or modifiable by any other concurrent Request.

---

### Requirement 2: Parallel Streaming Micro-Judges (Input Guardrails)

**User Story:** As a security and compliance officer, I want incoming prompts inspected in parallel by specialised safety judges so that toxicity, prompt injection, and PII violations are caught before any LLM is invoked.

#### Acceptance Criteria

1. WHEN a Request enters the Micro-Judge stage, THE Gateway SHALL invoke P1_Judge, P2_Judge, and P3_Judge concurrently before forwarding the prompt to the Model_Router.
2. WHEN P1_Judge evaluates a prompt, THE P1_Judge SHALL produce a binary verdict of `BLOCK` or `PASS` for toxicity and a separate binary verdict of `BLOCK` or `PASS` for adversarial prompt-injection using LLM Guard scanners.
3. WHEN P1_Judge produces a `BLOCK` verdict for either toxicity or prompt injection, THE Gateway SHALL immediately set the Triage_State to `HARD_BLOCK`, SHALL cancel any in-flight parallel judge executions, and SHALL NOT forward the prompt to the Model_Router or any downstream stage.
4. WHEN P2_Judge evaluates a prompt, THE P2_Judge SHALL detect PII tokens (including but not limited to Social Security Numbers, full names, email addresses, and phone numbers) in the prompt using LLM Guard PII scanners and SHALL report the count of detected PII tokens as its verdict.
5. WHEN P2_Judge reports a PII token count greater than zero and the Use_Case_Profile has `pii_masking_enabled` set to `true`, THE PII_Masking_Engine SHALL replace each detected token with a typed placeholder (e.g., `[SSN_REDACTED]`, `[NAME_REDACTED]`) before the prompt proceeds to the Model_Router.
6. IF the PII_Masking_Engine fails to mask one or more detected PII tokens, THEN THE Gateway SHALL set the Triage_State to `HARD_BLOCK` and SHALL NOT forward the prompt downstream.
7. WHEN P3_Judge evaluates a prompt, THE P3_Judge SHALL produce a clarity verdict of `CLEAR` or `AMBIGUOUS`, where `AMBIGUOUS` is assigned when the prompt contains 10 or fewer tokens or contains no identifiable main verb, and `CLEAR` otherwise.
8. IF any of P1_Judge, P2_Judge, or P3_Judge encounters an internal error during evaluation, THEN THE Gateway SHALL treat that judge's verdict as `BLOCK` (for P1) or the most restrictive safe default for P2 and P3, and SHALL record an error event in the Telemetry_Logger.
9. IF the Micro-Judge stage does not complete within the `inspection_timeout_ms` value defined in the Use_Case_Profile, THEN THE Gateway SHALL set the Triage_State to `ESCALATE_TO_HUMAN` and SHALL log a timeout event in the Telemetry_Logger.
10. THE Gateway SHALL record in the Telemetry_Logger, for every Request: the P1 toxicity verdict, the P1 injection verdict, the P2 PII token count, and the P3 clarity verdict, regardless of outcome.

---

### Requirement 3: Intelligent Model Router

**User Story:** As a cost optimisation engineer, I want prompt complexity to determine which model tier handles each request so that routine tasks are served by a cheaper SLM while only genuinely complex tasks consume Frontier_Model capacity.

#### Acceptance Criteria

1. WHEN a prompt passes all Micro_Judge checks, THE Model_Router SHALL evaluate prompt complexity using RouteLLM and classify the prompt as either `ROUTINE` or `COMPLEX`, using the P3_Judge clarity verdict as an additional input signal alongside RouteLLM's complexity score.
2. WHEN the Model_Router classifies a prompt as `ROUTINE`, THE Model_Router SHALL route the Request to the SLM tier.
3. WHEN the Model_Router classifies a prompt as `COMPLEX`, THE Model_Router SHALL route the Request to the Frontier_Model tier.
4. WHERE a Use_Case_Profile specifies a `complexity_threshold` value, THE Model_Router SHALL classify a prompt as `ROUTINE` when RouteLLM's confidence score is below that threshold and `COMPLEX` when at or above it; WHERE no `complexity_threshold` is specified in the Use_Case_Profile, THE Model_Router SHALL use a default threshold of 0.7.
5. THE Model_Router SHALL target a routing distribution where approximately 80% of Requests are classified as `ROUTINE` and approximately 20% are classified as `COMPLEX` under a representative mixed workload, within a tolerance of ±10 percentage points when measured over a minimum sample of 1,000 Requests.
6. THE Telemetry_Logger SHALL record the routing decision (`ROUTINE` or `COMPLEX`), the selected model tier, and the RouteLLM confidence score for every Request processed by the Model_Router.
7. IF the selected model tier is unavailable or returns an error, THEN THE Model_Router SHALL attempt the Request on the alternative model tier exactly once; IF the alternative model tier also fails, THEN THE Model_Router SHALL set the Triage_State to `HARD_BLOCK` and SHALL record a `MODEL_TIER_FAILURE` event in the Telemetry_Logger.

---

### Requirement 4: Groundedness Audit

**User Story:** As a data governance lead, I want every LLM response evaluated against the enterprise knowledge base so that hallucinated or unsupported content is detected before it reaches users.

#### Acceptance Criteria

1. WHEN an LLM response is received from either model tier, THE Groundedness_Auditor SHALL score the response against the Enterprise_Vector_Store and produce a Groundedness_Score in the range [0.0, 1.0].
2. WHEN the Groundedness_Auditor begins evaluating a response, THE Groundedness_Auditor SHALL produce an initial Groundedness_Score within 500 milliseconds of receiving the first token of the LLM response, without waiting for full response completion.
3. THE Groundedness_Auditor SHALL use at minimum one of the following detection techniques: embedding-based similarity, statistical anomaly detection, or AI-as-judge evaluation against retrieved documents.
4. THE Telemetry_Logger SHALL record the Groundedness_Score and the detection technique used for every Request processed by the Groundedness_Auditor.
5. IF the Enterprise_Vector_Store is unreachable during a Groundedness_Auditor evaluation, THEN THE Groundedness_Auditor SHALL set the Groundedness_Score to 0.0, SHALL flag the response as `UNVERIFIED` in the telemetry record, and SHALL log a `VECTOR_STORE_UNAVAILABLE` event in the Telemetry_Logger.
6. IF the Groundedness_Score produced by the Groundedness_Auditor is below 0.5, THEN THE Groundedness_Auditor SHALL emit a low-groundedness signal containing the Groundedness_Score and the detection technique used to the Action_Triage_Gateway before the response is delivered to the caller.

---

### Requirement 5: Action Triage Gateway (Output Decision Matrix)

**User Story:** As a platform operator, I want every AI response processed through a deterministic four-state decision matrix so that unsafe, hallucinated, or oversized responses are handled appropriately before delivery.

#### Acceptance Criteria

1. WHEN an LLM response reaches the Action_Triage_Gateway, THE Action_Triage_Gateway SHALL evaluate the response using the Groundedness_Score, response token count, and any upstream Triage_State set by earlier pipeline stages, and SHALL assign exactly one Triage_State per Request.
2. WHEN the Groundedness_Score is greater than 0.7 and the response token count does not exceed the token threshold defined in the Use_Case_Profile and no upstream stage has set a blocking Triage_State, THE Action_Triage_Gateway SHALL set the Triage_State to `PASS_AND_DELIVER`.
3. WHEN the response token count exceeds the token threshold defined in the Use_Case_Profile and the Groundedness_Score is greater than 0.7, THE Action_Triage_Gateway SHALL set the Triage_State to `COMPRESS_AND_EDIT`, SHALL apply response compression before delivery, and SHALL ensure the compressed response token count does not exceed the Use_Case_Profile token threshold.
4. WHEN the Groundedness_Score is between 0.5 and 0.7 inclusive and no higher-priority Triage_State applies, THE Action_Triage_Gateway SHALL set the Triage_State to `ESCALATE_TO_HUMAN` and SHALL log the full Request and response payload for human review.
5. WHEN P3_Judge produced an `AMBIGUOUS` verdict and no higher-priority Triage_State applies, THE Action_Triage_Gateway SHALL set the Triage_State to `ESCALATE_TO_HUMAN` and SHALL log the full Request and response payload for human review.
6. WHEN the Groundedness_Score is below 0.5, THE Action_Triage_Gateway SHALL set the Triage_State to `HARD_BLOCK` and SHALL return a structured error response to the caller containing the Triage_State value and the blocking reason, and SHALL NOT include any LLM-generated content in the response.
7. IF any upstream pipeline stage has already set the Triage_State to `HARD_BLOCK`, THEN THE Action_Triage_Gateway SHALL preserve that state and SHALL NOT override it with a lower-severity state.
8. THE Action_Triage_Gateway SHALL apply Triage_State evaluation rules in the following priority order: `HARD_BLOCK` (highest) → `ESCALATE_TO_HUMAN` → `COMPRESS_AND_EDIT` → `PASS_AND_DELIVER` (lowest), ensuring higher-severity states always take precedence.
9. THE Telemetry_Logger SHALL record the final Triage_State, the Groundedness_Score, and the response token count for every Request processed by the Action_Triage_Gateway.

---

### Requirement 6: Telemetry Logging and Observability

**User Story:** As a platform operator, I want every request to produce a structured telemetry record so that I can monitor safety triggers, routing decisions, latency overhead, and system health in a single observability feedback loop.

#### Acceptance Criteria

1. THE Telemetry_Logger SHALL produce one structured log record per Request containing: request ID, timestamp, Use_Case_Profile name, P1/P2/P3 judge verdicts, routing decision, selected model tier, Groundedness_Score, final Triage_State, and end-to-end latency in milliseconds.
2. THE Telemetry_Logger SHALL write each log record within 50 milliseconds of the Request completing the pipeline, and SHALL NOT add more than 5 milliseconds of observable latency to the response delivery to the caller.
3. THE Gateway SHALL assign a unique request_id (UUID v4) to every inbound Request.
4. THE Telemetry_Logger SHALL include the request_id assigned by the Gateway in every log record associated with that Request.
5. WHEN a Request results in a `HARD_BLOCK` Triage_State, THE Telemetry_Logger SHALL additionally record the specific blocking trigger (e.g., `P1_TOXICITY`, `P1_INJECTION`, `LOW_GROUNDEDNESS`) in the log record.
6. WHEN a Request results in an `ESCALATE_TO_HUMAN` Triage_State, THE Telemetry_Logger SHALL store the full prompt (post-masking) and full LLM response in the escalation log for human review.
7. THE Telemetry_Logger SHALL expose a summary metrics endpoint at `/v1/metrics` returning aggregate counts of each Triage_State, average Groundedness_Score, and routing distribution; the time window SHALL be configurable with a default of 60 minutes and a valid range of 1 minute to 1,440 minutes.
8. THE Feedback_Loop SHALL make escalated cases and human-reviewer-overridden cases — where an operator has manually changed the system's Triage_State verdict after the fact — available via a structured export at `/v1/feedback/export` so that they can be used to improve detection quality and policy thresholds.
9. IF the Telemetry_Logger fails to write a log record, THEN THE Telemetry_Logger SHALL complete the failure within 5 seconds, SHALL NOT retry more than 3 times, and SHALL NOT propagate the failure to the caller's response.

---

### Requirement 7: Configurable Policy Layer and Use-Case Governance

**User Story:** As a governance architect, I want pipeline behaviour to be driven by a configurable policy layer so that risk tolerance, routing thresholds, and latency budgets can vary by use case, geography, or organisational risk appetite without code changes.

#### Acceptance Criteria

1. THE Policy_Layer SHALL store Use_Case_Profile configurations in a YAML or JSON file that the Gateway reloads without a service restart when the file is modified, and SHALL apply the updated configuration to Requests received after the reload completes within 5 seconds of the file modification.
2. THE Policy_Layer SHALL support the following configurable fields per Use_Case_Profile: `latency_budget_ms` (positive integer, 1–300,000), `complexity_threshold` (float, [0.0, 1.0]), `token_compression_threshold` (positive integer), `groundedness_pass_threshold` (float, [0.0, 1.0]), `inspection_timeout_ms` (positive integer, 1–60,000), `pii_masking_enabled` (boolean), and `human_escalation_enabled` (boolean).
3. WHERE `pii_masking_enabled` is set to `false` in a Use_Case_Profile, THE PII_Masking_Engine SHALL skip masking for Requests under that profile and THE Telemetry_Logger SHALL record a `PII_MASKING_BYPASSED` event.
4. WHERE `human_escalation_enabled` is set to `false` in a Use_Case_Profile, THE Action_Triage_Gateway SHALL promote `ESCALATE_TO_HUMAN` decisions to `HARD_BLOCK` for Requests under that profile.
5. IF a Use_Case_Profile fails type or range validation on load, THEN THE Gateway SHALL reject that profile with an error message identifying the invalid field name, the provided value, and the expected type and range, and SHALL continue operating with the previously loaded valid configuration.
6. THE Gateway SHALL support at minimum 10 simultaneously active Use_Case_Profile configurations, and processing latency for any single Request SHALL NOT increase by more than 2 milliseconds compared to a baseline measured with a single active profile.
7. WHERE a Use_Case_Profile specifies a `groundedness_pass_threshold` value, THE Action_Triage_Gateway SHALL use that value in place of the system default of 0.9 when evaluating whether to assign a `PASS_AND_DELIVER` Triage_State.

---

### Requirement 8: False Positive / False Negative Measurement and Reporting

**User Story:** As a quality assurance lead, I want the system to define, measure, and report false positive and false negative rates for each safety judge so that the trustworthiness of the system can be quantified and improved over time.

#### Acceptance Criteria

1. THE Telemetry_Logger SHALL record ground-truth labels for Requests that have been reviewed and overridden by a human operator, storing both the system's original verdict and the human-assigned correct label, where valid label values are `PASS`, `SOFT_BLOCK`, and `HARD_BLOCK`.
2. THE Gateway SHALL expose a `/v1/metrics/accuracy` endpoint that returns false positive rate, false negative rate, and F1 score for P1_Judge (toxicity), P1_Judge (injection), and P2_Judge (PII) computed over an evaluation window of 1 to 30 days.
3. WHEN a human operator overrides a `HARD_BLOCK`, `SOFT_BLOCK`, or `ALLOW` decision via the `/v1/feedback/override` endpoint, THE Feedback_Loop SHALL record the override with the operator ID, timestamp, and stated reason before updating the ground-truth label.
4. IF a request to `/v1/metrics/accuracy` specifies an evaluation window outside the range of 1 to 30 days, THEN THE endpoint SHALL return an HTTP 422 response with an error body indicating the valid range; otherwise, THE endpoint SHALL return accuracy metrics within 2 seconds for valid window values.
5. THE Telemetry_Logger SHALL retain raw telemetry records, including any override records captured in criterion 3, for a minimum of 90 days to support retrospective accuracy analysis.

---

### Requirement 9: Round-Trip Prompt Integrity

**User Story:** As a security engineer, I want to verify that PII masking and any response compression applied by the pipeline are invertible and do not corrupt prompt or response content so that information fidelity is preserved throughout the pipeline.

#### Acceptance Criteria

1. THE PII_Masking_Engine SHALL maintain a per-Request mapping from each placeholder token to its original PII value, and SHALL discard that mapping once the final response for that Request has been delivered to the caller.
2. IF PII masking has been applied to a prompt, THEN THE PII_Masking_Engine SHALL restore all placeholder tokens in the masked prompt using the per-Request mapping and the resulting prompt SHALL be byte-for-byte identical to the original input after whitespace normalisation.
3. WHEN `COMPRESS_AND_EDIT` is applied to a response, THE Action_Triage_Gateway SHALL ensure the compressed output contains no statements, claims, or named entities that were not present in the original response.
4. WHEN the Gateway starts up, THE Gateway SHALL execute a round-trip validation suite of five synthetic test prompts containing known PII patterns against the PII_Masking_Engine.
5. IF any prompt in the startup round-trip validation suite fails to produce a byte-for-byte identical result after whitespace normalisation, THEN THE Gateway SHALL log a `MASKING_INTEGRITY_FAILURE` event, SHALL reject all incoming requests with an error indicating the gateway is unavailable, and SHALL resume serving production traffic only after a subsequent execution of the full validation suite passes without failures.
