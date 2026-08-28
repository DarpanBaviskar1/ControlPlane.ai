"""Shared Pydantic data models for the ControlPlane.ai Enterprise AI Proxy Gateway.

All field-level constraints are encoded as Pydantic Field validators to enforce
correctness at parse time and surface clear validation errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Triage state literals
# ---------------------------------------------------------------------------
TriageState = Literal[
    "PASS_AND_DELIVER",
    "COMPRESS_AND_EDIT",
    "ESCALATE_TO_HUMAN",
    "HARD_BLOCK",
]

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ConversationTurn(BaseModel):
    """One turn of conversation history for multi-turn agentic oversight (Req. 12.7)."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=32_768)


class ChatRequest(BaseModel):
    """Inbound request body for POST /v1/chat."""

    prompt: str = Field(min_length=1, max_length=32_768)
    use_case_profile: str = Field(min_length=1, max_length=256)
    metadata: dict[str, str] = Field(default_factory=dict)
    # Optional multi-turn history for Worldsense agentic oversight (Req. 12.7)
    conversation_history: list[ConversationTurn] = Field(default_factory=list, max_length=50)


class ChatResponse(BaseModel):
    """Outbound response body for POST /v1/chat (all four triage states)."""

    request_id: str  # UUID v4
    triage_state: TriageState
    response: str | None  # None for HARD_BLOCK
    blocking_reason: str | None  # populated for HARD_BLOCK
    latency_ms: int


class ErrorResponse(BaseModel):
    """Standard error envelope for 4xx/5xx responses."""

    error_code: str
    detail: str
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Use-case profile
# ---------------------------------------------------------------------------


class UseCaseProfile(BaseModel):
    """Per-profile pipeline configuration loaded from the Policy Layer."""

    name: str = Field(min_length=1, max_length=256)
    latency_budget_ms: int = Field(ge=1, le=300_000)
    complexity_threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    token_compression_threshold: int = Field(ge=1)
    groundedness_pass_threshold: float = Field(ge=0.0, le=1.0, default=0.9)
    inspection_timeout_ms: int = Field(ge=1, le=60_000)
    pii_masking_enabled: bool = True
    human_escalation_enabled: bool = True
    # Worldsense agentic oversight (Req. 12.1, 12.8)
    agentic_oversight_enabled: bool = False
    # Obot tool governance (Req. 11.5)
    allowed_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    max_tool_calls_per_request: int = Field(ge=1, default=10)
    # Feedback loop sensitivity floor — minimum value a threshold can be reduced to (Req. 6.8)
    sensitivity_floor: float = Field(ge=0.0, le=1.0, default=0.3)
    # Sensitivity decrement step applied on each human override (Req. 6.8)
    sensitivity_decrement: float = Field(ge=0.0, le=0.5, default=0.02)
    # Phase 3 — Semantic Cache (Req. 1.1)
    cache_enabled: bool = False
    cache_ttl_seconds: int = Field(ge=1, default=300)
    cache_similarity_threshold: float = Field(ge=0.0, le=1.0, default=0.92)
    # Phase 3 — GLiNER custom entity masking (Req. 3.1)
    custom_entity_terms: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Judge verdict dataclasses
# ---------------------------------------------------------------------------


@dataclass
class P1Verdict:
    toxicity_verdict: Literal["BLOCK", "PASS"]
    injection_verdict: Literal["BLOCK", "PASS"]

    @property
    def is_blocked(self) -> bool:
        return self.toxicity_verdict == "BLOCK" or self.injection_verdict == "BLOCK"

    @property
    def blocking_trigger(self) -> str | None:
        if self.toxicity_verdict == "BLOCK":
            return "P1_TOXICITY"
        if self.injection_verdict == "BLOCK":
            return "P1_INJECTION"
        return None


@dataclass
class P2Verdict:
    pii_count: int
    masked_prompt: str | None  # populated if pii_count > 0
    placeholder_map: dict[str, str] = field(default_factory=dict)


@dataclass
class GuardrailsVerdict:
    """Output of the Guardrails AI output validation chain (Req. 2.11-13)."""

    passed: bool
    # on_fail action that fired, or None if all validators passed
    action: Literal["exception", "filter", "fix"] | None = None
    # Name of the validator that triggered the action
    triggered_validator: str | None = None
    # Fixed output when action == 'fix'; None otherwise
    fixed_output: str | None = None


# Worldsense oversight verdict literals (Req. 12.2)
WorldsenseVerdictLiteral = Literal["SAFE", "RISK_DETECTED", "CONSEQUENCE_ALERT"]


@dataclass
class WorldsenseVerdict:
    """Output of the Worldsense multi-turn agentic oversight stage (Req. 12)."""

    verdict: WorldsenseVerdictLiteral
    # Turn index where risk was first detected (for RISK_DETECTED / CONSEQUENCE_ALERT)
    risk_turn_index: int | None = None
    # Human-readable description of the detected risk or consequence
    risk_description: str | None = None


@dataclass
class RoutingDecision:
    classification: Literal["ROUTINE", "COMPLEX"]
    selected_tier: Literal["SLM", "FRONTIER"]
    routellm_score: float
    response: str | None
    triage_state: TriageState | None  # set only on failure


@dataclass
class AuditResult:
    groundedness_score: float  # [0.0, 1.0]
    technique: str  # "embedding_similarity" or "nli_embedding_similarity"
    is_unverified: bool
    # Phase 3 — NLI groundedness auditor (Req. 2.1)
    nli_label: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"] | None = None


@dataclass
class TriageResult:
    triage_state: TriageState
    blocking_reason: str | None
    response_content: str | None


# ---------------------------------------------------------------------------
# Pipeline request context (per-request mutable state)
# ---------------------------------------------------------------------------


@dataclass
class RequestContext:
    """Carries all mutable pipeline state for a single request.

    Created fresh in the Ingress layer; passed through all stages; destroyed
    after response delivery.  No instance is shared across concurrent requests.
    """

    request_id: str
    profile: UseCaseProfile
    original_prompt: str
    working_prompt: str  # may be masked by P2
    conversation_history: list[ConversationTurn] = field(default_factory=list)
    placeholder_map: dict[str, str] = field(default_factory=dict)
    p1_verdict: P1Verdict | None = None
    p2_verdict: P2Verdict | None = None
    p3_verdict: Literal["CLEAR", "AMBIGUOUS"] | None = None
    guardrails_verdict: GuardrailsVerdict | None = None
    worldsense_verdict: WorldsenseVerdict | None = None
    # Obot tool call counter (Req. 11.6)
    tool_call_count: int = 0
    routing_decision: RoutingDecision | None = None
    llm_response: str | None = None
    audit_result: AuditResult | None = None
    triage_result: TriageResult | None = None
    pipeline_start_ts: float = 0.0
    upstream_triage_state: TriageState | None = None
    # Langfuse trace ID equals the request_id (Req. 6.3)
    langfuse_trace_id: str | None = None
    # Red team session marker (Req. 10.8)
    redteam_session_id: str | None = None


# ---------------------------------------------------------------------------
# Telemetry models
# ---------------------------------------------------------------------------


class TelemetryRecord(BaseModel):
    """One structured log record per request written by the Telemetry Logger."""

    request_id: str
    timestamp: datetime
    use_case_profile: str
    p1_toxicity_verdict: Literal["BLOCK", "PASS"] | None = None
    p1_injection_verdict: Literal["BLOCK", "PASS"] | None = None
    p2_pii_count: int | None = None
    p3_clarity_verdict: Literal["CLEAR", "AMBIGUOUS"] | None = None
    routing_decision: Literal["ROUTINE", "COMPLEX"] | None = None
    selected_model_tier: Literal["SLM", "FRONTIER"] | None = None
    routellm_score: float | None = None
    groundedness_score: float | None = None
    groundedness_technique: str | None = None
    groundedness_unverified: bool = False
    final_triage_state: str
    blocking_trigger: str | None = None  # e.g. "P1_TOXICITY", "LOW_GROUNDEDNESS"
    response_token_count: int | None = None
    latency_ms: int
    pii_masking_bypassed: bool = False
    # Phase 3 — Semantic Cache telemetry (Req. 1.2, 6.6)
    cache_hit: bool = False
    # Phase 3 — NLI groundedness telemetry (Req. 2.2, 6.6)
    nli_label: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"] | None = None


class OverrideRecord(BaseModel):
    """Human-operator override of a triage decision, written via /v1/feedback/override."""

    request_id: str
    operator_id: str
    timestamp: datetime
    original_verdict: Literal["PASS", "SOFT_BLOCK", "HARD_BLOCK"]
    human_label: Literal["PASS", "SOFT_BLOCK", "HARD_BLOCK"]
    stated_reason: str


class FeedbackRecord(BaseModel):
    """Export record combining telemetry + override for /v1/feedback/export."""

    telemetry: TelemetryRecord
    override: OverrideRecord | None = None
    human_label: Literal["PASS", "SOFT_BLOCK", "HARD_BLOCK"] | None = None


# ---------------------------------------------------------------------------
# Metrics models
# ---------------------------------------------------------------------------


class TriageStateCounts(BaseModel):
    PASS_AND_DELIVER: int = 0
    COMPRESS_AND_EDIT: int = 0
    ESCALATE_TO_HUMAN: int = 0
    HARD_BLOCK: int = 0


class RoutingDistribution(BaseModel):
    ROUTINE: float = 0.0
    COMPLEX: float = 0.0


class MetricsSummary(BaseModel):
    window_minutes: int
    total_requests: int
    triage_state_counts: TriageStateCounts
    average_groundedness_score: float
    routing_distribution: RoutingDistribution


class JudgeAccuracy(BaseModel):
    false_positive_rate: float
    false_negative_rate: float
    f1_score: float


class AccuracyMetrics(BaseModel):
    window_days: int
    p1_toxicity: JudgeAccuracy
    p1_injection: JudgeAccuracy
    p2_pii: JudgeAccuracy


# ---------------------------------------------------------------------------
# Policy file schema helper
# ---------------------------------------------------------------------------


class PolicyFileSchema(BaseModel):
    """Validates the top-level structure of a YAML/JSON policy file."""

    profiles: list[UseCaseProfile]


# ---------------------------------------------------------------------------
# UUID v4 validation helper
# ---------------------------------------------------------------------------

_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def is_valid_uuid4(value: str) -> bool:
    return bool(_UUID4_RE.match(value))
