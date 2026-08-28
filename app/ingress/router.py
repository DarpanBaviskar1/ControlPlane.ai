"""Enterprise Ingress — POST /v1/chat handler.

Responsibilities:
- Parse and validate ChatRequest (Pydantic raises 422 automatically for
  missing/invalid fields; custom 422 for unrecognised use_case_profile).
- Assign UUID v4 request_id via dependency.
- Load UseCaseProfile via PolicyLoader.
- Gate traffic when PIIMaskingEngine startup validation failed (503).
- Wrap the full downstream pipeline in asyncio.wait_for(latency_budget_ms).
- Create a fresh RequestContext per request (no shared mutable state).
- In the finally block: call TelemetryLogger.record() and
  PIIMaskingEngine.discard_mapping(request_id).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.dependencies import get_policy_loader, get_pii_engine, request_id_dep
from app.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    RequestContext,
    TriageResult,
    UseCaseProfile,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helper — build a ChatResponse from a finalised RequestContext
# ---------------------------------------------------------------------------


def _build_response(ctx: RequestContext, latency_ms: int) -> ChatResponse:
    tr: TriageResult | None = ctx.triage_result
    if tr is None:
        return ChatResponse(
            request_id=ctx.request_id,
            triage_state="HARD_BLOCK",
            response=None,
            blocking_reason="INTERNAL_ERROR",
            latency_ms=latency_ms,
        )
    return ChatResponse(
        request_id=ctx.request_id,
        triage_state=tr.triage_state,
        response=tr.response_content,
        blocking_reason=tr.blocking_reason,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# POST /v1/chat
# ---------------------------------------------------------------------------


@router.post(
    "/v1/chat",
    response_model=ChatResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Gateway unavailable"},
        504: {"model": ErrorResponse, "description": "Latency budget exceeded"},
    },
)
async def handle_chat(
    body: ChatRequest,
    request: Request,
    request_id: Annotated[str, Depends(request_id_dep)],
    policy_loader=Depends(get_policy_loader),
) -> ChatResponse:
    """Main request handler.  Runs the full five-stage pipeline."""

    # ------------------------------------------------------------------
    # Resolve UseCaseProfile — 422 on unknown name
    # ------------------------------------------------------------------
    try:
        profile: UseCaseProfile = await policy_loader.get_profile(body.use_case_profile)
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "UNRECOGNISED_PROFILE",
                "detail": (
                    f"use_case_profile '{body.use_case_profile}' does not match "
                    "any configured profile"
                ),
                "request_id": request_id,
            },
        )

    # ------------------------------------------------------------------
    # 503 gate: PIIMaskingEngine startup validation
    # ------------------------------------------------------------------
    pii_engine = getattr(request.app.state, "pii_engine", None)
    if pii_engine is not None and not pii_engine.is_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "MASKING_INTEGRITY_FAILURE",
                "detail": "Gateway unavailable: PII masking startup validation failed",
                "request_id": request_id,
            },
        )

    # ------------------------------------------------------------------
    # Build per-request context — no shared mutable state
    # ------------------------------------------------------------------
    ctx = RequestContext(
        request_id=request_id,
        profile=profile,
        original_prompt=body.prompt,
        working_prompt=body.prompt,
        conversation_history=body.conversation_history,
        pipeline_start_ts=time.monotonic(),
    )

    telemetry = getattr(request.app.state, "telemetry_logger", None)
    langfuse_tracer = getattr(request.app.state, "langfuse_tracer", None)

    if langfuse_tracer:
        langfuse_tracer.start_trace(
            request_id=request_id,
            use_case_profile=profile.name,
            metadata={
                "original_prompt": body.prompt,
                "metadata": body.metadata,
                **(body.metadata.get("redteam_session_id") and {"redteam_session_id": body.metadata["redteam_session_id"]} or {})
            },
        )

    try:
        # ------------------------------------------------------------------
        # Enforce latency budget over the entire downstream pipeline
        # ------------------------------------------------------------------
        pipeline_fn = getattr(request.app.state, "pipeline_fn", None)
        timeout_s = profile.latency_budget_ms / 1000.0

        if pipeline_fn is not None:
            await asyncio.wait_for(pipeline_fn(ctx), timeout=timeout_s)
        else:
            # Stub: used before full pipeline is wired
            await asyncio.wait_for(_stub_pipeline(ctx), timeout=timeout_s)

    except asyncio.TimeoutError:
        elapsed = int((time.monotonic() - ctx.pipeline_start_ts) * 1000)
        logger.warning(
            "request_id=%s latency budget exceeded (%d ms > %d ms)",
            request_id,
            elapsed,
            profile.latency_budget_ms,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error_code": "LATENCY_BUDGET_EXCEEDED",
                "detail": (
                    f"Pipeline exceeded latency budget of {profile.latency_budget_ms} ms "
                    f"(elapsed {elapsed} ms)"
                ),
                "request_id": request_id,
            },
        )

    finally:
        # Always discard PII mapping
        if pii_engine is not None:
            pii_engine.discard_mapping(request_id)
        # Always record telemetry (fire-and-forget)
        if telemetry is not None and ctx.triage_result is not None:
            latency_for_tel = int((time.monotonic() - ctx.pipeline_start_ts) * 1000)
            await _record_telemetry(telemetry, ctx, latency_for_tel)
        
        # Flush Langfuse trace
        if langfuse_tracer:
            if ctx.triage_result:
                langfuse_tracer.set_metadata(
                    request_id,
                    triage_state=ctx.triage_result.triage_state,
                    blocking_reason=ctx.triage_result.blocking_reason
                )
            langfuse_tracer.flush_trace(request_id)

    latency_ms = int((time.monotonic() - ctx.pipeline_start_ts) * 1000)
    return _build_response(ctx, latency_ms)


# ---------------------------------------------------------------------------
# Stub pipeline (replaced by full wiring in Wave 12)
# ---------------------------------------------------------------------------


async def _stub_pipeline(ctx: RequestContext) -> None:
    from app.triage.gateway import TriageResult as TR  # noqa: PLC0415

    ctx.triage_result = TR(
        triage_state="PASS_AND_DELIVER",
        blocking_reason=None,
        response_content="[stub response]",
    )


# ---------------------------------------------------------------------------
# Telemetry helper
# ---------------------------------------------------------------------------


async def _record_telemetry(telemetry: object, ctx: RequestContext, latency_ms: int) -> None:
    from datetime import datetime, timezone

    from app.models import TelemetryRecord

    tr = ctx.triage_result
    if tr is None:
        return

    blocking_trigger: str | None = None
    if tr.triage_state == "HARD_BLOCK":
        if tr.blocking_reason:
            blocking_trigger = tr.blocking_reason
        elif ctx.p1_verdict and ctx.p1_verdict.blocking_trigger:
            blocking_trigger = ctx.p1_verdict.blocking_trigger

    record = TelemetryRecord(
        request_id=ctx.request_id,
        timestamp=datetime.now(tz=timezone.utc),
        use_case_profile=ctx.profile.name,
        p1_toxicity_verdict=ctx.p1_verdict.toxicity_verdict if ctx.p1_verdict else None,
        p1_injection_verdict=ctx.p1_verdict.injection_verdict if ctx.p1_verdict else None,
        p2_pii_count=ctx.p2_verdict.pii_count if ctx.p2_verdict else None,
        p3_clarity_verdict=ctx.p3_verdict,
        routing_decision=(
            ctx.routing_decision.classification if ctx.routing_decision else None
        ),
        selected_model_tier=(
            ctx.routing_decision.selected_tier if ctx.routing_decision else None
        ),
        routellm_score=(
            ctx.routing_decision.routellm_score if ctx.routing_decision else None
        ),
        groundedness_score=(
            ctx.audit_result.groundedness_score if ctx.audit_result else None
        ),
        groundedness_technique=(
            ctx.audit_result.technique if ctx.audit_result else None
        ),
        groundedness_unverified=(
            ctx.audit_result.is_unverified if ctx.audit_result else False
        ),
        final_triage_state=tr.triage_state,
        blocking_trigger=blocking_trigger,
        response_token_count=None,
        latency_ms=latency_ms,
        pii_masking_bypassed=(
            not ctx.profile.pii_masking_enabled
            if ctx.p2_verdict and ctx.p2_verdict.pii_count and ctx.p2_verdict.pii_count > 0
            else False
        ),
    )

    try:
        await telemetry.record(record)
    except Exception:  # noqa: BLE001
        logger.exception("Telemetry record failed for request_id=%s", ctx.request_id)
