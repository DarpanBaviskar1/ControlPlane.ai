"""Feedback Loop endpoints — /v1/feedback/export and /v1/feedback/override."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.models import FeedbackRecord, OverrideRecord
from app.policy.loader import PolicyLoader

logger = logging.getLogger(__name__)

router = APIRouter()

class OverrideRequest(BaseModel):
    request_id: str
    profile_name: str | None = None
    operator_id: str
    original_verdict: Literal["PASS", "SOFT_BLOCK", "HARD_BLOCK"]
    human_label: Literal["PASS", "SOFT_BLOCK", "HARD_BLOCK"]
    stated_reason: str

class OverrideResponse(BaseModel):
    override_id: str
    timestamp: datetime

@router.post("/v1/feedback/override", response_model=OverrideResponse)
async def submit_override(
    body: OverrideRequest,
    request: Request,
) -> OverrideResponse:
    """Submit a human-operator override of a triage decision."""
    telemetry = getattr(request.app.state, "telemetry_logger", None)
    if not telemetry:
        raise HTTPException(status_code=500, detail="Telemetry logger not initialized")
    
    now = datetime.now(tz=timezone.utc)
    
    override = OverrideRecord(
        request_id=body.request_id,
        operator_id=body.operator_id,
        timestamp=now,
        original_verdict=body.original_verdict,
        human_label=body.human_label,
        stated_reason=body.stated_reason,
    )
    
    await telemetry.record_override(override)
    
    # --- Langfuse Integration (Req. 6.7) ---
    langfuse_tracer = getattr(request.app.state, "langfuse_tracer", None)
    if langfuse_tracer:
        # We record the human label as a score (e.g. 1.0 for PASS, 0.0 for HARD_BLOCK)
        score_val = 1.0 if body.human_label == "PASS" else 0.0
        langfuse_tracer.add_evaluation_score(
            request_id=body.request_id,
            name="human_override",
            value=score_val,
            comment=f"Operator {body.operator_id} overriding {body.original_verdict} to {body.human_label}: {body.stated_reason}",
        )

    # --- Profile Sensitivity Adjustment (Req. 6.8) ---
    policy_loader: PolicyLoader | None = getattr(request.app.state, "policy_loader", None)
    if policy_loader and body.profile_name:
        try:
            profile = await policy_loader.get_profile(body.profile_name)
            # Simple heuristic: if human says PASS but system blocked, it's too sensitive.
            # If human says HARD_BLOCK but system passed, it's not sensitive enough.
            if body.original_verdict in ("HARD_BLOCK", "SOFT_BLOCK") and body.human_label == "PASS":
                # Loosen (e.g., lower complexity_threshold but bounded by sensitivity_floor)
                profile.complexity_threshold = max(
                    profile.sensitivity_floor, 
                    profile.complexity_threshold - profile.sensitivity_decrement
                )
                logger.info("Adjusted complexity_threshold down for %s to %.2f", profile.name, profile.complexity_threshold)
            elif body.original_verdict == "PASS" and body.human_label in ("HARD_BLOCK", "SOFT_BLOCK"):
                # Tighten
                profile.complexity_threshold = min(
                    1.0, 
                    profile.complexity_threshold + profile.sensitivity_decrement
                )
                logger.info("Adjusted complexity_threshold up for %s to %.2f", profile.name, profile.complexity_threshold)
        except KeyError:
            logger.warning("Could not adjust thresholds: profile '%s' not found", body.profile_name)
            
    # In a real system, override_id would be a UUID or database ID
    return OverrideResponse(
        override_id=f"override-{body.request_id}",
        timestamp=now,
    )

@router.get("/v1/feedback/export", response_model=list[FeedbackRecord])
async def export_feedback(
    request: Request,
) -> list[FeedbackRecord]:
    """Export all escalated and overridden cases."""
    telemetry = getattr(request.app.state, "telemetry_logger", None)
    if not telemetry:
        raise HTTPException(status_code=500, detail="Telemetry logger not initialized")
    
    return await telemetry.export_feedback()
