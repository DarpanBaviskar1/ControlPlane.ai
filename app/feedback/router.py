"""Feedback Loop endpoints — /v1/feedback/export and /v1/feedback/override."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.models import FeedbackRecord, OverrideRecord

logger = logging.getLogger(__name__)

router = APIRouter()

class OverrideRequest(BaseModel):
    request_id: str
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
