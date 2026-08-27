"""Observability endpoints — /v1/metrics and /v1/metrics/accuracy."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.models import AccuracyMetrics, MetricsSummary

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/v1/metrics", response_model=MetricsSummary)
async def get_metrics(
    request: Request,
    window_minutes: Annotated[int, Query(ge=1, le=1440)] = 60,
) -> MetricsSummary:
    """Return aggregated metrics over the requested time window."""
    telemetry = getattr(request.app.state, "telemetry_logger", None)
    if not telemetry:
        raise HTTPException(status_code=500, detail="Telemetry logger not initialized")
    
    return await telemetry.get_metrics(window_minutes)

@router.get("/v1/metrics/accuracy", response_model=AccuracyMetrics)
async def get_accuracy_metrics(
    request: Request,
    window_days: Annotated[int, Query(ge=1, le=30)] = 7,
) -> AccuracyMetrics:
    """Return accuracy metrics based on operator overrides."""
    telemetry = getattr(request.app.state, "telemetry_logger", None)
    if not telemetry:
        raise HTTPException(status_code=500, detail="Telemetry logger not initialized")
    
    return await telemetry.get_accuracy_metrics(window_days)
