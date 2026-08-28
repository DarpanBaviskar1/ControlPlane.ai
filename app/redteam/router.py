"""Red Team Runner API router."""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, Request

from app.redteam.runner import get_runner, RedTeamReport

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/redteam", tags=["redteam"])


@router.post("/run", response_model=RedTeamReport)
async def run_redteam(request: Request) -> RedTeamReport:
    """Trigger a new automated red team testing run."""
    runner = get_runner()
    
    if getattr(runner, "_running", False):
        raise HTTPException(
            status_code=409,
            detail="A red team run is already in progress",
        )
        
    tracer = getattr(request.app.state, "langfuse_tracer", None)
    
    # Fire and forget or await? Usually we'd want this to be async background,
    # but for simplicity of the prototype we await it (or the client can use a longer timeout).
    # Since this is a testing endpoint, returning the report is fine.
    try:
        report = await runner.run(tracer=tracer)
        return report
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/report", response_model=RedTeamReport)
async def get_latest_report() -> RedTeamReport:
    """Fetch the results of the last red team run."""
    runner = get_runner()
    report = runner.last_report
    
    if not report:
        raise HTTPException(
            status_code=404,
            detail="No red team run has been completed yet",
        )
        
    return report
