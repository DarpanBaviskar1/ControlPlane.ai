"""Micro-Judge Stage Orchestrator.

Runs P1, P2, and P3 judges concurrently via asyncio.gather inside
asyncio.wait_for(timeout=inspection_timeout_ms).

Short-circuit rules (applied in order after all judges return):
1. P1 BLOCK → set upstream_triage_state=HARD_BLOCK immediately.
2. P2 masking failure (pii_count=sys.maxsize after masking attempt) →
   set upstream_triage_state=HARD_BLOCK.
3. P2 pii_count>0 + masking enabled → apply masking; on engine failure →
   HARD_BLOCK.
4. P2 pii_masking_enabled=False + pii_count>0 → log PII_MASKING_BYPASSED.

On asyncio.TimeoutError → set upstream_triage_state=ESCALATE_TO_HUMAN
and log a timeout event (requirement 2.9).

Judge failure isolation (requirement 2.8):
- P1 exception → BLOCK/BLOCK verdict.
- P2 exception → pii_count=sys.maxsize (handled by PIIMaskingEngine.mask error path).
- P3 exception → AMBIGUOUS.
All failures are logged as error events.

After a successful run, ctx is populated with:
  ctx.p1_verdict, ctx.p2_verdict, ctx.p3_verdict,
  ctx.working_prompt (masked if PII found),
  ctx.placeholder_map,
  ctx.upstream_triage_state (set only on HARD_BLOCK or ESCALATE_TO_HUMAN).
"""

from __future__ import annotations

import asyncio
import logging
import sys

from app.judges.p1_judge import p1_judge
from app.judges.p2_judge import p2_judge
from app.judges.p3_judge import p3_judge
from app.models import P1Verdict, P2Verdict, RequestContext

logger = logging.getLogger(__name__)

# Valid blocking trigger identifiers referenced by telemetry
_TRIGGER_INSPECTION_TIMEOUT = "INSPECTION_TIMEOUT"
_TRIGGER_PII_MASKING_FAILURE = "PII_MASKING_FAILURE"


async def run_micro_judges(ctx: RequestContext, pii_engine: object) -> None:
    """Execute the Micro-Judge stage, populating ctx in-place.

    Concurrently runs P1/P2/P3 inside inspection_timeout_ms, then applies
    short-circuit logic and PII masking before returning.
    """
    timeout_s = ctx.profile.inspection_timeout_ms / 1000.0

    try:
        p1_verdict, p2_verdict, p3_verdict = await asyncio.wait_for(
            _run_all_judges(ctx, pii_engine),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "request_id=%s micro-judge timeout after %dms",
            ctx.request_id,
            ctx.profile.inspection_timeout_ms,
        )
        # Apply safe defaults so telemetry still has values
        ctx.p1_verdict = P1Verdict(toxicity_verdict="PASS", injection_verdict="PASS")
        ctx.p2_verdict = P2Verdict(pii_count=0, masked_prompt=None)
        ctx.p3_verdict = "AMBIGUOUS"
        ctx.upstream_triage_state = "ESCALATE_TO_HUMAN"
        return

    # Store verdicts on context
    ctx.p1_verdict = p1_verdict
    ctx.p2_verdict = p2_verdict
    ctx.p3_verdict = p3_verdict

    # ------------------------------------------------------------------
    # Short-circuit 1: P1 BLOCK
    # ------------------------------------------------------------------
    if p1_verdict.is_blocked:
        logger.info(
            "request_id=%s P1 BLOCK trigger=%s",
            ctx.request_id,
            p1_verdict.blocking_trigger,
        )
        ctx.upstream_triage_state = "HARD_BLOCK"
        return

    # ------------------------------------------------------------------
    # Short-circuit 2: P2 masking
    # ------------------------------------------------------------------
    if p2_verdict.pii_count > 0:
        if not ctx.profile.pii_masking_enabled:
            # Log bypass but continue with original prompt
            logger.info(
                "request_id=%s PII_MASKING_BYPASSED pii_count=%d",
                ctx.request_id,
                p2_verdict.pii_count,
            )
        else:
            # P2 already ran the masking via pii_engine.mask() in p2_judge;
            # pii_count==sys.maxsize signals an engine error.
            if p2_verdict.pii_count == sys.maxsize:
                logger.error(
                    "request_id=%s PII masking failure — HARD_BLOCK",
                    ctx.request_id,
                )
                ctx.upstream_triage_state = "HARD_BLOCK"
                # Override blocking trigger so telemetry captures it
                ctx.p1_verdict = P1Verdict(
                    toxicity_verdict=p1_verdict.toxicity_verdict,
                    injection_verdict=p1_verdict.injection_verdict,
                )
                return

            if p2_verdict.masked_prompt is not None:
                ctx.working_prompt = p2_verdict.masked_prompt
                ctx.placeholder_map = p2_verdict.placeholder_map

    # All checks passed — pipeline continues to the Model Router


async def _run_all_judges(
    ctx: RequestContext, pii_engine: object
) -> tuple[P1Verdict, P2Verdict, "Literal['CLEAR', 'AMBIGUOUS']"]:  # type: ignore[name-defined]
    """Run P1, P2, P3 concurrently; apply per-judge failure isolation."""
    results = await asyncio.gather(
        _safe_p1(ctx.working_prompt),
        _safe_p2(ctx.working_prompt, ctx.profile, pii_engine, ctx.request_id),
        _safe_p3(ctx.working_prompt),
        return_exceptions=False,
    )
    return results[0], results[1], results[2]  # type: ignore[return-value]


async def _safe_p1(prompt: str) -> P1Verdict:
    try:
        return await p1_judge(prompt)
    except Exception as exc:
        logger.error("P1 judge raised unexpectedly: %s — defaulting BLOCK/BLOCK", exc)
        return P1Verdict(toxicity_verdict="BLOCK", injection_verdict="BLOCK")


async def _safe_p2(
    prompt: str, profile: object, pii_engine: object, request_id: str
) -> P2Verdict:
    try:
        return await p2_judge(prompt, profile, pii_engine, request_id)
    except Exception as exc:
        logger.error("P2 judge raised unexpectedly: %s — defaulting maxsize PII", exc)
        return P2Verdict(pii_count=sys.maxsize, masked_prompt=None)


async def _safe_p3(prompt: str) -> str:
    try:
        return await p3_judge(prompt)
    except Exception as exc:
        logger.error("P3 judge raised unexpectedly: %s — defaulting AMBIGUOUS", exc)
        return "AMBIGUOUS"

# Alias used by app/main.py
run_orchestrator = run_micro_judges
