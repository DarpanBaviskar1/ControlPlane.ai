"""P2 Judge — PII detection.

Detects PII tokens using the PIIMaskingEngine (backed by LLM Guard Anonymize
or the regex fallback).  Returns a P2Verdict with the pii_count, the masked
prompt (if PII was found), and the placeholder_map.

The P2 Judge is responsible for detection and counting only.  The orchestrator
decides whether to apply masking based on the profile's pii_masking_enabled flag.

Failure handling: any internal exception returns pii_count=sys.maxsize, which
the orchestrator treats as "PII detected" and triggers masking.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from app.models import P2Verdict, UseCaseProfile

logger = logging.getLogger(__name__)


async def p2_judge(
    prompt: str,
    profile: UseCaseProfile,
    pii_engine: object,  # PIIMaskingEngine
    request_id: str,
    gliner_model: object | None = None,
) -> P2Verdict:
    """Detect PII in *prompt* and return a P2Verdict.

    If pii_masking_enabled is False the engine is not called and pii_count=0
    is returned immediately (the orchestrator logs PII_MASKING_BYPASSED).
    """
    if not profile.pii_masking_enabled:
        return P2Verdict(pii_count=0, masked_prompt=None, placeholder_map={})

    try:
        masked_prompt, placeholder_map = await asyncio.to_thread(
            pii_engine.mask,
            prompt,
            request_id,
            profile.custom_entity_terms or [],
            gliner_model,
        )
        pii_count = len(placeholder_map)
        return P2Verdict(
            pii_count=pii_count,
            masked_prompt=masked_prompt if pii_count > 0 else None,
            placeholder_map=placeholder_map,
        )
    except Exception as exc:
        logger.error("P2 judge error for request_id=%s: %s", request_id, exc)
        # Safe default: treat as maximum PII to force masking / HARD_BLOCK
        return P2Verdict(pii_count=sys.maxsize, masked_prompt=None, placeholder_map={})
