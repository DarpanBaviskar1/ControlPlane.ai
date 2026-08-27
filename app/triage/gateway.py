"""Action Triage Gateway — four-state decision matrix.

Implements the deterministic priority-ordered evaluation:
  Priority 1 (highest): HARD_BLOCK
  Priority 2:           ESCALATE_TO_HUMAN
  Priority 3:           COMPRESS_AND_EDIT
  Priority 4 (lowest):  PASS_AND_DELIVER

The compressor (COMPRESS_AND_EDIT path) sends a token-budget summarisation
prompt to the SLM tier via Portkey and validates that no new named entities
appear in the output (spaCy NER containment check).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from app.models import TriageResult, TriageState, UseCaseProfile

logger = logging.getLogger(__name__)


def evaluate(
    groundedness_score: float,
    response_token_count: int,
    upstream_triage_state: TriageState | None,
    p3_clarity: Literal["CLEAR", "AMBIGUOUS"],
    profile: UseCaseProfile,
    response_content: str | None = None,
) -> TriageResult:
    """Apply the four-state priority matrix and return a TriageResult.

    States are evaluated in strict priority order; the first matching rule wins.
    """
    threshold = profile.groundedness_pass_threshold

    # ------------------------------------------------------------------
    # Priority 1: HARD_BLOCK
    # ------------------------------------------------------------------
    if upstream_triage_state == "HARD_BLOCK":
        return TriageResult(
            triage_state="HARD_BLOCK",
            blocking_reason=_upstream_reason(upstream_triage_state),
            response_content=None,
        )

    if groundedness_score < 0.5:
        return TriageResult(
            triage_state="HARD_BLOCK",
            blocking_reason="LOW_GROUNDEDNESS",
            response_content=None,
        )

    # ------------------------------------------------------------------
    # Priority 2: ESCALATE_TO_HUMAN
    # ------------------------------------------------------------------
    escalate = False
    if 0.5 <= groundedness_score <= threshold:
        escalate = True
    if p3_clarity == "AMBIGUOUS" and profile.human_escalation_enabled:
        escalate = True

    if escalate:
        if not profile.human_escalation_enabled:
            # Promote to HARD_BLOCK per Requirement 7.4
            return TriageResult(
                triage_state="HARD_BLOCK",
                blocking_reason="ESCALATION_SUPPRESSED",
                response_content=None,
            )
        return TriageResult(
            triage_state="ESCALATE_TO_HUMAN",
            blocking_reason=None,
            response_content=response_content,
        )

    # ------------------------------------------------------------------
    # Priority 3: COMPRESS_AND_EDIT
    # ------------------------------------------------------------------
    if (
        groundedness_score > threshold
        and response_token_count > profile.token_compression_threshold
    ):
        # The actual compression is handled by the pipeline coroutine;
        # here we just signal the state.
        return TriageResult(
            triage_state="COMPRESS_AND_EDIT",
            blocking_reason=None,
            response_content=response_content,
        )

    # ------------------------------------------------------------------
    # Priority 4: PASS_AND_DELIVER
    # ------------------------------------------------------------------
    return TriageResult(
        triage_state="PASS_AND_DELIVER",
        blocking_reason=None,
        response_content=response_content,
    )


def _upstream_reason(state: TriageState | None) -> str:
    """Convert an upstream triage state to a blocking reason string."""
    return str(state) if state else "UPSTREAM_BLOCK"
