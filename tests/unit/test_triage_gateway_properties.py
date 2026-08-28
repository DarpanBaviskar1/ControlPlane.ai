"""Property-based tests for TriageGateway.

Tasks: 6.2
Properties:
  NLI-2 — CONTRADICTION → HARD_BLOCK: any evaluate() call with nli_label="CONTRADICTION"
           must return triage_state="HARD_BLOCK" and blocking_reason="NLI_CONTRADICTION"
           regardless of the numeric groundedness_score.

Requirements: 2.4
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.triage.gateway import evaluate
from app.models import UseCaseProfile, TriageResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile(**overrides) -> UseCaseProfile:
    base = dict(
        name="test",
        latency_budget_ms=10_000,
        complexity_threshold=0.7,
        token_compression_threshold=512,
        groundedness_pass_threshold=0.85,
        inspection_timeout_ms=3_000,
    )
    base.update(overrides)
    return UseCaseProfile.model_validate(base)


# ---------------------------------------------------------------------------
# NLI-2: CONTRADICTION → HARD_BLOCK
# ---------------------------------------------------------------------------

class TestNLI2ContradictionHardBlock:
    """Property NLI-2: nli_label=CONTRADICTION must always produce HARD_BLOCK."""

    @given(
        groundedness_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        response_token_count=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=300)
    def test_contradiction_always_hard_blocks(
        self, groundedness_score: float, response_token_count: int
    ) -> None:
        """
        Property NLI-2: For any call to evaluate() where nli_label='CONTRADICTION',
        the result must have triage_state='HARD_BLOCK' and
        blocking_reason='NLI_CONTRADICTION' regardless of groundedness_score.
        """
        profile = _profile()
        result = evaluate(
            groundedness_score=groundedness_score,
            response_token_count=response_token_count,
            upstream_triage_state=None,
            p3_clarity="CLEAR",
            profile=profile,
            response_content="some response",
            nli_label="CONTRADICTION",
        )
        assert result.triage_state == "HARD_BLOCK", (
            f"Expected HARD_BLOCK for score={groundedness_score:.3f}, got {result.triage_state}"
        )
        assert result.blocking_reason == "NLI_CONTRADICTION", (
            f"Expected NLI_CONTRADICTION, got {result.blocking_reason!r}"
        )
        assert result.response_content is None

    @given(
        groundedness_score=st.floats(min_value=0.9, max_value=1.0, allow_nan=False),
        token_count=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=100)
    def test_contradiction_overrides_high_groundedness(
        self, groundedness_score: float, token_count: int
    ) -> None:
        """CONTRADICTION must block even when groundedness_score is above the pass threshold."""
        profile = _profile(groundedness_pass_threshold=0.5)
        result = evaluate(
            groundedness_score=groundedness_score,
            response_token_count=token_count,
            upstream_triage_state=None,
            p3_clarity="CLEAR",
            profile=profile,
            response_content="great answer",
            nli_label="CONTRADICTION",
        )
        assert result.triage_state == "HARD_BLOCK"
        assert result.blocking_reason == "NLI_CONTRADICTION"

    @given(
        groundedness_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_entailment_does_not_block(self, groundedness_score: float) -> None:
        """nli_label=ENTAILMENT must NOT trigger the NLI_CONTRADICTION block."""
        profile = _profile()
        result = evaluate(
            groundedness_score=groundedness_score,
            response_token_count=100,
            upstream_triage_state=None,
            p3_clarity="CLEAR",
            profile=profile,
            nli_label="ENTAILMENT",
        )
        assert result.blocking_reason != "NLI_CONTRADICTION"

    @given(
        groundedness_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_none_nli_label_does_not_block(self, groundedness_score: float) -> None:
        """nli_label=None must not trigger Priority-0 block."""
        profile = _profile()
        result = evaluate(
            groundedness_score=groundedness_score,
            response_token_count=100,
            upstream_triage_state=None,
            p3_clarity="CLEAR",
            profile=profile,
            nli_label=None,
        )
        assert result.blocking_reason != "NLI_CONTRADICTION"

    def test_contradiction_with_upstream_hard_block_still_nli_blocks_first(self) -> None:
        """Priority 0 fires before Priority 1 (upstream HARD_BLOCK)."""
        profile = _profile()
        result = evaluate(
            groundedness_score=1.0,
            response_token_count=50,
            upstream_triage_state="HARD_BLOCK",
            p3_clarity="CLEAR",
            profile=profile,
            nli_label="CONTRADICTION",
        )
        # NLI_CONTRADICTION has highest priority — check it fires
        assert result.triage_state == "HARD_BLOCK"
        assert result.blocking_reason == "NLI_CONTRADICTION"
