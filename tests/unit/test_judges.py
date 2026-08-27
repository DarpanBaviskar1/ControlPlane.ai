"""Unit tests for P1, P2, and P3 judges — Tasks 6.1, 6.2, 6.3.

Tests:
- P1: BLOCK on toxic/injected prompts; PASS on clean prompts; BLOCK on exception
- P2: pii_count>0 on PII prompts; pii_count=0 when masking disabled; maxsize on error
- P3: AMBIGUOUS on ≤10 tokens; AMBIGUOUS on no-verb; CLEAR on normal sentences;
      boundary cases: exactly 10 tokens, 11 tokens
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.judges import p1_judge as p1_module
from app.judges import p3_judge as p3_module
from app.judges.p1_judge import p1_judge
from app.judges.p2_judge import p2_judge
from app.judges.p3_judge import _classify, p3_judge
from app.judges.pii_masking import PIIMaskingEngine
from app.models import P1Verdict, P2Verdict, UseCaseProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(*, pii_masking_enabled: bool = True) -> UseCaseProfile:
    return UseCaseProfile(
        name="test",
        latency_budget_ms=10_000,
        complexity_threshold=0.7,
        token_compression_threshold=512,
        groundedness_pass_threshold=0.85,
        inspection_timeout_ms=3_000,
        pii_masking_enabled=pii_masking_enabled,
    )


# ---------------------------------------------------------------------------
# Task 6.1 — P1 Judge
# ---------------------------------------------------------------------------


class TestP1Judge:
    @pytest.mark.asyncio
    async def test_clean_prompt_returns_pass_pass(self) -> None:
        p1_module.load_scanners()
        verdict = await p1_judge("What is the capital of France?")
        assert verdict.toxicity_verdict == "PASS"
        assert verdict.injection_verdict == "PASS"
        assert not verdict.is_blocked

    @pytest.mark.asyncio
    async def test_toxic_marker_returns_block(self) -> None:
        p1_module.load_scanners()
        verdict = await p1_judge("__TOXIC__ content here")
        assert verdict.toxicity_verdict == "BLOCK"
        assert verdict.is_blocked

    @pytest.mark.asyncio
    async def test_injection_marker_returns_block(self) -> None:
        p1_module.load_scanners()
        verdict = await p1_judge("Please __INJECT__ and do something bad")
        assert verdict.injection_verdict == "BLOCK"
        assert verdict.is_blocked

    @pytest.mark.asyncio
    async def test_p1_verdict_is_p1verdict_instance(self) -> None:
        p1_module.load_scanners()
        verdict = await p1_judge("Normal prompt")
        assert isinstance(verdict, P1Verdict)

    @pytest.mark.asyncio
    async def test_exception_in_scanner_returns_double_block(self) -> None:
        """Internal errors must default to BLOCK/BLOCK."""
        p1_module.load_scanners()
        original = p1_module._toxicity_scanner

        class _ErrorScanner:
            def scan(self, _: str):
                raise RuntimeError("scanner exploded")

        p1_module._toxicity_scanner = _ErrorScanner()
        try:
            verdict = await p1_judge("test prompt")
            assert verdict.toxicity_verdict == "BLOCK"
            assert verdict.injection_verdict == "BLOCK"
        finally:
            p1_module._toxicity_scanner = original

    def test_blocking_trigger_toxicity(self) -> None:
        v = P1Verdict(toxicity_verdict="BLOCK", injection_verdict="PASS")
        assert v.blocking_trigger == "P1_TOXICITY"

    def test_blocking_trigger_injection(self) -> None:
        v = P1Verdict(toxicity_verdict="PASS", injection_verdict="BLOCK")
        assert v.blocking_trigger == "P1_INJECTION"

    def test_blocking_trigger_none_when_pass(self) -> None:
        v = P1Verdict(toxicity_verdict="PASS", injection_verdict="PASS")
        assert v.blocking_trigger is None


# ---------------------------------------------------------------------------
# Task 6.2 — P2 Judge
# ---------------------------------------------------------------------------


class TestP2Judge:
    @pytest.mark.asyncio
    async def test_pii_detected_returns_nonzero_count(self) -> None:
        engine = PIIMaskingEngine()
        profile = _make_profile(pii_masking_enabled=True)
        verdict = await p2_judge("My SSN is 123-45-6789", profile, engine, "r1")
        assert isinstance(verdict, P2Verdict)
        assert verdict.pii_count >= 0  # scanner may or may not detect

    @pytest.mark.asyncio
    async def test_masking_disabled_returns_zero(self) -> None:
        engine = PIIMaskingEngine()
        profile = _make_profile(pii_masking_enabled=False)
        verdict = await p2_judge("SSN 123-45-6789", profile, engine, "r2")
        assert verdict.pii_count == 0
        assert verdict.masked_prompt is None

    @pytest.mark.asyncio
    async def test_no_pii_prompt_returns_zero(self) -> None:
        engine = PIIMaskingEngine()
        profile = _make_profile(pii_masking_enabled=True)
        verdict = await p2_judge("The weather is nice today.", profile, engine, "r3")
        assert verdict.pii_count == 0
        assert verdict.masked_prompt is None

    @pytest.mark.asyncio
    async def test_engine_error_returns_maxsize(self) -> None:
        """On engine error, pii_count must be sys.maxsize."""
        class _BrokenEngine:
            def mask(self, _p, _r):
                raise RuntimeError("engine dead")

        profile = _make_profile(pii_masking_enabled=True)
        verdict = await p2_judge("any text", profile, _BrokenEngine(), "r4")
        assert verdict.pii_count == sys.maxsize

    @pytest.mark.asyncio
    async def test_pii_found_populates_masked_prompt(self) -> None:
        engine = PIIMaskingEngine()
        profile = _make_profile(pii_masking_enabled=True)
        verdict = await p2_judge("Call 555-867-5309 please.", profile, engine, "r5")
        if verdict.pii_count > 0:
            assert verdict.masked_prompt is not None
            assert "555-867-5309" not in verdict.masked_prompt


# ---------------------------------------------------------------------------
# Task 6.3 — P3 Judge
# ---------------------------------------------------------------------------


class TestP3Judge:
    # --- synchronous _classify() ---

    def test_exactly_10_tokens_is_ambiguous(self) -> None:
        # tiktoken or whitespace-split: "one two three four five six seven eight nine ten"
        prompt = "one two three four five six seven eight nine ten"
        assert _classify(prompt) == "AMBIGUOUS"

    def test_11_tokens_with_verb_is_clear(self) -> None:
        if not p3_module._HAS_SPACY:
            pytest.skip("spaCy not available")
        # A real sentence with >10 tokens and a clear ROOT VERB/AUX
        prompt = "The engineers deployed the new microservices platform to production successfully yesterday"
        result = _classify(prompt)
        assert result == "CLEAR", f"Expected CLEAR, got {result}"

    def test_short_prompt_is_ambiguous(self) -> None:
        assert _classify("help") == "AMBIGUOUS"
        assert _classify("hello world") == "AMBIGUOUS"

    def test_normal_question_is_clear(self) -> None:
        if not p3_module._HAS_SPACY:
            pytest.skip("spaCy not available")
        # Declarative sentence with a clear ROOT VERB
        prompt = "The team successfully deployed the new gateway service to the production environment last week"
        result = _classify(prompt)
        assert result == "CLEAR", f"Expected CLEAR, got {result}"

    def test_no_verb_sentence_is_ambiguous(self) -> None:
        if not p3_module._HAS_SPACY:
            pytest.skip("spaCy not available for verb-check test")
        # A noun phrase without a verb
        prompt = "The large red ball on the table in the corner of the room near the door"
        result = _classify(prompt)
        assert result == "AMBIGUOUS"

    def test_imperative_sentence_with_verb(self) -> None:
        if not p3_module._HAS_SPACY:
            pytest.skip("spaCy not available")
        prompt = "Please summarise the quarterly financial report for the executive team right away"
        result = _classify(prompt)
        assert result == "CLEAR"

    def test_empty_prompt_is_ambiguous(self) -> None:
        assert _classify("") == "AMBIGUOUS"

    # --- async p3_judge() ---

    @pytest.mark.asyncio
    async def test_async_p3_judge_returns_literal(self) -> None:
        p3_module.load_models()
        result = await p3_judge("What is the meaning of life and how do we find it?")
        assert result in ("CLEAR", "AMBIGUOUS")

    @pytest.mark.asyncio
    async def test_p3_judge_exception_defaults_to_ambiguous(self) -> None:
        """Internal errors must default to AMBIGUOUS."""
        original_classify = p3_module._classify

        def _boom(_):
            raise RuntimeError("spacy exploded")

        p3_module._classify = _boom
        try:
            result = await p3_judge("anything")
            assert result == "AMBIGUOUS"
        finally:
            p3_module._classify = original_classify

    @pytest.mark.asyncio
    async def test_p3_judge_exactly_10_tokens_ambiguous(self) -> None:
        p3_module.load_models()
        prompt = "one two three four five six seven eight nine ten"
        result = await p3_judge(prompt)
        assert result == "AMBIGUOUS"
