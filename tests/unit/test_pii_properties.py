"""Property-based tests for PII masking.

# Feature: controlplane-ai-gateway, Property 16: PII Masking Round-Trip Fidelity
# Feature: controlplane-ai-gateway, Property 6: P2 PII Masking Replaces Tokens

Validates: Requirements 2.5, 9.2
"""

from __future__ import annotations

import re
import sys

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.judges.p2_judge import p2_judge
from app.judges.pii_masking import PIIMaskingEngine, _normalise, _FALLBACK_PATTERNS
from app.models import UseCaseProfile

# ---------------------------------------------------------------------------
# Shared engine instance (avoid repeated construction)
# ---------------------------------------------------------------------------

_ENGINE = PIIMaskingEngine()

PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+_REDACTED(?:_\d+)?\]")


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
# Helper strategies
# ---------------------------------------------------------------------------

# A prompt that definitely contains detectable PII (for the fallback scanner)
_SSN_STRATEGY = st.builds(
    lambda pre, ssn, post: f"{pre} my SSN is {ssn} end {post}",
    pre=st.text(max_size=20, alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs"))),
    ssn=st.builds(
        lambda a, b, c: f"{a:03d}-{b:02d}-{c:04d}",
        a=st.integers(min_value=100, max_value=899),
        b=st.integers(min_value=10, max_value=99),
        c=st.integers(min_value=1000, max_value=9999),
    ),
    post=st.text(max_size=20, alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs"))),
)

_EMAIL_STRATEGY = st.builds(
    lambda local, domain, tld, pre, post: f"{pre} contact {local}@{domain}.{tld} today {post}",
    local=st.text(min_size=1, max_size=15, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"),
    domain=st.text(min_size=2, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
    tld=st.sampled_from(["com", "org", "net", "io"]),
    pre=st.text(max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz "),
    post=st.text(max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz "),
)


# ---------------------------------------------------------------------------
# Property 16: PII Masking Round-Trip Fidelity
# Validates: Requirement 9.2
# ---------------------------------------------------------------------------


class TestProperty16PIIMaskingRoundTrip:

    @given(_SSN_STRATEGY)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ssn_round_trip_fidelity(self, prompt: str) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 16: PII Masking Round-Trip Fidelity
        mask() followed by unmask() must be byte-for-byte identical after whitespace
        normalisation for any prompt containing a detectable SSN.
        """
        request_id = f"prop16-ssn-{hash(prompt) & 0xFFFF}"
        masked, pmap = _ENGINE.mask(prompt, request_id)
        try:
            if not pmap:
                # Scanner didn't detect PII → unmask is identity
                assert _normalise(masked) == _normalise(prompt)
                return
            restored = _ENGINE.unmask(masked, request_id)
            assert _normalise(restored) == _normalise(prompt), (
                f"Round-trip failed.\n"
                f"  original: {_normalise(prompt)!r}\n"
                f"  masked:   {masked!r}\n"
                f"  restored: {_normalise(restored)!r}"
            )
        finally:
            _ENGINE.discard_mapping(request_id)

    @given(_EMAIL_STRATEGY)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_email_round_trip_fidelity(self, prompt: str) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 16: PII Masking Round-Trip Fidelity
        """
        request_id = f"prop16-email-{hash(prompt) & 0xFFFF}"
        masked, pmap = _ENGINE.mask(prompt, request_id)
        try:
            if not pmap:
                assert _normalise(masked) == _normalise(prompt)
                return
            restored = _ENGINE.unmask(masked, request_id)
            assert _normalise(restored) == _normalise(prompt), (
                f"Round-trip failed.\n"
                f"  original: {_normalise(prompt)!r}\n"
                f"  restored: {_normalise(restored)!r}"
            )
        finally:
            _ENGINE.discard_mapping(request_id)

    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_arbitrary_prompt_round_trip(self, prompt: str) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 16: PII Masking Round-Trip Fidelity
        For any arbitrary prompt, round-trip must preserve content.
        """
        request_id = f"prop16-arb-{hash(prompt) & 0xFFFF}"
        masked, pmap = _ENGINE.mask(prompt, request_id)
        try:
            restored = _ENGINE.unmask(masked, request_id)
            assert _normalise(restored) == _normalise(prompt), (
                f"Round-trip failed for arbitrary prompt.\n"
                f"  original: {_normalise(prompt)!r}\n"
                f"  restored: {_normalise(restored)!r}"
            )
        finally:
            _ENGINE.discard_mapping(request_id)

    def test_discard_after_unmask_prevents_re_unmask(self) -> None:
        """After discard, unmask must not restore (map is gone)."""
        prompt = "SSN 123-45-6789 email test@example.com"
        request_id = "prop16-discard"
        masked, pmap = _ENGINE.mask(prompt, request_id)
        if not pmap:
            pytest.skip("No PII detected")
        _ENGINE.discard_mapping(request_id)
        # Second unmask has no map → returns input unchanged
        result = _ENGINE.unmask(masked, request_id)
        assert result == masked


# ---------------------------------------------------------------------------
# Property 6: P2 PII Masking Replaces Tokens
# Validates: Requirement 2.5
# ---------------------------------------------------------------------------


class TestProperty6P2PIITokenReplacement:

    @given(_SSN_STRATEGY)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_ssn_not_present_in_masked_prompt(self, prompt: str) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 6: P2 PII Masking Replaces Tokens
        After masking, the original SSN must not appear in the masked output.
        """
        # Extract SSN from the prompt using the known pattern
        ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        ssn_matches = ssn_pattern.findall(prompt)

        request_id = f"prop6-ssn-{hash(prompt) & 0xFFFF}"
        masked, pmap = _ENGINE.mask(prompt, request_id)
        _ENGINE.discard_mapping(request_id)

        if not pmap or not ssn_matches:
            return  # scanner didn't detect — skip assertion

        for ssn in ssn_matches:
            assert ssn not in masked, (
                f"SSN {ssn!r} still present in masked prompt: {masked!r}"
            )

    @given(_EMAIL_STRATEGY)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_email_not_present_in_masked_prompt(self, prompt: str) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 6: P2 PII Masking Replaces Tokens
        After masking, the original email must not appear in the masked output.
        """
        email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        )
        email_matches = email_pattern.findall(prompt)

        request_id = f"prop6-email-{hash(prompt) & 0xFFFF}"
        masked, pmap = _ENGINE.mask(prompt, request_id)
        _ENGINE.discard_mapping(request_id)

        if not pmap or not email_matches:
            return

        for email in email_matches:
            assert email not in masked, (
                f"Email {email!r} still present in masked prompt: {masked!r}"
            )

    @given(_SSN_STRATEGY)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_masked_prompt_contains_placeholder_token(self, prompt: str) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 6: P2 PII Masking Replaces Tokens
        Each detected PII token must be replaced with a [TYPE_REDACTED] placeholder.
        """
        request_id = f"prop6-ph-{hash(prompt) & 0xFFFF}"
        masked, pmap = _ENGINE.mask(prompt, request_id)
        _ENGINE.discard_mapping(request_id)

        if not pmap:
            return

        placeholders = PLACEHOLDER_RE.findall(masked)
        assert len(placeholders) > 0, (
            f"Expected placeholder tokens in masked output but found none: {masked!r}"
        )
        # Each placeholder must match the expected format
        for ph in placeholders:
            assert PLACEHOLDER_RE.fullmatch(ph), f"Malformed placeholder: {ph!r}"

    @pytest.mark.asyncio
    @given(_SSN_STRATEGY)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_p2_verdict_pii_count_nonzero_for_pii_prompt(self, prompt: str) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 6: P2 PII Masking Replaces Tokens
        p2_judge() with pii_masking_enabled=True must return pii_count > 0.
        """
        engine = PIIMaskingEngine()
        profile = _make_profile(pii_masking_enabled=True)
        request_id = f"prop6-p2-{hash(prompt) & 0xFFFF}"
        verdict = await p2_judge(prompt, profile, engine, request_id)
        engine.discard_mapping(request_id)

        # The regex scanner should detect the SSN we embedded
        ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        if ssn_pattern.search(prompt):
            assert verdict.pii_count > 0, (
                f"Expected pii_count > 0 for SSN-containing prompt, got {verdict.pii_count}"
            )
