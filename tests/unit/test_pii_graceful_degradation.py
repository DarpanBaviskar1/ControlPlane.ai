"""Tests for PIIMaskingEngine two-tier graceful degradation (Item 5).

Covers:
- Engine defaults to regex tier when LLM Guard not installed
- NLP tier failure during validation → downgrade to regex, is_healthy=True
- Regex tier failure → is_healthy=False, 503 gate fires
- active_tier property reflects current scanner
- mask/unmask/discard_mapping still work correctly after tier switch
- _emit_degraded_alert does not crash when telemetry_logger is None
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.judges.pii_masking import (
    PIIMaskingEngine,
    RegexOnlyMasker,
    _normalise,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _BrokenScanner:
    """Scanner whose scan() always raises — simulates NLP model crash."""
    name = "nlp"

    def scan(self, prompt: str):
        raise RuntimeError("NLP model OOM")


class _BadRoundTripScanner:
    """Scanner that returns garbage — causes round-trip fidelity to fail."""
    name = "nlp"

    def scan(self, prompt: str):
        return "TOTALLY_DIFFERENT_OUTPUT", False, 1.0


class _PassScanner:
    """Scanner that always reports no PII (identity masker)."""
    name = "nlp"

    def scan(self, prompt: str):
        return prompt, True, 0.0


# ---------------------------------------------------------------------------
# active_tier reflects scanner
# ---------------------------------------------------------------------------

class TestActiveTier:
    def test_regex_tier_name(self) -> None:
        engine = PIIMaskingEngine()
        engine._scanner = RegexOnlyMasker()
        assert engine.active_tier == "regex"

    def test_nlp_tier_name(self) -> None:
        engine = PIIMaskingEngine()
        engine._scanner = _PassScanner()
        assert engine.active_tier == "nlp"


# ---------------------------------------------------------------------------
# NLP failure → downgrade to regex, gateway stays healthy
# ---------------------------------------------------------------------------

class TestNLPDowngradeToRegex:
    @pytest.mark.asyncio
    async def test_broken_nlp_scanner_downgrades_to_regex(self) -> None:
        """If the NLP scanner raises during scan(), validation fails and the
        engine switches to regex tier with is_healthy=True."""
        engine = PIIMaskingEngine()
        engine._scanner = _BrokenScanner()

        result = await engine.run_startup_validation()

        assert result is True, "Engine should remain healthy after downgrade"
        assert engine.is_healthy is True
        assert engine.active_tier == "regex", (
            f"Expected regex tier after NLP failure, got {engine.active_tier!r}"
        )

    @pytest.mark.asyncio
    async def test_bad_round_trip_nlp_downgrades_to_regex(self) -> None:
        """If the NLP scanner returns wrong output (round-trip fails), it
        should downgrade gracefully rather than going 503."""
        engine = PIIMaskingEngine()
        engine._scanner = _BadRoundTripScanner()

        result = await engine.run_startup_validation()

        assert result is True
        assert engine.is_healthy is True
        assert engine.active_tier == "regex"

    @pytest.mark.asyncio
    async def test_degraded_alert_emitted_on_nlp_failure(self, capsys) -> None:
        """When no telemetry_logger is wired, the alert goes to stderr."""
        engine = PIIMaskingEngine(telemetry_logger=None)
        engine._scanner = _BrokenScanner()

        await engine.run_startup_validation()

        captured = capsys.readouterr()
        assert "MASKING_DEGRADED_TO_REGEX" in captured.err, (
            "Expected MASKING_DEGRADED_TO_REGEX alert on stderr"
        )

    @pytest.mark.asyncio
    async def test_degraded_alert_uses_telemetry_logger_when_available(self) -> None:
        """When a telemetry logger is present, record_alert is called."""
        mock_logger = MagicMock()
        mock_logger.record_alert = AsyncMock()

        engine = PIIMaskingEngine(telemetry_logger=mock_logger)
        engine._scanner = _BrokenScanner()

        # Provide a running event loop so create_task works
        await engine.run_startup_validation()
        await asyncio.sleep(0)  # let the task schedule

        mock_logger.record_alert.assert_called_once()
        call_kwargs = mock_logger.record_alert.call_args.kwargs
        assert call_kwargs.get("alert_type") == "MASKING_DEGRADED_TO_REGEX"
        assert call_kwargs.get("severity") == "HIGH"


# ---------------------------------------------------------------------------
# Both tiers fail → is_healthy=False
# ---------------------------------------------------------------------------

class TestBothTiersFail:
    @pytest.mark.asyncio
    async def test_both_tiers_fail_sets_unhealthy(self) -> None:
        """If NLP fails AND regex fails, the gateway must enter 503 state."""
        engine = PIIMaskingEngine()
        engine._scanner = _BadRoundTripScanner()

        # Patch RegexOnlyMasker so its scan() also returns bad data
        with patch(
            "app.judges.pii_masking.RegexOnlyMasker.scan",
            return_value=("GARBAGE", False, 1.0),
        ):
            result = await engine.run_startup_validation()

        assert result is False
        assert engine.is_healthy is False

    @pytest.mark.asyncio
    async def test_unhealthy_engine_blocks_requests(self) -> None:
        """When is_healthy=False, the ingress must return 503."""
        import sys
        from unittest.mock import MagicMock

        # Stub optional heavy deps that may not be installed in the test env
        for mod in ("routellm", "routellm.controller", "portkey_ai", "langfuse"):
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()

        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            original_health = app.state.pii_engine.is_healthy
            try:
                app.state.pii_engine.is_healthy = False
                resp = client.post(
                    "/v1/chat",
                    json={"prompt": "Hello", "use_case_profile": "customer_chatbot"},
                )
                assert resp.status_code == 503
                body = resp.json()
                assert "MASKING_INTEGRITY_FAILURE" in str(body)
            finally:
                app.state.pii_engine.is_healthy = original_health


# ---------------------------------------------------------------------------
# Regex tier still works correctly after a tier switch
# ---------------------------------------------------------------------------

class TestRegexTierFunctionality:
    @pytest.mark.asyncio
    async def test_mask_unmask_round_trip_after_downgrade(self) -> None:
        """After an NLP→regex downgrade, mask/unmask must still work."""
        engine = PIIMaskingEngine()
        engine._scanner = _BrokenScanner()
        await engine.run_startup_validation()
        assert engine.active_tier == "regex"

        prompt = "My SSN is 123-45-6789 please help."
        masked, pmap = await asyncio.to_thread(engine.mask, prompt, "req-deg-1")
        try:
            if pmap:
                restored = await asyncio.to_thread(engine.unmask, masked, "req-deg-1")
                assert _normalise(restored) == _normalise(prompt)
        finally:
            engine.discard_mapping("req-deg-1")

    @pytest.mark.asyncio
    async def test_pii_still_redacted_after_downgrade(self) -> None:
        """After downgrade, PII tokens must still be replaced by placeholders."""
        engine = PIIMaskingEngine()
        engine._scanner = _BrokenScanner()
        await engine.run_startup_validation()

        prompt = "Email: test@example.com and SSN: 987-65-4321"
        masked, pmap = await asyncio.to_thread(engine.mask, prompt, "req-deg-2")
        engine.discard_mapping("req-deg-2")

        if pmap:
            assert "test@example.com" not in masked
            assert "987-65-4321" not in masked

    @pytest.mark.asyncio
    async def test_discard_mapping_safe_after_downgrade(self) -> None:
        """discard_mapping must not raise after a tier switch."""
        engine = PIIMaskingEngine()
        engine._scanner = _BrokenScanner()
        await engine.run_startup_validation()

        engine.discard_mapping("nonexistent-id")  # must not raise

    def test_regex_only_masker_standalone(self) -> None:
        """RegexOnlyMasker can be tested directly without the engine."""
        masker = RegexOnlyMasker()
        prompt = "SSN 123-45-6789 and email bob@corp.io"
        masked, is_valid, score = masker.scan(prompt)
        assert is_valid is False
        assert "123-45-6789" not in masked
        assert "bob@corp.io" not in masked
        assert score == 1.0

    def test_regex_only_masker_clean_prompt(self) -> None:
        masker = RegexOnlyMasker()
        prompt = "The weather is sunny today."
        masked, is_valid, score = masker.scan(prompt)
        assert is_valid is True
        assert masked == prompt
        assert score == 0.0


# ---------------------------------------------------------------------------
# Healthy NLP path is unchanged
# ---------------------------------------------------------------------------

class TestHealthyNLPPath:
    @pytest.mark.asyncio
    async def test_healthy_nlp_tier_stays_nlp(self) -> None:
        """A passing NLP scanner must not be downgraded."""
        engine = PIIMaskingEngine()
        engine._scanner = _PassScanner()

        result = await engine.run_startup_validation()

        # _PassScanner returns the prompt unchanged (no PII detected),
        # so round-trip fidelity passes (restored == original).
        assert result is True
        assert engine.is_healthy is True
        assert engine.active_tier == "nlp"
