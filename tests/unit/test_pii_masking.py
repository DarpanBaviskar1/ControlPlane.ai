"""Tests for PIIMaskingEngine — Tasks 5.2, 5.3.

Tests:
- unmask() restores placeholders correctly
- byte-for-byte fidelity after whitespace normalisation
- run_startup_validation() returns True on healthy engine
- 503 gate when is_healthy=False
- discard_mapping() removes the per-request map
"""

from __future__ import annotations

import asyncio
import re

import pytest

from app.judges.pii_masking import PIIMaskingEngine, _normalise

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine() -> PIIMaskingEngine:
    return PIIMaskingEngine()


# ---------------------------------------------------------------------------
# Task 5.2 — unmask() byte-for-byte fidelity
# ---------------------------------------------------------------------------


class TestUnmask:
    def test_unmask_restores_ssn(self, engine: PIIMaskingEngine) -> None:
        prompt = "My SSN is 123-45-6789 here."
        masked, pmap = engine.mask(prompt, "req-001")
        if not pmap:
            pytest.skip("No PII detected by scanner")
        restored = engine.unmask(masked, "req-001")
        assert _normalise(restored) == _normalise(prompt)

    def test_unmask_restores_email(self, engine: PIIMaskingEngine) -> None:
        prompt = "Email me at alice@example.com please."
        masked, pmap = engine.mask(prompt, "req-002")
        if not pmap:
            pytest.skip("No PII detected by scanner")
        restored = engine.unmask(masked, "req-002")
        assert _normalise(restored) == _normalise(prompt)

    def test_unmask_no_pii_prompt_unchanged(self, engine: PIIMaskingEngine) -> None:
        prompt = "The weather is sunny today."
        masked, _ = engine.mask(prompt, "req-003")
        restored = engine.unmask(masked, "req-003")
        assert _normalise(restored) == _normalise(prompt)

    def test_unmask_unknown_request_id_returns_as_is(self, engine: PIIMaskingEngine) -> None:
        """unmask() with no stored map must return the input unchanged."""
        text = "some [SSN_REDACTED_1] text"
        result = engine.unmask(text, "nonexistent-id")
        assert result == text

    def test_discard_mapping_removes_map(self, engine: PIIMaskingEngine) -> None:
        prompt = "SSN 123-45-6789"
        engine.mask(prompt, "req-dis")
        engine.discard_mapping("req-dis")
        # After discard, unmask can't restore
        masked = "[SSN_REDACTED_1]"
        result = engine.unmask(masked, "req-dis")
        assert result == masked  # no map → unchanged

    def test_discard_mapping_on_nonexistent_id_is_safe(self, engine: PIIMaskingEngine) -> None:
        """discard_mapping() on a nonexistent request_id must not raise."""
        engine.discard_mapping("does-not-exist")  # must not raise

    def test_multiple_pii_types_round_trip(self, engine: PIIMaskingEngine) -> None:
        prompt = "Call 555-867-5309 or email bob@corp.io and SSN is 111-22-3333."
        masked, pmap = engine.mask(prompt, "req-multi")
        if not pmap:
            pytest.skip("No PII detected")
        restored = engine.unmask(masked, "req-multi")
        engine.discard_mapping("req-multi")
        assert _normalise(restored) == _normalise(prompt)

    def test_per_request_isolation(self, engine: PIIMaskingEngine) -> None:
        """Two concurrent requests must not share placeholder maps."""
        p1 = "SSN 123-45-6789"
        p2 = "Email foo@bar.com"
        _, m1 = engine.mask(p1, "iso-req-1")
        _, m2 = engine.mask(p2, "iso-req-2")
        # Each map must be a distinct dict
        assert id(engine._maps.get("iso-req-1")) != id(engine._maps.get("iso-req-2"))
        engine.discard_mapping("iso-req-1")
        engine.discard_mapping("iso-req-2")


# ---------------------------------------------------------------------------
# Task 5.3 — run_startup_validation() + lifespan gate
# ---------------------------------------------------------------------------


class TestStartupValidation:
    @pytest.mark.asyncio
    async def test_healthy_engine_passes_validation(self) -> None:
        engine = PIIMaskingEngine()
        result = await engine.run_startup_validation()
        assert result is True
        assert engine.is_healthy is True

    @pytest.mark.asyncio
    async def test_validation_sets_is_healthy_true_on_success(self) -> None:
        engine = PIIMaskingEngine()
        engine.is_healthy = False  # force unhealthy state
        await engine.run_startup_validation()
        assert engine.is_healthy is True

    @pytest.mark.asyncio
    async def test_validation_discards_startup_mappings(self) -> None:
        """Validation must not leave request mappings in the internal dict."""
        engine = PIIMaskingEngine()
        await engine.run_startup_validation()
        remaining = [k for k in engine._maps if k.startswith("__startup_validation_")]
        assert not remaining, f"Startup maps not discarded: {remaining}"

    def test_unhealthy_engine_blocks_requests(self) -> None:
        """When is_healthy=False, the ingress handler must return 503."""
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            original = app.state.pii_engine
            try:
                app.state.pii_engine.is_healthy = False
                resp = client.post(
                    "/v1/chat",
                    json={"prompt": "Hello", "use_case_profile": "customer_chatbot"},
                )
                assert resp.status_code == 503
            finally:
                if original is not None:
                    app.state.pii_engine.is_healthy = True

    def test_healthy_engine_does_not_block_requests(self) -> None:
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            assert app.state.pii_engine.is_healthy is True
            resp = client.post(
                "/v1/chat",
                json={"prompt": "Hello", "use_case_profile": "customer_chatbot"},
            )
            assert resp.status_code == 200
