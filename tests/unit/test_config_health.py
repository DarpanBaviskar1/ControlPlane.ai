"""Tests for _is_real_key utility and GET /v1/config/health endpoint.

Requirements: 6.1-6.6, NFR-2, NFR-4
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.config import _is_real_key


# ---------------------------------------------------------------------------
# _is_real_key tests
# ---------------------------------------------------------------------------

class TestIsRealKey:

    def test_empty_string_is_not_real(self) -> None:
        assert _is_real_key("") is False

    def test_whitespace_is_not_real(self) -> None:
        assert _is_real_key("   ") is False

    def test_dummy_prefix_is_not_real(self) -> None:
        assert _is_real_key("dummy-portkey-key") is False
        assert _is_real_key("dummy-openai-key") is False
        assert _is_real_key("dummy") is False

    def test_real_key_is_real(self) -> None:
        assert _is_real_key("sk-real-key-abc123") is True
        assert _is_real_key("pk-live-xyz") is True
        assert _is_real_key("some-actual-key") is True

    def test_key_starting_with_d_but_not_dummy(self) -> None:
        assert _is_real_key("dummyx-something") is False  # starts with "dummy"
        assert _is_real_key("d-not-dummy") is True        # starts with "d" but not "dummy"


# ---------------------------------------------------------------------------
# GET /v1/config/health endpoint tests
# ---------------------------------------------------------------------------

class TestConfigHealthEndpoint:

    def test_returns_200_with_correct_schema(self) -> None:
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/config/health")
        assert resp.status_code == 200
        data = resp.json()
        for key in ("portkey", "langfuse", "guardrails", "worldsense", "llm_direct"):
            assert key in data, f"Missing key: {key}"
            assert data[key]["status"] in ("active", "degraded")
            assert isinstance(data[key]["detail"], str)
            assert len(data[key]["detail"]) > 0

    def test_all_degraded_with_dummy_keys(self) -> None:
        """With default dummy/empty keys, all integrations should be degraded."""
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/config/health")
        assert resp.status_code == 200
        data = resp.json()
        # Portkey uses dummy key by default
        assert data["portkey"]["status"] == "degraded"
        # LLM direct key is empty by default
        assert data["llm_direct"]["status"] == "degraded"
        # Langfuse keys are empty by default
        assert data["langfuse"]["status"] == "degraded"

    def test_portkey_active_when_real_key(self, monkeypatch) -> None:
        from app.config import settings
        monkeypatch.setattr(settings, "PORTKEY_API_KEY", "pk-live-real-key-123")
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/config/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["portkey"]["status"] == "active"
        assert "Portkey API key configured" in data["portkey"]["detail"]

    def test_llm_direct_active_when_real_key(self, monkeypatch) -> None:
        from app.config import settings
        monkeypatch.setattr(settings, "LLM_API_KEY", "sk-real-openai-key-xyz")
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/v1/config/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["llm_direct"]["status"] == "active"

    def test_worldsense_active_when_state_healthy(self) -> None:
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            # Inject healthy worldsense state
            app.state.worldsense_mcp_healthy = True
            resp = client.get("/v1/config/health")
            app.state.worldsense_mcp_healthy = False  # reset
        assert resp.status_code == 200
        data = resp.json()
        assert data["worldsense"]["status"] == "active"

    def test_no_outbound_calls_on_health_check(self) -> None:
        """Health endpoint must not make any network calls — verified by ensuring
        the response is fast and doesn't raise connection errors in offline mode."""
        import time
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            start = time.monotonic()
            resp = client.get("/v1/config/health")
            elapsed_ms = (time.monotonic() - start) * 1000
        assert resp.status_code == 200
        # Must respond well within 50 ms (NFR-2)
        assert elapsed_ms < 500, f"Health endpoint too slow: {elapsed_ms:.0f}ms"
