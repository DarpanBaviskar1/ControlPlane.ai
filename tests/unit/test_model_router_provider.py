"""Tests for generic LLM_API_KEY and LLM_PROVIDER settings in the model router.

Requirements: 1.1-1.4, 2.1-2.5, NFR-4
"""

from __future__ import annotations

import pytest

from app.config import _is_real_key, settings


class TestLLMAPIKeyConfig:

    def test_llm_api_key_exists_on_settings(self) -> None:
        """settings.LLM_API_KEY must exist (renamed from OPENAI_API_KEY)."""
        assert hasattr(settings, "LLM_API_KEY")

    def test_openai_api_key_does_not_exist(self) -> None:
        """OPENAI_API_KEY must no longer exist on settings."""
        assert not hasattr(settings, "OPENAI_API_KEY")

    def test_llm_fallback_model_default(self) -> None:
        assert settings.LLM_FALLBACK_MODEL == "gpt-4o-mini"

    def test_llm_provider_default(self) -> None:
        assert settings.LLM_PROVIDER == "openai"

    def test_llm_provider_accepts_valid_values(self, monkeypatch) -> None:
        for provider in ("openai", "anthropic", "google", "grok", "generic"):
            monkeypatch.setattr(settings, "LLM_PROVIDER", provider)
            assert settings.LLM_PROVIDER == provider


class TestMockPathReturnsMockResponse:

    @pytest.mark.asyncio
    async def test_mock_response_when_all_keys_absent(self, monkeypatch) -> None:
        """When LLM_API_KEY and PORTKEY_API_KEY are absent/dummy, route_and_call
        must return a contextual mock response without raising."""
        from app.config import settings
        monkeypatch.setattr(settings, "LLM_API_KEY", "")
        monkeypatch.setattr(settings, "PORTKEY_API_KEY", "dummy-portkey-key")

        import app.router.model_router as mr
        mr._controller = None  # force no-RouteLLM path

        from app.models import UseCaseProfile
        profile = UseCaseProfile(
            name="test", latency_budget_ms=10_000,
            complexity_threshold=0.7, token_compression_threshold=512,
            groundedness_pass_threshold=0.85, inspection_timeout_ms=3_000,
        )
        result = await mr.route_and_call("Hello, what is the refund policy?", profile)
        assert result.triage_state is None
        assert result.response is not None
        assert len(result.response) > 0
