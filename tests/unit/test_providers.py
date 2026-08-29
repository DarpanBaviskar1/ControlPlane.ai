"""LiteLLM egress layer — the single LLM call path.

All tests stub the transport: the suite must never touch the network.
"""
from __future__ import annotations

import pytest

from app.router import providers


# --------------------------------------------------------------------------
# Model resolution


class TestModelResolution:
    def test_bare_model_name_gets_provider_prefix(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_PROVIDER", "google")
        monkeypatch.setattr(providers.settings, "SLM_MODEL", "gemini-2.5-flash")
        assert providers._model_for_tier("SLM") == "gemini/gemini-2.5-flash"

    def test_already_qualified_model_is_left_alone(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_PROVIDER", "google")
        monkeypatch.setattr(providers.settings, "SLM_MODEL", "openai/gpt-4o-mini")
        assert providers._model_for_tier("SLM") == "openai/gpt-4o-mini"

    def test_grok_maps_to_litellm_xai_prefix(self, monkeypatch) -> None:
        """LiteLLM calls xAI 'xai', not 'grok' — the mapping must translate."""
        monkeypatch.setattr(providers.settings, "LLM_PROVIDER", "grok")
        monkeypatch.setattr(providers.settings, "FRONTIER_MODEL", "grok-2")
        assert providers._model_for_tier("FRONTIER") == "xai/grok-2"

    def test_frontier_and_slm_are_distinct(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_PROVIDER", "openai")
        monkeypatch.setattr(providers.settings, "SLM_MODEL", "gpt-4o-mini")
        monkeypatch.setattr(providers.settings, "FRONTIER_MODEL", "gpt-4o")
        assert providers._model_for_tier("SLM") != providers._model_for_tier("FRONTIER")

    def test_legacy_fallback_model_supplies_slm_when_slm_left_default(
        self, monkeypatch
    ) -> None:
        """An existing .env with only LLM_FALLBACK_MODEL keeps working."""
        monkeypatch.setattr(providers.settings, "LLM_PROVIDER", "google")
        monkeypatch.setattr(providers.settings, "SLM_MODEL", "gpt-4o-mini")  # default
        monkeypatch.setattr(providers.settings, "LLM_FALLBACK_MODEL", "gemini-2.5-flash")
        assert providers._model_for_tier("SLM") == "gemini/gemini-2.5-flash"


# --------------------------------------------------------------------------
# Mock path (no key configured)


class TestMockPath:
    @pytest.fixture(autouse=True)
    def _force_mock(self, monkeypatch) -> None:
        """is_live() honours LLM_API_BASE as well as the key, so a test that
        blanks only the key would pass or fail on ambient config (e.g. a
        developer machine running Ollama with LLM_API_BASE set in .env).
        Blank both here; individual tests override just the field they are
        exercising."""
        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "")
        monkeypatch.setattr(providers.settings, "LLM_API_BASE", "")

    def test_is_live_false_without_key(self, monkeypatch) -> None:
        assert providers.is_live() is False

    def test_is_live_false_for_dummy_key(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "dummy-key")
        assert providers.is_live() is False

    def test_is_live_false_when_litellm_absent(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "sk-real-abc")
        monkeypatch.setattr(providers, "_HAS_LITELLM", False)
        assert providers.is_live() is False

    def test_is_live_true_with_api_base_and_no_key(self, monkeypatch) -> None:
        """Keyless local backends (Ollama, vLLM) are live via LLM_API_BASE
        alone — this is the test that actually pins Ruling 25. Key stays
        blank (from the autouse fixture); only LLM_API_BASE is set."""
        monkeypatch.setattr(providers.settings, "LLM_API_BASE", "http://localhost:11434")
        monkeypatch.setattr(providers, "_HAS_LITELLM", True)
        assert providers.is_live() is True

    def test_call_kwargs_omits_api_key_when_blank(self, monkeypatch) -> None:
        """A blank key must never be passed through as api_key='' — litellm
        would treat that as a real (empty) credential rather than 'absent'."""
        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "")
        kwargs = providers._call_kwargs("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}])
        assert "api_key" not in kwargs

    def test_call_kwargs_includes_api_key_when_present(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "sk-real-abc")
        kwargs = providers._call_kwargs("openai/gpt-4o-mini", [{"role": "user", "content": "hi"}])
        assert kwargs["api_key"] == "sk-real-abc"

    async def test_acomplete_returns_mock_without_key(self, monkeypatch) -> None:
        """P-PR-1: absent credentials degrade to mock text, never an exception."""
        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "")
        text, model = await providers.acomplete("What is your return policy?", "SLM")
        assert "30 calendar days" in text
        assert model == "mock"

    async def test_astream_yields_mock_chunks_without_key(self, monkeypatch) -> None:
        monkeypatch.setattr(providers.settings, "LLM_API_KEY", "")
        chunks = [c async for c in providers.astream("refund please", "SLM")]
        assert len(chunks) > 1                      # genuinely chunked
        assert "30 calendar days" in "".join(chunks)


# --------------------------------------------------------------------------
# Live path, with litellm stubbed


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)
        self.delta = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


@pytest.fixture
def live(monkeypatch):
    """Put the module on its live path with a stubbed litellm."""
    monkeypatch.setattr(providers.settings, "LLM_API_KEY", "sk-real-abc")
    monkeypatch.setattr(providers.settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(providers.settings, "SLM_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(providers.settings, "FRONTIER_MODEL", "gpt-4o")
    monkeypatch.setattr(providers, "_HAS_LITELLM", True)
    return monkeypatch


class TestLivePath:
    async def test_acomplete_passes_resolved_model_and_returns_text(
        self, live
    ) -> None:
        seen: dict = {}

        async def fake_acompletion(**kwargs):
            seen.update(kwargs)
            return _FakeResponse("live answer")

        live.setattr(providers, "_acompletion", fake_acompletion)
        text, model = await providers.acomplete("hello", "FRONTIER")

        assert text == "live answer"
        assert model == "openai/gpt-4o"
        assert seen["model"] == "openai/gpt-4o"
        assert seen["messages"] == [{"role": "user", "content": "hello"}]
        assert seen["num_retries"] == providers.settings.LLM_MAX_RETRIES

    async def test_system_prompt_is_prepended(self, live) -> None:
        seen: dict = {}

        async def fake_acompletion(**kwargs):
            seen.update(kwargs)
            return _FakeResponse("ok")

        live.setattr(providers, "_acompletion", fake_acompletion)
        await providers.acomplete("q", "SLM", system="be careful")

        assert seen["messages"][0] == {"role": "system", "content": "be careful"}
        assert seen["messages"][1] == {"role": "user", "content": "q"}

    async def test_falls_back_to_other_tier_on_failure(self, live) -> None:
        """P-PR-2: SLM failure retries on FRONTIER — the fallback Portkey gave us."""
        attempts: list[str] = []

        async def flaky(**kwargs):
            attempts.append(kwargs["model"])
            if kwargs["model"] == "openai/gpt-4o-mini":
                raise RuntimeError("upstream 503")
            return _FakeResponse("frontier saved it")

        live.setattr(providers, "_acompletion", flaky)
        text, model = await providers.acomplete("hello", "SLM")

        assert attempts == ["openai/gpt-4o-mini", "openai/gpt-4o"]
        assert text == "frontier saved it"
        assert model == "openai/gpt-4o"

    async def test_mock_used_when_both_tiers_fail(self, live) -> None:
        """P-PR-3: total upstream failure still returns a response, never raises."""

        async def always_fails(**kwargs):
            raise RuntimeError("everything is down")

        live.setattr(providers, "_acompletion", always_fails)
        text, model = await providers.acomplete("refund", "SLM")

        assert model == "mock"
        assert "30 calendar days" in text

    async def test_astream_yields_deltas(self, live) -> None:
        async def fake_stream(**kwargs):
            for piece in ("Hel", "lo ", "world"):
                yield _FakeResponse(piece)

        live.setattr(providers, "_acompletion_stream", fake_stream)
        chunks = [c async for c in providers.astream("hi", "SLM")]
        assert "".join(chunks) == "Hello world"

    async def test_astream_raises_on_failure_so_caller_can_emit_stream_error(
        self, live
    ) -> None:
        """Streaming cannot silently fall back mid-response: the SSE layer
        needs the exception so it can emit [STREAM_ERROR]."""

        async def broken(**kwargs):
            raise RuntimeError("stream died")
            yield  # pragma: no cover

        live.setattr(providers, "_acompletion_stream", broken)
        with pytest.raises(RuntimeError):
            [c async for c in providers.astream("hi", "SLM")]


# --------------------------------------------------------------------------
# generate_contextual_response — public, unprefixed name


class TestGenerateContextualResponse:
    def test_is_public_and_unprefixed(self) -> None:
        """The intended public name has no leading underscore (unlike the
        legacy _generate_contextual_response in model_router.py)."""
        assert hasattr(providers, "generate_contextual_response")
        assert not hasattr(providers, "_generate_contextual_response")

    def test_return_policy_prompt(self) -> None:
        text = providers.generate_contextual_response("What is your return policy?")
        assert "30 calendar days" in text
