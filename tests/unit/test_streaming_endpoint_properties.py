"""Property-based tests for the SSE Streaming endpoint.

Tasks: 10.2, 10.3, 10.4, 10.5
Properties:
  SSE-1 — Non-streaming endpoint unchanged: POST /v1/chat must return the same
           ChatResponse structure and status codes after the streaming router is added.
  SSE-2 — HARD_BLOCK before streaming: when Orchestrator returns HARD_BLOCK, the SSE
           response body must contain zero 'data:' frames.
  SSE-3 — Violation severs stream: when a chunk fails output validation mid-stream,
           the next and only subsequent frame is 'data: [REDACTED DUE TO POLICY]'.
  SSE-4 — Clean stream ends with DONE: when all chunks pass, the final frame is
           'data: [DONE]'.

Requirements: 4.2, 4.4, 4.7, 4.8
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.main import app
from app.models import UseCaseProfile, GuardrailsVerdict


# ---------------------------------------------------------------------------
# Module-level test client (avoids hypothesis function-scoped fixture warning)
# ---------------------------------------------------------------------------

_CLIENT = TestClient(app, raise_server_exceptions=False)


def setup_module(_module: object) -> None:
    _CLIENT.__enter__()


def teardown_module(_module: object) -> None:
    _CLIENT.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_profile(profile: UseCaseProfile) -> None:
    app.state.policy_loader._profiles[profile.name] = profile


def _remove_profile(name: str) -> None:
    app.state.policy_loader._profiles.pop(name, None)


def _make_profile(name: str = "stream_test", **overrides) -> UseCaseProfile:
    return UseCaseProfile.model_validate({
        "name": name,
        "latency_budget_ms": 30_000,
        "complexity_threshold": 0.7,
        "token_compression_threshold": 512,
        "groundedness_pass_threshold": 0.85,
        "inspection_timeout_ms": 5_000,
        "cache_enabled": False,
        **overrides,
    })


def _parse_sse_frames(body: bytes) -> list[str]:
    """Extract the data values from SSE 'data: ...' lines."""
    lines = body.decode("utf-8", errors="replace").split("\n")
    frames = []
    for line in lines:
        line = line.rstrip("\r")
        if line.startswith("data: "):
            frames.append(line[len("data: "):])
    return frames


# ---------------------------------------------------------------------------
# SSE-1: Non-streaming endpoint (POST /v1/chat) is unchanged
# Requirements: 4.2
# ---------------------------------------------------------------------------

class TestSSE1NonStreamingEndpointUnchanged:
    """Property SSE-1: POST /v1/chat must still work after adding streaming router."""

    def test_valid_request_returns_200_with_chat_response_keys(self) -> None:
        """POST /v1/chat must return 200 with the standard ChatResponse fields."""
        profile = _make_profile("sse1_valid")
        _add_profile(profile)

        async def _fast_pipeline(ctx) -> None:
            from app.models import TriageResult
            ctx.triage_result = TriageResult(
                triage_state="PASS_AND_DELIVER",
                blocking_reason=None,
                response_content="hello",
            )

        saved = app.state.pipeline_fn
        app.state.pipeline_fn = _fast_pipeline
        try:
            resp = _CLIENT.post(
                "/v1/chat",
                json={"prompt": "hi", "use_case_profile": "sse1_valid"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "request_id" in data
            assert "triage_state" in data
            assert "latency_ms" in data
        finally:
            app.state.pipeline_fn = saved
            _remove_profile("sse1_valid")

    def test_empty_prompt_still_returns_422(self) -> None:
        """422 for empty prompt must be unaffected by the streaming router."""
        resp = _CLIENT.post(
            "/v1/chat",
            json={"prompt": "", "use_case_profile": "customer_chatbot"},
        )
        assert resp.status_code == 422

    @given(st.text(min_size=1, max_size=200))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_non_streaming_responses_have_correct_schema(self, prompt: str) -> None:
        """For any valid prompt, POST /v1/chat must return required keys."""
        profile = _make_profile("sse1_schema")
        _add_profile(profile)

        async def _pipeline(ctx) -> None:
            from app.models import TriageResult
            ctx.triage_result = TriageResult(
                triage_state="HARD_BLOCK",
                blocking_reason="TEST",
                response_content=None,
            )

        saved = app.state.pipeline_fn
        app.state.pipeline_fn = _pipeline
        try:
            resp = _CLIENT.post(
                "/v1/chat",
                json={"prompt": prompt, "use_case_profile": "sse1_schema"},
            )
            # Must not be 500; schema must be intact
            assert resp.status_code in (200, 422, 503)
            if resp.status_code == 200:
                data = resp.json()
                assert "triage_state" in data
        finally:
            app.state.pipeline_fn = saved
            _remove_profile("sse1_schema")


# ---------------------------------------------------------------------------
# SSE-2: HARD_BLOCK before streaming → zero data frames
# Requirements: 4.4
# ---------------------------------------------------------------------------

class TestSSE2HardBlockBeforeStreaming:
    """Property SSE-2: HARD_BLOCK must produce zero SSE data frames."""

    def test_hard_block_produces_no_sse_frames(self) -> None:
        """When orchestrator hard-blocks, the streaming response body is empty of data frames."""
        profile = _make_profile("sse2_hardblock")
        _add_profile(profile)

        # Patch run_orchestrator to set HARD_BLOCK
        async def _blocking_orchestrator(ctx, pii_engine) -> None:
            ctx.upstream_triage_state = "HARD_BLOCK"

        with patch("app.judges.orchestrator.run_orchestrator", _blocking_orchestrator):
            resp = _CLIENT.post(
                "/v1/chat/stream",
                json={"prompt": "attack", "use_case_profile": "sse2_hardblock"},
            )

        frames = _parse_sse_frames(resp.content)
        assert frames == [], f"Expected zero frames for HARD_BLOCK, got: {frames}"
        _remove_profile("sse2_hardblock")

    @given(st.text(min_size=1, max_size=200))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_hard_block_zero_frames_for_any_prompt(self, prompt: str) -> None:
        """For any prompt that triggers HARD_BLOCK, zero data frames must be emitted."""
        name = f"sse2_prop_{hash(prompt) & 0xFFFF}"
        profile = _make_profile(name)
        _add_profile(profile)

        async def _blocking_orchestrator(ctx, pii_engine) -> None:
            ctx.upstream_triage_state = "HARD_BLOCK"

        with patch("app.judges.orchestrator.run_orchestrator", _blocking_orchestrator):
            resp = _CLIENT.post(
                "/v1/chat/stream",
                json={"prompt": prompt, "use_case_profile": name},
            )

        frames = _parse_sse_frames(resp.content)
        assert frames == [], f"Expected zero frames, got: {frames}"
        _remove_profile(name)


# ---------------------------------------------------------------------------
# SSE-3: Violation severs stream
# Requirements: 4.7
# ---------------------------------------------------------------------------

class TestSSE3ViolationSeversStream:
    """Property SSE-3: mid-stream violation must produce [REDACTED DUE TO POLICY] and nothing after."""

    def test_failing_validator_produces_redacted_frame(self) -> None:
        """When validate_output returns passed=False (non-fix), emit REDACTED and close."""
        profile = _make_profile("sse3_violation", cache_enabled=False)
        _add_profile(profile)

        async def _pass_orchestrator(ctx, pii_engine) -> None:
            pass  # no block

        failing_verdict = GuardrailsVerdict(passed=False, action="exception", triggered_validator="toxic")

        async def _token_stream(*args, **kwargs):
            yield "Hello"
            yield " world."

        with (
            patch("app.judges.orchestrator.run_orchestrator", _pass_orchestrator),
            patch("app.ingress.streaming_router.validate_output", AsyncMock(return_value=failing_verdict)),
            patch("app.ingress.streaming_router._stream_tokens_from_llm", _token_stream),
        ):
            resp = _CLIENT.post(
                "/v1/chat/stream",
                json={"prompt": "test", "use_case_profile": "sse3_violation"},
            )

        frames = _parse_sse_frames(resp.content)
        # The REDACTED frame must appear
        assert "[REDACTED DUE TO POLICY]" in frames, f"Frames: {frames}"
        # Nothing must come after the REDACTED frame
        redacted_idx = frames.index("[REDACTED DUE TO POLICY]")
        assert redacted_idx == len(frames) - 1, (
            f"Expected REDACTED to be the last frame, but frames after: {frames[redacted_idx+1:]}"
        )
        _remove_profile("sse3_violation")


# ---------------------------------------------------------------------------
# SSE-4: Clean stream ends with [DONE]
# Requirements: 4.8
# ---------------------------------------------------------------------------

class TestSSE4CleanStreamEndsDone:
    """Property SSE-4: clean stream must have [DONE] as its final frame."""

    def test_clean_stream_final_frame_is_done(self) -> None:
        """When all chunks pass validation, the final frame must be [DONE]."""
        profile = _make_profile("sse4_clean", cache_enabled=False)
        _add_profile(profile)

        async def _pass_orchestrator(ctx, pii_engine) -> None:
            pass

        passing_verdict = GuardrailsVerdict(passed=True)

        async def _token_stream(*args, **kwargs):
            yield "All "
            yield "good. "
            yield "Done!"

        with (
            patch("app.judges.orchestrator.run_orchestrator", _pass_orchestrator),
            patch("app.ingress.streaming_router.validate_output", AsyncMock(return_value=passing_verdict)),
            patch("app.ingress.streaming_router._stream_tokens_from_llm", _token_stream),
            patch("app.ingress.streaming_router.audit", AsyncMock(
                return_value=MagicMock(groundedness_score=0.95, nli_label=None)
            )),
        ):
            resp = _CLIENT.post(
                "/v1/chat/stream",
                json={"prompt": "hello", "use_case_profile": "sse4_clean"},
            )

        frames = _parse_sse_frames(resp.content)
        assert frames, "Expected at least one SSE frame"
        assert frames[-1] == "[DONE]", (
            f"Expected last frame to be [DONE], got: {frames[-1]!r}. All frames: {frames}"
        )
        _remove_profile("sse4_clean")

    @given(
        tokens=st.lists(
            st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz "),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    def test_clean_stream_always_ends_with_done(self, tokens: list[str]) -> None:
        """For any token sequence that passes validation, final frame must be [DONE]."""
        name = f"sse4_prop_{abs(hash(str(tokens))) & 0xFFFF}"
        profile = _make_profile(name, cache_enabled=False)
        _add_profile(profile)

        async def _pass_orchestrator(ctx, pii_engine) -> None:
            pass

        passing_verdict = GuardrailsVerdict(passed=True)

        async def _token_stream(*args, **kwargs):
            for t in tokens:
                yield t

        with (
            patch("app.judges.orchestrator.run_orchestrator", _pass_orchestrator),
            patch("app.ingress.streaming_router.validate_output", AsyncMock(return_value=passing_verdict)),
            patch("app.ingress.streaming_router._stream_tokens_from_llm", _token_stream),
            patch("app.ingress.streaming_router.audit", AsyncMock(
                return_value=MagicMock(groundedness_score=0.95, nli_label=None)
            )),
        ):
            resp = _CLIENT.post(
                "/v1/chat/stream",
                json={"prompt": "test", "use_case_profile": name},
            )

        frames = _parse_sse_frames(resp.content)
        assert frames, "Expected at least one frame"
        assert frames[-1] == "[DONE]", f"Last frame: {frames[-1]!r}"
        _remove_profile(name)
