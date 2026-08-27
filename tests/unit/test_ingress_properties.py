"""Property-based tests for the Enterprise Ingress handler.

# Feature: controlplane-ai-gateway, Property 2: Invalid Request Rejection
# Feature: controlplane-ai-gateway, Property 3: Latency Budget Enforcement
# Feature: controlplane-ai-gateway, Property 4: Concurrent Request Isolation

Validates: Requirements 1.3, 1.4, 1.6, 1.7
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.main import app
from app.models import UseCaseProfile, is_valid_uuid4

# Module-level client reused across all @given tests.
# Hypothesis warns about function-scoped fixtures; using a module-level
# client is the recommended workaround.
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


# ---------------------------------------------------------------------------
# Property 2: Invalid Request Rejection
# Validates: Requirements 1.3, 1.4
# ---------------------------------------------------------------------------


class TestProperty2InvalidRequestRejection:

    @given(st.one_of(st.just(""), st.just(None)))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_absent_or_empty_prompt_returns_422(self, bad_prompt: Any) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 2: Invalid Request Rejection
        For any absent or empty prompt, the gateway must return HTTP 422.
        """
        body: dict = {"use_case_profile": "customer_chatbot"}
        if bad_prompt is not None:
            body["prompt"] = bad_prompt
        resp = _CLIENT.post("/v1/chat", json=body)
        assert resp.status_code == 422, (
            f"Expected 422 for prompt={bad_prompt!r}, got {resp.status_code}"
        )

    @given(
        st.text(min_size=1, max_size=200).filter(
            lambda s: s not in ("customer_chatbot", "internal_copilot") and s.strip() != ""
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_unknown_profile_returns_422(self, unknown_profile: str) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 2: Invalid Request Rejection
        For any use_case_profile not in the Policy Layer, return HTTP 422.
        """
        resp = _CLIENT.post(
            "/v1/chat",
            json={"prompt": "Hello world", "use_case_profile": unknown_profile},
        )
        assert resp.status_code == 422, (
            f"Expected 422 for profile={unknown_profile!r}, got {resp.status_code}"
        )

    @given(st.text(min_size=1, max_size=256))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_valid_prompt_with_known_profile_not_422(self, prompt: str) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 2: Invalid Request Rejection
        Any non-empty prompt with a known profile must NOT return 422.
        """
        resp = _CLIENT.post(
            "/v1/chat",
            json={"prompt": prompt, "use_case_profile": "customer_chatbot"},
        )
        assert resp.status_code != 422, (
            f"Got unexpected 422 for valid prompt len={len(prompt)}: {resp.text[:200]}"
        )


# ---------------------------------------------------------------------------
# Property 3: Latency Budget Enforcement
# Validates: Requirement 1.6
# ---------------------------------------------------------------------------


class TestProperty3LatencyBudgetEnforcement:

    @given(st.integers(min_value=50, max_value=400))
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_slow_pipeline_triggers_504(self, budget_ms: int) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 3: Latency Budget Enforcement
        When the pipeline sleeps beyond the budget, the gateway returns 504.
        """
        profile_name = f"prop3_budget_{budget_ms}"
        profile = UseCaseProfile(
            name=profile_name,
            latency_budget_ms=budget_ms,
            complexity_threshold=0.7,
            token_compression_threshold=512,
            groundedness_pass_threshold=0.85,
            inspection_timeout_ms=3_000,
        )
        _add_profile(profile)

        async def slow_pipeline(ctx):
            await asyncio.sleep(60)  # far beyond any budget tested

        app.state.pipeline_fn = slow_pipeline
        try:
            start = time.monotonic()
            resp = _CLIENT.post(
                "/v1/chat",
                json={"prompt": "Test latency", "use_case_profile": profile_name},
            )
            elapsed_ms = (time.monotonic() - start) * 1000

            assert resp.status_code == 504, (
                f"Expected 504 for budget={budget_ms}ms, got {resp.status_code}"
            )
            # Must time out within budget + 500 ms CI tolerance
            assert elapsed_ms < budget_ms + 500, (
                f"Elapsed {elapsed_ms:.0f}ms too long for budget={budget_ms}ms"
            )
        finally:
            app.state.pipeline_fn = None
            _remove_profile(profile_name)

    def test_fast_pipeline_does_not_trigger_504(self) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 3: Latency Budget Enforcement
        """
        async def fast_pipeline(ctx):
            from app.triage.gateway import TriageResult
            ctx.triage_result = TriageResult(
                triage_state="PASS_AND_DELIVER",
                blocking_reason=None,
                response_content="fast",
            )

        app.state.pipeline_fn = fast_pipeline
        try:
            resp = _CLIENT.post(
                "/v1/chat",
                json={"prompt": "Quick", "use_case_profile": "customer_chatbot"},
            )
            assert resp.status_code == 200
        finally:
            app.state.pipeline_fn = None


# ---------------------------------------------------------------------------
# Property 4: Concurrent Request Isolation
# Validates: Requirement 1.7
# ---------------------------------------------------------------------------


class TestProperty4ConcurrentRequestIsolation:

    def test_concurrent_requests_get_correct_profiles(self) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 4: Concurrent Request Isolation
        """
        observed: dict[str, str] = {}
        lock = threading.Lock()

        async def tracking_pipeline(ctx):
            from app.triage.gateway import TriageResult
            with lock:
                observed[ctx.request_id] = ctx.profile.name
            await asyncio.sleep(0.005)
            ctx.triage_result = TriageResult(
                triage_state="PASS_AND_DELIVER",
                blocking_reason=None,
                response_content="ok",
            )

        app.state.pipeline_fn = tracking_pipeline
        results: list[dict] = []
        errors: list[Exception] = []

        def post(profile_name: str) -> None:
            try:
                r = _CLIENT.post(
                    "/v1/chat",
                    json={"prompt": "Hello", "use_case_profile": profile_name},
                )
                results.append({"status": r.status_code, "body": r.json(), "profile": profile_name})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=post, args=("customer_chatbot",)),
            threading.Thread(target=post, args=("internal_copilot",)),
            threading.Thread(target=post, args=("customer_chatbot",)),
            threading.Thread(target=post, args=("internal_copilot",)),
        ]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"Errors: {errors}"
            assert len(results) == 4
            for r in results:
                assert r["status"] == 200
                rid = r["body"]["request_id"]
                assert rid in observed
                assert observed[rid] == r["profile"], (
                    f"Profile mismatch: expected {r['profile']}, saw {observed[rid]}"
                )
        finally:
            app.state.pipeline_fn = None

    def test_no_placeholder_map_leaks_between_requests(self) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 4: Concurrent Request Isolation
        Each request must get its own placeholder_map instance.
        """
        maps_seen: list[int] = []

        async def capture_pipeline(ctx):
            from app.triage.gateway import TriageResult
            maps_seen.append(id(ctx.placeholder_map))
            ctx.triage_result = TriageResult(
                triage_state="PASS_AND_DELIVER",
                blocking_reason=None,
                response_content="ok",
            )

        app.state.pipeline_fn = capture_pipeline
        try:
            for _ in range(3):
                _CLIENT.post(
                    "/v1/chat",
                    json={"prompt": "Test", "use_case_profile": "customer_chatbot"},
                )
            assert len(maps_seen) == 3
            assert len(set(maps_seen)) == 3, "placeholder_map shared across requests"
        finally:
            app.state.pipeline_fn = None

    @given(st.integers(min_value=2, max_value=8))
    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_n_concurrent_requests_all_get_unique_ids(self, n: int) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 4: Concurrent Request Isolation
        """
        ids: list[str] = []
        lock = threading.Lock()
        errors: list[Exception] = []

        def post() -> None:
            try:
                r = _CLIENT.post(
                    "/v1/chat",
                    json={"prompt": "Hello", "use_case_profile": "customer_chatbot"},
                )
                if r.status_code == 200:
                    with lock:
                        ids.append(r.json()["request_id"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=post) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(ids) == n
        assert len(set(ids)) == n, "Duplicate request_ids in concurrent batch"
        for rid in ids:
            assert is_valid_uuid4(rid), f"Not a UUID v4: {rid}"
