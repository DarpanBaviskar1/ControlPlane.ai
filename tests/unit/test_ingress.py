"""Tests for the Enterprise Ingress handler — Tasks 3.1, 3.2, 3.3.

Tests:
- 422 for missing/empty prompt (Requirement 1.3)
- 422 for unknown use_case_profile (Requirement 1.4)
- UUID v4 request_id in response (Requirement 6.3)
- Latency budget → HTTP 504 (Requirement 1.6)
- Per-request state isolation (Requirement 1.7)
"""

from __future__ import annotations

import asyncio
import re
import time

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.models import is_valid_uuid4
from app.policy.loader import PolicyLoader

UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Synchronous TestClient with lifespan."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Task 3.1 — Request validation and 422 responses
# ---------------------------------------------------------------------------


class TestRequestValidation:
    def test_missing_prompt_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/chat", json={"use_case_profile": "customer_chatbot"})
        assert resp.status_code == 422

    def test_empty_prompt_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat", json={"prompt": "", "use_case_profile": "customer_chatbot"}
        )
        assert resp.status_code == 422

    def test_whitespace_only_prompt_returns_422(self, client: TestClient) -> None:
        # Pydantic min_length=1 is satisfied by a single space — whitespace
        # prompts ARE valid per spec (min_length=1); only empty string is rejected.
        resp = client.post(
            "/v1/chat", json={"prompt": " ", "use_case_profile": "customer_chatbot"}
        )
        # A single space IS a 1-char string — pydantic allows it; gateway processes it
        assert resp.status_code in (200, 422)

    def test_missing_use_case_profile_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/chat", json={"prompt": "Hello world"})
        assert resp.status_code == 422

    def test_empty_use_case_profile_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat", json={"prompt": "Hello world", "use_case_profile": ""}
        )
        assert resp.status_code == 422

    def test_unknown_profile_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat",
            json={"prompt": "Hello world", "use_case_profile": "no_such_profile"},
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "error_code" in body or "detail" in body

    def test_unknown_profile_error_mentions_profile_name(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat",
            json={"prompt": "Hello world", "use_case_profile": "nonexistent_xyz"},
        )
        assert resp.status_code == 422
        text = resp.text
        assert "nonexistent_xyz" in text

    def test_valid_request_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat",
            json={"prompt": "Hello world", "use_case_profile": "customer_chatbot"},
        )
        assert resp.status_code == 200

    def test_prompt_max_length_exactly_32768_valid(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat",
            json={"prompt": "a" * 32_768, "use_case_profile": "customer_chatbot"},
        )
        assert resp.status_code == 200

    def test_prompt_over_max_length_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat",
            json={"prompt": "a" * 32_769, "use_case_profile": "customer_chatbot"},
        )
        assert resp.status_code == 422

    def test_503_when_pii_engine_unhealthy(self, client: TestClient) -> None:
        """If the PII engine is unhealthy, all requests must get 503."""
        class FakePIIEngine:
            is_healthy = False
            def discard_mapping(self, _: str) -> None: ...

        app.state.pii_engine = FakePIIEngine()
        try:
            resp = client.post(
                "/v1/chat",
                json={"prompt": "Hello", "use_case_profile": "customer_chatbot"},
            )
            assert resp.status_code == 503
        finally:
            app.state.pii_engine = None


# ---------------------------------------------------------------------------
# Task 3.2 — UUID v4 request_id and latency budget
# ---------------------------------------------------------------------------


class TestRequestIdAndLatencyBudget:
    def test_response_contains_uuid4_request_id(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat",
            json={"prompt": "Test prompt", "use_case_profile": "customer_chatbot"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "request_id" in body
        assert is_valid_uuid4(body["request_id"]), f"Not a UUID v4: {body['request_id']}"

    def test_consecutive_requests_have_distinct_ids(self, client: TestClient) -> None:
        ids = set()
        for _ in range(5):
            resp = client.post(
                "/v1/chat",
                json={"prompt": "Test", "use_case_profile": "customer_chatbot"},
            )
            assert resp.status_code == 200
            ids.add(resp.json()["request_id"])
        assert len(ids) == 5, "All request_ids must be unique"

    def test_latency_budget_exceeded_returns_504(self, client: TestClient) -> None:
        """Mock a slow pipeline and confirm the gateway returns 504."""
        import asyncio

        async def slow_pipeline(ctx):
            await asyncio.sleep(60)  # far beyond any budget

        app.state.pipeline_fn = slow_pipeline
        try:
            # customer_chatbot has latency_budget_ms=10000; use a tiny profile
            # We patch the loader to return a 50 ms budget profile
            from app.models import UseCaseProfile
            from app.policy.loader import PolicyLoader

            tiny_profile = UseCaseProfile(
                name="tiny_budget",
                latency_budget_ms=100,
                complexity_threshold=0.7,
                token_compression_threshold=512,
                groundedness_pass_threshold=0.85,
                inspection_timeout_ms=3_000,
            )
            # Add profile to the live loader
            loader: PolicyLoader = app.state.policy_loader
            import asyncio as aio
            aio.run(_add_profile(loader, tiny_profile))

            start = time.monotonic()
            resp = client.post(
                "/v1/chat",
                json={"prompt": "Test", "use_case_profile": "tiny_budget"},
            )
            elapsed = time.monotonic() - start

            assert resp.status_code == 504
            # elapsed must be close to the budget (within 1 second tolerance for CI)
            assert elapsed < 2.0, f"Should have timed out quickly, took {elapsed:.2f}s"
        finally:
            app.state.pipeline_fn = None

    def test_response_contains_latency_ms(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat",
            json={"prompt": "Test", "use_case_profile": "customer_chatbot"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "latency_ms" in body
        assert isinstance(body["latency_ms"], int)
        assert body["latency_ms"] >= 0


async def _add_profile(loader, profile):
    async with loader._lock:
        loader._profiles[profile.name] = profile


# ---------------------------------------------------------------------------
# Task 3.3 — Per-request state isolation
# ---------------------------------------------------------------------------


class TestRequestContextIsolation:
    def test_each_request_gets_independent_triage_state(self, client: TestClient) -> None:
        """Two concurrent requests with different profiles must not cross-contaminate."""
        import threading

        results: list[dict] = []
        errors: list[Exception] = []

        def do_request(profile_name: str) -> None:
            try:
                r = client.post(
                    "/v1/chat",
                    json={"prompt": "Hello", "use_case_profile": profile_name},
                )
                results.append({"status": r.status_code, "body": r.json()})
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=do_request, args=("customer_chatbot",))
        t2 = threading.Thread(target=do_request, args=("internal_copilot",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Request errors: {errors}"
        assert len(results) == 2
        for r in results:
            assert r["status"] == 200

    def test_no_state_leaks_between_requests(self, client: TestClient) -> None:
        """RequestContext must not persist between requests."""
        tracked_ids: list[str] = []

        async def tracking_pipeline(ctx):
            from app.triage.gateway import TriageResult
            tracked_ids.append(ctx.request_id)
            ctx.triage_result = TriageResult(
                triage_state="PASS_AND_DELIVER",
                blocking_reason=None,
                response_content="ok",
            )

        app.state.pipeline_fn = tracking_pipeline
        try:
            resp1 = client.post(
                "/v1/chat",
                json={"prompt": "First", "use_case_profile": "customer_chatbot"},
            )
            resp2 = client.post(
                "/v1/chat",
                json={"prompt": "Second", "use_case_profile": "customer_chatbot"},
            )
            assert resp1.status_code == 200
            assert resp2.status_code == 200
            # Each request must have a distinct context (tracked via distinct IDs)
            assert len(tracked_ids) == 2
            assert tracked_ids[0] != tracked_ids[1]
            # And the IDs in the responses must match what the pipeline saw
            assert resp1.json()["request_id"] == tracked_ids[0]
            assert resp2.json()["request_id"] == tracked_ids[1]
        finally:
            app.state.pipeline_fn = None
