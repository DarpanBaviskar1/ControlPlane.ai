"""Unit tests for TelemetryLogger async queue writer — Task 12.1.

Tests:
- record() enqueues without blocking
- Consumer writes records to in-memory store
- Retry on write failure: drops after 3 attempts, increments error counter
- record_override() persists OverrideRecord
- stop() drains the queue before returning
- RetentionManager refuses to delete records younger than 90 days
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.models import OverrideRecord, TelemetryRecord
from app.telemetry.logger import RetentionManager, TelemetryLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_telemetry_record(request_id: str = "req-001") -> TelemetryRecord:
    return TelemetryRecord(
        request_id=request_id,
        timestamp=datetime.now(tz=timezone.utc),
        use_case_profile="customer_chatbot",
        final_triage_state="PASS_AND_DELIVER",
        latency_ms=120,
        groundedness_unverified=False,
        pii_masking_bypassed=False,
    )


def _make_override_record(request_id: str = "req-001") -> OverrideRecord:
    return OverrideRecord(
        request_id=request_id,
        operator_id="ops-001",
        timestamp=datetime.now(tz=timezone.utc),
        original_verdict="HARD_BLOCK",
        human_label="PASS",
        stated_reason="False positive",
    )


# ---------------------------------------------------------------------------
# Basic enqueue + consume
# ---------------------------------------------------------------------------


class TestTelemetryQueueBasic:
    @pytest.mark.asyncio
    async def test_record_enqueues_without_blocking(self) -> None:
        tel = TelemetryLogger(sink="stdout")
        await tel.start()
        rec = _make_telemetry_record()
        await tel.record(rec)
        # Give the consumer a moment to process
        await asyncio.sleep(0.05)
        assert len(tel._records) == 1
        await tel.stop()

    @pytest.mark.asyncio
    async def test_multiple_records_all_consumed(self) -> None:
        tel = TelemetryLogger(sink="stdout")
        await tel.start()
        for i in range(5):
            await tel.record(_make_telemetry_record(f"req-{i}"))
        await asyncio.sleep(0.1)
        assert len(tel._records) == 5
        await tel.stop()

    @pytest.mark.asyncio
    async def test_record_override_stored(self) -> None:
        tel = TelemetryLogger(sink="stdout")
        await tel.start()
        ov = _make_override_record()
        await tel.record_override(ov)
        await asyncio.sleep(0.05)
        assert len(tel._override_records) == 1
        assert tel._override_records[0].request_id == "req-001"
        await tel.stop()

    @pytest.mark.asyncio
    async def test_stop_drains_queue(self) -> None:
        tel = TelemetryLogger(sink="stdout")
        await tel.start()
        for i in range(10):
            await tel.record(_make_telemetry_record(f"drain-{i}"))
        await tel.stop()
        assert len(tel._records) == 10


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestTelemetryRetry:
    @pytest.mark.asyncio
    async def test_error_counter_increments_after_exhausted_retries(self) -> None:
        tel = TelemetryLogger(sink="stdout")
        await tel.start()

        write_calls = 0
        original_write = tel._write

        async def _always_fail(item):
            nonlocal write_calls
            write_calls += 1
            raise OSError("disk full")

        tel._write = _always_fail
        await tel.record(_make_telemetry_record())
        # Wait for all retry attempts (≤ 3 s total back-off)
        await asyncio.sleep(3.5)
        assert tel.error_count == 1
        assert write_calls == 3  # 3 attempts exactly
        tel._write = original_write
        await tel.stop()

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self) -> None:
        tel = TelemetryLogger(sink="stdout")
        await tel.start()

        attempt = 0
        original_write = tel._write

        async def _fail_once(item):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise OSError("transient")
            await original_write(item)

        tel._write = _fail_once
        await tel.record(_make_telemetry_record("retry-ok"))
        await asyncio.sleep(0.5)
        assert tel.error_count == 0
        assert len(tel._records) == 1
        tel._write = original_write
        await tel.stop()

    @pytest.mark.asyncio
    async def test_caller_unaffected_by_write_failure(self) -> None:
        """record() must return immediately regardless of write errors."""
        tel = TelemetryLogger(sink="stdout")
        await tel.start()
        tel._write = AsyncMock(side_effect=RuntimeError("sink dead"))
        rec = _make_telemetry_record()
        # This must not raise or block the caller
        await tel.record(rec)
        await asyncio.sleep(3.5)
        assert tel.error_count == 1
        await tel.stop()


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


class TestMetricsAggregation:
    @pytest.mark.asyncio
    async def test_get_metrics_returns_correct_counts(self) -> None:
        tel = TelemetryLogger(sink="stdout")
        await tel.start()

        states = ["PASS_AND_DELIVER", "HARD_BLOCK", "HARD_BLOCK", "ESCALATE_TO_HUMAN"]
        for i, state in enumerate(states):
            rec = TelemetryRecord(
                request_id=f"metrics-{i}",
                timestamp=datetime.now(tz=timezone.utc),
                use_case_profile="customer_chatbot",
                final_triage_state=state,
                latency_ms=100,
                groundedness_unverified=False,
                pii_masking_bypassed=False,
            )
            await tel.record(rec)

        await asyncio.sleep(0.1)
        summary = await tel.get_metrics(window_minutes=60)
        assert summary.total_requests == 4
        assert summary.triage_state_counts.HARD_BLOCK == 2
        assert summary.triage_state_counts.PASS_AND_DELIVER == 1
        assert summary.triage_state_counts.ESCALATE_TO_HUMAN == 1
        await tel.stop()


# ---------------------------------------------------------------------------
# RetentionManager
# ---------------------------------------------------------------------------


class TestRetentionManager:
    def test_record_older_than_90_days_eligible(self) -> None:
        rm = RetentionManager()
        old_ts = datetime.now(tz=timezone.utc) - timedelta(days=91)
        assert rm.is_eligible_for_deletion(old_ts) is True

    def test_record_exactly_90_days_old_eligible(self) -> None:
        rm = RetentionManager()
        ts = datetime.now(tz=timezone.utc) - timedelta(days=90)
        assert rm.is_eligible_for_deletion(ts) is True

    def test_record_89_days_old_not_eligible(self) -> None:
        rm = RetentionManager()
        ts = datetime.now(tz=timezone.utc) - timedelta(days=89)
        assert rm.is_eligible_for_deletion(ts) is False

    def test_recent_record_not_eligible(self) -> None:
        rm = RetentionManager()
        ts = datetime.now(tz=timezone.utc)
        assert rm.is_eligible_for_deletion(ts) is False
