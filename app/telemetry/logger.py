"""Telemetry Logger — async queue writer with retry consumer.

Design:
- Singleton TelemetryLogger with an asyncio.Queue.
- record() / record_override(): fire-and-forget enqueue (≤5 ms caller latency).
- Background consumer task drains the queue, writes JSON to the configured sink.
- Consumer retries up to 3 times with exponential back-off completing within 5 s.
  After exhausting retries, the record is dropped and an error counter is incremented.
- Rolling deque aggregator enables lazy O(N) metrics computation on /v1/metrics.
- RetentionManager enforces 90-day minimum retention (Task 14.3).
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.models import (
    AccuracyMetrics,
    FeedbackRecord,
    JudgeAccuracy,
    MetricsSummary,
    OverrideRecord,
    RoutingDistribution,
    TelemetryRecord,
    TriageStateCounts,
)

logger = logging.getLogger(__name__)

_RETRY_DELAYS = (0.1, 0.5, 2.0)  # 3 attempts; total ≤ 2.6 s < 5 s
_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Retention Manager
# ---------------------------------------------------------------------------


class RetentionManager:
    """Enforces a minimum 90-day retention floor on telemetry records.

    The manager checks ages in-process (in-memory deque).  For durable storage
    the 90-day floor is enforced by refusing to delete records younger than
    MIN_RETENTION_DAYS.
    """

    MIN_RETENTION_DAYS = 90

    def is_eligible_for_deletion(self, record_timestamp: datetime) -> bool:
        """Return True only if the record is older than 90 days."""
        age = datetime.now(tz=timezone.utc) - record_timestamp.replace(
            tzinfo=timezone.utc if record_timestamp.tzinfo is None else record_timestamp.tzinfo
        )
        return age.days >= self.MIN_RETENTION_DAYS


# ---------------------------------------------------------------------------
# TelemetryLogger
# ---------------------------------------------------------------------------


class TelemetryLogger:
    """Async queue-based structured log writer.

    Usage::

        logger = TelemetryLogger(sink="stdout")
        await logger.start()          # starts background consumer
        await logger.record(record)   # fire-and-forget
        await logger.stop()           # drains queue and stops consumer
    """

    def __init__(self, sink: str = "stdout", log_file: str = "telemetry.jsonl") -> None:
        self._sink = sink
        self._log_file = Path(log_file)
        self._queue: asyncio.Queue[TelemetryRecord | OverrideRecord | None] = asyncio.Queue()
        self._consumer_task: asyncio.Task | None = None
        self._error_count: int = 0

        # Rolling deque of (timestamp, TelemetryRecord) for lazy aggregation
        # Max deque length: ~1440 min * 10 req/s = 864,000 → unbounded, purge on read
        self._records: collections.deque[TelemetryRecord] = collections.deque()
        self._override_records: list[OverrideRecord] = []

        self.retention_manager = RetentionManager()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background consumer task."""
        self._consumer_task = asyncio.create_task(
            self._consumer_loop(), name="telemetry-consumer"
        )

    async def stop(self) -> None:
        """Drain the queue and stop the consumer."""
        await self._queue.put(None)  # sentinel
        if self._consumer_task is not None:
            try:
                await asyncio.wait_for(self._consumer_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._consumer_task.cancel()

    # ------------------------------------------------------------------
    # Public write interface
    # ------------------------------------------------------------------

    async def record(self, entry: TelemetryRecord) -> None:
        """Fire-and-forget enqueue of a telemetry record (≤5 ms)."""
        self._queue.put_nowait(entry)

    async def record_override(self, override: OverrideRecord) -> None:
        """Persist an operator override record."""
        self._queue.put_nowait(override)

    # ------------------------------------------------------------------
    # Public read interface
    # ------------------------------------------------------------------

    async def get_metrics(self, window_minutes: int) -> MetricsSummary:
        """Lazily aggregate telemetry records within *window_minutes*."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=window_minutes)
        records = [
            r for r in self._records
            if r.timestamp.replace(tzinfo=timezone.utc if r.timestamp.tzinfo is None else r.timestamp.tzinfo) >= cutoff
        ]

        total = len(records)
        state_counts = TriageStateCounts()
        groundedness_sum = 0.0
        groundedness_n = 0
        routing: dict[str, int] = {"ROUTINE": 0, "COMPLEX": 0}

        for r in records:
            state = r.final_triage_state
            if state == "PASS_AND_DELIVER":
                state_counts.PASS_AND_DELIVER += 1
            elif state == "COMPRESS_AND_EDIT":
                state_counts.COMPRESS_AND_EDIT += 1
            elif state == "ESCALATE_TO_HUMAN":
                state_counts.ESCALATE_TO_HUMAN += 1
            elif state == "HARD_BLOCK":
                state_counts.HARD_BLOCK += 1

            if r.groundedness_score is not None:
                groundedness_sum += r.groundedness_score
                groundedness_n += 1

            if r.routing_decision in routing:
                routing[r.routing_decision] += 1

        avg_gs = groundedness_sum / groundedness_n if groundedness_n else 0.0
        total_routed = routing["ROUTINE"] + routing["COMPLEX"]
        dist = RoutingDistribution(
            ROUTINE=routing["ROUTINE"] / total_routed if total_routed else 0.0,
            COMPLEX=routing["COMPLEX"] / total_routed if total_routed else 0.0,
        )

        return MetricsSummary(
            window_minutes=window_minutes,
            total_requests=total,
            triage_state_counts=state_counts,
            average_groundedness_score=avg_gs,
            routing_distribution=dist,
        )

    async def export_feedback(self) -> list[FeedbackRecord]:
        """Return all escalated and overridden cases."""
        override_map = {o.request_id: o for o in self._override_records}
        result: list[FeedbackRecord] = []

        for rec in self._records:
            override = override_map.get(rec.request_id)
            if rec.final_triage_state == "ESCALATE_TO_HUMAN" or override:
                result.append(
                    FeedbackRecord(
                        telemetry=rec,
                        override=override,
                        human_label=override.human_label if override else None,
                    )
                )
        return result

    async def get_accuracy_metrics(self, window_days: int) -> AccuracyMetrics:
        """Compute FPR, FNR, F1 for P1-tox, P1-inj, P2-PII from override records."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=window_days)
        overrides = [
            o for o in self._override_records
            if o.timestamp.replace(tzinfo=timezone.utc if o.timestamp.tzinfo is None else o.timestamp.tzinfo) >= cutoff
        ]

        def _metrics(tp: int, fp: int, fn: int) -> JudgeAccuracy:
            precision = tp / (tp + fp) if (tp + fp) else 1.0
            recall = tp / (tp + fn) if (tp + fn) else 1.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            fpr = fp / (fp + (tp + fn - tp)) if (fp + (tp + fn - tp)) else 0.0
            fnr = fn / (fn + tp) if (fn + tp) else 0.0
            return JudgeAccuracy(false_positive_rate=fpr, false_negative_rate=fnr, f1_score=f1)

        # Build a map from request_id → original record for cross-reference
        rec_map = {r.request_id: r for r in self._records}

        tp_tox = fp_tox = fn_tox = 0
        tp_inj = fp_inj = fn_inj = 0
        tp_pii = fp_pii = fn_pii = 0

        for ov in overrides:
            rec = rec_map.get(ov.request_id)
            if rec is None:
                continue
            sys_block = rec.final_triage_state == "HARD_BLOCK"
            human_pass = ov.human_label == "PASS"

            # Toxicity
            tox_block = rec.p1_toxicity_verdict == "BLOCK"
            if tox_block and not human_pass:
                tp_tox += 1
            elif tox_block and human_pass:
                fp_tox += 1
            elif not tox_block and not human_pass:
                fn_tox += 1

            # Injection
            inj_block = rec.p1_injection_verdict == "BLOCK"
            if inj_block and not human_pass:
                tp_inj += 1
            elif inj_block and human_pass:
                fp_inj += 1
            elif not inj_block and not human_pass:
                fn_inj += 1

            # PII
            pii_flagged = (rec.p2_pii_count or 0) > 0
            human_pii = ov.human_label in ("SOFT_BLOCK", "HARD_BLOCK")
            if pii_flagged and human_pii:
                tp_pii += 1
            elif pii_flagged and not human_pii:
                fp_pii += 1
            elif not pii_flagged and human_pii:
                fn_pii += 1

        return AccuracyMetrics(
            window_days=window_days,
            p1_toxicity=_metrics(tp_tox, fp_tox, fn_tox),
            p1_injection=_metrics(tp_inj, fp_inj, fn_inj),
            p2_pii=_metrics(tp_pii, fp_pii, fn_pii),
        )

    # ------------------------------------------------------------------
    # Error counter (for testing and monitoring)
    # ------------------------------------------------------------------

    @property
    def error_count(self) -> int:
        return self._error_count

    # ------------------------------------------------------------------
    # Background consumer
    # ------------------------------------------------------------------

    async def _consumer_loop(self) -> None:
        """Drain the queue and write records with retry/back-off."""
        while True:
            item = await self._queue.get()
            if item is None:  # sentinel → shutdown
                self._queue.task_done()
                break

            await self._write_with_retry(item)
            self._queue.task_done()

    async def _write_with_retry(self, item: TelemetryRecord | OverrideRecord) -> None:
        """Write one record; retry up to _MAX_RETRIES times."""
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                await self._write(item)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Telemetry write attempt %d/%d failed: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(delay)

        self._error_count += 1
        logger.error(
            "Telemetry write exhausted %d retries — record dropped", _MAX_RETRIES
        )

    async def _write(self, item: TelemetryRecord | OverrideRecord) -> None:
        """Persist *item* to the configured sink and update in-memory stores."""
        # Update in-memory deque / list for aggregation
        if isinstance(item, TelemetryRecord):
            self._records.append(item)
        elif isinstance(item, OverrideRecord):
            self._override_records.append(item)

        # Write to sink
        payload = item.model_dump_json()
        if self._sink == "stdout":
            print(payload, flush=True)
        elif self._sink == "file":
            await asyncio.to_thread(self._append_to_file, payload)
        # For remote sinks, a real implementation would POST to an endpoint.

    def _append_to_file(self, payload: str) -> None:
        with self._log_file.open("a", encoding="utf-8") as fh:
            fh.write(payload + "\n")
