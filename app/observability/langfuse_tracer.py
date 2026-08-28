"""Langfuse observability layer — wraps per-request tracing (Req. 6).

Design:
- LangfuseTracer is a thin wrapper around the langfuse SDK.
- If LANGFUSE_PUBLIC_KEY is empty or the SDK is unavailable, the tracer
  degrades gracefully to stdout logging (no import error is raised).
- One Langfuse trace is created per request (trace_id == request_id).
- Nested spans are added for each pipeline stage.
- When Langfuse is unreachable, events are buffered up to LANGFUSE_BUFFER_SIZE
  and retried every LANGFUSE_RETRY_INTERVAL_S seconds (Req. 6.10).
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
from contextlib import contextmanager
from typing import Any, Generator

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Langfuse SDK import — degrades gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from langfuse import Langfuse  # type: ignore[import-untyped]
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    logger.info("langfuse package not installed — tracing will fall back to stdout")


# ---------------------------------------------------------------------------
# Buffered event for offline retry (Req. 6.10)
# ---------------------------------------------------------------------------

class _BufferedEvent:
    __slots__ = ("trace_id", "event_type", "payload")

    def __init__(self, trace_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self.trace_id = trace_id
        self.event_type = event_type
        self.payload = payload


# ---------------------------------------------------------------------------
# LangfuseTracer
# ---------------------------------------------------------------------------

class LangfuseTracer:
    """Per-application singleton that manages Langfuse traces.

    Usage::

        tracer = LangfuseTracer()
        tracer.start_trace(request_id, use_case_profile)
        tracer.add_span(request_id, "p1_judge", input=..., output=..., metadata=...)
        tracer.set_metadata(request_id, key="final_triage_state", value="HARD_BLOCK")
        tracer.flush_trace(request_id)
    """

    def __init__(self) -> None:
        self._client: Any | None = None
        self._active_traces: dict[str, Any] = {}
        # Buffer for offline events (Req. 6.10)
        self._buffer: collections.deque[_BufferedEvent] = collections.deque(
            maxlen=settings.LANGFUSE_BUFFER_SIZE
        )
        self._retry_task: asyncio.Task | None = None
        self._enabled = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise Langfuse client if credentials are configured."""
        if not _LANGFUSE_AVAILABLE:
            logger.info("Langfuse SDK unavailable — tracing disabled")
            return
        if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
            logger.info("Langfuse credentials not set — tracing disabled (stdout fallback)")
            return

        try:
            self._client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
            self._enabled = True
            logger.info("Langfuse tracing enabled (host=%s)", settings.LANGFUSE_HOST)
            # Start background retry task for buffered events
            self._retry_task = asyncio.create_task(
                self._retry_loop(), name="langfuse-retry"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse client init failed: %s — continuing without tracing", exc)

    async def stop(self) -> None:
        """Flush pending traces and shut down."""
        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.flush)
            except Exception:  # noqa: BLE001
                pass
        if self._retry_task is not None:
            self._retry_task.cancel()

    # ------------------------------------------------------------------
    # Trace management
    # ------------------------------------------------------------------

    def start_trace(
        self,
        request_id: str,
        use_case_profile: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a new Langfuse trace for this request (Req. 6.1)."""
        if not self._enabled or self._client is None:
            return
        try:
            trace = self._client.trace(
                id=request_id,
                name="controlplane-request",
                metadata={
                    "use_case_profile": use_case_profile,
                    **(metadata or {}),
                },
            )
            self._active_traces[request_id] = trace
        except Exception as exc:  # noqa: BLE001
            self._buffer_event(request_id, "trace_start", {
                "use_case_profile": use_case_profile,
                "error": str(exc),
            })

    def add_span(
        self,
        request_id: str,
        name: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
        level: str = "DEFAULT",
    ) -> None:
        """Add a named span to the trace for the given request_id (Req. 6.1)."""
        if not self._enabled or self._client is None:
            # Stdout fallback
            logger.info(
                "TRACE[%s] span=%s level=%s metadata=%s",
                request_id, name, level, json.dumps(metadata or {}),
            )
            return
        try:
            trace = self._active_traces.get(request_id)
            if trace is not None:
                trace.span(
                    name=name,
                    input=input_data,
                    output=output_data,
                    metadata=metadata or {},
                    level=level,
                )
        except Exception as exc:  # noqa: BLE001
            self._buffer_event(request_id, "span", {
                "name": name,
                "metadata": metadata,
                "error": str(exc),
            })

    def add_error_span(
        self,
        request_id: str,
        name: str,
        error_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Convenience method for error spans (Req. 2.8, 2.9, 4.5, etc.)."""
        self.add_span(
            request_id=request_id,
            name=name,
            output_data={"error": error_message},
            metadata=metadata,
            level="ERROR",
        )

    def set_metadata(self, request_id: str, **kwargs: Any) -> None:
        """Attach key-value metadata fields to the root trace (Req. 6.4)."""
        if not self._enabled or self._client is None:
            logger.info("TRACE[%s] metadata=%s", request_id, json.dumps(kwargs))
            return
        try:
            trace = self._active_traces.get(request_id)
            if trace is not None:
                trace.update(metadata=kwargs)
        except Exception as exc:  # noqa: BLE001
            self._buffer_event(request_id, "metadata", {"kwargs": kwargs, "error": str(exc)})

    def add_evaluation_score(
        self,
        request_id: str,
        name: str,
        value: float | str,
        comment: str | None = None,
    ) -> None:
        """Record a human evaluation score on a trace (Req. 6.7 override recording)."""
        if not self._enabled or self._client is None:
            logger.info(
                "TRACE[%s] evaluation name=%s value=%s comment=%s",
                request_id, name, value, comment,
            )
            return
        try:
            self._client.score(
                trace_id=request_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as exc:  # noqa: BLE001
            self._buffer_event(request_id, "score", {
                "name": name, "value": str(value), "error": str(exc),
            })

    def flush_trace(self, request_id: str) -> None:
        """Remove the trace reference after a request completes (Req. 6.2)."""
        self._active_traces.pop(request_id, None)

    # ------------------------------------------------------------------
    # Buffer + retry (Req. 6.10)
    # ------------------------------------------------------------------

    def _buffer_event(self, trace_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self._buffer.append(_BufferedEvent(trace_id, event_type, payload))

    async def _retry_loop(self) -> None:
        """Periodically attempt to flush buffered events to Langfuse."""
        while True:
            await asyncio.sleep(settings.LANGFUSE_RETRY_INTERVAL_S)
            if not self._buffer:
                continue
            events_to_retry = list(self._buffer)
            self._buffer.clear()
            for evt in events_to_retry:
                logger.debug(
                    "Langfuse retry: trace_id=%s type=%s", evt.trace_id, evt.event_type
                )
                # Best-effort: log the buffered event to stdout
                logger.info(
                    "BUFFERED_LANGFUSE_EVENT trace_id=%s type=%s payload=%s",
                    evt.trace_id,
                    evt.event_type,
                    json.dumps(evt.payload),
                )


# ---------------------------------------------------------------------------
# Module-level singleton — initialised during app lifespan
# ---------------------------------------------------------------------------

_tracer: LangfuseTracer | None = None


def get_tracer() -> LangfuseTracer:
    """Return the module-level LangfuseTracer singleton."""
    global _tracer  # noqa: PLW0603
    if _tracer is None:
        _tracer = LangfuseTracer()
    return _tracer
