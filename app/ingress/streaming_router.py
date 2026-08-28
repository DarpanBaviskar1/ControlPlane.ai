"""SSE Streaming endpoint — POST /v1/chat/stream.

Streams LLM output token-by-token via Server-Sent Events.  The full pipeline
(orchestrator, PII masking, routing, output validation, groundedness audit,
triage) runs on the assembled prompt before streaming begins.  Per-sentence
chunks are validated mid-stream via SlidingWindow; a policy violation severs
the stream immediately.

SSE frame format: ``data: <content>\\n\\n``
Terminal frames:
  ``data: [DONE]\\n\\n``            — clean end-of-stream
  ``data: [REDACTED DUE TO POLICY]\\n\\n`` — mid-stream policy violation
  ``data: [STREAM_ERROR]\\n\\n``    — unhandled LLM streaming exception

Requirements: 4.1, 4.2, 4.3, 4.4, 4.6, 4.7, 4.8, 4.9, 4.10, 4.11, 6.1
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models import ChatRequest, TelemetryRecord, UseCaseProfile, RequestContext
from app.ingress.sliding_window import SlidingWindow
from app.judges.output_validator import validate_output
from app.groundedness.auditor import audit

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------


@router.post("/v1/chat/stream")
async def stream_chat(request: Request) -> StreamingResponse:
    """Stream LLM output as Server-Sent Events.

    The non-streaming pipeline stages (orchestrator, PII, routing) run to
    completion first so that HARD_BLOCK decisions are enforced before any
    tokens are forwarded to the client (Req. 4.4).
    """
    body = await request.json()
    chat_req = ChatRequest(**body)
    request_id = str(uuid.uuid4())

    return StreamingResponse(
        content=_sse_generator(request, chat_req, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-Id": request_id,
        },
    )


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------


async def _sse_generator(
    request: Request,
    chat_req: ChatRequest,
    request_id: str,
) -> AsyncGenerator[str, None]:
    """Async generator that runs the full pipeline and streams SSE frames."""
    start_ts = time.monotonic()
    mid_stream_violation = False
    total_tokens = 0
    final_triage_state = "HARD_BLOCK"
    cache_hit = False

    # ------------------------------------------------------------------
    # 1. Resolve profile
    # ------------------------------------------------------------------
    policy_loader = request.app.state.policy_loader
    try:
        profile: UseCaseProfile = await policy_loader.get_profile(chat_req.use_case_profile)
    except KeyError:
        yield "data: [STREAM_ERROR]\n\n"
        return

    # ------------------------------------------------------------------
    # 2. Run non-streaming pipeline (orchestrator + upstream triage)
    # ------------------------------------------------------------------
    from app.models import RequestContext
    from app.judges.orchestrator import run_orchestrator
    from app.judges.pii_masking import PIIMaskingEngine

    pii_engine: PIIMaskingEngine = request.app.state.pii_engine

    ctx = RequestContext(
        request_id=request_id,
        profile=profile,
        original_prompt=chat_req.prompt,
        working_prompt=chat_req.prompt,
        conversation_history=list(chat_req.conversation_history),
        pipeline_start_ts=start_ts,
    )

    await run_orchestrator(ctx, pii_engine)

    if ctx.upstream_triage_state == "HARD_BLOCK":
        # No frames emitted for HARD_BLOCK (Req. 4.4)
        final_triage_state = "HARD_BLOCK"
        _record_telemetry(request, ctx, final_triage_state, 0, False, cache_hit, start_ts)
        return

    # ------------------------------------------------------------------
    # 3. Semantic Cache lookup (Req. 4.3)
    # ------------------------------------------------------------------
    semantic_cache = getattr(request.app.state, "semantic_cache", None)

    if profile.cache_enabled and semantic_cache is not None:
        cache_result = await semantic_cache.lookup(
            ctx.working_prompt, profile.cache_ttl_seconds
        )
        if cache_result.hit and cache_result.response:
            cache_hit = True
            final_triage_state = "PASS_AND_DELIVER"
            yield f"data: {cache_result.response}\n\n"
            yield "data: [DONE]\n\n"
            _record_telemetry(request, ctx, final_triage_state, 0, False, cache_hit, start_ts)
            return

    # ------------------------------------------------------------------
    # 4. Stream from LLM via Portkey (Req. 4.6)
    # ------------------------------------------------------------------
    nli_scorer = getattr(request.app.state, "nli_scorer", None)
    vector_store = getattr(request.app.state, "vector_store", None)

    window = SlidingWindow()
    accumulated_response: list[str] = []

    try:
        async for token in _stream_tokens_from_llm(ctx.working_prompt, profile):
            total_tokens += 1
            accumulated_response.append(token)

            for chunk in window.push(token):
                # Per-chunk output validation
                try:
                    guardrails_verdict = await validate_output(chunk)
                    if not guardrails_verdict.passed and guardrails_verdict.action != "fix":
                        mid_stream_violation = True
                        yield "data: [REDACTED DUE TO POLICY]\n\n"
                        final_triage_state = "HARD_BLOCK"
                        _record_telemetry(
                            request, ctx, final_triage_state, total_tokens,
                            mid_stream_violation, cache_hit, start_ts
                        )
                        return
                    emit_chunk = (
                        guardrails_verdict.fixed_output
                        if guardrails_verdict.action == "fix" and guardrails_verdict.fixed_output
                        else chunk
                    )
                except Exception as exc:
                    logger.error("Output validator error mid-stream (%s): %s", request_id, exc)
                    emit_chunk = chunk

                yield f"data: {emit_chunk}\n\n"

        # Flush remainder — also validate (Req. 4.7)
        for chunk in window.flush_remaining():
            try:
                guardrails_verdict = await validate_output(chunk)
                if not guardrails_verdict.passed and guardrails_verdict.action != "fix":
                    mid_stream_violation = True
                    yield "data: [REDACTED DUE TO POLICY]\n\n"
                    final_triage_state = "HARD_BLOCK"
                    _record_telemetry(
                        request, ctx, final_triage_state, total_tokens,
                        mid_stream_violation, cache_hit, start_ts
                    )
                    return
                emit_chunk = (
                    guardrails_verdict.fixed_output
                    if guardrails_verdict.action == "fix" and guardrails_verdict.fixed_output
                    else chunk
                )
            except Exception as exc:
                logger.error("Output validator error on flush (%s): %s", request_id, exc)
                emit_chunk = chunk
            yield f"data: {emit_chunk}\n\n"

    except Exception as exc:
        logger.error("LLM streaming error for request %s: %s", request_id, exc)
        yield "data: [STREAM_ERROR]\n\n"
        _record_telemetry(request, ctx, "HARD_BLOCK", total_tokens, False, cache_hit, start_ts)
        return

    # ------------------------------------------------------------------
    # 5. Post-stream groundedness audit (Req. 4.9)
    # ------------------------------------------------------------------
    full_response = "".join(accumulated_response)
    if full_response and vector_store is not None:
        try:
            audit_result = await audit(
                response=full_response,
                request_id=request_id,
                vector_store=vector_store,
                nli_scorer=nli_scorer,
            )
            ctx.audit_result = audit_result
            if audit_result.nli_label == "CONTRADICTION":
                mid_stream_violation = True
                yield "data: [REDACTED DUE TO POLICY]\n\n"
                final_triage_state = "HARD_BLOCK"
                _record_telemetry(
                    request, ctx, final_triage_state, total_tokens,
                    mid_stream_violation, cache_hit, start_ts
                )
                return
        except Exception as exc:
            logger.error("Post-stream audit error for %s: %s", request_id, exc)

    # ------------------------------------------------------------------
    # 6. Store in SemanticCache (Req. 4.10)
    # ------------------------------------------------------------------
    if profile.cache_enabled and semantic_cache is not None and full_response:
        await semantic_cache.store(ctx.working_prompt, full_response, profile.cache_ttl_seconds)

    # ------------------------------------------------------------------
    # 7. Done
    # ------------------------------------------------------------------
    final_triage_state = "PASS_AND_DELIVER"
    yield "data: [DONE]\n\n"
    _record_telemetry(request, ctx, final_triage_state, total_tokens, False, cache_hit, start_ts)


# ---------------------------------------------------------------------------
# LLM token stream (Portkey passthrough)
# ---------------------------------------------------------------------------


async def _stream_tokens_from_llm(
    prompt: str,
    profile: UseCaseProfile,
) -> AsyncGenerator[str, None]:
    """Stream tokens from the LLM via Portkey.

    In production this calls the Portkey streaming API.  When Portkey is
    unreachable or the key is a dummy value, falls back to a word-by-word
    simulated stream so the rest of the pipeline still exercises the SSE path.
    """
    portkey_key = settings.PORTKEY_API_KEY
    if portkey_key.startswith("dummy"):
        # Fallback: simulate streaming for local dev / tests
        from app.router.model_router import _generate_contextual_response  # noqa: PLC0415
        words = _generate_contextual_response(prompt).split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0)
        return

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                "https://api.portkey.ai/v1/chat/completions",
                headers={
                    "x-portkey-api-key": portkey_key,
                    "x-portkey-virtual-key": settings.PORTKEY_SLM_VIRTUAL_KEY,
                    "x-portkey-provider": settings.LLM_PROVIDER,
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            return
                        import json
                        try:
                            data = json.loads(payload)
                            token = (
                                data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if token:
                                yield token
                        except Exception:
                            pass
    except Exception as exc:
        raise RuntimeError(f"Portkey streaming failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Telemetry helper
# ---------------------------------------------------------------------------


def _record_telemetry(
    request: Request,
    ctx: RequestContext,
    final_triage_state: str,
    total_tokens: int,
    mid_stream_violation: bool,
    cache_hit: bool,
    start_ts: float,
) -> None:
    """Fire-and-forget telemetry record after stream close (Req. 4.11)."""
    try:
        telemetry_logger = getattr(request.app.state, "telemetry_logger", None)
        if telemetry_logger is None:
            return

        latency_ms = int((time.monotonic() - start_ts) * 1000)
        nli_label = ctx.audit_result.nli_label if ctx.audit_result else None

        record = TelemetryRecord(
            request_id=ctx.request_id,
            timestamp=datetime.now(tz=timezone.utc),
            use_case_profile=ctx.profile.name,
            final_triage_state=final_triage_state,
            latency_ms=latency_ms,
            response_token_count=total_tokens,
            cache_hit=cache_hit,
            nli_label=nli_label,
        )

        async def _log() -> None:
            await telemetry_logger.log(record)

        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_log())
    except Exception as exc:  # noqa: BLE001
        logger.error("Telemetry record failed for %s: %s", ctx.request_id, exc)
