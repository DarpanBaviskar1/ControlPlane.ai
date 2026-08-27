"""FastAPI application entry point.

The lifespan context manager handles startup/shutdown:
1. Initialises the PolicyLoader (with watchdog).
2. Loads P1 scanner models (LLM Guard Toxicity + PromptInjection).
3. Loads P3 spaCy model.
4. Creates PIIMaskingEngine and runs startup validation suite.
   - If validation fails: is_healthy=False; ingress returns 503.
5. Starts the TelemetryLogger background consumer (stub until Wave 12).
6. Starts RouteLLM Controller (stub until Wave 8).

All long-lived objects are stored in app.state so FastAPI dependency
injection can access them without globals.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import settings
from app.ingress.router import router as chat_router
from app.judges.p1_judge import load_scanners as load_p1_scanners
from app.judges.p3_judge import load_models as load_p3_models
from app.judges.pii_masking import PIIMaskingEngine
from app.policy.loader import PolicyLoader
from app.telemetry.logger import TelemetryLogger
from app.telemetry.router import router as metrics_router
from app.feedback.router import router as feedback_router
from app.router.model_router import init_router, route_and_call
from app.groundedness.vector_store import FAISSVectorStore
from app.groundedness.auditor import audit
from app.triage.gateway import evaluate
from app.triage.compressor import compress_and_edit
from app.judges.orchestrator import run_orchestrator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown sequence."""

    # 1. Policy Layer
    policy_loader = PolicyLoader(policy_file_path=settings.POLICY_FILE_PATH)
    await policy_loader.start()
    app.state.policy_loader = policy_loader

    # 2. P1 scanner models (CPU-bound; load in thread pool to avoid blocking startup)
    import asyncio
    await asyncio.to_thread(load_p1_scanners)

    # 3. P3 spaCy model
    await asyncio.to_thread(load_p3_models)

    # 4. PII Masking Engine + startup validation
    pii_engine = PIIMaskingEngine()
    validation_passed = await pii_engine.run_startup_validation()
    if not validation_passed:
        logger.error(
            "MASKING_INTEGRITY_FAILURE: PII startup validation failed — "
            "gateway will return 503 until re-validation passes"
        )
    app.state.pii_engine = pii_engine

    # 5. TelemetryLogger
    telemetry_logger = TelemetryLogger()
    await telemetry_logger.start()
    app.state.telemetry_logger = telemetry_logger

    # 6. RouteLLM Controller
    init_router()
    app.state.routellm_controller = True # flag to indicate initialization

    # Initialize Vector Store
    app.state.vector_store = FAISSVectorStore()

    # 7. Pipeline function
    async def run_pipeline(ctx) -> None:
        """Full five-stage deterministic pipeline."""
        # --- 1. Orchestrator (Micro-Judges) ---
        await run_orchestrator(ctx, pii_engine)
        if ctx.upstream_triage_state == "HARD_BLOCK":
            ctx.triage_result = evaluate(
                groundedness_score=1.0, 
                response_token_count=0,
                upstream_triage_state=ctx.upstream_triage_state,
                p3_clarity=ctx.p3_verdict or "AMBIGUOUS",
                profile=ctx.profile,
            )
            return

        # --- 2. Model Router ---
        decision = await route_and_call(
            prompt=ctx.working_prompt,
            profile=ctx.profile,
            p3_clarity=ctx.p3_verdict
        )
        ctx.routing_decision = decision
        if decision.triage_state == "HARD_BLOCK":
            ctx.upstream_triage_state = "HARD_BLOCK"
            ctx.triage_result = evaluate(
                groundedness_score=1.0,
                response_token_count=0,
                upstream_triage_state=ctx.upstream_triage_state,
                p3_clarity=ctx.p3_verdict or "AMBIGUOUS",
                profile=ctx.profile,
            )
            return
            
        ctx.llm_response = decision.response
        response_token_count = len(ctx.llm_response.split()) if ctx.llm_response else 0

        # --- 3. Groundedness Auditor ---
        audit_res = await audit(
            response=ctx.llm_response or "",
            request_id=ctx.request_id,
            vector_store=app.state.vector_store
        )
        ctx.audit_result = audit_res

        # --- 4. Triage Gateway ---
        triage_res = evaluate(
            groundedness_score=audit_res.groundedness_score,
            response_token_count=response_token_count,
            upstream_triage_state=ctx.upstream_triage_state,
            p3_clarity=ctx.p3_verdict or "AMBIGUOUS",
            profile=ctx.profile,
            response_content=ctx.llm_response
        )

        if triage_res.triage_state == "COMPRESS_AND_EDIT" and triage_res.response_content:
            compressed = await compress_and_edit(
                triage_res.response_content,
                ctx.profile.token_compression_threshold
            )
            triage_res.response_content = compressed
            
        if triage_res.response_content and pii_engine and ctx.placeholder_map:
            triage_res.response_content = pii_engine.unmask(
                triage_res.response_content,
                ctx.request_id
            )

        ctx.triage_result = triage_res

    app.state.pipeline_fn = run_pipeline

    logger.info("ControlPlane.ai Gateway started (pii_healthy=%s)", pii_engine.is_healthy)

    yield

    # Shutdown
    await policy_loader.stop()
    await telemetry_logger.stop()
    logger.info("ControlPlane.ai Gateway stopped")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ControlPlane.ai Enterprise AI Proxy Gateway",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a consistent 422 envelope for Pydantic validation errors."""
    errors = exc.errors()
    first = errors[0] if errors else {}
    field = ".".join(str(loc) for loc in first.get("loc", []))
    msg = first.get("msg", "Validation error")
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "VALIDATION_ERROR",
            "detail": f"{field}: {msg}" if field else msg,
            "request_id": None,
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(chat_router)
app.include_router(metrics_router)
app.include_router(feedback_router)
