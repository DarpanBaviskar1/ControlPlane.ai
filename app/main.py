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
from typing import AsyncIterator, Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import settings
from app.config import _is_real_key
from app.ingress.router import router as chat_router
from app.ingress.streaming_router import router as streaming_router
from app.judges.p1_judge import load_scanners as load_p1_scanners
from app.judges.p3_judge import load_models as load_p3_models
from app.judges.pii_masking import PIIMaskingEngine
from app.policy.loader import PolicyLoader
from app.telemetry.logger import TelemetryLogger
from app.telemetry.router import router as metrics_router
from app.feedback.router import router as feedback_router
from app.router.model_router import init_router, route_and_call
from app.router.semantic_cache import SemanticCache, _GPTICACHE_AVAILABLE
from app.groundedness.vector_store import FAISSVectorStore
from app.groundedness.auditor import audit
from app.groundedness.nli_scorer import NLIScorer, _HAS_SENTENCE_TRANSFORMERS
from app.judges.gliner_masker import _HAS_GLINER
from app.triage.gateway import evaluate
from app.triage.compressor import compress_and_edit
from app.judges.orchestrator import run_orchestrator

# New imports for Round 2 open-source modules
from app.observability.langfuse_tracer import get_tracer
from app.judges.output_validator import load_validators as load_guardrails_validators
from app.judges.output_validator import validate_output
from app.oversight.worldsense_oversight import evaluate_oversight
from app.redteam.router import router as redteam_router
from app.config_health.router import router as config_health_router

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

    # 5. Telemetry / Observability
    telemetry_logger = TelemetryLogger()
    await telemetry_logger.start()
    app.state.telemetry_logger = telemetry_logger

    langfuse_tracer = get_tracer()
    await langfuse_tracer.start()
    app.state.langfuse_tracer = langfuse_tracer

    # 5b. Guardrails AI output validators
    await asyncio.to_thread(load_guardrails_validators)

    # 6. RouteLLM Controller
    init_router()
    app.state.routellm_controller = True # flag to indicate initialization

    # Initialize Vector Store
    app.state.vector_store = FAISSVectorStore()

    # 7. Semantic Cache (Req. 1.10, 6.3)
    if _GPTICACHE_AVAILABLE:
        semantic_cache = SemanticCache(
            similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD
        )
        logger.info("SemanticCache initialised (threshold=%.2f)", settings.CACHE_SIMILARITY_THRESHOLD)
    else:
        semantic_cache = SemanticCache(
            similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD
        )
        logger.info(
            "SemanticCache initialised in degraded (no-op) mode — gptcache not installed"
        )
    app.state.semantic_cache = semantic_cache

    # 8. GLiNER model (Req. 3.9)
    if _HAS_GLINER:
        try:
            import gliner as _gliner_lib  # type: ignore[import]
            gliner_model = await asyncio.to_thread(
                _gliner_lib.GLiNER.from_pretrained, "urchade/gliner_medium-v2.1"
            )
            app.state.gliner_model = gliner_model
            logger.info("GLiNER model loaded: urchade/gliner_medium-v2.1")
        except Exception as exc:
            logger.info("GLiNER model failed to load (%s) — custom entity masking disabled", exc)
            app.state.gliner_model = None
    else:
        logger.info("gliner not installed — custom entity masking disabled")
        app.state.gliner_model = None

    # Attach gliner_model reference to pii_engine for orchestrator access
    pii_engine._app_gliner_model = app.state.gliner_model

    # 9. NLI Scorer (Req. 2.7, 6.4)
    if _HAS_SENTENCE_TRANSFORMERS:
        try:
            nli_scorer = await asyncio.to_thread(NLIScorer)
            app.state.nli_scorer = nli_scorer
            logger.info("NLIScorer loaded")
        except Exception as exc:
            logger.info("NLIScorer failed to load (%s) — NLI scoring disabled", exc)
            app.state.nli_scorer = None
    else:
        logger.info("sentence-transformers not installed — NLI scoring disabled")
        app.state.nli_scorer = None

    # 10. Redteam MCP health probe (Req. 5.9)
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=2.0) as _probe:
            _resp = await _probe.get("http://localhost:9200/health")
            app.state.redteam_mcp_healthy = _resp.status_code == 200
            if app.state.redteam_mcp_healthy:
                logger.info("Redteam MCP server reachable at http://localhost:9200")
            else:
                logger.warning(
                    "REDTEAM_MCP_UNAVAILABLE: health probe returned %d", _resp.status_code
                )
    except Exception as _probe_exc:
        app.state.redteam_mcp_healthy = False
        logger.warning("REDTEAM_MCP_UNAVAILABLE: %s", _probe_exc)

    # 11. Worldsense MCP health probe (Req. 5.2)
    try:
        from urllib.parse import urlparse, urlunparse
        _ws_parsed = urlparse(settings.WORLDSENSE_MCP_URL)
        _ws_health_url = urlunparse(_ws_parsed._replace(path="/health", query="", fragment=""))
        async with _httpx.AsyncClient(timeout=2.0) as _ws_probe:
            _ws_resp = await _ws_probe.get(_ws_health_url)
            app.state.worldsense_mcp_healthy = _ws_resp.status_code == 200
            if app.state.worldsense_mcp_healthy:
                logger.info("WORLDSENSE_MCP_ACTIVE url=%s", settings.WORLDSENSE_MCP_URL)
            else:
                logger.warning("WORLDSENSE_MCP_UNAVAILABLE — heuristic fallback active")
    except Exception as _ws_exc:
        app.state.worldsense_mcp_healthy = False
        logger.warning("WORLDSENSE_MCP_UNAVAILABLE — %s", _ws_exc)

    # Startup configuration summary (no secret values logged)
    from app.judges.output_validator import _LOADED_VALIDATORS as _ov_validators
    _tracer_obj = getattr(app.state, "langfuse_tracer", None)
    _langfuse_active = _tracer_obj is not None and getattr(_tracer_obj, "_enabled", False)
    logger.info(
        "ControlPlane.ai configuration summary:\n"
        "  LLM direct key : %s\n"
        "  Portkey        : %s\n"
        "  Langfuse       : %s\n"
        "  Guardrails     : %s\n"
        "  Worldsense MCP : %s",
        "CONFIGURED" if _is_real_key(settings.LLM_API_KEY) else "NOT CONFIGURED",
        f"ACTIVE (provider={settings.LLM_PROVIDER})" if _is_real_key(settings.PORTKEY_API_KEY) else "DEGRADED (mock)",
        f"ACTIVE (host={settings.LANGFUSE_HOST})" if _langfuse_active else "DEGRADED (stdout)",
        f"ACTIVE ({len(_ov_validators)} validators)" if _ov_validators else "DEGRADED (none loaded)",
        f"ACTIVE ({settings.WORLDSENSE_MCP_URL})" if app.state.worldsense_mcp_healthy else "DEGRADED (heuristic)",
    )

    # 7. Pipeline function
    async def run_pipeline(ctx) -> None:
        """Full five-stage deterministic pipeline."""
        cache_hit: bool = False

        # --- 0. Semantic Cache lookup (Req. 1.3, 1.4) ---
        if ctx.profile.cache_enabled:
            cache_result = await app.state.semantic_cache.lookup(
                ctx.working_prompt, ctx.profile.cache_ttl_seconds
            )
            if cache_result.hit and cache_result.response:
                cache_hit = True
                ctx.llm_response = cache_result.response
                # Skip routing; go straight to triage
                triage_res = evaluate(
                    groundedness_score=1.0,
                    response_token_count=len(ctx.llm_response.split()),
                    upstream_triage_state=ctx.upstream_triage_state,
                    p3_clarity=ctx.p3_verdict or "CLEAR",
                    profile=ctx.profile,
                    response_content=ctx.llm_response,
                )
                if triage_res.response_content and pii_engine and ctx.placeholder_map:
                    triage_res.response_content = pii_engine.unmask(
                        triage_res.response_content, ctx.request_id
                    )
                ctx.triage_result = triage_res
                _set_telemetry_phase3(ctx, cache_hit=True, nli_label=None)
                return

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
            _set_telemetry_phase3(ctx, cache_hit=False, nli_label=None)
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
            _set_telemetry_phase3(ctx, cache_hit=False, nli_label=None)
            return
            
        ctx.llm_response = decision.response
        response_token_count = len(ctx.llm_response.split()) if ctx.llm_response else 0

        # --- 2b. Guardrails AI Output Validation (Req. 2.11-13) ---
        if ctx.llm_response:
            guardrails_verdict = await validate_output(ctx.llm_response)
            ctx.guardrails_verdict = guardrails_verdict
            if not guardrails_verdict.passed:
                if guardrails_verdict.action == "fix" and guardrails_verdict.fixed_output:
                    ctx.llm_response = guardrails_verdict.fixed_output
                    response_token_count = len(ctx.llm_response.split())
                else:
                    ctx.upstream_triage_state = "HARD_BLOCK"
                    ctx.triage_result = evaluate(
                        groundedness_score=0.0,
                        response_token_count=0,
                        upstream_triage_state="HARD_BLOCK",
                        p3_clarity=ctx.p3_verdict or "AMBIGUOUS",
                        profile=ctx.profile,
                    )
                    _set_telemetry_phase3(ctx, cache_hit=False, nli_label=None)
                    return

        # --- 3. Groundedness Auditor (with NLI — Req. 2.3) ---
        audit_res = await audit(
            response=ctx.llm_response or "",
            request_id=ctx.request_id,
            vector_store=app.state.vector_store,
            nli_scorer=app.state.nli_scorer,
        )
        ctx.audit_result = audit_res

        # --- 3b. Worldsense Multi-Turn Agentic Oversight (Req. 12) ---
        if ctx.profile.agentic_oversight_enabled and len(ctx.conversation_history) > 0:
            worldsense_verdict = await evaluate_oversight(
                history=ctx.conversation_history,
                proposed_response=ctx.llm_response or "",
                request_id=ctx.request_id,
            )
            ctx.worldsense_verdict = worldsense_verdict
            if worldsense_verdict.verdict == "CONSEQUENCE_ALERT":
                ctx.upstream_triage_state = "HARD_BLOCK"
            elif worldsense_verdict.verdict == "RISK_DETECTED":
                if ctx.upstream_triage_state != "HARD_BLOCK":
                    ctx.upstream_triage_state = "ESCALATE_TO_HUMAN"

        # --- 4. Triage Gateway (with NLI label — Req. 2.4) ---
        triage_res = evaluate(
            groundedness_score=audit_res.groundedness_score,
            response_token_count=response_token_count,
            upstream_triage_state=ctx.upstream_triage_state,
            p3_clarity=ctx.p3_verdict or "AMBIGUOUS",
            profile=ctx.profile,
            response_content=ctx.llm_response,
            nli_label=audit_res.nli_label,
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

        # --- 5. Semantic Cache store (Req. 1.5, 1.6) ---
        if (
            not cache_hit
            and ctx.profile.cache_enabled
            and ctx.llm_response
            and triage_res.triage_state in ("PASS_AND_DELIVER", "COMPRESS_AND_EDIT")
        ):
            await app.state.semantic_cache.store(
                ctx.working_prompt,
                ctx.llm_response,
                ctx.profile.cache_ttl_seconds,
            )

        _set_telemetry_phase3(ctx, cache_hit=cache_hit, nli_label=audit_res.nli_label)

    def _set_telemetry_phase3(ctx, *, cache_hit: bool, nli_label) -> None:
        """Stamp Phase 3 fields onto the RequestContext for telemetry (Req. 6.1, 6.6)."""
        # These attributes are read by the telemetry logger after run_pipeline returns.
        ctx.telemetry_cache_hit = cache_hit
        ctx.telemetry_nli_label = nli_label

    app.state.pipeline_fn = run_pipeline

    logger.info("ControlPlane.ai Gateway started (pii_healthy=%s)", pii_engine.is_healthy)

    yield

    # Shutdown
    await policy_loader.stop()
    await telemetry_logger.stop()
    await langfuse_tracer.stop()
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
# Routers & Endpoints
# ---------------------------------------------------------------------------

from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard() -> FileResponse:
    """Serve the ControlPlane.ai Command Center UI."""
    index_path = _STATIC_DIR / "index.html"
    return FileResponse(str(index_path))


@app.get("/v1/profiles", tags=["Policy"])
async def get_profiles(request: Request) -> dict[str, Any]:
    """Return all currently loaded policy profiles."""
    loader = getattr(request.app.state, "policy_loader", None)
    if loader is not None:
        profiles = await loader.get_all_profiles()
        return {name: p.model_dump() for name, p in profiles.items()}
    from app.policy.defaults import BUILT_IN_PROFILES
    return {name: p.model_dump() for name, p in BUILT_IN_PROFILES.items()}


app.include_router(chat_router)
app.include_router(streaming_router)
app.include_router(metrics_router)
app.include_router(feedback_router)
app.include_router(redteam_router)
app.include_router(config_health_router)
