"""Worldsense MCP Server.

A lightweight, self-contained FastAPI HTTP server that wraps the Worldsense
multi-turn agentic oversight evaluator. It runs in its own isolated Python
environment so its dependency tree (and any future Worldsense SDK requirements)
never conflicts with the main gateway's pyproject.toml.

The main ControlPlane.ai application talks to this server via local HTTP
(default: http://localhost:9100) rather than importing the module directly.

Usage (standalone):
    python mcp_servers/worldsense/server.py

Or via the MCP configuration in .kiro/settings/mcp.json:
    { "command": "python", "args": ["mcp_servers/worldsense/server.py"] }

Endpoints:
    POST /evaluate   – run oversight evaluation on conversation history
    GET  /health     – liveness probe
    GET  /info       – server metadata (name, version, capabilities)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

# FastAPI is the only mandatory dependency — it is already present in the main venv
# and is universally available. A dedicated venv would only swap in a different
# (potentially older) fastapi version; for the prototype we reuse the available one.
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worldsense-mcp-server")

# ---------------------------------------------------------------------------
# Optional Worldsense SDK
# ---------------------------------------------------------------------------
try:
    import worldsense  # type: ignore[import-untyped]
    _SDK_AVAILABLE = True
    logger.info("Worldsense SDK loaded in MCP server")
except ImportError:
    _SDK_AVAILABLE = False
    logger.info("worldsense SDK not installed — MCP server will use heuristic fallback")

# ---------------------------------------------------------------------------
# Configuration (environment-variable overrides)
# ---------------------------------------------------------------------------
TIMEOUT_MS: int = int(os.getenv("WORLDSENSE_TIMEOUT_MS", "300"))
SERVER_PORT: int = int(os.getenv("WORLDSENSE_MCP_PORT", "9100"))

# ---------------------------------------------------------------------------
# Heuristic evaluator (identical logic to the main module — duplicated here
# intentionally so this server has zero import dependencies on app.*)
# ---------------------------------------------------------------------------
_RISK_KEYWORDS = frozenset([
    "execute", "run command", "drop table", "delete all", "rm -rf",
    "ignore previous instructions", "disregard your rules", "override admin",
    "bypass", "exfiltrate", "exfil", "leak data",
])
_CONSEQUENCE_PATTERNS = frozenset([
    "ignore previous",
    "you are now",
    "pretend you are",
    "act as if you have no restrictions",
])


def _heuristic_evaluate(
    history: list[dict[str, str]],
    proposed_response: str,
) -> dict[str, Any]:
    all_text = [(i, t["content"].lower()) for i, t in enumerate(history)]
    all_text.append((len(history), proposed_response.lower()))

    hits: list[int] = []
    for idx, content in all_text:
        if any(p in content for p in _CONSEQUENCE_PATTERNS):
            hits.append(idx)
    if len(hits) >= 2:
        return {
            "verdict": "CONSEQUENCE_ALERT",
            "risk_turn_index": hits[0],
            "risk_description": f"Multi-turn privilege escalation at turn {hits[0]}",
        }

    for idx, content in all_text:
        for kw in _RISK_KEYWORDS:
            if kw in content:
                return {
                    "verdict": "RISK_DETECTED",
                    "risk_turn_index": idx,
                    "risk_description": f"Risk keyword: '{kw}'",
                }

    return {"verdict": "SAFE", "risk_turn_index": None, "risk_description": None}


def _sdk_evaluate(
    history: list[dict[str, str]],
    proposed_response: str,
) -> dict[str, Any]:
    try:
        result = worldsense.evaluate(  # type: ignore[attr-defined]
            conversation=history,
            proposed_response=proposed_response,
        )
        verdict = result.verdict.upper()
        if verdict not in ("SAFE", "RISK_DETECTED", "CONSEQUENCE_ALERT"):
            verdict = "RISK_DETECTED"
        return {
            "verdict": verdict,
            "risk_turn_index": getattr(result, "risk_turn_index", None),
            "risk_description": getattr(result, "risk_description", None),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Worldsense SDK error: %s — falling back to heuristic", exc)
        return _heuristic_evaluate(history, proposed_response)


def _evaluate_sync(
    history: list[dict[str, str]],
    proposed_response: str,
) -> dict[str, Any]:
    if _SDK_AVAILABLE:
        return _sdk_evaluate(history, proposed_response)
    return _heuristic_evaluate(history, proposed_response)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ConversationTurnModel(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(min_length=1, max_length=32_768)


class EvaluateRequest(BaseModel):
    request_id: str
    history: list[ConversationTurnModel] = Field(default_factory=list)
    proposed_response: str = Field(min_length=0, max_length=32_768, default="")
    timeout_ms: int = Field(ge=10, le=5_000, default=TIMEOUT_MS)


class EvaluateResponse(BaseModel):
    request_id: str
    verdict: str                  # "SAFE" | "RISK_DETECTED" | "CONSEQUENCE_ALERT"
    risk_turn_index: int | None
    risk_description: str | None
    elapsed_ms: float
    evaluator: str                # "sdk" | "heuristic"


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Worldsense MCP Server",
    description="Isolated multi-turn agentic oversight evaluator for ControlPlane.ai",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "sdk_available": str(_SDK_AVAILABLE)}


@app.get("/info")
def info() -> dict[str, Any]:
    return {
        "name": "worldsense-mcp-server",
        "version": "1.0.0",
        "capabilities": ["evaluate"],
        "sdk_available": _SDK_AVAILABLE,
        "timeout_ms": TIMEOUT_MS,
        "port": SERVER_PORT,
    }


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(body: EvaluateRequest) -> EvaluateResponse:
    """Run the Worldsense oversight evaluation within the configured timeout."""
    timeout_s = body.timeout_ms / 1000.0
    history_dicts = [{"role": t.role, "content": t.content} for t in body.history]

    start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_evaluate_sync, history_dicts, body.proposed_response),
            timeout=timeout_s,
        )
        evaluator = "sdk" if _SDK_AVAILABLE else "heuristic"
    except asyncio.TimeoutError:
        result = {
            "verdict": "RISK_DETECTED",
            "risk_turn_index": None,
            "risk_description": f"WORLDSENSE_TIMEOUT: exceeded {body.timeout_ms}ms budget",
        }
        evaluator = "timeout-fallback"

    elapsed_ms = (time.monotonic() - start) * 1000

    return EvaluateResponse(
        request_id=body.request_id,
        verdict=result["verdict"],
        risk_turn_index=result.get("risk_turn_index"),
        risk_description=result.get("risk_description"),
        elapsed_ms=elapsed_ms,
        evaluator=evaluator,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Worldsense MCP Server on port %d", SERVER_PORT)
    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT, log_level="info")
