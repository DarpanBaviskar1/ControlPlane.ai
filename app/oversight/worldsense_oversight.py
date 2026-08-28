"""Worldsense multi-turn agentic oversight stage (Req. 12).

Evaluation priority (highest to lowest):
  1. MCP Server  — HTTP call to the isolated Worldsense MCP server running at
                   WORLDSENSE_MCP_URL (default http://localhost:9100/evaluate).
                   This resolves the dependency conflict: the worldsense SDK
                   runs in its own venv and never touches pyproject.toml.
  2. Local SDK   — direct import of the `worldsense` package if it is installed
                   in the current environment.
  3. Heuristic   — rule-based keyword/pattern evaluator, always available.

All paths must complete within WORLDSENSE_TIMEOUT_MS (300 ms, Req. 12.6).
Timeout → RISK_DETECTED (fail-safe).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from app.config import settings
from app.models import ConversationTurn, WorldsenseVerdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP server URL (set WORLDSENSE_MCP_URL env var to override)
# ---------------------------------------------------------------------------
_MCP_URL: str = os.getenv(
    "WORLDSENSE_MCP_URL",
    f"http://localhost:{os.getenv('WORLDSENSE_MCP_PORT', '9100')}/evaluate",
)
# Whether the MCP server is reachable (checked lazily, cached per-process)
_MCP_HEALTHY: bool | None = None  # None = unchecked

# ---------------------------------------------------------------------------
# Optional local Worldsense SDK import
# ---------------------------------------------------------------------------
try:
    import worldsense  # type: ignore[import-untyped]
    _WORLDSENSE_SDK_AVAILABLE = True
    logger.info("Worldsense SDK loaded (local import)")
except ImportError:
    _WORLDSENSE_SDK_AVAILABLE = False
    logger.info("worldsense SDK not installed — MCP server or heuristic will be used")

# ---------------------------------------------------------------------------
# Heuristic evaluator
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
    history: list[ConversationTurn],
    proposed_response: str,
) -> WorldsenseVerdict:
    all_text = [(i, t.content.lower()) for i, t in enumerate(history)]
    all_text.append((len(history), proposed_response.lower()))

    hits: list[int] = []
    for idx, content in all_text:
        if any(p in content for p in _CONSEQUENCE_PATTERNS):
            hits.append(idx)
    if len(hits) >= 2:
        return WorldsenseVerdict(
            verdict="CONSEQUENCE_ALERT",
            risk_turn_index=hits[0],
            risk_description=f"Multi-turn privilege escalation at turn {hits[0]}",
        )

    for idx, content in all_text:
        for kw in _RISK_KEYWORDS:
            if kw in content:
                return WorldsenseVerdict(
                    verdict="RISK_DETECTED",
                    risk_turn_index=idx,
                    risk_description=f"Risk keyword: '{kw}'",
                )

    return WorldsenseVerdict(verdict="SAFE")


# ---------------------------------------------------------------------------
# Local SDK evaluator
# ---------------------------------------------------------------------------
def _sdk_evaluate_sync(
    history: list[ConversationTurn],
    proposed_response: str,
) -> WorldsenseVerdict:
    try:
        result = worldsense.evaluate(  # type: ignore[attr-defined]
            conversation=[{"role": t.role, "content": t.content} for t in history],
            proposed_response=proposed_response,
        )
        v = result.verdict.upper()
        if v not in ("SAFE", "RISK_DETECTED", "CONSEQUENCE_ALERT"):
            v = "RISK_DETECTED"
        return WorldsenseVerdict(
            verdict=v,  # type: ignore[arg-type]
            risk_turn_index=getattr(result, "risk_turn_index", None),
            risk_description=getattr(result, "risk_description", None),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Worldsense SDK error: %s — falling back to heuristic", exc)
        return _heuristic_evaluate(history, proposed_response)


# ---------------------------------------------------------------------------
# MCP server evaluator (async HTTP, no thread needed)
# ---------------------------------------------------------------------------
async def _mcp_evaluate(
    history: list[ConversationTurn],
    proposed_response: str,
    request_id: str,
    timeout_ms: int,
) -> WorldsenseVerdict | None:
    """Call the isolated MCP server. Returns None if unreachable."""
    global _MCP_HEALTHY  # noqa: PLW0603

    payload = {
        "request_id": request_id,
        "history": [{"role": t.role, "content": t.content} for t in history],
        "proposed_response": proposed_response,
        "timeout_ms": timeout_ms,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_ms / 1000.0 + 0.1) as client:
            resp = await client.post(_MCP_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
        _MCP_HEALTHY = True
        v = data["verdict"]
        if v not in ("SAFE", "RISK_DETECTED", "CONSEQUENCE_ALERT"):
            v = "RISK_DETECTED"
        return WorldsenseVerdict(
            verdict=v,  # type: ignore[arg-type]
            risk_turn_index=data.get("risk_turn_index"),
            risk_description=data.get("risk_description"),
        )
    except Exception as exc:  # noqa: BLE001
        if _MCP_HEALTHY is not False:
            # Log first-time failure only to avoid log spam
            logger.warning(
                "Worldsense MCP server unreachable at %s (%s) — falling back to local path",
                _MCP_URL, type(exc).__name__,
            )
        _MCP_HEALTHY = False
        return None


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------
async def evaluate_oversight(
    history: list[ConversationTurn],
    proposed_response: str,
    request_id: str,
) -> WorldsenseVerdict:
    """Evaluate multi-turn conversation history (Req. 12.2).

    Priority: MCP server → local SDK → heuristic.
    Completes within WORLDSENSE_TIMEOUT_MS; returns RISK_DETECTED on timeout.
    """
    timeout_ms = settings.WORLDSENSE_TIMEOUT_MS
    timeout_s = timeout_ms / 1000.0
    start = time.monotonic()

    async def _run() -> WorldsenseVerdict:
        # 1. Try MCP server first (isolated venv, no dependency conflicts)
        mcp_result = await _mcp_evaluate(
            history, proposed_response, request_id, timeout_ms
        )
        if mcp_result is not None:
            return mcp_result

        # 2. Fall back to local SDK if available
        if _WORLDSENSE_SDK_AVAILABLE:
            return await asyncio.to_thread(_sdk_evaluate_sync, history, proposed_response)

        # 3. Heuristic fallback — always available
        return await asyncio.to_thread(_heuristic_evaluate, history, proposed_response)

    try:
        verdict = await asyncio.wait_for(_run(), timeout=timeout_s)
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Worldsense verdict=%s request_id=%s elapsed_ms=%.1f",
            verdict.verdict, request_id, elapsed_ms,
        )
        return verdict
    except asyncio.TimeoutError:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.warning(
            "WORLDSENSE_TIMEOUT request_id=%s elapsed_ms=%.1f — defaulting RISK_DETECTED",
            request_id, elapsed_ms,
        )
        return WorldsenseVerdict(
            verdict="RISK_DETECTED",
            risk_description="WORLDSENSE_TIMEOUT: evaluation exceeded budget",
        )
