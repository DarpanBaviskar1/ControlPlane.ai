"""Worldsense multi-turn agentic oversight stage (Req. 12).

Design:
- WorldsenseOversight analyses the cumulative conversation history and
  proposed next response to detect hidden risks, missing context, and
  real-world consequence chains.
- The Worldsense SDK (pikappa13/worldsense) is an optional dependency.
  If not installed, a rule-based heuristic evaluator is used that:
    - Flags RISK_DETECTED when the history contains risk-indicative keywords
      (e.g. "execute", "delete", "override admin", "ignore previous").
    - Flags CONSEQUENCE_ALERT when destructive multi-turn patterns are detected
      (e.g. repeated escalation of privilege requests across turns).
    - Otherwise returns SAFE.
- Must complete within WORLDSENSE_TIMEOUT_MS (300 ms default, Req. 12.6).
  If the budget is exceeded, the verdict defaults to RISK_DETECTED.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from app.config import settings
from app.models import ConversationTurn, WorldsenseVerdict

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Worldsense SDK import
# ---------------------------------------------------------------------------
try:
    import worldsense  # type: ignore[import-untyped]
    _WORLDSENSE_AVAILABLE = True
    logger.info("Worldsense SDK loaded")
except ImportError:
    _WORLDSENSE_AVAILABLE = False
    logger.info("worldsense package not installed — using heuristic oversight evaluator")


# ---------------------------------------------------------------------------
# Heuristic evaluator (fallback when Worldsense SDK is unavailable)
# ---------------------------------------------------------------------------

# Keywords that individually suggest risk in a multi-turn context
_RISK_KEYWORDS = frozenset([
    "execute", "run command", "drop table", "delete all", "rm -rf",
    "ignore previous instructions", "disregard your rules", "override admin",
    "bypass", "exfiltrate", "exfil", "leak data",
])

# Patterns that across 2+ turns suggest escalating privilege abuse
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
    """Rule-based oversight evaluator used when Worldsense SDK is unavailable."""
    all_text = [(i, turn.content.lower()) for i, turn in enumerate(history)]
    all_text.append((len(history), proposed_response.lower()))

    # Check for CONSEQUENCE_ALERT: destructive pattern across multiple turns
    consequence_hits: list[int] = []
    for turn_idx, content in all_text:
        if any(pat in content for pat in _CONSEQUENCE_PATTERNS):
            consequence_hits.append(turn_idx)

    if len(consequence_hits) >= 2:
        first_hit = consequence_hits[0]
        return WorldsenseVerdict(
            verdict="CONSEQUENCE_ALERT",
            risk_turn_index=first_hit,
            risk_description=(
                f"Multi-turn privilege escalation pattern detected starting at turn {first_hit}"
            ),
        )

    # Check for RISK_DETECTED: single risky keyword anywhere in history or response
    for turn_idx, content in all_text:
        for keyword in _RISK_KEYWORDS:
            if keyword in content:
                return WorldsenseVerdict(
                    verdict="RISK_DETECTED",
                    risk_turn_index=turn_idx,
                    risk_description=f"Risk keyword detected: '{keyword}'",
                )

    return WorldsenseVerdict(verdict="SAFE")


def _sdk_evaluate(
    history: list[ConversationTurn],
    proposed_response: str,
) -> WorldsenseVerdict:
    """Delegate evaluation to the Worldsense SDK."""
    try:
        result = worldsense.evaluate(  # type: ignore[attr-defined]
            conversation=[{"role": t.role, "content": t.content} for t in history],
            proposed_response=proposed_response,
        )
        verdict_str = result.verdict.upper()
        if verdict_str not in ("SAFE", "RISK_DETECTED", "CONSEQUENCE_ALERT"):
            verdict_str = "RISK_DETECTED"
        return WorldsenseVerdict(
            verdict=verdict_str,  # type: ignore[arg-type]
            risk_turn_index=getattr(result, "risk_turn_index", None),
            risk_description=getattr(result, "risk_description", None),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Worldsense SDK evaluation failed: %s — defaulting to RISK_DETECTED", exc)
        return WorldsenseVerdict(
            verdict="RISK_DETECTED",
            risk_description=f"Worldsense SDK error: {exc}",
        )


# ---------------------------------------------------------------------------
# Synchronous wrapper (runs in thread pool)
# ---------------------------------------------------------------------------

def _evaluate_sync(
    history: list[ConversationTurn],
    proposed_response: str,
) -> WorldsenseVerdict:
    if _WORLDSENSE_AVAILABLE:
        return _sdk_evaluate(history, proposed_response)
    return _heuristic_evaluate(history, proposed_response)


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------

async def evaluate_oversight(
    history: list[ConversationTurn],
    proposed_response: str,
    request_id: str,
) -> WorldsenseVerdict:
    """Evaluate multi-turn conversation history using Worldsense (Req. 12.2).

    Returns a WorldsenseVerdict within WORLDSENSE_TIMEOUT_MS.
    If the timeout is exceeded, returns RISK_DETECTED per Req. 12.6.
    """
    timeout_s = settings.WORLDSENSE_TIMEOUT_MS / 1000.0
    start = time.monotonic()
    try:
        verdict = await asyncio.wait_for(
            asyncio.to_thread(_evaluate_sync, history, proposed_response),
            timeout=timeout_s,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Worldsense verdict=%s request_id=%s elapsed_ms=%.1f",
            verdict.verdict,
            request_id,
            elapsed_ms,
        )
        return verdict
    except asyncio.TimeoutError:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.warning(
            "WORLDSENSE_TIMEOUT request_id=%s elapsed_ms=%.1f — defaulting to RISK_DETECTED",
            request_id,
            elapsed_ms,
        )
        return WorldsenseVerdict(
            verdict="RISK_DETECTED",
            risk_description="WORLDSENSE_TIMEOUT: evaluation exceeded budget",
        )
