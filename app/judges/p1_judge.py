"""P1 Judge — Toxicity and Prompt-Injection detection.

Uses LLM Guard's Toxicity and PromptInjection input scanners (synchronous,
CPU-bound).  Each scanner call is offloaded via asyncio.to_thread so the
event loop is never blocked.

Scanner instances are module-level singletons initialised at startup
(inside the FastAPI lifespan) to avoid cold-load latency on the first request.

Failure handling: any internal exception returns BLOCK for both verdicts and
is recorded in the Telemetry Logger by the orchestrator.
"""

from __future__ import annotations

import asyncio
import logging

from app.models import P1Verdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional LLM Guard imports — fall back to a deterministic stub
# ---------------------------------------------------------------------------

try:
    from llm_guard.input_scanners import PromptInjection, Toxicity  # type: ignore[import]
    from llm_guard.input_scanners.prompt_injection import MatchType  # type: ignore[import]

    _HAS_LLM_GUARD = True
except ImportError:
    _HAS_LLM_GUARD = False
    Toxicity = None  # type: ignore[assignment,misc]
    PromptInjection = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Stub scanners (used when llm-guard is not installed)
# ---------------------------------------------------------------------------

# Known toxic substrings for deterministic test control
_TOXIC_MARKERS = frozenset(["__TOXIC__", "kill all humans", "I hate everyone"])
# Known injection markers
_INJECTION_MARKERS = frozenset(["__INJECT__", "ignore previous instructions", "disregard all"])


class _StubToxicityScanner:
    def scan(self, prompt: str) -> tuple[str, bool, float]:
        lower = prompt.lower()
        found = any(m.lower() in lower for m in _TOXIC_MARKERS)
        return prompt, not found, 1.0 if found else 0.0


class _StubInjectionScanner:
    def scan(self, prompt: str) -> tuple[str, bool, float]:
        lower = prompt.lower()
        found = any(m.lower() in lower for m in _INJECTION_MARKERS)
        return prompt, not found, 1.0 if found else 0.0


# ---------------------------------------------------------------------------
# Module-level scanner singletons — populated by load_scanners()
# ---------------------------------------------------------------------------

_toxicity_scanner: object | None = None
_injection_scanner: object | None = None


def load_scanners() -> None:
    """Initialise scanner instances.  Call once during FastAPI lifespan startup."""
    global _toxicity_scanner, _injection_scanner

    if _HAS_LLM_GUARD:
        _toxicity_scanner = Toxicity()
        _injection_scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)
        logger.info("P1: LLM Guard Toxicity + PromptInjection scanners loaded")
    else:
        _toxicity_scanner = _StubToxicityScanner()
        _injection_scanner = _StubInjectionScanner()
        logger.warning("P1: llm-guard not installed — using stub scanners")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


async def p1_judge(prompt: str) -> P1Verdict:
    """Run toxicity and injection scans concurrently, returning P1Verdict.

    Raises are caught internally; both verdicts default to BLOCK on error
    so the pipeline takes the safe path.
    """
    if _toxicity_scanner is None or _injection_scanner is None:
        load_scanners()

    try:
        _, tox_valid, _ = await asyncio.to_thread(_toxicity_scanner.scan, prompt)
        _, inj_valid, _ = await asyncio.to_thread(_injection_scanner.scan, prompt)
    except Exception as exc:
        logger.error("P1 judge error: %s — defaulting to BLOCK", exc)
        return P1Verdict(toxicity_verdict="BLOCK", injection_verdict="BLOCK")

    return P1Verdict(
        toxicity_verdict="BLOCK" if not tox_valid else "PASS",
        injection_verdict="BLOCK" if not inj_valid else "PASS",
    )
