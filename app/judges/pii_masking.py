"""PII Masking Engine — two-tier graceful degradation.

Tier 1 (primary):  LLM Guard Anonymize scanner (NLP-based, Presidio under the hood).
                   Highest accuracy; requires llm-guard to be installed.

Tier 2 (fallback): RegexOnlyMasker — compiled-regex scanner that is always
                   available, zero extra dependencies, ~1 ms per call.

Startup behaviour (Item 5):
  - Engine always starts with whatever tier is available.
  - run_startup_validation() runs the 5 synthetic prompts through a full
    mask → unmask round-trip.
  - If the primary (NLP) scanner fails validation it is downgraded to the
    regex-only tier and a high-priority MASKING_DEGRADED_TO_REGEX alert is
    emitted to the Telemetry Logger.  The gateway stays **online** (is_healthy=True).
  - Only if the regex tier also fails validation does the engine set
    is_healthy=False and the ingress handler return HTTP 503.

This keeps the gateway available under NLP-model load failures while still
maintaining a baseline of safety through regex matching.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional LLM Guard import (Tier 1)
# ---------------------------------------------------------------------------
try:
    from llm_guard.input_scanners import Anonymize  # type: ignore[import]
    _HAS_LLM_GUARD = True
except ImportError:
    _HAS_LLM_GUARD = False
    Anonymize = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Regex patterns shared by both the fallback scanner and the placeholder map
# builder.  Extend here to add new entity types — see the performance-budget
# steering file before adding patterns.
# ---------------------------------------------------------------------------
_FALLBACK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN",          re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL_ADDRESS", re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    )),
    ("PHONE_NUMBER",  re.compile(
        r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"
    )),
    ("CREDIT_CARD",   re.compile(r"\b(?:\d{4}[\s\-]){3}\d{4}\b")),
]

_WS_RE = re.compile(r"\s+")


def _normalise(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


# ---------------------------------------------------------------------------
# Tier 2: RegexOnlyMasker
# ---------------------------------------------------------------------------

class RegexOnlyMasker:
    """Pure-regex PII scanner — always available, zero extra dependencies.

    Processes all pattern matches in document order (earliest position first)
    so the placeholder → original mapping is position-consistent.

    Per the performance-budget steering file, scan() must run in < 30 ms
    on worst-case 32 768-char input.  Do not add NLP calls here.
    """

    name: str = "regex"

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        """Return (masked_prompt, is_valid, risk_score).

        is_valid=False signals PII was detected (mirrors LLM Guard semantics).
        """
        all_matches: list[tuple[int, int, str]] = []

        for entity_type, pattern in _FALLBACK_PATTERNS:
            for m in pattern.finditer(prompt):
                all_matches.append((m.start(), m.end(), entity_type))

        if not all_matches:
            return prompt, True, 0.0

        # Sort by start position; remove overlaps (keep earliest)
        all_matches.sort(key=lambda x: x[0])
        deduped: list[tuple[int, int, str]] = []
        last_end = -1
        for start, end, etype in all_matches:
            if start >= last_end:
                deduped.append((start, end, etype))
                last_end = end

        # Build masked string in forward order
        counts: dict[str, int] = {}
        parts: list[str] = []
        prev = 0
        for start, end, etype in deduped:
            counts[etype] = counts.get(etype, 0) + 1
            parts.append(prompt[prev:start])
            parts.append(f"[{etype}_REDACTED_{counts[etype]}]")
            prev = end
        parts.append(prompt[prev:])
        return "".join(parts), False, 1.0


# ---------------------------------------------------------------------------
# Tier 1: NLPMasker (wraps LLM Guard Anonymize)
# ---------------------------------------------------------------------------

class NLPMasker:
    """LLM Guard Anonymize-backed scanner.  Loaded once at startup."""

    name: str = "nlp"

    def __init__(self) -> None:
        self._inner = Anonymize(preamble="", allowed_names=[], hidden_names=[])

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        return self._inner.scan(prompt)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Placeholder map builder (used by both tiers)
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+_REDACTED(?:_\d+)?\]")


def _build_placeholder_map(original: str, masked: str) -> dict[str, str]:
    """Reconstruct {placeholder: original_value} by aligning document-order spans."""
    placeholders = _PLACEHOLDER_RE.findall(masked)
    if not placeholders:
        return {}

    spans: list[tuple[int, str]] = []
    for _, pattern in _FALLBACK_PATTERNS:
        for m in pattern.finditer(original):
            spans.append((m.start(), m.group(0)))
    spans.sort(key=lambda x: x[0])

    return {ph: orig for ph, (_, orig) in zip(placeholders, spans)}


# ---------------------------------------------------------------------------
# PIIMaskingEngine — public interface
# ---------------------------------------------------------------------------

class PIIMaskingEngine:
    """Two-tier PII masking engine with graceful NLP → regex degradation.

    Tier selection at startup:
      1. NLPMasker (LLM Guard) if available.
      2. RegexOnlyMasker as the always-available fallback.

    Startup validation:
      - If the NLP tier fails round-trip validation → downgrade to regex tier,
        log MASKING_DEGRADED_TO_REGEX, keep is_healthy=True.
      - If regex tier also fails → log MASKING_INTEGRITY_FAILURE, is_healthy=False.
    """

    _VALIDATION_PROMPTS: list[str] = [
        "My SSN is 123-45-6789 and I live at 10 Main St.",
        "Contact me at john.doe@example.com for details.",
        "Call me at (555) 867-5309 anytime.",
        "Card number 4111 1111 1111 1111 expires 12/26.",
        "Name: Alice Smith, SSN: 987-65-4321, email: alice@corp.org",
    ]

    def __init__(self, telemetry_logger: Any | None = None) -> None:
        self._maps: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()
        self.is_healthy: bool = True
        self._telemetry_logger = telemetry_logger

        # Select initial scanner tier
        if _HAS_LLM_GUARD:
            try:
                self._scanner: RegexOnlyMasker | NLPMasker = NLPMasker()
                logger.info("PIIMaskingEngine: NLP tier (LLM Guard) loaded")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PIIMaskingEngine: NLP tier failed to initialise (%s) — "
                    "pre-loading regex tier", exc
                )
                self._scanner = RegexOnlyMasker()
        else:
            logger.info("PIIMaskingEngine: llm-guard not installed — using regex tier")
            self._scanner = RegexOnlyMasker()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def active_tier(self) -> str:
        return self._scanner.name

    def mask(self, prompt: str, request_id: str) -> tuple[str, dict[str, str]]:
        """Replace PII tokens with typed placeholders."""
        sanitised, is_valid, _ = self._scanner.scan(prompt)
        placeholder_map: dict[str, str] = (
            _build_placeholder_map(prompt, sanitised) if not is_valid else {}
        )
        with self._lock:
            self._maps[request_id] = placeholder_map
        return sanitised, placeholder_map

    def unmask(self, masked_prompt: str, request_id: str) -> str:
        """Restore all placeholders using the stored per-request mapping."""
        with self._lock:
            placeholder_map = self._maps.get(request_id, {})
        result = masked_prompt
        for placeholder, original in placeholder_map.items():
            result = result.replace(placeholder, original)
        return result

    def discard_mapping(self, request_id: str) -> None:
        """Remove the per-request map after response delivery."""
        with self._lock:
            self._maps.pop(request_id, None)

    # ------------------------------------------------------------------
    # Startup validation — graceful degradation
    # ------------------------------------------------------------------

    async def run_startup_validation(self) -> bool:
        """Round-trip fidelity check with two-tier graceful degradation.

        Returns True (and keeps is_healthy=True) in both of:
          a) NLP tier passes all 5 prompts.
          b) NLP tier fails but regex tier passes all 5 prompts (degraded mode).

        Returns False only if both tiers fail, setting is_healthy=False.
        """
        passed = await self._validate_tier(self._scanner)

        if not passed and self._scanner.name == "nlp":
            # Downgrade to regex tier — gateway stays online
            logger.error(
                "MASKING_DEGRADED_TO_REGEX: NLP PII scanner failed startup validation. "
                "Downgrading to regex-only masker. A high-priority alert has been raised."
            )
            self._emit_degraded_alert()
            self._scanner = RegexOnlyMasker()
            passed = await self._validate_tier(self._scanner)

        if not passed:
            logger.error(
                "MASKING_INTEGRITY_FAILURE: both NLP and regex PII scanners failed "
                "startup validation — gateway entering 503 state."
            )
            self.is_healthy = False
            return False

        tier = self._scanner.name
        logger.info(
            "PII masking startup validation passed using %s tier (%d prompts)",
            tier, len(self._VALIDATION_PROMPTS),
        )
        self.is_healthy = True
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _validate_tier(
        self, scanner: RegexOnlyMasker | NLPMasker
    ) -> bool:
        """Run all validation prompts through mask → unmask for the given scanner.

        Temporarily swaps self._scanner to the provided one so mask/unmask
        use it, then restores the previous scanner regardless of outcome.
        Scanner exceptions are caught and treated as a validation failure
        (not propagated), so a broken NLP model OOM never kills startup.
        """
        original = self._scanner
        self._scanner = scanner
        try:
            for i, prompt in enumerate(self._VALIDATION_PROMPTS):
                rid = f"__startup_validation_{i}__"
                try:
                    masked, _ = await asyncio.to_thread(self.mask, prompt, rid)
                    restored = await asyncio.to_thread(self.unmask, masked, rid)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Tier '%s' raised on prompt %d: %s — treating as fidelity failure",
                        scanner.name, i, exc,
                    )
                    return False
                finally:
                    self.discard_mapping(rid)

                if _normalise(restored) != _normalise(prompt):
                    logger.warning(
                        "Tier '%s' failed round-trip on prompt %d: "
                        "original=%r restored=%r",
                        scanner.name, i, _normalise(prompt), _normalise(restored),
                    )
                    return False
        finally:
            self._scanner = original
        return True

    def _emit_degraded_alert(self) -> None:
        """Log a high-priority telemetry alert when degrading to the regex tier."""
        if self._telemetry_logger is None:
            # No logger wired yet — write directly to stderr so it's never silent
            import sys
            print(
                "ALERT [MASKING_DEGRADED_TO_REGEX] NLP PII scanner failed startup "
                "validation. Gateway is running in reduced-accuracy regex-only mode.",
                file=sys.stderr,
                flush=True,
            )
            return

        # Fire-and-forget async alert via the telemetry logger
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(
                    self._telemetry_logger.record_alert(
                        alert_type="MASKING_DEGRADED_TO_REGEX",
                        severity="HIGH",
                        detail=(
                            "NLP PII scanner failed startup validation round-trip. "
                            "Gateway operating in regex-only masking mode."
                        ),
                    )
                )
        except Exception:  # noqa: BLE001
            pass  # alert emission must never crash startup
