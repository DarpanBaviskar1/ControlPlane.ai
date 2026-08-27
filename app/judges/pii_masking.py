"""PII Masking Engine.

Wraps the LLM Guard Anonymize scanner (or a regex-based fallback when LLM
Guard is not installed) and maintains a per-request placeholder mapping.

Responsibilities:
- mask(prompt, request_id)   → (masked_prompt, placeholder_map)
- unmask(masked, request_id) → original prompt
- discard_mapping(request_id)→ clears the per-request map
- run_startup_validation()   → round-trip fidelity check on 5 synthetic prompts

The engine is healthy (is_healthy=True) after a successful startup validation
pass and unhealthy (is_healthy=False) after any failure. The ingress handler
returns HTTP 503 while the engine is unhealthy.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional LLM Guard import
# ---------------------------------------------------------------------------

try:
    from llm_guard.input_scanners import Anonymize  # type: ignore[import]
    from llm_guard.input_scanners.anonymize_helpers import ANONYMIZE_PATTERNS  # type: ignore[import]

    _HAS_LLM_GUARD = True
except ImportError:
    _HAS_LLM_GUARD = False
    Anonymize = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Regex-based fallback scanner (used when llm-guard is not installed)
# ---------------------------------------------------------------------------

# Ordered list of (entity_type, compiled_pattern) used by the fallback scanner.
# Patterns are intentionally simple; production uses LLM Guard / Presidio.
_FALLBACK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # SSN: 123-45-6789
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Email
    ("EMAIL_ADDRESS", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    # US phone: various formats
    (
        "PHONE_NUMBER",
        re.compile(r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"),
    ),
    # Credit card (naive 16-digit groups)
    (
        "CREDIT_CARD",
        re.compile(r"\b(?:\d{4}[\s\-]){3}\d{4}\b"),
    ),
]

# Whitespace normalisation for round-trip fidelity checks
_WS_RE = re.compile(r"\s+")


def _normalise(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


class _FallbackScanner:
    """Simple regex-based PII scanner used when llm-guard is unavailable.

    Replaces matches in document order (earliest position first) so that the
    placeholder → original mapping stays position-consistent.
    """

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        """Returns (sanitised_prompt, is_valid, risk_score).

        is_valid=False means PII was found (matches LLM Guard's semantics).
        """
        # Collect all matches across all patterns, sorted by start position
        all_matches: list[tuple[int, int, str, str]] = []  # (start, end, entity_type, value)
        entity_counts: dict[str, int] = {}

        for entity_type, pattern in _FALLBACK_PATTERNS:
            for match in pattern.finditer(prompt):
                all_matches.append((match.start(), match.end(), entity_type, match.group(0)))

        if not all_matches:
            return prompt, True, 0.0

        # Sort by start position to process left-to-right
        all_matches.sort(key=lambda x: x[0])

        # Remove overlapping matches (keep first / earliest)
        deduplicated: list[tuple[int, int, str, str]] = []
        last_end = -1
        for start, end, etype, value in all_matches:
            if start >= last_end:
                deduplicated.append((start, end, etype, value))
                last_end = end

        # Build masked string by replacing in reverse order (to preserve indices)
        masked = prompt
        for start, end, entity_type, _ in reversed(deduplicated):
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            placeholder = f"[{entity_type}_REDACTED_{entity_counts[entity_type]}]"
            masked = masked[:start] + placeholder + masked[end:]

        # Re-count in forward order for consistent numbering
        entity_counts_fwd: dict[str, int] = {}
        parts: list[str] = []
        prev = 0
        for start, end, entity_type, _ in deduplicated:
            entity_counts_fwd[entity_type] = entity_counts_fwd.get(entity_type, 0) + 1
            placeholder = f"[{entity_type}_REDACTED_{entity_counts_fwd[entity_type]}]"
            parts.append(prompt[prev:start])
            parts.append(placeholder)
            prev = end
        parts.append(prompt[prev:])
        masked = "".join(parts)

        return masked, False, 1.0


# ---------------------------------------------------------------------------
# PIIMaskingEngine
# ---------------------------------------------------------------------------


class PIIMaskingEngine:
    """Thread-safe PII masking engine with per-request placeholder maps.

    All public methods are synchronous so they can be called from both sync
    and async contexts (the scanner's async path uses asyncio.to_thread).
    """

    # Five synthetic prompts used by the startup validation suite
    _VALIDATION_PROMPTS: list[str] = [
        "My SSN is 123-45-6789 and I live at 10 Main St.",
        "Contact me at john.doe@example.com for details.",
        "Call me at (555) 867-5309 anytime.",
        "Card number 4111 1111 1111 1111 expires 12/26.",
        "Name: Alice Smith, SSN: 987-65-4321, email: alice@corp.org",
    ]

    def __init__(self) -> None:
        # per-request placeholder maps: {request_id: {placeholder: original}}
        self._maps: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()
        self.is_healthy: bool = True  # optimistic; set False on validation failure

        if _HAS_LLM_GUARD:
            # Load once; expensive model download happens here
            self._scanner = Anonymize(preamble="", allowed_names=[], hidden_names=[])
        else:
            logger.warning(
                "llm-guard not installed; using regex fallback for PII detection"
            )
            self._scanner = _FallbackScanner()

    # ------------------------------------------------------------------
    # mask
    # ------------------------------------------------------------------

    def mask(self, prompt: str, request_id: str) -> tuple[str, dict[str, str]]:
        """Replace PII tokens with typed placeholders.

        Returns (masked_prompt, placeholder_map).  The placeholder_map maps
        each placeholder back to the original PII value and is stored internally
        keyed by request_id.
        """
        sanitised, is_valid, _ = self._scanner.scan(prompt)

        # Build the reverse mapping: placeholder → original
        placeholder_map: dict[str, str] = {}
        if not is_valid:
            placeholder_map = _build_placeholder_map(prompt, sanitised)

        with self._lock:
            self._maps[request_id] = placeholder_map

        return sanitised, placeholder_map

    # ------------------------------------------------------------------
    # unmask
    # ------------------------------------------------------------------

    def unmask(self, masked_prompt: str, request_id: str) -> str:
        """Restore all placeholders using the stored per-request mapping."""
        with self._lock:
            placeholder_map = self._maps.get(request_id, {})

        result = masked_prompt
        for placeholder, original in placeholder_map.items():
            result = result.replace(placeholder, original)
        return result

    # ------------------------------------------------------------------
    # discard_mapping
    # ------------------------------------------------------------------

    def discard_mapping(self, request_id: str) -> None:
        """Remove the per-request placeholder map after response delivery."""
        with self._lock:
            self._maps.pop(request_id, None)

    # ------------------------------------------------------------------
    # Startup validation
    # ------------------------------------------------------------------

    async def run_startup_validation(self) -> bool:
        """Round-trip fidelity check on five synthetic PII prompts.

        Returns True if all prompts survive mask → unmask with byte-for-byte
        identity after whitespace normalisation.  Sets is_healthy=False and
        logs MASKING_INTEGRITY_FAILURE on any failure.
        """
        for i, prompt in enumerate(self._VALIDATION_PROMPTS):
            request_id = f"__startup_validation_{i}__"
            try:
                masked, _ = await asyncio.to_thread(self.mask, prompt, request_id)
                restored = await asyncio.to_thread(self.unmask, masked, request_id)
            finally:
                self.discard_mapping(request_id)

            if _normalise(restored) != _normalise(prompt):
                logger.error(
                    "MASKING_INTEGRITY_FAILURE: prompt %d failed round-trip. "
                    "original=%r  restored=%r",
                    i,
                    _normalise(prompt),
                    _normalise(restored),
                )
                self.is_healthy = False
                return False

        self.is_healthy = True
        logger.info("PII masking startup validation passed (%d prompts)", len(self._VALIDATION_PROMPTS))
        return True


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _build_placeholder_map(original: str, masked: str) -> dict[str, str]:
    """Reconstruct a placeholder → original mapping by diffing the two strings.

    Extracts all PII values from the original in document order and all
    placeholder tokens from the masked string in document order, then
    zips them together positionally.
    """
    placeholder_map: dict[str, str] = {}

    # Extract placeholder tokens from masked string in order of appearance
    placeholder_re = re.compile(r"\[[A-Z_]+_REDACTED(?:_\d+)?\]")
    placeholders = placeholder_re.findall(masked)

    if not placeholders:
        return {}

    # Extract original PII spans in document order (by start position)
    spans: list[tuple[int, str]] = []
    for _, pattern in _FALLBACK_PATTERNS:
        for match in pattern.finditer(original):
            spans.append((match.start(), match.group(0)))

    # Sort by position in the original string
    spans.sort(key=lambda x: x[0])
    originals = [v for _, v in spans]

    for placeholder, original_val in zip(placeholders, originals):
        placeholder_map[placeholder] = original_val

    return placeholder_map
