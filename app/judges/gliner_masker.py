"""GLiNER Custom Entity Masker — optional Tier 1.5 in the PII Masking Engine.

This module provides zero-shot named-entity recognition for domain-specific
corporate terms that standard NLP/regex tiers do not cover.  It is inserted
between NLPMasker (Tier 1) and RegexOnlyMasker (Tier 2) when both conditions
hold at call-time:
  * the ``gliner`` package is installed in the current environment, AND
  * the active ``UseCaseProfile`` supplies a non-empty ``custom_entity_terms`` list.

The GLiNER model is loaded once during ``lifespan()`` and stored in
``app.state.gliner_model``; GLiNERMasker itself holds **no** model reference so
that the class can be instantiated cheaply without any I/O.

All GLiNER inference is offloaded via ``asyncio.to_thread`` (``scan`` method)
so the FastAPI event loop is never blocked.  The synchronous path (``scan_sync``)
is exposed for use in startup validation and direct thread-pool dispatch.
"""

from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional gliner import (graceful degradation — Req. 3.3, 6.3)
# ---------------------------------------------------------------------------
try:
    import gliner as _gliner_lib  # type: ignore[import]
    _HAS_GLINER = True
except ImportError:
    _HAS_GLINER = False
    _gliner_lib = None  # type: ignore[assignment]
    logger.info(
        "gliner not installed — custom entity masking disabled; "
        "GLiNERMasker tier will be skipped for all requests"
    )

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Compiled regex that matches any GLiNER-produced placeholder (Req. 3.5).
# Used by callers to detect and process these tokens without re-compiling.
_CUSTOM_PLACEHOLDER_RE = re.compile(r"\[CUSTOM_ENTITY_REDACTED_\d+\]")


# ---------------------------------------------------------------------------
# GLiNERMasker
# ---------------------------------------------------------------------------

class GLiNERMasker:
    """Zero-shot named-entity scanner for custom corporate terms.

    The GLiNER model is **injected at call-time** from ``app.state.gliner_model``
    so this class holds no model reference itself and is safe to instantiate
    at module import time.

    Produces ``[CUSTOM_ENTITY_REDACTED_N]`` placeholders in document order
    (N is a sequential integer starting at 1), and returns a mapping of
    ``{placeholder: original_text}`` for later unmasking.
    """

    name: str = "gliner"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scan_sync(
        self,
        prompt: str,
        entity_terms: list[str],
        gliner_model: object,
    ) -> tuple[str, dict[str, str]]:
        """Synchronous GLiNER scan.

        Args:
            prompt:        The text to scan (already partially masked by Tier 1).
            entity_terms:  List of entity label strings to detect (e.g. ``["Project Phoenix"]``).
            gliner_model:  A loaded ``gliner.GLiNER`` instance (from ``app.state.gliner_model``).

        Returns:
            A 2-tuple of:
              * ``masked_prompt`` — the input text with detected entities replaced
                by ``[CUSTOM_ENTITY_REDACTED_N]`` tokens.
              * ``placeholder_map`` — ``{placeholder: original_text}`` in document order.

        Notes:
            - Entities are replaced in document order (earliest start offset first).
            - Overlapping spans are handled by a greedy earliest-wins strategy.
            - When no entities are detected the original prompt and an empty map
              are returned unchanged.
        """
        # gliner_model.predict_entities returns a list of dicts with at least
        # {"text": str, "start": int, "end": int, "label": str, "score": float}.
        raw_entities: list[dict] = gliner_model.predict_entities(  # type: ignore[union-attr]
            prompt, entity_terms
        )

        if not raw_entities:
            return prompt, {}

        # Sort by start offset; greedy deduplication for overlapping spans
        raw_entities.sort(key=lambda e: e["start"])
        deduped: list[dict] = []
        last_end = -1
        for ent in raw_entities:
            if ent["start"] >= last_end:
                deduped.append(ent)
                last_end = ent["end"]

        # Build masked string and placeholder map in document order
        placeholder_map: dict[str, str] = {}
        parts: list[str] = []
        prev = 0
        for n, ent in enumerate(deduped, start=1):
            start: int = ent["start"]
            end: int = ent["end"]
            placeholder = f"[CUSTOM_ENTITY_REDACTED_{n}]"
            parts.append(prompt[prev:start])
            parts.append(placeholder)
            placeholder_map[placeholder] = ent["text"]
            prev = end
        parts.append(prompt[prev:])

        return "".join(parts), placeholder_map

    async def scan(
        self,
        prompt: str,
        entity_terms: list[str],
        gliner_model: object,
    ) -> tuple[str, dict[str, str]]:
        """Async wrapper around ``scan_sync`` — offloads to a thread pool.

        This keeps the FastAPI event loop unblocked during GLiNER inference
        (Req. 3.10, 6.5).
        """
        return await asyncio.to_thread(
            self.scan_sync, prompt, entity_terms, gliner_model
        )
