"""P3 Judge — Prompt Clarity classifier.

Lightweight local classifier — no external model call beyond spaCy's
en_core_web_sm dependency parse.

Rules (from Requirement 2.7 / Property 7):
  AMBIGUOUS if:
    - token_count (via tiktoken) <= 10, OR
    - spaCy dependency parse produces no token with both:
        dep_ == "ROOT"  AND  pos_ == "VERB"
  CLEAR otherwise.

Both tiktoken and spaCy are loaded once at startup via load_models().
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import tiktoken  # type: ignore[import]

    _enc = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False
    _enc = None

try:
    import spacy  # type: ignore[import]

    _HAS_SPACY = False  # set to True after load_models()
    _nlp = None
    # Eagerly try to load the model at import time so tests don't need
    # an explicit load_models() call. load_models() is still callable to
    # re-load or force a fresh attempt from the FastAPI lifespan.
    try:
        _nlp = spacy.load("en_core_web_sm")
        _HAS_SPACY = True
    except Exception:
        pass  # model not yet downloaded; load_models() must be called explicitly
except ImportError:
    _HAS_SPACY = False
    _nlp = None


# ---------------------------------------------------------------------------
# Startup loader
# ---------------------------------------------------------------------------


def load_models() -> None:
    """Load spaCy en_core_web_sm.  Call once during FastAPI lifespan startup."""
    global _HAS_SPACY, _nlp

    try:
        import spacy  # noqa: PLC0415

        _nlp = spacy.load("en_core_web_sm")
        _HAS_SPACY = True
        logger.info("P3: spaCy en_core_web_sm loaded")
    except Exception as exc:
        logger.warning("P3: could not load spaCy model — falling back to token-only rule: %s", exc)
        _HAS_SPACY = False


# ---------------------------------------------------------------------------
# Core classification logic (sync, safe to call from asyncio.to_thread)
# ---------------------------------------------------------------------------


def _count_tokens(prompt: str) -> int:
    """Count tokens using tiktoken cl100k_base, or whitespace split as fallback."""
    if _HAS_TIKTOKEN and _enc is not None:
        return len(_enc.encode(prompt))
    return len(prompt.split())


def _has_root_verb(prompt: str) -> bool:
    """Return True if the spaCy parse contains a ROOT-tagged verb token.

    Accepts ROOT tokens with pos_ in {"VERB", "AUX"} — auxiliary verbs
    (e.g. "are", "is", "do") function as the main verb in English questions
    and copular sentences and should count as CLEAR.
    """
    if not _HAS_SPACY or _nlp is None:
        # Without spaCy we cannot check for verbs; assume verb present so only
        # the token-count rule applies.
        return True

    doc = _nlp(prompt)
    return any(token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"} for token in doc)


def _classify(prompt: str) -> Literal["CLEAR", "AMBIGUOUS"]:
    """Synchronous classification — offloaded to a thread pool by p3_judge()."""
    token_count = _count_tokens(prompt)
    if token_count <= 10:
        return "AMBIGUOUS"
    if not _has_root_verb(prompt):
        return "AMBIGUOUS"
    return "CLEAR"


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------


async def p3_judge(prompt: str) -> Literal["CLEAR", "AMBIGUOUS"]:
    """Classify prompt clarity asynchronously.

    Offloads the CPU-bound spaCy parse to a thread pool.
    Failure defaults to AMBIGUOUS (most restrictive safe default).
    """
    try:
        return await asyncio.to_thread(_classify, prompt)
    except Exception as exc:
        logger.error("P3 judge error: %s — defaulting to AMBIGUOUS", exc)
        return "AMBIGUOUS"
