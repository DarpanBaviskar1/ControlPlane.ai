"""NLI-based groundedness scorer using a cross-encoder model.

Wraps the ``cross-encoder/nli-deberta-v3-small`` model from sentence-transformers
to produce ENTAILMENT / NEUTRAL / CONTRADICTION verdicts for (document, response)
pairs.  All inference is offloaded via ``asyncio.to_thread`` so the FastAPI event
loop is never blocked.

The module degrades gracefully when ``sentence-transformers`` is not installed:
``score_pairs()`` returns an empty list and callers should treat ``nli_label`` as
``None``.

Requirements: 2.3, 2.5, 2.7, 2.8, 6.3, 6.4, 6.5
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard (Req. 2.5, 6.3)
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import CrossEncoder  # type: ignore[import]

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
    logger.info(
        "sentence-transformers not installed — NLI scoring disabled; nli_label=None"
    )

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

NLILabel = Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]

# Map from cross-encoder label strings to the canonical NLILabel
# cross-encoder/nli-deberta-v3-small returns "entailment", "neutral", "contradiction"
_LABEL_MAP: dict[str, NLILabel] = {
    "entailment": "ENTAILMENT",
    "neutral": "NEUTRAL",
    "contradiction": "CONTRADICTION",
    # uppercase variants — defensive handling
    "ENTAILMENT": "ENTAILMENT",
    "NEUTRAL": "NEUTRAL",
    "CONTRADICTION": "CONTRADICTION",
}


# ---------------------------------------------------------------------------
# NLIScorer
# ---------------------------------------------------------------------------


class NLIScorer:
    """Cross-encoder NLI scorer wrapping sentence-transformers.

    Instantiated once in ``lifespan()`` (Req. 2.7) and stored in
    ``app.state.nli_scorer``.  All inference is executed via
    ``asyncio.to_thread`` (Req. 2.8, 6.5).

    When ``sentence-transformers`` is not installed, the constructor succeeds
    (no-op) and ``score_pairs()`` always returns an empty list (Req. 2.5).
    """

    _DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        """Load the cross-encoder model synchronously.

        This method is intended to be called inside ``asyncio.to_thread`` at
        startup so that the model download/load does not block the event loop
        (Req. 6.4).  When ``sentence-transformers`` is absent the constructor
        records ``None`` and all subsequent calls are no-ops (Req. 2.5).
        """
        self._model_name = model_name
        if _HAS_SENTENCE_TRANSFORMERS:
            # CrossEncoder loads / downloads the model weights synchronously.
            self._model: CrossEncoder | None = CrossEncoder(model_name)
            logger.info("NLIScorer loaded model: %s", model_name)
        else:
            self._model = None

    # ------------------------------------------------------------------
    # Synchronous scoring (runs inside asyncio.to_thread)
    # ------------------------------------------------------------------

    def score_pairs_sync(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[tuple[NLILabel, float]]:
        """Score a list of (document_text, response_text) pairs synchronously.

        Returns a list of ``(NLILabel, confidence_score)`` tuples in the same
        order as the input pairs.  Returns an empty list when
        ``sentence-transformers`` is not installed (Req. 2.5) or when ``pairs``
        is empty.
        """
        if not _HAS_SENTENCE_TRANSFORMERS or self._model is None or not pairs:
            return []

        # CrossEncoder.predict() returns a list of dicts:
        # [{"label": "entailment", "score": 0.95}, ...]
        # when apply_softmax=True (default for NLI models).
        raw_results: list[dict] = self._model.predict(
            pairs,
            apply_softmax=True,
            convert_to_numpy=False,
        )

        results: list[tuple[NLILabel, float]] = []
        for item in raw_results:
            raw_label: str = item.get("label", "neutral")
            score: float = float(item.get("score", 0.0))
            canonical = _LABEL_MAP.get(raw_label, "NEUTRAL")
            results.append((canonical, score))

        return results

    # ------------------------------------------------------------------
    # Async wrapper (Req. 2.8, 6.5)
    # ------------------------------------------------------------------

    async def score_pairs(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[tuple[NLILabel, float]]:
        """Async wrapper that offloads ``score_pairs_sync`` to a thread pool.

        Ensures the FastAPI event loop is never blocked by cross-encoder
        inference (Req. 2.8, 6.5).
        """
        return await asyncio.to_thread(self.score_pairs_sync, pairs)

    # ------------------------------------------------------------------
    # Aggregation (Req. 2.9)
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate(labels: list[NLILabel]) -> NLILabel:
        """Derive a single aggregate label from a list of per-pair labels.

        Priority rule: CONTRADICTION > ENTAILMENT > NEUTRAL (Req. 2.9).

        >>> NLIScorer.aggregate(["ENTAILMENT", "CONTRADICTION", "NEUTRAL"])
        'CONTRADICTION'
        >>> NLIScorer.aggregate(["ENTAILMENT", "NEUTRAL"])
        'ENTAILMENT'
        >>> NLIScorer.aggregate(["NEUTRAL"])
        'NEUTRAL'
        >>> NLIScorer.aggregate([])
        'NEUTRAL'
        """
        if "CONTRADICTION" in labels:
            return "CONTRADICTION"
        if "ENTAILMENT" in labels:
            return "ENTAILMENT"
        return "NEUTRAL"
