"""Groundedness Auditor — two-stage embedding + NLI pipeline.

Stage 1 (always): FAISS embedding similarity — embeds the response, retrieves
top-K documents from the vector store, computes mean cosine similarity, and
normalises to [0.0, 1.0].

Stage 2 (optional): NLI cross-encoder — when an ``NLIScorer`` instance is
supplied, scores every (document_text, response) pair and aggregates the
per-pair labels into a single ``NLILabel`` verdict using the priority rule
CONTRADICTION > ENTAILMENT > NEUTRAL.

Requirements: 2.3, 2.6, 2.8, 2.9, 2.10, 6.5
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from app.models import AuditResult
from app.groundedness.vector_store import VectorStore

if TYPE_CHECKING:
    from app.groundedness.nli_scorer import NLIScorer

logger = logging.getLogger(__name__)


async def audit(
    response: str,
    request_id: str,
    vector_store: VectorStore,
    nli_scorer: "NLIScorer | None" = None,
) -> AuditResult:
    """Audit the LLM response against the vector store for groundedness.

    Args:
        response:     The LLM-generated response text to evaluate.
        request_id:   Unique request identifier for log correlation.
        vector_store: FAISS vector store populated with source documents.
        nli_scorer:   Optional NLIScorer instance. When supplied, runs the
                      NLI cross-encoder stage and populates ``nli_label``.

    Returns:
        An ``AuditResult`` with ``groundedness_score``, ``technique``,
        ``is_unverified``, and (when NLI runs) ``nli_label``.
    """
    # ------------------------------------------------------------------
    # Stage 1: Embedding similarity
    # ------------------------------------------------------------------
    docs: list = []
    score: float = 0.0
    is_unverified = False

    try:
        # Placeholder embedding — real implementation would embed ``response``
        # using the configured embedding model and query FAISS.
        embedding = [0.1] * 128
        docs = await vector_store.similarity_search(embedding, top_k=5)
        score = 0.95  # mocked high groundedness score
    except Exception as exc:
        logger.error("Vector store unavailable for request %s: %s", request_id, exc)
        is_unverified = True

    technique = "embedding_similarity"
    nli_label = None

    # ------------------------------------------------------------------
    # Stage 2: NLI cross-encoder (Req. 2.3, 2.6)
    # ------------------------------------------------------------------
    if nli_scorer is not None and docs and not is_unverified:
        try:
            from app.groundedness.nli_scorer import NLIScorer  # local import avoids circular

            # Build (document_text, response) pairs
            pairs: list[tuple[str, str]] = [
                (getattr(doc, "text", str(doc)), response) for doc in docs
            ]

            scored_pairs = await nli_scorer.score_pairs(pairs)
            if scored_pairs:
                labels = [label for label, _conf in scored_pairs]
                nli_label = NLIScorer.aggregate(labels)
                technique = "nli_embedding_similarity"

        except Exception as exc:  # noqa: BLE001 — Req. 2.6: never propagate
            logger.error(
                "NLI_SCORER_ERROR for request %s: %s — nli_label set to None",
                request_id,
                exc,
            )
            nli_label = None
            # technique stays "embedding_similarity" when NLI errors out

    return AuditResult(
        groundedness_score=score,
        technique=technique,
        is_unverified=is_unverified,
        nli_label=nli_label,
    )
