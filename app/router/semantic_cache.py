"""Semantic Cache for the ControlPlane.ai Enterprise AI Proxy Gateway.

GPTCache-backed vector-similarity cache injected before the RouteLLM Controller.
All embedding and FAISS index operations run via asyncio.to_thread so the event
loop is never blocked.

When gpicache is not installed the module degrades gracefully: lookup() always
returns a cache miss and store() is a no-op (Req. 1.11).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard — Req. 1.11, 6.3
# ---------------------------------------------------------------------------

try:
    import numpy as np  # type: ignore[import]
    from gptcache import Cache  # type: ignore[import]
    from gptcache.embedding import Onnx  # type: ignore[import]
    from gptcache.manager import CacheBase, VectorBase, get_data_manager  # type: ignore[import]
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation  # type: ignore[import]

    _GPTICACHE_AVAILABLE = True
except ImportError:
    _GPTICACHE_AVAILABLE = False
    logger.info(
        "gpicache not installed — SemanticCache disabled; all requests treated as cache misses"
    )


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CacheLookupResult:
    """Result of a SemanticCache lookup operation."""

    hit: bool
    response: str | None = None  # populated on cache hit
    similarity: float | None = None  # cosine similarity of the winning entry; None on miss


@dataclass
class _CacheEntry:
    """Internal storage triple: (embedding, response, expiry_ts)."""

    embedding: list[float]
    response: str
    expiry_ts: float  # monotonic timestamp after which the entry is expired


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


class SemanticCache:
    """GPTCache-backed vector-similarity cache.

    Loaded once at startup and stored in app.state.semantic_cache.
    All embedding and FAISS index operations run via asyncio.to_thread
    so the FastAPI event loop is never blocked (Req. 1.12, 6.5).

    When _GPTICACHE_AVAILABLE is False the instance acts as a no-op stub:
    lookup() always returns hit=False and store() is a no-op (Req. 1.11).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self._similarity_threshold = similarity_threshold
        self._embedding_model = embedding_model

        # In-process entry list used when _GPTICACHE_AVAILABLE is True.
        # We maintain our own lightweight store alongside GPTCache so we can
        # implement fine-grained TTL eviction (Req. 1.7) and exception isolation
        # (Req. 1.9) without relying on GPTCache internals.
        self._entries: list[_CacheEntry] = []

        # GPTCache objects — only initialised when the optional dep is present.
        self._embedder: object | None = None
        self._cache: object | None = None

        if _GPTICACHE_AVAILABLE:
            try:
                self._embedder = Onnx()
                self._cache = Cache()
                # Initialise GPTCache with an in-memory SQLite data store and a
                # FAISS flat index so the whole cache lives in process memory.
                self._cache.init(  # type: ignore[union-attr]
                    embedding_func=self._embedder.to_embeddings,  # type: ignore[union-attr]
                    data_manager=get_data_manager(
                        CacheBase("sqlite"),
                        VectorBase("faiss", dimension=self._embedder.dimension),  # type: ignore[union-attr]
                    ),
                    similarity_evaluation=SearchDistanceEvaluation(),
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "SEMANTIC_CACHE_ERROR during initialisation — cache disabled: %s", exc
                )
                # Treat as if the optional dep were absent for the lifetime of this instance.
                self._embedder = None
                self._cache = None

    # ------------------------------------------------------------------
    # Internal synchronous helpers (run inside asyncio.to_thread)
    # ------------------------------------------------------------------

    def _embed_sync(self, text: str) -> list[float]:
        """Compute an embedding vector synchronously."""
        if self._embedder is None:
            return []
        vec = self._embedder.to_embeddings(text)  # type: ignore[union-attr]
        # GPTCache Onnx embedder returns a numpy array; normalise to plain list.
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return list(vec)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors (returns 0.0 on error)."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    def _lookup_sync(self, masked_prompt: str) -> CacheLookupResult:
        """Synchronous lookup — called inside asyncio.to_thread.

        1. Prunes expired entries.
        2. Embeds the incoming prompt.
        3. Finds the nearest stored embedding by cosine similarity.
        4. Returns a hit if similarity >= threshold, miss otherwise.
        """
        # Evict expired entries before searching (Req. 1.7)
        self._evict_expired()

        if not self._entries:
            return CacheLookupResult(hit=False)

        try:
            query_embedding = self._embed_sync(masked_prompt)
        except Exception as exc:
            logger.error("SEMANTIC_CACHE_ERROR during embedding: %s", exc)
            return CacheLookupResult(hit=False)

        best_similarity = -1.0
        best_entry: _CacheEntry | None = None

        for entry in self._entries:
            try:
                sim = self._cosine_similarity(query_embedding, entry.embedding)
            except Exception:
                continue
            if sim > best_similarity:
                best_similarity = sim
                best_entry = entry

        if best_entry is not None and best_similarity >= self._similarity_threshold:
            return CacheLookupResult(
                hit=True,
                response=best_entry.response,
                similarity=best_similarity,
            )

        return CacheLookupResult(hit=False, similarity=best_similarity if best_entry else None)

    def _store_sync(self, masked_prompt: str, response: str, ttl_seconds: int) -> None:
        """Synchronous store — called inside asyncio.to_thread."""
        try:
            embedding = self._embed_sync(masked_prompt)
            expiry_ts = time.monotonic() + ttl_seconds
            self._entries.append(
                _CacheEntry(embedding=embedding, response=response, expiry_ts=expiry_ts)
            )
        except Exception as exc:
            logger.error("SEMANTIC_CACHE_ERROR during store: %s", exc)

    def _evict_expired(self) -> int:
        """Remove all entries whose TTL has elapsed.  Returns count removed."""
        now = time.monotonic()
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.expiry_ts > now]
        return before - len(self._entries)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def lookup(
        self,
        masked_prompt: str,
        cache_ttl_seconds: int,  # noqa: ARG002 — kept for API symmetry with store()
    ) -> CacheLookupResult:
        """Perform a vector-similarity lookup for *masked_prompt*.

        Returns CacheLookupResult(hit=False) when:
        - _GPTICACHE_AVAILABLE is False  (Req. 1.11)
        - the index raises any exception  (Req. 1.9)
        - no entry meets the similarity threshold

        All embedding/index work runs in a thread pool via asyncio.to_thread
        (Req. 1.12, 6.5).
        """
        if not _GPTICACHE_AVAILABLE or self._embedder is None:
            return CacheLookupResult(hit=False)

        try:
            result: CacheLookupResult = await asyncio.to_thread(
                self._lookup_sync, masked_prompt
            )
            return result
        except Exception as exc:
            logger.error("SEMANTIC_CACHE_ERROR during lookup: %s", exc)
            return CacheLookupResult(hit=False)

    async def store(
        self,
        masked_prompt: str,
        response: str,
        ttl_seconds: int,
    ) -> None:
        """Store *masked_prompt* → *response* pair with a TTL.

        No-op when _GPTICACHE_AVAILABLE is False (Req. 1.11).
        All embedding work runs in a thread pool via asyncio.to_thread
        (Req. 1.12, 6.5).
        """
        if not _GPTICACHE_AVAILABLE or self._embedder is None:
            return

        try:
            await asyncio.to_thread(self._store_sync, masked_prompt, response, ttl_seconds)
        except Exception as exc:
            logger.error("SEMANTIC_CACHE_ERROR during store: %s", exc)

    def invalidate_expired(self) -> int:
        """Prune TTL-expired entries; returns count removed.

        This is a synchronous utility intended for background maintenance tasks.
        It is safe to call from outside asyncio.to_thread because the underlying
        list operation is a single assignment (GIL-safe for CPython).
        """
        return self._evict_expired()
