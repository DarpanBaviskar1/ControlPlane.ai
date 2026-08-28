"""Property-based tests for SemanticCache.

Tasks: 2.2, 2.3
Properties:
  SC-2 — TTL eviction: any entry with expiry_ts < time.monotonic() must produce hit=False
  SC-3 — Exception isolation: if the index raises, lookup() must return hit=False without propagating

Requirements: 1.7, 1.9
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from app.router.semantic_cache import SemanticCache, CacheLookupResult, _CacheEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache(threshold: float = 0.92) -> SemanticCache:
    """Return a SemanticCache instance that always acts in degraded no-op mode
    at the GPTCache level but with a populated _entries list we control directly."""
    return SemanticCache(similarity_threshold=threshold)


# ---------------------------------------------------------------------------
# SC-2: TTL eviction
# Property: for any CacheEntry with expiry_ts < time.monotonic(),
# lookup() must return hit=False.
# Requirements: 1.7
# ---------------------------------------------------------------------------

class TestSC2TTLEviction:
    """Property SC-2: expired entries must never produce a cache hit."""

    def test_expired_entry_returns_miss(self) -> None:
        """A single entry that expired 1 second ago must be evicted and return miss."""
        cache = _cache()
        # Inject a known embedding + expired timestamp directly
        past_ts = time.monotonic() - 1.0
        cache._entries = [
            _CacheEntry(
                embedding=[1.0] + [0.0] * 127,
                response="cached response",
                expiry_ts=past_ts,
            )
        ]
        # Eviction runs synchronously inside _lookup_sync
        result = cache._lookup_sync("any prompt")
        assert result.hit is False
        assert len(cache._entries) == 0, "Expired entry must be pruned"

    @given(st.floats(min_value=0.001, max_value=100.0, allow_nan=False))
    @settings(max_examples=100)
    def test_entry_expired_by_arbitrary_delta(self, delta_s: float) -> None:
        """For any positive past offset, the entry must be evicted."""
        cache = _cache()
        past_ts = time.monotonic() - delta_s
        cache._entries = [
            _CacheEntry(
                embedding=[1.0] + [0.0] * 127,
                response="stale",
                expiry_ts=past_ts,
            )
        ]
        result = cache._lookup_sync("query")
        assert result.hit is False

    def test_future_entry_not_evicted(self) -> None:
        """An entry with a far-future TTL must NOT be evicted."""
        cache = _cache(threshold=0.0)  # threshold=0 so any similarity is a hit
        future_ts = time.monotonic() + 3600.0
        # Use a known embedding so cosine similarity is predictable
        vec = [1.0] + [0.0] * 127
        cache._entries = [
            _CacheEntry(embedding=vec, response="fresh", expiry_ts=future_ts)
        ]
        # Patch _embed_sync to return the same vector so cosine sim == 1.0
        with patch.object(cache, "_embed_sync", return_value=vec):
            result = cache._lookup_sync("any prompt")
        assert result.hit is True
        assert result.response == "fresh"

    def test_invalidate_expired_returns_count(self) -> None:
        """invalidate_expired() must return the exact number of entries pruned."""
        cache = _cache()
        now = time.monotonic()
        cache._entries = [
            _CacheEntry(embedding=[], response="a", expiry_ts=now - 2),
            _CacheEntry(embedding=[], response="b", expiry_ts=now - 1),
            _CacheEntry(embedding=[], response="c", expiry_ts=now + 3600),
        ]
        removed = cache.invalidate_expired()
        assert removed == 2
        assert len(cache._entries) == 1
        assert cache._entries[0].response == "c"

    @given(
        expired=st.integers(min_value=0, max_value=20),
        fresh=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=100)
    def test_invalidate_count_matches_expired_entries(
        self, expired: int, fresh: int
    ) -> None:
        """invalidate_expired count must equal the number of expired entries."""
        cache = _cache()
        now = time.monotonic()
        cache._entries = (
            [_CacheEntry(embedding=[], response="x", expiry_ts=now - 1)] * expired
            + [_CacheEntry(embedding=[], response="y", expiry_ts=now + 3600)] * fresh
        )
        removed = cache.invalidate_expired()
        assert removed == expired
        assert len(cache._entries) == fresh


# ---------------------------------------------------------------------------
# SC-3: Exception isolation
# Property: if the GPTCache index raises any exception, lookup() must return
# hit=False and must not propagate the exception.
# Requirements: 1.9
# ---------------------------------------------------------------------------

class TestSC3ExceptionIsolation:
    """Property SC-3: exceptions inside lookup must be swallowed, returning hit=False."""

    def test_embed_exception_returns_miss(self) -> None:
        """If _embed_sync raises, lookup() must return hit=False."""
        cache = _cache()
        # Give it a live entry so the lookup path reaches embedding
        future_ts = time.monotonic() + 3600
        cache._entries = [
            _CacheEntry(embedding=[1.0], response="boom", expiry_ts=future_ts)
        ]
        with patch.object(cache, "_embed_sync", side_effect=RuntimeError("embed failure")):
            result = cache._lookup_sync("any prompt")
        assert result.hit is False

    @given(
        exc_type=st.sampled_from([RuntimeError, ValueError, MemoryError, OSError])
    )
    @settings(max_examples=50)
    def test_any_exception_type_returns_miss(self, exc_type: type) -> None:
        """Any exception type from _embed_sync must result in hit=False."""
        cache = _cache()
        future_ts = time.monotonic() + 3600
        cache._entries = [
            _CacheEntry(embedding=[1.0], response="x", expiry_ts=future_ts)
        ]
        with patch.object(cache, "_embed_sync", side_effect=exc_type("test error")):
            result = cache._lookup_sync("query")
        assert result.hit is False

    @pytest.mark.asyncio
    async def test_async_lookup_exception_returns_miss(self) -> None:
        """async lookup() must also absorb exceptions and return hit=False."""
        cache = _cache()
        future_ts = time.monotonic() + 3600
        cache._entries = [
            _CacheEntry(embedding=[1.0], response="async", expiry_ts=future_ts)
        ]
        with patch.object(cache, "_embed_sync", side_effect=RuntimeError("async fail")):
            result = await cache.lookup("any prompt", cache_ttl_seconds=300)
        assert result.hit is False

    def test_cosine_similarity_exception_skips_entry(self) -> None:
        """If _cosine_similarity raises for an entry, it must be skipped gracefully."""
        cache = _cache(threshold=0.0)
        future_ts = time.monotonic() + 3600
        vec = [1.0, 0.0]
        cache._entries = [_CacheEntry(embedding=vec, response="x", expiry_ts=future_ts)]

        with patch.object(cache, "_embed_sync", return_value=vec):
            with patch.object(SemanticCache, "_cosine_similarity", side_effect=ZeroDivisionError):
                result = cache._lookup_sync("query")
        # Entry was skipped — best_entry is None, so result is miss
        assert result.hit is False

    def test_store_exception_does_not_propagate(self) -> None:
        """_store_sync must not propagate exceptions from _embed_sync."""
        cache = _cache()
        with patch.object(cache, "_embed_sync", side_effect=RuntimeError("store fail")):
            # Must not raise
            cache._store_sync("prompt", "response", ttl_seconds=300)
        assert len(cache._entries) == 0
