"""Property-based tests for GroundednessAuditor.

Tasks: 5.2, 5.3
Properties:
  NLI-1 — Score range: AuditResult.groundedness_score must always be in [0.0, 1.0]
  NLI-4 — Scorer exception isolation: if cross-encoder raises, audit() must return
           a valid AuditResult with nli_label=None and must not propagate the exception.

Requirements: 2.3, 2.6
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from app.groundedness.auditor import audit
from app.groundedness.nli_scorer import NLIScorer
from app.models import AuditResult


# ---------------------------------------------------------------------------
# Fake vector store that always returns controllable docs
# ---------------------------------------------------------------------------

class _FakeDoc:
    def __init__(self, text: str = "source document text"):
        self.text = text


class _FakeVectorStore:
    def __init__(self, docs: list | None = None) -> None:
        self._docs = docs or [_FakeDoc()]

    async def similarity_search(self, embedding, top_k: int = 5) -> list:
        return self._docs[:top_k]


class _BrokenVectorStore:
    async def similarity_search(self, embedding, top_k: int = 5) -> list:
        raise RuntimeError("vector store unavailable")


# ---------------------------------------------------------------------------
# NLI-1: Score range
# Property: AuditResult.groundedness_score must always be in [0.0, 1.0]
# Requirements: 2.3
# ---------------------------------------------------------------------------

class TestNLI1ScoreRange:
    """Property NLI-1: groundedness_score must always be in [0.0, 1.0]."""

    @pytest.mark.asyncio
    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    async def test_score_in_range_no_nli(self, response_text: str) -> None:
        """Score must be in [0.0, 1.0] for any response text, without NLI scorer."""
        store = _FakeVectorStore()
        result = await audit(
            response=response_text,
            request_id="prop-nli1",
            vector_store=store,
            nli_scorer=None,
        )
        assert isinstance(result, AuditResult)
        assert 0.0 <= result.groundedness_score <= 1.0, (
            f"Score {result.groundedness_score} out of range for response={response_text[:50]!r}"
        )

    @pytest.mark.asyncio
    async def test_score_in_range_with_broken_store(self) -> None:
        """Even when the vector store is broken, score must stay in [0.0, 1.0]."""
        store = _BrokenVectorStore()
        result = await audit(
            response="test response",
            request_id="prop-nli1-broken",
            vector_store=store,
        )
        assert 0.0 <= result.groundedness_score <= 1.0
        assert result.is_unverified is True

    @pytest.mark.asyncio
    async def test_score_in_range_with_empty_response(self) -> None:
        """Empty response must still produce a score in range."""
        store = _FakeVectorStore()
        result = await audit(
            response="",
            request_id="prop-nli1-empty",
            vector_store=store,
        )
        assert 0.0 <= result.groundedness_score <= 1.0


# ---------------------------------------------------------------------------
# NLI-4: Scorer exception isolation
# Property: if the cross-encoder raises any exception, audit() must return
# a valid AuditResult with nli_label=None and must not propagate the exception.
# Requirements: 2.6
# ---------------------------------------------------------------------------

class TestNLI4ScorerExceptionIsolation:
    """Property NLI-4: NLI scorer exceptions must be absorbed; nli_label=None."""

    def _make_exploding_scorer(self, exc_type: type = RuntimeError) -> NLIScorer:
        """Build an NLIScorer whose score_pairs always raises."""
        scorer = MagicMock(spec=NLIScorer)
        scorer.score_pairs = AsyncMock(side_effect=exc_type("scorer exploded"))
        return scorer

    @pytest.mark.asyncio
    async def test_runtime_error_returns_none_nli_label(self) -> None:
        scorer = self._make_exploding_scorer(RuntimeError)
        store = _FakeVectorStore()
        result = await audit(
            response="some response",
            request_id="prop-nli4-rt",
            vector_store=store,
            nli_scorer=scorer,
        )
        assert isinstance(result, AuditResult)
        assert result.nli_label is None
        assert 0.0 <= result.groundedness_score <= 1.0

    @pytest.mark.asyncio
    @given(st.sampled_from([RuntimeError, ValueError, MemoryError, OSError, TypeError]))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    async def test_any_exception_type_absorbed(self, exc_type: type) -> None:
        scorer = self._make_exploding_scorer(exc_type)
        store = _FakeVectorStore()
        # Must not raise
        result = await audit(
            response="response text",
            request_id=f"prop-nli4-{exc_type.__name__}",
            vector_store=store,
            nli_scorer=scorer,
        )
        assert result.nli_label is None

    @pytest.mark.asyncio
    async def test_technique_falls_back_on_scorer_error(self) -> None:
        """When NLI errors out, technique must remain 'embedding_similarity'."""
        scorer = self._make_exploding_scorer()
        store = _FakeVectorStore()
        result = await audit(
            response="test",
            request_id="prop-nli4-technique",
            vector_store=store,
            nli_scorer=scorer,
        )
        assert result.technique == "embedding_similarity"

    @pytest.mark.asyncio
    async def test_valid_scorer_sets_technique_correctly(self) -> None:
        """When NLI runs successfully, technique must be 'nli_embedding_similarity'."""
        scorer = MagicMock(spec=NLIScorer)
        scorer.score_pairs = AsyncMock(return_value=[("ENTAILMENT", 0.95)])
        store = _FakeVectorStore(docs=[_FakeDoc("doc text")])
        result = await audit(
            response="test response",
            request_id="prop-nli4-success",
            vector_store=store,
            nli_scorer=scorer,
        )
        assert result.technique == "nli_embedding_similarity"
        assert result.nli_label is not None
