"""Property-based tests for Phase 3 data model fields.

Tasks: 1.2
Properties:
  SC-4  — TelemetryRecord.cache_hit defaults to False
  NLI-1 (partial) — TelemetryRecord.nli_label and AuditResult.nli_label default to None
  Validation: cache_ttl_seconds < 1 raises ValidationError
  Validation: cache_similarity_threshold outside [0.0, 1.0] raises ValidationError

Requirements: 1.1, 1.2, 2.1, 2.2, 6.6
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.models import AuditResult, TelemetryRecord, UseCaseProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_telemetry(**overrides) -> dict:
    base = dict(
        request_id="req-001",
        timestamp=datetime.now(tz=timezone.utc),
        use_case_profile="test",
        final_triage_state="PASS_AND_DELIVER",
        latency_ms=42,
    )
    base.update(overrides)
    return base


def _valid_profile(**overrides) -> dict:
    base = dict(
        name="test",
        latency_budget_ms=10_000,
        complexity_threshold=0.7,
        token_compression_threshold=512,
        groundedness_pass_threshold=0.85,
        inspection_timeout_ms=3_000,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# SC-4: cache_hit telemetry default
# ---------------------------------------------------------------------------

class TestSC4CacheHitDefault:
    """Property SC-4: TelemetryRecord.cache_hit must default to False."""

    def test_cache_hit_defaults_to_false(self) -> None:
        rec = TelemetryRecord(**_valid_telemetry())
        assert rec.cache_hit is False

    @given(st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_cache_hit_false_for_any_profile_name(self, profile_name: str) -> None:
        rec = TelemetryRecord(**_valid_telemetry(use_case_profile=profile_name))
        assert rec.cache_hit is False

    def test_cache_hit_can_be_set_true(self) -> None:
        rec = TelemetryRecord(**_valid_telemetry(cache_hit=True))
        assert rec.cache_hit is True


# ---------------------------------------------------------------------------
# NLI-1 (partial): nli_label defaults to None
# ---------------------------------------------------------------------------

class TestNLI1NliLabelDefault:
    """Property NLI-1 (partial): nli_label must default to None on TelemetryRecord and AuditResult."""

    def test_telemetry_nli_label_defaults_to_none(self) -> None:
        rec = TelemetryRecord(**_valid_telemetry())
        assert rec.nli_label is None

    def test_audit_result_nli_label_defaults_to_none(self) -> None:
        result = AuditResult(
            groundedness_score=0.9,
            technique="embedding_similarity",
            is_unverified=False,
        )
        assert result.nli_label is None

    @given(st.sampled_from(["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]))
    @settings(max_examples=30)
    def test_nli_label_accepts_valid_literals(self, label: str) -> None:
        rec = TelemetryRecord(**_valid_telemetry(nli_label=label))
        assert rec.nli_label == label

        result = AuditResult(
            groundedness_score=0.5,
            technique="nli_embedding_similarity",
            is_unverified=False,
            nli_label=label,  # type: ignore[arg-type]
        )
        assert result.nli_label == label


# ---------------------------------------------------------------------------
# Validation: cache_ttl_seconds < 1 raises ValidationError
# ---------------------------------------------------------------------------

class TestCacheTTLValidation:
    """cache_ttl_seconds must be >= 1."""

    @given(st.integers(max_value=0))
    @settings(max_examples=100)
    def test_cache_ttl_below_1_raises(self, v: int) -> None:
        with pytest.raises(ValidationError) as exc_info:
            UseCaseProfile.model_validate(_valid_profile(cache_ttl_seconds=v))
        fields = [e["loc"][-1] for e in exc_info.value.errors()]
        assert "cache_ttl_seconds" in fields

    @given(st.integers(min_value=1, max_value=86_400))
    @settings(max_examples=100)
    def test_cache_ttl_valid_range_accepted(self, v: int) -> None:
        profile = UseCaseProfile.model_validate(_valid_profile(cache_ttl_seconds=v))
        assert profile.cache_ttl_seconds == v

    def test_cache_ttl_default_is_300(self) -> None:
        profile = UseCaseProfile.model_validate(_valid_profile())
        assert profile.cache_ttl_seconds == 300


# ---------------------------------------------------------------------------
# Validation: cache_similarity_threshold outside [0.0, 1.0] raises ValidationError
# ---------------------------------------------------------------------------

class TestCacheSimilarityThresholdValidation:
    """cache_similarity_threshold must be in [0.0, 1.0]."""

    @given(
        st.one_of(
            st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.001, max_value=1e6, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100)
    def test_out_of_range_raises(self, v: float) -> None:
        with pytest.raises(ValidationError) as exc_info:
            UseCaseProfile.model_validate(_valid_profile(cache_similarity_threshold=v))
        fields = [e["loc"][-1] for e in exc_info.value.errors()]
        assert "cache_similarity_threshold" in fields

    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=100)
    def test_valid_range_accepted(self, v: float) -> None:
        profile = UseCaseProfile.model_validate(_valid_profile(cache_similarity_threshold=v))
        assert 0.0 <= profile.cache_similarity_threshold <= 1.0

    def test_default_is_0_92(self) -> None:
        profile = UseCaseProfile.model_validate(_valid_profile())
        assert profile.cache_similarity_threshold == 0.92
