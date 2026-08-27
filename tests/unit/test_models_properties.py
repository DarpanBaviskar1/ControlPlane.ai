"""Property-based tests for data model validation.

# Feature: controlplane-ai-gateway, Property 22: Policy Validation Rejects Invalid Fields
Validates: Requirement 7.5
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.models import UseCaseProfile

# ---------------------------------------------------------------------------
# Helpers — valid baseline dict
# ---------------------------------------------------------------------------

VALID_PROFILE_DICT = {
    "name": "test_profile",
    "latency_budget_ms": 10_000,
    "complexity_threshold": 0.7,
    "token_compression_threshold": 512,
    "groundedness_pass_threshold": 0.85,
    "inspection_timeout_ms": 3_000,
    "pii_masking_enabled": True,
    "human_escalation_enabled": True,
}


def _profile_with(**overrides: object) -> dict:
    d = dict(VALID_PROFILE_DICT)
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Property 22: Policy Validation Rejects Invalid Fields
# ---------------------------------------------------------------------------


class TestUseCaseProfileValidation:
    """Any UseCaseProfile with an out-of-range field must raise ValidationError."""

    # latency_budget_ms: ge=1, le=300_000
    @given(st.one_of(st.integers(max_value=0), st.integers(min_value=300_001)))
    @settings(max_examples=100)
    def test_invalid_latency_budget_ms(self, v: int) -> None:
        with pytest.raises(ValidationError) as exc_info:
            UseCaseProfile.model_validate(_profile_with(latency_budget_ms=v))
        errors = exc_info.value.errors()
        fields = [e["loc"][-1] for e in errors]
        assert "latency_budget_ms" in fields, f"Expected latency_budget_ms in errors, got {fields}"

    # complexity_threshold: ge=0.0, le=1.0
    @given(
        st.one_of(
            st.floats(max_value=-0.001, allow_nan=False),
            st.floats(min_value=1.001, max_value=1e10, allow_nan=False),
        )
    )
    @settings(max_examples=100)
    def test_invalid_complexity_threshold(self, v: float) -> None:
        with pytest.raises(ValidationError) as exc_info:
            UseCaseProfile.model_validate(_profile_with(complexity_threshold=v))
        errors = exc_info.value.errors()
        fields = [e["loc"][-1] for e in errors]
        assert "complexity_threshold" in fields

    # token_compression_threshold: ge=1
    @given(st.integers(max_value=0))
    @settings(max_examples=100)
    def test_invalid_token_compression_threshold(self, v: int) -> None:
        with pytest.raises(ValidationError) as exc_info:
            UseCaseProfile.model_validate(_profile_with(token_compression_threshold=v))
        errors = exc_info.value.errors()
        fields = [e["loc"][-1] for e in errors]
        assert "token_compression_threshold" in fields

    # groundedness_pass_threshold: ge=0.0, le=1.0
    @given(
        st.one_of(
            st.floats(max_value=-0.001, allow_nan=False),
            st.floats(min_value=1.001, max_value=1e10, allow_nan=False),
        )
    )
    @settings(max_examples=100)
    def test_invalid_groundedness_pass_threshold(self, v: float) -> None:
        with pytest.raises(ValidationError) as exc_info:
            UseCaseProfile.model_validate(_profile_with(groundedness_pass_threshold=v))
        errors = exc_info.value.errors()
        fields = [e["loc"][-1] for e in errors]
        assert "groundedness_pass_threshold" in fields

    # inspection_timeout_ms: ge=1, le=60_000
    @given(st.one_of(st.integers(max_value=0), st.integers(min_value=60_001)))
    @settings(max_examples=100)
    def test_invalid_inspection_timeout_ms(self, v: int) -> None:
        with pytest.raises(ValidationError) as exc_info:
            UseCaseProfile.model_validate(_profile_with(inspection_timeout_ms=v))
        errors = exc_info.value.errors()
        fields = [e["loc"][-1] for e in errors]
        assert "inspection_timeout_ms" in fields

    # name: min_length=1
    @given(st.just(""))
    @settings(max_examples=10)
    def test_invalid_empty_name(self, v: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            UseCaseProfile.model_validate(_profile_with(name=v))
        errors = exc_info.value.errors()
        fields = [e["loc"][-1] for e in errors]
        assert "name" in fields

    # Valid profiles must not raise
    @given(
        st.fixed_dictionaries(
            {
                "name": st.text(min_size=1, max_size=50),
                "latency_budget_ms": st.integers(min_value=1, max_value=300_000),
                "complexity_threshold": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                "token_compression_threshold": st.integers(min_value=1, max_value=100_000),
                "groundedness_pass_threshold": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                "inspection_timeout_ms": st.integers(min_value=1, max_value=60_000),
                "pii_masking_enabled": st.booleans(),
                "human_escalation_enabled": st.booleans(),
            }
        )
    )
    @settings(max_examples=100)
    def test_valid_profile_does_not_raise(self, d: dict) -> None:
        profile = UseCaseProfile.model_validate(d)
        assert profile.name == d["name"]
        assert 1 <= profile.latency_budget_ms <= 300_000
        assert 0.0 <= profile.complexity_threshold <= 1.0
        assert profile.token_compression_threshold >= 1
        assert 0.0 <= profile.groundedness_pass_threshold <= 1.0
        assert 1 <= profile.inspection_timeout_ms <= 60_000


# ---------------------------------------------------------------------------
# Unit spot checks (boundary values)
# ---------------------------------------------------------------------------


class TestUseCaseProfileBoundaries:
    def test_latency_budget_min_valid(self) -> None:
        p = UseCaseProfile.model_validate(_profile_with(latency_budget_ms=1))
        assert p.latency_budget_ms == 1

    def test_latency_budget_max_valid(self) -> None:
        p = UseCaseProfile.model_validate(_profile_with(latency_budget_ms=300_000))
        assert p.latency_budget_ms == 300_000

    def test_latency_budget_zero_invalid(self) -> None:
        with pytest.raises(ValidationError):
            UseCaseProfile.model_validate(_profile_with(latency_budget_ms=0))

    def test_complexity_threshold_zero_valid(self) -> None:
        p = UseCaseProfile.model_validate(_profile_with(complexity_threshold=0.0))
        assert p.complexity_threshold == 0.0

    def test_complexity_threshold_one_valid(self) -> None:
        p = UseCaseProfile.model_validate(_profile_with(complexity_threshold=1.0))
        assert p.complexity_threshold == 1.0

    def test_inspection_timeout_min_valid(self) -> None:
        p = UseCaseProfile.model_validate(_profile_with(inspection_timeout_ms=1))
        assert p.inspection_timeout_ms == 1

    def test_inspection_timeout_max_valid(self) -> None:
        p = UseCaseProfile.model_validate(_profile_with(inspection_timeout_ms=60_000))
        assert p.inspection_timeout_ms == 60_000

    def test_error_message_includes_field_name(self) -> None:
        """Validation error must identify the invalid field name."""
        with pytest.raises(ValidationError) as exc_info:
            UseCaseProfile.model_validate(_profile_with(latency_budget_ms=-1))
        assert "latency_budget_ms" in str(exc_info.value)
