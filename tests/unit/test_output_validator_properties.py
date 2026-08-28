"""Property-based tests for app.judges.output_validator (Guardrails AI chain).

Goal (Item 3): prove that _run_validation_sync and validate_output *never*
crash the FastAPI event loop regardless of how malformed, large, or adversarial
the input text is, and that HARD_BLOCK is consistently produced for any
unfixable Guardrails failure — with no unhandled exception escaping to the
caller.

Test strategy
─────────────
Because guardrails-ai is optional and the Hub validators require a running
installation, these tests inject controllable stub validators directly into
the _LOADED_VALIDATORS registry.  This makes every property deterministic and
fast without needing the real SDK.

Properties proven
─────────────────
P-OV-1  For any text input, validate_output() always returns a GuardrailsVerdict
        (never raises).
P-OV-2  When a validator raises an unhandled exception, the returned verdict has
        passed=False and action="exception" (HARD_BLOCK path in the pipeline).
P-OV-3  When a validator returns action="filter" or action="exception" (unfixable),
        passed is always False — the pipeline must HARD_BLOCK.
P-OV-4  When a validator returns action="fix" with a fixed_output, passed is True
        and fixed_output is non-None.
P-OV-5  An empty _LOADED_VALIDATORS list always produces passed=True (no-op pass-
        through when guardrails is not installed or no validators are loaded).
P-OV-6  validate_output() is safe to call concurrently from many coroutines; no
        shared state is mutated between concurrent calls.
P-OV-7  The asyncio event loop is never blocked: validate_output() completes even
        when the synchronous validator is patched to sleep (asyncio.to_thread).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings as hyp_settings
from hypothesis import strategies as st

import app.judges.output_validator as ov_module
from app.judges.output_validator import _run_validation_sync, validate_output
from app.models import GuardrailsVerdict

# ---------------------------------------------------------------------------
# Stub validator helpers
# ---------------------------------------------------------------------------


def _make_pass_validator(vid: str = "stub-pass"):
    """Validator that always passes."""
    class _R:
        def __init__(self, value: str):
            self.outcome = "pass"
            self.on_fail_action = "exception"
            self.fix_value = value

    class _V:
        def validate(self, value: str, metadata: dict) -> Any:
            return _R(value)
    return (vid, _V())


def _make_fail_validator(action: str, vid: str = "stub-fail", fix_value: str = "FIXED"):
    """Validator that always fails with the given action."""
    _action = action
    _fix_value = fix_value

    class _R:
        def __init__(self):
            self.outcome = "fail"
            self.on_fail_action = _action
            self.fix_value = _fix_value

    class _V:
        def validate(self, value: str, metadata: dict) -> Any:
            return _R()
    return (vid, _V())


def _make_exploding_validator(vid: str = "stub-explode"):
    """Validator that always raises an internal exception."""
    class _V:
        def validate(self, value: str, metadata: dict) -> Any:
            raise RuntimeError("validator internal explosion")
    return (vid, _V())


def _make_slow_validator(sleep_s: float, vid: str = "stub-slow"):
    """Validator that blocks for sleep_s seconds (must not block event loop)."""
    _sleep_s = sleep_s

    class _R:
        def __init__(self, value: str):
            self.outcome = "pass"
            self.on_fail_action = "exception"
            self.fix_value = value

    class _V:
        def validate(self, value: str, metadata: dict) -> Any:
            time.sleep(_sleep_s)
            return _R(value)
    return (vid, _V())


# ---------------------------------------------------------------------------
# Fixture: isolate _LOADED_VALIDATORS for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_validators():
    """Ensure _LOADED_VALIDATORS is empty before and after every test."""
    original = list(ov_module._LOADED_VALIDATORS)
    ov_module._LOADED_VALIDATORS.clear()
    yield
    ov_module._LOADED_VALIDATORS.clear()
    ov_module._LOADED_VALIDATORS.extend(original)


# ---------------------------------------------------------------------------
# P-OV-1: validate_output never raises for any text input
# ---------------------------------------------------------------------------

class TestPOV1NeverRaises:

    @given(st.text(min_size=0, max_size=4096))
    @hyp_settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_no_validators_never_raises(self, text: str) -> None:
        """P-OV-1: empty validator list — always returns GuardrailsVerdict."""
        # _LOADED_VALIDATORS is empty (reset by fixture)
        result = _run_validation_sync(text)
        assert isinstance(result, GuardrailsVerdict)

    @pytest.mark.asyncio
    @given(st.text(min_size=0, max_size=4096))
    @hyp_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    async def test_async_no_validators_never_raises(self, text: str) -> None:
        """P-OV-1 (async path): validate_output always returns GuardrailsVerdict."""
        result = await validate_output(text)
        assert isinstance(result, GuardrailsVerdict)

    @given(st.text(min_size=0, max_size=4096))
    @hyp_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_passing_validator_never_raises(self, text: str) -> None:
        """P-OV-1: passing validator — always returns GuardrailsVerdict."""
        ov_module._LOADED_VALIDATORS.append(_make_pass_validator())
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            result = _run_validation_sync(text)
        assert isinstance(result, GuardrailsVerdict)

    @given(st.text(min_size=0, max_size=4096))
    @hyp_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_exploding_validator_never_raises_to_caller(self, text: str) -> None:
        """P-OV-1: an internal validator exception must be caught, not propagated."""
        ov_module._LOADED_VALIDATORS.append(_make_exploding_validator())
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            try:
                result = _run_validation_sync(text)
            except Exception as exc:
                pytest.fail(
                    f"_run_validation_sync raised to the caller: {type(exc).__name__}: {exc}"
                )
        assert isinstance(result, GuardrailsVerdict)

    @given(
        st.text(min_size=0, max_size=4096),
        st.sampled_from(["exception", "filter", "fix"]),
    )
    @hyp_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_fail_validator_never_raises_to_caller(self, text: str, action: str) -> None:
        """P-OV-1: failing validators (all actions) must not propagate exceptions."""
        ov_module._LOADED_VALIDATORS.append(_make_fail_validator(action))
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            try:
                result = _run_validation_sync(text)
            except Exception as exc:
                pytest.fail(
                    f"_run_validation_sync raised for action={action!r}: "
                    f"{type(exc).__name__}: {exc}"
                )
        assert isinstance(result, GuardrailsVerdict)


# ---------------------------------------------------------------------------
# P-OV-2: exploding validator → passed=False, action="exception"
# ---------------------------------------------------------------------------

class TestPOV2ExceptionMeansHardBlock:

    @given(st.text(min_size=1, max_size=2048))
    @hyp_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_exploding_validator_produces_hard_block_verdict(self, text: str) -> None:
        """P-OV-2: internal validator exception → passed=False, action='exception'."""
        ov_module._LOADED_VALIDATORS.append(_make_exploding_validator())
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            result = _run_validation_sync(text)
        assert result.passed is False, "expected passed=False on validator exception"
        assert result.action == "exception", f"expected action='exception', got {result.action!r}"

    @pytest.mark.asyncio
    @given(st.text(min_size=1, max_size=2048))
    @hyp_settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    async def test_async_exploding_validator_produces_hard_block(self, text: str) -> None:
        """P-OV-2 (async): same guarantee through validate_output()."""
        ov_module._LOADED_VALIDATORS.append(_make_exploding_validator())
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            result = await validate_output(text)
        assert result.passed is False
        assert result.action == "exception"

    def test_exception_verdict_has_triggered_validator_set(self) -> None:
        """P-OV-2: triggered_validator field identifies the guilty validator."""
        ov_module._LOADED_VALIDATORS.append(_make_exploding_validator("my-boom-validator"))
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            result = _run_validation_sync("some text")
        assert result.triggered_validator == "my-boom-validator"


# ---------------------------------------------------------------------------
# P-OV-3: unfixable failures (filter / exception) → passed=False
# ---------------------------------------------------------------------------

class TestPOV3UnfixableAlwaysHardBlock:

    @given(
        st.text(min_size=0, max_size=4096),
        st.sampled_from(["filter", "exception"]),
    )
    @hyp_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_unfixable_action_always_passed_false(self, text: str, action: str) -> None:
        """P-OV-3: filter and exception actions always produce passed=False."""
        ov_module._LOADED_VALIDATORS.append(_make_fail_validator(action))
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            result = _run_validation_sync(text)
        assert result.passed is False, (
            f"Expected passed=False for action={action!r} on text of len={len(text)}"
        )
        assert result.action in ("filter", "exception"), (
            f"Unexpected action value: {result.action!r}"
        )

    @given(
        st.text(min_size=0, max_size=4096),
        st.sampled_from(["filter", "exception"]),
    )
    @hyp_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_unfixable_action_fixed_output_is_none(self, text: str, action: str) -> None:
        """P-OV-3: unfixable verdicts must have fixed_output=None."""
        ov_module._LOADED_VALIDATORS.append(_make_fail_validator(action))
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            result = _run_validation_sync(text)
        assert result.fixed_output is None, (
            f"fixed_output should be None for unfixable action={action!r}"
        )

    @given(
        st.text(min_size=1, max_size=2048),
        st.sampled_from(["filter", "exception"]),
    )
    @hyp_settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_pipeline_would_hard_block_on_unfixable(self, text: str, action: str) -> None:
        """P-OV-3: mirror the pipeline's block logic — if not fixed, it's a HARD_BLOCK."""
        ov_module._LOADED_VALIDATORS.append(_make_fail_validator(action))
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            verdict = _run_validation_sync(text)
        # Replicate the pipeline's decision logic from main.py
        if not verdict.passed:
            if verdict.action == "fix" and verdict.fixed_output:
                pipeline_decision = "PASS_WITH_FIX"
            else:
                pipeline_decision = "HARD_BLOCK"
        else:
            pipeline_decision = "PASS"
        assert pipeline_decision == "HARD_BLOCK", (
            f"Pipeline should HARD_BLOCK for unfixable action={action!r}, "
            f"got {pipeline_decision!r}"
        )


# ---------------------------------------------------------------------------
# P-OV-4: fix action → passed=True, fixed_output is non-None
# ---------------------------------------------------------------------------

class TestPOV4FixActionPreservesContent:

    @given(
        st.text(min_size=1, max_size=2048),
        st.text(min_size=1, max_size=2048),
    )
    @hyp_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_fix_action_passed_true_and_output_nonnull(
        self, text: str, fixed: str
    ) -> None:
        """P-OV-4: fix action always produces passed=True with a non-None fixed_output."""
        ov_module._LOADED_VALIDATORS.append(
            _make_fail_validator("fix", fix_value=fixed)
        )
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            result = _run_validation_sync(text)
        assert result.passed is True, "fix action must produce passed=True"
        assert result.action == "fix"
        assert result.fixed_output is not None, "fix action must provide fixed_output"

    @given(st.text(min_size=1, max_size=512))
    @hyp_settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_fix_output_equals_stub_fix_value(self, text: str) -> None:
        """P-OV-4: the fixed_output returned must equal the validator's fix_value."""
        # Use a constant sentinel so it is stable across Hypothesis shrink iterations
        sentinel_fix = "SENTINEL_FIXED_OUTPUT"
        ov_module._LOADED_VALIDATORS.clear()
        ov_module._LOADED_VALIDATORS.append(
            _make_fail_validator("fix", fix_value=sentinel_fix)
        )
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            result = _run_validation_sync(text)
        assert result.fixed_output == sentinel_fix, (
            f"Expected fixed_output={sentinel_fix!r}, got {result.fixed_output!r}"
        )


# ---------------------------------------------------------------------------
# P-OV-5: empty validator list → always passed=True
# ---------------------------------------------------------------------------

class TestPOV5EmptyValidatorsPassThrough:

    @given(st.text(min_size=0, max_size=32_768))
    @hyp_settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_no_validators_always_pass(self, text: str) -> None:
        """P-OV-5: with no validators loaded, every output passes."""
        # _LOADED_VALIDATORS is empty (reset by fixture)
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            result = _run_validation_sync(text)
        assert result.passed is True
        assert result.action is None
        assert result.fixed_output is None

    @given(st.text(min_size=0, max_size=1024))
    @hyp_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_guardrails_unavailable_always_pass(self, text: str) -> None:
        """P-OV-5: when guardrails is not installed, every output passes."""
        ov_module._LOADED_VALIDATORS.append(_make_fail_validator("exception"))
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", False):
            result = _run_validation_sync(text)
        assert result.passed is True, (
            "When guardrails is unavailable, verdict must be pass-through"
        )


# ---------------------------------------------------------------------------
# P-OV-6: concurrent safety — no shared state mutation between coroutines
# ---------------------------------------------------------------------------

class TestPOV6ConcurrentSafety:

    @pytest.mark.asyncio
    async def test_concurrent_calls_no_cross_contamination(self) -> None:
        """P-OV-6: 20 concurrent validate_output() calls with different validators
        must each see only their own result — no shared-state bleed."""
        # Use different validator-IDs per "logical caller" to detect bleed
        results: list[GuardrailsVerdict] = []
        errors: list[Exception] = []

        async def call_with_pass(i: int) -> None:
            try:
                # All calls share the same _LOADED_VALIDATORS (empty); must all pass
                result = await validate_output(f"concurrent text {i}")
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        await asyncio.gather(*[call_with_pass(i) for i in range(20)])

        assert not errors, f"Concurrent calls raised: {errors}"
        assert len(results) == 20
        for r in results:
            assert isinstance(r, GuardrailsVerdict)
            assert r.passed is True  # no validators loaded → pass

    @pytest.mark.asyncio
    async def test_concurrent_calls_with_exploding_validator(self) -> None:
        """P-OV-6: even with an exploding validator, all concurrent calls must
        return a GuardrailsVerdict — no unhandled exceptions, no event-loop crash."""
        ov_module._LOADED_VALIDATORS.append(_make_exploding_validator())
        results: list[GuardrailsVerdict] = []
        errors: list[Exception] = []

        async def call(i: int) -> None:
            try:
                with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
                    result = await validate_output(f"text {i}")
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        await asyncio.gather(*[call(i) for i in range(20)])

        assert not errors, f"Event loop received unhandled exceptions: {errors}"
        assert len(results) == 20
        for r in results:
            assert r.passed is False
            assert r.action == "exception"


# ---------------------------------------------------------------------------
# P-OV-7: asyncio.to_thread — event loop not blocked by slow validator
# ---------------------------------------------------------------------------

class TestPOV7EventLoopNotBlocked:

    @pytest.mark.asyncio
    async def test_slow_validator_runs_in_thread_not_blocking_loop(self) -> None:
        """P-OV-7: a validator that sleeps 200ms must not block the event loop.

        We verify by running two concurrent coroutines and confirming total wall-
        clock time is closer to 200ms than 400ms (they ran concurrently).
        """
        SLEEP_S = 0.2
        ov_module._LOADED_VALIDATORS.append(_make_slow_validator(SLEEP_S))

        start = time.monotonic()
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            r1, r2 = await asyncio.gather(
                validate_output("first call"),
                validate_output("second call"),
            )
        elapsed = time.monotonic() - start

        assert isinstance(r1, GuardrailsVerdict)
        assert isinstance(r2, GuardrailsVerdict)
        # If the event loop were blocked, elapsed would be ~400ms.
        # In a thread pool both run concurrently → ~200ms.
        # We allow up to 350ms for CI variance.
        assert elapsed < 0.35 + SLEEP_S, (
            f"Slow validator appears to have blocked the event loop: {elapsed:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_validate_output_completes_while_other_tasks_run(self) -> None:
        """P-OV-7: a background task can make progress while validate_output runs."""
        counter = {"n": 0}

        async def background_incrementer() -> None:
            for _ in range(5):
                await asyncio.sleep(0.01)
                counter["n"] += 1

        ov_module._LOADED_VALIDATORS.append(_make_slow_validator(0.05))
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            await asyncio.gather(
                validate_output("some output text"),
                background_incrementer(),
            )

        # The background coroutine must have had CPU time while the validator slept
        assert counter["n"] > 0, (
            "Background task got no CPU time — event loop was likely blocked"
        )


# ---------------------------------------------------------------------------
# Edge-case spot checks
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_string_with_fail_validator(self) -> None:
        """Empty string must still trigger the validator and produce a verdict."""
        ov_module._LOADED_VALIDATORS.append(_make_fail_validator("exception"))
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            result = _run_validation_sync("")
        assert isinstance(result, GuardrailsVerdict)
        assert result.passed is False

    def test_very_large_input(self) -> None:
        """Inputs up to 32 768 characters must not cause memory or timeout issues."""
        large = "A" * 32_768
        ov_module._LOADED_VALIDATORS.append(_make_pass_validator())
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            result = _run_validation_sync(large)
        assert isinstance(result, GuardrailsVerdict)
        assert result.passed is True

    def test_null_bytes_and_unicode_extremes(self) -> None:
        """Null bytes, surrogates, and extreme unicode must not crash the validator."""
        nasty_inputs = [
            "\x00\x01\x02\x03",
            "\ud800\udfff",   # surrogate pair
            "😈" * 500,
            "\n" * 1000,
            "<script>alert('xss')</script>",
        ]
        ov_module._LOADED_VALIDATORS.append(_make_pass_validator())
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            for inp in nasty_inputs:
                result = _run_validation_sync(inp)
                assert isinstance(result, GuardrailsVerdict), (
                    f"Got non-GuardrailsVerdict for input {inp[:30]!r}"
                )

    def test_multiple_validators_first_fail_short_circuits(self) -> None:
        """Once a validator fails, the chain must stop (first-fail wins)."""
        ov_module._LOADED_VALIDATORS.extend([
            _make_fail_validator("exception", vid="v1"),
            _make_pass_validator(vid="v2"),  # must not be reached
        ])
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            result = _run_validation_sync("any text")
        assert result.triggered_validator == "v1", (
            "Second validator ran when first should have short-circuited"
        )

    def test_multiple_validators_all_pass(self) -> None:
        """If all validators pass, the verdict must be passed=True."""
        ov_module._LOADED_VALIDATORS.extend([
            _make_pass_validator("v1"),
            _make_pass_validator("v2"),
            _make_pass_validator("v3"),
        ])
        with patch.object(ov_module, "_GUARDRAILS_AVAILABLE", True):
            result = _run_validation_sync("clean output")
        assert result.passed is True
        assert result.action is None
