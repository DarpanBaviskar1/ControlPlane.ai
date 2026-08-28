"""Property-based tests for GLiNERMasker + PIIMaskingEngine Phase 3 GLiNER integration.

Tasks: 7.2, 8.2, 8.3, 8.4
Properties:
  GL-1 — Placeholder format: every detected entity must produce a placeholder
          matching [CUSTOM_ENTITY_REDACTED_\\d+]
  GL-2 — Round-trip fidelity: mask() → unmask() must be byte-for-byte identical
          (after whitespace normalisation) for any prompt with a custom entity term
  GL-3 — Empty terms skips tier: mask() with custom_entity_terms=[] must produce
          output identical to Tier-1/Tier-2 alone, with no CUSTOM_ENTITY_REDACTED placeholders
  GL-4 — Exception isolation: if scan_sync() raises, mask() must return the
          Tier-1/Tier-2 result without propagating the exception

Requirements: 3.2, 3.4, 3.5, 3.6, 3.8
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from app.judges.gliner_masker import GLiNERMasker, _CUSTOM_PLACEHOLDER_RE
from app.judges.pii_masking import PIIMaskingEngine, _normalise


# ---------------------------------------------------------------------------
# Fake GLiNER model helpers
# ---------------------------------------------------------------------------

def _make_gliner_model(entities: list[dict]) -> MagicMock:
    """Return a mock gliner model that returns the provided entity list."""
    model = MagicMock()
    model.predict_entities = MagicMock(return_value=entities)
    return model


def _make_entity(text: str, start: int, end: int, label: str = "CORP_TERM") -> dict:
    return {"text": text, "start": start, "end": end, "label": label}


# ---------------------------------------------------------------------------
# GL-1: Placeholder format
# Task 7.2 — every entity span must produce a placeholder matching the regex
# Requirements: 3.5
# ---------------------------------------------------------------------------

class TestGL1PlaceholderFormat:
    """Property GL-1: every detected entity must produce a [CUSTOM_ENTITY_REDACTED_N] placeholder."""

    _masker = GLiNERMasker()

    @given(
        entities=st.lists(
            st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz "),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_every_entity_gets_correct_placeholder(self, entities: list[str]) -> None:
        """
        Property GL-1: every entity span detected by scan_sync() must produce a
        placeholder matching \\[CUSTOM_ENTITY_REDACTED_\\d+\\].
        """
        # Build a prompt that contains all entity strings
        prompt = " ".join(entities) + " extra text"

        # Construct entity dicts with correct start/end offsets
        entity_dicts = []
        pos = 0
        for ent_text in entities:
            idx = prompt.find(ent_text, pos)
            if idx == -1:
                continue
            entity_dicts.append(_make_entity(ent_text, idx, idx + len(ent_text)))
            pos = idx + len(ent_text)

        model = _make_gliner_model(entity_dicts)
        masked, placeholder_map = self._masker.scan_sync(prompt, entities, model)

        for placeholder in placeholder_map:
            assert _CUSTOM_PLACEHOLDER_RE.fullmatch(placeholder), (
                f"Placeholder {placeholder!r} does not match expected format"
            )

    def test_single_entity_placeholder_is_1(self) -> None:
        """Single detected entity must produce [CUSTOM_ENTITY_REDACTED_1]."""
        masker = GLiNERMasker()
        prompt = "Project Phoenix is confidential."
        model = _make_gliner_model([_make_entity("Project Phoenix", 0, 15)])
        masked, pmap = masker.scan_sync(prompt, ["Project Phoenix"], model)
        assert "[CUSTOM_ENTITY_REDACTED_1]" in masked
        assert pmap["[CUSTOM_ENTITY_REDACTED_1]"] == "Project Phoenix"

    def test_multiple_entities_numbered_sequentially(self) -> None:
        """N detected entities must produce placeholders _1 through _N."""
        masker = GLiNERMasker()
        prompt = "Alpha and Beta are projects."
        entities = [
            _make_entity("Alpha", 0, 5),
            _make_entity("Beta", 10, 14),
        ]
        model = _make_gliner_model(entities)
        masked, pmap = masker.scan_sync(prompt, ["Alpha", "Beta"], model)
        assert "[CUSTOM_ENTITY_REDACTED_1]" in pmap
        assert "[CUSTOM_ENTITY_REDACTED_2]" in pmap

    def test_no_entities_returns_original(self) -> None:
        """When no entities are detected, original prompt and empty map are returned."""
        masker = GLiNERMasker()
        model = _make_gliner_model([])
        prompt = "Nothing special here."
        masked, pmap = masker.scan_sync(prompt, ["MissingTerm"], model)
        assert masked == prompt
        assert pmap == {}


# ---------------------------------------------------------------------------
# GL-2: Round-trip fidelity
# Task 8.2 — mask() followed by unmask() must restore the original prompt
# Requirements: 3.6
# ---------------------------------------------------------------------------

class TestGL2RoundTripFidelity:
    """Property GL-2: mask() → unmask() round-trip must be byte-for-byte identical."""

    _engine = PIIMaskingEngine()

    @given(
        prefix=st.text(min_size=0, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz "),
        suffix=st.text(min_size=0, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz "),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_round_trip_with_gliner_entity(self, prefix: str, suffix: str) -> None:
        """
        Property GL-2: mask() + unmask() on any prompt containing a custom entity
        term must be byte-for-byte identical to the original after whitespace normalisation.
        """
        entity_text = "ProjectPhoenix"
        prompt = f"{prefix} {entity_text} {suffix}".strip()
        request_id = f"gl2-{hash(prompt) & 0xFFFFFF}"

        # Build a fake GLiNER model that detects "ProjectPhoenix"
        idx = prompt.find(entity_text)
        if idx == -1:
            return  # entity was absorbed by surrounding text — skip this example
        model = _make_gliner_model([_make_entity(entity_text, idx, idx + len(entity_text))])

        try:
            masked, pmap = self._engine.mask(
                prompt,
                request_id,
                custom_entity_terms=[entity_text],
                gliner_model=model,
            )
            restored = self._engine.unmask(masked, request_id)
            assert _normalise(restored) == _normalise(prompt), (
                f"Round-trip mismatch.\n"
                f"  original: {_normalise(prompt)!r}\n"
                f"  masked:   {masked!r}\n"
                f"  restored: {_normalise(restored)!r}"
            )
        finally:
            self._engine.discard_mapping(request_id)

    def test_round_trip_no_custom_terms_unchanged(self) -> None:
        """When no custom entity terms are provided, mask/unmask is identity."""
        engine = PIIMaskingEngine()
        prompt = "Hello world, no entities here."
        request_id = "gl2-noterms"
        try:
            masked, pmap = engine.mask(prompt, request_id, custom_entity_terms=[], gliner_model=None)
            restored = engine.unmask(masked, request_id)
            assert _normalise(restored) == _normalise(prompt)
        finally:
            engine.discard_mapping(request_id)


# ---------------------------------------------------------------------------
# GL-3: Empty terms skips GLiNER tier
# Task 8.3 — mask() with custom_entity_terms=[] must not produce any CUSTOM_ENTITY_REDACTED
# Requirements: 3.4
# ---------------------------------------------------------------------------

class TestGL3EmptyTermsSkipsTier:
    """Property GL-3: empty custom_entity_terms must skip the GLiNER tier entirely."""

    _engine = PIIMaskingEngine()

    @given(st.text(min_size=0, max_size=200))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_no_custom_placeholders_when_terms_empty(self, prompt: str) -> None:
        """
        Property GL-3: mask() with custom_entity_terms=[] must produce output
        with no [CUSTOM_ENTITY_REDACTED_N] placeholders.
        """
        request_id = f"gl3-{hash(prompt) & 0xFFFFFF}"
        # Provide a model that would detect something if called — it must NOT be called
        spy_model = MagicMock()
        spy_model.predict_entities = MagicMock(return_value=[])

        try:
            masked, _pmap = self._engine.mask(
                prompt,
                request_id,
                custom_entity_terms=[],
                gliner_model=spy_model,
            )
            # No CUSTOM_ENTITY_REDACTED placeholders must appear
            assert _CUSTOM_PLACEHOLDER_RE.search(masked) is None, (
                f"Found custom placeholder in masked output: {masked!r}"
            )
            # GLiNER model must not have been called when terms is empty
            spy_model.predict_entities.assert_not_called()
        finally:
            self._engine.discard_mapping(request_id)

    @given(st.text(min_size=0, max_size=200))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_no_custom_placeholders_when_model_is_none(self, prompt: str) -> None:
        """mask() with gliner_model=None must also skip the GLiNER tier."""
        request_id = f"gl3-nomodel-{hash(prompt) & 0xFFFFFF}"
        try:
            masked, _pmap = self._engine.mask(
                prompt,
                request_id,
                custom_entity_terms=["SomeEntity"],
                gliner_model=None,
            )
            assert _CUSTOM_PLACEHOLDER_RE.search(masked) is None
        finally:
            self._engine.discard_mapping(request_id)


# ---------------------------------------------------------------------------
# GL-4: Exception isolation
# Task 8.4 — if scan_sync() raises, mask() must return Tier-1/Tier-2 result
# without propagating, and no CUSTOM_ENTITY_REDACTED placeholders in output
# Requirements: 3.8
# ---------------------------------------------------------------------------

class TestGL4ExceptionIsolation:
    """Property GL-4: GLiNER scan_sync exceptions must be absorbed by mask()."""

    @given(
        prompt=st.text(min_size=0, max_size=200),
        exc_type=st.sampled_from([RuntimeError, ValueError, MemoryError, OSError]),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_exception_absorbed_no_propagation(self, prompt: str, exc_type: type) -> None:
        """
        Property GL-4: if scan_sync() raises any exception, mask() must return the
        Tier-1/Tier-2 result without propagating and with no CUSTOM_ENTITY_REDACTED.
        """
        engine = PIIMaskingEngine()
        request_id = f"gl4-{hash(prompt) & 0xFFFFFF}"
        model = MagicMock()
        model.predict_entities = MagicMock(side_effect=exc_type("gliner exploded"))

        try:
            # Must not raise
            masked, pmap = engine.mask(
                prompt,
                request_id,
                custom_entity_terms=["SomeTerm"],
                gliner_model=model,
            )
            # No custom entity placeholders must appear
            assert _CUSTOM_PLACEHOLDER_RE.search(masked) is None, (
                f"Found custom placeholder after GLiNER exception: {masked!r}"
            )
        finally:
            engine.discard_mapping(request_id)

    def test_exception_in_scan_sync_does_not_affect_tier1_masking(self) -> None:
        """When GLiNER raises, Tier-1 PII masking must still have been applied."""
        engine = PIIMaskingEngine()
        prompt = "My SSN is 123-45-6789 and entity ProjectX is here."
        request_id = "gl4-tier1-check"
        model = MagicMock()
        model.predict_entities = MagicMock(side_effect=RuntimeError("boom"))

        try:
            masked, pmap = engine.mask(
                prompt,
                request_id,
                custom_entity_terms=["ProjectX"],
                gliner_model=model,
            )
            # SSN should still be masked by Tier-1/Tier-2
            assert "123-45-6789" not in masked
            # No GLiNER custom placeholder
            assert _CUSTOM_PLACEHOLDER_RE.search(masked) is None
        finally:
            engine.discard_mapping(request_id)
