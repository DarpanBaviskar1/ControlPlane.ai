"""Test isolation: pin `settings` to its declared defaults.

`Settings` reads `.env` (app/config.py:34), so a developer with real
credentials on disk silently changes what the suite exercises.  Two concrete
failures this caused:

  * `test_whitespace_only_prompt_returns_422` and
    `test_prompt_max_length_exactly_32768_valid` dispatched real requests to
    the configured provider instead of exercising request validation — which
    exhausted a free-tier daily quota and then failed on the 429.
  * `test_llm_provider_default` / `test_llm_fallback_model_default` assert the
    *declared* defaults, so any `.env` naming a provider or model failed them.

Resetting every field to its declared default makes each test run as though no
`.env` existed.  Tests that need a live-looking key still opt in explicitly via
`monkeypatch.setattr(settings, ...)`; this fixture only sets the baseline, so
that ordering keeps working.
"""
from __future__ import annotations

import pytest

from app.config import Settings, settings


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset every Settings field to its declared default for each test."""
    for name, field in Settings.model_fields.items():
        monkeypatch.setattr(settings, name, field.default, raising=False)
