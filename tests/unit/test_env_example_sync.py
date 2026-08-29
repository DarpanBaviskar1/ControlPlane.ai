"""Config contract: `.env.example` documents exactly the Settings fields.

Ruling 27 requires every task that touches `Settings` to update `.env.example`
in the same commit.  Until now that was enforced only by review vigilance,
while Tasks 3, 6 and 7 each change both sides — so the rule drifted the moment
nobody checked.  This test makes the drift a test failure instead.

A key may appear commented out (`# LLM_API_BASE=...`); documenting an optional
setting by example is the file's own convention, so commented keys count as
documented.  `LLM_API_KEY` appears five times (one live, four provider
examples), which is why both sides are compared as SETS, never as counts.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"

# Matches `KEY=`, `# KEY=` and `#KEY=`; captures the key only.
_KEY = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)


def _settings_fields() -> set[str]:
    """The env-configurable field names, which are the UPPER_SNAKE ones."""
    return {name for name in Settings.model_fields if name.isupper()}


@pytest.fixture(scope="module")
def documented() -> set[str]:
    return set(_KEY.findall(ENV_EXAMPLE.read_text()))


def test_every_setting_is_documented(documented: set[str]) -> None:
    undocumented = _settings_fields() - documented
    assert undocumented == set(), (
        "these Settings fields are missing from .env.example: "
        f"{sorted(undocumented)}"
    )


def test_no_documented_key_is_a_dead_setting(documented: set[str]) -> None:
    """A key removed from Settings must be removed from .env.example too.

    This is the half that catches a deleted field whose documentation was left
    behind — `.env.example` describing a setting the code no longer reads.
    """
    orphaned = documented - _settings_fields()
    assert orphaned == set(), (
        "these .env.example keys are not Settings fields: " f"{sorted(orphaned)}"
    )
