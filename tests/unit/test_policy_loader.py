"""Tests for PolicyLoader — Tasks 2.1, 2.2, 2.3.

# Feature: controlplane-ai-gateway, Property 1: Profile Load Correctness
Validates: Requirements 1.2, 7.1, 7.2, 7.5
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from app.models import UseCaseProfile
from app.policy.defaults import BUILT_IN_PROFILES
from app.policy.loader import PolicyLoader

# ---------------------------------------------------------------------------
# Helpers
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


def _write_yaml(path: Path, profiles: list[dict]) -> None:
    path.write_text(yaml.dump({"profiles": profiles}), encoding="utf-8")


def _write_json(path: Path, profiles: list[dict]) -> None:
    path.write_text(json.dumps({"profiles": profiles}), encoding="utf-8")


# ---------------------------------------------------------------------------
# Task 2.1 — Built-in profiles always available
# ---------------------------------------------------------------------------


class TestPolicyLoaderBuiltins:
    @pytest.mark.asyncio
    async def test_built_in_profiles_available_without_file(self) -> None:
        loader = PolicyLoader(policy_file_path=None)
        await loader.start()
        for name in ("customer_chatbot", "internal_copilot"):
            profile = await loader.get_profile(name)
            assert profile.name == name
        await loader.stop()

    @pytest.mark.asyncio
    async def test_built_in_profiles_not_overwritten_by_file(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        custom = dict(VALID_PROFILE_DICT)
        custom["name"] = "my_custom_profile"
        _write_yaml(policy_file, [custom])
        loader = PolicyLoader(policy_file_path=policy_file)
        await loader.start()
        # Built-ins must still be present
        cc = await loader.get_profile("customer_chatbot")
        assert cc.name == "customer_chatbot"
        # Custom profile must also be present
        cp = await loader.get_profile("my_custom_profile")
        assert cp.name == "my_custom_profile"
        await loader.stop()

    @pytest.mark.asyncio
    async def test_list_profiles_includes_builtins(self) -> None:
        loader = PolicyLoader(policy_file_path=None)
        await loader.start()
        names = loader.list_profiles()
        assert "customer_chatbot" in names
        assert "internal_copilot" in names
        await loader.stop()


# ---------------------------------------------------------------------------
# Task 2.1 — Hot-reload within 5 seconds
# ---------------------------------------------------------------------------


class TestPolicyLoaderHotReload:
    @pytest.mark.asyncio
    async def test_reload_method_updates_config(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        _write_yaml(policy_file, [VALID_PROFILE_DICT])
        loader = PolicyLoader(policy_file_path=policy_file)
        await loader.start()

        # First load
        p = await loader.get_profile("test_profile")
        assert p.latency_budget_ms == 10_000

        # Update the file with a new latency budget and reload
        updated = dict(VALID_PROFILE_DICT)
        updated["latency_budget_ms"] = 20_000
        _write_yaml(policy_file, [updated])
        await loader.reload()

        p2 = await loader.get_profile("test_profile")
        assert p2.latency_budget_ms == 20_000
        await loader.stop()

    @pytest.mark.asyncio
    async def test_reload_records_last_reload_ts(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        _write_yaml(policy_file, [VALID_PROFILE_DICT])
        loader = PolicyLoader(policy_file_path=policy_file)
        before = time.monotonic()
        await loader.start()
        after = time.monotonic()
        assert loader.last_reload_ts >= before
        assert loader.last_reload_ts <= after + 0.1
        await loader.stop()

    @pytest.mark.asyncio
    async def test_invalid_file_keeps_previous_config(self, tmp_path: Path) -> None:
        """Broken YAML must NOT overwrite the valid live config."""
        policy_file = tmp_path / "policy.yaml"
        _write_yaml(policy_file, [VALID_PROFILE_DICT])
        loader = PolicyLoader(policy_file_path=policy_file)
        await loader.start()

        # Overwrite with garbage
        policy_file.write_text("profiles: [\n  - broken: [invalid yaml", encoding="utf-8")
        await loader.reload()

        # Previous config must still be active
        p = await loader.get_profile("test_profile")
        assert p.name == "test_profile"
        await loader.stop()

    @pytest.mark.asyncio
    async def test_invalid_profile_field_keeps_previous_config(self, tmp_path: Path) -> None:
        """A profile with an out-of-range field must reject the whole file."""
        policy_file = tmp_path / "policy.yaml"
        _write_yaml(policy_file, [VALID_PROFILE_DICT])
        loader = PolicyLoader(policy_file_path=policy_file)
        await loader.start()

        bad = dict(VALID_PROFILE_DICT)
        bad["latency_budget_ms"] = -1  # invalid
        _write_yaml(policy_file, [bad])
        await loader.reload()

        # Still has the original valid profile
        p = await loader.get_profile("test_profile")
        assert p.latency_budget_ms == 10_000
        await loader.stop()

    @pytest.mark.asyncio
    async def test_json_policy_file_loads(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.json"
        _write_json(policy_file, [VALID_PROFILE_DICT])
        loader = PolicyLoader(policy_file_path=policy_file)
        await loader.start()
        p = await loader.get_profile("test_profile")
        assert p.name == "test_profile"
        await loader.stop()


# ---------------------------------------------------------------------------
# Task 2.2 — get_profile() raises for unknown names
# ---------------------------------------------------------------------------


class TestGetProfile:
    @pytest.mark.asyncio
    async def test_get_profile_known_name_returns_profile(self) -> None:
        loader = PolicyLoader()
        await loader.start()
        p = await loader.get_profile("customer_chatbot")
        assert isinstance(p, UseCaseProfile)
        await loader.stop()

    @pytest.mark.asyncio
    async def test_get_profile_unknown_name_raises_key_error(self) -> None:
        loader = PolicyLoader()
        await loader.start()
        with pytest.raises(KeyError, match="unknown_profile"):
            await loader.get_profile("unknown_profile")
        await loader.stop()

    @pytest.mark.asyncio
    async def test_get_profile_empty_string_raises(self) -> None:
        loader = PolicyLoader()
        await loader.start()
        with pytest.raises(KeyError):
            await loader.get_profile("")
        await loader.stop()


# ---------------------------------------------------------------------------
# Property 1: Profile Load Correctness
# ---------------------------------------------------------------------------


class TestProfileLoadCorrectness:
    """For any valid profile name present in the loader,
    get_profile(name).name == name and all fields satisfy constraints.
    """

    @given(
        st.fixed_dictionaries(
            {
                "name": st.text(min_size=1, max_size=50).filter(
                    lambda s: s not in ("customer_chatbot", "internal_copilot")
                ),
                "latency_budget_ms": st.integers(min_value=1, max_value=300_000),
                "complexity_threshold": st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False
                ),
                "token_compression_threshold": st.integers(min_value=1, max_value=100_000),
                "groundedness_pass_threshold": st.floats(
                    min_value=0.0, max_value=1.0, allow_nan=False
                ),
                "inspection_timeout_ms": st.integers(min_value=1, max_value=60_000),
                "pii_masking_enabled": st.booleans(),
                "human_escalation_enabled": st.booleans(),
            }
        )
    )
    @settings(max_examples=100)
    def test_loaded_profile_name_matches_request(self, profile_dict: dict) -> None:
        """
        # Feature: controlplane-ai-gateway, Property 1: Profile Load Correctness
        """
        profile = UseCaseProfile.model_validate(profile_dict)
        # Simulate what the loader does: validate and store
        assert profile.name == profile_dict["name"]
        # All field constraints satisfied
        assert 1 <= profile.latency_budget_ms <= 300_000
        assert 0.0 <= profile.complexity_threshold <= 1.0
        assert profile.token_compression_threshold >= 1
        assert 0.0 <= profile.groundedness_pass_threshold <= 1.0
        assert 1 <= profile.inspection_timeout_ms <= 60_000

    @pytest.mark.asyncio
    async def test_get_profile_name_field_matches_requested_name(
        self, tmp_path: Path
    ) -> None:
        """get_profile(name).name == name for all profiles in the loader."""
        policy_file = tmp_path / "policy.yaml"
        profiles = [
            dict(VALID_PROFILE_DICT, name="alpha"),
            dict(VALID_PROFILE_DICT, name="beta"),
        ]
        _write_yaml(policy_file, profiles)
        loader = PolicyLoader(policy_file_path=policy_file)
        await loader.start()

        for name in ("alpha", "beta", "customer_chatbot", "internal_copilot"):
            p = await loader.get_profile(name)
            assert p.name == name, f"Profile name mismatch: expected {name!r}, got {p.name!r}"

        await loader.stop()
