"""Guardrails AI output validation chain (Req. 2.11-13).

Design:
- Wraps guardrails-ai validators in a sequential chain.
- Each validator is loaded from the Guardrails AI Hub on startup.
- If guardrails is not installed, the validator returns a passing verdict
  with a warning log so the rest of the pipeline is unaffected.
- on_fail actions:
    "exception"  -> GuardrailsVerdict(passed=False, action="exception")
    "filter"     -> GuardrailsVerdict(passed=False, action="filter")
    "fix"        -> GuardrailsVerdict(passed=True, action="fix", fixed_output=...)
- Validators run in asyncio.to_thread since guardrails is synchronous/CPU-bound.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import settings
from app.models import GuardrailsVerdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional guardrails import — degrades gracefully
# ---------------------------------------------------------------------------
try:
    import guardrails as gd  # type: ignore[import-untyped]
    _GUARDRAILS_AVAILABLE = True
except ImportError:
    _GUARDRAILS_AVAILABLE = False
    logger.info(
        "guardrails-ai package not installed — output validation will be skipped"
    )


# ---------------------------------------------------------------------------
# Validator registry
# ---------------------------------------------------------------------------

# Map from Hub validator ID to its Python class (populated at startup)
_LOADED_VALIDATORS: list[Any] = []


def _hub_install(validator_id: str) -> None:
    """Attempt `guardrails hub install <id>` via subprocess."""
    import os
    import subprocess
    import sys
    env = os.environ.copy()
    if settings.GUARDRAILS_HUB_TOKEN:
        env["GUARDRAILS_TOKEN"] = settings.GUARDRAILS_HUB_TOKEN
    subprocess.check_call(
        [sys.executable, "-m", "guardrails", "hub", "install", validator_id],
        env=env,
        timeout=60,
    )


def load_validators() -> None:
    """Load validators from Guardrails Hub.  Called once at startup."""
    if not _GUARDRAILS_AVAILABLE:
        logger.warning("GUARDRAILS_DEGRADED — no validators active")
        return

    validator_ids = [v.strip() for v in settings.GUARDRAILS_VALIDATORS.split(",") if v.strip()]
    loaded, skipped = [], []

    for vid in validator_ids:
        try:
            validator = gd.hub.load(vid)  # type: ignore[attr-defined]
            _LOADED_VALIDATORS.append((vid, validator))
            loaded.append(vid)
        except Exception:
            try:
                _hub_install(vid)
                validator = gd.hub.load(vid)  # type: ignore[attr-defined]
                _LOADED_VALIDATORS.append((vid, validator))
                loaded.append(vid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GUARDRAILS_SKIPPED validator=%s reason=%s", vid, exc)
                skipped.append(vid)

    if loaded:
        logger.info("GUARDRAILS_LOADED validators=%s", ",".join(loaded))
    if skipped:
        logger.warning("GUARDRAILS_SKIPPED validators=%s", ",".join(skipped))
    if not loaded:
        logger.warning("GUARDRAILS_DEGRADED — no validators active")


# ---------------------------------------------------------------------------
# Synchronous validation (runs in thread pool)
# ---------------------------------------------------------------------------

def _run_validation_sync(text: str) -> GuardrailsVerdict:
    """Run all loaded validators against *text* synchronously."""
    if not _GUARDRAILS_AVAILABLE or not _LOADED_VALIDATORS:
        return GuardrailsVerdict(passed=True)

    for validator_id, validator_cls in _LOADED_VALIDATORS:
        try:
            # Guardrails validators expose a validate(value, metadata) interface.
            result = validator_cls.validate(value=text, metadata={})

            # Check if the validation outcome signals a block or fix.
            # The exact API depends on guardrails version; we use duck-typing.
            outcome = getattr(result, "outcome", None)
            if outcome in ("fail", "error"):
                action = getattr(result, "on_fail_action", "exception")
                if action == "fix":
                    fixed = getattr(result, "fix_value", text)
                    logger.info(
                        "Guardrails validator '%s' triggered fix action", validator_id
                    )
                    return GuardrailsVerdict(
                        passed=True,
                        action="fix",
                        triggered_validator=validator_id,
                        fixed_output=fixed,
                    )
                else:
                    logger.info(
                        "Guardrails validator '%s' triggered %s action", validator_id, action
                    )
                    return GuardrailsVerdict(
                        passed=False,
                        action=action if action in ("exception", "filter") else "exception",
                        triggered_validator=validator_id,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Guardrails validator '%s' raised during validation: %s", validator_id, exc
            )
            # Treat unexpected errors as a hard block to be safe
            return GuardrailsVerdict(
                passed=False,
                action="exception",
                triggered_validator=validator_id,
            )

    return GuardrailsVerdict(passed=True)


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------

async def validate_output(text: str) -> GuardrailsVerdict:
    """Validate an LLM output string through the Guardrails AI chain (Req. 2.11).

    Returns a GuardrailsVerdict describing whether the output passed, was
    fixed, or should be blocked.
    """
    return await asyncio.to_thread(_run_validation_sync, text)
