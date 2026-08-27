"""Policy Layer — hot-reload config loader.

Loads Use-Case Profile configurations from a YAML or JSON file on disk.
Uses watchdog to detect file modifications and reloads within 5 seconds.
The live configuration dict is protected by an asyncio.Lock and swapped
atomically: the new config is only activated if every profile passes
Pydantic validation.

Startup bootstrap: the two built-in profiles (customer_chatbot,
internal_copilot) are always merged in so the gateway starts even without
a config file present.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from app.models import UseCaseProfile
from app.policy.defaults import BUILT_IN_PROFILES

logger = logging.getLogger(__name__)


class _PolicyFileHandler(FileSystemEventHandler):
    """Watchdog handler that triggers an async reload on file modification."""

    def __init__(self, loader: "PolicyLoader") -> None:
        super().__init__()
        self._loader = loader
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._loader.reload(), self._loop)


class PolicyLoader:
    """Thread-safe, hot-reloading policy configuration loader.

    Usage::

        loader = PolicyLoader(policy_file_path="/etc/gateway/policy.yaml")
        await loader.start()                 # starts watchdog observer
        profile = await loader.get_profile("customer_chatbot")
        await loader.stop()                  # stops observer
    """

    def __init__(self, policy_file_path: str | Path | None = None) -> None:
        self._path: Path | None = Path(policy_file_path) if policy_file_path else None
        self._profiles: dict[str, UseCaseProfile] = dict(BUILT_IN_PROFILES)
        self._lock = asyncio.Lock()
        self._observer: Observer | None = None
        self._handler = _PolicyFileHandler(self)
        self._last_reload_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Bootstrap: merge built-in profiles with the file (if any) and start watchdog."""
        if self._path and self._path.exists():
            await self.reload()

        if self._path:
            loop = asyncio.get_running_loop()
            self._handler.set_loop(loop)
            observer = Observer()
            # Watch the parent directory so renames/writes are both caught
            watch_dir = str(self._path.parent)
            observer.schedule(self._handler, path=watch_dir, recursive=False)
            observer.daemon = True
            observer.start()
            self._observer = observer
            logger.info("PolicyLoader: watchdog started on %s", watch_dir)

    async def stop(self) -> None:
        """Stop the watchdog observer."""
        if self._observer is not None:
            self._observer.stop()
            # Join in a thread to avoid blocking the event loop
            await asyncio.to_thread(self._observer.join)
            self._observer = None

    async def get_profile(self, name: str) -> UseCaseProfile:
        """Return the named profile, or raise KeyError if unknown."""
        async with self._lock:
            profile = self._profiles.get(name)
        if profile is None:
            raise KeyError(f"Unknown use_case_profile: '{name}'")
        return profile

    def list_profiles(self) -> list[str]:
        """Return a snapshot list of currently loaded profile names."""
        return list(self._profiles.keys())

    async def reload(self) -> None:
        """Parse the policy file and atomically swap the live config on success.

        If the file is missing, malformed, or any profile fails validation,
        the previous (valid) configuration remains active and an error is logged.
        """
        if self._path is None:
            return

        try:
            raw = await asyncio.to_thread(self._path.read_text, encoding="utf-8")
        except OSError as exc:
            logger.error("PolicyLoader: cannot read %s — %s", self._path, exc)
            return

        try:
            data: Any = yaml.safe_load(raw) if self._path.suffix in (".yaml", ".yml") else json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.error("PolicyLoader: parse error in %s — %s", self._path, exc)
            return

        if not isinstance(data, dict) or "profiles" not in data:
            logger.error("PolicyLoader: %s must contain a top-level 'profiles' list", self._path)
            return

        candidate: dict[str, UseCaseProfile] = dict(BUILT_IN_PROFILES)
        for raw_profile in data["profiles"]:
            try:
                profile = UseCaseProfile.model_validate(raw_profile)
            except ValidationError as exc:
                logger.error(
                    "PolicyLoader: validation failed for profile in %s — %s",
                    self._path,
                    exc,
                )
                return  # abort — keep previous config
            candidate[profile.name] = profile

        async with self._lock:
            self._profiles = candidate
            self._last_reload_ts = time.monotonic()

        logger.info(
            "PolicyLoader: loaded %d profiles from %s",
            len(candidate),
            self._path,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def last_reload_ts(self) -> float:
        return self._last_reload_ts
