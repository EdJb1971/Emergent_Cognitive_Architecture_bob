"""Cached, durable identity settings with optimistic concurrency."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from src.models.identity_models import IdentityProfile, IdentityUpdateRequest


class IdentityConflictError(ValueError):
    """Raised when a stale browser attempts to overwrite newer settings."""


class IdentityService:
    def __init__(self, path: str | Path, default_assistant_name: str = "Bob") -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()
        self._profile = IdentityProfile(assistant_name=default_assistant_name)

    @property
    def current(self) -> IdentityProfile:
        """Return the in-memory snapshot; no I/O occurs on cognitive hot paths."""
        return self._profile

    @property
    def assistant_name(self) -> str:
        return self._profile.assistant_name

    @property
    def user_name(self) -> str | None:
        return self._profile.user_name

    async def connect(self) -> IdentityProfile:
        async with self._lock:
            if self.path.exists():
                self._profile = IdentityProfile.model_validate_json(
                    self.path.read_text(encoding="utf-8")
                )
            else:
                self._write(self._profile)
            return self._profile

    async def update(self, request: IdentityUpdateRequest) -> IdentityProfile:
        async with self._lock:
            if request.expected_revision != self._profile.revision:
                raise IdentityConflictError(
                    "Identity settings changed in another session; reload and try again."
                )
            aliases = list(self._profile.assistant_aliases)
            if request.assistant_name.casefold() != self._profile.assistant_name.casefold():
                previous = self._profile.assistant_name
                if all(previous.casefold() != item.casefold() for item in aliases):
                    aliases.append(previous)
            profile = IdentityProfile(
                assistant_name=request.assistant_name,
                user_name=request.user_name,
                assistant_aliases=tuple(aliases[-8:]),
                revision=self._profile.revision + 1,
                updated_at=datetime.now(timezone.utc),
            )
            self._write(profile)
            self._profile = profile
            return profile

    def prompt_context(self) -> str:
        profile = self._profile
        user = profile.user_name or "not configured; address the person generically"
        aliases = ", ".join(profile.assistant_aliases) or "none"
        return (
            "Authoritative configured identity (overrides names inferred from memory):\n"
            f"- Your current name: {profile.assistant_name}\n"
            f"- Operator name: {user}\n"
            f"- Your former configured names: {aliases}\n"
            "Never infer the operator's name from entities, the operating system, or unrelated memories."
        )

    def _write(self, profile: IdentityProfile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(self.path)
