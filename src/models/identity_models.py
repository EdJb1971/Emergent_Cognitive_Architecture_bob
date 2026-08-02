"""Authoritative operator-configured identity contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _clean_name(value: str, *, field_name: str, maximum: int) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise ValueError(f"{field_name} cannot be blank")
    if len(cleaned) > maximum:
        raise ValueError(f"{field_name} must be {maximum} characters or fewer")
    if any(ord(char) < 32 for char in cleaned):
        raise ValueError(f"{field_name} contains unsupported control characters")
    return cleaned


class IdentityProfile(BaseModel):
    """The sole authority for names used by the runtime and operator UI."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    assistant_name: str = Field(min_length=1, max_length=40)
    user_name: Optional[str] = Field(default=None, max_length=80)
    assistant_aliases: tuple[str, ...] = ()
    revision: int = Field(default=1, ge=1)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("assistant_name")
    @classmethod
    def validate_assistant_name(cls, value: str) -> str:
        return _clean_name(value, field_name="Assistant name", maximum=40)

    @field_validator("user_name")
    @classmethod
    def validate_user_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not value.strip():
            return None
        return _clean_name(value, field_name="Your name", maximum=80)


class IdentityUpdateRequest(BaseModel):
    assistant_name: str = Field(min_length=1, max_length=40)
    user_name: Optional[str] = Field(default=None, max_length=80)
    expected_revision: int = Field(ge=1)

    @field_validator("assistant_name")
    @classmethod
    def validate_assistant_name(cls, value: str) -> str:
        return _clean_name(value, field_name="Assistant name", maximum=40)

    @field_validator("user_name")
    @classmethod
    def validate_user_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not value.strip():
            return None
        return _clean_name(value, field_name="Your name", maximum=80)


class CleanStartRequest(BaseModel):
    confirmation: str
    preserve_identity: bool = True


class CleanStartStatus(BaseModel):
    pending_restart: bool
    preserve_identity: bool = True
    requested_at: Optional[datetime] = None
    message: str
