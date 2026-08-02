"""Typed, process-local telemetry contracts for the operator observability plane."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TelemetryDomain(str, Enum):
    COGNITIVE = "cognitive"
    MEMORY = "memory"
    RESEARCH = "research"
    SALIENCE = "salience"
    SLEEP = "sleep"
    AUTONOMOUS_WORK = "autonomous_work"
    SYSTEM = "system"


class TelemetryEvent(BaseModel):
    """A bounded projection of an authoritative domain event."""

    schema_version: int = 1
    sequence: int = Field(ge=1)
    event_id: UUID = Field(default_factory=uuid4)
    domain: TelemetryDomain
    event_type: str = Field(min_length=1, max_length=96)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)
    cycle_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    source_reference: Optional[str] = None


class TelemetryGap(BaseModel):
    """Signals that a cursor or subscriber fell behind a bounded buffer."""

    requested_after: int = Field(ge=0)
    available_from: int = Field(ge=1)
    latest_sequence: int = Field(ge=0)
    dropped_for_subscriber: int = Field(default=0, ge=0)
    reason: str


class TelemetryHello(BaseModel):
    schema_version: int = 1
    stream_id: UUID
    replay_capacity: int
    subscriber_queue_capacity: int
    oldest_sequence: int
    latest_sequence: int
    domains: list[TelemetryDomain]

