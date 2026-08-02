"""Contracts for governed sleep-cycle scheduling and audit."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SleepLedgerEventType(str, Enum):
    COORDINATOR_STARTED = "coordinator_started"
    COORDINATOR_STOPPED = "coordinator_stopped"
    CYCLE_STARTED = "cycle_started"
    CYCLE_SKIPPED = "cycle_skipped"
    CYCLE_COMPLETED = "cycle_completed"
    CYCLE_FAILED = "cycle_failed"
    CYCLE_CANCELLED = "cycle_cancelled"
    JOB_CREATED = "job_created"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"


class SleepLedgerEvent(BaseModel):
    sequence: int = Field(..., ge=1)
    event_id: UUID
    event_type: SleepLedgerEventType
    user_id: UUID
    run_id: Optional[UUID] = None
    job_id: Optional[UUID] = None
    created_at: datetime = Field(default_factory=utc_now)
    payload: Dict[str, Any] = Field(default_factory=dict)
    previous_hash: str
    event_hash: str

