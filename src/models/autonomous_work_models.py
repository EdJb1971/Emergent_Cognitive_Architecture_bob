"""Shared contracts for every bounded autonomous cognitive task."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AutonomousTaskType(str, Enum):
    SLEEP = "sleep"
    REFLECTION = "reflection"
    DISCOVERY = "discovery"
    CURIOSITY = "curiosity"
    SELF_ASSESSMENT = "self_assessment"
    PROACTIVE_ENGAGEMENT = "proactive_engagement"
    SUMMARY_UPDATE = "summary_update"
    STM_FLUSH = "stm_flush"


class AutonomousTaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class AutonomousProviderPolicy(str, Enum):
    LOCAL_ONLY = "local_only"
    NO_INFERENCE = "no_inference"


class AutonomousEventType(str, Enum):
    GOVERNOR_STARTED = "governor_started"
    GOVERNOR_STOPPED = "governor_stopped"
    RUNTIME_CHANGED = "runtime_changed"
    TASK_QUEUED = "task_queued"
    TASK_REJECTED = "task_rejected"
    TASK_DUPLICATE = "task_duplicate"
    TASK_STARTED = "task_started"
    TASK_RETRYING = "task_retrying"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"


class AutonomousTaskPolicy(BaseModel):
    task_type: AutonomousTaskType
    enabled: bool = False
    cooldown_seconds: float = Field(0.0, ge=0.0)
    timeout_seconds: float = Field(300.0, gt=0.0)
    max_retries: int = Field(0, ge=0, le=10)
    max_per_hour: int = Field(12, ge=1, le=10000)
    max_concurrent_per_user: int = Field(1, ge=1, le=10)
    provider_policy: AutonomousProviderPolicy = AutonomousProviderPolicy.LOCAL_ONLY
    cancel_on_user_activity: bool = True
    description: str = ""


class AutonomousTaskRequest(BaseModel):
    task_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    task_type: AutonomousTaskType
    trigger_reason: str = Field(..., min_length=1, max_length=500)
    signals: Dict[str, Any] = Field(default_factory=dict)
    payload: Dict[str, Any] = Field(default_factory=dict)
    deduplication_key: str = Field(..., min_length=1, max_length=500)
    provider_policy: AutonomousProviderPolicy = AutonomousProviderPolicy.LOCAL_ONLY
    priority: float = Field(0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)


class AutonomousTaskRecord(BaseModel):
    request: AutonomousTaskRequest
    status: AutonomousTaskStatus = AutonomousTaskStatus.QUEUED
    attempt: int = Field(0, ge=0)
    max_attempts: int = Field(1, ge=1)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    rejection_reason: Optional[str] = None


class AutonomousRuntimeState(BaseModel):
    master_enabled: bool
    max_concurrent_global: int
    active_count: int = 0
    queued_count: int = 0
    policies: Dict[AutonomousTaskType, AutonomousTaskPolicy]
    changed_at: datetime = Field(default_factory=utc_now)
    persistence: str = "sqlite"


class AutonomousRuntimeUpdate(BaseModel):
    master_enabled: Optional[bool] = None
    category_enabled: Dict[AutonomousTaskType, bool] = Field(default_factory=dict)
    reason: str = Field(..., min_length=3, max_length=500)


class AutonomousLedgerEvent(BaseModel):
    sequence: int = Field(..., ge=1)
    event_id: UUID
    event_type: AutonomousEventType
    user_id: UUID
    task_id: Optional[UUID] = None
    task_type: Optional[AutonomousTaskType] = None
    created_at: datetime
    payload: Dict[str, Any] = Field(default_factory=dict)
    previous_hash: str
    event_hash: str

