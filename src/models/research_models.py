"""Provider-neutral contracts for explicit, auditable research escalation."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EscalationReason(str, Enum):
    EXPLICIT_RESEARCH_REQUEST = "explicit_research_request"
    TIME_SENSITIVE = "time_sensitive"
    LOW_CONFIDENCE = "low_confidence"
    NAMED_FACT_MISSING = "named_fact_missing"
    METACOGNITIVE_GAP = "metacognitive_gap"


class EscalationDisposition(str, Enum):
    NOT_REQUIRED = "not_required"
    BLOCKED_DISABLED = "blocked_disabled"
    BLOCKED_LOCAL_ONLY = "blocked_local_only"
    BLOCKED_UNAVAILABLE = "blocked_unavailable"
    APPROVED = "approved"


class ResearchPacketStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchContextPolicy(str, Enum):
    QUESTION_ONLY = "question_only"


class EscalationDecision(BaseModel):
    decision_id: UUID = Field(default_factory=uuid4)
    source: str
    disposition: EscalationDisposition
    reasons: List[EscalationReason] = Field(default_factory=list)
    rationale: str
    research_enabled: bool
    local_only: bool
    provider_available: bool
    provider: str
    model: Optional[str] = None
    query_chars: int = Field(ge=0)
    estimated_query_tokens: int = Field(ge=0)
    context_policy: ResearchContextPolicy = ResearchContextPolicy.QUESTION_ONLY
    policy_version: str = "1"
    decided_at: datetime = Field(default_factory=utc_now)

    @property
    def approved(self) -> bool:
        return self.disposition == EscalationDisposition.APPROVED


class ResearchRequest(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    decision_id: UUID
    query: str = Field(min_length=1)
    reasons: List[EscalationReason]
    context_policy: ResearchContextPolicy = ResearchContextPolicy.QUESTION_ONLY
    context_summary: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class ResearchSource(BaseModel):
    source_id: str
    title: str
    url: str
    publication_date: Optional[str] = None
    accessed_at: datetime = Field(default_factory=utc_now)


class ResearchClaim(BaseModel):
    text: str
    source_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ResearchPacket(BaseModel):
    request_id: UUID
    decision_id: UUID
    query: str
    status: ResearchPacketStatus
    provider: str
    model: Optional[str] = None
    claims: List[ResearchClaim] = Field(default_factory=list)
    sources: List[ResearchSource] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    caveats: List[str] = Field(default_factory=list)
    context_policy: ResearchContextPolicy = ResearchContextPolicy.QUESTION_ONLY
    context_chars: int = Field(default=0, ge=0)
    latency_ms: Optional[float] = Field(default=None, ge=0.0)
    estimated_cost: Optional[float] = Field(default=None, ge=0.0)
    completed_at: datetime = Field(default_factory=utc_now)


class ResearchOutcome(BaseModel):
    decision: EscalationDecision
    packets: List[ResearchPacket] = Field(default_factory=list)
