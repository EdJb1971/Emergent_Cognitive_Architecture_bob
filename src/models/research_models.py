"""Provider-neutral contracts for explicit, auditable research escalation."""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional
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
    BLOCKED_COGNITIVE_GATE = "blocked_cognitive_gate"
    APPROVED = "approved"


class ResearchPacketStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchContextPolicy(str, Enum):
    QUESTION_ONLY = "question_only"


class CognitiveEffortAction(str, Enum):
    ROUTINE_LOCAL = "routine_local"
    DEEPEN_LOCAL = "deepen_local"
    ASK_CLARIFICATION = "ask_clarification"
    ACKNOWLEDGE_UNCERTAINTY = "acknowledge_uncertainty"
    QUEUE_INQUIRY = "queue_inquiry"
    AUTHORIZE_RESEARCH = "authorize_research"


class InquirySourceType(str, Enum):
    WAKING = "waking"
    REFLECTION = "reflection"
    DREAM = "dream"


class InquiryStatus(str, Enum):
    QUEUED = "queued"
    UNDER_REVIEW = "under_review"
    RESOLVED_LOCALLY = "resolved_locally"
    APPROVED = "approved"
    RESEARCHED = "researched"
    RESEARCH_FAILED = "research_failed"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


class InquiryReviewDisposition(str, Enum):
    RESOLVED_LOCALLY = "resolved_locally"
    DEFERRED = "deferred"
    AWAITING_USER_APPROVAL = "awaiting_user_approval"
    RESEARCHED = "researched"
    RESEARCH_FAILED = "research_failed"


class CognitiveResearchSignals(BaseModel):
    epistemic_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    cognitive_conflict: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_prediction_error: float = Field(default=0.0, ge=0.0, le=1.0)
    temporal_volatility: float = Field(default=0.0, ge=0.0, le=1.0)
    task_stakes: float = Field(default=0.0, ge=0.0, le=1.0)
    persistence_after_local_attempts: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_information_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    privacy_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    cloud_cost: float = Field(default=0.0, ge=0.0, le=1.0)
    explicit_user_request: bool = False
    metacognitive_gap: bool = False
    needs_clarification: bool = False


class CognitiveResearchAssessment(BaseModel):
    assessment_id: UUID = Field(default_factory=uuid4)
    source: str
    signals: CognitiveResearchSignals
    drive_score: float = Field(ge=0.0, le=1.0)
    excitation: float = Field(ge=0.0)
    inhibition: float = Field(ge=0.0)
    hysteresis_contribution: float = Field(ge=0.0)
    signal_contributions: Dict[str, float] = Field(default_factory=dict)
    dominant_signals: List[str] = Field(default_factory=list)
    recommended_action: CognitiveEffortAction
    effective_action: CognitiveEffortAction
    shadow_mode: bool
    cooldown_remaining_seconds: float = Field(default=0.0, ge=0.0)
    rationale: str
    controller_version: str = "1"
    assessed_at: datetime = Field(default_factory=utc_now)


def default_inquiry_expiry() -> datetime:
    return utc_now() + timedelta(days=14)


class InquiryCandidate(BaseModel):
    inquiry_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    question: str = Field(min_length=1, max_length=1000)
    hypothesis: Optional[str] = Field(default=None, max_length=2000)
    source_type: InquirySourceType
    source_cycle_ids: List[UUID] = Field(default_factory=list)
    source_pattern_ids: List[UUID] = Field(default_factory=list)
    assessment: CognitiveResearchAssessment
    priority: float = Field(ge=0.0, le=1.0)
    expected_information_gain: float = Field(ge=0.0, le=1.0)
    status: InquiryStatus = InquiryStatus.QUEUED
    shadow_mode: bool = True
    fingerprint: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime = Field(default_factory=default_inquiry_expiry)
    resolution: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


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
    source_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2048)
    publication_date: Optional[str] = None
    accessed_at: datetime = Field(default_factory=utc_now)


class ResearchClaim(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    source_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    start_index: Optional[int] = Field(default=None, ge=0)
    end_index: Optional[int] = Field(default=None, ge=0)


class ResearchPacket(BaseModel):
    request_id: UUID
    decision_id: UUID
    query: str
    status: ResearchPacketStatus
    provider: str
    model: Optional[str] = None
    answer: Optional[str] = Field(default=None, max_length=30000)
    claims: List[ResearchClaim] = Field(default_factory=list)
    sources: List[ResearchSource] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    grounding_verified: bool = False
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


class WakingInquiryReviewOutcome(BaseModel):
    candidate: InquiryCandidate
    disposition: InquiryReviewDisposition
    assessment: CognitiveResearchAssessment
    research_outcome: Optional[ResearchOutcome] = None
    rationale: str
