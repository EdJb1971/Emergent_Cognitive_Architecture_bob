from typing import Optional
from uuid import uuid4

import pytest

from src.models.research_models import (
    CognitiveResearchSignals,
    InquiryCandidate,
    InquiryReviewDisposition,
    InquirySourceType,
    InquiryStatus,
    ResearchClaim,
    ResearchPacket,
    ResearchPacketStatus,
    ResearchRequest,
    ResearchSource,
)
from src.services.cognitive_research_drive import CognitiveResearchDrive
from src.services.escalation_policy import EscalationPolicy
from src.services.inquiry_candidate_store import InquiryCandidateStore
from src.services.research_service import ResearchService
from src.services.waking_inquiry_service import WakingInquiryService


class GroundedProvider:
    provider_name = "grounded-test"
    model_name: Optional[str] = "grounded-test-v1"

    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid
        self.requests = []

    def is_available(self) -> bool:
        return True

    async def research(self, request: ResearchRequest) -> ResearchPacket:
        self.requests.append(request)
        return ResearchPacket(
            request_id=request.request_id,
            decision_id=request.decision_id,
            query=request.query,
            status=ResearchPacketStatus.COMPLETED,
            provider=self.provider_name,
            model=self.model_name,
            answer="Grounded answer",
            sources=[
                ResearchSource(source_id="s1", title="Primary", url="https://example.test/source")
            ],
            claims=[
                ResearchClaim(text="Grounded answer", source_ids=["s1"], confidence=0.9)
            ],
            grounding_verified=self.valid,
        )


def _signals(**updates):
    values = {
        "epistemic_uncertainty": 0.95,
        "cognitive_conflict": 0.9,
        "novelty_prediction_error": 0.8,
        "temporal_volatility": 0.9,
        "task_stakes": 0.9,
        "persistence_after_local_attempts": 0.8,
        "expected_information_gain": 0.95,
        "metacognitive_gap": True,
    }
    values.update(updates)
    return CognitiveResearchSignals(**values)


async def _queued_candidate(store, user_id):
    assessment = CognitiveResearchDrive().assess(_signals(), source="dream")
    candidate = InquiryCandidate(
        user_id=user_id,
        question="What is the latest verified status of the named project?",
        source_type=InquirySourceType.DREAM,
        assessment=assessment,
        priority=assessment.drive_score,
        expected_information_gain=assessment.signals.expected_information_gain,
    )
    return (await store.enqueue(candidate))[0]


@pytest.mark.asyncio
async def test_offline_candidate_requires_waking_user_approval(tmp_path):
    store = InquiryCandidateStore(tmp_path / "inquiries.sqlite3")
    await store.connect()
    user_id = uuid4()
    candidate = await _queued_candidate(store, user_id)
    drive = CognitiveResearchDrive(enabled=True, shadow_mode=False)
    provider = GroundedProvider()
    service = WakingInquiryService(
        store,
        drive,
        ResearchService(EscalationPolicy(research_enabled=True), provider),
        require_user_approval=True,
    )

    outcome = await service.review_candidate(
        user_id=user_id,
        inquiry_id=candidate.inquiry_id,
        signals=_signals(),
    )

    assert outcome.disposition == InquiryReviewDisposition.AWAITING_USER_APPROVAL
    assert outcome.candidate.status == InquiryStatus.QUEUED
    assert provider.requests == []


@pytest.mark.asyncio
async def test_approved_waking_review_completes_grounded_research(tmp_path):
    store = InquiryCandidateStore(tmp_path / "inquiries.sqlite3")
    await store.connect()
    user_id = uuid4()
    candidate = await _queued_candidate(store, user_id)
    drive = CognitiveResearchDrive(enabled=True, shadow_mode=False)
    provider = GroundedProvider()
    service = WakingInquiryService(
        store,
        drive,
        ResearchService(EscalationPolicy(research_enabled=True), provider),
    )

    outcome = await service.review_candidate(
        user_id=user_id,
        inquiry_id=candidate.inquiry_id,
        signals=_signals(),
        user_approved=True,
    )

    assert outcome.disposition == InquiryReviewDisposition.RESEARCHED
    assert outcome.candidate.status == InquiryStatus.RESEARCHED
    assert outcome.research_outcome.packets[0].grounding_verified is True
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_fresh_low_drive_resolves_candidate_locally(tmp_path):
    store = InquiryCandidateStore(tmp_path / "inquiries.sqlite3")
    await store.connect()
    user_id = uuid4()
    candidate = await _queued_candidate(store, user_id)
    drive = CognitiveResearchDrive(enabled=True, shadow_mode=False)
    provider = GroundedProvider()
    service = WakingInquiryService(
        store,
        drive,
        ResearchService(EscalationPolicy(research_enabled=True), provider),
    )

    outcome = await service.review_candidate(
        user_id=user_id,
        inquiry_id=candidate.inquiry_id,
        signals=CognitiveResearchSignals(epistemic_uncertainty=0.1),
    )

    assert outcome.disposition == InquiryReviewDisposition.RESOLVED_LOCALLY
    assert outcome.candidate.status == InquiryStatus.RESOLVED_LOCALLY
    assert provider.requests == []


@pytest.mark.asyncio
async def test_invalid_grounding_enters_retryable_failure_state(tmp_path):
    store = InquiryCandidateStore(tmp_path / "inquiries.sqlite3")
    await store.connect()
    user_id = uuid4()
    candidate = await _queued_candidate(store, user_id)
    provider = GroundedProvider(valid=False)
    drive = CognitiveResearchDrive(enabled=True, shadow_mode=False)
    service = WakingInquiryService(
        store,
        drive,
        ResearchService(EscalationPolicy(research_enabled=True), provider),
    )

    outcome = await service.review_candidate(
        user_id=user_id,
        inquiry_id=candidate.inquiry_id,
        signals=_signals(),
        user_approved=True,
    )

    assert outcome.disposition == InquiryReviewDisposition.RESEARCH_FAILED
    assert outcome.candidate.status == InquiryStatus.RESEARCH_FAILED
    assert outcome.research_outcome.packets[0].status == ResearchPacketStatus.FAILED
