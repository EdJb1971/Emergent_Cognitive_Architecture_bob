from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from src.api.research_review import router
from src.dependencies import SYSTEM_USER_ID, get_api_key_user_id
from src.models.research_models import (
    CognitiveResearchSignals,
    CognitiveEffortAction,
    InquiryCandidate,
    InquirySourceType,
    InquiryStatus,
    ResearchClaim,
    ResearchLedgerEventType,
    ResearchPacket,
    ResearchPacketStatus,
    ResearchSource,
)
from src.services.cognitive_research_drive import CognitiveResearchDrive
from src.services.escalation_policy import EscalationPolicy
from src.services.inquiry_candidate_store import InquiryCandidateStore
from src.services.inquiry_review_service import InquiryReviewService
from src.services.research_calibration_ledger import ResearchCalibrationLedger
from src.services.research_service import DisabledResearchProvider, ResearchService
from src.services.waking_inquiry_service import WakingInquiryService


async def _app_and_candidate(tmp_path, *, failed=False):
    path = tmp_path / "inquiries.sqlite3"
    store = InquiryCandidateStore(path)
    ledger = ResearchCalibrationLedger(path)
    await store.connect()
    await ledger.connect()
    drive = CognitiveResearchDrive(enabled=False, shadow_mode=True)
    waking = WakingInquiryService(
        store,
        drive,
        ResearchService(EscalationPolicy(research_enabled=False), DisabledResearchProvider()),
        ledger=ledger,
    )
    service = InquiryReviewService(store, waking, ledger)
    assessment = drive.assess(
        CognitiveResearchSignals(
            epistemic_uncertainty=0.95,
            temporal_volatility=0.9,
            expected_information_gain=0.95,
            metacognitive_gap=True,
        ),
        source="dream",
    )
    candidate, _ = await store.enqueue(
        InquiryCandidate(
            user_id=SYSTEM_USER_ID,
            question="What is the current verified state?",
            source_type=InquirySourceType.DREAM,
            assessment=assessment,
            priority=assessment.drive_score,
            expected_information_gain=assessment.signals.expected_information_gain,
        )
    )
    if failed:
        candidate = await store.transition(candidate.inquiry_id, SYSTEM_USER_ID, InquiryStatus.APPROVED)
        candidate = await store.transition(
            candidate.inquiry_id, SYSTEM_USER_ID, InquiryStatus.RESEARCH_FAILED
        )

    app = FastAPI()
    app.include_router(router)
    app.state.inquiry_review_service = service
    app.state.research_calibration_ledger = ledger
    app.dependency_overrides[get_api_key_user_id] = lambda: SYSTEM_USER_ID
    return app, candidate


@pytest.mark.asyncio
async def test_review_api_lists_inspects_and_approves_without_bypassing_shadow(tmp_path):
    app, candidate = await _app_and_candidate(tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listed = await client.get("/api/research/inquiries")
        inspected = await client.get(f"/api/research/inquiries/{candidate.inquiry_id}")
        approved = await client.post(
            f"/api/research/inquiries/{candidate.inquiry_id}/approve",
            json={"reason": "I approve checking external sources."},
        )
        ledger = await client.get(
            "/api/research/ledger", params={"inquiry_id": str(candidate.inquiry_id)}
        )

    assert listed.status_code == 200
    assert listed.json()["count"] == 1
    assert inspected.status_code == 200
    assert approved.status_code == 200
    assert approved.json()["disposition"] == "deferred"
    assert approved.json()["candidate"]["status"] == "queued"
    event_types = [event["event_type"] for event in ledger.json()["events"]]
    assert event_types == ["review_requested", "waking_revalidation", "review_resolved"]


@pytest.mark.asyncio
async def test_review_api_retries_failed_and_rejects_invalid_repeat(tmp_path):
    app, candidate = await _app_and_candidate(tmp_path, failed=True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        retried = await client.post(
            f"/api/research/inquiries/{candidate.inquiry_id}/retry",
            json={"reason": "New evidence makes a retry worthwhile."},
        )
        repeated = await client.post(
            f"/api/research/inquiries/{candidate.inquiry_id}/retry",
            json={"reason": "Retry again."},
        )

    assert retried.status_code == 200
    assert retried.json()["status"] == "queued"
    assert repeated.status_code == 409


@pytest.mark.asyncio
async def test_review_api_requires_authentication(tmp_path):
    app, _ = await _app_and_candidate(tmp_path)
    app.dependency_overrides.clear()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.get("/api/research/inquiries")
        invalid = await client.get(
            "/api/research/inquiries", headers={"X-API-Key": "invalid"}
        )

    assert missing.status_code == 422
    assert invalid.status_code == 401


@pytest.mark.asyncio
async def test_review_api_records_verified_source_feedback(tmp_path):
    app, candidate = await _app_and_candidate(tmp_path)
    store = app.state.inquiry_review_service.store
    ledger = app.state.research_calibration_ledger
    await store.transition(candidate.inquiry_id, SYSTEM_USER_ID, InquiryStatus.APPROVED)
    await store.transition(candidate.inquiry_id, SYSTEM_USER_ID, InquiryStatus.RESEARCHED)
    packet = ResearchPacket(
        request_id=uuid4(),
        decision_id=uuid4(),
        query=candidate.question,
        status=ResearchPacketStatus.COMPLETED,
        provider="grounded-test",
        grounding_verified=True,
        sources=[ResearchSource(source_id="s1", title="Primary", url="https://example.test")],
        claims=[ResearchClaim(text="Verified.", source_ids=["s1"], confidence=0.9)],
    )
    await ledger.record_packet(packet, user_id=SYSTEM_USER_ID, inquiry_id=candidate.inquiry_id)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/api/research/inquiries/{candidate.inquiry_id}/source-feedback",
            json={
                "request_id": str(packet.request_id),
                "source_id": "s1",
                "verdict": "trustworthy",
                "relevance": 5,
                "authority": 5,
                "freshness": 4,
                "citation_support": 5,
                "claim_supported": True,
                "research_changed_answer": True,
                "research_resolved_inquiry": True,
                "worth_cost": True,
            },
        )

    assert response.status_code == 201
    assert response.json()["event_type"] == "source_feedback"


@pytest.mark.asyncio
async def test_review_api_labels_real_shadow_observation_and_reports_summary(tmp_path):
    app, candidate = await _app_and_candidate(tmp_path)
    assessment = candidate.assessment
    await app.state.research_calibration_ledger.record_assessment(
        assessment,
        user_id=SYSTEM_USER_ID,
        cycle_id=uuid4(),
        event_type=ResearchLedgerEventType.SHADOW_ASSESSMENT,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        label = await client.post(
            f"/api/research/calibration/{assessment.assessment_id}/labels",
            json={
                "appropriate_action": CognitiveEffortAction.AUTHORIZE_RESEARCH.value,
                "should_external_research": True,
                "local_answer_sufficient": False,
                "rationale": "Current evidence was required.",
            },
        )
        summary = await client.get("/api/research/calibration/summary")

    assert label.status_code == 201
    assert summary.status_code == 200
    assert summary.json()["labeled_observations"] == 1
    assert summary.json()["automatic_non_explicit_research_eligible"] is False
