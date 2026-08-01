import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from src.models.research_models import (
    CognitiveResearchSignals,
    InquiryCandidate,
    InquirySourceType,
    InquiryStatus,
)
from src.services.cognitive_research_drive import CognitiveResearchDrive
from src.services.inquiry_candidate_service import InquiryCandidateService
from src.services.inquiry_candidate_store import InquiryCandidateStore


def _assessment(*, shadow_mode: bool = True):
    drive = CognitiveResearchDrive(enabled=True, shadow_mode=shadow_mode)
    return drive.assess(
        CognitiveResearchSignals(
            epistemic_uncertainty=0.9,
            cognitive_conflict=0.8,
            novelty_prediction_error=0.8,
            task_stakes=0.8,
            expected_information_gain=0.9,
            metacognitive_gap=True,
        ),
        source="test",
    )


def _candidate(user_id, question="What unresolved mechanism explains the anomaly?"):
    return InquiryCandidate(
        user_id=user_id,
        question=question,
        source_type=InquirySourceType.DREAM,
        source_cycle_ids=[uuid4()],
        assessment=_assessment(),
        priority=0.9,
        expected_information_gain=0.9,
    )


@pytest.mark.asyncio
async def test_queue_persists_across_store_instances(tmp_path):
    path = tmp_path / "inquiries.sqlite3"
    user_id = uuid4()
    first = InquiryCandidateStore(path)
    await first.connect()
    stored, created = await first.enqueue(_candidate(user_id))
    await first.close()

    second = InquiryCandidateStore(path)
    await second.connect()
    loaded = await second.get(stored.inquiry_id, user_id)

    assert created is True
    assert loaded is not None
    assert loaded.question == stored.question
    assert loaded.source_type == InquirySourceType.DREAM


@pytest.mark.asyncio
async def test_queue_deduplicates_active_questions_and_merges_provenance(tmp_path):
    store = InquiryCandidateStore(tmp_path / "inquiries.sqlite3")
    await store.connect()
    user_id = uuid4()
    first = _candidate(user_id, "Why does this anomaly persist?")
    second = _candidate(user_id, "  WHY does this   anomaly persist?  ")

    stored_first, created_first = await store.enqueue(first)
    stored_second, created_second = await store.enqueue(second)
    candidates = await store.list_candidates(user_id)

    assert created_first is True
    assert created_second is False
    assert stored_second.inquiry_id == stored_first.inquiry_id
    assert len(stored_second.source_cycle_ids) == 2
    assert len(candidates) == 1


@pytest.mark.asyncio
async def test_queue_enforces_state_machine_transitions(tmp_path):
    store = InquiryCandidateStore(tmp_path / "inquiries.sqlite3")
    await store.connect()
    user_id = uuid4()
    candidate, _ = await store.enqueue(_candidate(user_id))

    approved = await store.transition(candidate.inquiry_id, user_id, InquiryStatus.APPROVED)
    researched = await store.transition(approved.inquiry_id, user_id, InquiryStatus.RESEARCHED)

    assert researched.status == InquiryStatus.RESEARCHED
    with pytest.raises(ValueError, match="cannot transition"):
        await store.transition(researched.inquiry_id, user_id, InquiryStatus.QUEUED)


@pytest.mark.asyncio
async def test_claim_next_is_atomic_across_store_instances(tmp_path):
    path = tmp_path / "inquiries.sqlite3"
    first_store = InquiryCandidateStore(path)
    second_store = InquiryCandidateStore(path)
    await first_store.connect()
    await second_store.connect()
    user_id = uuid4()
    queued, _ = await first_store.enqueue(_candidate(user_id))

    claims = await asyncio.gather(
        first_store.claim_next(user_id),
        second_store.claim_next(user_id),
    )

    claimed = [candidate for candidate in claims if candidate is not None]
    assert len(claimed) == 1
    assert claimed[0].inquiry_id == queued.inquiry_id
    assert claimed[0].status == InquiryStatus.UNDER_REVIEW


@pytest.mark.asyncio
async def test_queue_expires_stale_open_candidates(tmp_path):
    store = InquiryCandidateStore(tmp_path / "inquiries.sqlite3")
    await store.connect()
    user_id = uuid4()
    old = _candidate(user_id).model_copy(
        update={"expires_at": datetime(2026, 8, 1, tzinfo=timezone.utc)}
    )
    stored, _ = await store.enqueue(old)

    count = await store.expire_due(now=datetime(2026, 8, 2, tzinfo=timezone.utc))
    loaded = await store.get(stored.inquiry_id, user_id)

    assert count == 1
    assert loaded.status == InquiryStatus.EXPIRED


@pytest.mark.asyncio
async def test_new_evidence_requeues_a_failed_duplicate(tmp_path):
    store = InquiryCandidateStore(tmp_path / "inquiries.sqlite3")
    await store.connect()
    user_id = uuid4()
    candidate, _ = await store.enqueue(_candidate(user_id))
    await store.transition(candidate.inquiry_id, user_id, InquiryStatus.APPROVED)
    await store.transition(candidate.inquiry_id, user_id, InquiryStatus.RESEARCH_FAILED)

    retried, created = await store.enqueue(_candidate(user_id))

    assert created is False
    assert retried.inquiry_id == candidate.inquiry_id
    assert retried.status == InquiryStatus.QUEUED


@pytest.mark.asyncio
async def test_offline_service_queues_dream_candidate_without_provider_capability(tmp_path):
    store = InquiryCandidateStore(tmp_path / "inquiries.sqlite3")
    await store.connect()
    drive = CognitiveResearchDrive(enabled=False, shadow_mode=True)
    service = InquiryCandidateService(store, drive, enabled=True, ttl_days=7)
    user_id = uuid4()
    source_cycle = uuid4()

    candidate = await service.propose_offline(
        user_id=user_id,
        question="What missing mechanism explains the recurring anomaly?",
        source_type=InquirySourceType.DREAM,
        source_cycle_ids=[source_cycle],
        signals=CognitiveResearchSignals(
            epistemic_uncertainty=0.9,
            cognitive_conflict=0.8,
            novelty_prediction_error=0.9,
            task_stakes=0.8,
            persistence_after_local_attempts=0.7,
            expected_information_gain=0.95,
            metacognitive_gap=True,
        ),
    )

    assert candidate is not None
    assert candidate.source_type == InquirySourceType.DREAM
    assert candidate.shadow_mode is True
    assert candidate.source_cycle_ids == [source_cycle]
    assert candidate.expires_at > datetime.now(timezone.utc) + timedelta(days=6)


@pytest.mark.asyncio
async def test_low_value_offline_curiosity_is_not_queued(tmp_path):
    store = InquiryCandidateStore(tmp_path / "inquiries.sqlite3")
    await store.connect()
    service = InquiryCandidateService(store, CognitiveResearchDrive(), enabled=True)

    candidate = await service.propose_offline(
        user_id=uuid4(),
        question="A low-value passing curiosity",
        source_type=InquirySourceType.REFLECTION,
        signals=CognitiveResearchSignals(
            epistemic_uncertainty=0.2,
            novelty_prediction_error=0.2,
            expected_information_gain=0.2,
        ),
    )

    assert candidate is None
