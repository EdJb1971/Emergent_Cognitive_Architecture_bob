from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.models.agent_models import MemoryConsolidationJob
from src.models.research_models import InquirySourceType
from src.services.inquiry_candidate_service import InquiryCandidateService
from src.services.memory_consolidation_service import MemoryConsolidationService


@pytest.mark.asyncio
async def test_dream_consolidation_only_proposes_marked_unresolved_inquiries():
    inquiry_service = MagicMock(spec=InquiryCandidateService)
    inquiry_service.propose_offline = AsyncMock(return_value=None)
    service = MemoryConsolidationService(
        memory_service=MagicMock(),
        autobiographical_system=MagicMock(),
        llm_service=MagicMock(),
        inquiry_candidate_service=inquiry_service,
    )
    user_id = uuid4()
    cycle_id = uuid4()
    job = MemoryConsolidationJob(
        job_id="dream-1",
        user_id=str(user_id),
        cycle_ids_to_process=[str(cycle_id)],
        consolidation_type="pattern_extraction",
        priority=0.8,
        patterns_discovered=[
            "The user prefers concise answers.",
            "An unresolved contradiction is missing a causal explanation.",
        ],
    )

    await service._queue_dream_inquiries(job)

    inquiry_service.propose_offline.assert_awaited_once()
    call = inquiry_service.propose_offline.await_args.kwargs
    assert call["user_id"] == user_id
    assert call["source_type"] == InquirySourceType.DREAM
    assert call["source_cycle_ids"] == [cycle_id]
    assert call["question"] == "An unresolved contradiction is missing a causal explanation."
    assert not hasattr(service, "research_service")


@pytest.mark.asyncio
async def test_dream_handoff_failure_does_not_gain_a_provider_fallback():
    inquiry_service = MagicMock(spec=InquiryCandidateService)
    inquiry_service.propose_offline = AsyncMock(side_effect=RuntimeError("queue unavailable"))
    service = MemoryConsolidationService(
        memory_service=MagicMock(),
        autobiographical_system=MagicMock(),
        llm_service=MagicMock(),
        inquiry_candidate_service=inquiry_service,
    )
    job = MemoryConsolidationJob(
        job_id="dream-2",
        user_id=str(uuid4()),
        consolidation_type="pattern_extraction",
        priority=0.8,
        patterns_discovered=["A missing fact needs research."],
    )
    service.consolidation_jobs[job.job_id] = job
    service._extract_patterns = AsyncMock()
    service._queue_dream_inquiries = AsyncMock(side_effect=RuntimeError("queue unavailable"))

    result = await service.execute_consolidation_job(job.job_id)

    assert result.status == "completed"
    assert not hasattr(service, "research_service")
