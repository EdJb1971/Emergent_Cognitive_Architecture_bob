from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.models.agent_models import MemoryConsolidationJob
from src.models.core_models import CognitiveCycle
from src.models.sleep_models import SleepLedgerEventType
from src.services.memory_consolidation_service import MemoryConsolidationService
from src.services.sleep_cycle_ledger import SleepCycleLedger


def _cycle(user_id, *, priority=0.8, metadata=None):
    values = {
        "consolidation_metadata": {"consolidation_priority": priority},
        **(metadata or {}),
    }
    return CognitiveCycle(
        user_id=user_id,
        user_input="Remember this",
        final_response="I will remember it",
        metadata=values,
    )


@pytest.mark.asyncio
async def test_candidate_selection_excludes_only_the_completed_stage():
    user_id = uuid4()
    first = _cycle(
        user_id,
        metadata={"sleep_consolidation": {"episodic_to_semantic": {"job_id": "old"}}},
    )
    second = _cycle(user_id)
    memory = MagicMock()
    memory.get_user_cycles = AsyncMock(return_value=[first, second])
    service = MemoryConsolidationService(memory, MagicMock(), MagicMock())

    episodic, _ = await service.get_consolidation_candidates(
        str(user_id), consolidation_type="episodic_to_semantic"
    )
    replay, _ = await service.get_consolidation_candidates(
        str(user_id), consolidation_type="memory_replay"
    )

    assert episodic == [str(second.cycle_id)]
    assert replay == [str(first.cycle_id), str(second.cycle_id)]


@pytest.mark.asyncio
async def test_replay_uses_user_scoped_retrieval_patches_metadata_and_audits(tmp_path):
    user_id = uuid4()
    cycle = _cycle(user_id)
    memory = MagicMock()
    memory.get_cycle_by_id = AsyncMock(return_value=cycle)
    memory.patch_cycle_metadata = AsyncMock(return_value=cycle)
    ledger = SleepCycleLedger(tmp_path / "sleep.sqlite3")
    await ledger.connect()
    service = MemoryConsolidationService(
        memory,
        MagicMock(),
        MagicMock(),
        audit_ledger=ledger,
    )
    run_id = uuid4()
    job = await service.create_consolidation_job(
        user_id=str(user_id),
        consolidation_type="memory_replay",
        cycle_ids=[str(cycle.cycle_id)],
        run_id=run_id,
    )

    result = await service.execute_consolidation_job(job.job_id, run_id=run_id)

    assert result.status == "completed"
    memory.get_cycle_by_id.assert_awaited_once_with(user_id, cycle.cycle_id)
    replay_patch = memory.patch_cycle_metadata.await_args_list[0].args[2]
    assert replay_patch["consolidation_metadata"]["replay_count"] == 1
    assert memory.patch_cycle_metadata.await_count == 2
    assert str(user_id) not in service.last_consolidation
    events = await ledger.list_events(user_id, run_id=run_id)
    assert [event.event_type for event in events] == [
        SleepLedgerEventType.JOB_CREATED,
        SleepLedgerEventType.JOB_STARTED,
        SleepLedgerEventType.JOB_COMPLETED,
    ]
    assert events[1].payload["job"]["status"] == "processing"


@pytest.mark.asyncio
async def test_missing_source_cycle_fails_job_without_completion_marker(tmp_path):
    user_id = uuid4()
    cycle_id = uuid4()
    memory = MagicMock()
    memory.get_cycle_by_id = AsyncMock(return_value=None)
    memory.patch_cycle_metadata = AsyncMock()
    ledger = SleepCycleLedger(tmp_path / "sleep.sqlite3")
    await ledger.connect()
    service = MemoryConsolidationService(
        memory,
        MagicMock(),
        MagicMock(),
        audit_ledger=ledger,
    )
    job = await service.create_consolidation_job(
        user_id=str(user_id),
        consolidation_type="episodic_to_semantic",
        cycle_ids=[str(cycle_id)],
    )

    result = await service.execute_consolidation_job(job.job_id)

    assert result.status == "failed"
    assert "not found" in result.error_message
    memory.patch_cycle_metadata.assert_not_awaited()
    events = await ledger.list_events(user_id)
    assert events[-1].event_type == SleepLedgerEventType.JOB_FAILED
