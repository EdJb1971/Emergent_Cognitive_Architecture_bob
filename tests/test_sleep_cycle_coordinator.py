import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.models.agent_models import MemoryConsolidationJob
from src.models.sleep_models import SleepLedgerEventType
from src.services.sleep_cycle_coordinator import SleepCycleCoordinator
from src.services.sleep_cycle_ledger import SleepCycleLedger


def _service(*, local: bool = True):
    service = MagicMock()
    service.llm_service.capabilities = SimpleNamespace(is_local=local)
    service.should_consolidate = AsyncMock(return_value=True)
    service.get_consolidation_candidates = AsyncMock()
    service.record_consolidation_completed = MagicMock()
    return service


@pytest.mark.asyncio
async def test_disabled_coordinator_creates_no_scheduler_or_ledger_event(tmp_path):
    user_id = uuid4()
    ledger = SleepCycleLedger(tmp_path / "sleep.sqlite3")
    await ledger.connect()
    coordinator = SleepCycleCoordinator(
        consolidation_service=_service(),
        ledger=ledger,
        user_ids=(user_id,),
        enabled=False,
    )

    assert await coordinator.start() is False
    assert coordinator.running is False
    assert await ledger.list_events(user_id) == []


@pytest.mark.asyncio
async def test_enabled_coordinator_has_one_idempotent_owned_lifecycle(tmp_path):
    user_id = uuid4()
    ledger = SleepCycleLedger(tmp_path / "sleep.sqlite3")
    await ledger.connect()
    coordinator = SleepCycleCoordinator(
        consolidation_service=_service(),
        ledger=ledger,
        user_ids=(user_id,),
        enabled=True,
        check_interval_seconds=60,
    )

    assert await coordinator.start() is True
    assert await coordinator.start() is False
    assert coordinator.running is True
    await coordinator.shutdown()
    await coordinator.shutdown()

    assert coordinator.running is False
    events = await ledger.list_events(user_id)
    assert [event.event_type for event in events] == [
        SleepLedgerEventType.COORDINATOR_STARTED,
        SleepLedgerEventType.COORDINATOR_STOPPED,
    ]


@pytest.mark.asyncio
async def test_activity_during_async_admission_prevents_sleep_start(tmp_path):
    user_id = uuid4()
    service = _service()
    admission_started = asyncio.Event()
    release_admission = asyncio.Event()

    async def delayed_admission(_user_id):
        admission_started.set()
        await release_admission.wait()
        return True

    service.should_consolidate = AsyncMock(side_effect=delayed_admission)
    ledger = SleepCycleLedger(tmp_path / "sleep.sqlite3")
    await ledger.connect()
    coordinator = SleepCycleCoordinator(
        consolidation_service=service,
        ledger=ledger,
        user_ids=(user_id,),
        enabled=True,
        idle_seconds=0,
    )

    pending = asyncio.create_task(coordinator.run_once(user_id))
    await asyncio.wait_for(admission_started.wait(), timeout=1)
    coordinator.note_activity(user_id)
    release_admission.set()
    result = await asyncio.wait_for(pending, timeout=1)

    assert result == {"status": "skipped", "reason": "user_activity", "idle_seconds": 0.0}
    service.get_consolidation_candidates.assert_not_awaited()
    assert await ledger.list_events(user_id) == []


@pytest.mark.asyncio
async def test_sleep_pipeline_runs_each_incomplete_stage_and_starts_cooldown_once(tmp_path):
    user_id = uuid4()
    cycle_id = str(uuid4())
    service = _service()
    service.get_consolidation_candidates.side_effect = [
        ([cycle_id], {"enabled": True}),
        ([cycle_id], None),
        ([], None),
    ]

    created_jobs = []

    async def create_job(**kwargs):
        job = MemoryConsolidationJob(
            job_id=str(uuid4()),
            user_id=kwargs["user_id"],
            run_id=str(kwargs["run_id"]),
            cycle_ids_to_process=kwargs["cycle_ids"],
            consolidation_type=kwargs["consolidation_type"],
            priority=kwargs["priority"],
            salience_advisory=kwargs["salience_advisory"],
        )
        created_jobs.append(job)
        return job

    async def execute_job(job_id, **_kwargs):
        job = next(item for item in created_jobs if item.job_id == job_id)
        job.status = "completed"
        return job

    service.create_consolidation_job = AsyncMock(side_effect=create_job)
    service.execute_consolidation_job = AsyncMock(side_effect=execute_job)
    ledger = SleepCycleLedger(tmp_path / "sleep.sqlite3")
    await ledger.connect()
    coordinator = SleepCycleCoordinator(
        consolidation_service=service,
        ledger=ledger,
        user_ids=(user_id,),
        enabled=True,
        idle_seconds=10,
    )
    coordinator.note_activity(user_id, at_monotonic=0)

    result = await coordinator.run_once(user_id, now_monotonic=20)

    assert result["status"] == "completed"
    assert [job.consolidation_type for job in created_jobs] == [
        "episodic_to_semantic",
        "memory_replay",
    ]
    service.record_consolidation_completed.assert_called_once_with(str(user_id))
    events = await ledger.list_events(user_id)
    assert [event.event_type for event in events] == [
        SleepLedgerEventType.CYCLE_STARTED,
        SleepLedgerEventType.CYCLE_COMPLETED,
    ]


@pytest.mark.asyncio
async def test_activity_cancels_active_sleep_work_and_audits_it(tmp_path):
    user_id = uuid4()
    cycle_id = str(uuid4())
    service = _service()
    service.get_consolidation_candidates.side_effect = [
        ([cycle_id], None),
        ([], None),
        ([], None),
    ]
    job = MemoryConsolidationJob(
        job_id=str(uuid4()),
        user_id=str(user_id),
        cycle_ids_to_process=[cycle_id],
        consolidation_type="episodic_to_semantic",
        priority=0.7,
    )
    service.create_consolidation_job = AsyncMock(return_value=job)
    entered = asyncio.Event()

    async def block_job(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    service.execute_consolidation_job = AsyncMock(side_effect=block_job)
    ledger = SleepCycleLedger(tmp_path / "sleep.sqlite3")
    await ledger.connect()
    coordinator = SleepCycleCoordinator(
        consolidation_service=service,
        ledger=ledger,
        user_ids=(user_id,),
        enabled=True,
        idle_seconds=0,
    )

    running = asyncio.create_task(coordinator.run_once(user_id))
    await asyncio.wait_for(entered.wait(), timeout=1)
    coordinator.note_activity(user_id)
    result = await asyncio.wait_for(running, timeout=1)

    assert result["status"] == "cancelled"
    events = await ledger.list_events(user_id)
    assert events[-1].event_type == SleepLedgerEventType.CYCLE_CANCELLED
    service.record_consolidation_completed.assert_not_called()


@pytest.mark.asyncio
async def test_non_local_provider_is_rejected_before_candidate_access(tmp_path):
    user_id = uuid4()
    service = _service(local=False)
    ledger = SleepCycleLedger(tmp_path / "sleep.sqlite3")
    await ledger.connect()
    coordinator = SleepCycleCoordinator(
        consolidation_service=service,
        ledger=ledger,
        user_ids=(user_id,),
        enabled=True,
        idle_seconds=0,
        require_local_provider=True,
    )

    result = await coordinator.run_once(user_id)

    assert result["reason"] == "non_local_provider"
    service.get_consolidation_candidates.assert_not_awaited()
