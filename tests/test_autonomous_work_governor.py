import asyncio
import sqlite3
from uuid import uuid4

import pytest

from src.models.autonomous_work_models import (
    AutonomousProviderPolicy,
    AutonomousRuntimeUpdate,
    AutonomousTaskPolicy,
    AutonomousTaskRequest,
    AutonomousTaskStatus,
    AutonomousTaskType,
)
from src.services.autonomous_work_governor import AutonomousWorkGovernor
from src.services.autonomous_work_store import AutonomousWorkStore


def policies(*, enabled=True, retries=0, cancel_on_activity=True):
    return {
        task_type: AutonomousTaskPolicy(
            task_type=task_type,
            enabled=enabled,
            cooldown_seconds=0,
            timeout_seconds=1,
            max_retries=retries,
            max_per_hour=100,
            provider_policy=AutonomousProviderPolicy.LOCAL_ONLY,
            cancel_on_user_activity=cancel_on_activity,
        )
        for task_type in AutonomousTaskType
    }


async def make_governor(tmp_path, **kwargs):
    store = AutonomousWorkStore(tmp_path / "autonomous.sqlite3")
    await store.connect()
    governor = AutonomousWorkGovernor(
        store=store,
        policies=kwargs.pop("policies", policies()),
        master_enabled=kwargs.pop("master_enabled", True),
        max_concurrent_global=kwargs.pop("max_concurrent_global", 1),
        provider_is_local=kwargs.pop("provider_is_local", lambda: True),
    )
    system_user = uuid4()
    await governor.start(system_user)
    return governor, store, system_user


def request(user_id, task_type=AutonomousTaskType.REFLECTION, key="one"):
    return AutonomousTaskRequest(
        user_id=user_id,
        task_type=task_type,
        trigger_reason="deterministic test signal",
        deduplication_key=key,
    )


@pytest.mark.asyncio
async def test_disabled_category_rejects_and_audits(tmp_path):
    configured = policies()
    configured[AutonomousTaskType.REFLECTION].enabled = False
    governor, store, system_user = await make_governor(tmp_path, policies=configured)
    called = False

    async def handler(_):
        nonlocal called
        called = True

    record = await governor.submit(request(system_user), executor=handler)

    assert record.status == AutonomousTaskStatus.REJECTED
    assert record.rejection_reason == "category_disabled"
    assert called is False
    events = await store.list_events(system_user)
    assert events[-1].event_type.value == "task_rejected"
    assert await store.verify_integrity() is True
    await governor.shutdown(system_user)


@pytest.mark.asyncio
async def test_duplicate_returns_original_and_executes_once(tmp_path):
    governor, store, system_user = await make_governor(tmp_path)
    gate = asyncio.Event()
    calls = 0

    async def handler(_):
        nonlocal calls
        calls += 1
        await gate.wait()

    first = await governor.submit(request(system_user), executor=handler)
    await asyncio.sleep(0)
    second = await governor.submit(request(system_user), executor=handler)
    gate.set()
    await asyncio.sleep(0.05)

    assert second.request.task_id == first.request.task_id
    assert calls == 1
    assert any(event.event_type.value == "task_duplicate" for event in await store.list_events(system_user))
    await governor.shutdown(system_user)


@pytest.mark.asyncio
async def test_failure_retries_within_budget(tmp_path):
    governor, store, system_user = await make_governor(tmp_path, policies=policies(retries=1))
    attempts = 0

    async def flaky(_):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")
        return {"recovered": True}

    record = await governor.submit(request(system_user), executor=flaky, wait=True)

    assert record.status == AutonomousTaskStatus.COMPLETED
    assert record.attempt == 2
    assert record.result == {"recovered": True}
    assert attempts == 2
    await governor.shutdown(system_user)


@pytest.mark.asyncio
async def test_waking_activity_preempts_interruptible_work(tmp_path):
    governor, store, system_user = await make_governor(tmp_path)
    started = asyncio.Event()

    async def long_running(_):
        started.set()
        await asyncio.Event().wait()

    queued = await governor.submit(request(system_user), executor=long_running)
    await started.wait()
    assert await governor.note_activity(system_user) == 1
    await asyncio.sleep(0.05)
    record = await store.get_task(queued.request.task_id)

    assert record.status == AutonomousTaskStatus.CANCELLED
    await governor.shutdown(system_user)


@pytest.mark.asyncio
async def test_runtime_toggles_persist_across_governor_restart(tmp_path):
    governor, store, system_user = await make_governor(tmp_path)
    state = await governor.update_runtime(
        system_user,
        AutonomousRuntimeUpdate(
            master_enabled=False,
            category_enabled={AutonomousTaskType.CURIOSITY: False},
            reason="operator calibration pause",
        ),
    )
    assert state.master_enabled is False
    await governor.shutdown(system_user)

    restored = AutonomousWorkGovernor(store=store, policies=policies(), provider_is_local=lambda: True)
    await restored.start(system_user)
    assert restored.master_enabled is False
    assert restored.policies[AutonomousTaskType.CURIOSITY].enabled is False
    await restored.shutdown(system_user)


@pytest.mark.asyncio
async def test_local_only_policy_rejects_nonlocal_provider(tmp_path):
    governor, _, system_user = await make_governor(tmp_path, provider_is_local=lambda: False)
    record = await governor.submit(request(system_user), executor=lambda _: asyncio.sleep(0))
    assert record.status == AutonomousTaskStatus.REJECTED
    assert record.rejection_reason == "local_provider_required"
    await governor.shutdown(system_user)


@pytest.mark.asyncio
async def test_ledger_update_and_delete_are_database_rejected(tmp_path):
    governor, store, system_user = await make_governor(tmp_path)
    await governor.submit(request(system_user), executor=lambda _: asyncio.sleep(0), wait=True)

    with sqlite3.connect(store.path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE autonomous_work_ledger SET payload='{}' WHERE sequence=1")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM autonomous_work_ledger WHERE sequence=1")
    await governor.shutdown(system_user)

