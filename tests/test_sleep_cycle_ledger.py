import sqlite3
from uuid import uuid4

import pytest

from src.models.sleep_models import SleepLedgerEventType
from src.services.sleep_cycle_ledger import SleepCycleLedger


@pytest.mark.asyncio
async def test_sleep_ledger_is_persistent_hash_chained_and_append_only(tmp_path):
    path = tmp_path / "sleep.sqlite3"
    user_id = uuid4()
    run_id = uuid4()
    ledger = SleepCycleLedger(path)
    await ledger.connect()

    first = await ledger.append(
        SleepLedgerEventType.CYCLE_STARTED,
        user_id=user_id,
        run_id=run_id,
        payload={"cycles": 3},
    )
    second = await ledger.append(
        SleepLedgerEventType.CYCLE_COMPLETED,
        user_id=user_id,
        run_id=run_id,
        payload={"status": "completed"},
    )

    assert second.previous_hash == first.event_hash
    assert await ledger.verify_integrity() is True
    await ledger.close()

    reopened = SleepCycleLedger(path)
    await reopened.connect()
    events = await reopened.list_events(user_id, run_id=run_id)
    assert [event.event_type for event in events] == [
        SleepLedgerEventType.CYCLE_STARTED,
        SleepLedgerEventType.CYCLE_COMPLETED,
    ]

    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE sleep_cycle_ledger SET payload = '{}' WHERE sequence = 1"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM sleep_cycle_ledger WHERE sequence = 1")


@pytest.mark.asyncio
async def test_sleep_ledger_validates_pagination(tmp_path):
    ledger = SleepCycleLedger(tmp_path / "sleep.sqlite3")
    await ledger.connect()

    with pytest.raises(ValueError, match="between 1 and 500"):
        await ledger.list_events(uuid4(), limit=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        await ledger.list_events(uuid4(), after_sequence=-1)
