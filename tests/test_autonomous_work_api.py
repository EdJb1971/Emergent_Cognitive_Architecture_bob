import httpx
import pytest
from fastapi import FastAPI

from src.api.autonomous_work import router
from src.dependencies import SYSTEM_USER_ID, get_api_key_user_id
from src.models.autonomous_work_models import (
    AutonomousTaskPolicy,
    AutonomousTaskType,
)
from src.services.autonomous_work_governor import AutonomousWorkGovernor
from src.services.autonomous_work_store import AutonomousWorkStore


@pytest.mark.asyncio
async def test_operator_api_toggles_lists_actions_and_verifies_ledger(tmp_path):
    store = AutonomousWorkStore(tmp_path / "work.sqlite3")
    await store.connect()
    policies = {
        task_type: AutonomousTaskPolicy(
            task_type=task_type,
            enabled=task_type in {AutonomousTaskType.SUMMARY_UPDATE, AutonomousTaskType.STM_FLUSH},
            timeout_seconds=1,
        )
        for task_type in AutonomousTaskType
    }
    governor = AutonomousWorkGovernor(store=store, policies=policies)
    await governor.start(SYSTEM_USER_ID)
    app = FastAPI()
    app.include_router(router)
    app.state.autonomous_work_governor = governor
    app.dependency_overrides[get_api_key_user_id] = lambda: SYSTEM_USER_ID

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        initial = await client.get("/api/autonomous-work/runtime")
        changed = await client.put(
            "/api/autonomous-work/runtime",
            json={
                "category_enabled": {"reflection": True, "sleep": True},
                "reason": "operator enabled calibrated categories",
            },
        )
        tasks = await client.get("/api/autonomous-work/tasks")
        ledger = await client.get("/api/autonomous-work/ledger")

    assert initial.status_code == 200
    assert initial.json()["policies"]["reflection"]["enabled"] is False
    assert changed.status_code == 200
    assert changed.json()["policies"]["reflection"]["enabled"] is True
    assert changed.json()["policies"]["sleep"]["enabled"] is True
    assert tasks.status_code == 200 and tasks.json()["count"] == 0
    assert ledger.status_code == 200
    assert ledger.json()["integrity_verified"] is True
    assert any(event["event_type"] == "runtime_changed" for event in ledger.json()["events"])
    await governor.shutdown(SYSTEM_USER_ID)
