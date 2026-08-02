from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from src.api.settings import router
from src.dependencies import get_api_key_user_id
from src.models.identity_models import IdentityUpdateRequest
from src.services.clean_start_service import CLEAN_START_CONFIRMATION, CleanStartService
from src.services.identity_service import IdentityConflictError, IdentityService


@pytest.mark.asyncio
async def test_identity_is_durable_cached_and_tracks_former_names(tmp_path):
    path = tmp_path / "runtime_data" / "identity.json"
    service = IdentityService(path, "Bob")
    initial = await service.connect()
    assert initial.assistant_name == "Bob"
    assert initial.user_name is None

    updated = await service.update(
        IdentityUpdateRequest(
            assistant_name="  Ada  ",
            user_name="  Ed  ",
            expected_revision=initial.revision,
        )
    )
    assert updated.assistant_name == "Ada"
    assert updated.user_name == "Ed"
    assert updated.assistant_aliases == ("Bob",)
    assert "Your current name: Ada" in service.prompt_context()

    reloaded = IdentityService(path, "Ignored default")
    assert (await reloaded.connect()) == updated
    with pytest.raises(IdentityConflictError):
        await reloaded.update(
            IdentityUpdateRequest(
                assistant_name="Stale",
                user_name=None,
                expected_revision=1,
            )
        )


def test_clean_start_requires_exact_confirmation_and_consumes_at_restart(tmp_path):
    data = tmp_path / "chroma_db"
    data.mkdir()
    (data / "chroma.sqlite3").write_text("development memory", encoding="utf-8")
    identity = tmp_path / "runtime_data" / "identity.json"
    identity.parent.mkdir()
    identity.write_text('{"assistant_name":"Ada"}', encoding="utf-8")
    service = CleanStartService(tmp_path / "runtime_data" / "reset.json", data)

    with pytest.raises(ValueError):
        service.arm(confirmation="reset", preserve_identity=True)
    armed = service.arm(
        confirmation=CLEAN_START_CONFIRMATION,
        preserve_identity=True,
    )
    assert armed.pending_restart is True
    assert service.consume_before_startup(identity_path=identity) is True
    assert list(data.iterdir()) == []
    assert identity.exists()
    assert service.status().pending_restart is False


@pytest.mark.asyncio
async def test_settings_api_updates_identity_and_gates_clean_start(tmp_path):
    identity = IdentityService(tmp_path / "runtime_data" / "identity.json", "Bob")
    await identity.connect()
    clean_start = CleanStartService(
        tmp_path / "runtime_data" / "reset.json",
        tmp_path / "chroma_db",
    )
    app = FastAPI()
    app.include_router(router)
    app.state.identity_service = identity
    app.state.clean_start_service = clean_start
    app.dependency_overrides[get_api_key_user_id] = lambda: uuid4()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        current = await client.get("/api/settings/identity")
        updated = await client.put(
            "/api/settings/identity",
            json={"assistant_name": "Nova", "user_name": "", "expected_revision": 1},
        )
        rejected = await client.post(
            "/api/settings/clean-start",
            json={"confirmation": "almost", "preserve_identity": True},
        )
        armed = await client.post(
            "/api/settings/clean-start",
            json={"confirmation": CLEAN_START_CONFIRMATION, "preserve_identity": True},
        )
        cancelled = await client.delete("/api/settings/clean-start")

    assert current.json()["assistant_name"] == "Bob"
    assert updated.json()["assistant_name"] == "Nova"
    assert updated.json()["user_name"] is None
    assert rejected.status_code == 400
    assert armed.json()["pending_restart"] is True
    assert cancelled.json()["pending_restart"] is False
