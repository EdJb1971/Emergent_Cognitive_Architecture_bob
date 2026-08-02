import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import main
from src.core.config import settings
from src.services.metrics_service import MetricType, MetricsService


async def build_metrics(monkeypatch):
    async def no_chroma_init(_self):
        return None

    async def no_chroma_save(_self, _event):
        return None

    monkeypatch.setattr(MetricsService, "_init_chroma", no_chroma_init)
    monkeypatch.setattr(MetricsService, "_save_event_to_db", no_chroma_save)
    metrics = MetricsService(telemetry_replay_size=32, telemetry_subscriber_queue_size=8)
    await metrics.record_metric(MetricType.COGNITIVE_CYCLE, {"event": "cycle_started"})
    return metrics


def test_websocket_is_authenticated_and_streams_typed_replay(monkeypatch):
    metrics = asyncio.run(build_metrics(monkeypatch))
    main.app.state.metrics_service = metrics
    client = TestClient(main.app)

    with pytest.raises(WebSocketDisconnect) as rejected:
        with client.websocket_connect("/ws/dashboard?replay=0"):
            pass
    assert rejected.value.code == 4401

    with client.websocket_connect(
        "/ws/dashboard?after=0&replay=8&domains=cognitive",
        headers={settings.API_KEY_HEADER_NAME: settings.API_KEY},
        subprotocols=["eca.telemetry.v1"],
    ) as websocket:
        assert websocket.accepted_subprotocol == "eca.telemetry.v1"
        hello = websocket.receive_json()
        snapshot = websocket.receive_json()
        event = websocket.receive_json()

    assert hello["type"] == "hello"
    assert hello["data"]["schema_version"] == 1
    assert snapshot["type"] == "snapshot"
    assert event["type"] == "event"
    assert event["data"]["domain"] == "cognitive"
    assert event["data"]["event_type"] == "cycle_started"

