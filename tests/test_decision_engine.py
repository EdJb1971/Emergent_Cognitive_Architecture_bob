import asyncio
import pytest
from datetime import datetime, timedelta

from src.services.decision_engine import DecisionEngine, TriggerPolicy


class DummyQueue:
    def __init__(self):
        self.enqueued = []

    async def enqueue(self, name, payload):
        self.enqueued.append((name, payload))


@pytest.mark.asyncio
async def test_reflection_policy_triggers_after_cycles():
    queue = DummyQueue()
    engine = DecisionEngine(queue)

    user_id = "user-1"
    signals = {"cycles_since_reflection": 10}
    await engine.ingest_signals(user_id, signals)

    assert any(t[0] == "autonomous:reflection" for t in queue.enqueued)


@pytest.mark.asyncio
async def test_cooldown_prevents_immediate_retrigger():
    queue = DummyQueue()
    engine = DecisionEngine(queue)

    user_id = "user-2"
    signals = {"cycles_since_reflection": 10}
    await engine.ingest_signals(user_id, signals)
    # immediate second ingestion should not enqueue again due to cooldown
    await engine.ingest_signals(user_id, signals)

    enqueued = [t for t in queue.enqueued if t[0] == "autonomous:reflection"]
    assert len(enqueued) == 1


@pytest.mark.asyncio
async def test_discovery_on_low_satisfaction():
    queue = DummyQueue()
    engine = DecisionEngine(queue)

    user_id = "user-3"
    signals = {"avg_user_satisfaction": 0.2}
    await engine.ingest_signals(user_id, signals)

    assert any(t[0] == "autonomous:discovery" for t in queue.enqueued)
