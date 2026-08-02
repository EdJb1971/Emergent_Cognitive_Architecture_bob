import asyncio
from uuid import uuid4

import pytest

from src.models.autonomous_work_models import AutonomousEventType, AutonomousTaskType
from src.models.research_models import ResearchLedgerEventType
from src.models.predictive_models import PredictivePerceptionAssessment
from src.models.telemetry_models import TelemetryDomain
from src.services.autonomous_work_store import AutonomousWorkStore
from src.services.metrics_service import MetricType, MetricsService
from src.services.research_calibration_ledger import ResearchCalibrationLedger
from src.services.predictive_calibration_store import PredictiveCalibrationStore


async def make_metrics(monkeypatch, *, replay_size=32, queue_size=8):
    async def no_chroma_init(_self):
        return None

    async def no_chroma_save(_self, _event):
        return None

    monkeypatch.setattr(MetricsService, "_init_chroma", no_chroma_init)
    monkeypatch.setattr(MetricsService, "_save_event_to_db", no_chroma_save)
    service = MetricsService(
        telemetry_replay_size=replay_size,
        telemetry_subscriber_queue_size=queue_size,
    )
    await asyncio.sleep(0)
    return service


@pytest.mark.asyncio
async def test_typed_domains_and_cursor_replay(monkeypatch):
    metrics = await make_metrics(monkeypatch)
    user_id = str(uuid4())
    cycle_id = str(uuid4())

    await metrics.record_metric(
        MetricType.COGNITIVE_CYCLE,
        {"event": "cycle_started"},
        cycle_id=cycle_id,
        user_id=user_id,
    )
    await metrics.record_metric(
        MetricType.MEMORY_ACCESS,
        {"tier_accessed": "ltm", "hit_rate": 1.0},
        cycle_id=cycle_id,
        user_id=user_id,
    )
    await metrics.record_metric(
        MetricType.SALIENCE_ASSESSMENT,
        {"candidate_count": 4, "shadow_mode": True},
        cycle_id=cycle_id,
        user_id=user_id,
    )
    await metrics.record_metric(
        MetricType.SLEEP_CYCLE,
        {"status": "completed"},
        user_id=user_id,
    )

    subscription, gap = metrics.subscribe(
        after_sequence=1,
        domains=[TelemetryDomain.MEMORY, TelemetryDomain.SALIENCE],
        replay_limit=8,
    )
    assert gap is None
    first = await subscription.next_message()
    second = await subscription.next_message()
    assert [first["data"]["domain"], second["data"]["domain"]] == ["memory", "salience"]
    assert first["data"]["cycle_id"] == cycle_id
    assert metrics.telemetry_hello([TelemetryDomain.MEMORY]).latest_sequence == 4
    metrics.unsubscribe(subscription)


@pytest.mark.asyncio
async def test_slow_subscriber_gets_gap_without_blocking_publish(monkeypatch):
    metrics = await make_metrics(monkeypatch, queue_size=8)
    subscription, _ = metrics.subscribe(replay_limit=0)

    for index in range(12):
        await metrics.record_metric(
            MetricType.COGNITIVE_CYCLE,
            {"event": "cycle_progress", "index": index},
        )

    gap = await subscription.next_message()
    assert gap["type"] == "gap"
    assert gap["data"]["reason"] == "subscriber_backpressure"
    assert gap["data"]["dropped_for_subscriber"] == 4
    event = await subscription.next_message()
    assert event["data"]["sequence"] == 5


@pytest.mark.asyncio
async def test_old_cursor_reports_replay_window_gap(monkeypatch):
    metrics = await make_metrics(monkeypatch, replay_size=32)
    for index in range(40):
        await metrics.record_metric(MetricType.COGNITIVE_CYCLE, {"event": "tick", "index": index})

    subscription, gap = metrics.subscribe(after_sequence=2, replay_limit=8)
    assert gap is not None
    assert gap.reason == "cursor_older_than_replay_window"
    assert gap.available_from == 9
    assert (await subscription.next_message())["data"]["sequence"] == 33


@pytest.mark.asyncio
async def test_authoritative_ledgers_project_minimal_live_events(monkeypatch, tmp_path):
    metrics = await make_metrics(monkeypatch)
    user_id = uuid4()
    autonomous = AutonomousWorkStore(
        tmp_path / "autonomous.sqlite3",
        event_sink=metrics.record_autonomous_event,
    )
    research = ResearchCalibrationLedger(
        tmp_path / "research.sqlite3",
        event_sink=metrics.record_research_event,
    )
    await autonomous.connect()
    await research.connect()
    subscription, _ = metrics.subscribe(replay_limit=0)

    task_id = uuid4()
    await autonomous.append_event(
        AutonomousEventType.TASK_QUEUED,
        user_id=user_id,
        task_id=task_id,
        task_type=AutonomousTaskType.REFLECTION,
        payload={"reason": "test signal", "private_body": "must not be projected"},
    )
    inquiry_id = uuid4()
    await research.append(
        ResearchLedgerEventType.REVIEW_REQUESTED,
        user_id=user_id,
        inquiry_id=inquiry_id,
        payload={"question": "A deliberately unprojected inquiry body"},
    )

    autonomous_message = await subscription.next_message()
    research_message = await subscription.next_message()
    assert autonomous_message["data"]["domain"] == "autonomous_work"
    assert autonomous_message["data"]["source_reference"] == "autonomous_ledger:1"
    assert "private_body" not in autonomous_message["data"]["payload"]
    assert research_message["data"]["domain"] == "research"
    assert research_message["data"]["source_reference"] == "research_ledger:1"
    assert "question" not in research_message["data"]["payload"]
    await autonomous.close()
    await research.close()


@pytest.mark.asyncio
async def test_predictive_ledger_projects_content_free_typed_telemetry(monkeypatch, tmp_path):
    metrics = await make_metrics(monkeypatch)
    user_id = uuid4()
    assessment = PredictivePerceptionAssessment(
        assessment_id=uuid4(),
        cycle_id=uuid4(),
        sensory_episode_id=uuid4(),
        enabled=False,
        assessment_status="disabled",
        hypothesis_count=0,
        matched_count=0,
        mismatch_count=0,
        unobserved_count=0,
        low_reliability_count=0,
        material_error_count=0,
    )
    predictive = PredictiveCalibrationStore(
        tmp_path / "predictive.sqlite3",
        event_sink=metrics.record_predictive_event,
    )
    await predictive.connect()
    subscription, _ = metrics.subscribe(replay_limit=0)

    await predictive.record_assessment(assessment, user_id=user_id)

    message = await subscription.next_message()
    assert message["data"]["domain"] == "predictive"
    assert message["data"]["event_type"] == "assessment_recorded"
    assert message["data"]["source_reference"] == "predictive_ledger:1"
    assert message["data"]["payload"]["shadow_mode"] is True
    assert message["data"]["payload"]["predictive_influence_eligible"] is False
    assert "assessment" not in message["data"]["payload"]
    await predictive.close()
