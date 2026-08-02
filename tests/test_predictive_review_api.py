from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from src.api.predictive_review import router
from src.dependencies import SYSTEM_USER_ID, get_api_key_user_id
from src.models.core_models import CognitiveCycle
from src.models.multimodal_models import VisualAnalysis, VisualEvidence
from src.services.multisensory_binding_service import MultisensoryBindingService
from src.services.predictive_calibration_store import PredictiveCalibrationStore
from src.services.predictive_perception_service import PredictivePerceptionService


def _make_assessment():
    now = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)
    prior = CognitiveCycle(
        user_id=SYSTEM_USER_ID,
        session_id=uuid4(),
        user_input="The car is red.",
        final_response="Recorded.",
    )
    evidence = VisualEvidence(
        provider="ollama",
        model="vision-test",
        mime_type="image/png",
        byte_count=100,
        width=640,
        height=480,
        input_quality_score=0.9,
        sha256="a" * 64,
        observed_at=now,
        analysis=VisualAnalysis(
            description="A blue car",
            scene_description="A blue car",
            objects_detected=["car"],
            confidence=0.9,
        ),
    )
    cycle_id = uuid4()
    episode = MultisensoryBindingService().bind_turn(
        cycle_id=cycle_id,
        user_id=SYSTEM_USER_ID,
        session_id=uuid4(),
        request_timestamp=now,
        text="What is visible?",
        visual_evidence=evidence,
    )
    assessment = PredictivePerceptionService().assess(
        cycle_id=cycle_id,
        sensory_episode=episode,
        prior_cycles=[prior],
        current_text="What is visible?",
        visual_evidence=evidence,
    )
    error = next(
        item for item in assessment.prediction_errors
        if item.feature_name == "colour:car"
    )
    return assessment, error


async def _make_app(tmp_path, *, authenticated=True):
    store = PredictiveCalibrationStore(tmp_path / "predictive.sqlite3")
    await store.connect()
    assessment, error = _make_assessment()
    await store.record_assessment(assessment, user_id=SYSTEM_USER_ID)
    app = FastAPI()
    app.include_router(router)
    app.state.predictive_calibration_store = store
    if authenticated:
        app.dependency_overrides[get_api_key_user_id] = lambda: SYSTEM_USER_ID
    return app, assessment, error


@pytest.mark.asyncio
async def test_predictive_api_lists_inspects_labels_and_reports_calibration(tmp_path):
    app, assessment, error = await _make_app(tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listed = await client.get(
            "/api/predictive/assessments",
            params={"review_status": "unreviewed", "material_only": True},
        )
        inspected = await client.get(
            f"/api/predictive/assessments/{assessment.assessment_id}"
        )
        labeled = await client.post(
            f"/api/predictive/assessments/{assessment.assessment_id}/labels",
            json={
                "error_id": str(error.error_id),
                "hypothesis_verdict": "incorrect",
                "observation_quality": "reliable",
                "prediction_outcome": "confirmed_mismatch",
                "recommendation_verdict": "useful",
                "preferred_action": "ask_user",
                "rationale": "The current blue observation was reliable.",
            },
        )
        summary = await client.get("/api/predictive/calibration/summary")
        ledger = await client.get("/api/predictive/ledger")

    assert listed.status_code == 200
    assert listed.json()["count"] == 1
    assert inspected.status_code == 200
    assert inspected.json()["assessment"]["primary_evidence_rewritten"] is False
    assert labeled.status_code == 201
    assert labeled.json()["event_type"] == "calibration_label"
    assert summary.status_code == 200
    assert summary.json()["labeled_errors"] == 1
    assert summary.json()["predictive_influence_eligible"] is False
    assert summary.json()["ledger_integrity_verified"] is True
    assert ledger.status_code == 200
    assert [item["event_type"] for item in ledger.json()["events"]] == [
        "assessment_recorded",
        "calibration_label",
    ]


@pytest.mark.asyncio
async def test_predictive_api_enforces_auth_and_target_ownership(tmp_path):
    app, assessment, _error = await _make_app(tmp_path, authenticated=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.get("/api/predictive/assessments")
        invalid = await client.get(
            "/api/predictive/assessments",
            headers={"X-API-Key": "invalid"},
        )
    app.dependency_overrides[get_api_key_user_id] = lambda: uuid4()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        other_user = await client.get(
            f"/api/predictive/assessments/{assessment.assessment_id}"
        )

    assert missing.status_code == 422
    assert invalid.status_code == 401
    assert other_user.status_code == 404
