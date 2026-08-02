import sqlite3
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.models.core_models import CognitiveCycle
from src.models.multimodal_models import VisualAnalysis, VisualEvidence
from src.models.predictive_models import (
    HypothesisVerdict,
    ObservationQualityVerdict,
    PredictionOutcomeVerdict,
    PredictiveCalibrationLabelRequest,
    PredictivePreferredAction,
    PredictiveReviewStatus,
    RecommendationVerdict,
)
from src.services.multisensory_binding_service import MultisensoryBindingService
from src.services.predictive_calibration_store import PredictiveCalibrationStore
from src.services.predictive_perception_service import PredictivePerceptionService


NOW = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)


def _assessment(*, current_colour: str, user_id=None):
    user_id = user_id or uuid4()
    prior = CognitiveCycle(
        user_id=user_id,
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
        sha256=("a" if current_colour == "blue" else "b") * 64,
        observed_at=NOW,
        analysis=VisualAnalysis(
            description=f"A {current_colour} car",
            scene_description=f"A {current_colour} car",
            objects_detected=["car"],
            confidence=0.9,
        ),
    )
    cycle_id = uuid4()
    episode = MultisensoryBindingService().bind_turn(
        cycle_id=cycle_id,
        user_id=user_id,
        session_id=uuid4(),
        request_timestamp=NOW,
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
    colour_error = next(
        item for item in assessment.prediction_errors
        if item.feature_name == "colour:car"
    )
    return user_id, assessment, colour_error


@pytest.mark.asyncio
async def test_assessment_record_is_idempotent_append_only_and_user_isolated(tmp_path):
    sink = AsyncMock()
    store = PredictiveCalibrationStore(tmp_path / "predictive.sqlite3", event_sink=sink)
    await store.connect()
    user_id, assessment, _error = _assessment(current_colour="blue")

    first = await store.record_assessment(assessment, user_id=user_id)
    repeated = await store.record_assessment(assessment, user_id=user_id)
    listed = await store.list_assessments(user_id)

    assert repeated.event_id == first.event_id
    assert len(listed) == 1
    assert listed[0].review_status == PredictiveReviewStatus.UNREVIEWED
    assert listed[0].assessment == assessment
    sink.assert_awaited_once()
    with pytest.raises(KeyError):
        await store.get_assessment(uuid4(), assessment.assessment_id)
    with sqlite3.connect(tmp_path / "predictive.sqlite3") as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE predictive_calibration_ledger SET payload = '{}' WHERE sequence = 1"
            )


@pytest.mark.asyncio
async def test_labels_append_supersession_without_rewriting_original_assessment(tmp_path):
    store = PredictiveCalibrationStore(tmp_path / "predictive.sqlite3")
    await store.connect()
    user_id, assessment, error = _assessment(current_colour="blue")
    await store.record_assessment(assessment, user_id=user_id)
    first_request = PredictiveCalibrationLabelRequest(
        error_id=error.error_id,
        hypothesis_verdict=HypothesisVerdict.INCORRECT,
        observation_quality=ObservationQualityVerdict.RELIABLE,
        prediction_outcome=PredictionOutcomeVerdict.CONFIRMED_MISMATCH,
        recommendation_verdict=RecommendationVerdict.USEFUL,
        preferred_action=PredictivePreferredAction.ASK_USER,
        rationale="The blue observation was clear and the prior was stale.",
    )
    first = await store.append_label(
        user_id=user_id,
        assessment_id=assessment.assessment_id,
        label=first_request,
    )
    correction = await store.append_label(
        user_id=user_id,
        assessment_id=assessment.assessment_id,
        label=first_request.model_copy(update={
            "recommendation_verdict": RecommendationVerdict.UNNECESSARY,
            "preferred_action": PredictivePreferredAction.NONE,
            "rationale": "Correction: the mismatch was real but clarification was unnecessary.",
        }),
    )
    review = await store.get_assessment(user_id, assessment.assessment_id)

    assert correction.sequence > first.sequence
    assert correction.payload["supersedes_event_id"] == str(first.event_id)
    assert review.assessment == assessment
    assert review.reviewed_target_count == 1
    assert review.review_status == PredictiveReviewStatus.PARTIALLY_REVIEWED
    assert len(review.latest_labels) == 1
    assert review.latest_labels[0].event_id == correction.event_id
    assert await store.verify_integrity() is True


@pytest.mark.asyncio
async def test_label_scope_rejects_unknown_errors_and_unrelated_recommendations(tmp_path):
    store = PredictiveCalibrationStore(tmp_path / "predictive.sqlite3")
    await store.connect()
    user_id, assessment, error = _assessment(current_colour="blue")
    await store.record_assessment(assessment, user_id=user_id)

    with pytest.raises(KeyError, match="not part"):
        await store.append_label(
            user_id=user_id,
            assessment_id=assessment.assessment_id,
            label=PredictiveCalibrationLabelRequest(
                error_id=uuid4(),
                rationale="This target does not exist.",
            ),
        )
    unrelated = next(
        item for item in assessment.prediction_errors
        if item.error_id not in assessment.recommendation.source_error_ids
    )
    with pytest.raises(ValueError, match="does not target"):
        await store.append_label(
            user_id=user_id,
            assessment_id=assessment.assessment_id,
            label=PredictiveCalibrationLabelRequest(
                error_id=unrelated.error_id,
                recommendation_verdict=RecommendationVerdict.USEFUL,
                rationale="A recommendation judgement cannot attach here.",
            ),
        )
    with pytest.raises(ValueError, match="reserved"):
        await store.append_label(
            user_id=user_id,
            assessment_id=assessment.assessment_id,
            label=PredictiveCalibrationLabelRequest(
                rationale="This assessment recommendation already points to errors.",
            ),
        )
    assert error.error_id in assessment.recommendation.source_error_ids


@pytest.mark.asyncio
async def test_longitudinal_summary_reports_confusion_calibration_and_strata(tmp_path):
    store = PredictiveCalibrationStore(tmp_path / "predictive.sqlite3")
    await store.connect()
    user_id = uuid4()
    _user, mismatch, mismatch_error = _assessment(
        current_colour="blue", user_id=user_id
    )
    _user, match, match_error = _assessment(
        current_colour="red", user_id=user_id
    )
    await store.record_assessment(mismatch, user_id=user_id)
    await store.record_assessment(match, user_id=user_id)
    await store.append_label(
        user_id=user_id,
        assessment_id=mismatch.assessment_id,
        label=PredictiveCalibrationLabelRequest(
            error_id=mismatch_error.error_id,
            hypothesis_verdict=HypothesisVerdict.INCORRECT,
            observation_quality=ObservationQualityVerdict.RELIABLE,
            prediction_outcome=PredictionOutcomeVerdict.CONFIRMED_MISMATCH,
            recommendation_verdict=RecommendationVerdict.USEFUL,
            preferred_action=PredictivePreferredAction.ASK_USER,
            rationale="The mismatch and clarification were appropriate.",
        ),
    )
    await store.append_label(
        user_id=user_id,
        assessment_id=match.assessment_id,
        label=PredictiveCalibrationLabelRequest(
            error_id=match_error.error_id,
            hypothesis_verdict=HypothesisVerdict.CORRECT,
            observation_quality=ObservationQualityVerdict.RELIABLE,
            prediction_outcome=PredictionOutcomeVerdict.CONFIRMED_MATCH,
            rationale="The prior and observation matched.",
        ),
    )

    summary = await store.calibration_summary(user_id)

    assert summary.assessments == 2
    assert summary.errors >= 2
    assert summary.labeled_errors == 2
    assert summary.mismatch_precision == 1.0
    assert summary.mismatch_recall == 1.0
    assert summary.false_conflict_rate == 0.0
    assert summary.hypothesis_accuracy == 0.5
    assert summary.recommendation_usefulness_rate == 1.0
    assert summary.expected_calibration_error is not None
    assert summary.strata["feature:categorical_attribute"].labeled == 2
    assert summary.strata["modality:image"].labeled == 2
    assert summary.daily[0].assessments == 2
    assert summary.ledger_integrity_verified is True
    assert summary.predictive_influence_eligible is False


@pytest.mark.asyncio
async def test_unknown_outcomes_count_as_reviewed_but_not_calibration_truth(tmp_path):
    store = PredictiveCalibrationStore(tmp_path / "predictive.sqlite3")
    await store.connect()
    user_id, assessment, error = _assessment(current_colour="blue")
    await store.record_assessment(assessment, user_id=user_id)
    await store.append_label(
        user_id=user_id,
        assessment_id=assessment.assessment_id,
        label=PredictiveCalibrationLabelRequest(
            error_id=error.error_id,
            hypothesis_verdict=HypothesisVerdict.INCORRECT,
            observation_quality=ObservationQualityVerdict.RELIABLE,
            prediction_outcome=PredictionOutcomeVerdict.CONFIRMED_MISMATCH,
            recommendation_verdict=RecommendationVerdict.USEFUL,
            preferred_action=PredictivePreferredAction.ASK_USER,
            outcome_known=False,
            rationale="Reviewed, but there is not enough outcome evidence yet.",
        ),
    )

    summary = await store.calibration_summary(user_id)

    assert summary.labeled_errors == 1
    assert summary.observation_reliable_rate == 1.0
    assert summary.mismatch_precision is None
    assert summary.hypothesis_accuracy is None
    assert summary.recommendation_usefulness_rate is None
    assert summary.expected_calibration_error is None
    assert summary.strata["modality:image"].labeled == 1
    assert summary.strata["modality:image"].confirmed_mismatch == 0
