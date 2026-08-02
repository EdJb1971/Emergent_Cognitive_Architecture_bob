from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.models.core_models import CognitiveCycle
from src.models.multimodal_models import VisualAnalysis, VisualEvidence
from src.models.predictive_models import PredictivePerceptionAssessment
from src.services.multisensory_binding_service import MultisensoryBindingService
from src.services.predictive_perception_service import PredictivePerceptionService


NOW = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)


def _visual(description: str, *, quality: float = 0.9, confidence: float = 0.9):
    return VisualEvidence(
        provider="ollama", model="gemma4:e4b", mime_type="image/png",
        byte_count=100, width=640, height=480, input_quality_score=quality,
        sha256="a" * 64, observed_at=NOW,
        analysis=VisualAnalysis(
            description=description, scene_description=description,
            objects_detected=["car"] if "car" in description else [],
            confidence=confidence,
        ),
    )


def _prior(text: str, *, final_response: str = "Assistant-generated guess"):
    return CognitiveCycle(
        user_id=uuid4(), session_id=uuid4(), user_input=text,
        final_response=final_response,
    )


def _assess(priors, *, text="What is visible?", visual=None, service=None):
    user_id = priors[0].user_id if priors else uuid4()
    cycle_id = uuid4()
    episode = MultisensoryBindingService().bind_turn(
        cycle_id=cycle_id, user_id=user_id, session_id=uuid4(),
        request_timestamp=NOW, text=text, visual_evidence=visual,
    )
    assessment = (service or PredictivePerceptionService()).assess(
        cycle_id=cycle_id, sensory_episode=episode, prior_cycles=priors,
        current_text=text, visual_evidence=visual,
    )
    return episode, assessment


def test_no_prior_context_produces_empty_immutable_shadow_assessment():
    _episode, assessment = _assess([])

    assert assessment.schema_version == "predictive-perception-v1"
    assert assessment.assessment_status == "assessed"
    assert assessment.shadow_mode is True
    assert assessment.hypotheses == ()
    assert assessment.prediction_errors == ()
    assert assessment.recommendation is None
    assert assessment.response_influenced is False
    assert assessment.routing_influenced is False
    assert assessment.research_invoked is False
    assert assessment.learning_update_applied is False
    with pytest.raises(ValidationError):
        assessment.response_influenced = True


def test_hypotheses_are_labelled_prior_only_and_never_use_assistant_or_prediction_text():
    prior = _prior(
        "The car is red.",
        final_response="The forbidden assistant hypothesis is that the bicycle is purple.",
    )
    prior.metadata["predictive_perception"] = {
        "hypotheses": [{"feature_name": "secret_prediction", "predicted_value": "present"}]
    }

    _episode, assessment = _assess([prior], visual=_visual("A red car"))

    assert assessment.hypotheses
    assert all(item.label == "prior_hypothesis_not_observation" for item in assessment.hypotheses)
    assert all(item.formed_from_prior_context_only for item in assessment.hypotheses)
    assert all(item.semantic_truth_verified is False for item in assessment.hypotheses)
    serialized = assessment.model_dump_json()
    assert "forbidden" not in serialized
    assert "bicycle" not in serialized
    assert "secret_prediction" not in serialized


def test_cross_user_prior_is_excluded_at_the_service_boundary():
    same_user_prior = _prior("The car is red.")
    other_user_prior = _prior("The dog is purple.")
    cycle_id = uuid4()
    visual = _visual("A red car")
    episode = MultisensoryBindingService().bind_turn(
        cycle_id=cycle_id,
        user_id=same_user_prior.user_id,
        session_id=uuid4(),
        request_timestamp=NOW,
        text="What is visible?",
        visual_evidence=visual,
    )

    assessment = PredictivePerceptionService().assess(
        cycle_id=cycle_id,
        sensory_episode=episode,
        prior_cycles=[other_user_prior, same_user_prior],
        current_text="What is visible?",
        visual_evidence=visual,
    )

    assert assessment.prior_cycle_ids == (same_user_prior.cycle_id,)
    assert "dog" not in assessment.model_dump_json()
    assert "purple" not in assessment.model_dump_json()


def test_matching_prior_and_observation_have_zero_prediction_error():
    prior = _prior("The car is red.")
    _episode, assessment = _assess([prior], visual=_visual("A red car"))

    colour = next(
        item for item in assessment.prediction_errors
        if item.feature_name == "colour:car"
    )
    assert colour.status == "matched"
    assert colour.signed_error == 0.0
    assert colour.surprise_magnitude == 0.0
    assert colour.calibration_eligible is True
    assert assessment.recommendation is None


def test_categorical_mismatch_emits_material_signed_error_and_unexecuted_clarification():
    prior = _prior("The car is red.")
    episode, assessment = _assess([prior], visual=_visual("A blue car"))

    colour = next(
        item for item in assessment.prediction_errors
        if item.feature_name == "colour:car"
    )
    assert colour.status == "mismatch"
    assert colour.direction == "categorical_mismatch"
    assert colour.signed_error > 0.0
    assert colour.material is True
    assert colour.observation_reference == "image_sha256:" + "a" * 64
    assert colour.sensory_episode_id == episode.episode_id
    assert assessment.recommendation.action == "ask_user"
    assert assessment.recommendation.shadow_only is True
    assert assessment.recommendation.executed is False
    assert assessment.recommendation.cloud_research_allowed is False
    assert assessment.response_influenced is False
    assert assessment.primary_evidence_rewritten is False


def test_presence_error_has_direction_and_never_promotes_hypothesis_to_observation():
    prior = _prior("There is no dog.")
    visual = _visual("A dog is standing nearby")
    visual.analysis.objects_detected.append("dog")
    _episode, assessment = _assess(
        [prior], visual=visual,
        service=PredictivePerceptionService(clarification_threshold=0.45),
    )

    dog = next(item for item in assessment.prediction_errors if item.feature_name == "dog")
    assert dog.predicted_value == "absent"
    assert dog.observed_value == "present"
    assert dog.direction == "unexpected_presence"
    assert dog.signed_error > 0
    assert dog.derived_only is True
    assert dog.primary_evidence_changed is False


def test_low_reliability_check_recommends_recapture_but_cannot_execute_it():
    prior = _prior("The car is red.")
    _episode, assessment = _assess(
        [prior], visual=_visual("A blue car", quality=0.1, confidence=0.1),
    )

    colour = next(
        item for item in assessment.prediction_errors
        if item.feature_name == "colour:car"
    )
    assert colour.status == "low_reliability"
    assert colour.material is False
    assert assessment.recommendation.action == "request_image_recapture"
    assert assessment.recommendation.reason == "low_reliability_prediction_check"
    assert assessment.recommendation.executed is False


def test_current_cross_modal_conflict_can_recommend_clarification_without_a_prior():
    episode, assessment = _assess([], text="There is no dog", visual=_visual("A dog"))

    assert any(item.relation_type == "contradiction" for item in episode.relations)
    assert assessment.hypothesis_count == 0
    assert assessment.recommendation.reason == "unresolved_cross_modal_conflict"
    assert assessment.recommendation.action == "ask_user"
    assert assessment.recommendation.executed is False


def test_disabled_mode_and_non_shadow_guard_fail_safe():
    prior = _prior("The car is red.")
    _episode, assessment = _assess(
        [prior], visual=_visual("A blue car"),
        service=PredictivePerceptionService(enabled=False),
    )
    assert assessment.enabled is False
    assert assessment.assessment_status == "disabled"
    assert assessment.hypotheses == ()
    assert assessment.recommendation is None
    with pytest.raises(ValueError, match="shadow-only"):
        PredictivePerceptionService(shadow_mode=False)


def test_degraded_assessment_is_empty_immutable_and_explicit():
    assessment = PredictivePerceptionService().degraded_assessment(
        cycle_id=uuid4(), sensory_episode_id=uuid4(),
    )

    assert assessment.enabled is True
    assert assessment.assessment_status == "degraded"
    assert assessment.degradation_reason == "assessment_failed"
    assert assessment.hypotheses == ()
    assert assessment.prediction_errors == ()
    assert assessment.recommendation is None
    assert assessment.response_influenced is False


def test_assessment_contract_rejects_count_or_reference_tampering():
    prior = _prior("The car is red.")
    _episode, assessment = _assess([prior], visual=_visual("A blue car"))
    invalid = assessment.model_dump()
    invalid["hypothesis_count"] = 999
    with pytest.raises(ValidationError):
        PredictivePerceptionAssessment.model_validate(invalid)

    missing_error = assessment.model_dump()
    missing_error["prediction_errors"] = []
    missing_error["mismatch_count"] = 0
    missing_error["material_error_count"] = 0
    with pytest.raises(ValidationError):
        PredictivePerceptionAssessment.model_validate(missing_error)

    rewritten_feature = assessment.model_dump()
    rewritten_feature["prediction_errors"][0]["feature_name"] = "invented"
    with pytest.raises(ValidationError):
        PredictivePerceptionAssessment.model_validate(rewritten_feature)


def test_malformed_legacy_sensory_metadata_is_ignored_fail_safe():
    prior = _prior("What was present?")
    prior.metadata.update({
        "sensory_episode": {"relations": [{"relation_type": "agreement", "strength": "nan", "anchors": "dog"}]},
        "visual_evidence": {"input_quality_score": "bad", "analysis": {"confidence": None, "objects_detected": "car"}},
        "auditory_evidence": {"signal_quality_score": float("inf"), "analysis": {"confidence": {}, "audio_events": "bell"}},
    })

    _episode, assessment = _assess([prior], visual=_visual("A car"))

    assert assessment.hypotheses == ()
    assert assessment.prediction_errors == ()
