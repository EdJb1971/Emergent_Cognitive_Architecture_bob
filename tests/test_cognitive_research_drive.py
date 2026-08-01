from datetime import datetime, timedelta, timezone

import pytest

from src.models.research_models import (
    CognitiveEffortAction,
    CognitiveResearchSignals,
    EscalationReason,
)
from src.services.cognitive_research_drive import CognitiveResearchDrive


def test_low_signal_cycle_stays_with_routine_local_cognition():
    drive = CognitiveResearchDrive(enabled=True, shadow_mode=False)

    assessment = drive.assess(
        CognitiveResearchSignals(
            epistemic_uncertainty=0.15,
            cognitive_conflict=0.05,
            novelty_prediction_error=0.1,
            task_stakes=0.2,
            expected_information_gain=0.2,
        ),
        source="waking",
    )

    assert assessment.recommended_action == CognitiveEffortAction.ROUTINE_LOCAL
    assert assessment.effective_action == CognitiveEffortAction.ROUTINE_LOCAL
    assert assessment.drive_score < drive.deepen_threshold


def test_converging_high_value_signals_recruit_external_research():
    drive = CognitiveResearchDrive(enabled=True, shadow_mode=False)

    assessment = drive.assess(
        CognitiveResearchSignals(
            epistemic_uncertainty=0.9,
            cognitive_conflict=0.85,
            novelty_prediction_error=0.75,
            temporal_volatility=0.8,
            task_stakes=0.9,
            persistence_after_local_attempts=0.8,
            expected_information_gain=0.95,
            privacy_risk=0.05,
            cloud_cost=0.2,
            metacognitive_gap=True,
        ),
        source="waking",
    )

    assert assessment.recommended_action == CognitiveEffortAction.AUTHORIZE_RESEARCH
    assert assessment.effective_action == CognitiveEffortAction.AUTHORIZE_RESEARCH
    assert len(assessment.dominant_signals) == 3


def test_shadow_mode_observes_research_drive_without_applying_it():
    drive = CognitiveResearchDrive(enabled=True, shadow_mode=True)

    assessment = drive.assess(
        CognitiveResearchSignals(explicit_user_request=True, expected_information_gain=0.9),
        source="waking",
    )

    assert assessment.recommended_action == CognitiveEffortAction.AUTHORIZE_RESEARCH
    assert assessment.effective_action == CognitiveEffortAction.ROUTINE_LOCAL
    assert assessment.shadow_mode is True


def test_privacy_inhibition_prevents_external_authorization():
    drive = CognitiveResearchDrive(enabled=True, shadow_mode=False)

    assessment = drive.assess(
        CognitiveResearchSignals(
            explicit_user_request=True,
            epistemic_uncertainty=0.9,
            expected_information_gain=0.9,
            privacy_risk=1.0,
        ),
        source="waking",
    )

    assert assessment.recommended_action == CognitiveEffortAction.QUEUE_INQUIRY
    assert assessment.inhibition >= 0.3


def test_refractory_cooldown_prevents_repeated_external_research():
    now = datetime(2026, 8, 2, tzinfo=timezone.utc)
    drive = CognitiveResearchDrive(
        enabled=True,
        shadow_mode=False,
        cooldown_minutes=30,
    )
    drive.record_research_execution("user-1", now=now)

    assessment = drive.assess(
        CognitiveResearchSignals(explicit_user_request=True, expected_information_gain=1.0),
        source="waking",
        user_id="user-1",
        now=now + timedelta(minutes=5),
    )

    assert assessment.recommended_action == CognitiveEffortAction.QUEUE_INQUIRY
    assert assessment.cooldown_remaining_seconds == pytest.approx(25 * 60)


def test_hysteresis_retains_recent_control_signal_without_latching_forever():
    now = datetime(2026, 8, 2, tzinfo=timezone.utc)
    drive = CognitiveResearchDrive(enabled=True, shadow_mode=False, hysteresis_minutes=15)
    drive.assess(
        CognitiveResearchSignals(
            epistemic_uncertainty=0.8,
            cognitive_conflict=0.7,
            task_stakes=0.8,
            expected_information_gain=0.8,
            metacognitive_gap=True,
        ),
        source="waking",
        user_id="user-1",
        now=now,
    )
    weak = CognitiveResearchSignals(
        epistemic_uncertainty=0.35,
        task_stakes=0.3,
        expected_information_gain=0.3,
    )

    recent = drive.assess(weak, source="waking", user_id="user-1", now=now + timedelta(minutes=1))
    expired = drive.assess(weak, source="waking", user_id="user-1", now=now + timedelta(minutes=30))

    assert recent.hysteresis_contribution > 0
    assert expired.hysteresis_contribution == 0
    assert recent.drive_score > expired.drive_score


def test_waking_signal_builder_combines_policy_conflict_and_effort_evidence():
    drive = CognitiveResearchDrive()

    signals = drive.build_waking_signals(
        confidence=0.25,
        coherence_score=0.8,
        conflict_severities=["high"],
        novelty_score=0.6,
        urgency="high",
        policy_reasons=[
            EscalationReason.EXPLICIT_RESEARCH_REQUEST,
            EscalationReason.TIME_SENSITIVE,
            EscalationReason.METACOGNITIVE_GAP,
        ],
        local_attempts=2,
        needs_clarification=True,
    )

    assert signals.epistemic_uncertainty == pytest.approx(0.75)
    assert signals.cognitive_conflict == pytest.approx(0.9)
    assert signals.temporal_volatility == pytest.approx(0.9)
    assert signals.persistence_after_local_attempts == pytest.approx(2 / 3)
    assert signals.explicit_user_request is True
    assert signals.needs_clarification is True


def test_invalid_threshold_order_fails_closed():
    with pytest.raises(ValueError, match="ordered"):
        CognitiveResearchDrive(deepen_threshold=0.8, research_threshold=0.5)

