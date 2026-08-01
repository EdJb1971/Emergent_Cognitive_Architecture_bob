"""Brain-inspired, bounded effort allocation for local and external research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Sequence

from src.models.research_models import (
    CognitiveEffortAction,
    CognitiveResearchAssessment,
    CognitiveResearchSignals,
    EscalationReason,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class _DriveState:
    score: float
    assessed_at: datetime
    last_research_at: Optional[datetime] = None


class CognitiveResearchDrive:
    """Accumulate cognitive-control evidence without directly invoking a provider."""

    VERSION = "1"
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "epistemic_uncertainty": 0.22,
        "cognitive_conflict": 0.16,
        "novelty_prediction_error": 0.10,
        "temporal_volatility": 0.12,
        "task_stakes": 0.14,
        "persistence_after_local_attempts": 0.10,
        "expected_information_gain": 0.16,
    }

    def __init__(
        self,
        *,
        enabled: bool = False,
        shadow_mode: bool = True,
        deepen_threshold: float = 0.28,
        uncertainty_threshold: float = 0.48,
        inquiry_threshold: float = 0.64,
        research_threshold: float = 0.78,
        cooldown_minutes: float = 30.0,
        hysteresis_minutes: float = 15.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        thresholds = [deepen_threshold, uncertainty_threshold, inquiry_threshold, research_threshold]
        if thresholds != sorted(thresholds) or any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("Research-drive thresholds must be ordered values between 0 and 1.")
        if cooldown_minutes < 0 or hysteresis_minutes <= 0:
            raise ValueError("Cooldown must be non-negative and hysteresis must be positive.")
        self.enabled = enabled
        self.shadow_mode = shadow_mode or not enabled
        self.deepen_threshold = deepen_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.inquiry_threshold = inquiry_threshold
        self.research_threshold = research_threshold
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.hysteresis_window = timedelta(minutes=hysteresis_minutes)
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)
        if set(self.weights) != set(self.DEFAULT_WEIGHTS):
            raise ValueError("Research-drive weights must define every supported excitatory signal.")
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("Research-drive weights cannot be negative.")
        self._state_by_user: Dict[str, _DriveState] = {}

    def assess(
        self,
        signals: CognitiveResearchSignals,
        *,
        source: str,
        user_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> CognitiveResearchAssessment:
        now = now or _utc_now()
        state = self._state_by_user.get(user_id) if user_id else None
        contributions = {
            name: getattr(signals, name) * weight for name, weight in self.weights.items()
        }
        excitation = sum(contributions.values())

        synergy = (
            0.10 * signals.epistemic_uncertainty * signals.task_stakes
            + 0.08 * signals.epistemic_uncertainty * signals.cognitive_conflict
            + 0.06 * signals.novelty_prediction_error * signals.expected_information_gain
        )
        if signals.explicit_user_request:
            excitation = max(excitation + synergy, 0.86)
            contributions["explicit_user_request"] = max(0.0, 0.86 - sum(contributions.values()))
        else:
            excitation += synergy
        if signals.metacognitive_gap:
            excitation += 0.10
            contributions["metacognitive_gap"] = 0.10

        hysteresis = self._hysteresis_contribution(state, now)
        cooldown_remaining = self._cooldown_remaining(state, now)
        cooldown_inhibition = 0.25 if cooldown_remaining > 0 else 0.0
        inhibition = (
            0.30 * signals.privacy_risk
            + 0.10 * signals.cloud_cost
            + cooldown_inhibition
        )
        drive_score = _clamp(excitation + hysteresis - inhibition)
        recommended = self._select_action(drive_score, signals)

        if signals.explicit_user_request and (signals.privacy_risk >= 0.9 or cooldown_remaining > 0):
            recommended = CognitiveEffortAction.QUEUE_INQUIRY
        if signals.privacy_risk >= 0.9 and recommended == CognitiveEffortAction.AUTHORIZE_RESEARCH:
            recommended = CognitiveEffortAction.QUEUE_INQUIRY
        if cooldown_remaining > 0 and recommended == CognitiveEffortAction.AUTHORIZE_RESEARCH:
            recommended = CognitiveEffortAction.QUEUE_INQUIRY

        effective = recommended if self.enabled and not self.shadow_mode else CognitiveEffortAction.ROUTINE_LOCAL
        dominant = [
            name
            for name, value in sorted(contributions.items(), key=lambda item: item[1], reverse=True)
            if value > 0
        ][:3]
        rationale = self._rationale(recommended, dominant, inhibition, cooldown_remaining)
        assessment = CognitiveResearchAssessment(
            source=source,
            signals=signals,
            drive_score=drive_score,
            excitation=max(0.0, excitation),
            inhibition=max(0.0, inhibition),
            hysteresis_contribution=hysteresis,
            signal_contributions={key: round(value, 4) for key, value in contributions.items()},
            dominant_signals=dominant,
            recommended_action=recommended,
            effective_action=effective,
            shadow_mode=self.shadow_mode,
            cooldown_remaining_seconds=cooldown_remaining,
            rationale=rationale,
            controller_version=self.VERSION,
            assessed_at=now,
        )
        if user_id:
            last_research_at = state.last_research_at if state else None
            self._state_by_user[user_id] = _DriveState(
                score=drive_score,
                assessed_at=now,
                last_research_at=last_research_at,
            )
        return assessment

    def record_research_execution(self, user_id: str, *, now: Optional[datetime] = None) -> None:
        now = now or _utc_now()
        state = self._state_by_user.get(user_id)
        self._state_by_user[user_id] = _DriveState(
            score=state.score if state else 0.0,
            assessed_at=state.assessed_at if state else now,
            last_research_at=now,
        )

    def build_waking_signals(
        self,
        *,
        confidence: float,
        coherence_score: float,
        conflict_severities: Sequence[str],
        novelty_score: float,
        urgency: str,
        policy_reasons: Sequence[EscalationReason],
        local_attempts: int = 0,
        privacy_risk: float = 0.05,
        cloud_cost: float = 0.25,
        needs_clarification: bool = False,
    ) -> CognitiveResearchSignals:
        severity_values = {"low": 0.25, "medium": 0.55, "high": 0.9}
        severity_conflict = max((severity_values.get(value, 0.0) for value in conflict_severities), default=0.0)
        conflict = max(1.0 - _clamp(coherence_score), severity_conflict)
        reason_set = set(policy_reasons)
        uncertainty = 1.0 - _clamp(confidence)
        volatility = 0.9 if EscalationReason.TIME_SENSITIVE in reason_set else 0.05
        stakes = {"low": 0.2, "normal": 0.45, "high": 0.85}.get(urgency, 0.45)
        persistence = _clamp(local_attempts / 3.0)
        information_gain = _clamp(
            0.45 * uncertainty + 0.25 * conflict + 0.20 * volatility + 0.10 * _clamp(novelty_score)
        )
        return CognitiveResearchSignals(
            epistemic_uncertainty=uncertainty,
            cognitive_conflict=conflict,
            novelty_prediction_error=_clamp(novelty_score),
            temporal_volatility=volatility,
            task_stakes=stakes,
            persistence_after_local_attempts=persistence,
            expected_information_gain=information_gain,
            privacy_risk=_clamp(privacy_risk),
            cloud_cost=_clamp(cloud_cost),
            explicit_user_request=EscalationReason.EXPLICIT_RESEARCH_REQUEST in reason_set,
            metacognitive_gap=EscalationReason.METACOGNITIVE_GAP in reason_set,
            needs_clarification=needs_clarification,
        )

    def _select_action(
        self, drive_score: float, signals: CognitiveResearchSignals
    ) -> CognitiveEffortAction:
        if drive_score < self.deepen_threshold:
            return CognitiveEffortAction.ROUTINE_LOCAL
        if drive_score < self.uncertainty_threshold:
            return CognitiveEffortAction.DEEPEN_LOCAL
        if drive_score < self.inquiry_threshold:
            if signals.needs_clarification:
                return CognitiveEffortAction.ASK_CLARIFICATION
            return CognitiveEffortAction.ACKNOWLEDGE_UNCERTAINTY
        if drive_score < self.research_threshold:
            return CognitiveEffortAction.QUEUE_INQUIRY
        return CognitiveEffortAction.AUTHORIZE_RESEARCH

    def _hysteresis_contribution(self, state: Optional[_DriveState], now: datetime) -> float:
        if not state:
            return 0.0
        elapsed = max(0.0, (now - state.assessed_at).total_seconds())
        window = self.hysteresis_window.total_seconds()
        if elapsed >= window:
            return 0.0
        decay = math.exp(-3.0 * elapsed / window)
        return max(0.0, state.score - self.deepen_threshold) * 0.12 * decay

    def _cooldown_remaining(self, state: Optional[_DriveState], now: datetime) -> float:
        if not state or not state.last_research_at or self.cooldown.total_seconds() == 0:
            return 0.0
        remaining = self.cooldown.total_seconds() - (now - state.last_research_at).total_seconds()
        return max(0.0, remaining)

    @staticmethod
    def _rationale(
        action: CognitiveEffortAction,
        dominant: Sequence[str],
        inhibition: float,
        cooldown_remaining: float,
    ) -> str:
        signals = ", ".join(dominant) if dominant else "no strong excitatory signal"
        suffix = ""
        if cooldown_remaining > 0:
            suffix = f" Research cooldown has {cooldown_remaining:.0f}s remaining."
        elif inhibition > 0:
            suffix = f" Inhibitory cost/privacy contribution was {inhibition:.2f}."
        return f"Recommended {action.value}; dominant evidence: {signals}.{suffix}"
