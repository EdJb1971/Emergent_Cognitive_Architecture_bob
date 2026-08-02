"""Immutable contracts for shadow predictive perception and active clarification."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


Modality = Literal["text", "image", "audio"]


class PerceptualHypothesis(BaseModel):
    """A prior-derived, reviewable prediction that is explicitly not evidence."""

    model_config = ConfigDict(frozen=True)

    hypothesis_id: UUID
    label: Literal["prior_hypothesis_not_observation"] = "prior_hypothesis_not_observation"
    source_cycle_id: UUID
    source_reference: str = Field(min_length=1, max_length=180)
    source_kind: Literal[
        "prior_user_assertion",
        "prior_cross_modal_corroboration",
        "prior_visual_object_observation",
        "prior_auditory_event_observation",
    ]
    feature_kind: Literal["presence", "categorical_attribute"]
    feature_name: str = Field(min_length=1, max_length=96)
    predicted_value: str = Field(min_length=1, max_length=96)
    prior_confidence: float = Field(ge=0.0, le=1.0)
    reviewable_modalities: Tuple[Modality, ...]
    formed_from_prior_context_only: Literal[True] = True
    semantic_truth_verified: Literal[False] = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _modalities_are_reviewable(self) -> "PerceptualHypothesis":
        if not self.reviewable_modalities:
            raise ValueError("a hypothesis requires at least one reviewable modality")
        if len(set(self.reviewable_modalities)) != len(self.reviewable_modalities):
            raise ValueError("reviewable modalities must be unique")
        return self


class PerceptualPredictionError(BaseModel):
    """Signed discrepancy between one hypothesis and one primary observation."""

    model_config = ConfigDict(frozen=True)

    error_id: UUID
    hypothesis_id: UUID
    sensory_episode_id: UUID
    feature_kind: Literal["presence", "categorical_attribute"]
    feature_name: str = Field(min_length=1, max_length=96)
    predicted_value: str = Field(min_length=1, max_length=96)
    observed_value: Optional[str] = Field(None, max_length=96)
    observed_modality: Optional[Modality] = None
    observation_reference: Optional[str] = Field(None, max_length=180)
    status: Literal["matched", "mismatch", "unobserved", "low_reliability"]
    direction: Literal[
        "zero",
        "unexpected_presence",
        "unexpected_absence",
        "categorical_mismatch",
    ]
    signed_error: float = Field(ge=-1.0, le=1.0)
    surprise_magnitude: float = Field(ge=0.0, le=1.0)
    prior_confidence: float = Field(ge=0.0, le=1.0)
    observation_reliability: Optional[float] = Field(None, ge=0.0, le=1.0)
    calibration_eligible: bool
    material: bool
    derived_only: Literal[True] = True
    primary_evidence_changed: Literal[False] = False

    @model_validator(mode="after")
    def _status_is_consistent(self) -> "PerceptualPredictionError":
        observation_fields = (
            self.observed_value,
            self.observed_modality,
            self.observation_reference,
            self.observation_reliability,
        )
        has_complete_observation = all(value is not None for value in observation_fields)
        if self.status == "unobserved":
            if any(value is not None for value in observation_fields):
                raise ValueError("unobserved errors cannot claim observation fields")
            if any((self.direction != "zero", self.signed_error != 0.0, self.surprise_magnitude != 0.0,
                    self.calibration_eligible, self.material)):
                raise ValueError("unobserved errors must remain zero and non-calibratable")
        elif not has_complete_observation:
            raise ValueError("observed statuses require complete observation provenance")

        if self.status == "low_reliability":
            if any((self.direction != "zero", self.signed_error != 0.0, self.surprise_magnitude != 0.0,
                    self.calibration_eligible, self.material)):
                raise ValueError("low-reliability errors must remain zero and non-calibratable")
        elif self.status == "matched":
            if self.observed_value != self.predicted_value:
                raise ValueError("matched predictions require equal values")
            if any((self.direction != "zero", self.signed_error != 0.0, self.surprise_magnitude != 0.0,
                    not self.calibration_eligible, self.material)):
                raise ValueError("matched predictions must have zero calibratable error")
        elif self.status == "mismatch":
            if self.observed_value == self.predicted_value:
                raise ValueError("mismatch requires different values")
            if self.direction == "zero" or not self.calibration_eligible:
                raise ValueError("mismatch requires a signed calibratable direction")
            if abs(abs(self.signed_error) - self.surprise_magnitude) > 0.0001:
                raise ValueError("signed error magnitude must equal surprise magnitude")
            if self.direction == "unexpected_absence" and self.signed_error >= 0.0:
                raise ValueError("unexpected absence requires a negative signed error")
            if self.direction in {"unexpected_presence", "categorical_mismatch"} and self.signed_error <= 0.0:
                raise ValueError("unexpected presence/categorical mismatch requires a positive signed error")
        if self.material and self.status != "mismatch":
            raise ValueError("only mismatches can be material")
        return self


class ClarificationRecommendation(BaseModel):
    """One bounded action recommendation that is never executed in shadow mode."""

    model_config = ConfigDict(frozen=True)

    recommendation_id: UUID
    action: Literal["ask_user", "request_image_recapture", "request_audio_recapture"]
    reason: Literal[
        "material_prediction_error",
        "low_reliability_prediction_check",
        "unresolved_cross_modal_conflict",
    ]
    target_modalities: Tuple[Modality, ...]
    prompt: str = Field(min_length=1, max_length=320)
    priority: float = Field(ge=0.0, le=1.0)
    expected_information_gain: float = Field(ge=0.0, le=1.0)
    source_error_ids: Tuple[UUID, ...] = ()
    source_relation_indexes: Tuple[int, ...] = ()
    shadow_only: Literal[True] = True
    executed: Literal[False] = False
    cloud_research_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _sources_match_reason(self) -> "ClarificationRecommendation":
        if not self.target_modalities:
            raise ValueError("a clarification requires a target modality")
        if len(set(self.target_modalities)) != len(self.target_modalities):
            raise ValueError("target modalities must be unique")
        if self.reason == "unresolved_cross_modal_conflict":
            if not self.source_relation_indexes or self.source_error_ids:
                raise ValueError("cross-modal clarification requires relation indexes only")
        elif not self.source_error_ids or self.source_relation_indexes:
            raise ValueError("prediction clarification requires prediction-error ids only")
        if self.action == "request_image_recapture" and "image" not in self.target_modalities:
            raise ValueError("image recapture must target image")
        if self.action == "request_audio_recapture" and "audio" not in self.target_modalities:
            raise ValueError("audio recapture must target audio")
        return self


class PredictivePerceptionAssessment(BaseModel):
    """Complete immutable shadow result for one cognitive cycle."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["predictive-perception-v1"] = "predictive-perception-v1"
    assessment_id: UUID
    cycle_id: UUID
    sensory_episode_id: UUID
    enabled: bool
    assessment_status: Literal["assessed", "disabled", "degraded"]
    degradation_reason: Optional[Literal["assessment_failed"]] = None
    shadow_mode: Literal[True] = True
    prior_cycle_ids: Tuple[UUID, ...] = ()
    hypotheses: Tuple[PerceptualHypothesis, ...] = ()
    prediction_errors: Tuple[PerceptualPredictionError, ...] = ()
    recommendation: Optional[ClarificationRecommendation] = None
    hypothesis_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    mismatch_count: int = Field(ge=0)
    unobserved_count: int = Field(ge=0)
    low_reliability_count: int = Field(ge=0)
    material_error_count: int = Field(ge=0)
    response_influenced: Literal[False] = False
    routing_influenced: Literal[False] = False
    research_invoked: Literal[False] = False
    learning_update_applied: Literal[False] = False
    primary_evidence_rewritten: Literal[False] = False
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _counts_and_references_are_consistent(self) -> "PredictivePerceptionAssessment":
        if self.assessment_status == "assessed" and not self.enabled:
            raise ValueError("assessed status requires an enabled service")
        if self.assessment_status == "disabled" and self.enabled:
            raise ValueError("disabled status requires a disabled service")
        if self.assessment_status == "degraded":
            if not self.enabled or self.degradation_reason != "assessment_failed":
                raise ValueError("degraded status requires an enabled failed assessment")
        elif self.degradation_reason is not None:
            raise ValueError("only degraded assessments can carry a degradation reason")
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count must match hypotheses")
        hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
        error_hypothesis_ids = [item.hypothesis_id for item in self.prediction_errors]
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise ValueError("hypothesis ids must be unique")
        if len(set(item.error_id for item in self.prediction_errors)) != len(self.prediction_errors):
            raise ValueError("prediction-error ids must be unique")
        if sorted(map(str, error_hypothesis_ids)) != sorted(map(str, hypothesis_ids)):
            raise ValueError("every included hypothesis requires exactly one prediction error")
        if any(item.sensory_episode_id != self.sensory_episode_id for item in self.prediction_errors):
            raise ValueError("prediction errors must reference this sensory episode")
        hypotheses_by_id = {item.hypothesis_id: item for item in self.hypotheses}
        for error in self.prediction_errors:
            hypothesis = hypotheses_by_id[error.hypothesis_id]
            if (
                error.feature_kind != hypothesis.feature_kind
                or error.feature_name != hypothesis.feature_name
                or error.predicted_value != hypothesis.predicted_value
                or error.prior_confidence != hypothesis.prior_confidence
            ):
                raise ValueError("prediction error must preserve its hypothesis fields")
        if any(item.source_cycle_id not in self.prior_cycle_ids for item in self.hypotheses):
            raise ValueError("hypotheses must reference included prior cycles")
        if len(set(self.prior_cycle_ids)) != len(self.prior_cycle_ids):
            raise ValueError("prior cycle ids must be unique")
        expected = {
            "matched_count": sum(item.status == "matched" for item in self.prediction_errors),
            "mismatch_count": sum(item.status == "mismatch" for item in self.prediction_errors),
            "unobserved_count": sum(item.status == "unobserved" for item in self.prediction_errors),
            "low_reliability_count": sum(item.status == "low_reliability" for item in self.prediction_errors),
            "material_error_count": sum(item.material for item in self.prediction_errors),
        }
        for field_name, value in expected.items():
            if getattr(self, field_name) != value:
                raise ValueError(f"{field_name} does not match prediction errors")
        if self.recommendation:
            error_ids = {item.error_id for item in self.prediction_errors}
            if not set(self.recommendation.source_error_ids).issubset(error_ids):
                raise ValueError("recommendation references an unknown prediction error")
        if self.assessment_status in {"disabled", "degraded"} and (
            self.prior_cycle_ids or self.hypotheses or self.prediction_errors or self.recommendation
        ):
            raise ValueError("disabled/degraded assessment cannot contain predictive work")
        return self


class PredictiveLedgerEventType(str, Enum):
    ASSESSMENT_RECORDED = "assessment_recorded"
    CALIBRATION_LABEL = "calibration_label"


class PredictiveReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    PARTIALLY_REVIEWED = "partially_reviewed"
    REVIEWED = "reviewed"
    NOT_APPLICABLE = "not_applicable"


class HypothesisVerdict(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"
    NOT_REVIEWED = "not_reviewed"


class ObservationQualityVerdict(str, Enum):
    RELIABLE = "reliable"
    UNRELIABLE = "unreliable"
    INSUFFICIENT = "insufficient"
    UNCERTAIN = "uncertain"
    NOT_REVIEWED = "not_reviewed"


class PredictionOutcomeVerdict(str, Enum):
    CONFIRMED_MATCH = "confirmed_match"
    CONFIRMED_MISMATCH = "confirmed_mismatch"
    FALSE_CONFLICT = "false_conflict"
    MISSED_MISMATCH = "missed_mismatch"
    INDETERMINATE = "indeterminate"


class RecommendationVerdict(str, Enum):
    USEFUL = "useful"
    UNNECESSARY = "unnecessary"
    WRONG_ACTION = "wrong_action"
    NOT_APPLICABLE = "not_applicable"
    UNCERTAIN = "uncertain"


class PredictivePreferredAction(str, Enum):
    NONE = "none"
    ASK_USER = "ask_user"
    REQUEST_IMAGE_RECAPTURE = "request_image_recapture"
    REQUEST_AUDIO_RECAPTURE = "request_audio_recapture"


class PredictiveCalibrationLabelRequest(BaseModel):
    """Independent human judgement appended without changing its target."""

    error_id: Optional[UUID] = None
    hypothesis_verdict: HypothesisVerdict = HypothesisVerdict.NOT_REVIEWED
    observation_quality: ObservationQualityVerdict = (
        ObservationQualityVerdict.NOT_REVIEWED
    )
    prediction_outcome: PredictionOutcomeVerdict = (
        PredictionOutcomeVerdict.INDETERMINATE
    )
    recommendation_verdict: RecommendationVerdict = (
        RecommendationVerdict.NOT_APPLICABLE
    )
    preferred_action: PredictivePreferredAction = PredictivePreferredAction.NONE
    outcome_known: bool = True
    rationale: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _scope_is_consistent(self) -> "PredictiveCalibrationLabelRequest":
        if self.error_id is None and (
            self.hypothesis_verdict != HypothesisVerdict.NOT_REVIEWED
            or self.observation_quality != ObservationQualityVerdict.NOT_REVIEWED
            or self.prediction_outcome != PredictionOutcomeVerdict.INDETERMINATE
        ):
            raise ValueError(
                "hypothesis, observation, and prediction-outcome labels require error_id"
            )
        return self


class PredictiveLedgerEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    sequence: int = Field(ge=1)
    event_id: UUID
    event_type: PredictiveLedgerEventType
    user_id: UUID
    cycle_id: UUID
    assessment_id: UUID
    error_id: Optional[UUID] = None
    created_at: datetime
    payload: Dict[str, object] = Field(default_factory=dict)
    previous_hash: str = Field(min_length=64, max_length=64)
    event_hash: str = Field(min_length=64, max_length=64)


class PredictiveAssessmentReview(BaseModel):
    model_config = ConfigDict(frozen=True)

    assessment: PredictivePerceptionAssessment
    recorded_at: datetime
    ledger_sequence: int = Field(ge=1)
    review_status: PredictiveReviewStatus
    review_target_count: int = Field(ge=0)
    reviewed_target_count: int = Field(ge=0)
    latest_labels: Tuple[PredictiveLedgerEvent, ...] = ()


class PredictiveAssessmentListResponse(BaseModel):
    assessments: List[PredictiveAssessmentReview] = Field(default_factory=list)
    count: int = Field(ge=0)


class PredictiveLedgerResponse(BaseModel):
    events: List[PredictiveLedgerEvent] = Field(default_factory=list)
    count: int = Field(ge=0)
    next_after_sequence: Optional[int] = None


class PredictiveCalibrationStratum(BaseModel):
    observations: int = Field(ge=0)
    labeled: int = Field(ge=0)
    predicted_mismatch: int = Field(ge=0)
    confirmed_mismatch: int = Field(ge=0)
    false_conflict: int = Field(ge=0)
    missed_mismatch: int = Field(ge=0)
    hypothesis_correct: int = Field(ge=0)
    hypothesis_incorrect: int = Field(ge=0)
    recommendation_reviewed: int = Field(ge=0)
    recommendation_useful: int = Field(ge=0)


class PredictiveConfidenceBin(BaseModel):
    label: str
    lower_bound: float = Field(ge=0.0, le=1.0)
    upper_bound: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0)
    average_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    empirical_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    absolute_gap: Optional[float] = Field(None, ge=0.0, le=1.0)


class PredictiveCalibrationDay(BaseModel):
    date: str
    assessments: int = Field(ge=0)
    errors: int = Field(ge=0)
    labeled_errors: int = Field(ge=0)
    material_errors: int = Field(ge=0)
    confirmed_mismatches: int = Field(ge=0)
    false_conflicts: int = Field(ge=0)
    label_coverage: float = Field(ge=0.0, le=1.0)
    false_conflict_rate: Optional[float] = Field(None, ge=0.0, le=1.0)


class PredictiveCalibrationSummary(BaseModel):
    assessments: int = Field(ge=0)
    actionable_assessments: int = Field(ge=0)
    labeled_assessments: int = Field(ge=0)
    assessment_label_coverage: float = Field(ge=0.0, le=1.0)
    errors: int = Field(ge=0)
    labeled_errors: int = Field(ge=0)
    error_label_coverage: float = Field(ge=0.0, le=1.0)
    material_errors: int = Field(ge=0)
    mismatch_confusion_matrix: Dict[str, int]
    mismatch_precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    mismatch_recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    false_conflict_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    hypothesis_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    observation_reliable_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    recommendation_usefulness_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    preferred_action_agreement: Optional[float] = Field(None, ge=0.0, le=1.0)
    expected_calibration_error: Optional[float] = Field(None, ge=0.0, le=1.0)
    assessment_status_counts: Dict[str, int]
    recommendation_counts: Dict[str, int]
    strata: Dict[str, PredictiveCalibrationStratum]
    confidence_bins: List[PredictiveConfidenceBin]
    daily: List[PredictiveCalibrationDay]
    ledger_integrity_verified: bool
    predictive_influence_eligible: Literal[False] = False
    eligibility_reason: str
