export type PredictiveReviewStatus = 'unreviewed' | 'partially_reviewed' | 'reviewed' | 'not_applicable';
export type PredictionStatus = 'matched' | 'mismatch' | 'unobserved' | 'low_reliability';
export type Modality = 'text' | 'image' | 'audio';

export interface PerceptualHypothesis {
  hypothesis_id: string;
  label: 'prior_hypothesis_not_observation';
  source_cycle_id: string;
  source_reference: string;
  source_kind: string;
  feature_kind: 'presence' | 'categorical_attribute';
  feature_name: string;
  predicted_value: string;
  prior_confidence: number;
  reviewable_modalities: Modality[];
  formed_from_prior_context_only: true;
  semantic_truth_verified: false;
  created_at: string;
}

export interface PredictionError {
  error_id: string;
  hypothesis_id: string;
  sensory_episode_id: string;
  feature_kind: 'presence' | 'categorical_attribute';
  feature_name: string;
  predicted_value: string;
  observed_value?: string | null;
  observed_modality?: Modality | null;
  observation_reference?: string | null;
  status: PredictionStatus;
  direction: string;
  signed_error: number;
  surprise_magnitude: number;
  prior_confidence: number;
  observation_reliability?: number | null;
  calibration_eligible: boolean;
  material: boolean;
  derived_only: true;
  primary_evidence_changed: false;
}

export interface ClarificationRecommendation {
  recommendation_id: string;
  action: 'ask_user' | 'request_image_recapture' | 'request_audio_recapture';
  reason: string;
  target_modalities: Modality[];
  prompt: string;
  priority: number;
  expected_information_gain: number;
  source_error_ids: string[];
  source_relation_indexes: number[];
  shadow_only: true;
  executed: false;
  cloud_research_allowed: false;
}

export interface PredictiveAssessment {
  schema_version: 'predictive-perception-v1';
  assessment_id: string;
  cycle_id: string;
  sensory_episode_id: string;
  enabled: boolean;
  assessment_status: 'assessed' | 'disabled' | 'degraded';
  degradation_reason?: string | null;
  shadow_mode: true;
  prior_cycle_ids: string[];
  hypotheses: PerceptualHypothesis[];
  prediction_errors: PredictionError[];
  recommendation?: ClarificationRecommendation | null;
  hypothesis_count: number;
  matched_count: number;
  mismatch_count: number;
  unobserved_count: number;
  low_reliability_count: number;
  material_error_count: number;
  response_influenced: false;
  routing_influenced: false;
  research_invoked: false;
  learning_update_applied: false;
  primary_evidence_rewritten: false;
  assessed_at: string;
}

export interface PredictiveLedgerEvent {
  sequence: number;
  event_id: string;
  event_type: 'assessment_recorded' | 'calibration_label';
  user_id: string;
  cycle_id: string;
  assessment_id: string;
  error_id?: string | null;
  created_at: string;
  payload: Record<string, any>;
  previous_hash: string;
  event_hash: string;
}

export interface PredictiveAssessmentReview {
  assessment: PredictiveAssessment;
  recorded_at: string;
  ledger_sequence: number;
  review_status: PredictiveReviewStatus;
  review_target_count: number;
  reviewed_target_count: number;
  latest_labels: PredictiveLedgerEvent[];
}

export type HypothesisVerdict = 'correct' | 'incorrect' | 'uncertain' | 'not_reviewed';
export type ObservationQualityVerdict = 'reliable' | 'unreliable' | 'insufficient' | 'uncertain' | 'not_reviewed';
export type PredictionOutcomeVerdict = 'confirmed_match' | 'confirmed_mismatch' | 'false_conflict' | 'missed_mismatch' | 'indeterminate';
export type RecommendationVerdict = 'useful' | 'unnecessary' | 'wrong_action' | 'not_applicable' | 'uncertain';
export type PreferredAction = 'none' | 'ask_user' | 'request_image_recapture' | 'request_audio_recapture';

export interface PredictiveLabelRequest {
  error_id?: string;
  hypothesis_verdict: HypothesisVerdict;
  observation_quality: ObservationQualityVerdict;
  prediction_outcome: PredictionOutcomeVerdict;
  recommendation_verdict: RecommendationVerdict;
  preferred_action: PreferredAction;
  outcome_known: boolean;
  rationale: string;
}

export interface PredictiveCalibrationStratum {
  observations: number;
  labeled: number;
  predicted_mismatch: number;
  confirmed_mismatch: number;
  false_conflict: number;
  missed_mismatch: number;
  hypothesis_correct: number;
  hypothesis_incorrect: number;
  recommendation_reviewed: number;
  recommendation_useful: number;
}

export interface PredictiveConfidenceBin {
  label: string;
  lower_bound: number;
  upper_bound: number;
  count: number;
  average_confidence?: number | null;
  empirical_accuracy?: number | null;
  absolute_gap?: number | null;
}

export interface PredictiveCalibrationDay {
  date: string;
  assessments: number;
  errors: number;
  labeled_errors: number;
  material_errors: number;
  confirmed_mismatches: number;
  false_conflicts: number;
  label_coverage: number;
  false_conflict_rate?: number | null;
}

export interface PredictiveCalibrationSummary {
  assessments: number;
  actionable_assessments: number;
  labeled_assessments: number;
  assessment_label_coverage: number;
  errors: number;
  labeled_errors: number;
  error_label_coverage: number;
  material_errors: number;
  mismatch_confusion_matrix: Record<string, number>;
  mismatch_precision?: number | null;
  mismatch_recall?: number | null;
  false_conflict_rate?: number | null;
  hypothesis_accuracy?: number | null;
  observation_reliable_rate?: number | null;
  recommendation_usefulness_rate?: number | null;
  preferred_action_agreement?: number | null;
  expected_calibration_error?: number | null;
  assessment_status_counts: Record<string, number>;
  recommendation_counts: Record<string, number>;
  strata: Record<string, PredictiveCalibrationStratum>;
  confidence_bins: PredictiveConfidenceBin[];
  daily: PredictiveCalibrationDay[];
  ledger_integrity_verified: boolean;
  predictive_influence_eligible: false;
  eligibility_reason: string;
}
