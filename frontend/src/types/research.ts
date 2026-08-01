export type InquiryStatus =
  | 'queued'
  | 'under_review'
  | 'resolved_locally'
  | 'approved'
  | 'researched'
  | 'research_failed'
  | 'dismissed'
  | 'expired';

export type CognitiveAction =
  | 'routine_local'
  | 'deepen_local'
  | 'ask_clarification'
  | 'acknowledge_uncertainty'
  | 'queue_inquiry'
  | 'authorize_research';

export interface ResearchSignals {
  epistemic_uncertainty: number;
  cognitive_conflict: number;
  novelty_prediction_error: number;
  temporal_volatility: number;
  task_stakes: number;
  persistence_after_local_attempts: number;
  expected_information_gain: number;
  privacy_risk: number;
  cloud_cost: number;
  explicit_user_request: boolean;
  metacognitive_gap: boolean;
  needs_clarification: boolean;
}

export interface ResearchAssessment {
  assessment_id: string;
  signals: ResearchSignals;
  drive_score: number;
  dominant_signals: string[];
  recommended_action: CognitiveAction;
  effective_action: CognitiveAction;
  shadow_mode: boolean;
  rationale: string;
  assessed_at: string;
}

export interface InquiryCandidate {
  inquiry_id: string;
  question: string;
  hypothesis?: string;
  source_type: 'waking' | 'reflection' | 'dream';
  assessment: ResearchAssessment;
  priority: number;
  expected_information_gain: number;
  status: InquiryStatus;
  shadow_mode: boolean;
  created_at: string;
  updated_at: string;
  expires_at: string;
  resolution?: string;
}

export interface ResearchSource {
  source_id: string;
  title: string;
  url: string;
}

export interface ResearchPacket {
  request_id: string;
  decision_id: string;
  status: 'completed' | 'failed';
  provider: string;
  model?: string;
  answer?: string;
  grounding_verified: boolean;
  confidence: number;
  latency_ms?: number;
  sources: ResearchSource[];
}

export interface LedgerEvent {
  sequence: number;
  event_id: string;
  event_type: string;
  assessment_id?: string;
  request_id?: string;
  created_at: string;
  payload: Record<string, any>;
  event_hash: string;
}

export interface InquiryDetail {
  candidate: InquiryCandidate;
  ledger_events: LedgerEvent[];
}

export interface RuntimeState {
  provider_enabled: boolean;
  controller_active: boolean;
  automatic_non_explicit_enabled: boolean;
  emergency_stop: boolean;
  provider_configured: boolean;
  provider_available: boolean;
  provider: string;
  model?: string;
  local_only: boolean;
  explicit_approval_required: boolean;
  automation_confirmation: string;
  persistence: string;
  changed_at: string;
}

export interface RuntimeUpdate {
  provider_enabled?: boolean;
  controller_active?: boolean;
  automatic_non_explicit_enabled?: boolean;
  emergency_stop?: boolean;
  reason: string;
  confirmation?: string;
}

export type SourceQualityVerdict =
  | 'trustworthy'
  | 'useful_with_caveats'
  | 'poor'
  | 'incorrect';

export interface SourceFeedback {
  request_id: string;
  source_id: string;
  verdict: SourceQualityVerdict;
  relevance: number;
  authority: number;
  freshness: number;
  citation_support: number;
  claim_supported: boolean;
  notes?: string;
  research_changed_answer?: boolean;
  research_resolved_inquiry?: boolean;
  worth_cost?: boolean;
}

export interface CalibrationStratum {
  observations: number;
  labeled: number;
  recommended_external: number;
  should_external: number;
  false_positive: number;
  false_negative: number;
}

export interface CalibrationSummary {
  observations: number;
  shadow_observations: number;
  labeled_observations: number;
  label_coverage: number;
  recommended_action_counts: Record<string, number>;
  external_research_confusion_matrix: Record<string, number>;
  external_research_precision?: number;
  external_research_recall?: number;
  recommended_action_accuracy?: number;
  calibration_strata: Record<string, CalibrationStratum>;
  review_outcome_counts: Record<string, number>;
  research_packet_counts: Record<string, number>;
  source_feedback_count: number;
  source_quality_averages: Record<string, number | null>;
  source_claim_support_rate?: number;
  research_changed_answer_rate?: number;
  research_resolved_inquiry_rate?: number;
  research_worth_cost_rate?: number;
  automatic_non_explicit_research_eligible: boolean;
  eligibility_reason: string;
  ledger_integrity_verified: boolean;
}
