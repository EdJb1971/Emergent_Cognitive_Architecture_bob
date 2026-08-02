import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  FiActivity,
  FiAlertTriangle,
  FiBarChart2,
  FiBookOpen,
  FiCheck,
  FiChevronRight,
  FiClock,
  FiEye,
  FiLock,
  FiRefreshCw,
  FiSearch,
  FiShield,
  FiTag,
  FiTarget,
  FiX,
} from 'react-icons/fi';
import {
  getPredictiveAssessment,
  getPredictiveAssessments,
  getPredictiveCalibration,
  getPredictiveLedger,
  labelPredictiveAssessment,
} from 'api/predictiveApi';
import {
  HypothesisVerdict,
  ObservationQualityVerdict,
  PredictionError,
  PredictionOutcomeVerdict,
  PredictiveAssessmentReview,
  PredictiveCalibrationSummary,
  PredictiveLabelRequest,
  PredictiveLedgerEvent,
  PredictiveReviewStatus,
  PreferredAction,
  RecommendationVerdict,
} from 'types/predictive';

type View = 'review' | 'calibration' | 'ledger';
type ReviewTarget = { kind: 'error'; error: PredictionError } | { kind: 'recommendation' };

const percent = (value?: number | null) =>
  value === undefined || value === null ? '—' : `${Math.round(value * 100)}%`;
const humanize = (value: string) =>
  value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
const short = (value: string) => value.slice(0, 8);

const reviewTone = (status: PredictiveReviewStatus) => {
  if (status === 'reviewed') return 'positive';
  if (status === 'partially_reviewed') return 'amber';
  if (status === 'not_applicable') return 'muted';
  return 'cyan';
};

const defaultLabel = (target: ReviewTarget): PredictiveLabelRequest => ({
  error_id: target.kind === 'error' ? target.error.error_id : undefined,
  hypothesis_verdict: 'not_reviewed',
  observation_quality: 'not_reviewed',
  prediction_outcome: 'indeterminate',
  recommendation_verdict: 'not_applicable',
  preferred_action: 'none',
  outcome_known: true,
  rationale: '',
});

const SelectField: React.FC<{
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}> = ({ label, value, options, onChange }) => (
  <label><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}>
    {options.map((option) => <option key={option} value={option}>{humanize(option)}</option>)}
  </select></label>
);

const Metric: React.FC<{ eyebrow: string; value: React.ReactNode; detail: string; tone?: string }> = ({ eyebrow, value, detail, tone = 'cyan' }) => (
  <article className={`ops-metric tone-${tone}`}><span className="ops-kicker">{eyebrow}</span><strong>{value}</strong><p>{detail}</p></article>
);

const PredictiveCalibrationCenter: React.FC = () => {
  const [view, setView] = useState<View>('review');
  const [assessments, setAssessments] = useState<PredictiveAssessmentReview[]>([]);
  const [calibration, setCalibration] = useState<PredictiveCalibrationSummary | null>(null);
  const [ledger, setLedger] = useState<PredictiveLedgerEvent[]>([]);
  const [selected, setSelected] = useState<PredictiveAssessmentReview | null>(null);
  const [filter, setFilter] = useState<'all' | PredictiveReviewStatus>('unreviewed');
  const [materialOnly, setMaterialOnly] = useState(false);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [target, setTarget] = useState<ReviewTarget | null>(null);
  const [label, setLabel] = useState<PredictiveLabelRequest | null>(null);

  const refresh = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    setError(null);
    const results = await Promise.allSettled([
      getPredictiveAssessments(filter === 'all' ? undefined : filter, materialOnly),
      getPredictiveCalibration(),
      getPredictiveLedger(),
    ]);
    if (results[0].status === 'fulfilled') setAssessments(results[0].value.assessments);
    if (results[1].status === 'fulfilled') setCalibration(results[1].value);
    if (results[2].status === 'fulfilled') setLedger(results[2].value.events.slice().reverse());
    const failure = results.find((result) => result.status === 'rejected') as PromiseRejectedResult | undefined;
    if (failure) setError(failure.reason?.message || 'Predictive calibration plane is unavailable.');
    setLoading(false);
  }, [filter, materialOnly]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!notice) return;
    const timeout = window.setTimeout(() => setNotice(null), 4200);
    return () => window.clearTimeout(timeout);
  }, [notice]);

  const inspect = async (item: PredictiveAssessmentReview) => {
    setSelected(item);
    try { setSelected(await getPredictiveAssessment(item.assessment.assessment_id)); }
    catch (failure) { setError(failure instanceof Error ? failure.message : 'Assessment could not be loaded.'); }
  };

  const openLabel = (nextTarget: ReviewTarget) => {
    setTarget(nextTarget);
    setLabel(defaultLabel(nextTarget));
  };

  const submitLabel = async () => {
    if (!selected || !label || !label.rationale.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await labelPredictiveAssessment(selected.assessment.assessment_id, label);
      const updated = await getPredictiveAssessment(selected.assessment.assessment_id);
      setSelected(updated);
      setTarget(null);
      setLabel(null);
      setNotice('Independent judgement appended. The original prediction and evidence remain unchanged.');
      await refresh(true);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Label could not be appended.');
    } finally { setBusy(false); }
  };

  const visible = useMemo(() => {
    const needle = query.toLowerCase().trim();
    if (!needle) return assessments;
    return assessments.filter(({ assessment }) => [
      assessment.assessment_id,
      assessment.cycle_id,
      ...assessment.hypotheses.flatMap((item) => [item.feature_name, item.predicted_value, item.source_reference]),
    ].some((value) => value.toLowerCase().includes(needle)));
  }, [assessments, query]);

  const latestByError = useMemo(() => new Map(
    (selected?.latest_labels || []).filter((event) => event.error_id).map((event) => [event.error_id!, event])
  ), [selected]);

  const renderReview = () => (
    <>
      <div className="predictive-toolbar">
        <div className="search-field"><FiSearch /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search feature, value, cycle or assessment" /></div>
        <select value={filter} onChange={(event) => setFilter(event.target.value as typeof filter)}>
          <option value="unreviewed">Needs review</option><option value="partially_reviewed">Partially reviewed</option><option value="reviewed">Reviewed</option><option value="not_applicable">Not applicable</option><option value="all">All assessments</option>
        </select>
        <label className="material-filter"><input type="checkbox" checked={materialOnly} onChange={(event) => setMaterialOnly(event.target.checked)} /><span>Material only</span></label>
        <span className="result-count">{visible.length} records</span>
      </div>
      <div className="predictive-review-layout">
        <section className="prediction-list">
          {loading ? <div className="empty-state compact"><FiRefreshCw className="spin-slow" /><h3>Loading assessments</h3></div> : visible.length === 0 ? <div className="empty-state"><FiCheck /><h3>Review queue clear</h3><p>New waking sensory predictions will appear here after a cognitive cycle.</p></div> : visible.map((item) => {
            const a = item.assessment;
            const first = a.hypotheses[0];
            return <button key={a.assessment_id} className={`prediction-row ${selected?.assessment.assessment_id === a.assessment_id ? 'is-selected' : ''}`} onClick={() => inspect(item)}>
              <span className={`prediction-state is-${a.material_error_count ? 'material' : a.mismatch_count ? 'mismatch' : 'quiet'}`}><FiTarget /></span>
              <span className="prediction-row-copy"><span><b className={`status-badge tone-${reviewTone(item.review_status)}`}>{humanize(item.review_status)}</b><small>#{item.ledger_sequence}</small></span><strong>{first ? `${humanize(first.feature_name)} · ${first.predicted_value}` : humanize(a.assessment_status)}</strong><p>{a.hypothesis_count} hypotheses · {a.mismatch_count} mismatches · {a.material_error_count} material</p></span>
              <span className="prediction-row-time"><time>{new Date(a.assessed_at).toLocaleDateString()}</time><small>{new Date(a.assessed_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</small><FiChevronRight /></span>
            </button>;
          })}
        </section>
        <section className="prediction-inspector">
          {!selected ? <div className="detail-placeholder"><FiEye /><h3>Select an assessment</h3><p>Inspect every prior, observation and immutable operator judgement in context.</p></div> : <AssessmentInspector review={selected} latestByError={latestByError} onReview={openLabel} />}
        </section>
      </div>
    </>
  );

  const renderCalibration = () => {
    const bins = calibration?.confidence_bins.filter((bin) => bin.count > 0) || [];
    const days = calibration?.daily.slice(-30) || [];
    const maxErrors = Math.max(1, ...days.map((day) => day.errors));
    return <>
      <div className="ops-metric-grid">
        <Metric eyebrow="Assessment coverage" value={percent(calibration?.assessment_label_coverage)} detail={`${calibration?.labeled_assessments || 0} of ${calibration?.actionable_assessments || 0} actionable cycles`} />
        <Metric eyebrow="Mismatch precision" value={percent(calibration?.mismatch_precision)} detail="Confirmed among predicted mismatches" tone="violet" />
        <Metric eyebrow="Mismatch recall" value={percent(calibration?.mismatch_recall)} detail="Detected among known mismatches" tone="green" />
        <Metric eyebrow="Calibration error" value={percent(calibration?.expected_calibration_error)} detail="Confidence-to-accuracy gap (lower is better)" tone="amber" />
      </div>
      <div className="predictive-calibration-grid">
        <section className="ops-panel calibration-history">
          <div className="section-heading"><div><span>30-day history</span><h3>Prediction error and label coverage</h3></div><FiActivity /></div>
          {days.length ? <div className="calibration-bars" aria-label="Daily prediction errors">
            {days.map((day) => <div key={day.date} title={`${day.date}: ${day.errors} errors, ${percent(day.label_coverage)} labelled`}><i style={{ height: `${Math.max(5, (day.errors / maxErrors) * 100)}%` }}><b style={{ height: `${day.label_coverage * 100}%` }} /></i><span>{day.date.slice(5)}</span></div>)}
          </div> : <div className="empty-state compact"><FiBarChart2 /><h3>Awaiting longitudinal evidence</h3><p>The series begins with real waking cycles.</p></div>}
          <div className="chart-legend"><span><i className="is-errors" /> prediction errors</span><span><i className="is-labels" /> reviewed proportion</span></div>
        </section>
        <section className="ops-panel confidence-panel">
          <div className="section-heading"><div><span>Reliability curve</span><h3>Confidence calibration</h3></div><FiTarget /></div>
          {bins.length ? <div className="confidence-bins">{bins.map((bin) => <div key={bin.label}><span>{bin.label}</span><i><b className="confidence-prior" style={{ width: percent(bin.average_confidence) }} /><b className="confidence-actual" style={{ width: percent(bin.empirical_accuracy) }} /></i><strong>{percent(bin.empirical_accuracy)}</strong><small>n={bin.count}</small></div>)}</div> : <div className="empty-state compact"><FiTarget /><h3>No labelled confidence bins</h3><p>Correct/incorrect hypothesis labels populate this curve.</p></div>}
          <div className="chart-legend"><span><i className="is-prior" /> predicted confidence</span><span><i className="is-actual" /> empirical accuracy</span></div>
        </section>
      </div>
      <section className="ops-panel predictive-strata">
        <div className="section-heading"><div><span>Diagnostic cohorts</span><h3>Where prediction quality changes</h3></div><FiBarChart2 /></div>
        <div className="predictive-strata-head"><span>Cohort</span><span>Observed</span><span>Labelled</span><span>Mismatch</span><span>False conflict</span><span>Hypothesis accuracy</span></div>
        {Object.entries(calibration?.strata || {}).slice(0, 18).map(([name, data]) => {
          const judged = data.hypothesis_correct + data.hypothesis_incorrect;
          return <div className="predictive-strata-row" key={name}><strong>{humanize(name.replace(/:/g, ' · '))}</strong><span>{data.observations}</span><span>{data.labeled}</span><span>{data.confirmed_mismatch}</span><span>{data.false_conflict}</span><span>{judged ? percent(data.hypothesis_correct / judged) : '—'}</span></div>;
        })}
        {!Object.keys(calibration?.strata || {}).length && <p className="panel-note">Cohorts appear after review labels are appended.</p>}
      </section>
    </>;
  };

  const renderLedger = () => <section className="ops-panel predictive-ledger">
    <div className="section-heading"><div><span>Append-only audit</span><h3>Predictive decision ledger</h3></div><span className={`integrity-pill ${calibration?.ledger_integrity_verified ? 'is-good' : ''}`}><FiShield /> {calibration?.ledger_integrity_verified ? 'Chain verified' : 'Verification unavailable'}</span></div>
    <div className="predictive-ledger-head"><span>Sequence / event</span><span>Target</span><span>Judgement</span><span>Recorded</span><span>Hash</span></div>
    {ledger.map((event) => <div className="predictive-ledger-row" key={event.event_id}><div><i className={`event-${event.event_type}`} /><span><b>#{event.sequence} · {humanize(event.event_type)}</b><small>{short(event.event_id)}</small></span></div><p>{short(event.assessment_id)}{event.error_id ? ` / ${short(event.error_id)}` : ''}</p><p>{event.event_type === 'calibration_label' ? humanize(String(event.payload.prediction_outcome || event.payload.recommendation_verdict || 'label')) : `${event.payload.hypothesis_count || 0} hypotheses`}</p><time>{new Date(event.created_at).toLocaleString()}</time><code>{event.event_hash.slice(0, 12)}…</code></div>)}
    {!ledger.length && <div className="empty-state compact"><FiBookOpen /><h3>No ledger events yet</h3></div>}
  </section>;

  return <div className="ops-stack predictive-workspace">
    <section className="predictive-hero">
      <div><span className="ops-overline"><FiEye /> predictive review plane</span><h2>Calibrate expectation against reality.</h2><p>Review labelled priors beside the observations that tested them. Every judgement is additive, attributable and permanently separate from primary evidence.</p></div>
      <div className="shadow-seal"><FiLock /><span><b>Influence locked</b><small>Shadow observation only</small></span></div>
    </section>
    <div className="predictive-safety"><FiShield /><div><b>No predictive signal enters attention or learning.</b><span>Response, routing, research, evidence and learning influence are structurally false. Longitudinal calibration is evidence for a future policy decision—not permission to activate.</span></div><span className="status-badge tone-positive">enforced</span></div>
    {error && <div className="ops-banner is-error"><FiAlertTriangle /> {error}<button onClick={() => setError(null)}>Dismiss</button></div>}
    {notice && <div className="ops-banner is-success"><FiCheck /> {notice}</div>}
    <div className="predictive-viewbar">
      <div role="tablist" aria-label="Predictive calibration views"><button className={view === 'review' ? 'is-active' : ''} onClick={() => setView('review')}><FiEye /> Review</button><button className={view === 'calibration' ? 'is-active' : ''} onClick={() => setView('calibration')}><FiBarChart2 /> Calibration</button><button className={view === 'ledger' ? 'is-active' : ''} onClick={() => setView('ledger')}><FiBookOpen /> Ledger</button></div>
      <button className="icon-button" onClick={() => refresh()} aria-label="Refresh predictive data"><FiRefreshCw /></button>
    </div>
    {view === 'review' ? renderReview() : view === 'calibration' ? renderCalibration() : renderLedger()}
    {target && label && selected && <LabelDialog target={target} assessment={selected} label={label} setLabel={setLabel} busy={busy} onClose={() => { setTarget(null); setLabel(null); }} onSubmit={submitLabel} />}
  </div>;
};

const AssessmentInspector: React.FC<{
  review: PredictiveAssessmentReview;
  latestByError: Map<string, PredictiveLedgerEvent>;
  onReview: (target: ReviewTarget) => void;
}> = ({ review, latestByError, onReview }) => {
  const a = review.assessment;
  const recommendationError = a.recommendation?.source_error_ids.length
    ? a.prediction_errors.find((item) => a.recommendation!.source_error_ids.includes(item.error_id))
    : undefined;
  return <>
    <div className="prediction-inspector-head"><div><span className={`status-badge tone-${reviewTone(review.review_status)}`}>{humanize(review.review_status)}</span><small>Ledger #{review.ledger_sequence}</small></div><time><FiClock /> {new Date(a.assessed_at).toLocaleString()}</time></div>
    <h3>Assessment {short(a.assessment_id)}</h3>
    <p className="assessment-provenance">Cycle {short(a.cycle_id)} · sensory episode {short(a.sensory_episode_id)} · {a.prior_cycle_ids.length} prior cycle{a.prior_cycle_ids.length === 1 ? '' : 's'}</p>
    <div className="immutability-strip"><FiLock /><span><b>Immutable evidence boundary</b><small>Prior hypotheses are not observations. Operator labels do not rewrite either.</small></span></div>
    <div className="assessment-counts"><div><span>Hypotheses</span><b>{a.hypothesis_count}</b></div><div><span>Matched</span><b>{a.matched_count}</b></div><div><span>Mismatched</span><b>{a.mismatch_count}</b></div><div><span>Material</span><b>{a.material_error_count}</b></div></div>
    <div className="prediction-error-list">{a.prediction_errors.map((predictionError) => {
      const hypothesis = a.hypotheses.find((item) => item.hypothesis_id === predictionError.hypothesis_id)!;
      const latest = latestByError.get(predictionError.error_id);
      return <article className={`prediction-error-card is-${predictionError.status} ${predictionError.material ? 'is-material' : ''}`} key={predictionError.error_id}>
        <div className="error-card-head"><div><span>{humanize(hypothesis.feature_name)}</span><b className={`status-badge tone-${predictionError.material ? 'danger' : predictionError.status === 'matched' ? 'positive' : 'amber'}`}>{predictionError.material ? 'Material mismatch' : humanize(predictionError.status)}</b></div><button className="secondary-button" onClick={() => onReview({ kind: 'error', error: predictionError })}><FiTag /> {latest ? 'Revise judgement' : 'Review'}</button></div>
        <div className="evidence-comparison"><div className="is-prior"><span>Prior prediction</span><strong>{predictionError.predicted_value}</strong><small>{percent(predictionError.prior_confidence)} confidence · {humanize(hypothesis.source_kind)}</small></div><FiChevronRight /><div className="is-observed"><span>Primary observation</span><strong>{predictionError.observed_value || 'Not observed'}</strong><small>{predictionError.observed_modality ? `${humanize(predictionError.observed_modality)} · ${percent(predictionError.observation_reliability)} reliable` : 'No qualifying observation'}</small></div></div>
        <div className="error-metadata"><span>surprise <b>{percent(predictionError.surprise_magnitude)}</b></span><span>direction <b>{humanize(predictionError.direction)}</b></span><span>eligible <b>{predictionError.calibration_eligible ? 'yes' : 'no'}</b></span></div>
        <p className="source-reference">Source: {hypothesis.source_reference}</p>
        {latest && <div className="latest-judgement"><FiCheck /><span><b>{humanize(String(latest.payload.prediction_outcome || 'reviewed'))}</b><small>{humanize(String(latest.payload.hypothesis_verdict || 'not reviewed'))} hypothesis · {humanize(String(latest.payload.observation_quality || 'not reviewed'))} observation</small></span><time>{new Date(latest.created_at).toLocaleDateString()}</time></div>}
      </article>;
    })}</div>
    {a.recommendation && <article className="recommendation-review"><div><span>Shadow recommendation</span><h4>{humanize(a.recommendation.action)}</h4><p>{a.recommendation.prompt}</p><small>{humanize(a.recommendation.reason)} · {percent(a.recommendation.expected_information_gain)} expected information gain · never executed</small></div><button className="secondary-button" onClick={() => onReview(recommendationError ? { kind: 'error', error: recommendationError } : { kind: 'recommendation' })}><FiTag /> Review action</button></article>}
    {!a.prediction_errors.length && <div className="empty-state compact"><FiTarget /><h3>No reviewable hypotheses</h3><p>This disabled or degraded assessment is retained for completeness.</p></div>}
  </>;
};

const LabelDialog: React.FC<{
  target: ReviewTarget;
  assessment: PredictiveAssessmentReview;
  label: PredictiveLabelRequest;
  setLabel: React.Dispatch<React.SetStateAction<PredictiveLabelRequest | null>>;
  busy: boolean;
  onClose: () => void;
  onSubmit: () => void;
}> = ({ target, assessment, label, setLabel, busy, onClose, onSubmit }) => {
  const update = <K extends keyof PredictiveLabelRequest>(key: K, value: PredictiveLabelRequest[K]) => setLabel({ ...label, [key]: value });
  const recommendation = assessment.assessment.recommendation;
  return <div className="ops-dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose(); }}><div className="ops-dialog predictive-label-dialog" role="dialog" aria-modal="true" aria-label="Append predictive calibration label">
    <div className="label-dialog-head"><div className="dialog-icon"><FiTag /></div><button onClick={onClose} aria-label="Close"><FiX /></button></div>
    <h3>{target.kind === 'error' ? `Review ${humanize(target.error.feature_name)}` : 'Review clarification action'}</h3>
    <p>This creates a new ledger event. Any earlier judgement remains in history and the assessment itself is never edited.</p>
    {target.kind === 'error' && <div className="dialog-comparison"><div><span>Predicted</span><b>{target.error.predicted_value}</b></div><FiChevronRight /><div><span>Observed</span><b>{target.error.observed_value || 'Not observed'}</b></div></div>}
    {target.kind === 'error' && <div className="label-field-grid">
      <SelectField label="Hypothesis" value={label.hypothesis_verdict} options={['not_reviewed', 'correct', 'incorrect', 'uncertain']} onChange={(value) => update('hypothesis_verdict', value as HypothesisVerdict)} />
      <SelectField label="Observation quality" value={label.observation_quality} options={['not_reviewed', 'reliable', 'unreliable', 'insufficient', 'uncertain']} onChange={(value) => update('observation_quality', value as ObservationQualityVerdict)} />
      <SelectField label="Prediction outcome" value={label.prediction_outcome} options={['indeterminate', 'confirmed_match', 'confirmed_mismatch', 'false_conflict', 'missed_mismatch']} onChange={(value) => update('prediction_outcome', value as PredictionOutcomeVerdict)} />
    </div>}
    {recommendation && <div className="label-field-grid is-recommendation">
      <SelectField label="Recommendation" value={label.recommendation_verdict} options={['not_applicable', 'useful', 'unnecessary', 'wrong_action', 'uncertain']} onChange={(value) => update('recommendation_verdict', value as RecommendationVerdict)} />
      <SelectField label="Preferred action" value={label.preferred_action} options={['none', 'ask_user', 'request_image_recapture', 'request_audio_recapture']} onChange={(value) => update('preferred_action', value as PreferredAction)} />
    </div>}
    <label className="outcome-known"><input type="checkbox" checked={label.outcome_known} onChange={(event) => update('outcome_known', event.target.checked)} /><span><b>Outcome is known</b><small>Clear this when the evidence cannot support a final judgement.</small></span></label>
    <label>Rationale<textarea rows={4} value={label.rationale} onChange={(event) => update('rationale', event.target.value)} placeholder="State the evidence for this independent judgement…" maxLength={2000} /></label>
    <div className="dialog-actions"><button onClick={onClose}>Cancel</button><button className="primary-button" disabled={busy || !label.rationale.trim()} onClick={onSubmit}>{busy ? 'Appending…' : 'Append judgement'}</button></div>
  </div></div>;
};

export default PredictiveCalibrationCenter;
