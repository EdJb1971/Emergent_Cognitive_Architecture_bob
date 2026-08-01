import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  FiActivity,
  FiAlertOctagon,
  FiArrowUpRight,
  FiCheck,
  FiChevronRight,
  FiClock,
  FiDatabase,
  FiExternalLink,
  FiRefreshCw,
  FiSearch,
  FiShield,
  FiSliders,
  FiX,
  FiZap,
} from 'react-icons/fi';
import {
  approveInquiry,
  dismissInquiry,
  getCalibrationSummary,
  getInquiries,
  getInquiry,
  getResearchLedger,
  getResearchRuntime,
  labelAssessment,
  recordSourceFeedback,
  retryInquiry,
  updateResearchRuntime,
} from 'api/researchApi';
import {
  CalibrationSummary,
  CognitiveAction,
  InquiryCandidate,
  InquiryDetail,
  InquiryStatus,
  LedgerEvent,
  ResearchPacket,
  ResearchSource,
  RuntimeState,
  RuntimeUpdate,
  SourceFeedback,
  SourceQualityVerdict,
} from 'types/research';

export type ResearchView = 'command' | 'inquiries' | 'calibration' | 'ledger';

interface Props {
  view: ResearchView;
}

const percent = (value?: number | null) =>
  value === undefined || value === null ? '—' : `${Math.round(value * 100)}%`;

const humanize = (value: string) =>
  value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());

const statusTone = (status: InquiryStatus) => {
  if (status === 'researched' || status === 'resolved_locally') return 'positive';
  if (status === 'research_failed' || status === 'expired') return 'danger';
  if (status === 'approved' || status === 'under_review') return 'amber';
  if (status === 'dismissed') return 'muted';
  return 'cyan';
};

const eventLabel: Record<string, string> = {
  shadow_assessment: 'Shadow observation',
  waking_revalidation: 'Waking revalidation',
  review_requested: 'Operator decision',
  review_resolved: 'Review resolved',
  research_decision: 'Policy decision',
  research_packet: 'Grounded packet',
  source_feedback: 'Source review',
  calibration_label: 'Calibration label',
  runtime_control_changed: 'Runtime control',
};

const openStatuses: InquiryStatus[] = ['queued', 'under_review', 'approved', 'research_failed'];

const freshSourceFeedback = (): Omit<SourceFeedback, 'request_id' | 'source_id'> => ({
  verdict: 'trustworthy',
  relevance: 4,
  authority: 4,
  freshness: 4,
  citation_support: 4,
  claim_supported: true,
  research_changed_answer: true,
  research_resolved_inquiry: true,
  worth_cost: true,
});

const Toggle: React.FC<{
  active: boolean;
  disabled?: boolean;
  label: string;
  onChange: () => void;
}> = ({ active, disabled, label, onChange }) => (
  <button
    type="button"
    role="switch"
    aria-checked={active}
    aria-label={label}
    disabled={disabled}
    onClick={onChange}
    className={`control-toggle ${active ? 'is-on' : ''}`}
  >
    <span />
  </button>
);

const Metric: React.FC<{
  eyebrow: string;
  value: React.ReactNode;
  detail: string;
  tone?: string;
}> = ({ eyebrow, value, detail, tone = 'cyan' }) => (
  <article className={`ops-metric tone-${tone}`}>
    <span className="ops-kicker">{eyebrow}</span>
    <strong>{value}</strong>
    <p>{detail}</p>
  </article>
);

const ResearchCommandCenter: React.FC<Props> = ({ view }) => {
  const [runtime, setRuntime] = useState<RuntimeState | null>(null);
  const [calibration, setCalibration] = useState<CalibrationSummary | null>(null);
  const [inquiries, setInquiries] = useState<InquiryCandidate[]>([]);
  const [ledger, setLedger] = useState<LedgerEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'open' | InquiryStatus>('open');
  const [selected, setSelected] = useState<InquiryDetail | null>(null);
  const [confirmAutomation, setConfirmAutomation] = useState(false);
  const [confirmationText, setConfirmationText] = useState('');
  const [emergencyConfirm, setEmergencyConfirm] = useState(false);
  const [action, setAction] = useState<{ type: 'approve' | 'dismiss' | 'retry'; item: InquiryCandidate } | null>(null);
  const [actionReason, setActionReason] = useState('');
  const [labelTarget, setLabelTarget] = useState<LedgerEvent | null>(null);
  const [labelAction, setLabelAction] = useState<CognitiveAction>('authorize_research');
  const [labelShouldResearch, setLabelShouldResearch] = useState(true);
  const [labelRationale, setLabelRationale] = useState('');
  const [sourceTarget, setSourceTarget] = useState<{ inquiry: InquiryCandidate; packet: ResearchPacket; source: ResearchSource } | null>(null);
  const [sourceNotes, setSourceNotes] = useState('');
  const [sourceFeedback, setSourceFeedback] = useState(freshSourceFeedback);

  const refresh = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    setError(null);
    const results = await Promise.allSettled([
      getResearchRuntime(),
      getCalibrationSummary(),
      getInquiries(),
      getResearchLedger(200),
    ]);
    const failures = results.filter((result) => result.status === 'rejected');
    if (results[0].status === 'fulfilled') setRuntime(results[0].value);
    if (results[1].status === 'fulfilled') setCalibration(results[1].value);
    if (results[2].status === 'fulfilled') setInquiries(results[2].value.inquiries);
    if (results[3].status === 'fulfilled') setLedger(results[3].value.events);
    if (failures.length) {
      const first = failures[0] as PromiseRejectedResult;
      setError(first.reason?.message || 'The control plane is unavailable.');
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (!notice) return;
    const timeout = window.setTimeout(() => setNotice(null), 4200);
    return () => window.clearTimeout(timeout);
  }, [notice]);

  const mutateRuntime = async (update: RuntimeUpdate, success: string) => {
    setBusy(true);
    setError(null);
    try {
      setRuntime(await updateResearchRuntime(update));
      setNotice(success);
      await refresh(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Runtime update failed.');
    } finally {
      setBusy(false);
    }
  };

  const openInquiry = async (item: InquiryCandidate) => {
    setBusy(true);
    try {
      setSelected(await getInquiry(item.inquiry_id));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inquiry details could not be loaded.');
    } finally {
      setBusy(false);
    }
  };

  const runInquiryAction = async () => {
    if (!action || !actionReason.trim()) return;
    setBusy(true);
    try {
      if (action.type === 'approve') {
        await approveInquiry(action.item.inquiry_id, actionReason.trim());
      } else if (action.type === 'dismiss') {
        await dismissInquiry(action.item.inquiry_id, actionReason.trim());
      } else {
        await retryInquiry(action.item.inquiry_id, actionReason.trim());
      }
      setNotice(`${humanize(action.type)} recorded for the inquiry.`);
      setAction(null);
      setActionReason('');
      setSelected(null);
      await refresh(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inquiry action failed.');
    } finally {
      setBusy(false);
    }
  };

  const submitLabel = async () => {
    if (!labelTarget?.assessment_id || !labelRationale.trim()) return;
    setBusy(true);
    try {
      await labelAssessment(
        labelTarget.assessment_id,
        labelAction,
        labelShouldResearch,
        labelRationale.trim(),
      );
      setNotice('Independent calibration label added to the ledger.');
      setLabelTarget(null);
      setLabelRationale('');
      await refresh(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Calibration label failed.');
    } finally {
      setBusy(false);
    }
  };

  const submitSourceReview = async () => {
    if (!sourceTarget) return;
    setBusy(true);
    try {
      await recordSourceFeedback(sourceTarget.inquiry.inquiry_id, {
        request_id: sourceTarget.packet.request_id,
        source_id: sourceTarget.source.source_id,
        ...sourceFeedback,
        notes: sourceNotes.trim() || undefined,
      });
      setNotice('Source-quality review appended.');
      setSourceTarget(null);
      setSourceNotes('');
      setSourceFeedback(freshSourceFeedback());
      await refresh(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Source review failed.');
    } finally {
      setBusy(false);
    }
  };

  const filtered = useMemo(() => inquiries.filter((item) => {
    const statusMatch = statusFilter === 'all'
      || (statusFilter === 'open' && openStatuses.includes(item.status))
      || item.status === statusFilter;
    const textMatch = !query.trim()
      || `${item.question} ${item.hypothesis || ''}`.toLowerCase().includes(query.toLowerCase());
    return statusMatch && textMatch;
  }), [inquiries, query, statusFilter]);

  const labeledIds = useMemo(() => new Set(
    ledger.filter((event) => event.event_type === 'calibration_label').map((event) => event.assessment_id),
  ), [ledger]);
  const unlabeled = ledger.filter((event) =>
    ['shadow_assessment', 'waking_revalidation'].includes(event.event_type)
    && event.assessment_id
    && !labeledIds.has(event.assessment_id)
  ).reverse();

  const verifiedPackets = selected?.ledger_events
    .filter((event) => event.event_type === 'research_packet')
    .map((event) => event.payload.packet as ResearchPacket)
    .filter((packet) => packet?.status === 'completed' && packet.grounding_verified) || [];

  if (loading && !runtime) {
    return <div className="ops-loading"><span /><p>Synchronising the cognitive control plane…</p></div>;
  }

  const banner = (error || notice) && (
    <div className={`ops-banner ${error ? 'is-error' : 'is-success'}`}>
      {error ? <FiAlertOctagon /> : <FiCheck />}
      <span>{error || notice}</span>
      <button onClick={() => { setError(null); setNotice(null); }} aria-label="Dismiss message"><FiX /></button>
    </div>
  );

  if (view === 'command') {
    return (
      <div className="ops-stack">
        {banner}
        <section className="ops-hero">
          <div>
            <span className="ops-overline"><FiActivity /> live governance</span>
            <h2>Research control plane</h2>
            <p>Move deliberately from local cognition to grounded external evidence. Every change is authenticated, restored after restart, and written to the immutable ledger.</p>
          </div>
          <div className={`runtime-orbit ${runtime?.automatic_non_explicit_enabled ? 'is-live' : ''}`}>
            <span className="orbit-core"><FiZap /></span>
            <span className="orbit-copy"><b>{runtime?.automatic_non_explicit_enabled ? 'AUTONOMOUS' : runtime?.controller_active ? 'ACTIVE' : 'SHADOW'}</b><small>research posture</small></span>
          </div>
        </section>

        <section className="control-grid">
          <article className="control-card">
            <div className="control-icon"><FiExternalLink /></div>
            <div className="control-copy"><strong>Grounded provider</strong><span>{runtime?.provider_configured ? `${runtime.provider} · ${runtime.model}` : 'Gemini is not configured'}</span></div>
            <Toggle
              active={!!runtime?.provider_enabled}
              disabled={busy || !runtime?.provider_configured || !!runtime?.local_only}
              label="Grounded provider"
              onChange={() => mutateRuntime(
                { provider_enabled: !runtime?.provider_enabled, controller_active: runtime?.provider_enabled ? false : undefined, automatic_non_explicit_enabled: runtime?.provider_enabled ? false : undefined, reason: runtime?.provider_enabled ? 'Operator disabled provider from control room.' : 'Operator enabled grounded provider from control room.' },
                runtime?.provider_enabled ? 'Grounded provider disabled.' : 'Grounded provider ready.',
              )}
            />
          </article>
          <article className="control-card">
            <div className="control-icon"><FiSliders /></div>
            <div className="control-copy"><strong>Active controller</strong><span>{runtime?.controller_active ? 'Recommendations can govern effort' : 'Observing real cycles in shadow'}</span></div>
            <Toggle
              active={!!runtime?.controller_active}
              disabled={busy || !runtime?.provider_enabled}
              label="Active research controller"
              onChange={() => mutateRuntime(
                { controller_active: !runtime?.controller_active, automatic_non_explicit_enabled: runtime?.controller_active ? false : undefined, reason: runtime?.controller_active ? 'Operator returned controller to shadow.' : 'Operator activated calibrated controller.' },
                runtime?.controller_active ? 'Controller returned to shadow.' : 'Controller is active.',
              )}
            />
          </article>
          <article className="control-card is-sensitive">
            <div className="control-icon"><FiZap /></div>
            <div className="control-copy"><strong>Non-explicit research</strong><span>{runtime?.automatic_non_explicit_enabled ? 'High-drive inquiries may research automatically' : 'Explicit approval remains required'}</span></div>
            <Toggle
              active={!!runtime?.automatic_non_explicit_enabled}
              disabled={busy || !runtime?.controller_active}
              label="Automatic non-explicit research"
              onChange={() => runtime?.automatic_non_explicit_enabled
                ? mutateRuntime({ automatic_non_explicit_enabled: false, reason: 'Operator disabled non-explicit automation.' }, 'Automatic research disabled.')
                : setConfirmAutomation(true)}
            />
          </article>
        </section>

        <div className="ops-metric-grid">
          <Metric eyebrow="Shadow observations" value={calibration?.shadow_observations || 0} detail={`${calibration?.labeled_observations || 0} independently labeled`} />
          <Metric eyebrow="Label coverage" value={percent(calibration?.label_coverage)} detail="Representative evidence, not synthetic scores" tone="violet" />
          <Metric eyebrow="Research precision" value={percent(calibration?.external_research_precision)} detail="Of proposed external escalations" tone="amber" />
          <Metric eyebrow="Ledger integrity" value={calibration?.ledger_integrity_verified ? 'Verified' : 'Attention'} detail="SHA-256 event chain" tone={calibration?.ledger_integrity_verified ? 'green' : 'red'} />
        </div>

        <section className="ops-panel split-panel">
          <div>
            <div className="section-heading"><div><span>Readiness</span><h3>Activation evidence</h3></div><FiShield /></div>
            <div className="readiness-track"><span style={{ width: `${Math.min(100, (calibration?.label_coverage || 0) * 100)}%` }} /></div>
            <p className="panel-note">{calibration?.eligibility_reason || 'Collect real-cycle labels before considering automatic research.'}</p>
          </div>
          <div className="kill-zone">
            <span className="ops-kicker">Immediate containment</span>
            <strong>Emergency stop</strong>
            <p>Disable provider access, return the controller to shadow, and require explicit approval.</p>
            {!emergencyConfirm ? (
              <button className="danger-button" onClick={() => setEmergencyConfirm(true)}><FiAlertOctagon /> Arm emergency stop</button>
            ) : (
              <div className="confirm-row"><button onClick={() => setEmergencyConfirm(false)}>Cancel</button><button className="danger-button" onClick={() => { setEmergencyConfirm(false); mutateRuntime({ emergency_stop: true, reason: 'Operator emergency stop.' }, 'Emergency stop engaged.'); }}>Confirm stop</button></div>
            )}
          </div>
        </section>

        {confirmAutomation && runtime && (
          <div className="ops-dialog-backdrop">
            <div className="ops-dialog danger-dialog">
              <span className="dialog-icon"><FiZap /></span><h3>Enable autonomous research?</h3>
              <p>This permits strong non-explicit waking, reflection, and dream inquiries to cross the existing cognitive and policy gates without case-by-case approval.</p>
              <label>Type <b>{runtime.automation_confirmation}</b> to continue</label>
              <input autoFocus value={confirmationText} onChange={(event) => setConfirmationText(event.target.value)} />
              <div className="dialog-actions"><button onClick={() => { setConfirmAutomation(false); setConfirmationText(''); }}>Keep approval required</button><button className="primary-button" disabled={confirmationText !== runtime.automation_confirmation || busy} onClick={() => { setConfirmAutomation(false); mutateRuntime({ automatic_non_explicit_enabled: true, confirmation: confirmationText, reason: 'Operator enabled non-explicit automation after reviewing calibration.' }, 'Automatic non-explicit research enabled.'); setConfirmationText(''); }}>Enable automation</button></div>
            </div>
          </div>
        )}
      </div>
    );
  }

  if (view === 'inquiries') {
    return (
      <div className="ops-stack">
        {banner}
        <div className="workspace-heading"><div><span className="ops-overline"><FiSearch /> waking review</span><h2>Inquiry workspace</h2><p>Inspect unresolved questions and make explicit, auditable decisions.</p></div><button className="icon-button" onClick={() => refresh()} title="Refresh"><FiRefreshCw /></button></div>
        <div className="inquiry-toolbar">
          <label className="search-field"><FiSearch /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search questions and hypotheses" /></label>
          <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as any)}><option value="open">Open queue</option><option value="all">All states</option><option value="queued">Queued</option><option value="research_failed">Failed</option><option value="researched">Researched</option><option value="dismissed">Dismissed</option></select>
          <span className="result-count">{filtered.length} inquiries</span>
        </div>
        <div className="inquiry-layout">
          <div className="inquiry-list">
            {filtered.length === 0 ? <div className="empty-state"><FiDatabase /><h3>The queue is clear</h3><p>New waking, reflection, and dream inquiries will appear here.</p></div> : filtered.map((item) => (
              <button key={item.inquiry_id} className={`inquiry-row ${selected?.candidate.inquiry_id === item.inquiry_id ? 'is-selected' : ''}`} onClick={() => openInquiry(item)}>
                <div className="inquiry-score"><span style={{ '--score': `${item.priority * 360}deg` } as React.CSSProperties}>{Math.round(item.priority * 100)}</span></div>
                <div className="inquiry-main"><div><span className={`status-badge tone-${statusTone(item.status)}`}>{humanize(item.status)}</span><span className="source-label">{item.source_type}</span></div><strong>{item.question}</strong><p>{item.hypothesis || item.assessment.rationale}</p></div>
                <div className="inquiry-time"><FiClock />{new Date(item.updated_at).toLocaleDateString()}<FiChevronRight /></div>
              </button>
            ))}
          </div>
          <aside className={`inquiry-detail ${selected ? 'is-open' : ''}`}>
            {!selected ? <div className="detail-placeholder"><FiArrowUpRight /><p>Select an inquiry to inspect its evidence, signals, and immutable history.</p></div> : <>
              <div className="detail-top"><div><span className={`status-badge tone-${statusTone(selected.candidate.status)}`}>{humanize(selected.candidate.status)}</span><span className="source-label">{selected.candidate.source_type}</span></div><button onClick={() => setSelected(null)}><FiX /></button></div>
              <h3>{selected.candidate.question}</h3><p className="detail-hypothesis">{selected.candidate.hypothesis || 'No hypothesis was attached to this inquiry.'}</p>
              <div className="signal-grid"><div><span>Drive</span><b>{percent(selected.candidate.assessment.drive_score)}</b></div><div><span>Information gain</span><b>{percent(selected.candidate.expected_information_gain)}</b></div><div><span>Recommended</span><b>{humanize(selected.candidate.assessment.recommended_action)}</b></div><div><span>Mode</span><b>{selected.candidate.assessment.shadow_mode ? 'Shadow' : 'Active'}</b></div></div>
              <div className="detail-actions">
                {selected.candidate.status === 'queued' && <button className="primary-button" onClick={() => { setAction({ type: 'approve', item: selected.candidate }); setActionReason('Approved after waking review.'); }}><FiCheck /> Approve & revalidate</button>}
                {selected.candidate.status === 'research_failed' && <button className="primary-button" onClick={() => { setAction({ type: 'retry', item: selected.candidate }); setActionReason('Retry after reviewing the previous failure.'); }}><FiRefreshCw /> Re-queue</button>}
                {openStatuses.includes(selected.candidate.status) && <button className="secondary-button" onClick={() => { setAction({ type: 'dismiss', item: selected.candidate }); setActionReason('Dismissed after operator review.'); }}>Dismiss</button>}
              </div>
              {verifiedPackets.map((packet) => <div className="source-cluster" key={packet.request_id}><span className="ops-kicker">Verified sources</span>{packet.sources.map((source) => <div className="source-row" key={source.source_id}><a href={source.url} target="_blank" rel="noreferrer"><span>{source.title}</span><small>{new URL(source.url).hostname}</small></a><button onClick={() => { setSourceTarget({ inquiry: selected.candidate, packet, source }); setSourceFeedback(freshSourceFeedback()); setSourceNotes(''); }}>Review</button></div>)}</div>)}
              <div className="timeline"><span className="ops-kicker">Decision history</span>{selected.ledger_events.slice().reverse().map((event) => <div className="timeline-event" key={event.event_id}><i /><div><b>{eventLabel[event.event_type] || humanize(event.event_type)}</b><span>{new Date(event.created_at).toLocaleString()}</span><p>{event.payload.reason || event.payload.rationale || event.payload.resolution || `Ledger event #${event.sequence}`}</p></div></div>)}</div>
            </>}
          </aside>
        </div>
        {action && <div className="ops-dialog-backdrop"><div className="ops-dialog"><span className="ops-kicker">{humanize(action.type)} inquiry</span><h3>Record the reason</h3><p>{action.item.question}</p><textarea autoFocus rows={4} value={actionReason} onChange={(event) => setActionReason(event.target.value)} /><div className="dialog-actions"><button onClick={() => setAction(null)}>Cancel</button><button className="primary-button" disabled={!actionReason.trim() || busy} onClick={runInquiryAction}>Record decision</button></div></div></div>}
        {sourceTarget && <div className="ops-dialog-backdrop"><div className="ops-dialog source-review-dialog"><span className="ops-kicker">Source quality</span><h3>{sourceTarget.source.title}</h3><p>Record what this source actually contributed. This review is appended, never overwritten.</p><label>Overall verdict</label><select value={sourceFeedback.verdict} onChange={(event) => setSourceFeedback((current) => ({ ...current, verdict: event.target.value as SourceQualityVerdict }))}><option value="trustworthy">Trustworthy</option><option value="useful_with_caveats">Useful with caveats</option><option value="poor">Poor</option><option value="incorrect">Incorrect</option></select><div className="quality-grid">{(['relevance', 'authority', 'freshness', 'citation_support'] as const).map((dimension) => <label key={dimension}><span>{humanize(dimension)} <b>{sourceFeedback[dimension]}/5</b></span><input type="range" min="1" max="5" value={sourceFeedback[dimension]} onChange={(event) => setSourceFeedback((current) => ({ ...current, [dimension]: Number(event.target.value) }))} /></label>)}</div><div className="source-outcomes"><label className="check-line"><input type="checkbox" checked={sourceFeedback.claim_supported} onChange={(event) => setSourceFeedback((current) => ({ ...current, claim_supported: event.target.checked }))} /> Cited claim was supported</label><label className="check-line"><input type="checkbox" checked={sourceFeedback.research_changed_answer} onChange={(event) => setSourceFeedback((current) => ({ ...current, research_changed_answer: event.target.checked }))} /> Research changed the answer</label><label className="check-line"><input type="checkbox" checked={sourceFeedback.research_resolved_inquiry} onChange={(event) => setSourceFeedback((current) => ({ ...current, research_resolved_inquiry: event.target.checked }))} /> Inquiry was resolved</label><label className="check-line"><input type="checkbox" checked={sourceFeedback.worth_cost} onChange={(event) => setSourceFeedback((current) => ({ ...current, worth_cost: event.target.checked }))} /> Evidence was worth the cost</label></div><textarea rows={3} placeholder="Optional review notes" value={sourceNotes} onChange={(event) => setSourceNotes(event.target.value)} /><div className="dialog-actions"><button onClick={() => setSourceTarget(null)}>Cancel</button><button className="primary-button" disabled={busy} onClick={submitSourceReview}>Append source review</button></div></div></div>}
      </div>
    );
  }

  if (view === 'calibration') {
    return (
      <div className="ops-stack">
        {banner}
        <div className="workspace-heading"><div><span className="ops-overline"><FiActivity /> evidence lab</span><h2>Shadow calibration</h2><p>Compare the controller’s proposed effort with independent human judgment.</p></div><button className="icon-button" onClick={() => refresh()}><FiRefreshCw /></button></div>
        <div className="ops-metric-grid"><Metric eyebrow="Observations" value={calibration?.observations || 0} detail={`${unlabeled.length} await review`} /><Metric eyebrow="Action accuracy" value={percent(calibration?.recommended_action_accuracy)} detail="Exact action agreement" tone="violet" /><Metric eyebrow="Escalation recall" value={percent(calibration?.external_research_recall)} detail="Research-worthy cases detected" tone="amber" /><Metric eyebrow="Source value" value={percent(calibration?.research_worth_cost_rate)} detail={`${calibration?.source_feedback_count || 0} source reviews`} tone="green" /></div>
        <section className="ops-panel"><div className="section-heading"><div><span>Coverage by condition</span><h3>Calibration strata</h3></div><span className={`integrity-pill ${calibration?.ledger_integrity_verified ? 'is-good' : ''}`}><FiShield /> {calibration?.ledger_integrity_verified ? 'Chain verified' : 'Check ledger'}</span></div><div className="strata-table"><div className="strata-head"><span>Condition</span><span>Observed</span><span>Labeled</span><span>False +</span><span>False −</span></div>{Object.entries(calibration?.calibration_strata || {}).map(([name, stratum]) => <div className="strata-row" key={name}><strong>{humanize(name)}</strong><span>{stratum.observations}</span><span>{stratum.labeled}</span><span>{stratum.false_positive}</span><span>{stratum.false_negative}</span></div>)}</div></section>
        <section className="ops-panel"><div className="section-heading"><div><span>Independent review queue</span><h3>Unlabeled observations</h3></div><span className="result-count">{unlabeled.length} pending</span></div><div className="observation-list">{unlabeled.length === 0 ? <div className="empty-state compact"><FiCheck /><h3>All caught up</h3><p>Run ordinary conversations in shadow mode to collect more observations.</p></div> : unlabeled.slice(0, 20).map((event) => { const assessment = event.payload.assessment; return <button className="observation-row" key={event.event_id} onClick={() => { setLabelTarget(event); setLabelAction(assessment.recommended_action); setLabelShouldResearch(assessment.recommended_action === 'authorize_research'); }}><div><span>{event.event_type === 'shadow_assessment' ? 'Real cycle' : 'Waking review'}</span><strong>{humanize(assessment.recommended_action)}</strong><p>{assessment.rationale}</p></div><div className="observation-score"><b>{percent(assessment.drive_score)}</b><small>drive</small><FiChevronRight /></div></button>; })}</div></section>
        {labelTarget && <div className="ops-dialog-backdrop"><div className="ops-dialog"><span className="ops-kicker">Independent calibration</span><h3>What should the brain have done?</h3><label>Appropriate cognitive action</label><select value={labelAction} onChange={(event) => setLabelAction(event.target.value as CognitiveAction)}>{['routine_local', 'deepen_local', 'ask_clarification', 'acknowledge_uncertainty', 'queue_inquiry', 'authorize_research'].map((choice) => <option key={choice} value={choice}>{humanize(choice)}</option>)}</select><label className="check-line"><input type="checkbox" checked={labelShouldResearch} onChange={(event) => setLabelShouldResearch(event.target.checked)} /> External research was warranted</label><textarea rows={4} placeholder="Explain the independent judgment" value={labelRationale} onChange={(event) => setLabelRationale(event.target.value)} /><div className="dialog-actions"><button onClick={() => setLabelTarget(null)}>Cancel</button><button className="primary-button" disabled={!labelRationale.trim() || busy} onClick={submitLabel}>Append label</button></div></div></div>}
      </div>
    );
  }

  return (
    <div className="ops-stack">
      {banner}
      <div className="workspace-heading"><div><span className="ops-overline"><FiDatabase /> immutable audit</span><h2>Decision ledger</h2><p>A hash-chained record of every assessment, decision, packet, review, and runtime change.</p></div><button className="icon-button" onClick={() => refresh()}><FiRefreshCw /></button></div>
      <section className="ops-panel"><div className="ledger-header"><span>Event</span><span>Context</span><span>Time</span><span>Chain</span></div><div className="ledger-list">{ledger.slice().reverse().map((event) => <article className="ledger-row" key={event.event_id}><div><i className={`event-dot event-${event.event_type}`} /><span><b>{eventLabel[event.event_type] || humanize(event.event_type)}</b><small>#{event.sequence}</small></span></div><p>{event.payload.reason || event.payload.rationale || event.payload.disposition || event.assessment_id || 'Recorded governance event'}</p><time>{new Date(event.created_at).toLocaleString()}</time><code>{event.event_hash.slice(0, 10)}</code></article>)}</div></section>
    </div>
  );
};

export default ResearchCommandCenter;
