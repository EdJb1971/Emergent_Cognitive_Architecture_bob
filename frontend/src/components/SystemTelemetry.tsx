import React, { useEffect, useMemo, useState } from 'react';
import { FiActivity, FiAlertTriangle, FiRadio, FiRefreshCw } from 'react-icons/fi';
import { connectDashboardTelemetry, getDashboardMetrics } from 'api/dashboardApi';
import {
  DashboardMetrics,
  TelemetryConnectionState,
  TelemetryDomain,
  TelemetryEvent,
  TelemetryGap,
} from 'types/dashboard';

const domains: Array<{ id: TelemetryDomain; label: string; hint: string }> = [
  { id: 'cognitive', label: 'Cognition', hint: 'cycles & agents' },
  { id: 'memory', label: 'Memory', hint: 'retrieval & storage' },
  { id: 'research', label: 'Research', hint: 'decisions & packets' },
  { id: 'predictive', label: 'Predictive', hint: 'assessments & labels' },
  { id: 'salience', label: 'Salience', hint: 'attention ranking' },
  { id: 'sleep', label: 'Sleep', hint: 'consolidation runs' },
  { id: 'autonomous_work', label: 'Autonomy', hint: 'governed work' },
];

const humanize = (value: string) => value.replace(/_/g, ' ');

const eventSummary = (event: TelemetryEvent): string => {
  const payload = event.payload;
  if (event.domain === 'memory') {
    return `${String(payload.tier_accessed || 'memory')} · ${Math.round(Number(payload.retrieval_time_ms || 0))}ms`;
  }
  if (event.domain === 'cognitive' && Array.isArray(payload.agents_activated)) {
    return `${payload.agents_activated.length} neural agents recruited`;
  }
  if (event.domain === 'salience') {
    return `${Number(payload.candidate_count || 0)} candidates · ${payload.shadow_mode ? 'shadow' : 'active'}`;
  }
  if (event.domain === 'sleep') return String(payload.status || event.event_type);
  if (event.domain === 'autonomous_work') {
    return [payload.task_type, payload.reason].filter(Boolean).map(String).join(' · ') || 'Governor decision';
  }
  if (event.domain === 'research') {
    return payload.inquiry_id ? `Inquiry ${String(payload.inquiry_id).slice(0, 8)}` : 'Research control event';
  }
  if (event.domain === 'predictive') {
    if (event.event_type === 'calibration_label') {
      return payload.error_id ? `Error ${String(payload.error_id).slice(0, 8)} reviewed` : 'Recommendation reviewed';
    }
    return `${Number(payload.hypothesis_count || 0)} hypotheses · ${Number(payload.material_error_count || 0)} material`;
  }
  return event.cycle_id ? `Cycle ${event.cycle_id.slice(0, 8)}` : 'Operational signal';
};

const SystemTelemetry: React.FC = () => {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [events, setEvents] = useState<TelemetryEvent[]>([]);
  const [state, setState] = useState<TelemetryConnectionState>('connecting');
  const [gap, setGap] = useState<TelemetryGap | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    getDashboardMetrics()
      .then((data) => { if (mounted) setMetrics(data); })
      .catch((failure) => { if (mounted) setError(failure.message); });
    const connection = connectDashboardTelemetry({
      onSnapshot: (data) => { setMetrics(data); setError(null); },
      onEvent: (event) => setEvents((current) => {
        if (current.some((item) => item.event_id === event.event_id)) return current;
        return [event, ...current].slice(0, 120);
      }),
      onGap: setGap,
      onStateChange: setState,
      onStreamReset: () => setEvents([]),
      domains: domains.map((domain) => domain.id),
    });
    return () => { mounted = false; connection.close(); };
  }, []);

  const domainState = useMemo(() => Object.fromEntries(domains.map((domain) => {
    const matching = events.filter((event) => event.domain === domain.id);
    return [domain.id, { count: matching.length, latest: matching[0]?.occurred_at }];
  })) as Record<TelemetryDomain, { count: number; latest?: string }>, [events]);

  const latestSequence = events[0]?.sequence || 0;
  const connectionLabel = state === 'live' ? 'Live stream' : state === 'connecting' ? 'Connecting' : state === 'reconnecting' ? 'Reconnecting' : 'Closed';

  return (
    <div className="ops-stack telemetry-workspace">
      <div className="workspace-heading telemetry-heading">
        <div><span className="ops-overline"><FiRadio /> event-driven observability</span><h2>Cognitive system</h2><p>Typed signals from the living architecture, with bounded replay and explicit gaps.</p></div>
        <div className={`stream-health is-${state}`}><span /><div><b>{connectionLabel}</b><small>{latestSequence ? `cursor ${latestSequence}` : 'awaiting signal'}</small></div></div>
      </div>
      {error && <div className="ops-banner is-error">{error}</div>}
      {gap && <div className="ops-banner telemetry-gap"><FiAlertTriangle /><span><b>Telemetry gap detected.</b> {gap.dropped_for_subscriber || Math.max(0, gap.available_from - gap.requested_after - 1)} projections were outside this viewer&apos;s buffer. Authoritative records are intact.</span><button onClick={() => setGap(null)}>Dismiss</button></div>}
      <div className="ops-metric-grid">
        <article className="ops-metric tone-cyan"><span className="ops-kicker">Analytics buffer</span><strong>{metrics?.summary?.total_events || 0}</strong><p>Recorded metric projections</p></article>
        <article className="ops-metric tone-violet"><span className="ops-kicker">Throughput</span><strong>{(metrics?.summary?.events_per_minute || 0).toFixed(1)}</strong><p>Events per minute</p></article>
        <article className="ops-metric tone-amber"><span className="ops-kicker">Cycle latency</span><strong>{Math.round(metrics?.summary?.avg_processing_time_ms || 0)}<small>ms</small></strong><p>Average processing time</p></article>
        <article className="ops-metric tone-green"><span className="ops-kicker">Live cursor</span><strong>{latestSequence || '—'}</strong><p>Current process stream</p></article>
      </div>

      <section className="telemetry-domain-grid" aria-label="Telemetry domains">
        {domains.map((domain) => <article key={domain.id} className={`telemetry-domain domain-${domain.id}`}><i /><span>{domain.label}</span><strong>{domainState[domain.id].count}</strong><small>{domainState[domain.id].latest ? new Date(domainState[domain.id].latest!).toLocaleTimeString() : domain.hint}</small></article>)}
      </section>

      <div className="telemetry-main-grid">
        <section className="ops-panel telemetry-feed">
          <div className="section-heading"><div><span>Ordered event stream</span><h3>Neural activity feed</h3></div>{state === 'live' ? <FiActivity /> : <FiRefreshCw className="spin-slow" />}</div>
          <div className="telemetry-events">
            {events.length === 0 ? <div className="empty-state compact"><FiRadio /><h3>Listening</h3><p>New cycles and governed background work will appear here without polling.</p></div> : events.map((event) => (
              <article key={event.event_id} className={`telemetry-event domain-${event.domain}`}>
                <i />
                <div><span>{event.domain === 'autonomous_work' ? 'autonomy' : humanize(event.domain)}</span><strong>{humanize(event.event_type)}</strong><p>{eventSummary(event)}</p></div>
                <time>{new Date(event.occurred_at).toLocaleTimeString()}</time>
                <code>#{event.sequence}</code>
              </article>
            ))}
          </div>
        </section>

        <section className="ops-panel activation-panel">
          <div className="section-heading"><div><span>Agent recruitment</span><h3>Cognitive activation map</h3></div><FiActivity /></div>
          <div className="agent-bars">{Object.entries(metrics?.agent_metrics?.activation_frequencies || {}).sort((a, b) => b[1] - a[1]).map(([agent, count]) => { const max = Math.max(...Object.values(metrics?.agent_metrics?.activation_frequencies || { one: 1 })); return <div key={agent}><span>{agent.replace(/_agent$/, '').replace(/_/g, ' ')}</span><i><b style={{ width: `${(count / max) * 100}%` }} /></i><strong>{count}</strong></div>; })}</div>
          {!Object.keys(metrics?.agent_metrics?.activation_frequencies || {}).length && <p className="panel-note">Agent activation data will populate after a cognitive cycle.</p>}
        </section>
      </div>
    </div>
  );
};

export default SystemTelemetry;
