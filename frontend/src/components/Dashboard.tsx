import React, { useEffect, useState } from 'react';
import {
  FiActivity,
  FiBarChart2,
  FiBookOpen,
  FiCommand,
  FiCpu,
  FiDatabase,
  FiInbox,
  FiShield,
  FiX,
} from 'react-icons/fi';
import { getDashboardMetrics } from 'api/dashboardApi';
import ResearchCommandCenter, { ResearchView } from 'components/ResearchCommandCenter';
import { DashboardMetrics } from 'types/dashboard';

interface DashboardProps {
  isOpen: boolean;
  onClose: () => void;
}

type View = ResearchView | 'system';

const nav: Array<{ id: View; label: string; description: string; icon: React.ReactNode }> = [
  { id: 'command', label: 'Command', description: 'Research posture', icon: <FiCommand /> },
  { id: 'inquiries', label: 'Inquiries', description: 'Waking review', icon: <FiInbox /> },
  { id: 'calibration', label: 'Calibration', description: 'Shadow evidence', icon: <FiBarChart2 /> },
  { id: 'ledger', label: 'Ledger', description: 'Immutable history', icon: <FiBookOpen /> },
  { id: 'system', label: 'System', description: 'Cognitive telemetry', icon: <FiCpu /> },
];

const Dashboard: React.FC<DashboardProps> = ({ isOpen, onClose }) => {
  const [view, setView] = useState<View>('command');
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen || view !== 'system') return;
    getDashboardMetrics().then(setMetrics).catch((error) => setMetricsError(error.message));
  }, [isOpen, view]);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    if (isOpen) document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="operator-overlay" role="dialog" aria-modal="true" aria-label="ECA operator control room">
      <aside className="operator-sidebar">
        <div className="operator-brand">
          <span className="brand-mark"><FiActivity /></span>
          <span><b>ECA</b><small>Operator system</small></span>
        </div>
        <nav className="operator-nav" aria-label="Control room sections">
          {nav.map((item) => (
            <button key={item.id} className={view === item.id ? 'is-active' : ''} onClick={() => setView(item.id)}>
              <span className="nav-icon">{item.icon}</span>
              <span><b>{item.label}</b><small>{item.description}</small></span>
              {item.id === 'inquiries' && <i />}
            </button>
          ))}
        </nav>
        <div className="operator-sidebar-foot">
          <div><FiShield /><span><b>Local first</b><small>Question-only cloud boundary</small></span></div>
          <div><FiDatabase /><span><b>Ledger backed</b><small>Changes survive restart</small></span></div>
        </div>
      </aside>

      <main className="operator-main">
        <header className="operator-topbar">
          <div><span className="live-dot" /> <b>BOB / COGNITIVE OPERATIONS</b><small>Single-operator secure console</small></div>
          <button onClick={onClose} aria-label="Close operator console"><span>Return to conversation</span><FiX /></button>
        </header>
        <div className="operator-content">
          {view !== 'system' ? <ResearchCommandCenter view={view} /> : (
            <div className="ops-stack">
              <div className="workspace-heading"><div><span className="ops-overline"><FiCpu /> system telemetry</span><h2>Cognitive system</h2><p>Live operational signals from the complete architecture.</p></div></div>
              {metricsError && <div className="ops-banner is-error">{metricsError}</div>}
              <div className="ops-metric-grid">
                <article className="ops-metric tone-cyan"><span className="ops-kicker">Total events</span><strong>{metrics?.summary?.total_events || 0}</strong><p>Persisted telemetry events</p></article>
                <article className="ops-metric tone-violet"><span className="ops-kicker">Throughput</span><strong>{(metrics?.summary?.events_per_minute || 0).toFixed(1)}</strong><p>Events per minute</p></article>
                <article className="ops-metric tone-amber"><span className="ops-kicker">Cycle latency</span><strong>{Math.round(metrics?.summary?.avg_processing_time_ms || 0)}<small>ms</small></strong><p>Average processing time</p></article>
                <article className="ops-metric tone-green"><span className="ops-kicker">Active agents</span><strong>{Object.keys(metrics?.agent_metrics?.activation_frequencies || {}).length}</strong><p>Cognitive specialisations observed</p></article>
              </div>
              <section className="ops-panel"><div className="section-heading"><div><span>Agent recruitment</span><h3>Cognitive activation map</h3></div><FiActivity /></div><div className="agent-bars">{Object.entries(metrics?.agent_metrics?.activation_frequencies || {}).sort((a, b) => b[1] - a[1]).map(([agent, count]) => { const max = Math.max(...Object.values(metrics?.agent_metrics?.activation_frequencies || { one: 1 })); return <div key={agent}><span>{agent.replace(/_agent$/, '').replace(/_/g, ' ')}</span><i><b style={{ width: `${(count / max) * 100}%` }} /></i><strong>{count}</strong></div>; })}</div></section>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
