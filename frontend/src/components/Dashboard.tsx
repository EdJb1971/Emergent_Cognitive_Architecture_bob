import React, { useEffect, useState } from 'react';
import {
  FiActivity,
  FiBarChart2,
  FiBookOpen,
  FiCommand,
  FiCpu,
  FiDatabase,
  FiInbox,
  FiGitBranch,
  FiTarget,
  FiShield,
  FiX,
} from 'react-icons/fi';
import ResearchCommandCenter, { ResearchView } from 'components/ResearchCommandCenter';
import AutonomousWorkCenter from 'components/AutonomousWorkCenter';
import SystemTelemetry from 'components/SystemTelemetry';
import PredictiveCalibrationCenter from 'components/PredictiveCalibrationCenter';

interface DashboardProps {
  isOpen: boolean;
  onClose: () => void;
}

type View = ResearchView | 'autonomy' | 'predictive' | 'system';

const nav: Array<{ id: View; label: string; description: string; icon: React.ReactNode }> = [
  { id: 'command', label: 'Command', description: 'Research posture', icon: <FiCommand /> },
  { id: 'autonomy', label: 'Autonomy', description: 'Executive control', icon: <FiGitBranch /> },
  { id: 'inquiries', label: 'Inquiries', description: 'Waking review', icon: <FiInbox /> },
  { id: 'calibration', label: 'Calibration', description: 'Shadow evidence', icon: <FiBarChart2 /> },
  { id: 'ledger', label: 'Ledger', description: 'Immutable history', icon: <FiBookOpen /> },
  { id: 'predictive', label: 'Predictive', description: 'Perception calibration', icon: <FiTarget /> },
  { id: 'system', label: 'System', description: 'Cognitive telemetry', icon: <FiCpu /> },
];

const Dashboard: React.FC<DashboardProps> = ({ isOpen, onClose }) => {
  const [view, setView] = useState<View>('command');
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
          {view === 'autonomy' ? <AutonomousWorkCenter /> : view === 'predictive' ? <PredictiveCalibrationCenter /> : view !== 'system' ? <ResearchCommandCenter view={view} /> : <SystemTelemetry />}
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
