import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  FiActivity, FiAlertOctagon, FiCheck, FiClock, FiCpu, FiMoon,
  FiPlay, FiRefreshCw, FiRepeat, FiShield, FiSquare, FiZap,
} from 'react-icons/fi';
import {
  cancelAutonomousTask, getAutonomousLedger, getAutonomousRuntime,
  getAutonomousTasks, retryAutonomousTask, updateAutonomousRuntime,
} from 'api/autonomousApi';
import {
  AutonomousEvent, AutonomousRuntime, AutonomousTask, AutonomousTaskType,
} from 'types/autonomous';

const labels: Record<AutonomousTaskType, { title: string; region: string; icon: React.ReactNode }> = {
  sleep: { title: 'Sleep consolidation', region: 'Hippocampal replay', icon: <FiMoon /> },
  reflection: { title: 'Reflection', region: 'Default mode network', icon: <FiRepeat /> },
  discovery: { title: 'Discovery', region: 'Anterior prefrontal', icon: <FiZap /> },
  curiosity: { title: 'Curiosity', region: 'Information seeking', icon: <FiActivity /> },
  self_assessment: { title: 'Self-assessment', region: 'Metacognitive monitor', icon: <FiCpu /> },
  proactive_engagement: { title: 'Proactive engagement', region: 'Social initiative', icon: <FiPlay /> },
  summary_update: { title: 'Summary memory', region: 'Semantic compression', icon: <FiRepeat /> },
  stm_flush: { title: 'STM flush', region: 'Working-memory pressure', icon: <FiSquare /> },
};

const initiativeTypes: AutonomousTaskType[] = [
  'sleep', 'reflection', 'discovery', 'curiosity', 'self_assessment', 'proactive_engagement',
];
const housekeepingTypes: AutonomousTaskType[] = ['summary_update', 'stm_flush'];
const humanize = (value: string) => value.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
const tone = (status: string) => status === 'completed' ? 'positive'
  : ['failed', 'rejected'].includes(status) ? 'danger'
  : status === 'running' ? 'amber' : 'muted';

const Toggle: React.FC<{ active: boolean; disabled?: boolean; label: string; onChange: () => void }> =
  ({ active, disabled, label, onChange }) => <button type="button" role="switch" aria-checked={active}
    aria-label={label} disabled={disabled} onClick={onChange}
    className={`control-toggle ${active ? 'is-on' : ''}`}><span /></button>;

const AutonomousWorkCenter: React.FC = () => {
  const [runtime, setRuntime] = useState<AutonomousRuntime | null>(null);
  const [tasks, setTasks] = useState<AutonomousTask[]>([]);
  const [events, setEvents] = useState<AutonomousEvent[]>([]);
  const [integrity, setIntegrity] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const refresh = useCallback(async (quiet = false) => {
    if (!quiet) setBusy(true);
    const results = await Promise.allSettled([
      getAutonomousRuntime(), getAutonomousTasks(), getAutonomousLedger(),
    ]);
    if (results[0].status === 'fulfilled') setRuntime(results[0].value);
    if (results[1].status === 'fulfilled') setTasks(results[1].value.tasks);
    if (results[2].status === 'fulfilled') {
      setEvents(results[2].value.events); setIntegrity(results[2].value.integrity_verified);
    }
    const failure = results.find((result) => result.status === 'rejected') as PromiseRejectedResult | undefined;
    setError(failure ? failure.reason?.message || 'Executive control is unavailable.' : null);
    setBusy(false);
  }, []);

  useEffect(() => { refresh(); const timer = window.setInterval(() => refresh(true), 5000); return () => clearInterval(timer); }, [refresh]);
  useEffect(() => { if (!notice) return; const timer = window.setTimeout(() => setNotice(null), 3500); return () => clearTimeout(timer); }, [notice]);

  const mutate = async (body: Parameters<typeof updateAutonomousRuntime>[0], message: string) => {
    setBusy(true); setError(null);
    try { setRuntime(await updateAutonomousRuntime(body)); setNotice(message); await refresh(true); }
    catch (err) { setError(err instanceof Error ? err.message : 'Control change failed.'); }
    finally { setBusy(false); }
  };

  const act = async (task: AutonomousTask, action: 'cancel' | 'retry') => {
    setBusy(true);
    try {
      await (action === 'cancel' ? cancelAutonomousTask(task.request.task_id) : retryAutonomousTask(task.request.task_id));
      setNotice(`${humanize(action)} accepted.`); await refresh(true);
    } catch (err) { setError(err instanceof Error ? err.message : 'Task action failed.'); }
    finally { setBusy(false); }
  };

  const counts = useMemo(() => tasks.reduce((map, task) => {
    map[task.status] = (map[task.status] || 0) + 1; return map;
  }, {} as Record<string, number>), [tasks]);

  const renderPolicy = (taskType: AutonomousTaskType) => {
    const policy = runtime?.policies[taskType];
    if (!policy) return null;
    const meta = labels[taskType];
    return <article className={`autonomy-card ${policy.enabled ? 'is-enabled' : ''}`} key={taskType}>
      <div className="autonomy-icon">{meta.icon}</div>
      <div className="autonomy-copy"><strong>{meta.title}</strong><span>{meta.region}</span><p>{policy.description}</p>
        <div><small>{policy.max_per_hour}/hour</small><small>{Math.round(policy.timeout_seconds)}s timeout</small><small>{policy.max_retries} retries</small><small>{policy.cancel_on_user_activity ? 'waking preempts' : 'integrity protected'}</small></div>
      </div>
      <Toggle active={policy.enabled} disabled={busy || !runtime?.master_enabled} label={meta.title}
        onChange={() => mutate({ category_enabled: { [taskType]: !policy.enabled }, reason: `Operator ${policy.enabled ? 'disabled' : 'enabled'} ${taskType}.` }, `${meta.title} ${policy.enabled ? 'disabled' : 'enabled'}.`)} />
    </article>;
  };

  if (!runtime && busy) return <div className="ops-loading"><span /><p>Synchronising executive control…</p></div>;
  return <div className="ops-stack">
    {(error || notice) && <div className={`ops-banner ${error ? 'is-error' : 'is-success'}`}>{error ? <FiAlertOctagon /> : <FiCheck />}<span>{error || notice}</span></div>}
    <section className="ops-hero autonomy-hero"><div><span className="ops-overline"><FiShield /> unified executive control</span><h2>Autonomous work governor</h2><p>One bounded contract arbitrates sleep, reflection, discovery, curiosity, self-assessment, outreach, and memory housekeeping. Foreground cognition always has priority.</p></div>
      <div className={`runtime-orbit ${runtime?.master_enabled ? 'is-live' : ''}`}><span className="orbit-core"><FiCpu /></span><span className="orbit-copy"><b>{runtime?.master_enabled ? 'GOVERNED' : 'PAUSED'}</b><small>{runtime?.active_count || 0} active tasks</small></span></div>
    </section>
    <section className="ops-panel autonomy-master"><div><span className="ops-kicker">Global posture</span><h3>Autonomous cognition</h3><p>Pausing the master switch rejects new work and cancels interruptible activity. No autonomous category can access cloud inference.</p></div><Toggle active={!!runtime?.master_enabled} disabled={busy} label="Autonomous work master" onChange={() => mutate({ master_enabled: !runtime?.master_enabled, reason: `Operator ${runtime?.master_enabled ? 'paused' : 'resumed'} autonomous work.` }, runtime?.master_enabled ? 'Autonomous work paused.' : 'Autonomous work resumed.')} /></section>
    <div className="ops-metric-grid">
      <article className="ops-metric tone-amber"><span className="ops-kicker">Active</span><strong>{runtime?.active_count || 0}</strong><p>Global capacity {runtime?.max_concurrent_global || 0}</p></article>
      <article className="ops-metric tone-green"><span className="ops-kicker">Completed</span><strong>{counts.completed || 0}</strong><p>Durable task outcomes</p></article>
      <article className="ops-metric tone-red"><span className="ops-kicker">Failed / rejected</span><strong>{(counts.failed || 0) + (counts.rejected || 0)}</strong><p>Visible policy and execution failures</p></article>
      <article className="ops-metric tone-violet"><span className="ops-kicker">Ledger</span><strong>{integrity ? 'Verified' : 'Attention'}</strong><p>{events.length} immutable events loaded</p></article>
    </div>
    <div className="autonomy-section-head"><div><span>Initiative network</span><h3>Optional cognitive drives</h3></div><button className="icon-button" onClick={() => refresh()} title="Refresh"><FiRefreshCw /></button></div>
    <section className="autonomy-grid">{initiativeTypes.map(renderPolicy)}</section>
    <div className="autonomy-section-head"><div><span>Memory integrity</span><h3>Essential housekeeping</h3></div></div>
    <section className="autonomy-grid housekeeping-grid">{housekeepingTypes.map(renderPolicy)}</section>
    <section className="ops-panel"><div className="section-heading"><div><span>Bounded execution</span><h3>Recent work</h3></div><FiClock /></div>
      <div className="autonomy-table"><div className="autonomy-table-head"><span>Task</span><span>Trigger</span><span>Attempts</span><span>Status</span><span>Action</span></div>
        {tasks.length === 0 ? <div className="empty-state compact"><FiActivity /><h3>No governed work yet</h3><p>Accepted and rejected tasks will appear here.</p></div> : tasks.slice(0, 30).map((task) => <div className="autonomy-task-row" key={task.request.task_id}><div><b>{labels[task.request.task_type].title}</b><small>{new Date(task.request.created_at).toLocaleString()}</small></div><p title={task.request.trigger_reason}>{task.request.trigger_reason}</p><span>{task.attempt} / {task.max_attempts}</span><span className={`status-badge tone-${tone(task.status)}`}>{humanize(task.status)}</span><div className="task-actions">{['queued', 'running'].includes(task.status) && <button onClick={() => act(task, 'cancel')} disabled={busy}><FiSquare /> Cancel</button>}{['failed', 'cancelled'].includes(task.status) && <button onClick={() => act(task, 'retry')} disabled={busy}><FiRepeat /> Retry</button>}</div></div>)}
      </div>
    </section>
  </div>;
};

export default AutonomousWorkCenter;
