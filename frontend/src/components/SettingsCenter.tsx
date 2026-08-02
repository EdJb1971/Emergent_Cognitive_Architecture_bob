import React, { useEffect, useState } from 'react';
import { FiAlertTriangle, FiCheck, FiRefreshCw, FiSave, FiShield, FiUser } from 'react-icons/fi';
import { armCleanStart, cancelCleanStart, getCleanStartStatus, updateIdentity } from 'api/settingsApi';
import { CleanStartStatus, IdentityProfile } from 'types/settings';

const RESET_PHRASE = 'RESET COGNITIVE MEMORY';

interface Props {
  identity: IdentityProfile;
  onIdentityChange: (identity: IdentityProfile) => void;
}

const SettingsCenter: React.FC<Props> = ({ identity, onIdentityChange }) => {
  const [assistantName, setAssistantName] = useState(identity.assistant_name);
  const [userName, setUserName] = useState(identity.user_name || '');
  const [resetStatus, setResetStatus] = useState<CleanStartStatus | null>(null);
  const [confirmation, setConfirmation] = useState('');
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<{ tone: 'ok' | 'error'; text: string } | null>(null);

  useEffect(() => { setAssistantName(identity.assistant_name); setUserName(identity.user_name || ''); }, [identity]);
  useEffect(() => { getCleanStartStatus().then(setResetStatus).catch((error) => setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'Could not read clean-start status.' })); }, []);

  const save = async () => {
    setBusy(true); setNotice(null);
    try {
      const next = await updateIdentity({ assistant_name: assistantName, user_name: userName.trim() || null, expected_revision: identity.revision });
      onIdentityChange(next);
      setNotice({ tone: 'ok', text: 'Identity updated across the interface and cognitive runtime.' });
    } catch (error) { setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'Identity update failed.' }); }
    finally { setBusy(false); }
  };

  const armReset = async () => {
    setBusy(true); setNotice(null);
    try { setResetStatus(await armCleanStart(confirmation)); setConfirmation(''); setNotice({ tone: 'ok', text: 'Clean start armed. Restart the backend to clear cognitive memory.' }); }
    catch (error) { setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'Could not arm clean start.' }); }
    finally { setBusy(false); }
  };

  const cancelReset = async () => {
    setBusy(true); setNotice(null);
    try { setResetStatus(await cancelCleanStart()); setNotice({ tone: 'ok', text: 'Pending clean start cancelled.' }); }
    catch (error) { setNotice({ tone: 'error', text: error instanceof Error ? error.message : 'Could not cancel clean start.' }); }
    finally { setBusy(false); }
  };

  const unchanged = assistantName.trim() === identity.assistant_name && userName.trim() === (identity.user_name || '');
  return (
    <section className="settings-workspace">
      <header className="workspace-heading"><div><span className="ops-overline">local configuration</span><h2>Identity & clean start</h2><p>Names are explicit operator settings—not guesses learned from conversation entities or the machine running the app.</p></div><span className="status-badge tone-positive"><FiShield /> local only</span></header>
      {notice && <div className={`ops-banner ${notice.tone === 'error' ? 'is-error' : ''}`}>{notice.tone === 'ok' ? <FiCheck /> : <FiAlertTriangle />}{notice.text}</div>}
      <div className="settings-grid">
        <article className="ops-panel identity-card">
          <div className="section-heading"><div><span>Authoritative profile</span><h3>How this mind identifies us</h3></div><FiUser /></div>
          <label className="settings-field"><span>Assistant name</span><small>Appears in the title, conversation, control room, and runtime prompts.</small><input value={assistantName} maxLength={40} onChange={(event) => setAssistantName(event.target.value)} /></label>
          <label className="settings-field"><span>Your name <i>optional</i></span><small>Leave blank to be addressed generically as “you”.</small><input value={userName} maxLength={80} placeholder="Not configured" onChange={(event) => setUserName(event.target.value)} /></label>
          <div className="settings-actions"><span>Revision {identity.revision} · saved {new Date(identity.updated_at).toLocaleString()}</span><button className="primary-button" disabled={busy || unchanged || !assistantName.trim()} onClick={save}>{busy ? <FiRefreshCw className="spin-slow" /> : <FiSave />} Save identity</button></div>
          {identity.assistant_aliases.length > 0 && <p className="settings-aliases">Former configured names: {identity.assistant_aliases.join(', ')}</p>}
        </article>
        <article className="ops-panel clean-start-card">
          <div className="section-heading"><div><span>Destructive maintenance</span><h3>Start with a clean mind</h3></div><FiAlertTriangle /></div>
          <p>Clears conversations, summaries, learned memory, calibration, inquiry, sleep, and autonomous-work ledgers on the next backend restart. Your configured names are preserved.</p>
          {resetStatus?.pending_restart ? <div className="pending-reset"><FiAlertTriangle /><div><b>Clean start pending restart</b><small>Requested {resetStatus.requested_at ? new Date(resetStatus.requested_at).toLocaleString() : 'recently'}. No data is removed while databases are live.</small></div><button className="secondary-button" disabled={busy} onClick={cancelReset}>Cancel</button></div> : <><label className="settings-field danger-field"><span>Confirmation phrase</span><small>Type <code>{RESET_PHRASE}</code> exactly.</small><input value={confirmation} placeholder={RESET_PHRASE} onChange={(event) => setConfirmation(event.target.value)} /></label><div className="settings-actions"><span>This cannot be recovered unless you made a backup.</span><button className="danger-button" disabled={busy || confirmation !== RESET_PHRASE} onClick={armReset}><FiAlertTriangle /> Arm clean start</button></div></>}
        </article>
      </div>
    </section>
  );
};

export default SettingsCenter;
