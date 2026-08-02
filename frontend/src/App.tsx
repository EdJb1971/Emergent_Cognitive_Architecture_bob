import React, { useEffect, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { FiActivity, FiCommand, FiLock, FiRadio } from 'react-icons/fi';
import ChatWindow from 'components/ChatWindow';
import ChatInput from 'components/ChatInput';
import Dashboard from 'components/Dashboard';
import { Message, ChatRequest } from 'types/chat';
import { sendMessage } from 'api/chatApi';
import { getProactiveMessage, recordProactiveReaction } from 'api/dashboardApi';
import { getIdentity } from 'api/settingsApi';
import { IdentityProfile } from 'types/settings';

const DEFAULT_IDENTITY: IdentityProfile = {
  schema_version: 1,
  assistant_name: 'Bob',
  user_name: null,
  assistant_aliases: [],
  revision: 1,
  updated_at: new Date(0).toISOString(),
};

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [userId, setUserId] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [identity, setIdentity] = useState<IdentityProfile>(DEFAULT_IDENTITY);
  const [isDashboardOpen, setIsDashboardOpen] = useState(
    () => new URLSearchParams(window.location.search).has('control'),
  );

  useEffect(() => {
    getIdentity().then(setIdentity).catch((error) => console.debug('Identity settings unavailable:', error));
    let storedUserId = localStorage.getItem('user_id');
    if (!storedUserId) {
      storedUserId = uuidv4();
      localStorage.setItem('user_id', storedUserId);
    }
    setUserId(storedUserId);
    setSessionId(uuidv4());
    setMessages([{
      id: uuidv4(),
      sender: 'ai',
      text: 'I’m here. What should we think through?',
      timestamp: new Date().toISOString(),
    }]);
  }, []);

  useEffect(() => {
    document.title = `${identity.assistant_name} · Cognitive Operations`;
  }, [identity.assistant_name]);

  useEffect(() => {
    if (!userId) return;
    const poll = async () => {
      try {
        const response = await getProactiveMessage();
        if (response.has_message && response.message) {
          setMessages((current) => [...current, {
            id: uuidv4(),
            sender: 'ai',
            text: response.message!,
            timestamp: new Date().toISOString(),
            is_proactive: true,
            proactive_id: response.message_id,
            trigger_type: response.trigger_type,
          }]);
        }
      } catch (error) {
        console.debug('Proactive message check unavailable:', error);
      }
    };
    poll();
    const interval = window.setInterval(poll, 30000);
    return () => window.clearInterval(interval);
  }, [userId]);

  useEffect(() => {
    const openControlRoom = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setIsDashboardOpen(true);
      }
    };
    document.addEventListener('keydown', openControlRoom);
    return () => document.removeEventListener('keydown', openControlRoom);
  }, []);

  const handleSendMessage = async (
    text: string,
    imageBase64?: string,
    audioBase64?: string,
    imageMimeType?: string,
    audioMimeType?: string,
    audioSource?: 'direct_user_upload' | 'live_microphone_capture',
  ) => {
    if (!text.trim() && !imageBase64 && !audioBase64) return;
    const userMessage: Message = {
      id: uuidv4(),
      sender: 'user',
      text: text.trim(),
      timestamp: new Date().toISOString(),
      image_base64: imageBase64,
      image_mime_type: imageMimeType,
      audio_base64: audioBase64,
      audio_mime_type: audioMimeType,
      audio_source: audioSource,
    };
    setMessages((current) => [...current, userMessage]);
    setIsLoading(true);
    try {
      const request: ChatRequest = {
        user_id: userId,
        input_text: text.trim(),
        session_id: sessionId,
        timestamp: new Date().toISOString(),
        image_base64: imageBase64,
        image_mime_type: imageMimeType,
        audio_base64: audioBase64,
        audio_mime_type: audioMimeType,
        audio_source: audioSource,
      };
      const response = await sendMessage(request);
      const proactive = [...messages, userMessage].filter((message) => message.is_proactive).pop();
      if (proactive?.proactive_id) {
        recordProactiveReaction(proactive.proactive_id, text.trim()).catch(() => undefined);
      }
      setMessages((current) => [...current, {
        id: uuidv4(),
        sender: 'ai',
        text: response.response,
        timestamp: new Date().toISOString(),
      }]);
    } catch (error) {
      setMessages((current) => [...current, {
        id: uuidv4(),
        sender: 'ai',
        text: error instanceof Error ? error.message : 'The cognitive cycle could not complete.',
        timestamp: new Date().toISOString(),
        is_error: true,
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="eca-app-shell">
      <header className="eca-topbar">
        <div className="eca-brand"><span><FiActivity /></span><div><b>{identity.assistant_name.toUpperCase()}</b><small>Emergent cognitive architecture</small></div></div>
        <div className="eca-top-actions">
          <span className="privacy-indicator"><FiLock /> Local cognition</span>
          <span className="system-indicator"><i /><FiRadio /> Online</span>
          <button onClick={() => setIsDashboardOpen(true)} className="open-operator"><FiCommand /><span>Open control room</span><kbd>Ctrl K</kbd></button>
        </div>
      </header>
      <main className="conversation-stage">
        <section className="conversation-frame">
          <div className="conversation-header"><div><span className="ops-overline">continuous cognition</span><h1>Conversation</h1></div><span className="session-code">SESSION / {sessionId.slice(0, 8).toUpperCase()}</span></div>
          <ChatWindow messages={messages} isLoading={isLoading} assistantName={identity.assistant_name} userName={identity.user_name} />
          <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
        </section>
        <footer className="conversation-footer"><FiLock /><span>Routine cognition stays local. External research crosses a question-only, audited boundary.</span></footer>
      </main>
      <Dashboard isOpen={isDashboardOpen} onClose={() => setIsDashboardOpen(false)} identity={identity} onIdentityChange={setIdentity} />
    </div>
  );
};

export default App;
