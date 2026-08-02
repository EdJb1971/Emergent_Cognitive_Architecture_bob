import React, { useEffect, useRef } from 'react';
import { FiCpu, FiUser, FiZap } from 'react-icons/fi';
import { Message } from 'types/chat';

interface Props {
  messages: Message[];
  isLoading: boolean;
}

const ChatWindow: React.FC<Props> = ({ messages, isLoading }) => {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="chat-stream">
      <div className="chat-column">
        {messages.map((message) => (
          <article key={message.id} className={`chat-message ${message.sender === 'user' ? 'is-user' : 'is-ai'} ${message.is_proactive ? 'is-proactive' : ''} ${message.is_error ? 'is-error' : ''}`}>
            <div className="message-avatar">{message.sender === 'user' ? <FiUser /> : message.is_proactive ? <FiZap /> : <FiCpu />}</div>
            <div className="message-body">
              <div className="message-meta"><b>{message.sender === 'user' ? 'You' : message.is_proactive ? 'Bob / initiative' : 'Bob'}</b><time>{new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time></div>
              <p>{message.text}</p>
              {message.image_url && <img src={message.image_url} alt="Attachment" />}
              {message.audio_url && <audio controls src={message.audio_url} />}
              {message.image_base64 && <img src={`data:${message.image_mime_type || 'image/jpeg'};base64,${message.image_base64}`} alt="Attachment" />}
              {message.audio_base64 && <audio controls src={`data:${message.audio_mime_type || 'audio/wav'};base64,${message.audio_base64}`} />}
            </div>
          </article>
        ))}
        {isLoading && <article className="chat-message is-ai"><div className="message-avatar"><FiCpu /></div><div className="message-body thinking"><div className="message-meta"><b>Bob</b><span>cognitive cycle active</span></div><p><i /><i /><i /></p></div></article>}
        <div ref={endRef} />
      </div>
    </div>
  );
};

export default ChatWindow;
