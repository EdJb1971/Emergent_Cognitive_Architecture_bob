import React, { useRef, useEffect, useState } from 'react';
import { Message } from 'types/chat';
import { testProactiveMessage } from '../api/dashboardApi';

interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
}

const ChatWindow: React.FC<ChatWindowProps> = ({ messages, isLoading }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showTestPanel, setShowTestPanel] = useState(false);
  const [testTriggerType, setTestTriggerType] = useState('discovery_insight');
  const [testMessageContent, setTestMessageContent] = useState('Hey! I just discovered something interesting about quantum computing...');
  const [isTesting, setIsTesting] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleTestProactiveMessage = async () => {
    setIsTesting(true);
    try {
      const result = await testProactiveMessage(testTriggerType, testMessageContent);
      console.log('Test proactive message created:', result);
      alert(`Test message queued! Check for proactive messages in the next poll.`);
    } catch (error) {
      console.error('Failed to create test proactive message:', error);
      alert('Failed to create test proactive message. Check console for details.');
    } finally {
      setIsTesting(false);
    }
  };

  return (
    <div className="flex-1 p-4 overflow-y-auto bg-gray-50 dark:bg-gray-800 rounded-lg shadow-inner">
      {/* Test Panel */}
      <div className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
        <button
          onClick={() => setShowTestPanel(!showTestPanel)}
          className="text-sm text-yellow-700 dark:text-yellow-300 font-medium hover:text-yellow-800 dark:hover:text-yellow-200"
        >
          🧪 Test Proactive Messages {showTestPanel ? '▼' : '▶'}
        </button>
        {showTestPanel && (
          <div className="mt-2 space-y-2">
            <select
              value={testTriggerType}
              onChange={(e) => setTestTriggerType(e.target.value)}
              className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              aria-label="Test trigger type"
            >
              <option value="discovery_insight">Discovery Insight</option>
              <option value="self_reflection">Self Reflection</option>
              <option value="boredom">Boredom</option>
              <option value="knowledge_gap">Knowledge Gap</option>
              <option value="emotional_checkin">Emotional Check-in</option>
            </select>
            <textarea
              value={testMessageContent}
              onChange={(e) => setTestMessageContent(e.target.value)}
              placeholder="Enter test message content..."
              className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 resize-none"
              rows={2}
              aria-label="Test message content"
            />
            <button
              onClick={handleTestProactiveMessage}
              disabled={isTesting}
              className="px-3 py-1 text-sm bg-yellow-600 hover:bg-yellow-700 disabled:bg-yellow-400 text-white rounded font-medium"
            >
              {isTesting ? 'Creating...' : 'Queue Test Message'}
            </button>
          </div>
        )}
      </div>

      {messages.map((message) => (
        <div
          key={message.id}
          className={`flex mb-4 ${
            message.sender === 'user' ? 'justify-end' : 'justify-start'
          }`}
        >
          <div
            className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg shadow ${
              message.sender === 'user'
                ? 'bg-blue-500 text-white'
                : message.is_proactive
                ? 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200 border-l-4 border-purple-500'
                : 'bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
            }`}
          >
            {message.is_proactive && (
              <div className="text-xs text-purple-600 dark:text-purple-400 mb-1 font-medium">
                🤖 AI Initiative • {message.trigger_type?.replace('_', ' ').toUpperCase()}
              </div>
            )}
            <p className="text-sm">{message.text}</p>
            {message.image_url && (
              <img src={message.image_url} alt="User attachment" className="mt-2 max-w-full h-auto rounded-md" />
            )}
            {message.audio_url && (
              <audio controls src={message.audio_url} className="mt-2 w-full"></audio>
            )}
            {message.image_base64 && (
              <img src={`data:image/jpeg;base64,${message.image_base64}`} alt="User attachment" className="mt-2 max-w-full h-auto rounded-md" />
            )}
            {message.audio_base64 && (
              <audio controls src={`data:audio/webm;base64,${message.audio_base64}`} className="mt-2 w-full"></audio>
            )}
            <p className="text-xs mt-1 opacity-75">
              {new Date(message.timestamp).toLocaleTimeString()}
            </p>
          </div>
        </div>
      ))}
      {isLoading && (
        <div className="flex justify-start mb-4">
          <div className="max-w-xs lg:max-w-md px-4 py-2 rounded-lg shadow bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200">
            <p className="text-sm animate-pulse">AI is thinking...</p>
          </div>
        </div>
      )}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatWindow;
