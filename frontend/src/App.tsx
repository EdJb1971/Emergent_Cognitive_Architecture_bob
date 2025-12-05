import React, { useState, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import ChatWindow from 'components/ChatWindow';
import ChatInput from 'components/ChatInput';
import Dashboard from 'components/Dashboard';
import { Message, ChatRequest } from 'types/chat';
import { sendMessage } from 'api/chatApi';
import { getProactiveMessage, recordProactiveReaction } from 'api/dashboardApi';

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [userId, setUserId] = useState<string>('');
  const [sessionId, setSessionId] = useState<string>('');
  const [isDashboardOpen, setIsDashboardOpen] = useState(false);
  const [lastProactiveCheck, setLastProactiveCheck] = useState<Date>(new Date());

  useEffect(() => {
    // Initialize user_id and session_id from local storage or generate new ones
    let storedUserId = localStorage.getItem('user_id');
    if (!storedUserId) {
      storedUserId = uuidv4();
      localStorage.setItem('user_id', storedUserId);
    }
    setUserId(storedUserId);

    const newSessionId = uuidv4();
    setSessionId(newSessionId);

    setMessages([
      {
        id: uuidv4(),
        sender: 'ai',
        text: 'Hello! How can I assist you today?',
        timestamp: new Date().toISOString(),
      },
    ]);
  }, []);

  // Proactive message polling
  useEffect(() => {
    if (!userId) return;

    const pollProactiveMessages = async () => {
      try {
        const proactiveResponse = await getProactiveMessage();
        if (proactiveResponse.has_message && proactiveResponse.message) {
          // Add proactive message to chat
          const proactiveMessage: Message = {
            id: uuidv4(),
            sender: 'ai',
            text: proactiveResponse.message,
            timestamp: new Date().toISOString(),
            is_proactive: true,
            proactive_id: proactiveResponse.message_id,
            trigger_type: proactiveResponse.trigger_type,
          };

          setMessages((prevMessages) => [...prevMessages, proactiveMessage]);
          setLastProactiveCheck(new Date());
        }
      } catch (error) {
        // Silently fail - proactive messages are optional
        console.debug('Proactive message check failed:', error);
      }
    };

    // Poll immediately and then every 30 seconds
    pollProactiveMessages();
    const interval = setInterval(pollProactiveMessages, 30000);

    return () => clearInterval(interval);
  }, [userId]);

  const handleSendMessage = async (text: string, imageBase64?: string, audioBase64?: string) => {
    if (!text.trim() && !imageBase64 && !audioBase64) return;

    const newMessage: Message = {
      id: uuidv4(),
      sender: 'user',
      text: text.trim(),
      timestamp: new Date().toISOString(),
      image_base64: imageBase64,
      audio_base64: audioBase64,
    };

    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setIsLoading(true);

    try {
      const requestBody: ChatRequest = {
        user_id: userId,
        input_text: text.trim(),
        session_id: sessionId,
        timestamp: new Date().toISOString(),
        image_base64: imageBase64,
        audio_base64: audioBase64,
      };
      
      const response = await sendMessage(requestBody);

      // Record reaction to any recent proactive message
      const recentMessages = [...messages, newMessage];
      const lastProactiveMessage = recentMessages
        .filter(m => m.sender === 'ai' && m.is_proactive)
        .pop();
      
      if (lastProactiveMessage?.proactive_id) {
        try {
          await recordProactiveReaction(lastProactiveMessage.proactive_id, text.trim());
        } catch (error) {
          console.debug('Failed to record proactive reaction:', error);
        }
      }

      setMessages((prevMessages) => [
        ...prevMessages,
        {
          id: uuidv4(),
          sender: 'ai',
          text: response.response,
          timestamp: new Date().toISOString(),
        },
      ]);
    } catch (error: any) {
      console.error('Failed to send message:', error);
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          id: uuidv4(),
          sender: 'ai',
          text: `Error: ${error.message || 'Could not get a response from the AI.'}`,
          timestamp: new Date().toISOString(),
          is_error: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold">Multi-Agent AI Chat</h1>
          <button
            onClick={() => setIsDashboardOpen(true)}
            className="px-4 py-2 bg-blue-700 hover:bg-blue-800 rounded-lg transition-colors font-medium"
          >
            📊 Dashboard
          </button>
        </div>
      </header>
      <main className="flex-1 flex flex-col overflow-hidden">
        <ChatWindow messages={messages} isLoading={isLoading} />
        <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
      </main>
      <Dashboard isOpen={isDashboardOpen} onClose={() => setIsDashboardOpen(false)} />
    </div>
  );
};

export default App;
