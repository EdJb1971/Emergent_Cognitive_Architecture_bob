import React, { useState, useRef, ChangeEvent } from 'react';
import * as FiIcons from 'react-icons/fi';

interface ChatInputProps {
  onSendMessage: (text: string, imageBase64?: string, audioBase64?: string) => void;
  isLoading: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, isLoading }) => {
  const [inputText, setInputText] = useState('');
  const [selectedImage, setSelectedImage] = useState<string | undefined>(undefined);
  const [selectedAudio, setSelectedAudio] = useState<string | undefined>(undefined);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  const handleTextChange = (e: ChangeEvent<HTMLInputElement>) => {
    setInputText(e.target.value);
  };

  const handleImageSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = (reader.result as string).split(',')[1];
        setSelectedImage(base64String);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAudioSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = (reader.result as string).split(',')[1];
        setSelectedAudio(base64String);
      };
      reader.readAsDataURL(file);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64String = (reader.result as string).split(',')[1];
          setSelectedAudio(base64String);
        };
        reader.readAsDataURL(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Could not access microphone. Please ensure it is enabled.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop()); // Stop microphone access
      setIsRecording(false);
    }
  };

  const handleSend = () => {
    if (inputText.trim() || selectedImage || selectedAudio) {
      onSendMessage(inputText, selectedImage, selectedAudio);
      setInputText('');
      setSelectedImage(undefined);
      setSelectedAudio(undefined);
      if (fileInputRef.current) fileInputRef.current.value = '';
      if (audioInputRef.current) audioInputRef.current.value = '';
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && (inputText.trim() || selectedImage || selectedAudio)) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="p-4 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 flex flex-col space-y-2">
      {(selectedImage || selectedAudio) && (
        <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
          {selectedImage && (
            <span className="flex items-center">
              {React.createElement(FiIcons.FiImage as any, { className: "mr-1", size: 16 })} Image attached
              <button onClick={() => setSelectedImage(undefined)} className="ml-2 text-red-500 hover:text-red-700">x</button>
            </span>
          )}
          {selectedAudio && (
            <span className="flex items-center">
              {React.createElement(FiIcons.FiMic as any, { className: "mr-1", size: 16 })} Audio attached
              <button onClick={() => setSelectedAudio(undefined)} className="ml-2 text-red-500 hover:text-red-700">x</button>
            </span>
          )}
        </div>
      )}
      <div className="flex items-center space-x-2">
        <input
          type="text"
          className="flex-1 p-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-gray-100"
          placeholder="Type your message..."
          value={inputText}
          onChange={handleTextChange}
          onKeyPress={handleKeyPress}
          disabled={isLoading || isRecording}
        />
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          onChange={handleImageSelect}
          className="hidden"
          disabled={isLoading || isRecording}
          aria-label="Select image file"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="p-2 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors duration-200"
          title="Attach Image"
          disabled={isLoading || isRecording}
        >
          {React.createElement(FiIcons.FiImage as any, { className: "text-gray-700 dark:text-gray-300", size: 20 })}
        </button>

        <input
          type="file"
          accept="audio/*"
          ref={audioInputRef}
          onChange={handleAudioSelect}
          className="hidden"
          disabled={isLoading || isRecording}
          aria-label="Select audio file"
        />
        <button
          onClick={isRecording ? stopRecording : startRecording}
          className={`p-2 rounded-lg transition-colors duration-200 ${
            isRecording ? 'bg-red-500 hover:bg-red-600 text-white' : 'bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600'
          }`}
          title={isRecording ? "Stop Recording" : "Record Audio"}
          disabled={isLoading}
        >
          {isRecording ? React.createElement(FiIcons.FiStopCircle as any, { className: "text-white", size: 20 }) : React.createElement(FiIcons.FiMic as any, { className: "text-gray-700 dark:text-gray-300", size: 20 })}
        </button>

        <button
          onClick={handleSend}
          className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors duration-200"
          title="Send Message"
          disabled={isLoading || (!inputText.trim() && !selectedImage && !selectedAudio)}
        >
          {React.createElement(FiIcons.FiSend as any, { size: 20 })}
        </button>
      </div>
    </div>
  );
};

export default ChatInput;
