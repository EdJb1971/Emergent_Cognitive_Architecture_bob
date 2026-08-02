import React, { useEffect, useState, useRef, ChangeEvent } from 'react';
import * as FiIcons from 'react-icons/fi';

interface ChatInputProps {
  onSendMessage: (
    text: string,
    imageBase64?: string,
    audioBase64?: string,
    imageMimeType?: string,
    audioMimeType?: string,
    audioSource?: 'direct_user_upload' | 'live_microphone_capture',
  ) => void;
  isLoading: boolean;
}

const AUDIO_SAMPLE_RATE = 16_000;
const AUDIO_MAX_BYTES = 4 * 1024 * 1024;
const AUDIO_MAX_SECONDS = 60;

const resampleMono = (input: Float32Array, inputRate: number): Float32Array => {
  if (inputRate === AUDIO_SAMPLE_RATE) return input;
  const outputLength = Math.max(1, Math.round(input.length * AUDIO_SAMPLE_RATE / inputRate));
  const output = new Float32Array(outputLength);
  const scale = (input.length - 1) / Math.max(1, outputLength - 1);
  for (let index = 0; index < outputLength; index += 1) {
    const position = index * scale;
    const lower = Math.floor(position);
    const upper = Math.min(lower + 1, input.length - 1);
    const fraction = position - lower;
    output[index] = input[lower] * (1 - fraction) + input[upper] * fraction;
  }
  return output;
};

const encodePcmWav = (samples: Float32Array): Blob => {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeAscii = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };
  writeAscii(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(8, 'WAVE');
  writeAscii(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, AUDIO_SAMPLE_RATE, true);
  view.setUint32(28, AUDIO_SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(44 + index * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
};

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, isLoading }) => {
  const [inputText, setInputText] = useState('');
  const [selectedImage, setSelectedImage] = useState<string | undefined>(undefined);
  const [selectedImageMimeType, setSelectedImageMimeType] = useState<string | undefined>(undefined);
  const [selectedAudio, setSelectedAudio] = useState<string | undefined>(undefined);
  const [selectedAudioMimeType, setSelectedAudioMimeType] = useState<string | undefined>(undefined);
  const [selectedAudioSource, setSelectedAudioSource] = useState<'direct_user_upload' | 'live_microphone_capture' | undefined>(undefined);
  const [isRecording, setIsRecording] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioChunksRef = useRef<Float32Array[]>([]);
  const inputSampleRateRef = useRef(AUDIO_SAMPLE_RATE);
  const recordingTimerRef = useRef<number | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => () => {
    if (recordingTimerRef.current !== null) window.clearTimeout(recordingTimerRef.current);
    processorRef.current?.disconnect();
    mediaSourceRef.current?.disconnect();
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      void audioContextRef.current.close();
    }
  }, []);

  const handleTextChange = (e: ChangeEvent<HTMLInputElement>) => {
    setInputText(e.target.value);
  };

  const handleImageSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!['image/jpeg', 'image/png'].includes(file.type)) {
        alert('Please choose a JPEG or PNG image.');
        e.target.value = '';
        return;
      }
      if (file.size > 8 * 1024 * 1024) {
        alert('Images must be 8 MB or smaller.');
        e.target.value = '';
        return;
      }
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = (reader.result as string).split(',')[1];
        setSelectedImage(base64String);
        setSelectedImageMimeType(file.type);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAudioSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const isWav = ['audio/wav', 'audio/x-wav', 'audio/wave'].includes(file.type)
        || file.name.toLowerCase().endsWith('.wav');
      if (!isWav) {
        alert('Please choose a PCM WAV file. Other audio formats are not accepted.');
        e.target.value = '';
        return;
      }
      if (file.size > AUDIO_MAX_BYTES) {
        alert('Audio must be 4 MB or smaller.');
        e.target.value = '';
        return;
      }
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = (reader.result as string).split(',')[1];
        setSelectedAudio(base64String);
        setSelectedAudioMimeType('audio/wav');
        setSelectedAudioSource('direct_user_upload');
      };
      reader.readAsDataURL(file);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const context = new AudioContext();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        audioChunksRef.current.push(new Float32Array(event.inputBuffer.getChannelData(0)));
        event.outputBuffer.getChannelData(0).fill(0);
      };
      source.connect(processor);
      processor.connect(context.destination);
      audioContextRef.current = context;
      mediaSourceRef.current = source;
      processorRef.current = processor;
      inputSampleRateRef.current = context.sampleRate;
      audioChunksRef.current = [];
      setIsRecording(true);
      recordingTimerRef.current = window.setTimeout(stopRecording, AUDIO_MAX_SECONDS * 1000);
    } catch (error) {
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        await audioContextRef.current.close();
      }
      audioContextRef.current = null;
      console.error('Error accessing microphone:', error);
      alert('Could not access microphone. Please ensure it is enabled.');
    }
  };

  async function stopRecording() {
    if (!audioContextRef.current) return;
    if (recordingTimerRef.current !== null) window.clearTimeout(recordingTimerRef.current);
    processorRef.current?.disconnect();
    mediaSourceRef.current?.disconnect();
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    await audioContextRef.current.close();
    const sampleCount = audioChunksRef.current.reduce((total, chunk) => total + chunk.length, 0);
    const combined = new Float32Array(sampleCount);
    let offset = 0;
    audioChunksRef.current.forEach((chunk) => {
      combined.set(chunk, offset);
      offset += chunk.length;
    });
    const wavBlob = encodePcmWav(resampleMono(combined, inputSampleRateRef.current));
    audioContextRef.current = null;
    mediaStreamRef.current = null;
    mediaSourceRef.current = null;
    processorRef.current = null;
    audioChunksRef.current = [];
    recordingTimerRef.current = null;
    setIsRecording(false);
    if (wavBlob.size <= 44 || wavBlob.size > AUDIO_MAX_BYTES) {
      alert('The recording was empty or exceeded the 4 MB audio limit.');
      return;
    }
    const reader = new FileReader();
    reader.onloadend = () => {
      setSelectedAudio((reader.result as string).split(',')[1]);
      setSelectedAudioMimeType('audio/wav');
      setSelectedAudioSource('live_microphone_capture');
    };
    reader.readAsDataURL(wavBlob);
  }

  const handleSend = () => {
    if (inputText.trim() || selectedImage || selectedAudio) {
      onSendMessage(
        inputText,
        selectedImage,
        selectedAudio,
        selectedImageMimeType,
        selectedAudioMimeType,
        selectedAudioSource,
      );
      setInputText('');
      setSelectedImage(undefined);
      setSelectedImageMimeType(undefined);
      setSelectedAudio(undefined);
      setSelectedAudioMimeType(undefined);
      setSelectedAudioSource(undefined);
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
    <div className="composer-wrap">
      {(selectedImage || selectedAudio) && (
        <div className="composer-attachments">
          {selectedImage && (
            <span className="flex items-center">
              {React.createElement(FiIcons.FiImage as any, { className: "mr-1", size: 16 })} Image attached
              <button onClick={() => { setSelectedImage(undefined); setSelectedImageMimeType(undefined); }} className="ml-2 text-red-500 hover:text-red-700">x</button>
            </span>
          )}
          {selectedAudio && (
            <span className="flex items-center">
              {React.createElement(FiIcons.FiMic as any, { className: "mr-1", size: 16 })} Audio attached
              <button onClick={() => { setSelectedAudio(undefined); setSelectedAudioMimeType(undefined); setSelectedAudioSource(undefined); }} className="ml-2 text-red-500 hover:text-red-700">x</button>
            </span>
          )}
        </div>
      )}
      <div className="composer">
        <input
          type="text"
          className="composer-input"
          placeholder="Share a thought, question, or problem…"
          value={inputText}
          onChange={handleTextChange}
          onKeyPress={handleKeyPress}
          disabled={isLoading || isRecording}
        />
        <input
          type="file"
          accept="image/jpeg,image/png"
          ref={fileInputRef}
          onChange={handleImageSelect}
          className="hidden"
          disabled={isLoading || isRecording}
          aria-label="Select image file"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="composer-tool"
          title="Attach Image"
          disabled={isLoading || isRecording}
        >
          {React.createElement(FiIcons.FiImage as any, { className: "text-gray-700 dark:text-gray-300", size: 20 })}
        </button>

        <input
          type="file"
          accept="audio/wav,audio/x-wav,.wav"
          ref={audioInputRef}
          onChange={handleAudioSelect}
          className="hidden"
          disabled={isLoading || isRecording}
          aria-label="Select audio file"
        />
        <button
          onClick={() => audioInputRef.current?.click()}
          className="composer-tool"
          title="Attach PCM WAV"
          disabled={isLoading || isRecording}
        >
          {React.createElement(FiIcons.FiUpload as any, { className: "text-gray-700 dark:text-gray-300", size: 20 })}
        </button>
        <button
          onClick={isRecording ? stopRecording : startRecording}
          className={`composer-tool ${isRecording ? 'is-recording' : ''}`}
          title={isRecording ? "Stop Recording" : "Record Audio"}
          disabled={isLoading}
        >
          {isRecording ? React.createElement(FiIcons.FiStopCircle as any, { className: "text-white", size: 20 }) : React.createElement(FiIcons.FiMic as any, { className: "text-gray-700 dark:text-gray-300", size: 20 })}
        </button>

        <button
          onClick={handleSend}
          className="composer-send"
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
