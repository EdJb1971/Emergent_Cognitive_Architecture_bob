export interface ChatRequest {
  user_id: string;
  input_text: string;
  session_id: string;
  timestamp: string;
  image_base64?: string;
  audio_base64?: string;
}

export interface ChatResponse {
  user_id: string;
  session_id: string;
  response: string;
}

export interface Message {
  id: string;
  sender: 'user' | 'ai';
  text: string;
  timestamp: string;
  image_url?: string; // For displaying images (e.g., if AI sends one, or user's own)
  audio_url?: string; // For displaying audio (e.g., if AI sends one, or user's own)
  image_base64?: string; // For user's own image input
  audio_base64?: string; // For user's own audio input
  is_loading?: boolean;
  is_error?: boolean;
  is_proactive?: boolean; // Whether this message was initiated by the AI
  proactive_id?: string; // ID of the proactive message for reaction tracking
  trigger_type?: string; // Type of trigger that caused this proactive message
}
