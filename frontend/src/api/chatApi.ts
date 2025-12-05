import axios from 'axios';
import { ChatRequest, ChatResponse } from 'types/chat';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
// SEC-001 Fix: Remove hardcoded default API key.
// If REACT_APP_API_KEY is not set, API_KEY will be undefined, and sendMessage will throw an error.
const API_KEY = process.env.REACT_APP_API_KEY; 

export const sendMessage = async (request: ChatRequest): Promise<ChatResponse> => {
  // SEC-001 Fix: Explicitly check for API_KEY before making the request.
  if (!API_KEY) {
    console.error('Configuration Error: REACT_APP_API_KEY is not set.');
    throw new Error('API key is missing. Please configure REACT_APP_API_KEY.');
  }

  try {
    const response = await axios.post<ChatResponse>(`${API_BASE_URL}/chat`, request, {
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
      },
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('API Error:', error.response?.data || error.message);
      throw new Error(error.response?.data?.detail || 'An unexpected error occurred.');
    }
    console.error('Unknown error:', error);
    throw new Error('An unknown error occurred.');
  }
};
