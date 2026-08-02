import axios from 'axios';
import {
  DashboardMetrics,
  HistoricalData,
  TelemetryConnectionState,
  TelemetryDomain,
  TelemetryEvent,
  TelemetryGap,
  TelemetryHello,
} from 'types/dashboard';
import { API_BASE_URL, API_KEY, WS_BASE_URL } from './config';

export const getDashboardMetrics = async (): Promise<DashboardMetrics> => {
  if (!API_KEY) {
    console.error('Configuration Error: VITE_API_KEY is not set.');
    throw new Error('API key is missing. Please configure VITE_API_KEY.');
  }

  try {
    const response = await axios.get<DashboardMetrics>(`${API_BASE_URL}/api/dashboard/metrics`, {
      headers: {
        'X-API-Key': API_KEY,
      },
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Dashboard API Error:', error.response?.data || error.message);
      throw new Error(error.response?.data?.detail || 'Failed to fetch dashboard metrics.');
    }
    console.error('Unknown error:', error);
    throw new Error('An unknown error occurred while fetching dashboard metrics.');
  }
};

export const getDashboardHistory = async (hours: number = 24, metricTypes?: string[]): Promise<HistoricalData> => {
  if (!API_KEY) {
    console.error('Configuration Error: VITE_API_KEY is not set.');
    throw new Error('API key is missing. Please configure VITE_API_KEY.');
  }

  try {
    const params = new URLSearchParams({ hours: hours.toString() });
    if (metricTypes && metricTypes.length > 0) {
      params.append('metric_types', metricTypes.join(','));
    }

    const response = await axios.get<HistoricalData>(`${API_BASE_URL}/api/dashboard/history?${params}`, {
      headers: {
        'X-API-Key': API_KEY,
      },
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Dashboard History API Error:', error.response?.data || error.message);
      throw new Error(error.response?.data?.detail || 'Failed to fetch dashboard history.');
    }
    console.error('Unknown error:', error);
    throw new Error('An unknown error occurred while fetching dashboard history.');
  }
};

export const getDashboardCorrelations = async (hours: number = 24, userId?: string): Promise<any> => {
  if (!API_KEY) {
    console.error('Configuration Error: VITE_API_KEY is not set.');
    throw new Error('API key is missing. Please configure VITE_API_KEY.');
  }

  try {
    const params = new URLSearchParams({ hours: hours.toString() });
    if (userId) {
      params.append('user_id', userId);
    }

    const response = await axios.get(`${API_BASE_URL}/api/dashboard/correlations?${params}`, {
      headers: {
        'X-API-Key': API_KEY,
      },
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Dashboard Correlations API Error:', error.response?.data || error.message);
      throw new Error(error.response?.data?.detail || 'Failed to fetch dashboard correlations.');
    }
    console.error('Unknown error:', error);
    throw new Error('An unknown error occurred while fetching dashboard correlations.');
  }
};

interface TelemetryCallbacks {
  onSnapshot: (data: DashboardMetrics) => void;
  onEvent: (event: TelemetryEvent) => void;
  onGap: (gap: TelemetryGap) => void;
  onStateChange: (state: TelemetryConnectionState) => void;
  onStreamReset?: () => void;
  domains?: TelemetryDomain[];
}

export interface DashboardTelemetryConnection {
  close: () => void;
}

const cursorKey = 'eca.telemetry.cursor.v1';

const encodeProtocolCredential = (value: string): string => {
  const bytes = new TextEncoder().encode(value);
  let binary = '';
  bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};

/** Resumable WebSocket client with bounded exponential reconnect backoff. */
export const connectDashboardTelemetry = (callbacks: TelemetryCallbacks): DashboardTelemetryConnection => {
  let stopped = false;
  let socket: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let attempts = 0;
  let currentStream = '';
  let lastSequence = 0;

  try {
    const stored = JSON.parse(sessionStorage.getItem(cursorKey) || '{}');
    currentStream = typeof stored.streamId === 'string' ? stored.streamId : '';
    lastSequence = Number.isInteger(stored.sequence) ? stored.sequence : 0;
  } catch {
    sessionStorage.removeItem(cursorKey);
  }

  const saveCursor = () => sessionStorage.setItem(
    cursorKey,
    JSON.stringify({ streamId: currentStream, sequence: lastSequence }),
  );

  const scheduleReconnect = () => {
    if (stopped || reconnectTimer) return;
    callbacks.onStateChange('reconnecting');
    const base = Math.min(15000, 500 * (2 ** Math.min(attempts, 5)));
    const delay = Math.round(base * (0.8 + Math.random() * 0.4));
    attempts += 1;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, delay);
  };

  const connect = () => {
    if (stopped) return;
    callbacks.onStateChange(attempts ? 'reconnecting' : 'connecting');
    const params = new URLSearchParams({ after: String(lastSequence), replay: '100' });
    if (callbacks.domains?.length) params.set('domains', callbacks.domains.join(','));
    const protocols = ['eca.telemetry.v1'];
    if (API_KEY) protocols.push(`auth.${encodeProtocolCredential(API_KEY)}`);
    socket = new WebSocket(`${WS_BASE_URL}/ws/dashboard?${params}`, protocols);

    socket.onopen = () => {
      attempts = 0;
      callbacks.onStateChange('live');
    };
    socket.onmessage = (message) => {
      try {
        const envelope = JSON.parse(message.data);
        if (envelope.type === 'hello') {
          const hello = envelope.data as TelemetryHello;
          if (currentStream && currentStream !== hello.stream_id && lastSequence) {
            currentStream = hello.stream_id;
            lastSequence = 0;
            saveCursor();
            callbacks.onStreamReset?.();
            socket?.close(4000, 'stream changed; resubscribe');
            return;
          }
          currentStream = hello.stream_id;
          saveCursor();
        } else if (envelope.type === 'snapshot') {
          callbacks.onSnapshot(envelope.data as DashboardMetrics);
        } else if (envelope.type === 'event') {
          const event = envelope.data as TelemetryEvent;
          if (event.sequence > lastSequence) {
            lastSequence = event.sequence;
            saveCursor();
            callbacks.onEvent(event);
          }
        } else if (envelope.type === 'gap') {
          callbacks.onGap(envelope.data as TelemetryGap);
        }
      } catch (error) {
        console.error('Invalid dashboard telemetry envelope:', error);
      }
    };
    socket.onerror = () => socket?.close();
    socket.onclose = () => scheduleReconnect();
  };

  connect();
  return {
    close: () => {
      stopped = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close(1000, 'operator view closed');
      callbacks.onStateChange('closed');
    },
  };
};

// Statistical Analysis APIs
export const getDashboardStatisticalAnalysis = async (metricSeries: string, analysisType: string = 'comprehensive') => {
  if (!API_KEY) {
    console.error('Configuration Error: VITE_API_KEY is not set.');
    throw new Error('API key is missing. Please configure VITE_API_KEY.');
  }

  try {
    const response = await axios.get(`${API_BASE_URL}/api/dashboard/analysis/statistical`, {
      headers: {
        'X-API-Key': API_KEY,
      },
      params: {
        metric_series: metricSeries,
        analysis_type: analysisType
      }
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Statistical Analysis API Error:', error.response?.data || error.message);
      throw new Error(error.response?.data?.detail || 'Failed to perform statistical analysis.');
    }
    console.error('Unknown error:', error);
    throw new Error('An unknown error occurred during statistical analysis.');
  }
};

export const compareDashboardMetrics = async (group1Metric: string, group2Metric: string, testType: string = 'auto') => {
  if (!API_KEY) {
    console.error('Configuration Error: VITE_API_KEY is not set.');
    throw new Error('API key is missing. Please configure VITE_API_KEY.');
  }

  try {
    const response = await axios.get(`${API_BASE_URL}/api/dashboard/analysis/compare`, {
      headers: {
        'X-API-Key': API_KEY,
      },
      params: {
        group1_metric: group1Metric,
        group2_metric: group2Metric,
        test_type: testType
      }
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Metric Comparison API Error:', error.response?.data || error.message);
      throw new Error(error.response?.data?.detail || 'Failed to compare metrics.');
    }
    console.error('Unknown error:', error);
    throw new Error('An unknown error occurred during metric comparison.');
  }
};

export const getLearningCurvesAnalysis = async () => {
  if (!API_KEY) {
    console.error('Configuration Error: VITE_API_KEY is not set.');
    throw new Error('API key is missing. Please configure VITE_API_KEY.');
  }

  try {
    const response = await axios.get(`${API_BASE_URL}/api/dashboard/analysis/learning-curves`, {
      headers: {
        'X-API-Key': API_KEY,
      }
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Learning Curves API Error:', error.response?.data || error.message);
      throw new Error(error.response?.data?.detail || 'Failed to analyze learning curves.');
    }
    console.error('Unknown error:', error);
    throw new Error('An unknown error occurred during learning curve analysis.');
  }
};

// Research Export APIs
export const exportDashboardData = async (format: 'csv' | 'json', dataType: string) => {
  if (!API_KEY) {
    console.error('Configuration Error: VITE_API_KEY is not set.');
    throw new Error('API key is missing. Please configure VITE_API_KEY.');
  }

  try {
    const endpoint = format === 'csv'
      ? `${API_BASE_URL}/api/dashboard/export/csv`
      : `${API_BASE_URL}/api/dashboard/export/json`;

    const response = await axios.get(endpoint, {
      headers: {
        'X-API-Key': API_KEY,
      },
      params: {
        data_type: dataType,
        ...(format === 'json' && { include_metadata: true })
      },
      responseType: format === 'csv' ? 'text' : 'json'
    });

    if (format === 'csv') {
      // Trigger download for CSV
      const blob = new Blob([response.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `eca_${dataType}_${new Date().toISOString().slice(0, 19).replace(/:/g, '')}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    }

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Export API Error:', error.response?.data || error.message);
      throw new Error(error.response?.data?.detail || 'Failed to export data.');
    }
    console.error('Unknown error:', error);
    throw new Error('An unknown error occurred during data export.');
  }
};

export const generateResearchReport = async (analysisPeriodDays: number = 30) => {
  if (!API_KEY) {
    console.error('Configuration Error: VITE_API_KEY is not set.');
    throw new Error('API key is missing. Please configure VITE_API_KEY.');
  }

  try {
    const response = await axios.get(`${API_BASE_URL}/api/dashboard/export/report`, {
      headers: {
        'X-API-Key': API_KEY,
      },
      params: {
        analysis_period_days: analysisPeriodDays
      }
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Research Report API Error:', error.response?.data || error.message);
      throw new Error(error.response?.data?.detail || 'Failed to generate research report.');
    }
    console.error('Unknown error:', error);
    throw new Error('An unknown error occurred during report generation.');
  }
};

// Proactive Engagement APIs
export interface ProactiveMessage {
  has_message: boolean;
  message_id?: string;
  message?: string;
  trigger_type?: string;
  priority?: number;
  timestamp?: string;
}

export const getProactiveMessage = async (): Promise<ProactiveMessage> => {
  if (!API_KEY) {
    console.error('Configuration Error: VITE_API_KEY is not set.');
    throw new Error('API key is missing. Please configure VITE_API_KEY.');
  }

  try {
    const response = await axios.get<ProactiveMessage>(`${API_BASE_URL}/chat/proactive`, {
      headers: {
        'X-API-Key': API_KEY,
      },
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Proactive Message API Error:', error.response?.data || error.message);
      throw new Error(error.response?.data?.detail || 'Failed to fetch proactive message.');
    }
    console.error('Unknown error:', error);
    throw new Error('An unknown error occurred while fetching proactive message.');
  }
};

export const recordProactiveReaction = async (messageId: string, userResponse: string): Promise<void> => {
  if (!API_KEY) {
    console.error('Configuration Error: VITE_API_KEY is not set.');
    throw new Error('API key is missing. Please configure VITE_API_KEY.');
  }

  try {
    await axios.post(`${API_BASE_URL}/chat/proactive/reaction`, {
      message_id: messageId,
      user_response: userResponse,
    }, {
      headers: {
        'X-API-Key': API_KEY,
      },
    });
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Proactive Reaction API Error:', error.response?.data || error.message);
      throw new Error(error.response?.data?.detail || 'Failed to record proactive reaction.');
    }
    console.error('Unknown error:', error);
    throw new Error('An unknown error occurred while recording proactive reaction.');
  }
};

export const testProactiveMessage = async (triggerType: string, messageContent: string): Promise<any> => {
  if (!API_KEY) {
    console.error('Configuration Error: VITE_API_KEY is not set.');
    throw new Error('API key is missing. Please configure VITE_API_KEY.');
  }

  try {
    const params = new URLSearchParams({
      trigger_type: triggerType,
      message_content: messageContent,
    });
    
    const response = await axios.post(`${API_BASE_URL}/chat/proactive/test?${params.toString()}`, {}, {
      headers: {
        'X-API-Key': API_KEY,
      },
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Test Proactive Message API Error:', error.response?.data || error.message);
      throw new Error(error.response?.data?.detail || 'Failed to create test proactive message.');
    }
    console.error('Unknown error:', error);
    throw new Error('An unknown error occurred while creating test proactive message.');
  }
};
