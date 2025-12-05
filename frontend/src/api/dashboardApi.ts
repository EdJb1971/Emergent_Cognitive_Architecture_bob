import axios from 'axios';
import { DashboardMetrics, HistoricalData } from 'types/dashboard';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const API_KEY = process.env.REACT_APP_API_KEY;

export const getDashboardMetrics = async (): Promise<DashboardMetrics> => {
  if (!API_KEY) {
    console.error('Configuration Error: REACT_APP_API_KEY is not set.');
    throw new Error('API key is missing. Please configure REACT_APP_API_KEY.');
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
    console.error('Configuration Error: REACT_APP_API_KEY is not set.');
    throw new Error('API key is missing. Please configure REACT_APP_API_KEY.');
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
    console.error('Configuration Error: REACT_APP_API_KEY is not set.');
    throw new Error('API key is missing. Please configure REACT_APP_API_KEY.');
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

// WebSocket connection for real-time updates
export const createDashboardWebSocket = (onMessage: (data: any) => void, onError?: (error: Event) => void): WebSocket => {
  const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/ws/dashboard`;

  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('Dashboard WebSocket connected');
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  };

  ws.onerror = (error) => {
    console.error('Dashboard WebSocket error:', error);
    if (onError) {
      onError(error);
    }
  };

  ws.onclose = () => {
    console.log('Dashboard WebSocket disconnected');
  };

  return ws;
};

// Statistical Analysis APIs
export const getDashboardStatisticalAnalysis = async (metricSeries: string, analysisType: string = 'comprehensive') => {
  if (!API_KEY) {
    console.error('Configuration Error: REACT_APP_API_KEY is not set.');
    throw new Error('API key is missing. Please configure REACT_APP_API_KEY.');
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
    console.error('Configuration Error: REACT_APP_API_KEY is not set.');
    throw new Error('API key is missing. Please configure REACT_APP_API_KEY.');
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
    console.error('Configuration Error: REACT_APP_API_KEY is not set.');
    throw new Error('API key is missing. Please configure REACT_APP_API_KEY.');
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
    console.error('Configuration Error: REACT_APP_API_KEY is not set.');
    throw new Error('API key is missing. Please configure REACT_APP_API_KEY.');
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
    console.error('Configuration Error: REACT_APP_API_KEY is not set.');
    throw new Error('API key is missing. Please configure REACT_APP_API_KEY.');
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
    console.error('Configuration Error: REACT_APP_API_KEY is not set.');
    throw new Error('API key is missing. Please configure REACT_APP_API_KEY.');
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
    console.error('Configuration Error: REACT_APP_API_KEY is not set.');
    throw new Error('API key is missing. Please configure REACT_APP_API_KEY.');
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
    console.error('Configuration Error: REACT_APP_API_KEY is not set.');
    throw new Error('API key is missing. Please configure REACT_APP_API_KEY.');
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