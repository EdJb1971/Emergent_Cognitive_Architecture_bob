import axios from 'axios';
import { API_BASE_URL, API_KEY } from './config';
import {
  AutonomousEvent,
  AutonomousRuntime,
  AutonomousTask,
  AutonomousTaskType,
} from 'types/autonomous';

const client = axios.create({ baseURL: API_BASE_URL });
client.interceptors.request.use((config) => {
  if (!API_KEY) throw new Error('VITE_API_KEY is not configured.');
  config.headers['X-API-Key'] = API_KEY;
  return config;
});

const call = async <T>(operation: () => Promise<{ data: T }>): Promise<T> => {
  try { return (await operation()).data; }
  catch (error) {
    if (axios.isAxiosError(error)) throw new Error(error.response?.data?.detail || error.message);
    throw error;
  }
};

export const getAutonomousRuntime = () =>
  call<AutonomousRuntime>(() => client.get('/api/autonomous-work/runtime'));

export const updateAutonomousRuntime = (body: {
  master_enabled?: boolean;
  category_enabled?: Partial<Record<AutonomousTaskType, boolean>>;
  reason: string;
}) => call<AutonomousRuntime>(() => client.put('/api/autonomous-work/runtime', body));

export const getAutonomousTasks = () =>
  call<{ tasks: AutonomousTask[]; count: number }>(() =>
    client.get('/api/autonomous-work/tasks', { params: { limit: 200 } })
  );

export const getAutonomousLedger = () =>
  call<{ events: AutonomousEvent[]; count: number; integrity_verified: boolean }>(() =>
    client.get('/api/autonomous-work/ledger', { params: { limit: 200 } })
  );

export const cancelAutonomousTask = (taskId: string) =>
  call<AutonomousTask>(() => client.post(`/api/autonomous-work/tasks/${taskId}/cancel`, {
    reason: 'Operator cancellation from executive control.',
  }));

export const retryAutonomousTask = (taskId: string) =>
  call<AutonomousTask>(() => client.post(`/api/autonomous-work/tasks/${taskId}/retry`, {
    reason: 'Operator retry from executive control.',
  }));

