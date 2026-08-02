import axios from 'axios';
import { API_BASE_URL, API_KEY } from './config';
import { CleanStartStatus, IdentityProfile } from 'types/settings';

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

export const getIdentity = () => call<IdentityProfile>(() => client.get('/api/settings/identity'));
export const updateIdentity = (body: { assistant_name: string; user_name: string | null; expected_revision: number }) =>
  call<IdentityProfile>(() => client.put('/api/settings/identity', body));
export const getCleanStartStatus = () => call<CleanStartStatus>(() => client.get('/api/settings/clean-start'));
export const armCleanStart = (confirmation: string) => call<CleanStartStatus>(() => client.post('/api/settings/clean-start', { confirmation, preserve_identity: true }));
export const cancelCleanStart = () => call<CleanStartStatus>(() => client.delete('/api/settings/clean-start'));
