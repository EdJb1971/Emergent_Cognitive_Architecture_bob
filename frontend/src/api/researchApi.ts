import axios from 'axios';
import {
  CalibrationSummary,
  CognitiveAction,
  InquiryCandidate,
  InquiryDetail,
  InquiryStatus,
  LedgerEvent,
  RuntimeState,
  RuntimeUpdate,
  SourceFeedback,
} from 'types/research';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const API_KEY = process.env.REACT_APP_API_KEY;

const client = axios.create({ baseURL: API_BASE_URL });
client.interceptors.request.use((config) => {
  if (!API_KEY) throw new Error('REACT_APP_API_KEY is not configured.');
  config.headers['X-API-Key'] = API_KEY;
  return config;
});

const message = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    return error.response?.data?.detail || error.message;
  }
  return error instanceof Error ? error.message : 'The operation could not be completed.';
};

const call = async <T>(operation: () => Promise<{ data: T }>): Promise<T> => {
  try {
    return (await operation()).data;
  } catch (error) {
    throw new Error(message(error));
  }
};

export const getResearchRuntime = () =>
  call<RuntimeState>(() => client.get('/api/research/runtime'));

export const updateResearchRuntime = (update: RuntimeUpdate) =>
  call<RuntimeState>(() => client.put('/api/research/runtime', update));

export const getInquiries = (statuses?: InquiryStatus[]) =>
  call<{ inquiries: InquiryCandidate[]; count: number }>(() =>
    client.get('/api/research/inquiries', { params: { statuses, limit: 200 } })
  );

export const getInquiry = (inquiryId: string) =>
  call<InquiryDetail>(() => client.get(`/api/research/inquiries/${inquiryId}`));

export const approveInquiry = (inquiryId: string, reason: string) =>
  call<any>(() => client.post(`/api/research/inquiries/${inquiryId}/approve`, { reason }));

export const dismissInquiry = (inquiryId: string, reason: string) =>
  call<InquiryCandidate>(() =>
    client.post(`/api/research/inquiries/${inquiryId}/dismiss`, { reason })
  );

export const retryInquiry = (inquiryId: string, reason: string) =>
  call<InquiryCandidate>(() =>
    client.post(`/api/research/inquiries/${inquiryId}/retry`, { reason })
  );

export const getCalibrationSummary = () =>
  call<CalibrationSummary>(() => client.get('/api/research/calibration/summary'));

export const getResearchLedger = (limit = 100) =>
  call<{ events: LedgerEvent[]; count: number }>(() =>
    client.get('/api/research/ledger', { params: { limit } })
  );

export const labelAssessment = (
  assessmentId: string,
  appropriateAction: CognitiveAction,
  shouldResearch: boolean,
  rationale: string,
) =>
  call<LedgerEvent>(() =>
    client.post(`/api/research/calibration/${assessmentId}/labels`, {
      appropriate_action: appropriateAction,
      should_external_research: shouldResearch,
      outcome_known: true,
      rationale,
    })
  );

export const recordSourceFeedback = (
  inquiryId: string,
  feedback: SourceFeedback,
) =>
  call<LedgerEvent>(() =>
    client.post(`/api/research/inquiries/${inquiryId}/source-feedback`, feedback)
  );
