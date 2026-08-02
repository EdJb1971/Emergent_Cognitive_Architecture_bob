import axios from 'axios';
import {
  PredictiveAssessmentReview,
  PredictiveCalibrationSummary,
  PredictiveLabelRequest,
  PredictiveLedgerEvent,
  PredictiveReviewStatus,
} from 'types/predictive';
import { API_BASE_URL, API_KEY } from './config';

const client = axios.create({ baseURL: API_BASE_URL });
client.interceptors.request.use((config) => {
  if (!API_KEY) throw new Error('VITE_API_KEY is not configured.');
  config.headers['X-API-Key'] = API_KEY;
  return config;
});

const call = async <T>(operation: () => Promise<{ data: T }>): Promise<T> => {
  try {
    return (await operation()).data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.detail || error.message);
    }
    throw error instanceof Error ? error : new Error('Predictive review operation failed.');
  }
};

export const getPredictiveAssessments = (
  reviewStatus?: PredictiveReviewStatus,
  materialOnly = false,
) => call<{ assessments: PredictiveAssessmentReview[]; count: number }>(() =>
  client.get('/api/predictive/assessments', {
    params: { review_status: reviewStatus, material_only: materialOnly, limit: 200 },
  })
);

export const getPredictiveAssessment = (assessmentId: string) =>
  call<PredictiveAssessmentReview>(() =>
    client.get(`/api/predictive/assessments/${assessmentId}`)
  );

export const labelPredictiveAssessment = (assessmentId: string, label: PredictiveLabelRequest) =>
  call<PredictiveLedgerEvent>(() =>
    client.post(`/api/predictive/assessments/${assessmentId}/labels`, label)
  );

export const getPredictiveCalibration = () =>
  call<PredictiveCalibrationSummary>(() => client.get('/api/predictive/calibration/summary'));

export const getPredictiveLedger = (limit = 300) =>
  call<{ events: PredictiveLedgerEvent[]; count: number; next_after_sequence?: number | null }>(() =>
    client.get('/api/predictive/ledger', { params: { limit } })
  );
