import {
  HealthResponse,
  DatasetStatusResponse,
  TrainingConfigRequest,
  TrainingStatusResponse,
  EvaluationMetricsResponse,
  PredictionResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch (_) {}
    throw new Error(errorDetail);
  }

  return res.json();
}

export const api = {
  getHealth: (): Promise<HealthResponse> => fetchJSON('/api/health'),

  getDatasetStatus: (featureType: string = 'de_LDS'): Promise<DatasetStatusResponse> =>
    fetchJSON(`/api/dataset/status?feature_type=${featureType}`),

  inspectDataset: () => fetchJSON('/api/dataset/inspect', { method: 'POST' }),

  validateDataset: (featureType: string = 'de_LDS') =>
    fetchJSON('/api/dataset/validate', {
      method: 'POST',
      body: JSON.stringify({ feature_type: featureType }),
    }),

  startTraining: (config: TrainingConfigRequest): Promise<TrainingStatusResponse> =>
    fetchJSON('/api/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  getTrainingStatus: (): Promise<TrainingStatusResponse> => fetchJSON('/api/training/status'),

  getEvaluationMetrics: (): Promise<EvaluationMetricsResponse> => fetchJSON('/api/evaluation/metrics'),

  predictFeatures: (features: number[][]): Promise<PredictionResponse> =>
    fetchJSON('/api/predict/features', {
      method: 'POST',
      body: JSON.stringify({ features }),
    }),

  predictFile: async (file: File): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/api/predict/file`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || 'File prediction failed');
    }
    return res.json();
  },
};
