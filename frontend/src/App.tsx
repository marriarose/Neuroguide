import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { OverviewPage } from './pages/OverviewPage';
import { DatasetExplorerPage } from './pages/DatasetExplorerPage';
import { TrainingPage } from './pages/TrainingPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { PredictionPage } from './pages/PredictionPage';
import { AboutModelPage } from './pages/AboutModelPage';
import { api } from './services/api';
import { HealthResponse, DatasetStatusResponse, EvaluationMetricsResponse } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatusResponse | null>(null);
  const [metrics, setMetrics] = useState<EvaluationMetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchAppData = async () => {
    setLoading(true);
    try {
      const h = await api.getHealth().catch(() => null);
      setHealth(h);

      const ds = await api.getDatasetStatus('de_LDS').catch(() => null);
      setDatasetStatus(ds);

      const m = await api.getEvaluationMetrics().catch(() => null);
      setMetrics(m);
    } catch (err) {
      console.error('Failed to load application status:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAppData();
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-slate-950">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        datasetReady={datasetStatus?.training_permitted || false}
        modelTrained={health?.model_trained || false}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'overview' && (
          <OverviewPage
            health={health}
            datasetStatus={datasetStatus}
            metrics={metrics}
            setActiveTab={setActiveTab}
          />
        )}

        {activeTab === 'dataset' && (
          <DatasetExplorerPage status={datasetStatus} onRefresh={fetchAppData} />
        )}

        {activeTab === 'training' && (
          <TrainingPage datasetStatus={datasetStatus} onTrainingComplete={fetchAppData} />
        )}

        {activeTab === 'evaluation' && <EvaluationPage metrics={metrics} />}

        {activeTab === 'prediction' && (
          <PredictionPage isModelTrained={health?.model_trained || false} />
        )}

        {activeTab === 'about' && <AboutModelPage />}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>NeuroGuide — EEG-Based Emotion Classification Using EEGNet</span>
          <span className="font-mono">Backend: FastAPI 1.0 | PyTorch 2.x | React 18</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
