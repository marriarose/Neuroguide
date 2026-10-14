import React from 'react';
import { Database, Cpu, BarChart3, Brain, ArrowRight, AlertTriangle, CheckCircle2, ShieldAlert } from 'lucide-react';
import { StatCard } from '../components/StatCard';
import { HealthResponse, DatasetStatusResponse, EvaluationMetricsResponse } from '../types';

interface OverviewPageProps {
  health: HealthResponse | null;
  datasetStatus: DatasetStatusResponse | null;
  metrics: EvaluationMetricsResponse | null;
  setActiveTab: (tab: string) => void;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  health,
  datasetStatus,
  metrics,
  setActiveTab,
}) => {
  return (
    <div className="space-y-8">
      {/* Banner */}
      <div className="relative overflow-hidden bg-gradient-to-r from-slate-900 via-slate-850 to-slate-900 border border-slate-800 rounded-2xl p-8 shadow-xl">
        <div className="absolute -right-12 -top-12 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-800/60 text-cyan-400 text-xs font-semibold">
            <span>EEGNet-Inspired Deep Learning Pipeline</span>
          </div>
          <h2 className="text-3xl font-extrabold text-white tracking-tight sm:text-4xl">
            EEG-Based Emotion Classification
          </h2>
          <p className="text-slate-300 text-sm leading-relaxed">
            NeuroGuide extracts Differential Entropy (DE) and Linear Dynamical System (LDS) smoothed features from
            62-channel EEG signals across 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma) to classify emotional states into{' '}
            <span className="text-rose-400 font-semibold">Negative</span>,{' '}
            <span className="text-slate-300 font-semibold">Neutral</span>, or{' '}
            <span className="text-emerald-400 font-semibold">Positive</span> classes using PyTorch.
          </p>

          <div className="pt-2 flex flex-wrap gap-3">
            <button
              onClick={() => setActiveTab('dataset')}
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-sm transition-all shadow-lg shadow-cyan-500/20"
            >
              <Database className="w-4 h-4" />
              <span>Explore Dataset</span>
            </button>
            <button
              onClick={() => setActiveTab('prediction')}
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-sm border border-slate-700 transition-all"
            >
              <Brain className="w-4 h-4 text-cyan-400" />
              <span>Run Emotion Inference</span>
            </button>
          </div>
        </div>
      </div>

      {/* Dataset Verification Alert if Missing Labels */}
      {datasetStatus && !datasetStatus.training_permitted && (
        <div className="bg-amber-950/40 border border-amber-800/60 rounded-xl p-5 flex items-start space-x-4 shadow-sm">
          <AlertTriangle className="w-6 h-6 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="text-sm font-semibold text-amber-200">Dataset Verification & Labels Required</h4>
            <p className="text-xs text-amber-300/80 leading-relaxed">
              MATLAB feature files (<code className="bg-amber-900/50 px-1 py-0.5 rounded">1.mat</code>..
              <code className="bg-amber-900/50 px-1 py-0.5 rounded">20.mat</code>) were detected, but verified emotion labels are missing in <code className="bg-amber-900/50 px-1 py-0.5 rounded">labels.csv</code>.
              To preserve academic integrity, NeuroGuide will never synthesize random labels for training.
            </p>
            <button
              onClick={() => setActiveTab('dataset')}
              className="inline-flex items-center space-x-1 text-xs font-semibold text-amber-400 hover:underline pt-1"
            >
              <span>View label validation error details</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="MAT Files Detected"
          value={datasetStatus?.files_detected || 0}
          subtitle={`Raw files in data/raw/`}
          icon={Database}
          badge={datasetStatus?.files_detected ? 'Detected' : 'Empty'}
          badgeColor={datasetStatus?.files_detected ? 'cyan' : 'slate'}
        />
        <StatCard
          title="Usable EEG Windows"
          value={datasetStatus?.total_usable_windows || 0}
          subtitle="5 bands x 62 channels"
          icon={Brain}
          badge={`${datasetStatus?.selected_feature_type || 'de_LDS'}`}
          badgeColor="emerald"
        />
        <StatCard
          title="Model Status"
          value={health?.model_trained ? 'Trained' : 'Untrained'}
          subtitle={`Device: ${health?.device || 'CPU'}`}
          icon={Cpu}
          badge={health?.model_trained ? 'Checkpoint Ready' : 'Training Needed'}
          badgeColor={health?.model_trained ? 'cyan' : 'amber'}
        />
        <StatCard
          title="Verified Test Accuracy"
          value={metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : 'N/A'}
          subtitle={metrics ? `Macro F1: ${metrics.macro_f1.toFixed(3)}` : 'Model evaluation pending'}
          icon={BarChart3}
          badge={metrics ? 'Verified' : 'No Results'}
          badgeColor={metrics ? 'emerald' : 'slate'}
        />
      </div>

      {/* Quick Navigation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div
          onClick={() => setActiveTab('dataset')}
          className="bg-slate-900 border border-slate-800 hover:border-cyan-500/50 rounded-xl p-6 cursor-pointer group transition-all"
        >
          <div className="w-10 h-10 rounded-lg bg-cyan-950 border border-cyan-800/40 text-cyan-400 flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
            <Database className="w-5 h-5" />
          </div>
          <h3 className="text-base font-semibold text-white group-hover:text-cyan-400 transition-colors">
            1. Dataset Explorer
          </h3>
          <p className="mt-2 text-xs text-slate-400 leading-relaxed">
            Inspect MATLAB arrays, check shape consistency (num_windows, 5, 62), download <code className="text-slate-300">labels.csv</code> template, and validate array-to-label mappings.
          </p>
        </div>

        <div
          onClick={() => setActiveTab('training')}
          className="bg-slate-900 border border-slate-800 hover:border-cyan-500/50 rounded-xl p-6 cursor-pointer group transition-all"
        >
          <div className="w-10 h-10 rounded-lg bg-blue-950 border border-blue-800/40 text-blue-400 flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
            <Cpu className="w-5 h-5" />
          </div>
          <h3 className="text-base font-semibold text-white group-hover:text-cyan-400 transition-colors">
            2. EEGNet Training
          </h3>
          <p className="mt-2 text-xs text-slate-400 leading-relaxed">
            Configure learning rate, batch size, epochs, and grouped leakage-safe train/val/test splitting to train the PyTorch EEGNet-inspired model.
          </p>
        </div>

        <div
          onClick={() => setActiveTab('prediction')}
          className="bg-slate-900 border border-slate-800 hover:border-cyan-500/50 rounded-xl p-6 cursor-pointer group transition-all"
        >
          <div className="w-10 h-10 rounded-lg bg-purple-950 border border-purple-800/40 text-purple-400 flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
            <Brain className="w-5 h-5" />
          </div>
          <h3 className="text-base font-semibold text-white group-hover:text-cyan-400 transition-colors">
            3. Emotion Prediction
          </h3>
          <p className="mt-2 text-xs text-slate-400 leading-relaxed">
            Input a 5-band x 62-channel feature vector or upload a MATLAB file to classify Negative, Neutral, or Positive emotion states in real time.
          </p>
        </div>
      </div>
    </div>
  );
};
