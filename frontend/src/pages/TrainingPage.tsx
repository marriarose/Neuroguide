import React, { useState, useEffect } from 'react';
import { Cpu, Play, AlertCircle, RefreshCw, Sliders, Activity, Eye, Layers } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { api } from '../services/api';
import { DatasetStatusResponse, TrainingStatusResponse, TrainingConfigRequest } from '../types';

interface TrainingPageProps {
  datasetStatus: DatasetStatusResponse | null;
  onTrainingComplete: () => void;
}

export const TrainingPage: React.FC<TrainingPageProps> = ({ datasetStatus, onTrainingComplete }) => {
  const [config, setConfig] = useState<TrainingConfigRequest>({
    feature_type: 'de_LDS',
    task_mode: '7_class',
    modality: 'eeg_only',
    batch_size: 64,
    learning_rate: 0.001,
    epochs: 35,
    dropout: 0.25,
    optimizer: 'adam',
    random_seed: 42,
    early_stopping_patience: 8,
    class_weighted_loss: false,
  });

  const [status, setStatus] = useState<TrainingStatusResponse | null>(null);
  const [starting, setStarting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(() => {
      fetchStatus();
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await api.getTrainingStatus();
      setStatus(res);
      if (res.status === 'completed') {
        onTrainingComplete();
      }
    } catch (err) {
      console.error('Failed to get training status:', err);
    }
  };

  const handleStartTraining = async () => {
    setStarting(true);
    setErrorMsg(null);
    try {
      const res = await api.startTraining(config);
      setStatus(res);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to start training.');
    } finally {
      setStarting(false);
    }
  };

  const isRunning = status?.status === 'running';

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5">
        <h2 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
          <Cpu className="w-6 h-6 text-cyan-400" />
          <span>SEED-VII EEGNet Model Training Control Center</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Train the 7-class EEGNet emotion classifier or multimodal EEG+Eye fusion model with participant-independent splitting.
        </p>
      </div>

      {errorMsg && (
        <div className="bg-rose-950/50 border border-rose-800 rounded-xl p-4 text-rose-300 text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Hyperparameters Config Form */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5">
          <h3 className="text-sm font-semibold text-white uppercase tracking-wider text-slate-300 flex items-center space-x-2">
            <Sliders className="w-4 h-4 text-cyan-400" />
            <span>Training Configuration</span>
          </h3>

          <div className="space-y-4 text-xs">
            <div>
              <label className="block font-medium text-slate-300 mb-1">Emotion Classification Task</label>
              <select
                value={config.task_mode}
                onChange={(e) => setConfig({ ...config, task_mode: e.target.value as any })}
                disabled={isRunning}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-cyan-400 font-bold focus:border-cyan-500 focus:outline-none"
              >
                <option value="7_class">7 Discrete Emotion Classes (Disgust, Fear, Sad, Neutral, Surprise, Happy, Anger)</option>
                <option value="3_class">3 Mapping Classes (Negative, Neutral, Positive)</option>
                <option value="regression">Continuous Rating Regression</option>
              </select>
            </div>

            <div>
              <label className="block font-medium text-slate-300 mb-1">Modality Strategy</label>
              <select
                value={config.modality}
                onChange={(e) => setConfig({ ...config, modality: e.target.value as any })}
                disabled={isRunning}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-purple-400 font-bold focus:border-cyan-500 focus:outline-none"
              >
                <option value="eeg_only">EEG Baseline (5 Bands x 62 Channels)</option>
                <option value="eeg_eye_multimodal">Multimodal Fusion (EEG Baseline + 33 Eye Features)</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block font-medium text-slate-300 mb-1">Epochs</label>
                <input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) || 35 })}
                  disabled={isRunning}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:border-cyan-500 focus:outline-none font-mono"
                />
              </div>

              <div>
                <label className="block font-medium text-slate-300 mb-1">Batch Size</label>
                <input
                  type="number"
                  value={config.batch_size}
                  onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) || 64 })}
                  disabled={isRunning}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:border-cyan-500 focus:outline-none font-mono"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block font-medium text-slate-300 mb-1">Learning Rate</label>
                <input
                  type="number"
                  step="0.0001"
                  value={config.learning_rate}
                  onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) || 0.001 })}
                  disabled={isRunning}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:border-cyan-500 focus:outline-none font-mono"
                />
              </div>

              <div>
                <label className="block font-medium text-slate-300 mb-1">Dropout Rate</label>
                <input
                  type="number"
                  step="0.05"
                  value={config.dropout}
                  onChange={(e) => setConfig({ ...config, dropout: parseFloat(e.target.value) || 0.25 })}
                  disabled={isRunning}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:border-cyan-500 focus:outline-none font-mono"
                />
              </div>
            </div>

            <button
              onClick={handleStartTraining}
              disabled={isRunning || starting}
              className="w-full mt-4 py-3 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-sm transition-all shadow-lg shadow-cyan-500/20 disabled:opacity-40 flex items-center justify-center space-x-2"
            >
              {starting || isRunning ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>{isRunning ? 'Training in Progress...' : 'Initializing...'}</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Start SEED-VII Model Training</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Live Progress Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6 lg:col-span-2">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider text-slate-300 flex items-center space-x-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>Training Live Progress</span>
            </h3>

            <span
              className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                status?.status === 'running'
                  ? 'bg-cyan-950 text-cyan-400 border border-cyan-800 animate-pulse'
                  : status?.status === 'completed'
                  ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                  : 'bg-slate-800 text-slate-400'
              }`}
            >
              {status?.status || 'idle'}
            </span>
          </div>

          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1.5 font-mono">
              <span>Epoch {status?.current_epoch || 0} / {status?.total_epochs || config.epochs}</span>
              <span>
                Val Acc:{' '}
                <strong className="text-cyan-400">
                  {status?.val_acc ? `${(status.val_acc * 100).toFixed(1)}%` : 'N/A'}
                </strong>
              </span>
            </div>
            <div className="w-full h-3 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
                style={{
                  width: `${
                    ((status?.current_epoch || 0) / (status?.total_epochs || config.epochs || 1)) * 100
                  }%`,
                }}
              />
            </div>
          </div>

          <div className="grid grid-cols-4 gap-3 text-center">
            <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-500 uppercase">Train Loss</div>
              <div className="text-sm font-bold font-mono text-slate-200">{status?.train_loss?.toFixed(4) || '—'}</div>
            </div>
            <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-500 uppercase">Val Loss</div>
              <div className="text-sm font-bold font-mono text-slate-200">{status?.val_loss?.toFixed(4) || '—'}</div>
            </div>
            <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-500 uppercase">Train Acc</div>
              <div className="text-sm font-bold font-mono text-cyan-400">
                {status?.train_acc ? `${(status.train_acc * 100).toFixed(1)}%` : '—'}
              </div>
            </div>
            <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-500 uppercase">Val Acc</div>
              <div className="text-sm font-bold font-mono text-cyan-400">
                {status?.val_acc ? `${(status.val_acc * 100).toFixed(1)}%` : '—'}
              </div>
            </div>
          </div>

          {status?.history && status.history.length > 0 ? (
            <div className="h-64 pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={status.history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                    itemStyle={{ fontSize: '12px' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                  <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#00bcd4" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="#ff4081" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val_acc" name="Val Acc" stroke="#4caf50" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-48 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-center text-slate-500 text-xs">
              No active training run. Configure parameters and click Start SEED-VII Model Training.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
