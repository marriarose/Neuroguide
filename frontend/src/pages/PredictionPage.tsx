import React, { useState } from 'react';
import { Brain, Upload, Sparkles, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { api } from '../services/api';
import { PredictionResponse } from '../types';

interface PredictionPageProps {
  isModelTrained: boolean;
}

export const PredictionPage: React.FC<PredictionPageProps> = ({ isModelTrained }) => {
  const [activeMode, setActiveMode] = useState<'vector' | 'file'>('vector');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [filePredictions, setFilePredictions] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleGenerateSampleVector = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const sampleFeatures: number[][] = [];
      for (let band = 0; band < 5; band++) {
        const bandRow: number[] = [];
        for (let ch = 0; ch < 62; ch++) {
          bandRow.push(parseFloat((Math.random() * 5.0 - 2.0).toFixed(4)));
        }
        sampleFeatures.push(bandRow);
      }

      const res = await api.predictFeatures(sampleFeatures);
      setPrediction(res);
    } catch (err: any) {
      setErrorMsg(err.message || 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await api.predictFile(selectedFile);
      setFilePredictions(res);
    } catch (err: any) {
      setErrorMsg(err.message || 'File prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  const chartData = prediction
    ? Object.entries(prediction.probabilities).map(([name, prob]) => ({
        name: name.toUpperCase(),
        probability: parseFloat((prob * 100).toFixed(1)),
      }))
    : [];

  const classColors: Record<string, string> = {
    DISGUST: '#a855f7',
    FEAR: '#ec4899',
    SAD: '#3b82f6',
    NEUTRAL: '#94a3b8',
    SURPRISE: '#eab308',
    HAPPY: '#10b981',
    ANGER: '#ef4444',
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5">
        <h2 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
          <Brain className="w-6 h-6 text-cyan-400" />
          <span>Real-Time Emotion Prediction</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Classify 7 SEED-VII discrete emotion categories (Disgust, Fear, Sad, Neutral, Surprise, Happy, Anger) from EEG DE features.
        </p>
      </div>

      {!isModelTrained && (
        <div className="bg-rose-950/40 border border-rose-800/60 rounded-xl p-4 text-xs text-rose-300 flex items-center space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0" />
          <span>
            The model has not been trained yet. Please train the model in the Model Training tab before requesting predictions.
          </span>
        </div>
      )}

      {errorMsg && (
        <div className="bg-rose-950/50 border border-rose-800 rounded-xl p-4 text-xs text-rose-300 flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Mode Switcher */}
      <div className="flex space-x-2 bg-slate-900 p-1.5 rounded-xl border border-slate-800 max-w-md">
        <button
          onClick={() => setActiveMode('vector')}
          className={`flex-1 py-2 rounded-lg text-xs font-semibold transition-all ${
            activeMode === 'vector' ? 'bg-cyan-500 text-slate-950 shadow-md' : 'text-slate-400 hover:text-white'
          }`}
        >
          Single 5x62 Feature Matrix
        </button>
        <button
          onClick={() => setActiveMode('file')}
          className={`flex-1 py-2 rounded-lg text-xs font-semibold transition-all ${
            activeMode === 'file' ? 'bg-cyan-500 text-slate-950 shadow-md' : 'text-slate-400 hover:text-white'
          }`}
        >
          Upload .MAT Feature File
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        {activeMode === 'vector' ? (
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider text-slate-300">
              EEG DE Feature Matrix (5 Bands x 62 Channels)
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Input a 5x62 EEG Differential Entropy feature matrix representing Delta, Theta, Alpha, Beta, Gamma across 62 electrode channels.
            </p>

            <button
              onClick={handleGenerateSampleVector}
              disabled={!isModelTrained || loading}
              className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-sm transition-all shadow-lg shadow-cyan-500/20 disabled:opacity-40 flex items-center justify-center space-x-2"
            >
              <Sparkles className="w-4 h-4" />
              <span>{loading ? 'Running Forward Pass...' : 'Generate Sample & Predict 7-Class Emotion'}</span>
            </button>
          </div>
        ) : (
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider text-slate-300">
              Upload MATLAB .MAT Feature File
            </h3>

            <form onSubmit={handleFileUpload} className="space-y-4">
              <input
                type="file"
                accept=".mat"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                disabled={!isModelTrained || loading}
                className="w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-cyan-400 hover:file:bg-slate-700"
              />

              <button
                type="submit"
                disabled={!selectedFile || !isModelTrained || loading}
                className="w-full py-3 px-4 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-sm transition-all shadow-md shadow-cyan-500/20 disabled:opacity-40 flex items-center justify-center space-x-2"
              >
                <Upload className="w-4 h-4" />
                <span>{loading ? 'Processing File...' : 'Upload & Predict File'}</span>
              </button>
            </form>
          </div>
        )}

        {/* Results Card & Chart */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6">
          <h3 className="text-sm font-semibold text-white uppercase tracking-wider text-slate-300">
            Prediction Probability Distribution
          </h3>

          {prediction ? (
            <div className="space-y-5">
              <div className="flex items-center justify-between bg-slate-950 p-4 rounded-xl border border-slate-800">
                <div>
                  <div className="text-[10px] uppercase text-slate-500 font-bold">Predicted Emotion State</div>
                  <div className="text-2xl font-extrabold capitalize mt-0.5 text-cyan-400">
                    {prediction.predicted_class}
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-[10px] uppercase text-slate-500 font-bold">Confidence Score</div>
                  <div className="text-2xl font-mono font-extrabold text-cyan-400">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* 7 Class Probability Chart */}
              <div className="h-56 pt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <XAxis dataKey="name" stroke="#64748b" fontSize={10} />
                    <YAxis stroke="#64748b" fontSize={10} domain={[0, 100]} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                      itemStyle={{ fontSize: '12px' }}
                    />
                    <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={classColors[entry.name] || '#00bcd4'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <p className="text-[11px] text-slate-500 italic leading-tight border-t border-slate-800/80 pt-3">
                {prediction.disclaimer}
              </p>
            </div>
          ) : filePredictions ? (
            <div className="space-y-3 max-h-80 overflow-y-auto">
              <div className="text-xs text-slate-300 font-bold">
                Processed {filePredictions.arrays_processed} feature arrays from <span className="font-mono text-cyan-400">{filePredictions.filename}</span>
              </div>
              {filePredictions.predictions.map((p: any, idx: number) => (
                <div key={idx} className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-xs flex justify-between items-center font-mono">
                  <span className="text-slate-300">{p.variable_name} ({p.num_windows} windows)</span>
                  <span className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800 font-bold capitalize">
                    {p.majority_predicted_class}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-64 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-center text-slate-500 text-xs">
              No prediction generated yet. Click "Generate Sample & Predict 7-Class Emotion".
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
