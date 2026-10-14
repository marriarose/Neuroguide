import React from 'react';
import { BarChart3, ShieldCheck } from 'lucide-react';
import { EvaluationMetricsResponse } from '../types';

interface EvaluationPageProps {
  metrics: EvaluationMetricsResponse | null;
}

export const EvaluationPage: React.FC<EvaluationPageProps> = ({ metrics }) => {
  if (!metrics) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center space-y-4">
        <BarChart3 className="w-12 h-12 text-slate-600 mx-auto" />
        <h3 className="text-lg font-bold text-white">No Evaluation Results Found</h3>
        <p className="text-xs text-slate-400 max-w-md mx-auto">
          Train the model on the SEED-VII dataset to generate participant-independent held-out test evaluation metrics, 7x7 confusion matrix, and classification report.
        </p>
      </div>
    );
  }

  const { accuracy, macro_precision, macro_recall, macro_f1, per_class, confusion_matrix, classes, split_info } = metrics;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5">
        <h2 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
          <BarChart3 className="w-6 h-6 text-cyan-400" />
          <span>SEED-VII Held-Out Test Set Evaluation Report</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Evaluated on independent participants ({split_info?.test_participants || 4} held-out subjects) isolated from training and validation sets.
        </p>
      </div>

      {/* Primary Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Test Accuracy</div>
          <div className="text-3xl font-extrabold text-cyan-400 mt-2 font-mono">{(accuracy * 100).toFixed(1)}%</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Macro F1-Score</div>
          <div className="text-3xl font-extrabold text-white mt-2 font-mono">{macro_f1.toFixed(4)}</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Macro Precision</div>
          <div className="text-3xl font-extrabold text-white mt-2 font-mono">{macro_precision.toFixed(4)}</div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Macro Recall</div>
          <div className="text-3xl font-extrabold text-white mt-2 font-mono">{macro_recall.toFixed(4)}</div>
        </div>
      </div>

      {/* Data Leakage Prevention Summary */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-start space-x-4">
        <ShieldCheck className="w-6 h-6 text-emerald-400 flex-shrink-0 mt-1" />
        <div className="space-y-1 text-xs">
          <h4 className="font-bold text-slate-200">Subject-Independent Data Leakage Prevention</h4>
          <p className="text-slate-400 leading-relaxed">
            Data split strategy: <span className="text-cyan-400 font-mono">{split_info?.strategy || 'subject_independent_group_split'}</span>.
            All windows from test participants (<strong className="text-slate-200">{split_info?.test_participants || 4} subjects</strong>) are strictly isolated.
            Train Windows: <strong className="text-slate-200">{split_info?.train_samples}</strong> | Test Windows:{' '}
            <strong className="text-slate-200">{split_info?.test_samples}</strong>
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Confusion Matrix Table */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
          <h3 className="text-sm font-semibold text-white uppercase tracking-wider text-slate-300">
            {classes.length}x{classes.length} Confusion Matrix
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-center border-collapse">
              <thead>
                <tr>
                  <th className="p-2 text-left text-slate-500 uppercase border-b border-slate-800">True \ Pred</th>
                  {classes.map((c) => (
                    <th key={c} className="p-2 text-slate-300 capitalize font-mono border-b border-slate-800">
                      {c.slice(0, 4)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {confusion_matrix.map((row, rIdx) => (
                  <tr key={rIdx}>
                    <td className="p-2.5 text-left font-bold text-slate-300 capitalize font-mono">{classes[rIdx]}</td>
                    {row.map((val, cIdx) => {
                      const isDiagonal = rIdx === cIdx;
                      return (
                        <td
                          key={cIdx}
                          className={`p-2.5 font-mono font-bold ${
                            isDiagonal
                              ? 'bg-cyan-950/80 text-cyan-400 border border-cyan-800/60'
                              : val > 0
                              ? 'bg-rose-950/40 text-rose-300'
                              : 'text-slate-600'
                          }`}
                        >
                          {val}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Per Class Breakdown Table */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
          <h3 className="text-sm font-semibold text-white uppercase tracking-wider text-slate-300">
            Per-Class Classification Performance
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="text-slate-500 uppercase border-b border-slate-800">
                  <th className="pb-2">Emotion Class</th>
                  <th className="pb-2 text-right">Precision</th>
                  <th className="pb-2 text-right">Recall</th>
                  <th className="pb-2 text-right">F1-Score</th>
                  <th className="pb-2 text-right">Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {Object.entries(per_class).map(([cname, m]) => (
                  <tr key={cname}>
                    <td className="py-2.5 font-sans font-bold text-slate-200 capitalize">{cname}</td>
                    <td className="py-2.5 text-right text-slate-300">{m.precision.toFixed(4)}</td>
                    <td className="py-2.5 text-right text-slate-300">{m.recall.toFixed(4)}</td>
                    <td className="py-2.5 text-right text-cyan-400 font-bold">{m.f1_score.toFixed(4)}</td>
                    <td className="py-2.5 text-right text-slate-500">{m.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
