import React from 'react';
import { Info, Brain, Layers, Cpu, ShieldCheck, AlertTriangle, Eye } from 'lucide-react';

export const AboutModelPage: React.FC = () => {
  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-800 pb-5">
        <h2 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
          <Info className="w-6 h-6 text-cyan-400" />
          <span>About SEED-VII & Multimodal EEGNet Architecture</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Technical specifications of the SEED-VII multimodal emotion classification platform across 20 participants and 7 emotion categories.
        </p>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3">
          <div className="w-8 h-8 rounded-lg bg-cyan-950 border border-cyan-800/40 text-cyan-400 flex items-center justify-center">
            <Brain className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-bold text-white">SEED-VII Dataset Protocol</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Collected from 20 participants across 4 sessions (20 trial videos per session = 80 total trials). Evaluated for 7 discrete emotion categories:
            <strong className="text-slate-200"> Disgust, Fear, Sad, Neutral, Surprise, Happy, Anger</strong>.
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3">
          <div className="w-8 h-8 rounded-lg bg-purple-950 border border-purple-800/40 text-purple-400 flex items-center justify-center">
            <Eye className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-bold text-white">Multimodal Eye Feature Integration</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Features 33 eye-tracking dimensions (pupil diameter, gaze dispersion, blink rates, saccade amplitudes) synchronized with 62-channel EEG features (5 bands x 62 channels).
          </p>
        </div>
      </div>

      {/* Architecture Deep Dive */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5">
        <h3 className="text-sm font-semibold text-white uppercase tracking-wider text-slate-300 flex items-center space-x-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span>Multimodal EEGNet Deep Neural Network</span>
        </h3>

        <div className="space-y-4 text-xs text-slate-300 leading-relaxed">
          <div className="grid grid-cols-2 gap-4 font-mono text-[11px]">
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <div className="text-cyan-400 font-bold">EEG Branch</div>
              <div className="text-slate-400 mt-0.5">(1, 5, 62) -&gt; Spatial Conv (1, 62) -&gt; Depthwise (5, 1) -&gt; 32-dim Vector</div>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <div className="text-purple-400 font-bold">Eye Branch</div>
              <div className="text-slate-400 mt-0.5">(33-dim) -&gt; Linear(33, 64) -&gt; ELU -&gt; Linear(64, 32) -&gt; 32-dim Vector</div>
            </div>
          </div>

          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 font-mono text-[11px]">
            <div className="text-emerald-400 font-bold">Fusion Classifier Head</div>
            <div className="text-slate-400 mt-0.5">Concatenates [32-dim EEG, 32-dim Eye] = 64-dim representation -&gt; Linear(64, 7 Logits)</div>
          </div>
        </div>
      </div>

      {/* Subject Independent Evaluation Scope */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-3">
        <div className="font-bold text-white flex items-center space-x-2">
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
          <span>Participant-Independent Generalization</span>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">
          NeuroGuide isolates entire participants during train/val/test splitting. Independent participant testing measures true inter-subject transferability across real-world neural variations.
        </p>
      </div>
    </div>
  );
};
