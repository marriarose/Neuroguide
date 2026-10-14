import React from 'react';
import { Activity, Database, Cpu, BarChart3, Brain, Info } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  datasetReady: boolean;
  modelTrained: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  datasetReady,
  modelTrained,
}) => {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'dataset', label: 'Dataset Explorer', icon: Database },
    { id: 'training', label: 'Model Training', icon: Cpu },
    { id: 'evaluation', label: 'Evaluation', icon: BarChart3 },
    { id: 'prediction', label: 'Prediction', icon: Brain },
    { id: 'about', label: 'About Model', icon: Info },
  ];

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('overview')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                NeuroGuide
              </h1>
              <p className="text-xs text-slate-400 font-mono">EEGNet Emotion Classifier</p>
            </div>
          </div>

          {/* Nav Items */}
          <nav className="hidden md:flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/60'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Status Badges */}
          <div className="flex items-center space-x-2 text-xs">
            <span
              className={`px-2.5 py-1 rounded-full font-medium flex items-center space-x-1.5 border ${
                datasetReady
                  ? 'bg-emerald-950/60 text-emerald-400 border-emerald-800/50'
                  : 'bg-amber-950/60 text-amber-400 border-amber-800/50'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${datasetReady ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
              <span>{datasetReady ? 'Dataset Verified' : 'Labels Required'}</span>
            </span>

            <span
              className={`px-2.5 py-1 rounded-full font-medium flex items-center space-x-1.5 border ${
                modelTrained
                  ? 'bg-cyan-950/60 text-cyan-400 border-cyan-800/50'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${modelTrained ? 'bg-cyan-400' : 'bg-slate-500'}`} />
              <span>{modelTrained ? 'Model Trained' : 'Untrained'}</span>
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
