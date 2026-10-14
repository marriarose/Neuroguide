import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  badge?: string;
  badgeColor?: 'emerald' | 'amber' | 'cyan' | 'slate' | 'rose';
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  badge,
  badgeColor = 'cyan',
}) => {
  const badgeClasses = {
    emerald: 'bg-emerald-950/60 text-emerald-400 border-emerald-800/40',
    amber: 'bg-amber-950/60 text-amber-400 border-amber-800/40',
    cyan: 'bg-cyan-950/60 text-cyan-400 border-cyan-800/40',
    slate: 'bg-slate-800 text-slate-400 border-slate-700',
    rose: 'bg-rose-950/60 text-rose-400 border-rose-800/40',
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm hover:border-slate-700 transition-all">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">{title}</span>
        <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center text-cyan-400">
          <Icon className="w-4 h-4" />
        </div>
      </div>
      <div className="mt-3 flex items-baseline justify-between">
        <div className="text-2xl font-bold text-white tracking-tight">{value}</div>
        {badge && (
          <span className={`px-2 py-0.5 rounded text-xs font-semibold border ${badgeClasses[badgeColor]}`}>
            {badge}
          </span>
        )}
      </div>
      {subtitle && <p className="mt-1 text-xs text-slate-400">{subtitle}</p>}
    </div>
  );
};
