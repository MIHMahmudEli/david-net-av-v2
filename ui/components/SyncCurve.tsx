"use client";

import React from "react";
import { Activity } from "lucide-react";

interface SyncCurveProps {
  syncCurve: number[];
  durationSec?: number;
  currentTime?: number;
  onSeek?: (timeSec: number) => void;
}

export const SyncCurve: React.FC<SyncCurveProps> = ({
  syncCurve,
  durationSec = 4.0,
  currentTime = 0,
  onSeek,
}) => {
  if (!syncCurve || syncCurve.length === 0) return null;

  const minVal = -1.0;
  const maxVal = 1.0;
  const range = maxVal - minVal;

  // Compute summary metrics
  const avgSync =
    syncCurve.reduce((acc, val) => acc + val, 0) / syncCurve.length;
  const minSync = Math.min(...syncCurve);
  const desyncCount = syncCurve.filter((v) => v < 0).length;
  const desyncPercent = ((desyncCount / syncCurve.length) * 100).toFixed(1);

  // SVG points
  const points = syncCurve
    .map((val, idx) => {
      const x = (idx / (syncCurve.length - 1)) * 100;
      const y = 100 - ((val - minVal) / range) * 100;
      return `${x},${y}`;
    })
    .join(" ");

  // Fill polygon under curve down to baseline (y=50 is s_t=0)
  const areaPolygon = `0,50 ${points} 100,50`;

  const safeDuration = durationSec > 0 ? durationSec : 1;
  const playheadPercent = Math.min(100, Math.max(0, (currentTime / safeDuration) * 100));

  const handleSeekFromEvent = (clientX: number, target: HTMLDivElement) => {
    if (!onSeek) return;
    const rect = target.getBoundingClientRect();
    const clickX = clientX - rect.left;
    const fraction = Math.max(0, Math.min(1, clickX / rect.width));
    onSeek(fraction * safeDuration);
  };

  const handleGraphClick = (e: React.MouseEvent<HTMLDivElement>) => {
    handleSeekFromEvent(e.clientX, e.currentTarget);
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLDivElement>) => {
    if (e.touches && e.touches[0]) {
      handleSeekFromEvent(e.touches[0].clientX, e.currentTarget);
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-4 sm:p-6 shadow-sm space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-200/80 dark:border-white/[0.08] gap-2">
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-sky-600 dark:text-sky-400 shrink-0" />
          <div>
            <h3 className="text-xs sm:text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider font-mono">
              Phoneme-Viseme Sync Curve ($s_t$)
            </h3>
            <p className="text-[10px] sm:text-[11px] text-slate-500 dark:text-zinc-400">
              Windowed InfoNCE Cross-Modal Contrastive Alignment
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3 text-xs font-mono text-slate-500 dark:text-zinc-400">
          <div>
            Mean:{" "}
            <span
              className={`font-bold ${
                avgSync >= 0.2
                  ? "text-emerald-700 dark:text-emerald-400"
                  : avgSync >= 0.0
                  ? "text-amber-700 dark:text-amber-400"
                  : "text-rose-700 dark:text-rose-400"
              }`}
            >
              {avgSync > 0 ? `+${avgSync.toFixed(3)}` : avgSync.toFixed(3)}
            </span>
          </div>
          <span className="text-slate-300 dark:text-zinc-700">|</span>
          <div>
            Desync Anomaly:{" "}
            <span
              className={`font-bold ${
                Number(desyncPercent) > 20 ? "text-rose-700 dark:text-rose-400" : "text-slate-700 dark:text-zinc-300"
              }`}
            >
              {desyncPercent}%
            </span>
          </div>
        </div>
      </div>

      {/* SVG Curve Container */}
      <div
        onClick={handleGraphClick}
        onTouchStart={handleTouchMove}
        onTouchMove={handleTouchMove}
        className="h-36 sm:h-44 w-full bg-[#FAFCFF] dark:bg-[#07090E] border border-slate-200 dark:border-white/10 rounded-xl p-3 relative overflow-hidden flex flex-col justify-between cursor-pointer group shadow-inner touch-pan-x"
        title="Tap or drag on the curve to scrub video playhead"
      >
        {/* Horizontal grid guide lines */}
        <div className="absolute inset-0 flex flex-col justify-between p-3 pointer-events-none opacity-60">
          <div className="flex items-center text-[9px] font-mono text-slate-400 dark:text-zinc-500">
            <span className="w-7">+1.0</span>
            <div className="border-b border-dashed border-slate-200 dark:border-white/[0.08] w-full" />
          </div>
          <div className="flex items-center text-[9px] font-mono text-sky-700 dark:text-sky-400 font-semibold">
            <span className="w-7">0.0</span>
            <div className="border-b border-dashed border-sky-300 dark:border-sky-500/30 w-full" />
          </div>
          <div className="flex items-center text-[9px] font-mono text-rose-600 dark:text-rose-400 font-semibold">
            <span className="w-7">-1.0</span>
            <div className="border-b border-dashed border-rose-200 dark:border-rose-500/30 w-full" />
          </div>
        </div>

        {/* SVG Visualization */}
        <div className="w-full h-full relative pl-7 sm:pl-8">
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            className="w-full h-full overflow-visible"
          >
            <defs>
              <linearGradient id="syncGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0284c7" stopOpacity="0.25" />
                <stop offset="50%" stopColor="#0284c7" stopOpacity="0.04" />
                <stop offset="100%" stopColor="#e11d48" stopOpacity="0.2" />
              </linearGradient>
            </defs>

            {/* Filled Area */}
            <polygon
              points={areaPolygon}
              fill="url(#syncGradient)"
              className="transition-all duration-300"
            />

            {/* Main Polyline */}
            <polyline
              fill="none"
              stroke="#0284c7"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={points}
            />
          </svg>

          {/* Synchronized Playhead Needle */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-sky-600 shadow-[0_0_8px_rgba(2,132,199,0.5)] pointer-events-none z-10"
            style={{ left: `${playheadPercent}%` }}
          >
            <div className="w-2.5 h-2 -ml-1 -top-0 bg-sky-600 rounded-b-sm" />
          </div>
        </div>

        {/* Graph Subtitle Ruler */}
        <div className="flex justify-between text-[9px] sm:text-[10px] font-mono text-slate-400 dark:text-zinc-500 select-none pl-7 sm:pl-8 pt-1">
          <span>0.00s</span>
          <span className="text-slate-600 dark:text-zinc-300 font-medium">
            Time: {currentTime.toFixed(2)}s ({playheadPercent.toFixed(0)}%)
          </span>
          <span>{safeDuration.toFixed(2)}s</span>
        </div>
      </div>
    </div>
  );
};
