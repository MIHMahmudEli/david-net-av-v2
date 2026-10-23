"use client";

import React from "react";
import { TemporalInterval } from "../lib/api";
import { Clock, Play, CheckCircle2 } from "lucide-react";

interface DualTimelineProps {
  videoIntervals: TemporalInterval[];
  audioIntervals: TemporalInterval[];
  durationSec: number;
  currentTime?: number;
  onSeek?: (timeSec: number) => void;
}

export const DualTimeline: React.FC<DualTimelineProps> = ({
  videoIntervals,
  audioIntervals,
  durationSec,
  currentTime = 0,
  onSeek,
}) => {
  const safeDuration = durationSec > 0 ? durationSec : 1;
  const playheadPercent = Math.min(100, Math.max(0, (currentTime / safeDuration) * 100));

  const handleSeekFromEvent = (clientX: number, target: HTMLDivElement) => {
    if (!onSeek) return;
    const rect = target.getBoundingClientRect();
    const clickX = clientX - rect.left;
    const fraction = Math.max(0, Math.min(1, clickX / rect.width));
    onSeek(fraction * safeDuration);
  };

  const handleTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
    handleSeekFromEvent(e.clientX, e.currentTarget);
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLDivElement>) => {
    if (e.touches && e.touches[0]) {
      handleSeekFromEvent(e.touches[0].clientX, e.currentTarget);
    }
  };

  const renderTrack = (
    label: string,
    intervals: TemporalInterval[],
    colorClass: string,
    badgeColor: string
  ) => {
    return (
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center space-x-2">
            <span className={`w-2 h-2 rounded-full ${badgeColor}`} />
            <span className="font-mono font-bold text-slate-800 dark:text-zinc-200 uppercase tracking-wider text-[11px]">
              {label} Modality
            </span>
          </div>
          <span className="font-mono text-[11px] text-slate-500 dark:text-zinc-400">
            {intervals.length > 0 ? (
              <span className="text-rose-600 dark:text-rose-400 font-bold">
                {intervals.length} Tampered Interval(s)
              </span>
            ) : (
              <span className="text-emerald-700 dark:text-emerald-400 font-medium">Continuous 100% Pristine</span>
            )}
          </span>
        </div>

        {/* Track Container with Ruler */}
        <div
          onClick={handleTrackClick}
          onTouchStart={handleTouchMove}
          onTouchMove={handleTouchMove}
          className="w-full h-9 sm:h-10 bg-slate-100 dark:bg-[#08090D] border border-slate-200 dark:border-white/[0.08] rounded-xl relative overflow-hidden cursor-pointer group shadow-inner touch-pan-x"
          title="Tap or drag anywhere to seek video playhead"
        >
          {/* Subtle grid tick lines */}
          <div className="absolute inset-0 flex justify-between px-2 pointer-events-none opacity-40">
            <div className="border-r border-slate-300 dark:border-white/[0.08] h-full" />
            <div className="border-r border-slate-300 dark:border-white/[0.08] h-full" />
            <div className="border-r border-slate-300 dark:border-white/[0.08] h-full" />
            <div className="border-r border-slate-300 dark:border-white/[0.08] h-full" />
          </div>

          {/* Authentic Baseline */}
          {intervals.length === 0 ? (
            <div className="absolute inset-0 flex items-center pl-3 text-[10px] sm:text-[11px] font-mono text-emerald-700 dark:text-emerald-400 font-medium">
              <CheckCircle2 className="w-3.5 h-3.5 mr-1.5 text-emerald-600 dark:text-emerald-400 shrink-0" />
              Unbroken Temporal Integrity [0.0s – {safeDuration.toFixed(1)}s]
            </div>
          ) : (
            intervals.map((intv, idx) => {
              const left = Math.max(0, Math.min(100, (intv.start / safeDuration) * 100));
              const width = Math.max(
                3,
                Math.min(100 - left, ((intv.end - intv.start) / safeDuration) * 100)
              );
              return (
                <div
                  key={idx}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (onSeek) onSeek(intv.start);
                  }}
                  className={`absolute top-1 bottom-1 border rounded-lg flex items-center justify-center text-[9px] sm:text-[10px] font-mono font-bold text-white shadow-sm cursor-pointer transition-transform hover:scale-[1.02] ${colorClass}`}
                  style={{ left: `${left}%`, width: `${width}%` }}
                  title={`Tampered: ${intv.start.toFixed(2)}s – ${intv.end.toFixed(
                    2
                  )}s (Confidence: ${(intv.score * 100).toFixed(1)}%) • Tap to seek`}
                >
                  <span className="truncate px-1">
                    {intv.start.toFixed(1)}s–{intv.end.toFixed(1)}s
                  </span>
                </div>
              );
            })
          )}

          {/* Synchronized Playhead Needle */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-sky-600 shadow-[0_0_8px_rgba(2,132,199,0.5)] pointer-events-none z-10"
            style={{ left: `${playheadPercent}%` }}
          >
            <div className="w-2.5 h-2 -ml-1 -top-0 bg-sky-600 rounded-b-sm" />
          </div>
        </div>
      </div>
    );
  };

  const allIntervals = [
    ...videoIntervals.map((i) => ({ ...i, type: "Video" as const })),
    ...audioIntervals.map((i) => ({ ...i, type: "Audio" as const })),
  ].sort((a, b) => a.start - b.start);

  return (
    <div className="glass-panel rounded-2xl p-4 sm:p-6 shadow-sm space-y-4 sm:space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-200/80 dark:border-white/[0.08] gap-2">
        <div className="flex items-center space-x-2">
          <Clock className="w-4 h-4 text-sky-600 dark:text-sky-400 shrink-0" />
          <h3 className="text-xs sm:text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider font-mono">
            Temporal Forgery Localization Heads
          </h3>
        </div>
        <div className="flex items-center space-x-3 text-xs font-mono text-slate-500 dark:text-zinc-400">
          <span>
            Playhead:{" "}
            <strong className="text-sky-700 dark:text-sky-400 font-bold">
              {currentTime.toFixed(2)}s / {safeDuration.toFixed(2)}s
            </strong>
          </span>
          <span className="text-slate-300 dark:text-zinc-700">|</span>
          <span className="text-[10px] text-slate-400 dark:text-zinc-500">Dense Boundary MLP</span>
        </div>
      </div>

      {/* Tracks */}
      <div className="space-y-3.5 sm:space-y-4">
        {renderTrack(
          "Video Track",
          videoIntervals,
          "bg-rose-500 border-rose-600 hover:bg-rose-600",
          "bg-rose-500"
        )}
        {renderTrack(
          "Audio Track",
          audioIntervals,
          "bg-indigo-600 border-indigo-700 hover:bg-indigo-700",
          "bg-indigo-600"
        )}
      </div>

      {/* Detected Interval Chips (Touch / Click to Jump) */}
      {allIntervals.length > 0 && (
        <div className="pt-2 border-t border-slate-200/80 dark:border-white/[0.08] space-y-2">
          <span className="text-[10px] font-mono text-slate-500 dark:text-zinc-400 uppercase tracking-wider font-semibold block">
            Discovered Forgery Intervals (Tap to jump):
          </span>
          <div className="flex flex-wrap gap-2">
            {allIntervals.map((intv, idx) => (
              <button
                key={idx}
                onClick={() => onSeek && onSeek(intv.start)}
                className={`inline-flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg text-xs font-mono border transition-all ${
                  intv.type === "Video"
                    ? "bg-rose-50 dark:bg-rose-500/10 text-rose-800 dark:text-rose-300 border-rose-200 dark:border-rose-500/30 hover:bg-rose-100 dark:hover:bg-rose-500/20 active:scale-95"
                    : "bg-indigo-50 dark:bg-indigo-500/10 text-indigo-800 dark:text-indigo-300 border-indigo-200 dark:border-indigo-500/30 hover:bg-indigo-100 dark:hover:bg-indigo-500/20 active:scale-95"
                }`}
              >
                <Play className="w-3 h-3 fill-current opacity-70 shrink-0" />
                <span className="font-semibold">
                  {intv.type}: {intv.start.toFixed(2)}s – {intv.end.toFixed(2)}s
                </span>
                <span className="text-[10px] opacity-80 font-bold">
                  ({(intv.score * 100).toFixed(0)}%)
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
